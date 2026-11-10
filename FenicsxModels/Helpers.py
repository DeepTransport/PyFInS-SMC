import os
from dolfinx import fem
from petsc4py import PETSc
from mpi4py import MPI
import pyvista
from dolfinx import plot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

def owned_dof_coords(V: fem.FunctionSpace) -> np.ndarray:
    """
    Owned DOF coordinates on this rank.
    Works for Lagrange-type spaces where tabulate_dof_coordinates is defined.
    Shape: (n_owned, gdim)
    """
    n = V.dofmap.index_map.size_local
    X = V.tabulate_dof_coordinates()
    return np.asarray(X[:n, :], dtype=np.float64)

def coerce_space_value(
    x,
    V: fem.FunctionSpace,
    *,
    out: str = "function",
    dtype=np.float64,
):
    """
    Convert input x into either:
      - owned-local numpy vector in V ('out="owned"')
      - fem.Function(V) with ghosts updated ('out="function"')

    Accepted x:
      - scalar (broadcast)
      - numpy array (owned-local, length n_owned)
      - PETSc.Vec (uses first n_owned entries of local storage)
      - fem.Function (any space): sampled by its DOF vector if same space; otherwise interpolated
      - callable f(X): evaluated at owned DOF coordinates of V
          * tries f(X.T) then f(X)
          * expects length n_owned

    Notes:
      - For MPI, we always write/return owned entries and then scatter_forward() if returning a Function.
      - If you pass a fem.Function in a different space, we interpolate to V.
    """
    n_owned = V.dofmap.index_map.size_local

    # 1) scalar
    if np.isscalar(x):
        owned = np.full(n_owned, float(x), dtype=dtype)

    # 2) dolfinx Function
    elif isinstance(x, fem.Function):
        # If it's already in the same function space, just read dofs
        if x.function_space == V:
            x.x.scatter_forward()
            owned = np.asarray(x.x.array[:n_owned], dtype=dtype).copy()
        else:
            # Interpolate to V
            tmp = fem.Function(V)
            tmp.interpolate(x)
            tmp.x.scatter_forward()
            owned = np.asarray(tmp.x.array[:n_owned], dtype=dtype).copy()
    elif isinstance(x, PETSc.Vec):
        arr = x.getArray(readonly=False)
        owned = np.asarray(arr[:n_owned], dtype=dtype).copy()
    elif callable(x):
        X = owned_dof_coords(V)  # (n_owned, gdim)
        try:
            vals = np.asarray(x(X.T), dtype=dtype).reshape(-1)
        except Exception:
            vals = np.asarray(x(X), dtype=dtype).reshape(-1)
        if vals.size != n_owned:
            raise ValueError(f"Callable returned {vals.size} values, expected {n_owned}.")
        owned = vals
    # 5) numpy array
    else:
        arr = np.asarray(x, dtype=dtype).reshape(-1)
        if arr.size != n_owned:
            raise ValueError(f"Array length {arr.size} != n_owned {n_owned}.")
        owned = arr

    if out == "owned":
        return owned
    if out == "function":
        f = fem.Function(V)
        f.x.array[:n_owned] = owned
        f.x.scatter_forward()
        return f

    raise ValueError("out must be 'owned' or 'function'")

def plotting_grid(f_h, name="fun"):
    Vh = f_h.function_space
    # for plotting
    if Vh.ufl_element().degree == 0:
        topology, cell_types, geometry = plot.vtk_mesh(Vh.mesh, Vh.mesh.topology.dim)
        plot_g = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        n_owned_cells = Vh.dofmap.index_map.size_local
        plot_g.cell_data[name] = f_h.x.array[:n_owned_cells].copy()
        plot_g.set_active_scalars(name)
    else:
        cells, celltypes, x = plot.vtk_mesh(Vh)
        plot_g = pyvista.UnstructuredGrid(cells, celltypes, x)
        n_owned_cells = Vh.dofmap.index_map.size_local
        plot_g.point_data[name] = f_h.x.array[:n_owned_cells].copy()
        plot_g.set_active_scalars(name)

    return plot_g


# -----------------------------------------------------------------------------
# MPI plotting
# -----------------------------------------------------------------------------

def plot_prior_draw_mpi(
    prior,
    k,
    filename,
    *,
    title=None,
    show=False,
    root=0,
):
    """
    Plot one prior draw in a single figure.

    Columns:
        fields

    Rows:
        combined field
        rank 0
        rank 1
        ...

    Only owned cells are plotted on each MPI rank. The combined field is formed
    by gathering owned cells, so cells are not duplicated across MPI ranks.
    """

    V = prior.Kh
    msh = V.mesh
    comm = msh.comm
    rank = comm.rank

    x_local = msh.geometry.x[:, 0]
    y_local = msh.geometry.x[:, 1]

    xmin = comm.allreduce(x_local.min(), op=MPI.MIN)
    xmax = comm.allreduce(x_local.max(), op=MPI.MAX)

    ymin = comm.allreduce(y_local.min(), op=MPI.MIN)
    ymax = comm.allreduce(y_local.max(), op=MPI.MAX)

    if hasattr(prior, "n_fields"):
        k_fields = prior.split_parameter(k)
        names = prior.names
        n_fields = prior.n_fields
    else:
        k_fields = k.reshape(1, -1)
        names = ["field"]
        n_fields = 1

    tdim = msh.topology.dim
    n_cells = msh.topology.index_map(tdim).size_local
    n_owned = V.dofmap.index_map.size_local

    # Build one Function per field so ghost values are available on cells
    # touching an MPI interface.
    functions = []
    for i in range(n_fields):
        f = fem.Function(V)
        f.x.array[:n_owned] = k_fields[..., i, :]
        f.x.scatter_forward()
        functions.append(f)

    degree = V.ufl_element().degree

    # Store the actual owned-cell geometry rather than triangulating DOF clouds.
    cell_points = np.empty((n_cells, 3, 2))

    if degree == 0:
        field_values = np.empty((n_fields, n_cells))
        for cell in range(n_cells):
            geometry_dofs = msh.geometry.dofmap[cell]
            cell_points[cell] = msh.geometry.x[geometry_dofs[:3], :2]
            dof = V.dofmap.cell_dofs(cell)[0]
            for i, f in enumerate(functions):
                field_values[i, cell] = f.x.array[dof]
    else:
        field_values = np.empty((n_fields, n_cells, 3))
        dof_coordinates = V.tabulate_dof_coordinates()
        for cell in range(n_cells):
            dofs = V.dofmap.cell_dofs(cell)
            cell_points[cell] = dof_coordinates[dofs[:3], :2]
            for i, f in enumerate(functions):
                field_values[i, cell] = f.x.array[dofs[:3]]

    points_all = comm.gather(cell_points, root=root)
    values_all = comm.gather(field_values, root=root)

    if rank != root:
        return

    rows = comm.size + 1
    fig, axes = plt.subplots(
        rows,
        n_fields,
        figsize=(4.5 * n_fields, 3.8 * rows),
        squeeze=False,
    )

    def plot_cells(ax, points, values, vmin, vmax):
        n = points.shape[0]
        xy = points.reshape(-1, 2)
        triangles = np.arange(3 * n).reshape(n, 3)
        triangulation = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles)

        if degree == 0:
            im = ax.tripcolor(
                triangulation,
                facecolors=values,
                shading="flat",
                vmin=vmin,
                vmax=vmax,
            )
        else:
            im = ax.tripcolor(
                triangulation,
                values.reshape(-1),
                shading="gouraud",
                vmin=vmin,
                vmax=vmax,
            )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        return im

    combined_points = np.concatenate(points_all, axis=0)

    for i in range(n_fields):
        combined_values = np.concatenate(
            [values_all[r][i] for r in range(comm.size)],
            axis=0,
        )

        vmin = combined_values.min()
        vmax = combined_values.max()

        im = plot_cells(
            axes[0, i],
            combined_points,
            combined_values,
            vmin,
            vmax,
        )

        axes[0, i].set_title(names[i])
        if i == 0:
            axes[0, i].set_ylabel("combined")

        for r in range(comm.size):
            plot_cells(
                axes[r + 1, i],
                points_all[r],
                values_all[r][i],
                vmin,
                vmax,
            )

            if i == 0:
                axes[r + 1, i].set_ylabel(f"rank {r}")

        fig.colorbar(im, ax=axes[:, i], shrink=0.85)

    if title is not None:
        fig.suptitle(title)

    if filename is not None:
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(f"{filename}.png", dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

