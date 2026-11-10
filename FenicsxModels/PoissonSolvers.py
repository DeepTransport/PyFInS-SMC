from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import scipy.sparse as sp

import ufl
import dolfinx
from dolfinx import mesh, fem
from dolfinx.fem import petsc as fpetsc
from dolfinx.fem.petsc import LinearProblem

from BaseModels import SolverAdjointCache
from FenicsxModels.Helpers import coerce_space_value

# Solution scalar (e.g., float32, complex128) and geometry (float32/64)
# types
dtype = dolfinx.default_scalar_type
xdtype = dolfinx.default_real_type

#
def make_lift(Vh, xmax, xmin, pL, pR):
    """
    Make the lift function for given Dirichlet data on left and right
    """
    varphi = fem.Function(Vh)
    def lift_eval(X):
        y = X[1]
        return (xmax - X[0])/(xmax-xmin) * pL(y) + (X[0] - xmin)/(xmax-xmin) * pR(y)
    varphi.interpolate(lift_eval)
    return varphi

#
def make_qoi(Vh, xmax, xmin):
    """
    Make the qoi functional for the integrated flux on the right
    """
    phi = fem.Function(Vh)
    def qoi_eval(X):
        y = X[1]
        return (X[0] - xmin)/(xmax-xmin)
    phi.interpolate(qoi_eval)
    return phi

#
def make_observation_points(mesh, n):
    """
    Make a grid of interior points for the observation operator 
    """
    Xg = mesh.geometry.x  # (num_geom_dofs, gdim)

    #xmin, ymin = Xg[:, 0].min(), Xg[:, 1].min()
    #xmax, ymax = Xg[:, 0].max(), Xg[:, 1].max()

    comm = mesh.comm
    xmin = comm.allreduce(Xg[:, 0].min(), op=MPI.MIN)
    ymin = comm.allreduce(Xg[:, 1].min(), op=MPI.MIN)
    xmax = comm.allreduce(Xg[:, 0].max(), op=MPI.MAX)
    ymax = comm.allreduce(Xg[:, 1].max(), op=MPI.MAX)

    grid = np.linspace(0.0, 1.0, n+2)     
    interior = grid[1:-1]               
    Xhat, Yhat = np.meshgrid(interior, interior, indexing="xy")

    # Affine map from [0,1]^2 -> [xmin,xmax]x[ymin,ymax]
    X = xmin + (xmax - xmin) * Xhat
    Y = ymin + (ymax - ymin) * Yhat

    pts = np.column_stack([X.ravel(), Y.ravel()])  
    return pts

#    
def bump_compact_ufl(x, x0, eps):
    """
    A ufl function for a bump centered at the observation point 
    """
    dx = x[0] - x0[0]
    dy = x[1] - x0[1]
    r2 = dx*dx + dy*dy
    s2 = r2 / (eps*eps)
    return ufl.conditional(
        ufl.lt(s2, 1.0),
        ufl.exp(-1.0 / (1.0 - s2)),
        0.0
    )

#
def precompute_observation_operator_local(mesh, Vh, sensor_points, eps):
    """
    Precompute local column-block observation operator C_local (SciPy CSR) on each rank.

    C_local shape = (m_sensors, n_owned_dofs_on_rank)

    Forward:
        y_local = C_local @ p_owned
        y = allreduce(y_local, SUM)

    Transpose:
        b_owned = C_local.T @ r
    """
    comm = mesh.comm
    sensor_points = np.asarray(sensor_points, dtype=np.float64)
    m = sensor_points.shape[0]

    imap = Vh.dofmap.index_map
    n_owned = imap.size_local
    n_ghost = imap.num_ghosts

    x = ufl.SpatialCoordinate(mesh)
    w = ufl.TestFunction(Vh)

    rows, cols, data = [], [], []

    for i, x0 in enumerate(sensor_points):
        rho = bump_compact_ufl(x, x0, eps)

        mass_local = fem.assemble_scalar(fem.form(rho * ufl.dx))
        mass = comm.allreduce(mass_local, op=MPI.SUM)
        if mass <= 0.0:
            continue
        rho = rho / mass

        # Assemble full FE vector for this sensor functional
        bi = fpetsc.assemble_vector(fem.form(rho * w * ufl.dx))
        bi.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                       mode=PETSc.ScatterMode.REVERSE)

        # Keep only OWNED dofs on this rank
        arr_owned = bi.array[:n_owned]
        nz = np.nonzero(arr_owned)[0]
        if nz.size == 0:
            continue

        rows.append(np.full(nz.size, i, dtype=np.int32))
        cols.append(nz.astype(np.int32))     # local-owned column index
        data.append(arr_owned[nz].astype(np.float64))

        #print((comm.rank,i,nz), flush=True)

    if len(rows) == 0:
        C_local = sp.csr_matrix((m, n_owned))
    else:
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)
        C_local = sp.csr_matrix((data, (rows, cols)), shape=(m, n_owned))

    return C_local

#
class PoissonCG(SolverAdjointCache):
    """Solves the Poisson equation using CG.
    """
    
    def __init__(
        self, 
        corner_points,
        Nx,
        Ny, 
        pL,                 # Python callable gL(y): float or array
        pR,                 # Python callable gR(y): float or array
        *, 
        order = 1,
        const_k = False,
        log_normal = True,
        high_accuracy = False,
    ):
        super().__init__()
        
        self.mesh = mesh.create_rectangle(
            comm=MPI.COMM_WORLD,
            points=corner_points,
            n=(Nx, Ny),
            cell_type=mesh.CellType.triangle,
            dtype=xdtype
        )

        self.order = order
        self.const_k = const_k
        self.log_normal = log_normal

        # spaces
        self.Vh = fem.functionspace(self.mesh, ("Lagrange", self.order))
        if self.const_k:
            self.Kh = fem.functionspace(self.mesh, ("DG", 0)) 
        else:
            self.Kh = self.Vh

        self.nV_owned = self.Vh.dofmap.index_map.size_local
        self.nK_owned = self.Kh.dofmap.index_map.size_local
        self.dof_coordinates = self.Kh.tabulate_dof_coordinates()


        # not safe for MPI
        #Xg = self.mesh.geometry.x  # (num_geom_dofs, gdim)
        #xmin, ymin = Xg[:, 0].min(), Xg[:, 1].min()
        #xmax, ymax = Xg[:, 0].max(), Xg[:, 1].max()

        xmin, ymin = corner_points[0]
        xmax, ymax = corner_points[1]

        # ---- Lift varphi(x,y) = (1-x) gL(y) + x gR(y)
        self.lift_fun = make_lift(self.Vh, xmax, xmin, pL, pR)
        self.lift_fun.x.scatter_forward()

        # Unknown v in V0 (zero on x=0 and x=1)
        self.v = ufl.TrialFunction(self.Vh)
        self.w = ufl.TestFunction(self.Vh)

        # Dirichlet BCs for v: v=0 on x=min and x=xmax
        def on_left(X):  return np.isclose(X[0], xmin)
        def on_right(X): return np.isclose(X[0], xmax)
        left_dofs = fem.locate_dofs_geometrical(self.Vh, on_left)
        right_dofs = fem.locate_dofs_geometrical(self.Vh, on_right)
        self.bcs = [
            fem.dirichletbc(PETSc.ScalarType(0.0), left_dofs, self.Vh),
            fem.dirichletbc(PETSc.ScalarType(0.0), right_dofs, self.Vh),
        ]

        # QoI via phi(x,y)=x (your derivation): Q = ∫Ω k∇p·∇phi dx
        self.qoi_fun = make_qoi(self.Vh, xmax, xmin)
        self.qoi_fun.x.scatter_forward()

        # the forcing vector, default is 0
        self.f = fem.Function(self.Vh)
        self.f.x.array[:] = 0.0
        self.f.x.scatter_forward()

        # for the pre computed observation operator   
        self.sensor_points = None
        self.obs_eps = None
        self.C_local = None
        self.number_of_sensors = None

        self.n_parameter = self.Kh.dofmap.index_map.size_global # not in use
        self.n_data = 0 # set in make_observation
        self.n_qoi = 1  # 1 QoI 
        self.sensor2data = None

        # cached solutions
        self.p_h = None
        self.k_fun = None
        self.a_form = None
        self.A = None
        self.ksp = None

        self.high_accuracy = high_accuracy
    
    def make_observation_operator(self, obs_n_grid, obs_eps):
        # pre compute observation operator   
        self.sensor_points = make_observation_points(self.mesh, obs_n_grid)
        self.obs_eps = obs_eps
        self.C_local = precompute_observation_operator_local(
            self.mesh, self.Vh, self.sensor_points, self.obs_eps)
        self.number_of_sensors = self.C_local.shape[0]
        self.n_data = self.number_of_sensors

    def solve(self, k_in):
        # x = ufl.SpatialCoordinate(self.mesh)
        # k as a Function

        if self.log_normal:
            k_fun = coerce_space_value(np.exp(k_in), self.Kh)
        else:
            k_fun = coerce_space_value(k_in, self.Kh)

        a = ufl.inner(k_fun * ufl.grad(self.v), ufl.grad(self.w)) * ufl.dx
        L = ufl.inner(self.f, self.w) * ufl.dx - ufl.inner(k_fun * ufl.grad(self.lift_fun), ufl.grad(self.w)) * ufl.dx

        # Solve for v_h
        problem = LinearProblem(a, L, bcs=self.bcs, 
                                petsc_options={"ksp_type": "cg", "pc_type": "hypre"}, 
                                petsc_options_prefix="Poisson",)
        v_h = problem.solve()
        v_h.x.scatter_forward()

        # Recover p_h = v_h + varphi
        p_h = fem.Function(self.Vh)
        p_h.x.array[:] = v_h.x.array + self.lift_fun.x.array
        p_h.x.scatter_forward()

        Y = self.apply_observation(p_h)
        Q = self.apply_QoI(p_h, k_fun)

        return Y, Q

    #
    def apply_QoI(self, p_h, k_fun):
        Q_form = fem.form( ufl.inner(k_fun * ufl.grad(p_h), ufl.grad(self.qoi_fun)) * ufl.dx )
        Q = self.mesh.comm.allreduce(fem.assemble_scalar(Q_form), op=MPI.SUM)
        return Q


    def apply_observation(self, p_h):
        """
        y = C p using local column blocks + allreduce.
        Returns y as numpy array (length m) on all ranks.
        """
        if self.C_local is None:
            raise RuntimeError("Call make_observation_operator(...) first.")
        
        p_h.x.scatter_forward()  # safe; ensures ghosts consistent if needed
        #nV_owned = self.Vh.dofmap.index_map.size_local
        p_owned = p_h.x.array[:self.nV_owned]  # owned dofs only

        y_local = self.C_local @ p_owned       # shape (m,)
        y = np.zeros_like(y_local)
        self.mesh.comm.Allreduce(y_local, y, op=MPI.SUM)
        return y

    
    def _apply_observation_transpose(self, delta_y):
        """
        Compute b_owned = C_local.T @ r on this rank (owned dofs only).
        r: numpy array of length m (same on all ranks, e.g. from allreduce/bcast)
        Returns numpy array length n_owned.
        """
        delta_y = np.asarray(delta_y, dtype=np.float64)
        b_owned = self.C_local.T @ delta_y
        b_owned = np.asarray(b_owned).reshape(-1)
    
        dummy_form = fem.form(self.f * self.w * ufl.dx)
        #b = fpetsc.create_vector(zero_form)
        b = fpetsc.assemble_vector(dummy_form)   
        b.set(0.0)

        #nK_owned = self.Kh.dofmap.index_map.size_local
        #nV_owned = self.Vh.dofmap.index_map.size_local
        b.array[:self.nV_owned] = b_owned
        # ghost entries (if any) can remain zero
        return b
    
    def _assemble_operator(self, k_fun):
        if self.ksp is not None:
            self.ksp.destroy()
        if self.A is not None:
            self.A.destroy()
            
        """Assemble and cache A(k) and a reusable KSP. Call after setting k_fun."""
        self.a_form = fem.form(ufl.inner(k_fun * ufl.grad(self.v), ufl.grad(self.w)) * ufl.dx)
        self.A = fpetsc.assemble_matrix(self.a_form, bcs=self.bcs)
        self.A.assemble()

        self.ksp = PETSc.KSP().create(self.mesh.comm)
        self.ksp.setOperators(self.A)
        self.ksp.setType("cg")
        self.ksp.getPC().setType("hypre")
        self.ksp.setFromOptions()
        #for debugging adjoint
        if self.high_accuracy:
            self.ksp.setTolerances(rtol=1.0e-20, atol=1.0e-20)
        return 

    def _assemble_rhs_vec(self, rhs_form):
        """Assemble RHS PETSc.Vec in Vh dual, apply lifting + BC."""
        b = fpetsc.assemble_vector(rhs_form)
        fpetsc.apply_lifting(b, [self.a_form], bcs=[self.bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                      mode=PETSc.ScatterMode.REVERSE)
        fpetsc.set_bc(b, self.bcs)
        return b

    def _solve_with_cached_ksp(self, b):
        """Solve A x = b using cached KSP; return Function in Vh."""
        x_fun = fem.Function(self.Vh)
        self.ksp.solve(b, x_fun.x.petsc_vec)
        x_fun.x.scatter_forward()
        return x_fun

    def solve_and_cache(self, k_in, rebuild_operator=True):
        """
        Solve primal and cache operator/KSP for reuse in tangent/adjoint.
        Returns (Y, Q).
        """
        # set kappa
        if self.log_normal:
            k_phys_owned = np.exp(k_in)
        else:
            k_phys_owned = k_in

        k_fun = coerce_space_value(k_phys_owned, self.Kh)

        # Assemble operator (unless user says reuse)
        if rebuild_operator or (not hasattr(self, "A")) or (self.A is None):
            self._assemble_operator(k_fun)

        # RHS form: ∫ f w dx - ∫ k ∇lift · ∇w dx
        L_form = fem.form(
            ufl.inner(self.f, self.w) * ufl.dx
            - ufl.inner(k_fun * ufl.grad(self.lift_fun), ufl.grad(self.w)) * ufl.dx
        )

        # Assemble RHS vector b
        b = self._assemble_rhs_vec(L_form)
        v_h = self._solve_with_cached_ksp(b)

        # Recover p = v + lift
        p_h = fem.Function(self.Vh)
        p_h.x.array[:] = v_h.x.array + self.lift_fun.x.array
        p_h.x.scatter_forward()

        # save a copy
        self.p_h = p_h
        self.k_fun = k_fun
        self.k_phys_owned = k_phys_owned

        Y = self.apply_observation(p_h)
        Q = self.apply_QoI(p_h, k_fun)

        return Y, Q

    # ----------------------------
    # Tangent solve
    # ----------------------------
    def _tangent_solve(self, delta_k_in):
        """
        Tangent solve for δp given δk (cellwise DG0):
        A(k) δv = -∫ δk ∇p · ∇w dx
        with p = v + lift, and δp = δv (lift fixed).
        Requires self.p_h, self.k_fun, self.A/self.ksp cached (use solve_and_cache first).
        """        
        if self.ksp is None or self.p_h is None:
            raise RuntimeError("Call solve_and_cache(k_in) first.")

        dk = coerce_space_value(delta_k_in, self.Kh)

        # RHS: -∫ δk ∇p · ∇w
        L_form = fem.form(
            -ufl.inner(dk * ufl.grad(self.p_h), ufl.grad(self.w)) * ufl.dx
        )
        b = self._assemble_rhs_vec(L_form)
        dp_h = self._solve_with_cached_ksp(b)
        return dp_h
    
    # ----------------------------
    # Linearized (tangent) solve
    # ----------------------------
    def linearized_observation(self, delta_k_cell):
        #dp_h = self._tangent_solve(delta_k_cell)
        #dy = self.apply_observation(dp_h)

        arr = np.asarray(delta_k_cell, dtype=np.float64)   
        if self.log_normal:
            arr = arr * self.k_phys_owned 
        if arr.ndim == 1:
            dp_h = self._tangent_solve(arr)
            dy = self.apply_observation(dp_h)
            return np.asarray(dy, dtype=np.float64).reshape(-1)
        else:
            raise ValueError(f"only take one dk.")

    # ----------------------------
    # Adjoint for QoI: Q = ∫ k ∇p · ∇qoi_fun dx
    # ----------------------------
    def _adjoint_qoi(self):
        """
        Solve adjoint λ_Q:
        A(k) λ_Q = dQ/dp,  where dQ/dp[w]=∫ k ∇w·∇phi dx
        Requires cached operator.
        """
        if self.ksp is None or self.p_h is None:
            raise RuntimeError("Call solve_and_cache(k_in) first.")

        L_form = fem.form(
            ufl.inner(self.k_fun * ufl.grad(self.w), ufl.grad(self.qoi_fun)) * ufl.dx
        )
        b = self._assemble_rhs_vec(L_form)
        lam = self._solve_with_cached_ksp(b)
        return lam

    # ----------------------------
    # Adjoint for observation misfit: J=1/2||C p - d||^2
    # ----------------------------
    def _adjoint_observation(self, delta_y):
        """
        Solve adjoint λ_obs:
        A(k) λ_obs = C^T (C p - d)
        Returns (lam_obs, y_on_root).
        Requires self.C (observation operator) and cached operator.
        """
        if self.C_local is None:
            raise RuntimeError("Call make_observation_operator(...) first.")
        if self.ksp is None or self.p_h is None:
            raise RuntimeError("Call solve_and_cache(k_in) first.")

        b_obs = self._apply_observation_transpose(delta_y)  # PETSc Vec in pressure space
        # apply BC modifications consistent with adjoint space
        fpetsc.apply_lifting(b_obs, [self.a_form], bcs=[self.bcs])
        b_obs.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                          mode=PETSc.ScatterMode.REVERSE)
        fpetsc.set_bc(b_obs, self.bcs)

        lam = self._solve_with_cached_ksp(b_obs)
        return lam

    def _grad_k(self, density_ufl):
        test = ufl.TestFunction(self.Kh)
        g_vec = fpetsc.assemble_vector(fem.form(density_ufl * test * ufl.dx))
        g_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                          mode=PETSc.ScatterMode.REVERSE)
        #return g_vec.array[:self.Kh.dofmap.index_map.size_local].copy()
        return g_vec.array[:self.nK_owned].copy()

    def linearized_observation_transpose(self, delta_y):
        """
        """
        #lam = self._adjoint_observation(delta_y)
        #density_ufl = -ufl.inner(ufl.grad(self.p_h), ufl.grad(lam))
        #return self._grad_k(density_ufl)

        arr = np.asarray(delta_y, dtype=np.float64)    
        if arr.ndim == 1:
            lam = self._adjoint_observation(arr)
            density_ufl = -ufl.inner(ufl.grad(self.p_h), ufl.grad(lam))
            g_local = np.asarray(self._grad_k(density_ufl), dtype=np.float64).reshape(-1)
            if self.log_normal:
                g_local = g_local * self.k_phys_owned
        else:   
            raise ValueError(f"only take one dy.")
        return g_local

    def linearized_qoi_transpose(self):
        """
        Gradient wrt DG0 k for QoI Q(k)=∫ k ∇p·∇phi dx:
        dQ/dk[ξ] = ∫ ξ ∇p·∇phi dx  - ∫ ξ ∇p·∇λ_Q dx
        so cellwise:
        g_K = ∫_K (∇p·∇phi - ∇p·∇λ_Q) dx
        Returns numpy array of length num_local_cells.
        """
        lam = self._adjoint_qoi()
        term_explicit = ufl.inner(ufl.grad(self.p_h), ufl.grad(self.qoi_fun))
        term_adjoint = ufl.inner(ufl.grad(self.p_h), ufl.grad(lam))
        density_ufl = term_explicit - term_adjoint
        g_local = self._grad_k(density_ufl)
        if self.log_normal:
            g_local = g_local * self.k_phys_owned
        #if self.mesh.comm.size > 1:
        #    return self.gather_dg0_to_root(g_local, root=0)
        #else:
        return g_local





