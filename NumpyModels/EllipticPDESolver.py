import numpy as np
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.io import loadmat
from scipy.special import gamma

from sksparse.cholmod import cholesky
from scikits.umfpack import splu

from BaseModels import SolverAdjointCache, GaussianLikelihood, GaussianPriorBase


def structured_mesh(nx=120, ny=40, lx=3.0, ly=1.0):
    gx = np.linspace(0.0, lx, nx + 1)
    gy = np.linspace(0.0, ly, ny + 1)

    nodes = np.column_stack((
        np.repeat(gx, ny + 1),
        np.tile(gy, nx + 1),
    ))

    n0 = np.repeat(np.arange(nx), ny) * (ny + 1) + np.tile(np.arange(ny), nx)

    elements = np.column_stack((
        n0,
        n0 + ny + 1,
        n0 + ny + 2,
        n0 + 1,
    ))

    # Boundary edges
    left = np.column_stack((
        np.arange(ny),
        np.arange(1, ny + 1),
    ))

    right0 = nx * (ny + 1)
    right = np.column_stack((
        right0 + np.arange(ny),
        right0 + np.arange(1, ny + 1),
    ))

    bottom = np.column_stack((
        np.arange(nx) * (ny + 1),
        (np.arange(nx) + 1) * (ny + 1),
    ))

    top = np.column_stack((
        np.arange(nx) * (ny + 1) + ny,
        (np.arange(nx) + 1) * (ny + 1) + ny,
    ))

    boundary_edges = np.vstack((left, right, bottom, top))

    return nodes, elements, boundary_edges, gx, gy


def plot_mesh(nodes, elements):
    fig, ax = plt.subplots()

    for e in elements:
        xy = nodes[e]
        xy = np.vstack((xy, xy[0]))
        ax.plot(xy[:, 0], xy[:, 1], "k-", linewidth=0.5)

    ax.scatter(nodes[:, 0], nodes[:, 1], s=5)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Structured mesh")

    plt.show()


def plot_element(nodes, elements, element_idx, patch=2):
    e = elements[element_idx]
    xy = nodes[e]

    hx = xy[:, 0].max() - xy[:, 0].min()
    hy = xy[:, 1].max() - xy[:, 1].min()

    xmin = xy[:, 0].min() - patch * hx
    xmax = xy[:, 0].max() + patch * hx
    ymin = xy[:, 1].min() - patch * hy
    ymax = xy[:, 1].max() + patch * hy

    local_nodes = (
        (nodes[:, 0] >= xmin)
        & (nodes[:, 0] <= xmax)
        & (nodes[:, 1] >= ymin)
        & (nodes[:, 1] <= ymax)
    )

    fig, ax = plt.subplots()

    for elem in elements:
        if np.any(local_nodes[elem]):
            elem_xy = nodes[elem]
            closed = np.vstack((elem_xy, elem_xy[0]))
            ax.plot(closed[:, 0], closed[:, 1], "k-", linewidth=0.5)

    inds = np.flatnonzero(local_nodes)
    ax.scatter(nodes[inds, 0], nodes[inds, 1], s=30)

    for i in inds:
        ax.annotate(
            str(i),
            nodes[i],
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    closed = np.vstack((xy, xy[0]))
    ax.plot(closed[:, 0], closed[:, 1], linewidth=2)
    ax.scatter(xy[:, 0], xy[:, 1], s=80, zorder=3)

    for local_idx, global_idx in enumerate(e):
        ax.annotate(
            f"local {local_idx}",
            nodes[global_idx],
            xytext=(5, -15),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_title(f"Element {element_idx}: global nodes {e.tolist()}")

    plt.show()


class MRFPrecisionPrior(GaussianPriorBase):
    """
    Gaussian MRF prior in log-permeability space.

    Q = (tmp M^{-1/2} P)^T (tmp M^{-1/2} P),
    P = K + kappa^2 M_lump.
    """

    def __init__(
        self,
        nodes,
        elements,
        mu=np.log(1e-3),
        kappa=2.0,
        sigma=2.0,
        cond=(2.75, 2.75, 2.25),
    ):
        self.nodes = nodes
        self.elements = elements

        self.dim = nodes.shape[0]
        self.mu = np.full(self.dim, float(mu))

        lxx = np.array([
            [ 2, -2, -1,  1],
            [-2,  2,  1, -1],
            [-1,  1,  2, -2],
            [ 1, -1, -2,  2],
        ], dtype=float) / 6

        lyy = np.array([
            [ 2,  1, -1, -2],
            [ 1,  2, -2, -1],
            [-1, -2,  2,  1],
            [-2, -1,  1,  2],
        ], dtype=float) / 6

        lxy = np.array([
            [ 1,  0, -1,  0],
            [ 0, -1,  0,  1],
            [-1,  0,  1,  0],
            [ 0,  1,  0, -1],
        ], dtype=float) / 2

        lm = np.array([
            [4, 2, 1, 2],
            [2, 4, 2, 1],
            [1, 2, 4, 2],
            [2, 1, 2, 4],
        ], dtype=float) / 36

        rows = []
        cols = []
        vals = []

        ml = np.zeros(self.dim)
        local = cond[0] * lxx + cond[1] * lyy + cond[2] * lxy

        for e in elements:
            dx = nodes[e[2]] - nodes[e[0]]
            det = abs(dx[0] * dx[1])

            ml[e] += np.sum(lm * det, axis=1)

            rows.extend(np.repeat(e, 4))
            cols.extend(np.tile(e, 4))
            vals.extend(local.ravel())

        K = sparse.coo_matrix(
            (vals, (rows, cols)),
            shape=(self.dim, self.dim),
        ).tocsc()

        self.ml = ml
        self.P = (K + sparse.diags(kappa**2 * ml)).tocsc()

        # SuiteSparse CHOLMOD
        self.factor = cholesky(self.P)

        nor = gamma(1) / (gamma(2) * kappa**2 * 4 * np.pi)
        self.tmp = np.sqrt(nor) / sigma

    def random(self, n=1):
        z = np.random.randn(n, self.dim)
        return z[0] if n == 1 else z

    def matvec_prior_L(self, xi):
        xi = np.asarray(xi, float)

        if xi.ndim == 1:
            rhs = np.sqrt(self.ml) * xi / self.tmp
            return self.factor(rhs)

        return np.stack([self.matvec_prior_L(v) for v in xi])

    def matvec_prior_LT(self, x):
        x = np.asarray(x, float)

        if x.ndim == 1:
            return np.sqrt(self.ml) * self.factor(x) / self.tmp

        return np.stack([self.matvec_prior_LT(v) for v in x])

    def matvec_prior_invL(self, x):
        x = np.asarray(x, float)

        if x.ndim == 1:
            return self.tmp * (self.P @ x) / np.sqrt(self.ml)

        return np.stack([self.matvec_prior_invL(v) for v in x])


def _gaussian_source(nodes, centre, sigma):
    dist2 = np.sum((nodes - centre) ** 2, axis=1)
    return np.exp(-0.5 * dist2 / sigma**2) / (2 * np.pi * sigma**2)

def _build_source(nodes, lx, ly, tol=0.05):
    f1 = (
        2.0 * _gaussian_source(nodes, [lx - 0.02, 0.02], tol)
        + 4.0 * _gaussian_source(nodes, [lx - 0.02, ly - 0.02], tol)
    )
    f2 = (
        3.0 * _gaussian_source(nodes, [0.02, 0.02], tol)
        + 3.0 * _gaussian_source(nodes, [0.02, ly - 0.02], tol)
    )
    return f1 - f2 * f1.sum() / f2.sum()

class EllipticPDESolver(SolverAdjointCache):
    """
    Q1 FEM for

        -div(k grad u) = f,

    where k = exp(x).
    """

    def __init__(
        self,
        nx=120,
        ny=40,
        lx=3.0,
        ly=1.0,
        sensor_locations=None,
    ):
        super().__init__()

        (
            self.nodes,
            self.elements,
            self.boundary_edges,
            self.gx,
            self.gy,
        ) = structured_mesh(nx, ny, lx, ly)

        self.state_dim = len(self.nodes)
        self.n_parameter = len(self.elements)
        self.n_qoi = 0

        self.w = [
            np.array([1, -1, -1, 1]) / 2,
            np.array([-1, -1, 1, 1]) / 2,
            np.array([1, -1, 1, -1]) / 2 * np.sqrt(2 / 3),
        ]

        self._build_fem_matrices()

        f = _build_source(self.nodes, lx, ly)
        self.rhs = 1e-5 * (self.M @ f)

        if sensor_locations is None:
            x1, y1 = np.meshgrid(
                np.arange(0.3, 0.7001, 0.2),
                np.arange(0.2, 0.7001, 0.2),
            )

            x2, y2 = np.meshgrid(
                np.arange(2.4, 2.6001, 0.2),
                np.arange(0.4, 0.6001, 0.2),
            )

            sensor_locations = np.c_[
                np.r_[x1.ravel(), x2.ravel()],
                np.r_[y1.ravel(), y2.ravel()],
            ]

        self.sensor_locations = np.asarray(sensor_locations)

        self.sensor_idx = np.array([
            np.argmin(np.sum((self.nodes - q) ** 2, axis=1))
            for q in self.sensor_locations
        ])

        self.n_data = len(self.sensor_idx)

        self._x = None
        self._k = None
        self._u = None
        self._lu = None

    def _build_fem_matrices(self):
        nnode = self.state_dim
        nel = self.n_parameter

        # W1, W2, W3
        rows = self.elements.reshape(-1)
        cols = np.repeat(np.arange(nel), 4)

        self.W1 = sparse.coo_matrix(
            (np.tile(self.w[0], nel), (rows, cols)),
            shape=(nnode, nel),
        ).tocsc()

        self.W2 = sparse.coo_matrix(
            (np.tile(self.w[1], nel), (rows, cols)),
            shape=(nnode, nel),
        ).tocsc()

        self.W3 = sparse.coo_matrix(
            (np.tile(self.w[2], nel), (rows, cols)),
            shape=(nnode, nel),
        ).tocsc()

        # Mass matrix
        lm = np.array([
            [4, 2, 1, 2],
            [2, 4, 2, 1],
            [1, 2, 4, 2],
            [2, 1, 2, 4],
        ], dtype=float) / 36

        rows_M = []
        cols_M = []
        vals_M = []

        for e in self.elements:
            dx = self.nodes[e[2]] - self.nodes[e[0]]
            detJ = abs(dx[0] * dx[1])

            rows_M.extend(np.repeat(e, 4))
            cols_M.extend(np.tile(e, 4))
            vals_M.extend((lm * detJ).ravel())

        self.M = sparse.coo_matrix(
            (vals_M, (rows_M, cols_M)),
            shape=(nnode, nnode),
        ).tocsr()

        # Boundary mean penalty
        nbndf = len(self.boundary_edges)
        c = np.zeros(nnode)

        np.add.at(c, self.boundary_edges[:, 0], 1.0 / nbndf)
        np.add.at(c, self.boundary_edges[:, 1], 1.0 / nbndf)

        c = sparse.csc_matrix(c[:, None])
        self.penalty = (nel * (c @ c.T)).tocsc()

    def _assemble(self, k):
        A = (
            self.W1.multiply(k) @ self.W1.T
            + self.W2.multiply(k) @ self.W2.T
            + self.W3.multiply(k) @ self.W3.T
            + self.penalty
        )

        return A.tocsc()

    def _solve(self, x):
        k = np.exp(x)
        A = self._assemble(k)

        lu = splu(A)
        u = lu.solve(self.rhs)

        return u[self.sensor_idx]

    def solve(self, x):
        x = np.asarray(x, float)

        if x.ndim == 1:
            return self._solve(x), None

        return np.stack([self._solve(v) for v in x]), None

    def solve_and_cache(self, x):
        self._x = np.asarray(x, float)
        self._k = np.exp(self._x)

        A = self._assemble(self._k)

        self._lu = splu(A)
        self._u = self._lu.solve(self.rhs)

        return self._u, None

    def apply_observation(self, state):
        return np.asarray(state)[self.sensor_idx]

    def apply_QoI(self, state, params):
        return np.asarray(state)

    def _linearized_observation(self, delta_x):
        delta_k = self._k * delta_x

        W1u = self.W1.T @ self._u
        W2u = self.W2.T @ self._u
        W3u = self.W3.T @ self._u

        rhs = -(
            self.W1 @ (delta_k * W1u)
            + self.W2 @ (delta_k * W2u)
            + self.W3 @ (delta_k * W3u)
        )

        du = self._lu.solve(rhs)

        return du[self.sensor_idx]

    def linearized_observation(self, delta_x):
        delta_x = np.asarray(delta_x, float)

        if delta_x.ndim == 1:
            return self._linearized_observation(delta_x)

        return np.stack([
            self._linearized_observation(v)
            for v in delta_x
        ])

    def _linearized_observation_transpose(self, delta_y):
        r = np.zeros(self.state_dim)
        r[self.sensor_idx] = delta_y

        lam = self._lu.solve(r, "T")

        W1u = self.W1.T @ self._u
        W2u = self.W2.T @ self._u
        W3u = self.W3.T @ self._u

        W1lam = self.W1.T @ lam
        W2lam = self.W2.T @ lam
        W3lam = self.W3.T @ lam

        grad_k = -(W1lam * W1u + W2lam * W2u + W3lam * W3u)

        return self._k * grad_k

    def linearized_observation_transpose(self, delta_y):
        delta_y = np.asarray(delta_y, float)

        if delta_y.ndim == 1:
            return self._linearized_observation_transpose(delta_y)

        return np.stack([
            self._linearized_observation_transpose(v)
            for v in delta_y
        ])

    def linearized_qoi_transpose(self):
        raise NotImplementedError


def load_elliptic_mrf_problem(data_path=None):
    solver = EllipticPDESolver()

    # Original prior lives on the dual mesh:
    # one parameter per fine element.
    centres = solver.nodes[solver.elements].mean(axis=1)

    nx = len(solver.gx) - 1
    ny = len(solver.gy) - 1

    dual = []

    for i in range(nx - 1):
        for j in range(ny - 1):
            n0 = i * ny + j

            dual.append([
                n0,
                n0 + ny,
                n0 + ny + 1,
                n0 + 1,
            ])
    prior = MRFPrecisionPrior(
        centres,
        np.asarray(dual, dtype=int),
    )

    if data_path is not None:
        d = loadmat(
            data_path,
            squeeze_me=True,
        )

        obs = np.asarray(d["data"]).reshape(-1)
        std = np.asarray(d["std"])

        if std.size == 1:
            sigma = float(std.reshape(-1)[0])
        else:
            sigma = std.reshape(-1)

    else:
        obs = np.zeros(solver.n_data)
        sigma = np.ones(solver.n_data)

    like = GaussianLikelihood(
        obs,
        sigma,
    )

    return solver, like, prior