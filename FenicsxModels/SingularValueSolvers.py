from mpi4py import MPI
import numpy as np

from petsc4py import PETSc
from slepc4py import SLEPc

from BaseModels import SolverAdjointCache, LikelihoodBase, GaussianPriorBase


from typing import Literal
SolverType = Literal["Lanczos", "Random"]

# ============================================================================
# Generic matrix-free randomized SVD
# ============================================================================

def randomized_svd(
    A_mv,
    AT_mv,
    Omega,
    m,
    n_local,
    k,
    comm = MPI.COMM_WORLD,
    n_power: int = 0,     # optional power iterations on A A^T
    root: int = 0,
):
    """
    Randomized SVD of

        A : R^n -> R^m,

    where the parameter space is distributed across MPI ranks.

    A_mv(v_local)
        returns A v as a replicated vector of length m.

    AT_mv(y)
        returns the local part of A.T y.

    Omega
        has shape (l, n_local).

    Returns
    -------
    U : (m, k)
        Replicated left singular vectors.

    S : (k,)
        Singular values.

    V_local : (n_local, k)
        Local part of the right singular vectors.
    """
    l = Omega.shape[0]
    # ------------------------------------------------------------------
    # Range finder: Y = A Omega
    # ------------------------------------------------------------------
    Y = np.empty((m, l)) if comm.rank == root else None
    for j in range(l):
        y = A_mv(Omega[j])
        if comm.rank == root:
            Y[:, j] = y
    if comm.rank == root:
        Q, _ = np.linalg.qr(Y, mode="reduced")
    else:
        Q = np.empty((m, l))
    comm.Bcast(Q, root=root)

    # ------------------------------------------------------------------
    # Power iterations
    # ------------------------------------------------------------------
    # Apply A A.T to a small block
    def apply_AAT(Q):
        W = np.empty((m, l)) if comm.rank == root else None
        for j in range(l):
            z_local = AT_mv(Q[:, j])
            w = A_mv(z_local)
            if comm.rank == root:
                W[:, j] = w
        return W

    for _ in range(n_power):
        W = apply_AAT(Q)
        if comm.rank == root:
            Q, _ = np.linalg.qr(W, mode="reduced")
        comm.Bcast(Q, root=root)

    # ------------------------------------------------------------------
    # Projected eigenproblem
    # ------------------------------------------------------------------

    W = apply_AAT(Q)
    if comm.rank == root:
        T = Q.T @ W
        T = 0.5 * (T + T.T)
        eigvals, eigvecs = np.linalg.eigh(T)
        ind = np.argsort(eigvals)[::-1]
        eigvals = eigvals[ind]
        eigvecs = eigvecs[:, ind]
        S = np.sqrt(np.maximum(eigvals[:k], 0.0))
        U = Q @ eigvecs[:, :k]
    else:
        U = None
        S = None
    U = comm.bcast(U, root=root)
    S = comm.bcast(S, root=root)

    # ------------------------------------------------------------------
    # Recover right singular vectors
    #
    #     v_i = A.T u_i / sigma_i
    # ------------------------------------------------------------------

    V_local = np.empty((n_local, k))
    for i in range(k):
        if S[i] > 0.0:
            V_local[:, i] = AT_mv(U[:, i]) / S[i]
        else:
            V_local[:, i] = 0.0
    return U, S, V_local


# ============================================================================
# Generic matrix-free SLEPc SVD
# ============================================================================

def slepc_svd(
    A_mv,
    AT_mv,
    m,
    n_local,
    n_global,
    k,
    comm=MPI.COMM_WORLD,
    tol=1.0e-12,
    max_it=1000,
):
    """
    Direct matrix-free SVD using SLEPc.

    The rectangular operator is

        A : R^n -> R^m.

    The input space is distributed across MPI ranks. The output of A_mv
    is replicated, while PETSc distributes the data-space vector internally.

    Returns
    -------
    U : (m, k)
        Replicated left singular vectors.

    S : (k,)
        Singular values.

    V_local : (n_local, k)
        Local part of the right singular vectors.
    """

    class MatrixContext:
        def mult(self, A, x, y):
            Av = A_mv(x.array_r)
            start, end = y.getOwnershipRange()
            y.array[:] = Av[start:end]
        def multTranspose(self, A, x, y):
            u = np.concatenate(comm.allgather(x.array_r.copy()))
            y.array[:] = AT_mv(u)

    # PETSc distributes the m output rows automatically.
    A = PETSc.Mat().createPython(
        size=((PETSc.DECIDE, m), (n_local, n_global)),
        context=MatrixContext(),
        comm=comm,
    )
    A.setUp()

    # ------------------------------------------------------------------
    # Direct rectangular SVD
    # ------------------------------------------------------------------

    svd = SLEPc.SVD().create(comm=comm)
    svd.setOperators(A)
    svd.setType(SLEPc.SVD.Type.TRLANCZOS)
    svd.setWhichSingularTriplets(SLEPc.SVD.Which.LARGEST)
    svd.setDimensions(nsv=k)
    svd.setTolerances(tol=tol, max_it=max_it)
    # Allows command-line overrides such as
    #   -svd_tol 1e-13
    #   -svd_ncv 30
    #   -svd_view
    svd.setFromOptions()
    svd.solve()

    nconv = svd.getConverged()

    if nconv < k:
        raise RuntimeError(
            f"SLEPc converged {nconv} singular triplets, "
            f"but {k} were requested."
        )
    # Mat.createVecs() returns
    #     right vector : dimension n
    #     left vector  : dimension m
    v, u = A.createVecs()

    U = np.empty((m, k))
    S = np.empty(k)
    V_local = np.empty((n_local, k))
    for i in range(k):
        S[i] = svd.getSingularTriplet(i, u, v)
        U[:, i] = np.concatenate(comm.allgather(u.array.copy()))
        V_local[:, i] = v.array

    svd.destroy()
    A.destroy()

    return U, S, V_local


# ============================================================================
# Distributed SVD of explicit matrices
# ============================================================================

def _distributed_array_actions(A_local, comm):
    """
    A is stored by distributed column blocks:

        A = [A_0, A_1, ..., A_{P-1}].

    A_local has shape

        (m, n_local).
    """

    def A_mv(v_local):
        y_local = A_local @ v_local
        y = np.empty_like(y_local)
        comm.Allreduce(y_local, y, op=MPI.SUM)
        return y

    def AT_mv(y):
        return A_local.T @ y

    return A_mv, AT_mv

def svd_distributed_array(
    A_local,
    k: int,               # target rank
    comm=MPI.COMM_WORLD,
    root: int = 0,
    *,
    option: SolverType = "Lanczos",
    p: int = 10,          # oversampling
    n_power: int = 0,     # optional power iterations on A A^T
    tol:float = 1.0e-12,
    max_it: int = 1000,
):
    """
    Randomized SVD of an explicit matrix stored by distributed columns.
    """
    A_mv, AT_mv = _distributed_array_actions(A_local, comm)
    m = A_local.shape[0]
    n_local = A_local.shape[1]

    if option == "Random":
        l = min(k + p, m)
        Omega = np.random.standard_normal((l, n_local))
        return randomized_svd(
            A_mv,
            AT_mv,
            Omega,
            m,
            n_local,
            k,
            comm=comm,
            n_power=n_power,
            root=root,
        )
    else:
        n_global = comm.allreduce(n_local, op=MPI.SUM)
        return slepc_svd(
            A_mv,
            AT_mv,
            m,
            n_local,
            n_global,
            k,
            comm=comm,
            tol=tol,
            max_it=max_it,
        )

# ============================================================================
# Prior-conditioned, Fisher-weighted Jacobian
# ============================================================================

def _preconditioned_jacobian_actions(
    prior: GaussianPriorBase,
    pde_solver: SolverAdjointCache,
    like: LikelihoodBase,
    xi_owned,
):
    """
    Matrix-free actions for

        A = sqrt(F) J L,

    evaluated at xi.

    Here

        L       : prior square-root operator,
        J       : linearized observation operator,
        F       : diagonal Fisher information.

    Returns
    -------
    A_mv
        A v.

    AT_mv
        A.T y.

    m
        Number of active observations.
    """
    k_log_owned = prior.transform(xi_owned)
    Y, _ = pde_solver.solve_and_cache(k_log_owned)
    diag_Fisher = like.eval_potential_Fisher(Y)

    mask = like.mask
    m = np.count_nonzero(mask)
    sqrt_Fisher = np.sqrt(diag_Fisher[mask])
    def A_mv(v_local):
        dk_log = prior.matvec_prior_L(v_local)
        Jv = pde_solver.linearized_observation(dk_log)
        return Jv[mask] * sqrt_Fisher
    def AT_mv(dy):
        dy_full = np.zeros(pde_solver.n_data)
        dy_full[mask] = dy * sqrt_Fisher
        g_log = pde_solver.linearized_observation_transpose(dy_full)
        return prior.matvec_prior_LT(g_log)

    return A_mv, AT_mv, m


def preconditioned_svd_at_xi(
    prior: GaussianPriorBase,
    pde_solver: SolverAdjointCache,
    like: LikelihoodBase,
    xi_owned,
    k: int,               # target rank
    root: int = 0,
    *,
    option: SolverType = "Lanczos",
    p: int = 10,          # oversampling
    n_power: int = 0,     # optional power iterations on A A^T
    tol: float = 1.0e-12,
    max_it: int = 1000,
):
    comm = prior.comm
    A_mv, AT_mv, m = _preconditioned_jacobian_actions(
        prior,
        pde_solver,
        like,
        xi_owned,
    )
    n_local = prior.nV_owned

    if option == "Random":
        l = min(k + p, m)
        Omega = prior.random(l)
        return randomized_svd(
            A_mv,
            AT_mv,
            Omega,
            m,
            n_local,
            k,
            comm=comm,
            n_power=n_power,
            root=root,
        )
    else:
        n_global = prior.nV_global
        return slepc_svd(
            A_mv,
            AT_mv,
            m,
            n_local,
            n_global,
            k,
            comm=comm,
            tol=tol,
            max_it=max_it,
        )
