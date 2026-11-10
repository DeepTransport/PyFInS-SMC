import numpy as np
from mpi4py import MPI

from FenicsxModels.SingularValueSolvers import svd_distributed_array
from InferenceTools.Utilities import bound_based_truncation

class LikelihoodInformedSubspace:
    """
    Likelihood-informed subspace in distributed whitened coordinates.

    The full whitened parameter is stored by owned-local blocks on each MPI
    rank.  The LIS basis is stored in the same way:

        V_local.shape = (nV_owned, n_sub).

    Reduced LIS coordinates are replicated:

        xi_sub = sum_r xi_local,r @ V_local,r.

    Hence projection requires one MPI Allreduce, while reconstruction is local.
    """

    def __init__(
        self,
        prior,
        *,
        epsilon=0.5,
        criterion="trace",
        option="Lanczos",
        max_rank=100,
        p=10,
        n_power=0,
        tol=1.0e-12,
        max_it=1000,
        root=0,
    ):
        self.prior = prior
        self.comm = prior.comm
        self.root = root

        self.epsilon = epsilon
        self.criterion = criterion

        self.option = option
        self.max_rank = max_rank
        self.p = p
        self.n_power = n_power
        self.tol = tol
        self.max_it = max_it

        self.V_local = None
        self.eigvals = None

        self.n_sub = 0

    def copy_empty(self, *, criterion=None):
        return LikelihoodInformedSubspace(
            self.prior,
            epsilon=self.epsilon,
            criterion=self.criterion if criterion is None else criterion,
            option=self.option,
            max_rank=self.max_rank,
            p=self.p,
            n_power=self.n_power,
            tol=self.tol,
            max_it=self.max_it,
            root=self.root,
        )

    def assign(self, V_local, eigvals):
        self.V_local = V_local
        self.eigvals = eigvals
        self.n_sub = V_local.shape[1]

    def build_from_grads(self, A_local):
        """
        Build the LIS from a matrix of whitened gradients stored by distributed
        column blocks:

            A_local.shape = (n_gradients, nV_owned).
        """
        m = A_local.shape[0]
        k_svd = min(m, self.prior.nV_global)
        k_svd = min(k_svd, self.max_rank)

        if k_svd == 0:
            self.assign(np.empty((self.prior.nV_owned, 0)), np.empty(0))
            return

        _, singular_values, V_local = svd_distributed_array(
            A_local,
            k=k_svd,
            comm=self.comm,
            root=self.root,
            option=self.option,
            p=self.p,
            n_power=self.n_power,
            tol=self.tol,
            max_it=self.max_it,
        )

        eigvals = singular_values * singular_values
        k = bound_based_truncation(eigvals, self.epsilon, self.criterion)
        self.assign(V_local[:, :k], eigvals[:k])

    def projection_coeff(self, xi_local):
        """
        Project one or more distributed whitened parameters onto the LIS.

        Input
        -----
        single : (nV_owned,)
        batch  : (N, nV_owned)

        Output
        ------
        single : (n_sub,)
        batch  : (N, n_sub)

        The output is replicated on every rank.
        """
        if self.n_sub == 0:
            return np.zeros((*xi_local.shape[:-1], 0))
        coeff_local = xi_local @ self.V_local
        coeff = np.empty_like(coeff_local)
        self.comm.Allreduce(coeff_local, coeff, op=MPI.SUM)
        return coeff

    def projection_null(self, xi_local):
        xi_sub = self.projection_coeff(xi_local)
        if self.n_sub == 0:
            return xi_local.copy(), xi_sub
        xi_null = xi_local - xi_sub @ self.V_local.T
        return xi_null, xi_sub

    def fill_in(self, xi_sub, xi_null_local):
        if self.n_sub == 0:
            return xi_null_local.copy()
        return xi_sub @ self.V_local.T + xi_null_local

    def orthogonality_error(self):
        if self.n_sub == 0:
            return 0.0
        gram_local = self.V_local.T @ self.V_local
        gram = np.empty_like(gram_local)
        self.comm.Allreduce(gram_local, gram, op=MPI.SUM)
        return np.linalg.norm(gram - np.eye(self.n_sub))

    def save(self, filename):
        """
        Gather the distributed LIS basis to root and save it.

        The saved basis is ordered by global FEM dof index, so it can be
        redistributed according to the current MPI partition when loaded.
        """
        index_map = self.prior.V.dofmap.index_map

        local_ind = np.arange(index_map.size_local)
        global_ind = index_map.local_to_global(local_ind)

        global_inds = self.comm.gather(global_ind, root=self.root)
        V_locals = self.comm.gather(self.V_local, root=self.root)

        if self.comm.rank == self.root:
            V_global = np.empty((self.prior.nV_global, self.n_sub))
            for ind, V in zip(global_inds, V_locals):
                V_global[ind] = V
            np.savez_compressed(filename, V=V_global, eigvals=self.eigvals)

    def load(self, filename):
        """
        Load a globally stored LIS basis and distribute it according to the current FEM ownership.
        """
        if self.comm.rank == self.root:
            with np.load(filename) as data:
                V_global = data["V"]
                eigvals = data["eigvals"]
            if V_global.shape[0] != self.prior.nV_global:
                raise ValueError("Saved LIS and current prior have different global dimensions.")
        else:
            V_global = None
            eigvals = None

        index_map = self.prior.V.dofmap.index_map

        local_ind = np.arange(index_map.size_local, dtype=np.int32)
        global_ind = index_map.local_to_global(local_ind)

        global_inds = self.comm.gather(global_ind, root=self.root)

        if self.comm.rank == self.root:
            V_locals = [V_global[ind] for ind in global_inds]
        else:
            V_locals = None

        V_local = self.comm.scatter(V_locals, root=self.root)
        eigvals = self.comm.bcast(eigvals, root=self.root)

        self.assign(V_local, eigvals)
