
import numpy as np
from InferenceTools.Utilities import bound_based_truncation, TruncationCriteria, LikelihoodInformedSubspaceType

class LikelihoodInformedSubspace:
    def __init__(
        self, 
        *,
        epsilon=0.5,
        criterion:TruncationCriteria="trace",
        type:LikelihoodInformedSubspaceType="Bases",
    ):
        self.type = type
        self.epsilon = epsilon
        self.V = None
        self.eigvals = None
        self.active_idx = None
        self.diags = None
        self.criterion = criterion

    def assign(self, V, s):
        if self.type == "Canonical":
            if V.ndim != 1:
                raise ValueError("For Canonical type, V must be a 1D array.")
            self.active_idx = np.flatnonzero(V)
            self.n_dim = V.size
            self.n_sub = self.active_idx.size
            self.diags = np.asarray(s)
        else:
            if V.ndim != 2:
                raise ValueError("V must be a 2D array.")
            self.V = V
            self.n_dim, self.n_sub = V.shape
            self.eigvals = s

    def build_from_grads(self, A):
        if self.type == "Canonical":
            diagonal = np.sum(A**2, axis=0)
            if np.any(diagonal < 0):
                raise ValueError("Diagonal entries must be nonnegative.")
            selected = np.zeros(diagonal.size, dtype=bool)
            order = np.argsort(diagonal)[::-1]
            k = bound_based_truncation(diagonal[order], self.epsilon, self.criterion)
            selected[order[:k]] = True
            self.assign(selected, diagonal)
        else:
            V, s, _ = np.linalg.svd(A.T, full_matrices=False)
            eigvals = s*s
            k = bound_based_truncation(eigvals, self.epsilon, self.criterion)
            k = min(k, V.shape[1])
            self.assign(V[:,:k], eigvals)

    def build_from_gram(self, H):
        if self.type == "Canonical":
            diagonal = np.diag(H)
            if np.any(diagonal < 0):
                raise ValueError("Diagonal entries must be nonnegative.")
            selected = np.zeros(diagonal.size, dtype=bool)
            order = np.argsort(diagonal)[::-1]
            k = bound_based_truncation(diagonal[order], self.epsilon, self.criterion)
            selected[order[:k]] = True
            self.assign(selected, diagonal)
        else:
            eigvals, V = np.linalg.eigh(H)
            eigvals = np.maximum(eigvals, 0.0)
            ind = np.argsort(eigvals)[::-1]
            eigvals = eigvals[ind]
            V = V[:, ind]
            k = bound_based_truncation(eigvals, self.epsilon, self.criterion)
            k = min(k, V.shape[1])
            self.assign(V[:,:k], eigvals)

    def projection_coeff(self, xi):
        if self.type == "Canonical":
            if xi.ndim == 1:
                return xi[self.active_idx]
            return xi[:, self.active_idx]
        return xi @ self.V

    def projection_null(self, xi):
        xi_proj_coeff = self.projection_coeff(xi)
        if self.type == "Canonical":
            xi_null = xi.copy()
            if xi.ndim == 1:
                xi_null[self.active_idx] = 0.0
            else:
                xi_null[:, self.active_idx] = 0.0
        else:
            xi_null = xi - xi_proj_coeff @ self.V.T
        return xi_null, xi_proj_coeff

    def fill_in(self, xi_sub, xi_null):
        if self.type == "Canonical":
            xi = xi_null.copy()
            if xi.ndim == 1:
                xi[self.active_idx] = xi_sub
            else:
                xi[:,self.active_idx] = xi_sub
            return xi
        return xi_sub @ self.V.T + xi_null


    def save(self, filename):
        """
        Save the likelihood-informed subspace.
        """
        np.savez_compressed(filename, V=self.V, eigvals=self.eigvals)


    def load(self, filename):
        """
        Load the likelihood-informed subspace.
        """
        with np.load(filename) as data:
            V = data["V"]
            eigvals = data["eigvals"]
        self.assign(V, eigvals)