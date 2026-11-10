from typing import Union
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import ndtr, ndtri  # for standard Gaussian CDF
from scipy.special import gamma, kv
from scipy.linalg import solve_triangular

from BaseModels import PriorBase, GaussianPriorBase, KernelBase

class UniformPrior(PriorBase):
    """
    Uniform prior on a hypercube
    """

    def __init__(
        self,
        lower: Union[float, np.ndarray],
        upper: Union[float, np.ndarray],
        seed=12345,
    ):
        # no meaningful Gaussian mean, but keep base interface
        self.mu = 0.0
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        if self.lower.size != self.upper.size:
            raise ValueError("lower and upper must have the same shape.")
        self.dim = self.upper.size
        self.rng = np.random.default_rng(seed)

    def random(self, n: int = 1) -> np.ndarray:
        """
        Draw iid white-noise samples xi ~ N(0, I).

        Returns:
            shape (dim,) if n == 1
            shape (n, dim) if n > 1
        """
        xi = self.rng.standard_normal((n,self.dim))
        if n == 1:
            return xi[0]
        return xi

    def transform(self, xi: np.ndarray) -> np.ndarray:
        """
        Transform iid Gaussian xi into uniform samples on the box.

        If xi has shape (dim,), returns shape (dim,).
        If xi has shape (n, dim), returns shape (n, dim).
        """
        u = ndtr(xi)  # Phi(xi), maps R -> (0, 1)
        if xi.ndim == 1:
            return self.lower + (self.upper - self.lower) * u
        return self.lower[None, :] + (self.upper - self.lower)[None, :] * u
    
    def whiten(self, x: np.ndarray) -> np.ndarray:
        """
        Transform uniform samples on the box into iid Gaussian
        """
        if x.ndim == 1:
            u = (x - self.lower) / (self.upper - self.lower)
        else:
            u = (x - self.lower[None, :]) / (self.upper[None, :] - self.lower[None, :])
        # Avoid ndtri(0) = -inf and ndtri(1) = inf
        eps = np.finfo(float).eps
        u = np.clip(u, eps, 1.0 - eps)
        return ndtri(u)

    
class GaussianPriorNP(GaussianPriorBase):
    """
    Base class for Gaussian prior N(mu, cov) using Cholesky
    """

    def __init__(self, type:float=-1.0, seed=12345):
        self.type = type
        self.mu = None
        self.cov = None
        self.sqrt_lambda = None
        self.V = None
        self.L = None
        self.dim = None
        self.kl_dim = None
        self.rng = np.random.default_rng(seed)

    def build(self, mu, cov):
        self.cov = cov
        self.dim = self.cov.shape[0]
        self.kl_dim = self.dim
        mu = np.asarray(mu, dtype=float)
        if mu.ndim == 0:
            self.mu = np.full(self.dim, float(mu))
        else:
            self.mu = mu
        jitter = 1e-12
        if self.type < 0:
            self.L = np.linalg.cholesky(self.cov + jitter * np.eye(self.dim))
        else:
            if self.type >= 1.0:
                eigvals, eigvecs = np.linalg.eigh(self.cov + jitter * np.eye(self.dim))
                inds = np.argsort(eigvals)[::-1]
                self.V = eigvecs[:,inds]
                self.sqrt_lambda = np.sqrt(eigvals[inds])
            else:
                eigvals, eigvecs = np.linalg.eigh(self.cov)
                inds = np.argsort(eigvals)[::-1]
                eigvals = eigvals[inds]
                eigvecs = eigvecs[:, inds]
                cumulative_energy = np.cumsum(eigvals)/eigvals.sum()
                k = int(np.searchsorted(cumulative_energy, self.type) + 1)
                self.V = eigvecs[:,:k]
                self.sqrt_lambda = np.sqrt(eigvals[:k])
                self.kl_dim = k

    def matvec_prior_L(self, xi: np.ndarray) -> np.ndarray:
        # xi are iid Gaussian if GMRF
        if self.type < 0:
            if xi.ndim == 1:
                return self.L @ xi
            return xi @ self.L.T
        else:
            if xi.ndim == 1:
                return self.V @ (self.sqrt_lambda * xi)
            return (xi * self.sqrt_lambda ) @ self.V.T 

    def matvec_prior_LT(self, x: np.ndarray) -> np.ndarray:
        # x are nodal values if GMRF
        if self.type < 0:
            if x.ndim == 1:
                return self.L.T @ x
            return x @ self.L
        else:
            if x.ndim == 1:
                return (self.V.T @ x) * self.sqrt_lambda
            return (x @ self.V) * self.sqrt_lambda

    def matvec_prior_invL(self, x: np.ndarray) -> np.ndarray:
        # x are nodal values if GMRF
        if self.type < 0:
            if x.ndim == 1:
                return solve_triangular(self.L, x, lower=True)
            return solve_triangular(self.L, x.T, lower=True).T
        else:
            if x.ndim == 1:
                return (self.V.T @ x) / self.sqrt_lambda
            return (x @ self.V) / self.sqrt_lambda
    
    def random(self, n: int = 1) -> np.ndarray:
        """
        Draw iid white-noise samples xi ~ N(0, I).

        Returns:
            shape (dim,) if n == 1
            shape (n, dim) if n > 1
        """
        if n == 1:
            return self.rng.standard_normal(np.minimum(self.dim, self.kl_dim))
        else:
            return self.rng.standard_normal.standard_normal( (n, np.minimum(self.dim, self.kl_dim)) )

class GaussianPrior(GaussianPriorNP):
    """
    Gaussian prior N(mu, cov)
    """

    def __init__(
        self,
        mu: Union[float, np.ndarray],
        cov: np.ndarray,
        type: float = -1.0,
        seed=12345,
    ):
        super().__init__(type, seed=seed)
        self.build(mu, cov)


def _pairwise_distance(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    if xs.ndim == 1:
        xs = xs[:, None]
    elif xs.ndim != 2:
        raise ValueError("xs must have shape (n_points,) or (n_points, dimension).")
    if ys.ndim == 1:
        ys = ys[:, None]
    elif ys.ndim != 2:
        raise ValueError("ys must have shape (n_points,) or (n_points, dimension).")
    if xs.shape[1] != ys.shape[1]:
        raise ValueError("xs and ys must have the same spatial dimension.")
    return cdist(xs, ys, metric="euclidean")

class MaternKernel(KernelBase):
    """
    Matérn kernel
        k(x, y) = variance * 2^(1-nu) / Gamma(nu) * z^nu K_nu(z),
    where
        z = sqrt(2 nu) ||x-y|| / length_scale.
    Parameters
    ----------
    length_scale
        Positive correlation length.
    nu
        Positive smoothness parameter.
    variance
        Marginal variance k(x, x).
    """

    def __init__(self, 
        length_scale: float = 1.0, 
        nu: float = 1.5, 
        variance: float = 1.0
    ) -> None:
        if length_scale <= 0:
            raise ValueError("length_scale must be positive.")
        if nu <= 0:
            raise ValueError("nu must be positive.")
        if variance < 0:
            raise ValueError("variance must be nonnegative.")
        self.length_scale = float(length_scale)
        self.nu = float(nu)
        self.variance = float(variance)

    def eval(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        distances = _pairwise_distance(xs, ys)
        z = np.sqrt(2.0 * self.nu) * distances / self.length_scale
        covariance = np.empty_like(z)
        zero = z == 0.0
        covariance[zero] = self.variance
        nonzero = ~zero
        z_nonzero = z[nonzero]
        covariance[nonzero] = (self.variance*2.0**(1.0-self.nu)/gamma(self.nu)) * (z_nonzero**self.nu) * kv(self.nu, z_nonzero)
        return covariance

class PoweredExponentialKernel(KernelBase):
    """
    Powered exponential kernel

        k(x, y) = variance * exp[-(||x-y|| / length_scale)^power].
    where
        1 <= power <= 2.

    Special cases
    -------------
    power = 1
        Exponential kernel.
    power = 2
        Squared-exponential kernel.
    """

    def __init__(self, 
        length_scale: float = 1.0, 
        power: float = 1.0, 
        variance: float = 1.0
    ) -> None:
        if length_scale <= 0:
            raise ValueError("length_scale must be positive.")
        if not 1 <= power <= 2:
            raise ValueError("power must satisfy 1 <= power <= 2.")
        if variance < 0:
            raise ValueError("variance must be nonnegative.")

        self.length_scale = float(length_scale)
        self.power = float(power)
        self.variance = float(variance)

    def eval(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        distances = _pairwise_distance(xs, ys)
        return self.variance * np.exp(-(distances / self.length_scale)**self.power)

class ExponentialKernel(PoweredExponentialKernel):
    """
    Ordinary exponential kernel
        k(x, y) = variance * exp(-||x-y|| / length_scale).
    """
    def __init__(self, length_scale: float = 1.0, variance: float = 1.0) -> None:
        super().__init__(length_scale=length_scale, power=1.0, variance=variance)

class GaussianKernel(PoweredExponentialKernel):
    """
    Ordinary exponential kernel
        k(x, y) = variance * exp(-||x-y|| / length_scale).
    """
    def __init__(self, length_scale: float = 1.0, variance: float = 1.0) -> None:
        super().__init__(length_scale=length_scale, power=2.0, variance=variance)

class GaussianProcessPrior(GaussianPriorNP):
    """
    Gaussian prior N(mu, cov)
    """

    def __init__(
        self,
        xs: np.ndarray,
        mu: Union[float, np.ndarray],
        kernel: KernelBase,
        type: float = -1.0,
        seed=12345,
    ):
        super().__init__(type, seed=seed)
        self.xs = np.asarray(xs, dtype=float)
        self.kernel = kernel
        cov = self.kernel.eval(self.xs, self.xs)
        self.build(mu, cov)

class GaussianDiagonalPrior(GaussianPriorBase):
    """
    Base class for Gaussian prior N(mu, cov) using Cholesky
    """

    def __init__(self,
        dim: int, 
        mu: float,
        decay_rate: float = 1.0,  
        variance: float = 1.0,
        seed=12345,
    ):
        self.dim = dim
        self.mu = np.full(self.dim, mu)
        self.decay_rate = decay_rate
        self.D = np.sqrt(variance) * np.arange(1,dim+1,dtype=float) ** (-decay_rate/2.0)
        self.rng = np.random.default_rng(seed)

    def matvec_prior_L(self, xi: np.ndarray) -> np.ndarray:
        return xi * self.D

    def matvec_prior_LT(self, x: np.ndarray) -> np.ndarray:
        return x * self.D

    def matvec_prior_invL(self, x: np.ndarray) -> np.ndarray:
        return x / self.D
    
    def random(self, n: int = 1) -> np.ndarray:
        """
        Draw iid white-noise samples xi ~ N(0, I).

        Returns:
            shape (dim,) if n == 1
            shape (n, dim) if n > 1
        """
        xi = self.rng.standard_normal((n,self.dim))
        if n == 1:
            return xi[0]
        return xi

    