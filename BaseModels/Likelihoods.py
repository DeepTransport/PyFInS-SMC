
import abc
import numpy as np
from scipy.special import gammaln

def _as_2d(a):
    """Return a as 2D array (n_obs, n_batch)."""
    a = np.asarray(a)
    if a.ndim == 1:
        return a[None,:]
    if a.ndim == 2:
        return a
    raise ValueError(f"Expected 1D or 2D array, got shape {a.shape}.")

def _init_mask(mask, n_obs):
    if mask is None:
        return np.ones(n_obs, dtype=bool)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.size != n_obs:
        raise ValueError(f"mask length {mask.size} != n_obs {n_obs}.")
    return mask

class LikelihoodBase(abc.ABC):
    """
    Abstract likelihood interface.
    """

    def __init__(self, obs, mask=None, rng=None):
        self.data = _as_2d(obs)
        #self.n_obs = self.data.size
        self.n_obs = self.data.shape[1]
        self.mask = _init_mask(mask, self.n_obs)
        self.rng = np.random.default_rng() if rng is None else rng

    def set_rng(self, rng):
        self.rng = rng

    @abc.abstractmethod
    def eval_potential(self, observable):
        """
        return the negative log likelihood given the observables (computed by the forward model)
        """
        raise NotImplementedError
    
    def update_mask(self, ind):
        if np.any(np.asarray(ind) >= self.n_obs):
            raise ValueError(f"mask index {ind} >= n_obs {self.n_obs}.")
        if np.any(np.asarray(ind) < 0):
            raise ValueError(f"mask index {ind} < 0.")
        self.mask[ind] = True

    def set_mask(self, ind):
        if np.any(np.asarray(ind) >= self.n_obs):
            raise ValueError(f"mask index {ind} >= n_obs {self.n_obs}.")
        if np.any(np.asarray(ind) < 0):
            raise ValueError(f"mask index {ind} < 0.")
        self.mask = np.zeros(self.n_obs, dtype=bool)
        self.mask[ind] = True

    @abc.abstractmethod
    def eval_potential_grad(self, observable):
        raise NotImplementedError
    
    @abc.abstractmethod
    def eval_potential_Fisher(self, observable):
        raise NotImplementedError
    
    @abc.abstractmethod
    def simulate_data_masked(self, observable): 
        raise NotImplementedError

    @abc.abstractmethod
    def simulate_data_all(self, observable):
        raise NotImplementedError

def _broadcast_sigma(sigma, n_obs):
    sigma = np.asarray(sigma)
    if sigma.ndim == 0:
        return float(sigma)
    if sigma.ndim == 1:
        if sigma.size != n_obs:
            raise ValueError(f"sigma length {sigma.size} != n_obs {n_obs}.")
        return sigma.astype(float, copy=False)
    raise ValueError(f"sigma must be scalar or 1D vector, got shape {sigma.shape}.")

def _check_observable(observable, n_obs):
        y = _as_2d(observable)
        if y.shape[1] != n_obs:
            raise ValueError(f"observable first dim {y.shape[1]} must match n_obs {n_obs}.")
        return y

class GaussianLikelihood(LikelihoodBase):
    """
    Gaussian negative log-likelihood with masking:
      Phi(y) = 0.5 * sum_{active i} ((y_i - d_i)/sigma_i)^2

    data d: 1D shape (n_obs,)
    observable y: 1D (n_obs,) or 2D (n_obs, n_batch)
    mask: 1D bool shape (n_obs,), True=active.
    """

    def __init__(self, obs, sigma, mask=None):
        super().__init__(obs, mask)
        self.sigma = _broadcast_sigma(sigma, self.n_obs)

    def eval_potential(self, observable):
        y = _check_observable(observable, self.n_obs)
        diff = y - self.data
        if np.isscalar(self.sigma):
            diff_scaled = diff / self.sigma
        else:
            diff_scaled = diff / self.sigma[None,:]
        dm = diff_scaled[:,self.mask]
        return 0.5 * np.sum(dm * dm, axis=1)

    def eval_potential_grad(self, observable):
        y = _check_observable(observable, self.n_obs)
        diff = y - self.data
        if np.isscalar(self.sigma):
            grad = diff / ( self.sigma * self.sigma )
        else:
            grad = diff / ( self.sigma[None,:] * self.sigma[None,:] )
        grad_out = np.zeros_like(grad)
        grad_out[:,self.mask] = grad[:,self.mask]
        return grad_out[0,:] if observable.ndim == 1 else grad_out

    def eval_potential_Fisher(self, observable):
        diag = np.zeros(self.n_obs, dtype=float)
        if np.isscalar(self.sigma):
            diag[self.mask] = 1.0 / ( self.sigma * self.sigma )
        else:
            diag[self.mask] = 1.0 / ( self.sigma[self.mask] * self.sigma[self.mask] )
        return diag

    def simulate_data_masked(self, observable):
        y = _check_observable(observable, self.n_obs)
        r = self.rng.normal(0, self.sigma, size=y.shape)
        ynew = y + r
        if np.isscalar(self.sigma):
            diff_scaled = r / self.sigma
        else:
            diff_scaled = r / self.sigma[None,:]
        dm = diff_scaled[:,self.mask]
        potential = 0.5 * np.sum(dm * dm, axis=1)
        return ynew[:,self.mask], potential 
    
    def simulate_data_all(self, observable):
        y = _check_observable(observable, self.n_obs)
        r = self.rng.normal(0, self.sigma, size=y.shape)
        return (y + r)
    
class PoissonLikelihood(LikelihoodBase):
    """
    Poisson negative log-likelihood (up to constants) with masking:
      Phi(y) = - sum_{active i} d_i log(y_i) + sum_{active i} y_i

    data d: 1D shape (n_obs,)
    observable y: 1D (n_obs,) or 2D (n_obs, n_batch)
    mask: 1D bool shape (n_obs,), True=active.
    """

    def __init__(self, obs, mask=None):
        super().__init__(obs, mask)

    def eval_potential(self, observable):
        y = _check_observable(observable, self.n_obs)
        y = np.maximum(y, np.finfo(float).eps)
        return np.sum(y[:,self.mask] - self.data[:,self.mask] * np.log(y[:,self.mask]), axis=1)

    def eval_potential_grad(self, observable):
        y = _check_observable(observable, self.n_obs)
        y = np.maximum(y, np.finfo(float).eps)
        grad = 1.0 - self.data / y
        grad_out = np.zeros_like(grad)
        grad_out[:,self.mask] = grad[:,self.mask]
        return grad_out[0,:] if observable.ndim == 1 else grad_out

    def eval_potential_Fisher(self, observable):
        y = _check_observable(observable, self.n_obs)
        y = np.maximum(y, np.finfo(float).eps)
        if y.shape[0] != 1:
            raise ValueError("Fisher information currently supports one observable.")
        diag = np.zeros(self.n_obs)
        diag[self.mask] = 1.0/y[0,self.mask]
        return diag
    
    def simulate_data_masked(self, observable):
        y = _check_observable(observable, self.n_obs)
        y = np.maximum(y, np.finfo(float).eps)
        ynew = self.rng.poisson(lam=y)
        potential = np.sum(y[:,self.mask] - ynew[:,self.mask] * np.log(y[:,self.mask]) + gammaln(ynew[:,self.mask]+1), axis=1)
        return ynew[:,self.mask], potential 

    def simulate_data_all(self, observable):
        y = _check_observable(observable, self.n_obs)
        y = np.maximum(y, np.finfo(float).eps)
        return self.rng.poisson(lam=y)

