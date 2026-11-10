import abc
from typing import Union
import numpy as np

ArrayLike = [np.ndarray]
try:
    from petsc4py import PETSc
    ArrayLike.append("PETSc.Vec")
except ImportError:
    PETSc = None
try:
    from dolfinx import fem
    ArrayLike.append("fem.Function")
except ImportError:
    fem = None
ArrayLike = Union[tuple(ArrayLike)]

class PriorBase(abc.ABC):
    """
    Abstract prior interface.

    Base class only stores the mean (mu). Subclasses must:
      - define the parameterization / dimensions
      - implement random() and the linear operator matvecs
      - decide how ArrayLike objects are interpreted (numpy, PETSc Vec, dolfinx Function)

    This class does not assume that mu is scalar vs vector; subclass decides how to use it.
    """

    @abc.abstractmethod
    def random(self, n: int = 1) -> ArrayLike:
        raise NotImplementedError

    @abc.abstractmethod
    def transform(self, xi: ArrayLike) -> ArrayLike:
        raise NotImplementedError
    
    @abc.abstractmethod
    def whiten(self, xi: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def linearized_transform_transpose(self, k: ArrayLike) -> ArrayLike:
        return k

    def LIS_linear(self, k: ArrayLike) -> ArrayLike:
        # Jacobian for building LIS, identity map by default
        return k

    def sample(self, n: int = 1) -> ArrayLike:
        xi = self.random(n)
        return self.transform(xi)

class GaussianPriorBase(PriorBase):
    """
    Abstract prior interface.

    Base class only stores the mean (mu). Subclasses must:
      - define the parameterization / dimensions
      - implement random() and the linear operator matvecs
      - decide how ArrayLike objects are interpreted (numpy, PETSc Vec, dolfinx Function)

    Convention (typical):
      xi ~ N(0, I)
      k  = mu + L xi

    This class does not assume that mu is scalar vs vector; subclass decides how to use it.
    """

    # ---------
    # Sampling
    # ---------

    @abc.abstractmethod
    def matvec_prior_L(self, xi: ArrayLike) -> ArrayLike:
        """Return local-owned k = L xi (shape (n_owned,)) or batch (n_batch, n_owned)."""
        raise NotImplementedError

    @abc.abstractmethod
    def matvec_prior_LT(self, k_owned: ArrayLike) -> ArrayLike:
        """Return xi = L^T k. Typically requires an Allreduce if L is distributed."""
        raise NotImplementedError

    @abc.abstractmethod
    def matvec_prior_invL(self, k_owned: ArrayLike) -> ArrayLike:
        """Return xi = L^{-1} k (whitening)."""
        raise NotImplementedError

    # ---------------
    # Convenience APIs
    # ---------------
    def transform(self, xi: ArrayLike) -> ArrayLike:
        """Compute k = L xi + mu."""
        if xi.ndim == 1:
            return self.mu + self.matvec_prior_L(xi) 
        return self.mu + self.matvec_prior_L(xi) 

    def whiten(self, k: ArrayLike) -> ArrayLike:
        """Compute xi = L^{-1} (k - mu)."""
        if k.ndim == 1:
            return self.matvec_prior_invL(k - self.mu)
        return self.matvec_prior_invL(k - self.mu)

    def LIS_linear(self, k: ArrayLike) -> ArrayLike:
        return self.matvec_prior_LT(k)

    def linearized_transform_transpose(self, k: ArrayLike) -> ArrayLike:
        return self.matvec_prior_LT(k)
    
    def apply_cov(self, v: ArrayLike) -> ArrayLike:
        """
        Apply covariance C = L L^T:
          C v = L (L^T v)
        """
        return self.matvec_prior_L(self.matvec_prior_LT(v))
    
class KernelBase(abc.ABC):
    """
    Abstract prior interface.

    Base class only stores the mean (mu). Subclasses must:
      - define the parameterization / dimensions
      - implement random() and the linear operator matvecs
      - decide how ArrayLike objects are interpreted (numpy, PETSc Vec, dolfinx Function)

    This class does not assume that mu is scalar vs vector; subclass decides how to use it.
    """

    @abc.abstractmethod
    def eval(self, xs: ArrayLike, ys: ArrayLike) -> ArrayLike:
        raise NotImplementedError
