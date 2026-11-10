import numpy as np
#from scipy.linalg import solve_triangular
from typing import Literal

ProposalType = Literal["MALA", "RW", "OW"]

# proposal: Proposal = "MALA"
# sigma: float = 0.1
# dt: float = 0.1
# nstep: int = 1000
# nbatch: int = 100
# sbatch: int = 1

class OneStepKernel:

    type: str
    C: np.ndarray
    L: np.ndarray
    scale: float

    A: np.ndarray
    B: np.ndarray

    # For adaptive MCMC version
    sum: np.ndarray
    cross: np.ndarray
    num: int 

    potential_curr = None
    z_curr = None
    
    def __init__(self, 
        type: ProposalType = "OW",
        sigma: float = 0.1,
        adapt: bool = True,
        nbatch: int = 100,
        rng = None,
    ):
        self.type = type
        self.need_grad = type == "MALA"
        self.sigma = sigma
        self.adapt = adapt
        self.nbatch = nbatch
        if self.type == "RW":
            self.mh_rate = 0.234
        elif self.type == "OW":
            self.mh_rate = 0.5
        elif self.type == "MALA":
            self.mh_rate = 0.57
        self.sum = None
        self.cross = None
        self.num = 0
        self.rng = np.random.default_rng() if rng is None else rng

    def set_rng(self, rng):
        self.rng = rng

    def build_from_samples(self, samples, sigma):
        """
        Build proposal kernel from current LIS samples.

        samples : shape (nsamples, r)
        """
        self.accumulate(samples)
        self.rebuild(sigma)
        self.potential_curr = None
        self.z_curr = None

    def accumulate(self, samples):
        if samples.ndim == 1:
            samples = samples[None,:] 
        if self.num == 0:
            self.sum = np.zeros(samples.shape[1])
            self.cross = np.zeros((samples.shape[1], samples.shape[1]))
        elif self.sum.shape[0] != samples.shape[1]:
            raise ValueError("Sample dimension does not match accumulated samples.")
        self.sum = self.sum + np.sum(samples, axis=0)
        self.cross = self.cross + samples.T @ samples
        self.num = self.num + samples.shape[0]

    def rebuild(self, sigma):
        self.sigma = sigma
        r = len(self.sum)
        if self.num == 1:
            self.C = np.eye(r)
        else:
            M = self.sum / self.num
            self.C = self.cross / self.num - np.outer(M, M) + 1e-5 * np.eye(r)

        if self.type in ["MALA", "RW"]:
            self.scale = np.exp(sigma) / np.sqrt(np.max(np.diag(self.C)))
            self.L = np.linalg.cholesky(self.C, upper=False)
        elif self.type == "OW":
            eigvals, V = np.linalg.eigh(self.C)
            eigvals = np.maximum(eigvals, 0.0)
            dt = np.exp(sigma)
            tmp = 1.0 / (0.5 * dt * eigvals + 1.0)
            DB = np.sqrt(2.0 * dt) * np.sqrt(eigvals) * tmp
            DA = (1.0 - 0.5 * dt * eigvals) * tmp
            self.B = (V * DB[None, :]) @ V.T
            self.A = (V * DA[None, :]) @ V.T
        else:
            raise ValueError(f"Unknown proposal type: {self.type}")
        
    def propose(self, x_curr, like_potential_curr, like_grad_curr=None):
        self.potential_curr = like_potential_curr
        if self.type == "MALA":
            #   x' = x - 0.5 * scale^2 * grad V(x) C + scale * r L.T,
            # where C = L L.T and r ~ N(0, I).
            # With row-vector convention, define
            #   h = 0.5 * scale * grad V(x) L.
            # Then
            #   x' = x + scale * (r - h) L.T.
            # Store only
            #   z = r - h,
            # so that
            #   x' = x + scale * z L.T.
            # For the reverse move,
            #   r' - h' =  - z
            # because z = r_curr - h_curr, hence
            #   r' = h' - z.
            h_curr = 0.5 * self.scale * ((like_grad_curr+x_curr) @ self.L)
            r_curr = self.rng.standard_normal(size=x_curr.shape)
            z_curr = r_curr - h_curr
            x_next = x_curr + self.scale * (z_curr @ self.L.T)
            self.z_curr = z_curr
            log_c2n = -0.5 * np.sum(r_curr**2, axis=1)
            #self.potential_curr = self.potential_curr - log_c2n
            #
            #drift_curr = -(grad_curr @ self.C) * (self.scale**2 / 2.0)
            #r_curr = self.rng.standard_normal(size=x_curr.shape)
            #x_next = x_curr + drift_curr + self.scale * (r_curr @ self.L.T)
            #self.drift_curr = drift_curr
            #self.r_curr = r_curr
            self.potential_curr = self.potential_curr + 0.5 * np.sum(x_curr**2, axis=1) - log_c2n
        elif self.type == "RW":
            r_curr = self.rng.standard_normal(size=x_curr.shape)
            x_next = x_curr + self.scale * (r_curr @ self.L.T)
            self.potential_curr += 0.5 * np.sum(x_curr**2, axis=1)
        elif self.type == "OW":
            r_curr = self.rng.standard_normal(size=x_curr.shape)
            x_next = x_curr @ self.A.T + r_curr @ self.B.T
        else:
            raise ValueError(f"Unknown proposal type: {self.type}")
        return x_next

    def log_MH_ratio(self, like_potential_next, x_next=None, like_grad_next=None):
        potential_next = like_potential_next.copy()
        if self.type == "MALA":
            #drift_next = -(grad_next @ self.C.T) * (self.scale**2 / 2.0)
            #tmp = solve_triangular(self.L, (self.drift_curr + drift_next).T, lower=True,).T
            #r_next = -(tmp / self.scale + self.r_curr)
            #log_alpha = (self.potential_curr - potential_next) + (log_n2c - log_c2n)
            h_next = 0.5 * self.scale * ((like_grad_next+x_next) @ self.L)
            r_next = h_next - self.z_curr
            log_n2c = -0.5 * np.sum(r_next**2, axis=1)
            potential_next = potential_next + 0.5 * np.sum(x_next**2, axis=1) - log_n2c
            #log_alpha = self.potential_curr - potential_next
        elif self.type == "RW":
            potential_next += 0.5 * np.sum(x_next**2, axis=1)
            #log_alpha = self.potential_curr - potential_next
        #elif self.type == "OW":
            #log_alpha = self.potential_curr - potential_next
        return self.potential_curr - potential_next

    