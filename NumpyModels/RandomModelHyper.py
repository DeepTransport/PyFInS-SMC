import numpy as np
from typing import Optional
from BaseModels import SolverBase

class RandomModelHyper(SolverBase):
    """
    Model:
        x(u) = exp(u)
        G(u) = exp(F x(u))

    Prior:
        u_i ~ N(0, D_i^2),  D_i = 2 / i
    """

    def __init__(self, m: int, n: int, nonlinear:bool = True, seed: Optional[int] = None):
        super().__init__()
        rng = np.random.default_rng(seed)
        # MATLAB:
        # [Ql, ~] = qr(randn(m), 0)
        # [Qr, ~] = qr(randn(n, m), 0)
        Ql, _ = np.linalg.qr(rng.standard_normal((m, m)), mode="reduced")
        Qr, _ = np.linalg.qr(rng.standard_normal((n, m)), mode="reduced")
        self.Ql = Ql 
        self.Qr = Qr
        self.m = m
        self.n = n

        self.nonlinear = nonlinear
        self.n_parameter = 4
        self.n_data = m
        self.n_qoi = 0

        self.rng = np.random.default_rng(seed)

    def solve(self, u: np.ndarray):
        """
        Forward solve for the 2exp model.

        Parameters
        ----------
        u : ndarray, shape (n,) or (batch, n)

        Returns
        -------
        ForwardSolution
            d    : shape (m,) or (batch, m)
            dxdu : exp(u), same shape as u
        """
        if u.ndim > 1:
            raise ValueError("input parameter should be a one d array")

        F_decay_rate = u[0]
        mu = u[1]
        log_variance = u[2]
        D_decay_rate = u[3]

        sd = 10.0 ** np.linspace(2.0, 0.0, self.m) / np.exp(F_decay_rate)
        F = self.Ql @ (sd[:, None] * self.Qr.T)
        D = np.exp(0.5*log_variance) * np.arange(1,self.n+1,dtype=float) ** (-D_decay_rate/2.0)

        z = np.random.default_rng.standard_normal(self.n)
        u = D*z + mu

        if self.nonlinear: 
            x = np.exp(u)
            Y = self.F @ x
        else:
            Y = self.F @ u
        return Y, None
    


    