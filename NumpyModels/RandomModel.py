import numpy as np
from typing import Optional
from BaseModels import SolverAdjointCache

class RandomModel(SolverAdjointCache):
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
        sd = 10.0 ** np.linspace(2.0, 0.0, m) / 500.0
        # MATLAB:
        # model.F = Ql * (sd(:) .* Qr')
        #
        # Qr.T has shape (m, n)
        # sd[:, None] * Qr.T scales each row
        self.F = Ql @ (sd[:, None] * Qr.T)

        self.nonlinear = nonlinear
        self.n_parameter = n
        self.n_data = m
        self.n_qoi = 0

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
        if self.nonlinear: 
            x = np.exp(u)
        else:
            x = u
        if u.ndim == 1:
            #Y = np.exp(self.F @ x)
            Y = self.F @ x
        else:
            # Batch case: u is (batch, n), so x @ F.T gives (batch, m)
            #Y = np.exp(x @ self.F.T)
            Y = x @ self.F.T
        return Y, None
    
    def solve_and_cache(self, u: np.ndarray):
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
        if self.nonlinear: 
            x = np.exp(u)
            self.dxdu = x
        else:
            x = u
            self.dxdu = np.ones_like(u)

        if u.ndim == 1:
            #Y = np.exp(self.F @ x)
            Y = self.F @ x
        else:
            # Batch case: u is (batch, n), so x @ F.T gives (batch, m)
            #Y = np.exp(x @ self.F.T)
            Y = x @ self.F.T
        self.Y = Y

        return Y, None

    def linearized_observation(self, delta_u):
        # Ju = sol.d(:).*(model.F*(sol.dxdu.*du));
        if delta_u.ndim == 1:
            #Ju = self.Y * ( self.F @ (self.dxdu * delta_u) )
            Ju = self.F @ (self.dxdu * delta_u)
        else:
            #Ju = ( (delta_u * self.dxdu) @ self.F.T ) * self.Y
            Ju = (delta_u * self.dxdu) @ self.F.T
        return Ju.squeeze()
        
    def linearized_observation_transpose(self, delta_y):
        # Jty = model.F'*(sol.d.*dy);
        # Jty = sol.dxdu.*Jty;
        if delta_y.ndim == 1:
            #Jty = ( self.F.T @ (self.Y * delta_y) ) * self.dxdu
            Jty = ( self.F.T @ delta_y ) * self.dxdu
        else:
            #Jty = ( (delta_y * self.Y) @ self.F ) * self.dxdu
            Jty = ( delta_y @ self.F ) * self.dxdu
        return Jty.squeeze()

    def full_Jacobian(self):
        return self.linearized_observation_transpose(np.eye(self.m)).T

    ## functions below are only for consistency
    def linearized_qoi_transpose(self):
        return None

    