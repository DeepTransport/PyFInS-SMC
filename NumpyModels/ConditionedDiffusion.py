from typing import Union

import numpy as np
from scipy.io import loadmat
from pathlib import Path

# from Likelihoods import GaussianLikelihood
from BaseModels import GaussianPriorBase
from BaseModels import SolverAdjointCache
from BaseModels import GaussianLikelihood

class BrownianPrior(GaussianPriorBase):

    def __init__(self, dim: int, T: float = 1.0):
        self.dim = dim
        self.kl_dim = self.dim
        self.T = T
        self.dt = T / self.dim
        self.sqrt_dt = np.sqrt(self.dt)
        self.mu = np.zeros(self.dim)

    def random(self, n: int = 1) -> np.ndarray:
        if n == 1:
            return np.random.standard_normal(self.dim)
        else:
            return np.random.standard_normal( (n, self.dim) )

    def matvec_prior_L(self, xi: np.ndarray) -> np.ndarray:
        if xi.ndim == 1:
            return self.sqrt_dt * np.cumsum(xi)
        else:
            return self.sqrt_dt * np.cumsum(xi, axis=1)

    def matvec_prior_LT(self, xi: np.ndarray) -> np.ndarray:
        if xi.ndim == 1:
            return self.sqrt_dt * np.cumsum(xi[::-1])[::-1]
        else:
            return self.sqrt_dt * np.cumsum(xi[:, ::-1], axis=1)[:, ::-1]

    def matvec_prior_invL(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            w = np.empty_like(x)
            w[0] = x[0]
            w[1:] = np.diff(x)
        else:
            w = np.empty_like(x)
            w[:, 0] = x[:, 0]
            w[:, 1:] = np.diff(x, axis=1)
        return w/self.sqrt_dt

class ConditionedDiffusion(SolverAdjointCache):
    """Euler--Maruyama solver with tangent and adjoint Jacobian actions.

    Parameters
    ----------
    n_steps:
        Number of Euler--Maruyama increments and physical parameters.
    dt:
        Time-step size.
    drift_strength:
        The coefficient ``d`` in the nonlinear drift.
    observation_lag:
        Observe ``u[k], u[2k], ..., u[N]``. ``n_steps`` must be divisible by
        this value.
    initial_state:
        Deterministic value u[0]. The legacy example uses zero.

    Notes
    -----
    ``solve(w)`` returns the observable consumed by a likelihood. For a batch
    with shape ``(n_batch, n_steps)``, it returns shape
    ``(n_batch, n_observations)``.

    The derivative methods act at the most recent single-vector call to
    ``solve_and_cache`` (or ``solve``).
    """

    def __init__(
        self,
        n_steps: int = 1000,
        T: float = 10.0,
        drift_strength: float = 10.0,
        observation_lag: int = 20,
        initial_state: float = 0.0,
    ) -> None:

        super().__init__()
        self.n_steps = int(n_steps)
        self.T = float(T)
        self.dt = T/self.n_steps
        self.drift_strength = float(drift_strength)
        self.observation_lag = int(observation_lag)
        self.initial_state = float(initial_state)

        if self.n_steps % self.observation_lag != 0:
            raise ValueError("n_steps must be divisible by observation_lag.")

        self.sqrt_dt = np.sqrt(self.dt)
        self.n_parameter = self.n_steps
        self.observation_indices = np.arange(self.observation_lag, self.n_steps + 1, self.observation_lag, dtype=int)
        self.n_data = self.observation_indices.size
        self.n_qoi = 0

        self.state = None
        self.tangent_factors = None
        self.jacobian = None
    
    def drift(self, u: np.ndarray) -> np.ndarray:
        u2 = u * u
        return self.drift_strength * u * (1.0 - u2) / (1.0 + u2)

    def drift_derivative(self, u: np.ndarray) -> np.ndarray:
        u2 = u * u
        return self.drift_strength * ( (1.0 - 3.0 * u2) / (1.0 + u2) - 2.0 * u2 * (1.0 - u2) / (1.0 + u2) ** 2 )

    def _integrate(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            increments = np.empty_like(x)
            increments[0] = x[0]
            increments[1:] = np.diff(x)
            u = np.empty(self.n_steps + 1, dtype=float)
            u[0] = self.initial_state
            for n in range(self.n_steps):
                u[n + 1] = u[n] + self.dt * self.drift(u[n]) + increments[n]
            return u
        increments = np.empty_like(x)
        increments[:, 0] = x[:, 0]
        increments[:, 1:] = np.diff(x, axis=1)
        u = np.empty((x.shape[0], self.n_steps + 1), dtype=float)
        u[:, 0] = self.initial_state
        for n in range(self.n_steps):
            u[:, n + 1] = u[:, n] + self.dt * self.drift(u[:, n]) + increments[:, n]
        return u

    def apply_observation(self, state: np.ndarray) -> np.ndarray:
        if state.ndim == 1:
            return state[self.observation_indices]
        else:
            return state[:,self.observation_indices]
        
    def solve(self, x: np.ndarray) -> np.ndarray:
        state = self._integrate(x)
        return self.apply_observation(state), None

    def solve_and_cache(self, x: np.ndarray) -> np.ndarray:
        state = self._integrate(x)
        self.tangent_factors = 1.0 + self.dt * self.drift_derivative(state[:-1])
        return self.apply_observation(state), None

    def linearized_observation(self, delta_x: np.ndarray) -> np.ndarray:
        """Matrix-free J @ delta_x, with respect to the Brownian path."""
        a = self.tangent_factors
        if delta_x.ndim == 1:
            du = np.empty(self.n_steps + 1, dtype=float)
            du[0] = 0.0
            # First Brownian increment: delta_x[0] - 0
            du[1] = delta_x[0]
            for n in range(1, self.n_steps):
                delta_increment = delta_x[n] - delta_x[n - 1]
                du[n + 1] = a[n] * du[n] + delta_increment
            return du[self.observation_indices]
        du = np.empty( (delta_x.shape[0], self.n_steps + 1), dtype=float )
        du[:, 0] = 0.0
        du[:, 1] = delta_x[:, 0]
        for n in range(1, self.n_steps):
            delta_increment = delta_x[:, n] - delta_x[:, n - 1]
            du[:, n + 1] = a[n] * du[:, n] + delta_increment
        return du[:, self.observation_indices].squeeze()

    def linearized_observation_transpose(self, delta_y: np.ndarray) -> np.ndarray:
        """Matrix-free J.T @ delta_y, with respect to the Brownian path."""
        a = self.tangent_factors
        if delta_y.ndim == 1:
            adjoint = np.zeros(self.n_steps + 1, dtype=float)
            adjoint[self.observation_indices] = delta_y
            # Gradient with respect to Brownian increments.
            grad_increment = np.empty(self.n_steps, dtype=float)
            for n in range(self.n_steps - 1, -1, -1):
                grad_increment[n] = adjoint[n + 1]
                adjoint[n] += a[n] * adjoint[n + 1]
            # Apply the transpose of the first-difference operator.
            grad_path = np.empty_like(grad_increment)
            grad_path[:-1] = grad_increment[:-1] - grad_increment[1:]
            grad_path[-1] = grad_increment[-1]
            return grad_path
        adjoint = np.zeros((delta_y.shape[0], self.n_steps + 1), dtype=float)
        adjoint[:, self.observation_indices] = delta_y
        grad_increment = np.empty((delta_y.shape[0], self.n_steps), dtype=float)
        for n in range(self.n_steps - 1, -1, -1):
            grad_increment[:, n] = adjoint[:, n + 1]
            adjoint[:, n] += a[n] * adjoint[:, n + 1]
        grad_path = np.empty_like(grad_increment)
        grad_path[:, :-1] = (grad_increment[:, :-1] - grad_increment[:, 1:])
        grad_path[:, -1] = grad_increment[:, -1]
        return grad_path.squeeze()

    def linearized_qoi_transpose(self):
        return None

def load_conditioned_diffusion_problem(setup_file: Union[str, Path]):
    """Load the legacy ``CDsetup.mat`` into the common Python interfaces."""
    raw = loadmat(setup_file, squeeze_me=True, struct_as_record=False)
    model = raw["model"]
    obs = raw["obs"]

    N = int(model.N)
    T = float(model.dt) * N
    true_w = np.asarray(model.true_w, dtype=float).reshape(-1)
    data = np.asarray(obs.data, dtype=float).reshape(-1)
    sigma=float(obs.std)
    
    solver = ConditionedDiffusion(
        n_steps=N,
        T=T,
        drift_strength=float(model.d),
        observation_lag=int(model.k),
        initial_state=0.0,
    )
    prior = BrownianPrior(dim=N, T=T)
    like = GaussianLikelihood(data, sigma)

    return prior, solver, like, true_w

