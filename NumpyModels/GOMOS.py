import os
import numpy as np
import h5py
from scipy.io import loadmat
from scipy.linalg import block_diag

from BaseModels import SolverAdjointCache, GaussianLikelihood
from NumpyModels import GaussianPrior

import os
import numpy as np
import h5py
from scipy.io import loadmat

from BaseModels import SolverAdjointCache, GaussianLikelihood
from NumpyModels import GaussianPrior


class GOMOS(SolverAdjointCache):
    """
    GOMOS transmission model:

        x_phys = exp(x)
        y = exp(-A X C^T)

    where X has shape (nalts, ngas), using standard NumPy C-order:

        x = [
            gas0_alt0, gas1_alt0, ...,
            gas0_alt1, gas1_alt1, ...,
            ...
        ]
    """

    def __init__(self, A, C):
        super().__init__()

        self.A = np.asarray(A, dtype=float)
        self.C = np.asarray(C, dtype=float)

        self.nalts = self.A.shape[0]
        self.ngas = self.C.shape[1]
        self.nwave = self.C.shape[0]

        self.n_parameter = self.nalts * self.ngas
        self.n_data = self.nalts * self.nwave
        self.n_qoi = 0

        self._state = None
        self._x = None
        self._x_phys = None

    def _forward(self, x_phys):
        X = x_phys.reshape(self.nalts, self.ngas)
        # Shape: (nalts, nwave)
        Y = np.exp(-self.A @ X @ self.C.T)
        return Y.reshape(-1)

    def solve(self, x):
        x = np.asarray(x)
        x_phys = np.exp(x)
        if x.ndim == 1:
            return self._forward(x_phys), None
        return np.stack([self._forward(z) for z in x_phys], axis=0), None

    def solve_and_cache(self, x):
        self._x = np.asarray(x)
        self._x_phys = np.exp(self._x)
        self._state = self._forward(self._x_phys)
        return self._state, None

    def _linearized_observation(self, delta_x):
        # Chain rule through x_phys = exp(x)
        delta_x_phys = self._x_phys * delta_x
        D = delta_x_phys.reshape(self.nalts, self.ngas)
        return ((-self.A @ D @ self.C.T).reshape(-1) * self._state)

    def linearized_observation(self, delta_x):
        delta_x = np.asarray(delta_x)
        if delta_x.ndim == 1:
            return self._linearized_observation(delta_x)
        return np.stack([self._linearized_observation(v) for v in delta_x], axis=0)

    def _linearized_observation_transpose(self, delta_y):
        Y = (self._state * delta_y).reshape(self.nalts, self.nwave)
        grad_phys = (-self.A.T @ Y @ self.C).reshape(-1)
        # Chain rule through x_phys = exp(x)
        return grad_phys * self._x_phys

    def linearized_observation_transpose(self, delta_y):
        delta_y = np.asarray(delta_y)
        if delta_y.ndim == 1:
            return self._linearized_observation_transpose(delta_y)
        return np.stack([self._linearized_observation_transpose(v) for v in delta_y], axis=0)

    def linearized_qoi_transpose(self):
        raise NotImplementedError("Specify a scalar QoI before requesting its gradient.")


def _load_v73(path):
    out = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            a = np.array(f[k])
            out[k] = a.T if a.ndim == 2 else a
    return out

def load_gomos_problem(data_dir):
    d = _load_v73(os.path.join(data_dir, "transmission50.mat"))
    o = loadmat(os.path.join(data_dir, "Tobs.mat"),squeeze_me=True)

    A = np.asarray(d["A"])
    C = np.asarray(d["C"])
    X = np.asarray(d["X"])

    alts = np.asarray(d["alts"]).reshape(-1)
    weights = np.asarray(d["weights"])
    Tobs = np.asarray(o["Tobs"])

    # Rescale gases
    scale = C.mean(axis=0)
    Cscaled = C / scale[None, :]
    Xs = X * scale[None, :]

    # Gaussian prior for log(Xs)
    sig = np.var(np.log(Xs), axis=0, ddof=1)
    ell2 = np.full(4, 100.0)
    tau = 1e-10

    dist = (alts[:, None] - alts[None, :]) ** 2

    cov = np.zeros((4 * alts.size, 4 * alts.size))

    for j in range(4):
        block = sig[j] * np.exp(-dist / (2 * ell2[j])) + tau * np.eye(alts.size)
        cov[j::4, j::4] = block

    # gas-fast ordering at each altitude
    mu = np.tile(np.mean(np.log(Xs), axis=0), alts.size)

    prior = GaussianPrior(mu, cov)
    solver = GOMOS(A, Cscaled)

    like = GaussianLikelihood(
        Tobs.reshape(-1, order="F"),
        weights.reshape(-1, order="F"),
    )

    info = {
        "alts": alts,
        "wavelength": np.asarray(d["wavel"]).reshape(-1),
        "true_x": Xs.T.reshape(-1, order="F"),
    }

    return solver, like, prior, info



def _load_v73(path):
    out = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            a = np.array(f[k])
            out[k] = a.T if a.ndim == 2 else a
    return out


def load_gomos_problem(data_dir):
    d = _load_v73(os.path.join(data_dir, "transmission50.mat"))
    o = loadmat(os.path.join(data_dir, "Tobs.mat"), squeeze_me=True)

    A = np.asarray(d["A"])
    C = np.asarray(d["C"])
    X = np.asarray(d["X"])

    alts = np.asarray(d["alts"]).reshape(-1)
    weights = np.asarray(d["weights"])
    Tobs = np.asarray(o["Tobs"])

    nalts = alts.size
    ngas = C.shape[1]
    nwave = C.shape[0]

    # ---------------------------------------------------------
    # Rescale gases
    # ---------------------------------------------------------

    scale = C.mean(axis=0)

    Cscaled = C / scale[None, :]
    Xs = X * scale[None, :]

    # Xs should have shape (nalts, ngas)
    assert Xs.shape == (nalts, ngas)

    # ---------------------------------------------------------
    # Gaussian prior for log(Xs)
    # ---------------------------------------------------------

    log_Xs = np.log(Xs)
    sig = np.var(log_Xs, axis=0, ddof=1)
    ell2 = np.full(ngas, 100.0)
    tau = 1e-10
    dist = (alts[:, None] - alts[None, :]) ** 2
    cov = np.zeros((ngas * nalts, ngas * nalts))

    for j in range(ngas):
        block = sig[j] * np.exp(-dist / (2 * ell2[j])) + tau * np.eye(nalts)
        cov[j::ngas, j::ngas] = block

    # Mean gas concentration in log-space, repeated at each altitude.
    # Ordering:
    # [gas0_alt0, gas1_alt0, ..., gas0_alt1, gas1_alt1, ...]
    mu = np.tile(np.mean(log_Xs, axis=0), nalts)

    prior = GaussianPrior(mu, cov)
    solver = GOMOS(A, Cscaled)

    # ---------------------------------------------------------
    # Observation ordering
    # ---------------------------------------------------------
    #
    # Solver output has shape (nalts, nwave) before flattening:
    #
    # alt0: wave0, wave1, ...
    # alt1: wave0, wave1, ...
    #
    # Tobs and weights come from MATLAB as (nwave, nalts),
    # so transpose them explicitly and then use normal C-order.

    assert Tobs.shape == (nwave, nalts)
    assert weights.shape == (nwave, nalts)

    Tobs_vec = Tobs.T.reshape(-1)
    weights_vec = weights.T.reshape(-1)

    like = GaussianLikelihood(Tobs_vec, weights_vec)

    print("alts:", alts)
    print("wavelength:", np.asarray(d["wavel"]).reshape(-1))
    print("true_x:", Xs.reshape(-1))

    return solver, like, prior

