import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
import networkx as nx

from BaseModels import SolverBase

def periodic_lattice(d):
    W = np.zeros((d, d), dtype=float)
    for i in range(d):
        W[i, i] = -2
        W[i, (i - 1) % d] = 1
        W[i, (i + 1) % d] = 1
    W = 0.5 * W
    if d == 2:
        W = np.array([
            [-1.0,  1.0],
            [ 1.0, -1.0],
        ])
    elif d == 1:
        W = np.array([[0.0]])
    return W

def Austria_lattice():
    d = 9
    W = np.zeros((d, d), dtype=float)
    W[0, 1] = 1  # V-T
    W[1, 2] = 1  # T-Sa
    W[1, 3] = 1  # T-K
    W[2, 3] = 1  # Sa-K
    W[2, 5] = 1  # Sa-O
    W[2, 4] = 1  # Sa-St
    W[3, 4] = 1  # K-St
    W[4, 5] = 1  # St-O
    W[4, 6] = 1  # St-N
    W[4, 8] = 1  # St-B
    W[5, 6] = 1  # O-N
    W[6, 7] = 1  # N-W
    W[6, 8] = 1  # N-B
    W = W + W.T
    degree = np.sum(W, axis=1)
    W = W - np.diag(degree)
    W = 0.5 * W
    return W

def plot_labelled_graph(W, labels=None, pos=None, threshold=0.0):
    """
    Plot a labelled undirected graph from a sparse or dense matrix W.

    Assumes edges correspond to positive off-diagonal entries W[i, j] > threshold.
    This is suitable for your diffusion/Laplacian matrix, where diagonal entries are negative
    and off-diagonal entries are positive.
    """
    d = W.shape[0]

    if labels is None:
        labels = {i: str(i) for i in range(d)}
    elif isinstance(labels, list):
        labels = {i: labels[i] for i in range(d)}

    G = nx.Graph()

    for i in range(d):
        G.add_node(i)

    for i in range(d):
        for j in range(i + 1, d):
            if W[i, j] > threshold:
                G.add_edge(i, j)

    if pos is None:
        pos = nx.spring_layout(G, seed=1)

    plt.figure(figsize=(7, 5))
    nx.draw_networkx_edges(G, pos, width=2)
    nx.draw_networkx_nodes(G, pos, node_size=800)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12)

    plt.axis("off")
    plt.tight_layout()
    plt.show()

class SpatialSIR(SolverBase):
    """
        W         : d x d sparse matrix, converted into a full matrix 
        x0        : initial condition of size 3*d
        tobs      : observation times
    """

    def __init__(self, W, x0, tobs):
        super().__init__()
        self.block_mode = True
        self.W = W
        self.d = self.W.shape[0]
        self.tobs = tobs
        # Store as shape (d, 3)
        self.x0 = np.reshape(x0, (3, self.d))
        self.n_parameter = self.d*2
        self.n_data = self.d*(self.tobs.size-1)
        self.n_qoi = 0
        self.sensor2data = np.arange(self.n_data).reshape(self.tobs.size - 1, self.d)

    def rhs(self, t, x, beta, gamma):
        """
        x     : flattened array of size d*3*n
        beta  : shape (d, n)
        gamma : shape (d, n)
        """
        n = beta.shape[0]
        x = np.reshape(x, (n, 3, self.d))
        S = x[:, 0, :]   # shape (n, d)
        I = x[:, 1, :]   # shape (n, d)

        # Movement term
        f = x @ self.W.T
        f[:, 0, :] -= beta * S * I
        f[:, 1, :] += beta * S * I - gamma * I
        f[:, 2, :] += gamma * I

        return np.ravel(f)

    def solve(self, betagamma):
        """
        betagamma : array of shape (n, 2*d)
        Assumes columns are ordered as
            beta_1, ..., beta_d, gamma_1, ..., gamma_d
        returns : observables of shape ((len(tobs)-1)*d, n)
        """
        if betagamma.ndim == 1:
            betagamma = betagamma[None,:]
            n = 1
        else:
            n = betagamma.shape[0]

        beta = betagamma[:, :self.d]
        gamma = betagamma[:, self.d:2*self.d]

        # self.x0 has shape (3, d)
        x0 = np.tile(self.x0[None, :, :], (n, 1, 1))   # shape (n, 3, d)
        x0 = np.ravel(x0)

        sol = solve_ivp(
            fun=lambda t, x: self.rhs(t, x, beta, gamma),
            t_span=(self.tobs[0], self.tobs[-1]),
            y0=x0,
            t_eval=self.tobs,
            method="RK45",
            atol=1e-6,
            rtol=1e-6,
        )
        # sol.y shape: (n*3*d, len(tobs))
        # Remove initial time, then transpose to time-major layout
        x = sol.y[:, 1:].T      # shape (nt-1, n*3*d)
        # Shape: time x n x 3 x d
        x = np.reshape(x, (x.shape[0], n, 3, self.d))
        # Infected observable: (nt-1) x n x d
        Y = x[:, :, 1, :]
        # Convert to n x ((nt-1)*d)
        Y = np.transpose(Y, (1, 0, 2))   # n x time x d

        if n == 1:
            Y = np.ravel(Y)
        else:
            Y = np.reshape(Y, (n, -1))  # n x ((nt-1)*d)

        return Y, None
    