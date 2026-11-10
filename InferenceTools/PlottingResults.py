import os
import numpy as np
import matplotlib.pyplot as plt

def plot_values(values, ncols=3, show=True, filename=None, figsize=None, dpi=200):
    """
    Plot a list of one-dimensional values as subplots.

    Parameters
    ----------
    values : list of (value, title)
        Each value is plotted in one subplot.
    ncols : int
        Number of subplot columns.
    filename : str, optional
        Save the figure if provided.
    show : bool
        Display the figure if True.
    figsize : tuple, optional
        Figure size.  If None, choose it from the subplot layout.
    dpi : int
        Resolution used when saving.
    """
    nplots = len(values)
    nrows = (nplots + ncols - 1) // ncols

    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)

    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    ax = ax.ravel()

    for i, (value, title) in enumerate(values):
        ax[i].plot(value)
        ax[i].set_title(title)

    for i in range(nplots, ax.size):
        ax[i].axis("off")

    fig.tight_layout()

    if filename is not None:
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(filename, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    #return fig, ax


def pairwise_scatter(
    X,
    Y,
    log_W,
    xlabel="X",
    ylabel="Y",
    show=True, 
    filename=None,
    max_dim=6,
    figsize=None,
    dpi=200,
):
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]

    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")

    dx = min(X.shape[1], max_dim)
    dy = min(Y.shape[1], max_dim)

    if figsize is None:
        figsize = (4 * dx, 4 * dy)

    # fig, ax = plt.subplots(dy, dx, figsize=figsize, squeeze=False)

    # for j in range(dy):
    #     for i in range(dx):
    #         ax[j, i].scatter(X[:, i], Y[:, j], c=log_W, cmap="viridis", s=20)
    #         ax[j, i].set_xlabel(xlabel)
    #         ax[j, i].set_ylabel(ylabel)

    # fig.tight_layout()

    fig, ax = plt.subplots(dy, dx, figsize=figsize, squeeze=False, constrained_layout=True)
    for j in range(dy):
        for i in range(dx):
            sc = ax[j, i].scatter(X[:, i], Y[:, j], c=log_W, cmap="viridis", s=20)
            ax[j, i].set_xlabel(xlabel)
            ax[j, i].set_ylabel(ylabel)
    fig.colorbar(sc, ax=ax, label=r"$\log W$", pad=0.02)
    plt.show()

    if filename is not None:
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(filename, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    #return fig, ax
