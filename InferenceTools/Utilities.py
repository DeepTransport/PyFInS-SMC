
import numpy as np
from typing import Literal

class InferenceRNG:
    def __init__(self, seed=12345):
        ss = np.random.SeedSequence(seed)
        streams = ss.spawn(4)
        self.resample = np.random.default_rng(streams[0])
        self.kernel = np.random.default_rng(streams[1])
        self.likelihood = np.random.default_rng(streams[2])
        self.design = np.random.default_rng(streams[3])

LikelihoodInformedSubspaceType = Literal["Bases", "Canonical"]
TruncationCriteria = Literal["trace", "det", "value"]

def bound_based_truncation(value, epsilon, criterion:TruncationCriteria="trace"):
    """
    Return the number of leading modes to retain.

    The values are assumed to be sorted in descending order.
    """
    if criterion == "trace":
        tail = np.cumsum(value[::-1])[::-1]
        threshold = 4.0 * epsilon * epsilon
    elif criterion == "det":
        tail = np.cumsum((0.5 * np.log1p(value))[::-1])[::-1]
        threshold = epsilon
    elif criterion == "value":
        tail = value
        threshold = epsilon
    else:
        raise ValueError(f"Unknown truncation criterion '{criterion}'.")

    idx = np.flatnonzero(tail < threshold)

    if idx.size == 0:
        return value.size

    return int(idx[0])


def ess_ratio(delta_beta, like_potentials):
    log_weight = -delta_beta * like_potentials
    log_weight -= np.max(log_weight)

    weights = np.exp(log_weight)
    ess = np.sum(weights) ** 2 / np.sum(weights**2)

    return ess / len(like_potentials)

def adapt_temperature(
    beta_previous,
    like_potentials,
    ess_tol=0.5,
    min_beta=1.0e-6,
    beta_frac=1.05,
):
    beta = max(beta_previous, min_beta)
    ess2n = ess_ratio(beta - beta_previous, like_potentials)

    while beta < 1.0 and ess2n > ess_tol:
        beta *= beta_frac
        ess2n = ess_ratio(beta - beta_previous, like_potentials)

    beta = min(beta, 1.0)
    ess2n = ess_ratio(beta - beta_previous, like_potentials)

    return beta, ess2n

def convert_log_weights(log_weights):
    tmp = np.exp(log_weights-np.max(log_weights))
    return tmp/np.sum(tmp)
