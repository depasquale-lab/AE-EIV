"""Synthetic ring data with condition-dependent behavior noise."""
from __future__ import annotations

import numpy as np


def wrap(a):
    return np.angle(np.exp(1j * a))


def simulate(seed=0, N=30, T_per=1000, kappas=(8.0, 3.0, 0.5),
             tune_kappa=3.0, step_sd=0.3, gain=(3.0, 8.0)):
    """Latent x follows a periodic random walk. Neurons are von Mises tuned to x.
    b_t | x_t ~ VM(x_t, kappa_c) with one kappa per condition block.

    Returns dict with y (T, N), b, x, cond, and the true rate function.
    """
    rng = np.random.default_rng(seed)
    C = len(kappas)
    T = T_per * C
    x = wrap(np.cumsum(step_sd * rng.standard_normal(T)) + rng.uniform(-np.pi, np.pi))
    cond = np.repeat(np.arange(C), T_per)
    pref = wrap(np.linspace(-np.pi, np.pi, N, endpoint=False) + 0.1 * rng.standard_normal(N))
    gain = rng.uniform(gain[0], gain[1], N)
    base = 0.3

    def rate(xx):
        return base + gain[None, :] * np.exp(tune_kappa * (np.cos(xx[:, None] - pref[None, :]) - 1.0))

    y = rng.poisson(rate(x))
    kap_t = np.asarray(kappas)[cond]
    b = np.array([rng.vonmises(xi, ki) for xi, ki in zip(x, kap_t)])
    return {"y": y.astype(np.float32), "b": b.astype(np.float32), "x": x,
            "cond": cond, "kappas": np.asarray(kappas), "rate": rate}
