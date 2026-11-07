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


def _fourier(x, J):
    j = np.arange(1, J + 1)
    ang = np.asarray(x)[..., None] * j
    return np.concatenate([np.cos(ang), np.sin(ang)], axis=-1)


def simulate_shared(seed=0, N=30, T=1000, kappas=(8.0, 3.0, 0.5), tuning="gp",
                    step_sd=0.25, gp_ell=0.12, gp_J=6, bias=2.5, scale=1.0,
                    vm_kappa=3.0, vm_gain=(3.0, 8.0), vm_base=0.3, behavior_map=None):
    """One latent trajectory and one set of spikes, several behavior traces.

    As in the preprint's Fig 1 and Fig 2, spiking and the latent are identical
    across settings and only the behavioral observation changes. kappa = 0 gives
    uniform behavior, uncorrelated with the latent.

    tuning = "gp": log rate is a draw from a periodic squared-exponential GP
        (Fourier expansion, normalized to unit standard deviation over the ring).
    tuning = "vm": von Mises bumps, as in the preprint's conceptual Fig 1.
    gp_ell is in units of the period. The preprint's exact kernel is not
    reproduced. The value here gives one to three harmonics per neuron.
    """
    rng = np.random.default_rng(seed)
    x = wrap(np.cumsum(step_sd * rng.standard_normal(T)) + rng.uniform(-np.pi, np.pi))
    ring = np.linspace(-np.pi, np.pi, 512, endpoint=False)

    if tuning == "gp":
        j = np.arange(1, gp_J + 1)
        S = np.exp(-0.5 * (2 * np.pi * gp_ell * j) ** 2)
        coef = rng.standard_normal((N, 2 * gp_J)) * np.sqrt(np.concatenate([S, S]))
        coef /= (_fourier(ring, gp_J) @ coef.T).std(0)[:, None]

        def rate(xx):
            return bias * np.exp(scale * (_fourier(xx, gp_J) @ coef.T))
    else:
        pref = wrap(np.linspace(-np.pi, np.pi, N, endpoint=False) + 0.1 * rng.standard_normal(N))
        gain = rng.uniform(vm_gain[0], vm_gain[1], N)

        def rate(xx):
            xx = np.asarray(xx)
            return vm_base + gain[None, :] * np.exp(vm_kappa * (np.cos(xx[:, None] - pref[None, :]) - 1.0))

    y = rng.poisson(rate(x)).astype(np.float32)
    b_mean = x if behavior_map is None else behavior_map(x)
    bs = []
    for k in kappas:
        if k > 0:
            bs.append(np.array([rng.vonmises(xi, k) for xi in b_mean], dtype=np.float32))
        else:
            bs.append(rng.uniform(-np.pi, np.pi, T).astype(np.float32))
    return {"y": y, "x": x, "b": bs, "kappas": tuple(kappas), "rate": rate, "ring": ring,
            "b_mean": np.asarray(b_mean)}
