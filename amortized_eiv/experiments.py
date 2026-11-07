"""Simulation experiments shared by scripts/smoke_test.py and the tests."""
from __future__ import annotations

import jax
import numpy as np

from . import metrics as M
from . import model as m
from .simulate import simulate
from .train import fit


def _per_cond(mu, d):
    return {f"cond{c}": float(M.circ_err(mu[d["cond"] == c], d["x"][d["cond"] == c]).mean())
            for c in range(len(d["kappas"]))}


def run_ab(steps=3000, seed=0, T_per=1000, N=30, gain=(3.0, 8.0), kappas=(8.0, 3.0, 0.5)):
    """Shared tuning across conditions with different behavior noise.

    Reports learned kappa, latent error with the behavior gate on and off, and
    the error of raw behavior as a point estimate of the latent.
    """
    d = simulate(seed=seed, N=N, T_per=T_per, kappas=kappas, gain=gain)
    params, cfg, stats, feat, _ = fit(d["y"], d["b"], d["cond"], len(kappas),
                                      steps=steps, seed=seed, verbose=False)
    out = {"kappa_true": list(kappas),
           "kappa_learned": [round(float(k), 3) for k in m.kappas(params, cfg)],
           "behavior_mae": _per_cond(d["b"], d)}
    for gate, name in ((True, "latent_mae_gate_on"), (False, "latent_mae_gate_off")):
        mu, rho, _ = m.posterior(params, cfg, feat, d["b"], d["cond"], use_behavior=gate)
        out[name] = _per_cond(np.asarray(mu), d)
    return out


def run_noisy(steps=3000, seed=0, T=3000, kappa=0.1):
    """One condition with nearly uninformative behavior."""
    d = simulate(seed=seed + 1, kappas=(kappa,), T_per=T)
    params, cfg, stats, feat, _ = fit(d["y"], d["b"], d["cond"], 1,
                                      steps=steps, seed=seed, verbose=False)
    mu, _, _ = m.posterior(params, cfg, feat, d["b"], d["cond"])
    aligned, _ = M.align(np.asarray(mu), d["x"])
    y_gen, _ = m.generate(params, cfg, jax.random.PRNGKey(7), T)
    y_gen = np.asarray(y_gen, dtype=float)
    return {
        "kappa_true": kappa,
        "kappa_learned": round(float(m.kappas(params, cfg)[0]), 3),
        "latent_mae_aligned": float(M.circ_err(aligned, d["x"]).mean()),
        "behavior_mae": float(M.circ_err(d["b"], d["x"]).mean()),
        "mean_rate_corr": float(np.corrcoef(y_gen.mean(0), d["y"].mean(0))[0, 1]),
        "covariance_corr": M.offdiag_corr(np.cov(y_gen.T), np.cov(d["y"].T)),
    }


# ------------------------------------------------------- paper-style examples

def fit_free(d, i, steps=3000, seed=0, align=False):
    """Fit one behavior setting of a shared-latent simulation with kappa learned."""
    y, b = d["y"], d["b"][i]
    T = len(y)
    params, cfg, stats, feat, _ = fit(y, b, np.zeros(T, int), 1, steps=steps, seed=seed, verbose=False)
    mu, rho, _ = m.posterior(params, cfg, feat, b, np.zeros(T, int))
    mu = np.asarray(mu)
    s, shift = 1.0, 0.0
    if align:
        _, (s, shift) = M.align(mu, d["x"])
    mu_a = np.angle(np.exp(1j * (s * mu + shift)))
    est = np.exp(np.asarray(m.log_rate_at(params, cfg, s * (d["ring"] - shift))))
    return {"kappa": float(m.kappas(params, cfg)[0]), "mu": mu_a, "rho": np.asarray(rho),
            "est": est, "sign": s, "shift": shift}


def kappa_grid_cv(d, i, grid, folds=2, steps=1200, seed=0):
    """Preprint protocol: fit with kappa fixed on each grid value, score held-out
    log p(y | b) on random held-out bins. Returns (len(grid), folds)."""
    y, b = d["y"], d["b"][i]
    T = len(y)
    rng = np.random.default_rng(100 + seed)
    perm = rng.permutation(T)
    parts = np.array_split(perm, 5)
    out = np.zeros((len(grid), folds))
    for f in range(folds):
        te = np.sort(parts[f])
        tr = np.setdiff1d(perm, te)
        for ki, k in enumerate(grid):
            params, cfg, _, _, _ = fit(y[tr], b[tr], np.zeros(len(tr), int), 1, steps=steps,
                                       seed=seed, verbose=False, kappa_init=k, freeze_kappa=True)
            joint, marg_b = m.heldout_loglik(params, cfg, y[te], b[te], np.zeros(len(te), int))
            out[ki, f] = float(np.mean(joint - marg_b))
    return out
