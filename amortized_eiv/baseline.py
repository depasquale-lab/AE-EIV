"""Preprint-style baselines for timing comparisons (not the authors' code).

Same Fourier-basis Poisson decoder and von Mises behavior model as the main model,
but no encoder. The marginal likelihood over the latent is either

  fit_qmc  : approximated with quasi-Monte Carlo points from a Roberts sequence,
             with a fresh random offset each step (the preprint's sampler), or
  fit_grid : computed exactly as a sum over a fixed grid on the ring.

kappa is fixed (grid-over-kappa protocol) or learned by gradient. Gradients of the
marginal likelihood are the generalized-EM update.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from jax.scipy.special import gammaln, logsumexp

from .model import (Config, grid, init_params, kappas, log_beh_grid, log_i0,
                    log_lik_grid, log_prior_grid, log_rate_at, tau)

_ALPHA = 0.6180339887498949  # 1 / golden ratio, the one-dimensional Roberts generator


def qmc_points(key, M):
    off = jax.random.uniform(key)
    u = (off + jnp.arange(1, M + 1) * _ALPHA) % 1.0
    return 2 * jnp.pi * u - jnp.pi


def _penalty(params, cfg, T):
    return 0.5 * jnp.sum((params["W"] / tau(cfg)[:, None]) ** 2) / T


def qmc_loss(params, cfg, y, b, key, M):
    kap = kappas(params, cfg)[0]
    xs = qmc_points(key, M)
    logF = log_rate_at(params, cfg, xs)                       # (M, N)
    ll = (y @ logF.T - jnp.exp(logF).sum(-1)[None, :]
          - gammaln(y + 1.0).sum(-1)[:, None])                # (T, M)
    lb = kap * jnp.cos(b[:, None] - xs[None, :]) - jnp.log(2 * jnp.pi) - log_i0(kap)
    lm = logsumexp(ll + lb, axis=1) - jnp.log(M)
    return -lm.mean() + _penalty(params, cfg, y.shape[0])


def grid_loss(params, cfg, y, b):
    kap = kappas(params, cfg)[0] * jnp.ones(y.shape[0])
    joint = (log_prior_grid(params, cfg)[None, :] + log_lik_grid(params, cfg, y)
             + log_beh_grid(cfg, b, kap))
    return -logsumexp(joint, axis=1).mean() + _penalty(params, cfg, y.shape[0])


def _run(loss_of, params, steps, lr, seed, learn_kappa, uses_key):
    opt = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(lr))
    state = opt.init(params)

    @jax.jit
    def step(params, state, key):
        loss, g = jax.value_and_grad(loss_of)(params, key)
        if not learn_kappa:
            g = {**g, "log_kappa": jnp.zeros_like(g["log_kappa"])}
        upd, state = opt.update(g, state, params)
        return optax.apply_updates(params, upd), state, loss

    key = jax.random.PRNGKey(seed + 1)
    for _ in range(steps):
        key, k = jax.random.split(key)
        params, state, _ = step(params, state, k)
    return params


def fit_qmc(y, b, kappa, *, M=1024, steps=1200, lr=5e-3, seed=0, learn_kappa=False, cfg_kwargs=None):
    y = jnp.asarray(y, dtype=jnp.float32); b = jnp.asarray(b, dtype=jnp.float32)
    cfg = Config(N=y.shape[1], n_cond=1, learn_prior=False, **(cfg_kwargs or {}))
    params = init_params(jax.random.PRNGKey(seed), cfg, y)
    params["log_kappa"] = jnp.log(jnp.full((1,), kappa, dtype=jnp.float32))
    params = _run(lambda p, k: qmc_loss(p, cfg, y, b, k, M), params, steps, lr, seed, learn_kappa, True)
    return params, cfg


def fit_grid(y, b, kappa, *, G=128, steps=1200, lr=5e-3, seed=0, learn_kappa=False,
             learn_prior=False, cfg_kwargs=None):
    y = jnp.asarray(y, dtype=jnp.float32); b = jnp.asarray(b, dtype=jnp.float32)
    cfg = Config(N=y.shape[1], n_cond=1, G=G, learn_prior=learn_prior, **(cfg_kwargs or {}))
    params = init_params(jax.random.PRNGKey(seed), cfg, y)
    params["log_kappa"] = jnp.log(jnp.full((1,), kappa, dtype=jnp.float32))
    params = _run(lambda p, k: grid_loss(p, cfg, y, b), params, steps, lr, seed, learn_kappa, False)
    return params, cfg
