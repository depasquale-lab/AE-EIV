"""Amortized error-in-variables (EIV) model on the ring.

Generative model, per time bin t, with latent x_t on the circle:

    x_t       ~ p_phi(x)                         learned prior
    b_t | x_t ~ VonMises(x_t, kappa_t)           measured behavior
    y_nt | x_t ~ Poisson(exp(b_n + Phi(x_t) W_n))  Fourier-basis tuning curves

kappa_t = kappa_{c(t)} is a learned, condition-dependent behavior fidelity.

Encoder q(x_t | y_t, b_t) is von Mises with complex parameter

    z_t = z_psi(y_t) + kappa_gate_t * exp(i b_t)

so behavior enters through a gate. kappa_gate = 0 gives a neural-only encoder.

The latent is represented on a grid of G points, so every expectation in the
ELBO is an exact sum over the grid and no sampling is needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, i0e

from . import kernels as _kernels


@dataclass(frozen=True)
class Config:
    N: int                      # neurons
    n_cond: int = 1             # number of behavioral conditions
    J: int = 12                 # Fourier harmonics in the tuning curves
    G: int = 128                # grid points on the ring
    K_prior: int = 3            # harmonics in the learned prior log-density
    hidden: int = 64            # encoder hidden width
    tau0: float = 2.0           # tuning-curve prior scale
    ell: float = 3.0            # tuning-curve prior frequency scale
    nu: float = 2.0             # tuning-curve prior decay exponent
    kappa_min: float = 1e-3
    kappa_max: float = 500.0
    learn_prior: bool = True
    kernel: str = "matern"      # prior on tuning-curve weights: "matern" or "nngp"
    depth: int = 1              # ReLU network depth for kernel = "nngp"


def log_i0(k):
    return jnp.log(i0e(k)) + k


def grid(cfg: Config):
    return jnp.linspace(-jnp.pi, jnp.pi, cfg.G, endpoint=False)


def fourier_basis(x, J):
    j = jnp.arange(1, J + 1)
    ang = x[..., None] * j
    return jnp.concatenate([jnp.cos(ang), jnp.sin(ang)], axis=-1)


def tau(cfg: Config):
    t = jnp.asarray(_kernels.ring_tau(cfg.J, cfg.kernel, cfg.depth, cfg.tau0, cfg.ell, cfg.nu),
                    dtype=jnp.float32)
    return jnp.concatenate([t, t])


# ---------------------------------------------------------------- parameters

def features(y, stats=None):
    f = jnp.log1p(jnp.asarray(y, dtype=jnp.float32))
    if stats is None:
        stats = (f.mean(0), f.std(0) + 1e-3)
    return (f - stats[0]) / stats[1], stats


def _dense(key, n_in, n_out, scale=1.0):
    return (jax.random.normal(key, (n_in, n_out)) * scale / jnp.sqrt(n_in),
            jnp.zeros(n_out))


def init_params(key, cfg: Config, y):
    k = jax.random.split(key, 5)
    y = jnp.asarray(y, dtype=jnp.float32)
    return {
        "W": 0.05 * jax.random.normal(k[0], (2 * cfg.J, cfg.N)),
        "b0": jnp.log(y.mean(0) + 1e-2),
        "enc": [_dense(k[1], cfg.N, cfg.hidden),
                _dense(k[2], cfg.hidden, cfg.hidden),
                _dense(k[3], cfg.hidden, 2, scale=0.1)],
        "log_kappa": jnp.full((cfg.n_cond,), jnp.log(2.0)),
        "prior": jnp.zeros(2 * cfg.K_prior),
    }


def kappas(params, cfg: Config):
    return jnp.clip(jnp.exp(params["log_kappa"]), cfg.kappa_min, cfg.kappa_max)


# ------------------------------------------------------------------- pieces

def _mlp(layers, h):
    for W, b in layers[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = layers[-1]
    return h @ W + b


def encoder_z(params, feat, b, kappa_gate):
    """Complex parameter of q(x | y, b) as a (T, 2) real array."""
    z_y = _mlp(params["enc"], feat)
    beh = jnp.stack([jnp.cos(b), jnp.sin(b)], axis=-1)
    return z_y + kappa_gate[:, None] * beh


def log_q_grid(z, cfg: Config):
    g = grid(cfg)
    logits = z[:, 0:1] * jnp.cos(g) + z[:, 1:2] * jnp.sin(g)
    return jax.nn.log_softmax(logits, axis=-1)


def log_prior_grid(params, cfg: Config):
    g = grid(cfg)
    if not cfg.learn_prior:
        return jnp.full((cfg.G,), -jnp.log(cfg.G))
    k = jnp.arange(1, cfg.K_prior + 1)
    ang = g[:, None] * k
    basis = jnp.concatenate([jnp.cos(ang), jnp.sin(ang)], axis=-1)
    return jax.nn.log_softmax(basis @ params["prior"])


def log_rate_grid(params, cfg: Config):
    """log f_n(x_g), shape (G, N)."""
    return params["b0"][None, :] + fourier_basis(grid(cfg), cfg.J) @ params["W"]


def log_lik_grid(params, cfg: Config, y):
    """log p(y_t | x_g), shape (T, G)."""
    logF = log_rate_grid(params, cfg)
    return (y @ logF.T
            - jnp.exp(logF).sum(-1)[None, :]
            - gammaln(y + 1.0).sum(-1)[:, None])


def log_beh_grid(cfg: Config, b, kappa_t):
    """log VM(b_t | x_g, kappa_t), shape (T, G)."""
    d = b[:, None] - grid(cfg)[None, :]
    return (kappa_t[:, None] * jnp.cos(d)
            - jnp.log(2 * jnp.pi) - log_i0(kappa_t)[:, None])


# --------------------------------------------------------------------- ELBO

def elbo_per_bin(params, cfg: Config, feat, y, b, cond, gate_mask=None):
    """Per-bin ELBO on log p(y_t, b_t), and the exact grid log marginal."""
    kap = kappas(params, cfg)[cond]
    gate = kap if gate_mask is None else kap * gate_mask
    z = encoder_z(params, feat, b, gate)
    logq = log_q_grid(z, cfg)
    joint = (log_prior_grid(params, cfg)[None, :]
             + log_lik_grid(params, cfg, y)
             + log_beh_grid(cfg, b, kap))
    elbo = (jnp.exp(logq) * (joint - logq)).sum(-1)
    log_marg = jax.scipy.special.logsumexp(joint, axis=-1)
    return elbo, log_marg


def loss_fn(params, cfg: Config, feat, y, b, cond, gate_mask=None):
    elbo, _ = elbo_per_bin(params, cfg, feat, y, b, cond, gate_mask)
    T = y.shape[0]
    pen = 0.5 * jnp.sum((params["W"] / tau(cfg)[:, None]) ** 2)
    pen = pen + 0.5 * jnp.sum(params["prior"] ** 2)
    return -elbo.mean() + pen / T


def heldout_loglik(params, cfg: Config, y, b, cond):
    """Exact grid log p(y_t, b_t) and log p(b_t) per bin. Their difference is
    log p(y_t | b_t), the quantity compared across kappa."""
    kap = kappas(params, cfg)[cond]
    lp = log_prior_grid(params, cfg)[None, :]
    lb = log_beh_grid(cfg, b, kap)
    joint = jax.scipy.special.logsumexp(lp + log_lik_grid(params, cfg, y) + lb, axis=-1)
    marg_b = jax.scipy.special.logsumexp(lp + lb, axis=-1)
    return joint, marg_b


def log_rate_at(params, cfg: Config, x):
    """log f_n(x) at arbitrary x, shape (len(x), N)."""
    return params["b0"][None, :] + fourier_basis(jnp.asarray(x), cfg.J) @ params["W"]


# ------------------------------------------------------------------- readout

def posterior(params, cfg: Config, feat, b, cond, use_behavior=True):
    """Posterior mean direction mu_t, resultant length rho_t, and q on the grid."""
    kap = kappas(params, cfg)[cond]
    gate = kap if use_behavior else jnp.zeros_like(kap)
    q = jnp.exp(log_q_grid(encoder_z(params, feat, b, gate), cfg))
    c = (q * jnp.exp(1j * grid(cfg))[None, :]).sum(-1)
    return jnp.angle(c), jnp.abs(c), q


def generate(params, cfg: Config, key, T):
    """Sample neural activity from the generative model with behavior ignored."""
    k1, k2 = jax.random.split(key)
    idx = jax.random.categorical(k1, log_prior_grid(params, cfg), shape=(T,))
    rate = jnp.exp(log_rate_grid(params, cfg))[idx]
    return jax.random.poisson(k2, rate), grid(cfg)[idx]
