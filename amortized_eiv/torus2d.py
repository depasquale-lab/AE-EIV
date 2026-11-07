"""Two-dimensional torus latent: where grid and quasi-Monte Carlo integration get expensive.

Same model family as the ring, with x_t in [-pi, pi)^2, a 2D Fourier-basis Poisson
decoder, b_t | x_t ~ VM(x_t1, kappa) * VM(x_t2, kappa), and a uniform prior.

Three ways to fit it, all minibatched with the same decoder:
  amortized : encoder q(x | y, gate*b) = N(mu, diag sigma^2), reparameterized samples.
              Cost per bin is S decoder evaluations, independent of grid size.
  qmc       : M quasi-Monte Carlo points (2D Roberts R2 sequence, random shift).
  grid      : exact sum on a G x G grid.
All three learn kappa by gradient. Held-out scores use one fixed fine grid, so every
method is scored by the same integrator.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.scipy.special import gammaln, logsumexp

from . import kernels as _kernels
from .model import log_i0

_PLASTIC = 1.32471795724475
_A2 = jnp.array([1.0 / _PLASTIC, 1.0 / _PLASTIC ** 2])
LOG2PI = float(np.log(2 * np.pi))


# ------------------------------------------------------------------- decoder

def freqs(K):
    f = [(j1, 0) for j1 in range(1, K + 1)]
    f += [(j1, j2) for j2 in range(1, K + 1) for j1 in range(-K, K + 1)]
    return np.array(f, dtype=np.float32)


def basis(x, F):
    ang = x @ F.T
    return jnp.concatenate([jnp.cos(ang), jnp.sin(ang)], axis=-1)


def tau(F, tau0=2.0, ell=3.0, nu=2.0, kernel="matern", depth=1):
    t = jnp.asarray(_kernels.torus_tau(np.asarray(F), kernel, depth, tau0, ell, nu), dtype=jnp.float32)
    return jnp.concatenate([t, t])


def init(key, N, y, F, hidden=128, amortized=False):
    k = jax.random.split(key, 4)
    p = {"W": 0.05 * jax.random.normal(k[0], (2 * len(F), N)),
         "b0": jnp.log(jnp.asarray(y).mean(0) + 1e-2),
         "log_kappa": jnp.log(jnp.full((1,), 2.0))}
    if amortized:
        def dense(kk, i, o, s=1.0):
            return (jax.random.normal(kk, (i, o)) * s / jnp.sqrt(i), jnp.zeros(o))
        p["enc"] = [dense(k[1], N + 4, hidden), dense(k[2], hidden, hidden), dense(k[3], hidden, 4, 0.1)]
    return p


def _kap(p):
    return jnp.clip(jnp.exp(p["log_kappa"][0]), 1e-3, 500.0)


def _mlp(layers, h):
    for W, b in layers[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = layers[-1]
    return h @ W + b


# --------------------------------------------------------------------- losses

def _ll_points(p, F, y, xs):
    logF = p["b0"][None, :] + basis(xs, F) @ p["W"]                  # (M, N)
    return y @ logF.T - jnp.exp(logF).sum(-1)[None, :]               # (B, M), no gammaln


def _lb_points(kap, b, xs):
    c = jnp.cos(b[:, None, :] - xs[None, :, :]).sum(-1)              # (B, M)
    return kap * c - 2 * log_i0(kap) - 2 * LOG2PI


def _pen(p, F, T, tv=None):
    tv = tau(F) if tv is None else tv
    return 0.5 * jnp.sum((p["W"] / tv[:, None]) ** 2) / T


def qmc_points(key, M):
    off = jax.random.uniform(key, (2,))
    u = (off[None, :] + jnp.arange(1, M + 1)[:, None] * _A2[None, :]) % 1.0
    return 2 * jnp.pi * u - jnp.pi


def grid_points(G):
    g = jnp.linspace(-jnp.pi, jnp.pi, G, endpoint=False)
    a, b = jnp.meshgrid(g, g, indexing="ij")
    return jnp.stack([a.ravel(), b.ravel()], axis=-1)


def loss_points(p, F, y, b, xs, T, tv=None):
    lg = _ll_points(p, F, y, xs) + _lb_points(_kap(p), b, xs)
    return -(logsumexp(lg, axis=1) - jnp.log(xs.shape[0])).mean() + _pen(p, F, T, tv)


def loss_amortized(p, F, key, feat, y, b, T, S=2, tv=None):
    kap = _kap(p)
    inp = jnp.concatenate([feat, kap * jnp.cos(b), kap * jnp.sin(b)], axis=-1)
    out = _mlp(p["enc"], inp)
    mu, sig = out[:, :2], jnp.exp(jnp.clip(out[:, 2:], -4.0, 0.5))
    x = mu[None] + sig[None] * jax.random.normal(key, (S,) + mu.shape)    # (S, B, 2)
    logF = p["b0"][None, None, :] + basis(x, F) @ p["W"]                   # (S, B, N)
    ll = (y[None] * logF - jnp.exp(logF)).sum(-1).mean(0)
    lb = kap * (jnp.exp(-0.5 * sig ** 2) * jnp.cos(b - mu)).sum(-1) - 2 * log_i0(kap) - 2 * LOG2PI
    ent = (0.5 * jnp.log(2 * jnp.pi * jnp.e * sig ** 2)).sum(-1)
    return -(ll + lb - 2 * LOG2PI + ent).mean() + _pen(p, F, T, tv)


def supervised_init(p, F, y, b, steps=800, lr=1e-2, tv=None):
    """Poisson regression of y on the Fourier basis of behavior (kappa -> infinity).
    Initializes the decoder for every method, following the preprint's use of
    supervised tuning curves as a starting point."""
    y, b = jnp.asarray(y), jnp.asarray(b)
    T = y.shape[0]
    q = {"W": p["W"], "b0": p["b0"]}

    def loss(q):
        logF = q["b0"][None, :] + basis(b, F) @ q["W"]
        return -(y * logF - jnp.exp(logF)).sum(-1).mean() + _pen(q, F, T, tv)

    opt = optax.adam(lr)
    st = opt.init(q)

    @jax.jit
    def step(q, st):
        g = jax.grad(loss)(q)
        u, st = opt.update(g, st, q)
        return optax.apply_updates(q, u), st

    for _ in range(steps):
        q, st = step(q, st)
    return {**p, **q}


# ----------------------------------------------------------------- evaluation

def evaluate(p, F, y, b, x_true, G=96):
    """Held-out log p(y | b) per bin and exact-posterior latent error, on a fixed fine grid."""
    xs = grid_points(G)
    ll = _ll_points(p, F, y, xs) - gammaln(y + 1.0).sum(-1)[:, None]
    lg = ll + _kap(p) * jnp.cos(b[:, None, :] - xs[None, :, :]).sum(-1)
    lse = logsumexp(lg, axis=1)
    heldout = float(jnp.mean(lse - 2 * log_i0(_kap(p)) - 2 * jnp.log(G)))
    w = jax.nn.softmax(lg, axis=1)
    est = jnp.stack([jnp.angle(w @ jnp.exp(1j * xs[:, d])) for d in range(2)], axis=-1)
    err = np.abs(np.angle(np.exp(1j * (np.asarray(est) - np.asarray(x_true))))).mean()
    return heldout, float(err)


def encoder_error(p, F, feat, b, x_true):
    kap = _kap(p)
    inp = jnp.concatenate([feat, kap * jnp.cos(b), kap * jnp.sin(b)], axis=-1)
    mu = _mlp(p["enc"], inp)[:, :2]
    return float(np.abs(np.angle(np.exp(1j * (np.asarray(mu) - np.asarray(x_true))))).mean())


# ------------------------------------------------------------------ simulation

def simulate_torus(seed=0, N=60, T=2000, kappa=4.0, K=3, ell=0.12, step_sd=0.35, bias=2.5):
    rng = np.random.default_rng(seed)
    F = freqs(K)
    x = np.angle(np.exp(1j * (np.cumsum(step_sd * rng.standard_normal((T, 2)), axis=0)
                              + rng.uniform(-np.pi, np.pi, 2))))
    S = np.exp(-0.5 * (2 * np.pi * ell) ** 2 * (F ** 2).sum(1))
    coef = rng.standard_normal((N, 2 * len(F))) * np.sqrt(np.concatenate([S, S]))
    g = np.linspace(-np.pi, np.pi, 64, endpoint=False)
    ring = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    ang = ring @ F.T
    B = np.concatenate([np.cos(ang), np.sin(ang)], axis=-1)
    coef /= (B @ coef.T).std(0)[:, None]
    ang = x @ F.T
    h = np.concatenate([np.cos(ang), np.sin(ang)], axis=-1) @ coef.T
    y = rng.poisson(bias * np.exp(h)).astype(np.float32)
    b = np.stack([[rng.vonmises(xi[d], kappa) for xi in x] for d in range(2)], axis=-1).astype(np.float32)
    return {"y": y, "b": b, "x": x.astype(np.float32), "F": F}


# --------------------------------------------------------------------- training

def train(method, y, b, y_te, b_te, x_te, F, *, steps, checkpoints, batch=512, lr=5e-3,
          seed=0, S=2, M=4096, G=64, verbose=True, sup_init=True, hidden=128,
          kernel="matern", depth=1):
    """method in {'amortized', 'qmc', 'grid'}. Returns history with training time that
    excludes compilation and evaluation."""
    y, b = jnp.asarray(y), jnp.asarray(b)
    T = y.shape[0]
    feat = jnp.log1p(y)
    feat = (feat - feat.mean(0)) / (feat.std(0) + 1e-3)
    key = jax.random.PRNGKey(seed)
    key, k0 = jax.random.split(key)
    p = init(k0, y.shape[1], y, F, hidden=hidden, amortized=(method == "amortized"))
    tv = tau(F, kernel=kernel, depth=depth)
    if sup_init:
        p = supervised_init(p, F, y, b, tv=tv)
    opt = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(lr))
    state = opt.init(p)
    pts = grid_points(G) if method == "grid" else None

    def loss(p, key, idx):
        yb, bb = y[idx], b[idx]
        if method == "amortized":
            return loss_amortized(p, F, key, feat[idx], yb, bb, T, S, tv)
        xs = pts if method == "grid" else qmc_points(key, M)
        return loss_points(p, F, yb, bb, xs, T, tv)

    @jax.jit
    def step(p, state, key):
        k1, k2 = jax.random.split(key)
        idx = jax.random.randint(k1, (batch,), 0, T)
        l, g = jax.value_and_grad(loss)(p, k2, idx)
        u, state = opt.update(g, state, p)
        return optax.apply_updates(p, u), state, l

    t0 = time.perf_counter()          # compilation plus the retrace on the second call
    pw, sw = p, state
    for _ in range(3):
        pw, sw, _ = step(pw, sw, key)
    jax.block_until_ready(pw["W"])
    compile_s = time.perf_counter() - t0
    hist, train_t, marks = [], 0.0, set(checkpoints)
    clock = time.perf_counter()
    for i in range(1, steps + 1):
        key, k = jax.random.split(key)
        p, state, _ = step(p, state, k)
        if i in marks:
            jax.block_until_ready(p["W"])
            train_t += time.perf_counter() - clock
            ho, err = evaluate(p, F, jnp.asarray(y_te), jnp.asarray(b_te), x_te)
            row = {"step": i, "train_s": train_t, "heldout": ho, "latent_err": err,
                   "kappa": float(_kap(p))}
            if method == "amortized":
                f_te = jnp.log1p(jnp.asarray(y_te))
                f_te = (f_te - jnp.log1p(y).mean(0)) / (jnp.log1p(y).std(0) + 1e-3)
                row["encoder_err"] = encoder_error(p, F, f_te, jnp.asarray(b_te), x_te)
            hist.append(row)
            if verbose:
                print(method, {k_: round(v, 3) for k_, v in row.items()}, flush=True)
            clock = time.perf_counter()
    return {"compile_s": compile_s, "history": hist, "ms_per_step": 1000 * hist[-1]["train_s"] / steps}
