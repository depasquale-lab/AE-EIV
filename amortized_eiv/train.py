from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from .model import Config, features, init_params, kappas, loss_fn


def fit(y, b, cond, n_cond, *, steps=3000, lr=5e-3, p_drop=0.1, seed=0,
        cfg_kwargs=None, log_every=500, verbose=True, kappa_init=None,
        freeze_kappa=False):
    """Full-batch Adam on the negative ELBO.

    p_drop is the fraction of bins in which the encoder's behavior gate is set to
    zero. The generative model is unchanged, so the ELBO stays a valid bound.
    kappa_init sets the starting kappa. With freeze_kappa the kappa values are held
    fixed, which gives the preprint's grid-over-kappa protocol.
    """
    y = jnp.asarray(y, dtype=jnp.float32)
    b = jnp.asarray(b, dtype=jnp.float32)
    cond = jnp.asarray(cond)
    cfg = Config(N=y.shape[1], n_cond=n_cond, **(cfg_kwargs or {}))
    feat, stats = features(y)
    key = jax.random.PRNGKey(seed)
    key, k0 = jax.random.split(key)
    params = init_params(k0, cfg, y)
    if kappa_init is not None:
        params["log_kappa"] = jnp.log(jnp.broadcast_to(
            jnp.asarray(kappa_init, dtype=jnp.float32), (n_cond,)))

    opt = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(lr))
    state = opt.init(params)

    @jax.jit
    def step(params, state, key):
        mask = (jax.random.uniform(key, (y.shape[0],)) >= p_drop).astype(jnp.float32)
        loss, grads = jax.value_and_grad(loss_fn)(params, cfg, feat, y, b, cond, mask)
        if freeze_kappa:
            grads = {**grads, "log_kappa": jnp.zeros_like(grads["log_kappa"])}
        upd, state = opt.update(grads, state, params)
        return optax.apply_updates(params, upd), state, loss

    hist = []
    for i in range(steps):
        key, k = jax.random.split(key)
        params, state, loss = step(params, state, k)
        if i % log_every == 0 or i == steps - 1:
            kv = [float(v) for v in kappas(params, cfg)]
            hist.append((i, float(loss), kv))
            if verbose:
                print(f"step {i:5d}  loss {float(loss):9.4f}  kappa {[round(v, 2) for v in kv]}")
    return params, cfg, stats, feat, hist
