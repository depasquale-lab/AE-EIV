"""Held-out score versus training time, and what explains the gap to the QMC baseline.

Same data, fold, decoder, and kappa (3.0) as scripts/benchmark_speed.py.
Writes results/convergence_results.json.
"""
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

import amortized_eiv as ae
from amortized_eiv import baseline as B
from amortized_eiv import model as m

d = ae.simulate_shared(seed=0, kappas=(3.0,), tuning="gp")
y_all, b_all = d["y"], d["b"][0]
T = len(y_all)
rng = np.random.default_rng(100)
perm = rng.permutation(T)
te = np.sort(np.array_split(perm, 5)[0])
tr = np.setdiff1d(perm, te)
y, b = jnp.asarray(y_all[tr]), jnp.asarray(b_all[tr])
yt, bt = jnp.asarray(y_all[te]), jnp.asarray(b_all[te])
z, zt = jnp.zeros(len(tr), int), jnp.zeros(len(te), int)


def score(p, cfg, yy, bb, zz):
    j, mb = m.heldout_loglik(p, cfg, yy, bb, zz)
    return float(jnp.mean(j - mb))


out = {"qmc_M1024": [], "amortized": [], "exact_grid_direct": [], "prior_ablation": {}}
for steps in (1200, 4000, 10000):
    t = time.perf_counter(); p, cfg = B.fit_qmc(y, b, 3.0, M=1024, steps=steps); w = time.perf_counter() - t
    out["qmc_M1024"].append({"steps": steps, "wall_s": w, "train": score(p, cfg, y, b, z), "heldout": score(p, cfg, yt, bt, zt)})
for steps in (300, 600, 1200):
    t = time.perf_counter()
    p, cfg, _, _, _ = ae.fit(y, b, z, 1, steps=steps, kappa_init=3.0, freeze_kappa=True, verbose=False)
    w = time.perf_counter() - t
    out["amortized"].append({"steps": steps, "wall_s": w, "train": score(p, cfg, y, b, z), "heldout": score(p, cfg, yt, bt, zt)})
for lp in (True, False):
    p, cfg, _, _, _ = ae.fit(y, b, z, 1, steps=1200, kappa_init=3.0, freeze_kappa=True, verbose=False,
                             cfg_kwargs={"learn_prior": lp})
    out["prior_ablation"][f"learn_prior_{lp}"] = score(p, cfg, yt, bt, zt)

# exact grid marginal optimized directly: no encoder, no sampling
cfg = ae.Config(N=30, n_cond=1, learn_prior=False)
feat, _ = ae.features(y)
p = ae.init_params(jax.random.PRNGKey(0), cfg, y)
p["log_kappa"] = jnp.log(jnp.full((1,), 3.0))


def loss(p):
    _, lm = m.elbo_per_bin(p, cfg, feat, y, b, z)
    return -lm.mean() + 0.5 * jnp.sum((p["W"] / m.tau(cfg)[:, None]) ** 2) / len(tr)


opt = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(5e-3))
st = opt.init(p)


@jax.jit
def step(p, st):
    l, g = jax.value_and_grad(loss)(p)
    g = {**g, "log_kappa": jnp.zeros_like(g["log_kappa"])}
    u, st = opt.update(g, st, p)
    return optax.apply_updates(p, u), st, l


for i in range(1, 4001):
    p, st, _ = step(p, st)
    if i in (1200, 4000):
        out["exact_grid_direct"].append({"steps": i, "train": score(p, cfg, y, b, z), "heldout": score(p, cfg, yt, bt, zt)})
json.dump(out, open("results/convergence_results.json", "w"), indent=2)
print(json.dumps(out, indent=1))
