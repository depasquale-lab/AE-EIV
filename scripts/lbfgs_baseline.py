"""Does an L-BFGS baseline erase the convergence advantage?

Deterministic marginal likelihood (quasi-Monte Carlo points with a fixed offset,
M = 1024) with kappa fixed at 3, optimized by L-BFGS with line search. Same data,
fold, decoder, and held-out score as scripts/convergence_diagnostics.py.
Writes results/lbfgs_results.json.
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
zt = jnp.zeros(len(te), int)

cfg = ae.Config(N=30, n_cond=1, learn_prior=False)
base = ae.init_params(jax.random.PRNGKey(0), cfg, y)
base["log_kappa"] = jnp.log(jnp.full((1,), 3.0))
key = jax.random.PRNGKey(0)          # fixed offset: a deterministic objective for L-BFGS


def loss(p):
    return B.qmc_loss({**base, **p}, cfg, y, b, key, 1024)


def heldout(p):
    j, mb = m.heldout_loglik({**base, **p}, cfg, yt, bt, zt)
    return float(jnp.mean(j - mb))


p = {"W": base["W"], "b0": base["b0"]}
opt = optax.lbfgs()
state = opt.init(p)
vg = optax.value_and_grad_from_state(loss)


@jax.jit
def step(p, state):
    v, g = vg(p, state=state)
    u, state = opt.update(g, state, p, value=v, grad=g, value_fn=loss)
    return optax.apply_updates(p, u), state, v


rows, marks = [], {5, 10, 20, 40, 80, 160, 320}
t0 = time.perf_counter()
for _ in range(3):                    # compilation and the first retrace
    p, state, _ = step(p, state)
jax.block_until_ready(p["W"])
compile_s = time.perf_counter() - t0
clock = time.perf_counter()
for i in range(4, 321):
    p, state, v = step(p, state)
    if i in marks:
        jax.block_until_ready(p["W"])
        rows.append({"iterations": i, "wall_s_excl_compile": time.perf_counter() - clock,
                     "heldout": heldout(p)})
        print(rows[-1], flush=True)
json.dump({"compile_s": compile_s, "rows": rows}, open("results/lbfgs_results.json", "w"), indent=2)
