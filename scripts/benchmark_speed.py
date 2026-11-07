"""Timing and quality comparison against a preprint-style quasi-Monte Carlo baseline.

All runs: one CPU thread, T_train = 800 bins, N = 30, same decoder.
 steady : milliseconds per optimizer step
 fit    : wall time of one complete fit, including compilation
 quality: held-out log p(y | b) on the same random fold, at the true kappa
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

KAPPA_TRUE = 3.0
STEPS = 1200


def steady_ms(fn, args, n=150):
    out = fn(*args); jax.block_until_ready(out)
    t = time.perf_counter()
    for _ in range(n):
        out = fn(*args)
    jax.block_until_ready(out)
    return 1000 * (time.perf_counter() - t) / n


if __name__ == "__main__":
    d = ae.simulate_shared(seed=0, kappas=(KAPPA_TRUE,), tuning="gp")
    y_all, b_all = d["y"], d["b"][0]
    T = len(y_all)
    rng = np.random.default_rng(100)
    perm = rng.permutation(T); te = np.sort(np.array_split(perm, 5)[0]); tr = np.setdiff1d(perm, te)
    y, b = jnp.asarray(y_all[tr]), jnp.asarray(b_all[tr])
    yt, bt = jnp.asarray(y_all[te]), jnp.asarray(b_all[te])
    zero = jnp.zeros(len(tr), int); zero_t = jnp.zeros(len(te), int)
    res = {"T_train": int(len(tr)), "N": int(y.shape[1]), "steps_fixed_kappa": STEPS}

    # ---------------- steady-state cost per step
    cfg = ae.Config(N=30, n_cond=1)
    feat, _ = ae.features(y)
    p0 = ae.init_params(jax.random.PRNGKey(0), cfg, y)
    mask = jnp.ones(len(tr))

    ours = jax.jit(jax.value_and_grad(lambda p: ae.loss_fn(p, cfg, feat, y, b, zero, mask)))

    def direct_loss(p):  # exact grid marginal, no encoder
        _, lm = m.elbo_per_bin(p, cfg, feat, y, b, zero, mask)
        return -lm.mean() + 0.5 * jnp.sum((p["W"] / m.tau(cfg)[:, None]) ** 2) / len(tr)
    direct = jax.jit(jax.value_and_grad(direct_loss))

    cfg_q = ae.Config(N=30, n_cond=1, learn_prior=False)
    pq = ae.init_params(jax.random.PRNGKey(0), cfg_q, y)
    pq["log_kappa"] = jnp.log(jnp.full((1,), KAPPA_TRUE))
    qmc = {M: jax.jit(jax.value_and_grad(
        lambda p, k, M=M: B.qmc_loss(p, cfg_q, y, b, k, M))) for M in (1024, 128)}
    key = jax.random.PRNGKey(1)
    res["steady_ms_per_step"] = {
        "amortized_encoder_G128": steady_ms(ours, (p0,)),
        "exact_grid_marginal_no_encoder_G128": steady_ms(direct, (p0,)),
        "qmc_M1024_fixed_kappa": steady_ms(qmc[1024], (pq, key)),
        "qmc_M128_fixed_kappa": steady_ms(qmc[128], (pq, key)),
    }
    print(json.dumps(res["steady_ms_per_step"], indent=1), flush=True)

    # ---------------- one complete fit each, with held-out score
    def heldout(params, cfg_):
        j, mb = m.heldout_loglik(params, cfg_, yt, bt, zero_t)
        return float(jnp.mean(j - mb))

    fits = {}

    def record(name, fn, kappa_of=None):
        t = time.perf_counter()
        p, c = fn()
        jax.block_until_ready(p["W"])
        fits[name] = {"wall_s": round(time.perf_counter() - t, 2), "heldout": round(heldout(p, c), 3),
                      "kappa": round(float(m.kappas(p, c)[0]), 3)}
        print(name, fits[name], flush=True)

    def ours(**kw):
        def run():
            p, c, _, _, _ = ae.fit(y, b, zero, 1, verbose=False, **kw)
            return p, c
        return run

    record("amortized_fixed_kappa_1200", ours(steps=STEPS, kappa_init=KAPPA_TRUE, freeze_kappa=True))
    record("amortized_free_kappa_3000", ours(steps=3000))
    record("exact_grid_fixed_kappa_uniform_prior_1200", lambda: B.fit_grid(y, b, KAPPA_TRUE, steps=STEPS))
    record("exact_grid_fixed_kappa_learned_prior_1200", lambda: B.fit_grid(y, b, KAPPA_TRUE, steps=STEPS, learn_prior=True))
    record("exact_grid_free_kappa_learned_prior_3000", lambda: B.fit_grid(y, b, 2.0, steps=3000, learn_kappa=True, learn_prior=True))
    for M in (1024, 128):
        record(f"qmc_M{M}_fixed_kappa_1200", lambda M=M: B.fit_qmc(y, b, KAPPA_TRUE, M=M, steps=STEPS))
        record(f"qmc_M{M}_free_kappa_3000", lambda M=M: B.fit_qmc(y, b, 2.0, M=M, steps=3000, learn_kappa=True))
    res["fits"] = fits

    n_fits = 9 * 2   # 9-point kappa grid x 2 folds, as in scripts/fig2_validation.py
    res["kappa_selection_wall_s"] = {
        "grid_x_folds": n_fits,
        "preprint_style_qmc_M1024_grid_cv_extrapolated": round(n_fits * fits["qmc_M1024_fixed_kappa_1200"]["wall_s"], 1),
        "amortized_model_grid_cv_extrapolated": round(n_fits * fits["amortized_fixed_kappa_1200"]["wall_s"], 1),
        "amortized_single_free_kappa_fit": fits["amortized_free_kappa_3000"]["wall_s"],
        "qmc_M1024_single_free_kappa_fit": fits["qmc_M1024_free_kappa_3000"]["wall_s"],
    }
    json.dump(res, open("results/speed_results.json", "w"), indent=2)
    print(json.dumps(res["kappa_selection_wall_s"], indent=1))
