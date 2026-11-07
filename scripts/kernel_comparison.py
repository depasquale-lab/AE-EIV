"""Compare tuning-curve priors on the ring: Matern-type (default) and ReLU NNGP of depth 1, 2, 3.

smooth: tuning drawn from a periodic squared-exponential GP (the Fig 2 setup), N = 30,
        T = 1000, kappa = 8, 3, 0.5.
sharp : von Mises bump tuning (concentration 6), N = 30, T = 1000, kappa = 3.
Each fit uses 80% of the bins with kappa learned for 3000 steps and is scored on the
other 20%. Writes results/kernel_comparison.json.
"""
import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

import amortized_eiv as ae
from amortized_eiv import metrics as M
from amortized_eiv import model as m

KERNELS = {"matern": ("matern", 1), "nngp_d1": ("nngp", 1), "nngp_d2": ("nngp", 2), "nngp_d3": ("nngp", 3)}


def run_one(d, i, kernel, depth, steps, seed, extra=None):
    y_all, b_all, x_all = d["y"], d["b"][i], d["x"]
    T = len(y_all)
    rng = np.random.default_rng(100 + seed)
    perm = rng.permutation(T)
    te = np.sort(np.array_split(perm, 5)[0])
    tr = np.setdiff1d(perm, te)
    params, cfg, stats, feat, _ = ae.fit(y_all[tr], b_all[tr], np.zeros(len(tr), int), 1, steps=steps,
                                         seed=seed, verbose=False,
                                         cfg_kwargs={"kernel": kernel, "depth": depth, **(extra or {})})
    yt, bt = jnp.asarray(y_all[te]), jnp.asarray(b_all[te])
    zt = jnp.zeros(len(te), int)
    j, mb = m.heldout_loglik(params, cfg, yt, bt, zt)
    kap = m.kappas(params, cfg)[zt]
    joint = (m.log_prior_grid(params, cfg)[None, :] + m.log_lik_grid(params, cfg, yt)
             + m.log_beh_grid(cfg, bt, kap))
    q = jax.nn.softmax(joint, axis=-1)
    mu = np.asarray(jnp.angle((q * jnp.exp(1j * m.grid(cfg))[None, :]).sum(-1)))
    ring = d["ring"]
    est = np.exp(np.asarray(m.log_rate_at(params, cfg, ring)))
    true = d["rate"](ring)
    tuning_r = float(np.mean([np.corrcoef(est[:, n], true[:, n])[0, 1] for n in range(true.shape[1])]))
    return {"kappa": float(m.kappas(params, cfg)[0]), "heldout": float(jnp.mean(j - mb)),
            "latent_err": float(M.circ_err(mu, x_all[te]).mean()), "tuning_r": tuning_r}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--control", action="store_true",
                    help="Matern priors with other hyperparameters, variance-matched to the default")
    a = ap.parse_args()
    if a.control:
        from amortized_eiv import kernels as K
        j2 = np.arange(1, 13) ** 2
        v0 = (K.matern_tau(j2) ** 2).sum()
        variants = {"matern_default": {}}
        for name, (ell, nu) in {"matern_l1.5_nu2": (1.5, 2.0), "matern_l1.5_nu3": (1.5, 3.0)}.items():
            v1 = (K.matern_tau(j2, 2.0, ell, nu) ** 2).sum()
            variants[name] = {"ell": ell, "nu": nu, "tau0": float(2.0 * np.sqrt(v0 / v1))}
        out = {"smooth_kappa3": {}, "sharp_kappa3": {}}
        for seed in a.seeds:
            for grp, d, i in (("smooth_kappa3", ae.simulate_shared(seed=seed, kappas=(8.0, 3.0, 0.5), tuning="gp"), 1),
                              ("sharp_kappa3", ae.simulate_shared(seed=seed, kappas=(3.0,), tuning="vm", vm_kappa=6.0), 0)):
                for name, extra in variants.items():
                    r = run_one(d, i, "matern", 1, a.steps, seed, extra)
                    out[grp].setdefault(name, []).append(r)
                    print(grp, seed, name, {x: round(v, 3) for x, v in r.items()}, flush=True)
            json.dump({"variants": variants, **out}, open("results/matern_control.json", "w"), indent=2)
        raise SystemExit
    out = {"smooth": {}, "sharp": {}}
    for seed in a.seeds:
        d = ae.simulate_shared(seed=seed, kappas=(8.0, 3.0, 0.5), tuning="gp")
        for i, k in enumerate(d["kappas"]):
            for name, (kern, depth) in KERNELS.items():
                r = run_one(d, i, kern, depth, a.steps, seed)
                out["smooth"].setdefault(f"kappa{k:g}", {}).setdefault(name, []).append(r)
                print("smooth", k, seed, name, {x: round(v, 3) for x, v in r.items()}, flush=True)
        d = ae.simulate_shared(seed=seed, kappas=(3.0,), tuning="vm", vm_kappa=6.0)
        for name, (kern, depth) in KERNELS.items():
            r = run_one(d, 0, kern, depth, a.steps, seed)
            out["sharp"].setdefault("kappa3", {}).setdefault(name, []).append(r)
            print("sharp ", 3.0, seed, name, {x: round(v, 3) for x, v in r.items()}, flush=True)
        json.dump(out, open("results/kernel_comparison.json", "w"), indent=2)
