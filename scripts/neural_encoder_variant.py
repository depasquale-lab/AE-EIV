"""Neural-encoder-only variant versus the gated encoder.

gated_encoder             : the full model. Both encoders are present, and the behavior encoder (a von Mises
                            expert with weight kappa) drops out as kappa goes to 0. The behavior decoder
                            VM(b | x, kappa) is in the ELBO. The gate is on in 90% of bins during training.
gated_encoder_warp        : as above, with a learned map g(x) from latent to mean behavior (3 harmonics).
                            The behavior encoder uses the same map, kappa cos(b - g(x)).
neural_encoder_only       : ablation. The behavior encoder is removed (gate off in every bin), and behavior
                            enters only through the behavior decoder.
neural_encoder_only_warp  : the ablation with the learned map g(x).

All variants are scored on held-out bins with the behavior input removed at test time: the error of
decoding behavior from neural activity alone, g(mu(y)) against the noise-free mean behavior, plus held-out
log p(y | b) and learned kappa. Writes results/neural_encoder_variant.json.
"""
import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

import amortized_eiv as ae
from amortized_eiv import metrics as M
from amortized_eiv import model as m

VARIANTS = {
    "gated_encoder": dict(p_drop=0.1, cfg={}),
    "gated_encoder_warp": dict(p_drop=0.1, cfg={"beh_warp": 3}),
    "neural_encoder_only": dict(p_drop=1.0, cfg={}),
    "neural_encoder_only_warp": dict(p_drop=1.0, cfg={"beh_warp": 3}),
}


def score(d, i, v, steps, seed):
    y_all, b_all, x_all = d["y"], d["b"][i], d["x"]
    T = len(y_all)
    rng = np.random.default_rng(100 + seed)
    perm = rng.permutation(T)
    te = np.sort(np.array_split(perm, 5)[0])
    tr = np.setdiff1d(perm, te)
    params, cfg, stats, feat, _ = ae.fit(y_all[tr], b_all[tr], np.zeros(len(tr), int), 1, steps=steps,
                                         seed=seed, verbose=False, p_drop=v["p_drop"], cfg_kwargs=v["cfg"])
    yt, bt = jnp.asarray(y_all[te]), jnp.asarray(b_all[te])
    zt = jnp.zeros(len(te), int)
    feat_te, _ = ae.features(yt, stats)
    j, mb = m.heldout_loglik(params, cfg, yt, bt, zt)
    mu0, _, _ = ae.posterior(params, cfg, feat_te, bt, zt, use_behavior=False)
    bhat0 = np.asarray(m.beh_mean_at(params, cfg, mu0))
    target = d["b_mean"][te]
    out = {"kappa": float(m.kappas(params, cfg)[0]), "heldout": float(jnp.mean(j - mb)),
           "behavior_decode_err": float(M.circ_err(bhat0, target).mean()),
           "raw_behavior_err": float(M.circ_err(b_all[te], target).mean())}
    mu1, _, _ = ae.posterior(params, cfg, feat_te, bt, zt, use_behavior=True)
    out["behavior_decode_err_gate_on"] = float(M.circ_err(np.asarray(m.beh_mean_at(params, cfg, mu1)), target).mean())
    return out


def noisy(variant, steps, seed=0):
    v = VARIANTS[variant]
    d = ae.simulate(seed=1, kappas=(0.1,), T_per=3000)
    params, cfg, stats, feat, _ = ae.fit(d["y"], d["b"], d["cond"], 1, steps=steps, seed=seed, verbose=False,
                                         p_drop=v["p_drop"], cfg_kwargs=v["cfg"])
    mu, _, _ = ae.posterior(params, cfg, feat, d["b"], d["cond"], use_behavior=False)
    aligned, _ = M.align(np.asarray(mu), d["x"])
    yg, _ = ae.generate(params, cfg, jax.random.PRNGKey(7), 3000)
    yg = np.asarray(yg, dtype=float)
    return {"kappa": float(m.kappas(params, cfg)[0]), "latent_err_aligned": float(M.circ_err(aligned, d["x"]).mean()),
            "mean_rate_r": float(np.corrcoef(yg.mean(0), d["y"].mean(0))[0, 1]),
            "covariance_r": M.offdiag_corr(np.cov(yg.T), np.cov(d["y"].T))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--only", nargs="+", default=None, help="run only these variants and merge into the saved results")
    a = ap.parse_args()
    variants = {k: v for k, v in VARIANTS.items() if a.only is None or k in a.only}
    out = json.load(open("results/neural_encoder_variant.json")) if a.only else {"strong": {}, "weak": {}, "biased": {}, "noisy": {}}
    bias = lambda x: x + 0.4 + 0.5 * np.sin(x)
    for seed in a.seeds:
        sets = [("strong", ae.simulate_shared(seed=seed, N=30, kappas=(8.0, 3.0, 0.5), tuning="gp")),
                ("weak", ae.simulate_shared(seed=seed, N=8, kappas=(8.0, 3.0), tuning="vm", vm_gain=(0.8, 2.5))),
                ("biased", ae.simulate_shared(seed=seed, N=30, kappas=(6.0,), tuning="gp", behavior_map=bias))]
        for grp, d in sets:
            for i, k in enumerate(d["kappas"]):
                for name, v in variants.items():
                    r = score(d, i, v, a.steps, seed)
                    out[grp].setdefault(f"kappa{k:g}", {}).setdefault(name, []).append(r)
                    print(grp, k, seed, name, {x: round(y, 3) for x, y in r.items()}, flush=True)
        json.dump(out, open("results/neural_encoder_variant.json", "w"), indent=2)
    for name in (() if a.only else ("gated_encoder", "neural_encoder_only")):
        out["noisy"][name] = noisy(name, a.steps)
        print("noisy", name, {x: round(y, 3) for x, y in out["noisy"][name].items()}, flush=True)
    json.dump(out, open("results/neural_encoder_variant.json", "w"), indent=2)
