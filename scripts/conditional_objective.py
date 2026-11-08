"""Joint objective (ELBO on log p(y, b)) versus conditional objective (ELBO on log p(y | b)).

The conditional objective drops log Z(b), the normalizer of the conditional prior p(x | b), so the
model conditions on behavior and does not model it. The two are identical for a flat prior on x.
With a learned prior they differ. 'standard' is the Fig 2 setup. 'narrow' keeps the latent in a
small part of the ring (random-walk step 0.05), where a learned prior matters most.
Each fit uses 80% of the bins with kappa learned for 3000 steps. Writes results/conditional_objective.json.
"""
import json

import jax.numpy as jnp
import numpy as np

import amortized_eiv as ae
from amortized_eiv import metrics as M
from amortized_eiv import model as m


def run(d, i, objective, steps, seed):
    y_all, b_all, x_all = d["y"], d["b"][i], d["x"]
    T = len(y_all)
    rng = np.random.default_rng(100 + seed)
    perm = rng.permutation(T)
    te = np.sort(np.array_split(perm, 5)[0])
    tr = np.setdiff1d(perm, te)
    params, cfg, stats, feat, _ = ae.fit(y_all[tr], b_all[tr], np.zeros(len(tr), int), 1, steps=steps,
                                         seed=seed, verbose=False, cfg_kwargs={"objective": objective})
    yt, bt = jnp.asarray(y_all[te]), jnp.asarray(b_all[te])
    zt = jnp.zeros(len(te), int)
    j, mb = m.heldout_loglik(params, cfg, yt, bt, zt)
    feat_te, _ = ae.features(yt, stats)
    mu, _, _ = ae.posterior(params, cfg, feat_te, bt, zt, use_behavior=True)
    lp = np.asarray(m.log_prior_grid(params, cfg))
    return {"kappa": float(m.kappas(params, cfg)[0]), "heldout": float(jnp.mean(j - mb)),
            "latent_err": float(M.circ_err(np.asarray(mu), x_all[te]).mean()),
            "prior_entropy": float(-(np.exp(lp) * lp).sum())}


if __name__ == "__main__":
    out = {}
    for seed in (0, 1):
        sets = [("standard", ae.simulate_shared(seed=seed, kappas=(8.0, 3.0, 0.5), tuning="gp")),
                ("narrow", ae.simulate_shared(seed=seed, kappas=(3.0,), tuning="gp", step_sd=0.05))]
        for grp, d in sets:
            for i, k in enumerate(d["kappas"]):
                for obj in ("joint", "conditional"):
                    r = run(d, i, obj, 3000, seed)
                    out.setdefault(grp, {}).setdefault(f"kappa{k:g}", {}).setdefault(obj, []).append(r)
                    print(grp, k, seed, obj, {x: round(y, 3) for x, y in r.items()}, flush=True)
        json.dump(out, open("results/conditional_objective.json", "w"), indent=2)
