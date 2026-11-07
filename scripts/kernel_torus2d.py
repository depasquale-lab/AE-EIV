"""2D torus benchmark with the Matern-type prior replaced by the ReLU NNGP prior (depth 1, 2).

Same data, folds, budgets, and scoring as scripts/benchmark_torus2d.py. The Matern-type
results are read from results/torus2d_results.json. Writes results/kernel_torus2d.json.
"""
import json

import jax.numpy as jnp

from amortized_eiv import torus2d as T2

METHODS = {
    "amortized_S4_h256": ("amortized", {"S": 4, "hidden": 256, "lr": 3e-3}, 6000),
    "qmc_M1024": ("qmc", {"M": 1024}, 3000),
    "grid_G32": ("grid", {"G": 32}, 3000),
}
KERNELS = {"nngp_d1": ("nngp", 1), "nngp_d2": ("nngp", 2)}

if __name__ == "__main__":
    out = {}
    for seed in (0, 1):
        d = T2.simulate_torus(seed=seed, N=60, T=2000, kappa=4.0)
        F, n = jnp.asarray(d["F"]), 1600
        for kname, (kern, depth) in KERNELS.items():
            for mname, (method, kw, steps) in METHODS.items():
                r = T2.train(method, d["y"][:n], d["b"][:n], d["y"][n:], d["b"][n:], d["x"][n:], F,
                             steps=steps, checkpoints=[steps], seed=seed, verbose=False,
                             kernel=kern, depth=depth, **kw)
                h = r["history"][-1]
                out.setdefault(str(seed), {}).setdefault(kname, {})[mname] = {**h, "ms_per_step": r["ms_per_step"]}
                print(f"seed {seed} {kname} {mname:18s} held-out {h['heldout']:.2f} kappa {h['kappa']:.2f} "
                      f"latent err {h['latent_err']:.3f}", flush=True)
        json.dump(out, open("results/kernel_torus2d.json", "w"), indent=2)
