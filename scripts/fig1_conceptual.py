"""Reproduce the preprint's Fig 1: identical spikes and latent, behavior that is
aligned, noisy, or uncorrelated. Conventional tuning flattens as noise grows,
while tuning to the latent (truth, and the EIV estimate) stays sharp."""
import argparse
import os
import pickle

import numpy as np

import amortized_eiv as ae
from amortized_eiv import experiments as E

KAPPAS = (100.0, 2.0, 0.0)
OUT = "results/fig1_data.pkl"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    if not a.plot_only:
        d = ae.simulate_shared(seed=a.seed, kappas=KAPPAS, tuning="vm")
        fits = []
        for i, k in enumerate(KAPPAS):
            f = E.fit_free(d, i, steps=a.steps, seed=a.seed, align=True)
            print(f"true kappa {k}: learned {f['kappa']:.2f}, "
                  f"aligned latent MAE {float(ae.metrics.circ_err(f['mu'], d['x']).mean()):.3f} rad, "
                  f"behavior MAE {float(ae.metrics.circ_err(d['b'][i], d['x']).mean()):.3f} rad", flush=True)
            fits.append(f)
        res = {"x": d["x"], "y": d["y"], "b": d["b"], "ring": d["ring"],
               "true": d["rate"](d["ring"]), "kappas": KAPPAS, "fits": fits}
        pickle.dump(res, open(OUT, "wb"))
    from amortized_eiv.figures import plot_fig1
    plot_fig1(pickle.load(open(OUT, "rb")), "results/fig1_conceptual.png")
