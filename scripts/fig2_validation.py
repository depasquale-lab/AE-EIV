"""Reproduce the preprint's Fig 2 on simulated data (N = 30, T = 1000,
kappa = 8, 3, 0.5, identical spikes and latent across settings).

Saves results/fig2_data.npz. Plot with --plot-only after a run.
"""
import argparse
import os
import pickle

import numpy as np

import amortized_eiv as ae
from amortized_eiv import experiments as E

KAPPAS = (8.0, 3.0, 0.5)
GRID = np.array([0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 15.0, 50.0])
OUT = "results/fig2_data.pkl"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--cv-steps", type=int, default=1200)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    if not a.plot_only:
        d = ae.simulate_shared(seed=a.seed, kappas=KAPPAS, tuning="gp")
        res = {"x": d["x"], "y": d["y"], "b": d["b"], "ring": d["ring"],
               "true": d["rate"](d["ring"]), "kappas": KAPPAS, "grid": GRID, "free": [], "cv": []}
        for i, k in enumerate(KAPPAS):
            f = E.fit_free(d, i, steps=a.steps, seed=a.seed)
            print(f"free fit, true kappa {k}: learned {f['kappa']:.2f}", flush=True)
            res["free"].append(f)
        for i, k in enumerate(KAPPAS):
            cv = E.kappa_grid_cv(d, i, GRID, folds=a.folds, steps=a.cv_steps, seed=a.seed)
            res["cv"].append(cv)
            print(f"cv, true kappa {k}: best grid kappa {GRID[cv.mean(1).argmax()]}", flush=True)
        pickle.dump(res, open(OUT, "wb"))
    from amortized_eiv.figures import plot_fig2
    plot_fig2(pickle.load(open(OUT, "rb")), "results/fig2_validation.png")
