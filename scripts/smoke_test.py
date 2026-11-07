"""End-to-end smoke test on simulated ring data. Prints JSON.

strong : N = 30, high rates. kappa_c recovered, latent beats raw behavior.
weak   : N = 8, low rates. Behavior gate on beats neural-only decoding.
noisy  : one condition, kappa = 0.1. The fit is a generative model of neural
         activity: small learned kappa, latent up to rotation or reflection, and
         samples that reproduce mean rates and neural covariance.
"""
import argparse
import json

from amortized_eiv.experiments import run_ab, run_noisy

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    res = {
        "strong": run_ab(steps=a.steps, seed=a.seed),
        "weak": run_ab(steps=a.steps, seed=a.seed + 3, N=8, gain=(0.8, 2.5)),
        "noisy": run_noisy(steps=a.steps, seed=a.seed),
    }
    print(json.dumps(res, indent=2))
