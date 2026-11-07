# amortized-eiv

Amortized error-in-variables (EIV) regression for neural population activity on a ring, with a behavior-gated encoder, a learned prior on the latent, and a time-varying, condition-dependent behavior fidelity kappa.

It follows the setup of Garon, Keeley, and Williams (bioRxiv 2026, doi 10.64898/2026.04.22.720005), where measured behavior b is a noisy observation of a latent x and neural activity is tuned to x. The preprint scans one kappa per epoch by cross-validation and integrates over x with quasi-Monte Carlo. Here kappa is learned and an encoder replaces the integral.

## Model

```
x_t        ~ p_phi(x)                              learned prior on the ring
b_t | x_t  ~ VonMises(x_t, kappa_c(t))             behavior, one kappa per condition
y_nt | x_t ~ Poisson(exp(b_n + Phi(x_t) W_n))      Fourier-basis tuning, smoothness prior on W
q(x_t | y_t, b_t) = VM(arg z_t, |z_t|),   z_t = z_psi(y_t) + kappa_gate_t * exp(i b_t)
```

- The encoder always sees neural activity. Behavior enters through a gate, so kappa_gate = 0 gives a neural-only encoder and kappa_gate large pins x to b.
- The latent lives on a grid of G points, so the ELBO is an exact sum with no sampling.
- When kappa is small, behavior carries no gradient on x and the fit is a VAE on y with a learned prior. `generate` samples neural activity from it.
- Training randomly zeroes the encoder gate in a fraction of bins (`p_drop`). The generative model is unchanged, so the ELBO stays a valid bound.

## Use

```
pip install -e ".[dev]"
pytest -q                          # about 2 minutes on one CPU
python scripts/smoke_test.py       # full-length run, prints JSON
```

```python
import amortized_eiv as ae
d = ae.simulate(kappas=(8.0, 3.0, 0.5))
params, cfg, stats, feat, hist = ae.fit(d["y"], d["b"], d["cond"], n_cond=3)
mu, rho, q = ae.posterior(params, cfg, feat, d["b"], d["cond"])            # with behavior
mu0, rho0, _ = ae.posterior(params, cfg, feat, d["b"], d["cond"], use_behavior=False)
```

## Tests

`tests/test_model.py` checks the derivation without training: the ELBO is a lower bound on the exact grid log marginal, it is tight at the exact posterior, the joint and conditional-prior forms agree, the uniform-prior normalizer is constant, the gate limits behave, and the von Mises product adds complex parameters.

`tests/test_smoke.py` trains on simulated data with loose thresholds. `tests/test_examples.py` checks the shared-latent simulator and that conventional tuning flattens with noise.

## Smoke-test results

Simulated ring data, 3000 steps, seed 0. Full output is in `results/smoke_results.json`. Errors are mean absolute circular error of the posterior mean against the true latent, in radians.

Strong neural information (N = 30), conditions with kappa = 8, 3, 0.5:

| Condition | kappa true | kappa learned | Latent error | Raw behavior error |
| --- | --- | --- | --- | --- |
| 0 | 8.0 | 7.40 | 0.12 | 0.29 |
| 1 | 3.0 | 2.75 | 0.14 | 0.53 |
| 2 | 0.5 | 0.43 | 0.12 | 1.30 |

Weak neural information (N = 8, low rates), same conditions. The gate helps where behavior is informative and is irrelevant where it is not:

| Condition | kappa true | kappa learned | Gate on | Gate off | Raw behavior |
| --- | --- | --- | --- | --- | --- |
| 0 | 8.0 | 3.72 | 0.27 | 0.52 | 0.29 |
| 1 | 3.0 | 2.36 | 0.39 | 0.58 | 0.49 |
| 2 | 0.5 | 0.49 | 0.61 | 0.61 | 1.28 |

One noisy condition (kappa = 0.1, T = 3000): learned kappa is 0.108, the aligned latent error is 0.34 against 1.55 for raw behavior, and samples from the fitted model match the data in per-neuron mean rate (r = 0.99) and in neural covariance (r = 0.997).

## Examples from the preprint

Both examples use simulated data, with spikes and latent identical across settings and only the behavior trace changing, as in the preprint. The ADn and grid-cell recordings are not reproduced.

```
python scripts/fig1_conceptual.py        # about 3 minutes, writes results/fig1_conceptual.png
python scripts/fig2_validation.py        # about 12 minutes, writes results/fig2_validation.png
python scripts/fig2_validation.py --plot-only
```

Fig 1 (aligned, noisy, uncorrelated behavior; N = 30, T = 1000, von Mises tuning). Conventional tuning curves flatten as behavior noise grows, while tuning to the latent stays sharp and the population manifold is recovered. Latent errors are aligned for rotation and reflection.

| True kappa | Learned kappa | Aligned latent error (rad) | Raw behavior error (rad) |
| --- | --- | --- | --- |
| 100 | 80.6 | 0.06 | 0.08 |
| 2 | 2.04 | 0.11 | 0.66 |
| 0 (uniform) | 0.11 | 0.39 | 1.57 |

Fig 2 (kappa = 8, 3, 0.5; N = 30, T = 1000, tuning drawn from a periodic GP). Held-out log p(y | b) is computed on random held-out bins with kappa fixed on a 9-point grid, as in the preprint's protocol. The learned kappa comes from a separate fit in which kappa is free.

| True kappa | Learned kappa | Best grid kappa | Latent error (rad) | Raw behavior error (rad) | Tuning r, EIV | Tuning r, conventional | Tuning peak-to-trough, true / EIV / conventional |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 8.53 | 8 | 0.054 | 0.28 | 0.996 | 0.961 | 14.1 / 14.2 / 9.7 |
| 3 | 2.70 | 3 | 0.129 | 0.53 | 0.915 | 0.878 | 14.1 / 14.2 / 6.8 |
| 0.5 | 0.43 | 0.5 | 0.088 | 1.31 | 0.988 | 0.659 | 14.1 / 14.3 / 2.3 |

Differences from the preprint's protocol: the tuning prior is a periodic squared-exponential GP with length scale 0.12 of the period, not the preprint's exact kernel. Held-out scoring uses 2 folds and 1200 training steps per grid point, not 10 folds. There is one seed. For kappa = 0.5 the held-out curve is flat below about 1, so the grid argmax of 0.5 is within noise of its neighbors, as in the preprint's bottom panel.

## Speed

Measured on one CPU thread with T_train = 800, N = 30, and kappa = 3 (`scripts/benchmark_speed.py`, `scripts/convergence_diagnostics.py`, `scripts/lbfgs_baseline.py`). The comparison baselines are stand-ins written for these tests, not the preprint's code: the same decoder with no encoder, with the latent integrated by quasi-Monte Carlo points from a Roberts sequence (`fit_qmc`) or by an exact sum on a grid (`fit_grid`). Wall times include JIT compilation of about 2 to 3 seconds unless noted.

Cost per optimizer step (one run; the 1024-point QMC step measured 3.8 ms in an earlier run, so expect run-to-run variation of this size on a shared CPU):

| Method | ms per step |
| --- | --- |
| Amortized encoder, grid G = 128 | 3.9 |
| Exact grid marginal, no encoder, G = 128 | 1.9 |
| Quasi-Monte Carlo, M = 1024 | 6.6 |
| Quasi-Monte Carlo, M = 128 | 1.2 |

Learning kappa in one fit (3000 steps, true kappa = 3):

| Method | Wall time (s) | Learned kappa | Held-out |
| --- | --- | --- | --- |
| Amortized, learned prior | 17.1 | 3.05 | -57.09 |
| Exact grid, no encoder, learned prior | 8.5 | 2.75 | -57.24 |
| Quasi-Monte Carlo, M = 1024 | 22.1 | 2.79 | -57.33 |
| Quasi-Monte Carlo, M = 128 | 5.8 | 2.74 | -57.29 |

The preprint's protocol is a 9-point kappa grid with 2 folds, which is 18 fits. That is about 112 s for the 1024-point baseline, extrapolated from one fit.

Held-out log p(y | b) per bin against training time at fixed kappa, one random fold, Adam for all rows:

| Method | Steps | Wall time (s) | Train | Held-out |
| --- | --- | --- | --- | --- |
| Amortized | 300 | 3.1 | -56.44 | -57.57 |
| Amortized | 600 | 4.3 | -56.08 | -57.29 |
| Amortized | 1200 | 6.6 | -55.71 | -57.05 |
| Quasi-Monte Carlo, M = 1024 | 1200 | 7.6 | -56.25 | -57.68 |
| Quasi-Monte Carlo, M = 1024 | 4000 | 19.4 | -55.64 | -57.28 |
| Quasi-Monte Carlo, M = 1024 | 10000 | 52.1 | -55.64 | -57.28 |
| Exact grid marginal, optimized directly | 4000 | | -55.65 | -57.28 |

With L-BFGS on the deterministic 1024-point marginal likelihood, the baseline reaches held-out -57.29 in 320 iterations. That is about 1.3 s of compute at 4.1 ms per iteration, plus about 4 s of compilation and warm-up.

What this shows:

- The amortized model is not faster in this setting. Per step it is cheaper than 1024-point QMC only because it uses 128 grid points. In one dimension the grid marginal is exact, so the encoder roughly doubles the cost per step and is not needed for the likelihood.
- Learning kappa jointly is about 6 times cheaper than the grid-with-folds protocol, but the baselines can do the same with a learned kappa (5.8 to 22 s). The saving does not come from amortization. The amortized fit is the slowest of the four, and its kappa estimate (3.05) and held-out score are the best.
- The convergence advantage over the Adam baseline does not survive a better optimizer. Amortized Adam reaches held-out -57.29 in 600 steps and 4.3 s, which is about what L-BFGS on the baseline needs.
- The baseline's plateau at -57.28 is not a sampling artifact: the exact grid marginal optimized directly gives the same train and held-out scores. The amortized ELBO fit has a slightly worse train score (-55.71) and a better held-out score (-57.05), so it generalizes better on this fold. The mechanism is not tested. The ELBO may act as an implicit regularizer. The learned prior is not the cause (held-out -57.05 with it, -57.03 without).

## Known limitations

- kappa is biased low when neural information is weak. With N = 8 the cleanest condition gives 3.7 against a true 8. A 9000-step run stays near 3.8, so this is not slow convergence. The cause is not isolated. A broad posterior pulls the kappa update down, which fits the mechanism described above, but this is a hypothesis.
- With strong neural information the cleanest condition is 7.4 at 3000 steps and still rising slowly.
- In the kappa = 3 reproduction, 5% of bins (latent between about -0.2 and 1.05 rad, a sparsely visited stretch) are mislocated by more than 0.5 rad. The latent is warped over that stretch. This is plausibly a local optimum and was not investigated.
- In the uncorrelated case (kappa = 0) the latent is identified only up to rotation and reflection, and a residual warp remains after alignment (0.39 rad).
- Simulated data only. Simulated tuning curves are von Mises bumps, not draws from the model's own prior.
- Not compared against the authors' code, which is not public as far as I could find. The preprint refers to "the software package" without a link. The speed comparison uses stand-in baselines. An L-BFGS baseline was tried and the preprint's own implementation was not.
- The speed results are from one dataset, one fold, and a one-dimensional latent. Amortization is expected to matter for higher-dimensional latents, and that is untested.
- One kappa per condition. There is no within-condition drift, no dynamical prior, and no multi-region coupling. The latent is one-dimensional (a ring grid).
- Grid discretization (G = 128) is a fixed approximation.
