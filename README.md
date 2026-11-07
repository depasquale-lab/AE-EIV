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

`tests/test_smoke.py` trains on simulated data with loose thresholds.

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

## Known limitations

- kappa is biased low when neural information is weak. With N = 8 the cleanest condition gives 3.7 against a true 8. A 9000-step run stays near 3.8, so this is not slow convergence. The cause is not isolated. A broad posterior pulls the kappa update down, which fits the mechanism described above, but this is a hypothesis.
- With strong neural information the cleanest condition is 7.4 at 3000 steps and still rising slowly.
- Simulated data only. Simulated tuning curves are von Mises bumps, not draws from the model's own prior.
- Not compared against the authors' sampler or code.
- One kappa per condition. There is no within-condition drift, no dynamical prior, and no multi-region coupling. The latent is one-dimensional (a ring grid).
- Grid discretization (G = 128) is a fixed approximation.
