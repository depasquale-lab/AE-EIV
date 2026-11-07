"""Checks of the identities in the derivation. No training."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from amortized_eiv import Config, features, init_params, simulate
from amortized_eiv import model as m


@pytest.fixture(scope="module")
def setup():
    d = simulate(seed=1, N=12, T_per=60, kappas=(6.0, 1.0))
    y = jnp.asarray(d["y"]); b = jnp.asarray(d["b"]); cond = jnp.asarray(d["cond"])
    cfg = Config(N=12, n_cond=2, G=64)
    params = init_params(jax.random.PRNGKey(0), cfg, y)
    # make the model non-trivial so the checks are not vacuous
    params["W"] = 0.5 * jax.random.normal(jax.random.PRNGKey(2), params["W"].shape) * m.tau(cfg)[:, None]
    params["prior"] = jnp.array([0.4, -0.2, 0.1, 0.0, 0.3, -0.1])
    params["log_kappa"] = jnp.log(jnp.array([5.0, 0.7]))
    feat, _ = features(y)
    return cfg, params, feat, y, b, cond


def test_elbo_is_a_lower_bound(setup):
    cfg, params, feat, y, b, cond = setup
    elbo, log_marg = m.elbo_per_bin(params, cfg, feat, y, b, cond)
    assert bool(jnp.all(elbo <= log_marg + 1e-4))


def test_elbo_tight_at_exact_posterior(setup):
    cfg, params, feat, y, b, cond = setup
    kap = m.kappas(params, cfg)[cond]
    joint = (m.log_prior_grid(params, cfg)[None, :] + m.log_lik_grid(params, cfg, y)
             + m.log_beh_grid(cfg, b, kap))
    logq = jax.nn.log_softmax(joint, axis=-1)
    elbo = (jnp.exp(logq) * (joint - logq)).sum(-1)
    np.testing.assert_allclose(elbo, jax.scipy.special.logsumexp(joint, -1), atol=1e-4)


def test_conditional_prior_form_matches(setup):
    """E_q log p(y|x) - KL(q || p(x|b)) + log Z(b) equals the joint-form ELBO."""
    cfg, params, feat, y, b, cond = setup
    kap = m.kappas(params, cfg)[cond]
    z = m.encoder_z(params, feat, b, kap)
    logq = m.log_q_grid(z, cfg); q = jnp.exp(logq)
    lp = m.log_prior_grid(params, cfg)[None, :]
    lb = m.log_beh_grid(cfg, b, kap)
    ll = m.log_lik_grid(params, cfg, y)
    logZ = jax.scipy.special.logsumexp(lp + lb, axis=-1)
    log_pxb = lp + lb - logZ[:, None]
    form2 = (q * ll).sum(-1) - (q * (logq - log_pxb)).sum(-1) + logZ
    form1 = (q * (ll + lb + lp - logq)).sum(-1)
    np.testing.assert_allclose(form1, form2, atol=1e-4)


def test_uniform_prior_gives_constant_Z(setup):
    cfg, params, feat, y, b, cond = setup
    cfg_u = Config(N=cfg.N, n_cond=cfg.n_cond, G=cfg.G, learn_prior=False)
    kap = m.kappas(params, cfg_u)[cond]
    lp = m.log_prior_grid(params, cfg_u)[None, :]
    logZ = jax.scipy.special.logsumexp(lp + m.log_beh_grid(cfg_u, b, kap), axis=-1)
    # discrete-grid analogue of 1/(2 pi): constant in b
    assert float(logZ.max() - logZ.min()) < 1e-2


def test_gate_zero_is_neural_only(setup):
    cfg, params, feat, y, b, cond = setup
    z0 = m.encoder_z(params, feat, b, jnp.zeros(y.shape[0]))
    np.testing.assert_allclose(z0, m._mlp(params["enc"], feat), atol=1e-6)


def test_gate_large_pins_latent_to_behavior(setup):
    cfg, params, feat, y, b, cond = setup
    z = m.encoder_z(params, feat, b, jnp.full(y.shape[0], 1e4))
    mu = jnp.angle(z[:, 0] + 1j * z[:, 1])
    np.testing.assert_allclose(jnp.cos(mu - b), 1.0, atol=1e-4)


def test_vm_product_is_vm_with_summed_parameter():
    g = jnp.linspace(-jnp.pi, jnp.pi, 256, endpoint=False)
    z1 = jnp.array([1.3, -0.4]); z2 = 2.0 * jnp.array([jnp.cos(0.7), jnp.sin(0.7)])
    f = lambda z: z[0] * jnp.cos(g) + z[1] * jnp.sin(g)
    prod = jax.nn.log_softmax(f(z1) + f(z2))
    fused = jax.nn.log_softmax(f(z1 + z2))
    np.testing.assert_allclose(prod, fused, atol=1e-6)
