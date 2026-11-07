"""Learned behavior decoder (mean-direction warp). No training."""
import jax
import jax.numpy as jnp
import numpy as np

from amortized_eiv import Config, features, init_params, simulate_shared
from amortized_eiv import model as m


def test_default_behavior_map_is_identity_and_warp_starts_at_identity():
    x = jnp.linspace(-3, 3, 50)
    cfg0 = Config(N=4)
    np.testing.assert_allclose(m.beh_mean_at({}, cfg0, x), x)
    cfg1 = Config(N=4, beh_warp=3)
    p = init_params(jax.random.PRNGKey(0), cfg1, jnp.ones((10, 4)))
    np.testing.assert_allclose(m.beh_mean_at(p, cfg1, x), x, atol=1e-6)


def test_elbo_stays_a_lower_bound_with_a_nonzero_warp():
    d = simulate_shared(seed=1, N=10, T=80, kappas=(3.0,), tuning="gp",
                        behavior_map=lambda x: x + 0.3 + 0.4 * np.sin(x))
    y, b = jnp.asarray(d["y"]), jnp.asarray(d["b"][0])
    cfg = Config(N=10, beh_warp=2, G=64)
    p = init_params(jax.random.PRNGKey(0), cfg, y)
    p["warp"] = jnp.array([0.2, 0.3, -0.1, 0.1, 0.05])
    feat, _ = features(y)
    elbo, log_marg = m.elbo_per_bin(p, cfg, feat, y, b, jnp.zeros(80, int))
    assert bool(jnp.all(elbo <= log_marg + 1e-4))


def test_simulator_reports_the_mean_behavior():
    f = lambda x: x + 0.5
    d = simulate_shared(seed=0, kappas=(50.0,), behavior_map=f)
    err = np.abs(np.angle(np.exp(1j * (d["b"][0] - d["b_mean"]))))
    assert err.mean() < 0.2
    np.testing.assert_allclose(d["b_mean"], f(d["x"]))
