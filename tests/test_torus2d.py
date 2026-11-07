"""Cheap checks of the 2D torus module. No long training."""
import jax
import jax.numpy as jnp
import numpy as np

from amortized_eiv import torus2d as T2


def _setup():
    d = T2.simulate_torus(seed=0, N=20, T=200, kappa=4.0)
    F = jnp.asarray(d["F"])
    y, b = jnp.asarray(d["y"]), jnp.asarray(d["b"])
    p = T2.init(jax.random.PRNGKey(0), 20, y, F, hidden=32, amortized=True)
    p = T2.supervised_init(p, F, y, b, steps=200)
    return d, F, y, b, p


def test_grid_and_qmc_integrators_agree():
    d, F, y, b, p = _setup()
    lg = float(T2.loss_points(p, F, y, b, T2.grid_points(48), 200))
    lq = float(T2.loss_points(p, F, y, b, T2.qmc_points(jax.random.PRNGKey(1), 8192), 200))
    assert abs(lg - lq) < 0.1


def test_amortized_loss_and_gradients_are_finite():
    d, F, y, b, p = _setup()
    feat = jnp.log1p(y)
    feat = (feat - feat.mean(0)) / (feat.std(0) + 1e-3)
    loss, g = jax.value_and_grad(lambda q: T2.loss_amortized(q, F, jax.random.PRNGKey(2), feat, y, b, 200, 2))(p)
    assert np.isfinite(float(loss))
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree_util.tree_leaves(g))


def test_supervised_init_beats_random_decoder():
    d = T2.simulate_torus(seed=0, N=20, T=400, kappa=1e3)   # behavior is almost exact
    F = jnp.asarray(d["F"])
    y, b = jnp.asarray(d["y"]), jnp.asarray(d["b"])
    p0 = T2.init(jax.random.PRNGKey(0), 20, y, F)
    p1 = T2.supervised_init(p0, F, y, b, steps=300)

    def nll(q):
        logF = q["b0"][None, :] + T2.basis(b, F) @ q["W"]
        return float(-(y * logF - jnp.exp(logF)).sum(-1).mean())
    assert nll(p1) < nll(p0)
