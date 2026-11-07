"""Training smoke tests. Thresholds are loose: they catch breakage, not small drift."""
import pytest

from amortized_eiv.experiments import run_ab, run_noisy

STEPS = 2000


@pytest.fixture(scope="module")
def strong():
    return run_ab(steps=STEPS, seed=0, T_per=800)


def test_kappa_ordering_and_scale(strong):
    k = strong["kappa_learned"]
    assert k[0] > k[1] > k[2]
    for learned, true in zip(k, strong["kappa_true"]):
        assert 0.5 < learned / true < 2.0


def test_latent_beats_behavior_when_behavior_is_noisy(strong):
    assert strong["latent_mae_gate_on"]["cond2"] < 0.5 * strong["behavior_mae"]["cond2"]


def test_gate_helps_when_neural_information_is_weak():
    r = run_ab(steps=STEPS, seed=3, T_per=800, N=8, gain=(0.8, 2.5))
    k = r["kappa_learned"]
    assert k[0] > k[2] and k[1] > k[2]
    # behavior-informative conditions gain from the gate
    assert r["latent_mae_gate_on"]["cond0"] < r["latent_mae_gate_off"]["cond0"]
    assert r["latent_mae_gate_on"]["cond1"] < r["latent_mae_gate_off"]["cond1"]


def test_noisy_behavior_gives_generative_model_of_neural_activity():
    r = run_noisy(steps=STEPS, seed=0, T=2000)
    assert r["kappa_learned"] < 0.5
    assert r["latent_mae_aligned"] < 0.5 * r["behavior_mae"]
    assert r["mean_rate_corr"] > 0.9
    assert r["covariance_corr"] > 0.9
