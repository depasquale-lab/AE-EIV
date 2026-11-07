"""Cheap checks on the paper-style simulation helpers. No training."""
import numpy as np

import amortized_eiv as ae
from amortized_eiv import metrics as M


def test_shared_latent_and_spikes_across_settings():
    d = ae.simulate_shared(seed=0, kappas=(100.0, 2.0, 0.0), tuning="vm")
    assert len(d["b"]) == 3 and d["y"].shape == (1000, 30)
    err = [M.circ_err(b, d["x"]).mean() for b in d["b"]]
    assert err[0] < 0.15 < err[1] < err[2]
    assert abs(err[2] - np.pi / 2) < 0.15          # uniform behavior: E|error| = pi/2


def test_conventional_tuning_flattens_with_noise():
    d = ae.simulate_shared(seed=0, kappas=(100.0, 2.0, 0.0), tuning="gp")
    amp = [np.mean(np.ptp(M.kernel_tuning(d["y"], b, d["ring"]), axis=0)) for b in d["b"]]
    assert amp[0] > amp[1] > amp[2]
    assert amp[2] < 0.3 * amp[0]
