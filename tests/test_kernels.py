"""Checks of the tuning-prior kernels. No training."""
import numpy as np

from amortized_eiv import Config, kernels as K
from amortized_eiv import model as m


def test_depth1_spectrum_reproduces_the_arccosine_kernel():
    a0, a = K.ring_coefs(4000, depth=1)
    for d in (0.0, 0.5, 1.3, 2.4, np.pi):
        k = a0 + 2 * (a * np.cos(np.arange(1, 4001) * d)).sum()
        closed = (np.sin(d) + (np.pi - d) * np.cos(d)) / np.pi
        assert abs(k - closed) < 1e-4


def test_variance_matches_the_matern_default():
    j = np.arange(1, 13)
    target = (K.matern_tau(j ** 2) ** 2).sum()
    for depth in (1, 2, 3):
        t = K.ring_tau(12, "nngp", depth)
        assert abs((t ** 2).sum() - target) / target < 1e-3


def test_odd_harmonics_vanish_at_depth_1_but_not_deeper():
    t1 = K.ring_tau(12, "nngp", 1)
    t2 = K.ring_tau(12, "nngp", 2)
    odd = np.array([3, 5, 7, 9, 11]) - 1
    assert np.all(t1[odd] < 0.01 * t1.max())
    assert np.all(t2[odd] > 0.01 * t2.max())


def test_torus_spectrum_is_positive_and_variance_matched():
    from amortized_eiv.torus2d import freqs
    F = freqs(3)
    target = (K.matern_tau((F ** 2).sum(1)) ** 2).sum()
    for depth in (1, 2):
        t = K.torus_tau(F, "nngp", depth)
        assert t.shape == (len(F),) and np.all(t > 0)
        assert abs((t ** 2).sum() - target) / target < 1e-3


def test_model_tau_uses_the_kernel_option():
    base = np.asarray(m.tau(Config(N=3)))
    nn = np.asarray(m.tau(Config(N=3, kernel="nngp", depth=2)))
    assert base.shape == nn.shape == (24,)
    assert np.all(np.isfinite(nn)) and not np.allclose(base, nn)
