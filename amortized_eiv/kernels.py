"""Priors on the tuning-curve weights, as spectra of stationary kernels on the circle or torus.

The decoder writes the log rate as a Fourier expansion whose weights have a Gaussian prior
with standard deviation tau_j. That is the weight-space form of a Gaussian process whose
kernel has spectral weights tau_j^2.

  matern : tau_j = tau0 * (1 + (j / ell)^2)^(-nu / 2). The default.
  nngp   : the infinitely wide limit of a ReLU network (the neural network Gaussian
           process kernel) of a given depth, on the unit-norm input (cos x, sin x).
           For depth 1 this is the arc-cosine kernel
               k(d) = (1/pi) * (sin d + (pi - d) * cos d).

For a stationary kernel k(d) = sum_j a_j cos(j d), a weight pair (cos, sin) at harmonic j has
variance 2 a_j. NNGP spectra are rescaled so the total prior variance of the log rate matches
the Matern default, so the two priors differ in shape and not in overall scale.
"""
from __future__ import annotations

import numpy as np


def relu_corr(rho, depth=1):
    """Normalized ReLU NNGP correlation after `depth` layers, from input correlation rho."""
    r = np.clip(np.asarray(rho, dtype=np.float64), -1.0, 1.0)
    for _ in range(depth):
        r = (np.sqrt(1.0 - r ** 2) + (np.pi - np.arccos(r)) * r) / np.pi
    return r


def matern_tau(j2, tau0=2.0, ell=3.0, nu=2.0):
    """j2 is |j|^2 (a harmonic squared on the ring, a squared frequency length on the torus)."""
    return tau0 * (1.0 + np.asarray(j2) / ell ** 2) ** (-nu / 2.0)


def ring_coefs(J, depth=1, M=8192):
    d = np.arange(M) * 2 * np.pi / M
    a = np.fft.rfft(relu_corr(np.cos(d), depth)).real / M
    return a[0], a[1:J + 1]


def torus_coefs(F, depth=1, M=256):
    g = np.arange(M) * 2 * np.pi / M
    d1, d2 = np.meshgrid(g, g, indexing="ij")
    k = relu_corr((np.cos(d1) + np.cos(d2)) / 2.0, depth)
    A = np.fft.fft2(k).real / M ** 2
    j1 = np.asarray(F[:, 0]).astype(int) % M
    j2 = np.asarray(F[:, 1]).astype(int) % M
    return A[0, 0], A[j1, j2]


def _scale(a, target_var, floor=1e-3):
    tau2 = np.maximum(2.0 * a, 0.0)
    tau = np.sqrt(target_var / tau2.sum()) * np.sqrt(tau2)
    return np.maximum(tau, floor * tau.max())


def ring_tau(J, kernel="matern", depth=1, tau0=2.0, ell=3.0, nu=2.0):
    """Prior standard deviations for harmonics 1..J (the same value for cos and sin)."""
    j = np.arange(1, J + 1)
    m = matern_tau(j ** 2, tau0, ell, nu)
    if kernel == "matern":
        return m
    _, a = ring_coefs(J, depth)
    return _scale(a, (m ** 2).sum())


def torus_tau(F, kernel="matern", depth=1, tau0=2.0, ell=3.0, nu=2.0):
    r2 = (np.asarray(F) ** 2).sum(1)
    m = matern_tau(r2, tau0, ell, nu)
    if kernel == "matern":
        return m
    _, a = torus_coefs(F, depth)
    return _scale(a, (m ** 2).sum())


def ring_kernel_from_tau(tau, d):
    """k(d) implied by weight standard deviations tau_j: sum_j tau_j^2 cos(j d)."""
    j = np.arange(1, len(tau) + 1)
    return (np.asarray(tau)[None, :] ** 2 * np.cos(np.asarray(d)[:, None] * j[None, :])).sum(1)
