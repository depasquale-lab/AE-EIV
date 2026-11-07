import numpy as np


def circ_err(a, b):
    return np.abs(np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b)))))


def align(est, ref):
    """Best rotation and reflection of est onto ref. Returns aligned est and (sign, shift)."""
    best = None
    for s in (1.0, -1.0):
        shift = np.angle(np.mean(np.exp(1j * (ref - s * est))))
        e = circ_err(s * est + shift, ref).mean()
        if best is None or e < best[0]:
            best = (e, s, shift)
    _, s, shift = best
    return np.angle(np.exp(1j * (s * est + shift))), (s, shift)


def offdiag_corr(A, B):
    iu = np.triu_indices(A.shape[0], 1)
    return float(np.corrcoef(A[iu], B[iu])[0, 1])
