import numpy as np
import pytest

from kinneret_recon.diagnostics import ess_bulk, rhat


def ar1(phi, m, n, rng, shift=None):
    x = np.zeros((m, n))
    e = rng.normal(size=(m, n))
    for t in range(1, n):
        x[:, t] = phi * x[:, t - 1] + e[:, t]
    if shift is not None:
        x += np.asarray(shift)[:, None]
    return x


def test_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 2000))
    assert rhat(x) == pytest.approx(1.0, abs=0.01)
    assert ess_bulk(x) == pytest.approx(8000, rel=0.1)


def test_ar1_ess():
    rng = np.random.default_rng(1)
    phi = 0.8
    x = ar1(phi, 4, 20000, rng)
    assert ess_bulk(x) == pytest.approx(80000 * (1 - phi) / (1 + phi), rel=0.15)


def test_rhat_detects_shifted_chain():
    rng = np.random.default_rng(2)
    x = ar1(0.5, 4, 2000, rng, shift=[0, 0, 0, 3])
    assert rhat(x) > 1.1
