"""Validierung des ueberarbeiteten Samplers an Zielverteilungen mit bekannter Loesung."""

from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from kinneret_recon import sampler
from kinneret_recon.config import SamplerConfig
from kinneret_recon.data import load_inputs
from kinneret_recon.model import build_setup

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def inputs():
    return load_inputs(ROOT / "data/in")


def prior_only(inputs, fix, n=300_000, chains=4, factor=4.0, proposal="dirichlet"):
    cfg = SamplerConfig(sample_length=n, burn_in=n // 4, thin=10, n_chains=chains, seed=11, weights_proposal=proposal,
                        dirichlet_hastings_fix=fix, dirichlet_spread_factor=factor,
                        use_calibration=False, use_tf_likelihood=False)
    setup = build_setup(inputs, cfg)
    return setup, sampler.run(setup, cfg)


@pytest.mark.parametrize("proposal", ["dirichlet", "log_gamma"])
def test_prior_only_recovers_jeffreys_prior(inputs, proposal):
    """Ohne Daten muss die Kette den Prior Dir(0.5, ..., 0.5) und U(Grenzen) wiedergeben."""
    setup, post = prior_only(inputs, fix=True, proposal=proposal)
    w = post.pooled("taxa_weights")
    K = w.shape[1]
    ref = stats.beta(0.5, 0.5 * (K - 1))  # Randverteilung einer Komponente
    assert w.std() == pytest.approx(ref.std(), rel=0.1)
    assert np.median(w) == pytest.approx(ref.median(), rel=0.15)
    assert np.mean(w < 1e-3) == pytest.approx(ref.cdf(1e-3), abs=0.03)
    u = (post.pooled("tf_sample") - setup.tf_lower) / (setup.tf_upper - setup.tf_lower)
    np.testing.assert_allclose(u.mean(0), 0.5, atol=0.08)
    np.testing.assert_allclose(u.std(0), np.sqrt(1 / 12), atol=0.04)


def test_original_hastings_ratio_misses_prior(inputs):
    """Dokumentiert den Fehler des Originals: ohne kappa in der Hastings-Korrektur wird der
    Prior der Taxa-Gewichte nicht getroffen (Streuung deutlich zu klein)."""
    _, post = prior_only(inputs, fix=False, n=200_000, chains=2, factor=1.0)
    w = post.pooled("taxa_weights")
    ref = stats.beta(0.5, 0.5 * (w.shape[1] - 1))
    assert w.std() < 0.75 * ref.std()
    assert np.mean(w < 1e-3) < 0.5 * ref.cdf(1e-3)


def test_chains_reproducible(inputs):
    cfg = SamplerConfig(sample_length=5_000, burn_in=1_000, thin=10, n_chains=2, seed=5)
    assert cfg.weights_proposal == "log_gamma"
    setup = build_setup(inputs, cfg)
    a, b = sampler.run(setup, cfg), sampler.run(setup, cfg)
    np.testing.assert_array_equal(a.tf_sample, b.tf_sample)
    np.testing.assert_array_equal(a.taxa_weights, b.taxa_weights)
    assert not np.array_equal(a.tf_sample[0], a.tf_sample[1])
