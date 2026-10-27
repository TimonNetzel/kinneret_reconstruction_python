"""Bausteine des C++-Kerns gegen unabhaengige Python-Implementierungen."""

from pathlib import Path

import numpy as np
import pytest
from scipy.stats import gamma, gaussian_kde, norm

from kinneret_recon import _core
from kinneret_recon.config import SamplerConfig
from kinneret_recon.data import load_inputs
from kinneret_recon.model import _nnet_model, build_setup, normalize_pann, normalize_temp, transfer_function_grid

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def inputs():
    return load_inputs(ROOT / "data/in")


@pytest.fixture(scope="module")
def setup(inputs):
    return build_setup(inputs, SamplerConfig())


def test_norm_ppf_and_gamma_p():
    p = np.array([1e-12, 1e-4, 0.025, 0.3, 0.5, 0.9, 1 - 1e-9])
    np.testing.assert_allclose([_core.norm_ppf(x) for x in p], norm.ppf(p), rtol=1e-14)
    for s, z in [(2.15, 0.3), (2.15, 4.0), (0.5, 30.0), (10.0, 9.0)]:
        assert _core.gamma_p(s, z) == pytest.approx(gamma.cdf(z, s), rel=1e-12)


def test_nnet_matches_legacy_and_numpy(inputs):
    rng = np.random.default_rng(0)
    t, p = rng.normal(size=50), rng.normal(size=50)
    new = _core.nnet_probs(_nnet_model(inputs), t, p)
    old = _core.legacy_nnet_probs(t, p, inputs.wts_in_hidden, inputs.wts_hidden_out,
                                  inputs.wts_bias_hidden, inputs.wts_bias_out)
    np.testing.assert_allclose(new, old, rtol=1e-12)
    # unabhaengige NumPy-Implementierung des Netzes (6 Hidden, Sigmoid, Softmax)
    nh = len(inputs.wts_bias_hidden)
    h = 1 / (1 + np.exp(-(np.outer(t, inputs.wts_in_hidden[:nh]) + np.outer(p, inputs.wts_in_hidden[nh:])
                          + inputs.wts_bias_hidden)))
    o = h @ inputs.wts_hidden_out.reshape(-1, nh).T + inputs.wts_bias_out
    sm = np.exp(o - o.max(1, keepdims=True))
    np.testing.assert_allclose(new, sm / sm.sum(1, keepdims=True), rtol=1e-12)


def test_grid_orientation(inputs):
    """Kommentar '# wrong' in load_data.py: das Gitter ist korrekt orientiert ([P, T])."""
    g = transfer_function_grid(inputs)
    i, j = 123, 321  # i: P-Index, j: T-Index
    t = normalize_temp(inputs.temp_range[j], inputs.normal_params)
    p = normalize_pann(inputs.pann_range[i], inputs.normal_params)
    direct = _core.legacy_nnet_probs(np.array([t]), np.array([p]), inputs.wts_in_hidden, inputs.wts_hidden_out,
                                     inputs.wts_bias_hidden, inputs.wts_bias_out)[0]
    np.testing.assert_allclose(g[:, i, j], direct, rtol=1e-12)


def test_biome_ratios(setup):
    rng = np.random.default_rng(1)
    w = rng.dirichlet(np.ones(setup.num_taxa))
    b = _core.biome_ratios(setup.core_model(), w)
    ref = (setup.taxa_spectrum_age * w) @ setup.inputs.biomes_assign
    np.testing.assert_allclose(b, ref / ref.sum(1, keepdims=True), rtol=1e-13)


def test_evaluate_state_log_target(setup):
    """Log-Zielverteilung gegen eine Python-Implementierung des Modells."""
    from scipy import stats

    rng = np.random.default_rng(2)
    w = rng.dirichlet(np.full(setup.num_taxa, 5.0))
    c = rng.uniform(setup.tf_lower, setup.tf_upper)
    m = setup.core_model()
    r = _core.evaluate_state(m, {}, w, c)
    B = setup.num_biomes
    ratios = (setup.taxa_spectrum_age * w) @ setup.inputs.biomes_assign
    ratios /= ratios.sum(1, keepdims=True)
    rt, rp = ratios @ c[:B], ratios @ c[B:]
    corr = np.corrcoef(rp, setup.ap_age)[0, 1]
    r2 = corr**2 if corr > 0 else 0.0
    inp = setup.inputs
    lt = (stats.beta.logpdf(r2, 3, 3) + stats.norm.logpdf(rt[0], *inp.recent_climate[:2])
          + stats.gamma.logpdf(rp[0], setup.pann_shape, scale=1 / setup.pann_rate)
          + stats.dirichlet.logpdf(w, setup.jeffreys))
    tn = normalize_temp(c[:B], inp.normal_params)
    pn = normalize_pann(c[B:], inp.normal_params)
    probs = _core.legacy_nnet_probs(tn, pn, inp.wts_in_hidden, inp.wts_hidden_out,
                                    inp.wts_bias_hidden, inp.wts_bias_out)
    lt += np.sum(np.log(probs[np.arange(B), np.arange(B)]))
    assert r["expl_variance"] == pytest.approx(r2, rel=1e-10)
    assert r["recent_temp"] == pytest.approx(rt[0], rel=1e-12)
    assert r["log_target"] == pytest.approx(lt, rel=1e-9)


def test_kde_matches_scipy():
    rng = np.random.default_rng(3)
    x = rng.normal(5, 0.4, (4, 3000))
    g = np.linspace(-10, 20, 400)
    a = _core.kde1d_rows(x, g[0], g[1] - g[0], len(g))
    b = np.array([gaussian_kde(r).evaluate(g) for r in x])
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12 * b.max())

    u = rng.normal(700, 150, 3000)
    v = rng.normal(10, 2, 3000) + 0.004 * (u - 700)
    ga, gb = np.linspace(0, 2000, 120), np.linspace(-5, 25, 100)
    c = _core.kde2d_grid(u, v, ga[0], ga[1] - ga[0], len(ga), gb[0], gb[1] - gb[0], len(gb))
    gx, gy = np.meshgrid(ga, gb, indexing="ij")
    d = gaussian_kde(np.vstack([u, v]))(np.vstack([gx.ravel(), gy.ravel()])).reshape(len(ga), len(gb))
    np.testing.assert_allclose(c, d, rtol=0, atol=1e-12 * d.max())
