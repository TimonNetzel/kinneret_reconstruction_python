"""Only the anonymous counts plan; no legacy sampler, app or real inputs."""
import importlib.util
import itertools
import math
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "anonymous_counts_model", Path(__file__).resolve().parents[1] / "tools/taxon_counts/model.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
CFG = m.config()


def compositions(n, k):
    if k == 1:
        yield (n,)
    else:
        for j in range(n + 1):
            for rest in compositions(n - j, k - 1):
                yield (j,) + rest


@pytest.mark.parametrize("total", [0, 8])
@pytest.mark.parametrize("point", CFG["base"]["states"])
@pytest.mark.parametrize("kind", ["base", "sample_shared_mixture"])
def test_complete_pmf_enumeration(total, point, kind):
    counts = list(compositions(total, 5))
    assert len(counts) == (1 if total == 0 else 495)
    total_probability = sum(np.exp(m.log_likelihood([point], y, total, CFG, model=kind))[0] for y in counts)
    assert abs(total_probability - 1) < 1e-12


def test_sample_mixture_and_matching_single_grain_marginals():
    points = m.grid(CFG)
    shifts = CFG["shared"]["shifts"]
    p0, p1 = [m.probabilities(points, CFG, h) for h in shifts]
    mean = (p0 + p1) / 2
    for j in range(5):
        y = np.eye(5, dtype=int)[j]
        sample = np.exp(m.log_likelihood(points, y, 1, CFG, model="sample_shared_mixture"))
        grain = np.exp(m.log_likelihood(points, y, 1, CFG, model="grain_redrawn_matching_marginals"))
        np.testing.assert_allclose(sample, mean[:, j], atol=1e-12, rtol=0)
        np.testing.assert_allclose(sample, grain, atol=1e-12, rtol=0)
    a = np.exp(m.log_likelihood([[0, 0]], [2, 0, 0, 0, 0], 2, CFG, model="sample_shared_mixture"))[0]
    b = np.exp(m.log_likelihood([[0, 0]], [2, 0, 0, 0, 0], 2, CFG, model="grain_redrawn_matching_marginals"))[0]
    assert a == pytest.approx(17 / 288, abs=1e-14)
    assert b == pytest.approx(25 / 576, abs=1e-14)
    assert a - b == pytest.approx(1 / 64, abs=1e-14)


def test_missing_group_equals_explicit_sum():
    points = m.grid(CFG)
    collapsed = np.exp(m.log_likelihood(points, [2, None, 0, 2, None], 8, CFG, scheme="marginal"))
    explicit = sum(np.exp(m.log_likelihood(points, [2, j, 0, 2, 4 - j], 8, CFG)) for j in range(5))
    np.testing.assert_allclose(collapsed, explicit, atol=1e-12, rtol=0)


def test_conditional_subset_is_normalized_and_uses_selected_denominator():
    points = [[0, 0], [-0.5, 0.5]]
    total = sum(np.exp(m.log_likelihood(points, list(y) + [None], 5, CFG,
                scheme="conditional", selection_documented=True)) for y in compositions(5, 4))
    np.testing.assert_allclose(total, 1, atol=1e-12, rtol=0)
    q = m.probabilities(points, CFG)[:, :4]
    expected = m.log_multinomial(q / q.sum(1, keepdims=True), [2, 1, 0, 2])
    actual = m.log_likelihood(points, [2, 1, 0, 2, None], 5, CFG,
                              scheme="conditional", selection_documented=True)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("prior", CFG["priors"])
@pytest.mark.parametrize("case", ["all_missing", "zero_total"])
def test_prior_only(prior, case):
    points = m.grid(CFG)
    c = CFG[case]
    total = c["known_total"] if case == "all_missing" else c["total"]
    w, e, _ = m.invert(points, c["counts"], total, CFG, prior,
                       scheme="marginal" if case == "all_missing" else "complete")
    np.testing.assert_allclose(w, m.prior_mass(points, CFG, prior), atol=1e-12, rtol=0)
    assert abs(e) < 1e-12


@pytest.mark.parametrize("prior", CFG["priors"])
def test_merged_sign_not_identified(prior):
    points = m.grid(CFG)
    y, _ = m.generate([0.5, 0], 40, 404, CFG, allocation=CFG["allocation_merged"])
    w, _, ll = m.invert(points, y, 40, CFG, prior, allocation=CFG["allocation_merged"])
    np.testing.assert_allclose(ll.reshape(21, 21), ll.reshape(21, 21)[::-1], atol=1e-12, rtol=0)
    if prior == "uniform":
        np.testing.assert_allclose(w.reshape(21, 21), w.reshape(21, 21)[::-1], atol=1e-12, rtol=0)
    else:
        assert np.max(abs(w.reshape(21, 21) - w.reshape(21, 21)[::-1])) > 1e-4


def test_scale_production_and_type_mixture_invariants():
    points = m.grid(CFG)
    q = m.probabilities(points, CFG)
    np.testing.assert_allclose(q, m.probabilities(points, CFG, production=[7] * 5), atol=1e-12, rtol=0)
    r = [math.exp(-1), math.exp(1), 1, 1, 1]
    np.testing.assert_allclose(m.probabilities([[0, 0]], CFG),
                               m.probabilities([[0.5, 0]], CFG, production=r), atol=1e-12, rtol=0)
    x = np.array([[0.2] * 5])
    changed = x + [0.05, -0.05, 0, 0, 0]
    np.testing.assert_allclose(m.flux_probabilities(x, CFG["allocation_merged"]),
                               m.flux_probabilities(changed, CFG["allocation_merged"]), atol=1e-12, rtol=0)


def test_structural_zero_and_impossible_all_states():
    assert m.log_multinomial([[0, 1]], [0, 1])[0] == 0
    assert np.isneginf(m.log_multinomial([[0, 1]], [1, 0])[0])
    a = np.array(CFG["allocation_base"], dtype=float)
    a[0, 0], a[1, 0] = 0, 1
    for prior in CFG["priors"]:
        with pytest.raises(m.ModelConflict, match="all supported states impossible"):
            m.invert(m.grid(CFG), [1, 0, 0, 0, 0], 1, CFG, prior, allocation=a)


@pytest.mark.parametrize("counts,total,kwargs", [
    ([-1, 1, 0, 0, 0], 0, {}), ([0.5, 0.5, 0, 0, 0], 1, {}),
    ([1, 0, 0, 0, 0], 2, {}), ([1, 0, 0, 0, 0], None, {}),
    ([1, 0, 0, 0, 0], float("inf"), {}), ([1, 0, 0, 0, 0], 1.5, {}),
    ([1, float("inf"), 0, 0, 0], 1, {}), ([1, 0], 1, {}),
    ([None] * 5, 40, {}), ([2, None, 0, 0, 0], 1, {"scheme": "marginal"}),
    ([1, 0, 0, 0, None], 1, {"scheme": "conditional"}),
    ([1, 0, 0, 0, None], 2, {"scheme": "conditional", "selection_documented": True}),
    ([1, 0, 0, 0, 0], 1, {"scheme": "other"}),
    ([1, 0, 0, 0, 0], 1, {"model": "other"}),
])
def test_invalid_observations(counts, total, kwargs):
    with pytest.raises(ValueError):
        m.invert(m.grid(CFG), counts, total, CFG, **kwargs)


@pytest.mark.parametrize("point", [[-1.1, 0], [1.1, 0], [0, -1.1], [0, 1.1], [float("nan"), 0], [0, float("inf")]])
def test_invalid_climate(point):
    with pytest.raises(ValueError, match="support"):
        m.probabilities([point], CFG)


@pytest.mark.parametrize("x,a", [([[-1, 1]], [[1, 1]]), ([[0, 0]], [[1, 1]]),
    ([[1, 1]], [[0.5, 1]]), ([[1, 1]], [[-1, 1], [2, 0]]),
    ([[1, 1]], [[float("nan"), 1]]), ([[1, 1]], [[1]]), ([[float("inf"), 1]], [[1, 1]])])
def test_invalid_flux_or_allocation(x, a):
    with pytest.raises(ValueError):
        m.flux_probabilities(x, a)


def test_remaining_invalid_inputs_and_hpd_ties():
    for kwargs in ({"production": [-1] * 5}, {"background": [float("nan")] * 5},
                   {"detection": [1.1] * 5}, {"production": [1, 1]}):
        with pytest.raises(ValueError):
            m.probabilities([[0, 0]], CFG, **kwargs)
    for q in ([[float("inf"), 0]], [[0.5, 0.6]], [[-0.1, 1.1]]):
        with pytest.raises(ValueError):
            m.log_multinomial(q, [0, 1])
    with pytest.raises(ValueError):
        m.invert(m.grid(CFG), [0] * 5, 0, CFG, prior="other")
    w = m.prior_mass(m.grid(CFG), CFG, "uniform")
    assert m.summarize(m.grid(CFG), w, CFG)["hpd_cells"] == 441
