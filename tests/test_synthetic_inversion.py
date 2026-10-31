"""Structural tests of the new toy likelihood, not of existing fitted models."""
import itertools
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "synthetic_prototype", Path(__file__).resolve().parents[1] / "tools/taxon_synthetic/prototype.py")
prototype = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prototype)
generate, grid, invert, load_config = (prototype.generate, prototype.grid, prototype.invert, prototype.load_config)
log_likelihood, prior_mass = prototype.log_likelihood, prototype.prior_mass
probabilities, summarize = prototype.probabilities, prototype.summarize


@pytest.fixture
def config():
    return load_config()


@pytest.mark.parametrize("model", ["independent", "shared"])
def test_latent_state_enumeration(config, model):
    """Enumerate H and all 2^4 Z states including an omitted channel."""
    point = [0.25, -0.5]
    y = np.array([[1, 0, np.nan, 1], [0, 1, 1, 0]])
    shifts = [0] if model == "independent" else [-1.5, 1.5]
    likelihood = 1.0
    for row in y:
        row_probability = 0.0
        for h in shifts:
            a = np.array([1.0, -1.0, -2.0, 2.0]) + h
            p = 1 / (1 + np.exp(-a))
            for z in itertools.product([0, 1], repeat=4):
                probability = np.prod([p[j] if z[j] else 1 - p[j] for j in range(4)])
                for j in range(4):
                    if np.isnan(row[j]):
                        continue
                    detection = 0.9 if z[j] else 0.1
                    probability *= detection if row[j] else 1 - detection
                row_probability += probability / len(shifts)
        likelihood *= row_probability
    assert np.exp(log_likelihood([point], y, config, model)[0]) == pytest.approx(likelihood, abs=1e-14)


def test_shared_is_not_product_of_marginals(config):
    y = [[1, 1, 1, 1]]
    p = [0, 0]
    correct = np.exp(log_likelihood([p], y, config, "shared"))[0]
    averaged = (probabilities([p], config, -1.5) + probabilities([p], config, 1.5)) / 2
    assert correct > 2 * averaged.prod()


@pytest.mark.parametrize("prior", ["uniform", "informative"])
@pytest.mark.parametrize("model", ["independent", "shared"])
def test_missing_and_no_detection_information(config, prior, model):
    points = grid(config)
    w, e, _ = invert(points, np.full((40, 4), np.nan), config, prior, model=model)
    np.testing.assert_allclose(w, prior_mass(points, config, prior), rtol=0, atol=1e-15)
    assert abs(e) < 1e-12
    w, _, _ = invert(points, [[1, 0, 1, 1]], config, prior, model=model, detection=(0.1, 0.1))
    np.testing.assert_allclose(w, prior_mass(points, config, prior), rtol=0, atol=1e-15)


@pytest.mark.parametrize("prior", ["uniform", "informative"])
def test_missing_dimension_is_prior(config, prior):
    points = grid(config)
    w, _, _ = invert(points, [[1, 0, np.nan, np.nan]] * 40, config, prior)
    np.testing.assert_allclose(w.reshape(41, 41).sum(0),
                               prior_mass(points, config, prior).reshape(41, 41).sum(0), atol=1e-14, rtol=0)


def test_folded_likelihood_ambiguity(config):
    y = generate([0.5, 0], 606, config, family="folded")
    points = grid(config)
    w, _, ll = invert(points, y, config, family="folded")
    np.testing.assert_allclose(ll.reshape(41, 41), ll.reshape(41, 41)[::-1], atol=1e-12, rtol=0)
    np.testing.assert_allclose(w.reshape(41, 41), w.reshape(41, 41)[::-1], atol=1e-14, rtol=0)
    assert abs(w @ points[:, 0]) < 1e-14


@pytest.mark.parametrize("point", [[-1.05, 0], [1.05, 0], [0, -1.05], [0, 1.05], [np.nan, 0], [0, np.inf]])
def test_support_rejected(config, point):
    with pytest.raises(ValueError, match="support"):
        probabilities([point], config)


@pytest.mark.parametrize("bad", [[[2, 0, 0, 0]], [[-1, 0, 0, 0]], [[np.inf, 0, 0, 0]], [[0, 1]], [0, 1, 0, 1]])
def test_bad_observations(config, bad):
    with pytest.raises(ValueError):
        invert(grid(config), bad, config)


@pytest.mark.parametrize("rates", [(-0.1, 0), (1.1, 0), (0.1, 0.9), (np.nan, 0), (1, np.inf)])
def test_bad_detection(config, rates):
    with pytest.raises(ValueError):
        probabilities([[0, 0]], config, detection=rates)


def test_unknown_identifiers_and_shapes(config):
    for kwargs in [{"prior": "other"}, {"model": "other"}, {"family": "other"}]:
        with pytest.raises(ValueError):
            invert(grid(config), [[1, 0, 1, 0]], config, **kwargs)
    for points in [[], [0, 0], [[0, 0, 0]]]:
        with pytest.raises(ValueError):
            probabilities(points, config)


def test_conflict_can_concentrate_with_no_v_information(config):
    points = grid(config)
    y = [[1, 1, np.nan, np.nan]] * 40
    w, _, _ = invert(points, y, config)
    u = w.reshape(41, 41).sum(1)
    assert np.argmax(u) == 20
    q = probabilities(points, config)
    assert np.max(q[:, 0] * q[:, 1]) ** 40 < config["conflict_tail_threshold"]
    assert np.max(u) > 1 / 41


def test_hpd_ties_and_impossible_data(config):
    points = grid(config)
    w = np.ones(len(points)) / len(points)
    summary = summarize(points, w, config)
    assert summary["hpd_cells"] == len(points)
    with pytest.raises(ValueError, match="impossible"):
        invert(points, [[1, 0, 0, 0]], config, detection=(0, 0))
