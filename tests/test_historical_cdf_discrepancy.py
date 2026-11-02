"""Independent synthetic checks for the preserved raw cumulative discrepancy."""
from fractions import Fraction
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import historical_cdf_discrepancy as source

PACKET = ROOT / "docs/thesis_research/literal_cdf_discrepancy_2026_11_02"


def fixture():
    return json.loads((PACKET / "synthetic_fixture.json").read_text())


def rational(values):
    return np.array([float(Fraction(v)) for v in values])


def test_rectangular_raw_cumulative_and_zero_reference_scores_match_fractions():
    f = fixture()
    pair = source.prepare_reference_pair(1, 1, rho_rows=3, inherited_dims=4)
    result = source.cdf_discrepancy(f["density_age_by_cell"], pair.first, dims=4)
    assert pair.zero_placeholder_replaced is True
    assert result.cumulative.shape == (3, 4)
    expected = np.array([rational(row) for row in f["expected_raw_cumulative"]])
    np.testing.assert_array_equal(result.cumulative, expected)
    np.testing.assert_allclose(result.score, rational(f["expected_zero_reference_scores"]), rtol=0, atol=1e-14)
    # Last cumulative cells are raw masses 2, 1, 4, not all one.
    np.testing.assert_array_equal(result.cumulative[:, -1], [2, 1, 4])


def test_explicit_reference_is_not_cumulated_or_normalized_again():
    f = fixture()
    ref = np.array(f["explicit_reference_cdf"])
    pair = source.prepare_reference_pair(ref, ref * 2, rho_rows=None, inherited_dims=None)
    assert not pair.zero_placeholder_replaced
    result = source.cdf_discrepancy(f["density_age_by_cell"], pair.first, dims=4)
    np.testing.assert_allclose(result.score, rational(f["expected_explicit_reference_scores"]), rtol=0, atol=1e-14)
    np.testing.assert_array_equal(pair.second, ref * 2)
    altered = source.cdf_discrepancy(f["density_age_by_cell"], np.cumsum(ref, axis=1), dims=4)
    assert not np.allclose(result.score, altered.score)


@pytest.mark.parametrize("first", [1, -17, 0, [99], [[3]], np.array([[[2]]])])
def test_length_one_first_reference_value_is_ignored_and_both_are_replaced(first):
    class UnevaluatedSecond:
        def __array__(self, *args, **kwargs):
            raise AssertionError("Second reference must not be converted on replacement path")
    pair = source.prepare_reference_pair(first, UnevaluatedSecond(), rho_rows=2, inherited_dims=3)
    np.testing.assert_array_equal(pair.first, np.zeros((2, 3)))
    np.testing.assert_array_equal(pair.second, np.zeros((2, 3)))
    pair.first[0, 0] = 8
    assert pair.second[0, 0] == 0


def test_scalar_second_reference_does_not_trigger_symmetric_replacement():
    with pytest.raises(source.UnsupportedSourceDomain):
        source.prepare_reference_pair([[0, 1], [0, 1]], 1, rho_rows=2, inherited_dims=2)
    pair = source.prepare_reference_pair([[0, 1], [0, 1]], [[1]], rho_rows=2, inherited_dims=2)
    assert not pair.zero_placeholder_replaced
    with pytest.raises(source.UnsupportedSourceDomain):
        source.cdf_discrepancy([[1, 0], [0, 1]], pair.second, dims=2)


def test_inherited_width_is_used_before_current_width_and_never_repaired():
    pair = source.prepare_reference_pair(1, 1, rho_rows=3, inherited_dims=5)
    assert pair.first.shape == (3, 5)
    with pytest.raises(source.UnsupportedSourceDomain):
        source.cdf_discrepancy(np.ones((3, 4)), pair.first, dims=4)
    with pytest.raises(source.UnsupportedSourceDomain):
        source.prepare_reference_pair(1, 1, rho_rows=3, inherited_dims=None)


def test_density_is_not_prematurely_normalized():
    f = fixture()
    density = np.array(f["density_age_by_cell"], dtype=float)
    zeros = np.zeros_like(density)
    raw = source.cdf_discrepancy(density, zeros, dims=4).score
    early = source.cdf_discrepancy(density / density.sum(axis=1, keepdims=True), zeros, dims=4).score
    np.testing.assert_allclose(early, rational(f["expected_scores_if_density_were_prematurely_normalized"]), rtol=0, atol=1e-14)
    assert raw[0] == 2 * early[0]
    assert raw[2] == 4 * early[2]


def test_score_uses_cell_order_and_one_over_dims_without_coordinate_widths():
    f = fixture()
    density = np.array(f["density_age_by_cell"], dtype=float)
    raw = source.cdf_discrepancy(density, np.zeros_like(density), dims=4).score
    reversed_score = source.cdf_discrepancy(density[:, ::-1], np.zeros_like(density), dims=4).score
    assert raw[2] == 7 / 4
    assert reversed_score[2] == 13 / 4
    # A physical-width integral on the artificial nonuniform grid is a different number.
    widths = np.diff(f["synthetic_cell_coordinates_for_counterexample"])
    physical = np.sum(source.raw_row_cumsums(density)[:, :-1] * widths, axis=1)
    assert not np.allclose(raw, physical)


def test_signed_nonmonotone_reference_is_used_literally_without_cdf_repair():
    out = source.cdf_discrepancy([[1, 0, 1]], [[2, -1, 3]], dims=3)
    assert out.score[0] == pytest.approx(4 / 3)


def test_zero_rows_and_single_cells_are_preserved():
    result = source.cdf_discrepancy([[0], [2]], [[0], [1]], dims=1)
    np.testing.assert_array_equal(result.score, [0, 1])
    np.testing.assert_array_equal(result.cumulative, [[0], [2]])


def test_all_input_arrays_remain_unchanged_and_references_are_copied():
    density = np.array([[1., 2.], [3., 4.]])
    ref1, ref2 = np.array([[0., 1.], [0., 1.]]), np.ones((2, 2))
    before = [a.copy() for a in (density, ref1, ref2)]
    pair = source.prepare_reference_pair(ref1, ref2, rho_rows=2, inherited_dims=2)
    source.cdf_discrepancy(density, pair.first, dims=2)
    pair.first[0, 0] = 7
    for actual, expected in zip((density, ref1, ref2), before):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dims", [None, 0, -1, 4.0, True, 3])
def test_current_dims_is_explicit_positive_and_matches_width(dims):
    with pytest.raises(source.UnsupportedSourceDomain):
        source.cdf_discrepancy([[1, 0, 0, 0]], [[0, 0, 0, 0]], dims=dims)


@pytest.mark.parametrize("rows,width", [(0, 4), (-1, 4), (True, 4), (2, None), (2, 0), (2, 2.5)])
def test_missing_or_invalid_placeholder_dimensions_are_not_inferred(rows, width):
    with pytest.raises(source.UnsupportedSourceDomain):
        source.prepare_reference_pair(1, 1, rho_rows=rows, inherited_dims=width)


@pytest.mark.parametrize("density,reference", [
    ([], [[0]]), ([1, 2], [[0, 0]]),
    ([[1, -1]], [[0, 0]]), ([[1, float("nan")]], [[0, 0]]),
    ([[1, float("inf")]], [[0, 0]]), ([[1, 2]], [[0, float("nan")]]),
    ([[1, 2]], [[0, float("inf")]]), ([[1, 2]], [[0]]),
    ([[1, 2]], [[0, 0], [0, 0]]), ([[1e308, 1e308]], [[0, 0]]),
    ([[1e308, 0]], [[-1e308, 0]]),
])
def test_nonfinite_or_mismatched_comparisons_are_not_repaired(density, reference):
    with pytest.raises(source.UnsupportedSourceDomain):
        source.cdf_discrepancy(density, reference, dims=2)


@pytest.mark.parametrize("first,second", [([], []), ([0, 1], [[0, 1]]),
                                           (float("nan"), 1),
                                           ([[0, 1]], [[0, float("inf")]])])
def test_reference_domains_are_bounded(first, second):
    with pytest.raises(source.UnsupportedSourceDomain):
        source.prepare_reference_pair(first, second, rho_rows=1, inherited_dims=2)


def test_saved_synthetic_result_matches_independent_expectations():
    f = fixture()
    saved = json.loads((PACKET / "synthetic_result.json").read_text())
    np.testing.assert_allclose(saved["zero_reference_scores"], rational(f["expected_zero_reference_scores"]), rtol=0, atol=1e-14)
    np.testing.assert_allclose(saved["explicit_reference_scores"], rational(f["expected_explicit_reference_scores"]), rtol=0, atol=1e-14)
    assert saved["actual_original_numerical_inputs_used"] is False
