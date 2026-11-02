"""Synthetic row/age checks against explicit independent masks and rational outputs."""
from fractions import Fraction
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import historical_age_selection as source

PACKET = ROOT / "docs/thesis_research/literal_age_selection_2026_11_02"


def fixture():
    return json.loads((PACKET / "synthetic_fixture.json").read_text())


def rational(rows):
    return np.array([[float(Fraction(v)) for v in row] for row in rows])


def test_exact_grid_membership_is_not_interval_membership():
    f = fixture()
    grids = source.period_grids(f["age"])
    masks = source.period_membership(f["age"])
    for name in ("hol", "lg", "eem"):
        assert len(grids[name]) == f["expected_grid_counts"][name]
        np.testing.assert_array_equal(masks[name], f["expected_masks"][name])
    # 0.5 and 11550.5 are inside the period ranges, but outside the exact grids.
    assert not masks["hol"][2]
    assert not masks["lg"][4]
    np.testing.assert_array_equal(grids["hol"][[0, -1]], [-50, 11450])
    np.testing.assert_array_equal(grids["lg"][[0, -1]], [11500, 115000])
    np.testing.assert_array_equal(grids["eem"][[0, -1]], [115500, 125000])


def test_grid_endpoint_is_not_forced_when_step_does_not_land_on_it():
    age = [-50, 100, 11450, 11350, 11500, 125000, 124950]
    grids = source.period_grids(age)  # first difference 150, not a regular-age assertion
    assert len(grids["hol"]) == 77
    assert grids["hol"][-1] == 11350
    assert grids["eem"][-1] == 124950
    np.testing.assert_array_equal(source.period_membership(age)["hol"], [True, True, False, True, False, False, False])


def test_half_unit_first_step_is_kept_not_replaced_with_typical_resolution():
    grids = source.period_grids([0, .5, 100])
    assert len(grids["hol"]) == 23001
    assert grids["hol"][1] == -49.5
    assert source.period_membership([0, .5, 100])["hol"].all()


def test_large_positive_step_yields_each_start_without_fabricating_endpoint():
    grids = source.period_grids([-50, 200000])
    for name, start in (("hol", -50), ("lg", 11500), ("eem", 115500)):
        np.testing.assert_array_equal(grids[name], [start])


def test_shifted_age_and_site_age_are_not_silently_synchronized():
    f = fixture()
    age = np.array(f["age"])
    site_age = np.array(f["site_age_before_duplicate_shift"])
    age_before, site_before = age.copy(), site_age.copy()
    np.testing.assert_array_equal(source.period_membership(age)["hol"], [True, True, False, False, False, False])
    np.testing.assert_array_equal(source.period_membership(site_age)["hol"], [True, True, True, False, False, False])
    np.testing.assert_array_equal(age, age_before)
    np.testing.assert_array_equal(site_age, site_before)


@pytest.mark.parametrize("taxa_merge", [False, True])
def test_initial_holocene_mask_is_overwritten_on_preserved_control_flow(taxa_merge):
    f = fixture()
    trace = source.preserved_selection_trace(f["age"], taxa_merge=taxa_merge)
    expected = f["expected_initial_merge_mask"] if taxa_merge else f["expected_effective_mask"]
    np.testing.assert_array_equal(trace.initial_reconst_ages, expected)
    np.testing.assert_array_equal(trace.effective_reconst_ages, f["expected_effective_mask"])
    assert trace.taxa_change is False
    trace.initial_reconst_ages[:] = False
    assert trace.effective_reconst_ages.all()  # no accidental array alias


def test_rectangular_product_preserves_logical_row_selection_and_tf_mass():
    f = fixture()
    out = source.rw_selected_age_product(f["coefficients"], f["transfer_functions"], f["expected_initial_merge_mask"])
    assert out.shape == (2, 4)
    np.testing.assert_allclose(out, rational(f["expected_initial_product"]), rtol=0, atol=1e-14)
    full = source.rw_selected_age_product(f["coefficients"], f["transfer_functions"], f["expected_effective_mask"])
    assert full.shape == (6, 4)
    np.testing.assert_allclose(full, rational(f["expected_effective_product"]), rtol=0, atol=1e-14)
    assert full[2].sum() == 4  # no extra TF or result normalization


def test_scattered_mask_keeps_source_row_order_without_sorting_or_deduplication():
    f = fixture()
    mask = [False, True, False, False, True, False]
    out = source.rw_selected_age_product(f["coefficients"], f["transfer_functions"], mask)
    np.testing.assert_allclose(out, [[0, 1, 0, 0], [.5, 0, 2.25, .75]], rtol=0, atol=1e-14)


def test_product_inputs_remain_unchanged():
    f = fixture()
    arrays = [np.array(f[key]) for key in ("coefficients", "transfer_functions", "expected_initial_merge_mask")]
    before = [a.copy() for a in arrays]
    source.rw_selected_age_product(*arrays)
    for a, original in zip(arrays, before):
        np.testing.assert_array_equal(a, original)


@pytest.mark.parametrize("age", [[], [0], [[0, 50]], [0, 0], [50, 0],
                                  [0, float("nan")], [0, float("inf")],
                                  [0, .25], [0, 50, 50.1], [0, 1e10]])
def test_unsupported_age_grids_are_not_repaired(age):
    with pytest.raises(source.UnsupportedSourceDomain):
        source.period_membership(age)


@pytest.mark.parametrize("flag", [None, 0, 1, "False"])
def test_taxa_merge_is_explicit_boolean(flag):
    with pytest.raises(source.UnsupportedSourceDomain):
        source.preserved_selection_trace([-50, 0], taxa_merge=flag)


@pytest.mark.parametrize("mask", [[1, 1, 0, 0, 0, 0], [True], [True] * 7,
                                   [[True] * 6], [None] * 6, [False] * 6,
                                   [True, False, False, False, False, False]])
def test_mask_recycling_integer_subscripts_and_dimension_drop_are_excluded(mask):
    f = fixture()
    with pytest.raises(source.UnsupportedSourceDomain):
        source.rw_selected_age_product(f["coefficients"], f["transfer_functions"], mask)


@pytest.mark.parametrize("coefficients,functions", [
    ([[1], [2]], [[1, 2]]),
    ([[1, 2], [3, 4]], [[1, 2]]),
    ([[1, -2], [3, 4]], [[1], [2]]),
    ([[1, 2], [3, 4]], [[-1], [2]]),
    ([[1, float("nan")], [3, 4]], [[1], [2]]),
    ([[1, 2], [3, 4]], [[1], [float("inf")]]),
    ([[1e308, 1e308], [1e308, 1e308]], [[1e308], [1e308]]),
])
def test_unsupported_products_are_not_normalized_or_imputed(coefficients, functions):
    with pytest.raises(source.UnsupportedSourceDomain):
        source.rw_selected_age_product(coefficients, functions, [True, True])


def test_saved_result_matches_synthetic_fixture():
    f = fixture()
    saved = json.loads((PACKET / "synthetic_result.json").read_text())
    for name, expected in f["expected_masks"].items():
        assert saved["period_masks"][name] == expected
    np.testing.assert_allclose(saved["initial_product"], rational(f["expected_initial_product"]), rtol=0, atol=1e-14)
    np.testing.assert_allclose(saved["effective_product"], rational(f["expected_effective_product"]), rtol=0, atol=1e-14)
    assert saved["actual_original_numerical_inputs_used"] is False
