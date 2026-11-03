"""Independent rational examples for the preserved age/depth source fragment."""
from fractions import Fraction
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import historical_age_depth as source

PACKET = ROOT / "docs/thesis_research/literal_age_depth_2026_11_02"


def fixture():
    return json.loads((PACKET / "synthetic_fixture.json").read_text())


def rational(values):
    return np.array([[float(Fraction(v)) for v in row] for row in values])


def test_rectangular_kernel_orientation_and_exact_one_based_depth_offsets():
    f = fixture()
    rho = np.array(f["rho_age_by_depth_grid"])
    weights = source.depth_kernel_at_samples(rho, f["depths"])
    assert rho.shape == (4, 5)
    assert weights.shape == (4, 3)
    np.testing.assert_allclose(weights, rational(f["expected_weights"]), rtol=0, atol=1e-14)
    # Very large unselected columns distinguish sampling from whole-grid normalization.
    rho[:, [1, 3]] = 1e6
    np.testing.assert_array_equal(weights, source.depth_kernel_at_samples(rho, f["depths"]))


def test_post_normalization_zero_replacement_is_1e_minus_9_without_renormalizing():
    f = fixture()
    w = source.depth_kernel_at_samples(f["rho_age_by_depth_grid"], f["depths"])
    assert w[1, 0] == 1e-9
    assert w[2, 1] == 1e-9
    assert w[1].sum() == pytest.approx(1 + 1e-9, abs=1e-15, rel=0)
    assert w[1].sum() != 1
    assert w[0].sum() == 1


def test_transfer_has_age_rows_taxon_columns_and_matches_hand_derived_fractions():
    f = fixture()
    w = source.depth_kernel_at_samples(f["rho_age_by_depth_grid"], f["depths"])
    out = source.pollen_at_ages(w, f["all_available_depth_by_taxon"])
    assert out.shape == (4, 2)
    np.testing.assert_allclose(out, rational(f["expected_pollen_at_ages"]), rtol=0, atol=1e-14)
    # Taxon values are not normalized into composition rows by this source block.
    assert out[0].sum() == pytest.approx(4.50000075)
    renorm = source.pollen_at_ages(w / w.sum(axis=1, keepdims=True), f["all_available_depth_by_taxon"])
    assert abs(out[1, 1] - renorm[1, 1]) > 1e-9


def test_pollen_zeros_replaced_before_product_without_changing_inputs():
    pollen = np.array([[0., 2.], [4., 0.], [1., 6.]])
    w = np.array([[.5, .25, .25], [0., 1., 0.]])
    original_pollen, original_w = pollen.copy(), w.copy()
    np.testing.assert_allclose(source.pollen_at_ages(w, pollen),
                               [[1.2500005, 2.50000025], [4., 1e-6]], rtol=0, atol=1e-14)
    np.testing.assert_array_equal(pollen, original_pollen)
    np.testing.assert_array_equal(w, original_w)


def test_kernel_row_scaling_cancels_but_column_scaling_does_not():
    f = fixture()
    rho = np.array(f["rho_age_by_depth_grid"], dtype=float)
    w = source.depth_kernel_at_samples(rho, f["depths"])
    np.testing.assert_allclose(source.depth_kernel_at_samples(rho * np.array([2, 4, 8, 16])[:, None], f["depths"]), w)
    rho[:, 0] *= 2
    assert not np.allclose(source.depth_kernel_at_samples(rho, f["depths"]), w)


def test_kernel_does_not_mutate_inputs_and_accepts_single_sample():
    rho = np.array([[0., 9., 1.], [2., 8., 2.]])
    depth = np.array([.5, 2.5])
    rho_before, depth_before = rho.copy(), depth.copy()
    source.depth_kernel_at_samples(rho, depth)
    np.testing.assert_array_equal(rho, rho_before)
    np.testing.assert_array_equal(depth, depth_before)
    np.testing.assert_array_equal(source.depth_kernel_at_samples([[2.]], [10.]), [[1.]])


def test_duplicate_age_shift_is_once_and_may_leave_collisions():
    f = fixture()
    ages = np.array(f["ages"], dtype=float)
    before = ages.copy()
    np.testing.assert_array_equal(source.shift_duplicate_ages_once(ages), f["expected_shifted_ages"])
    np.testing.assert_array_equal(ages, before)  # site.age must not be silently changed.
    np.testing.assert_array_equal(source.shift_duplicate_ages_once([5, 5, 5, 5.5, 5.5]), [5, 5.5, 5.5, 5.5, 6])
    np.testing.assert_array_equal(source.shift_duplicate_ages_once([2, -1, 0]), [2, -1, 0])


def test_list_selection_uses_explicit_one_based_id_and_returns_copies():
    matrices = [np.array([[2., 3.]]), np.array([[4., 5.], [6., 7.]])]
    ages = [np.array([-50.]), np.array([0., 50.])]
    rho, age = source.select_list_age_grid(matrices, ages, 2)
    np.testing.assert_array_equal(rho, matrices[1])
    np.testing.assert_array_equal(age, ages[1])
    rho[0, 0], age[0] = 99, 99
    assert matrices[1][0, 0] == 4
    assert ages[1][0] == 0


@pytest.mark.parametrize("resolution", [0, -1, 2, 1.5, True])
def test_resolution_id_is_not_clamped_or_defaulted(resolution):
    with pytest.raises(source.UnsupportedSourceDomain):
        source.select_list_age_grid([[[1]]], [[0]], resolution)


@pytest.mark.parametrize("rho,ages", [
    (np.ones((2, 3)), np.array([0, 50])),
    ([[1, 2], [3, 4]], [0, 50]),  # matrix/flat-vector as JSON-like lists
    ([[[1, 2]]], [[0, 50]]),  # age count != matrix rows
    ([[[1]]], []),
    ([[]], [[]]),  # no automatic empty-grid fallback
])
def test_matrix_flat_ages_or_inconsistent_lists_are_not_substituted(rho, ages):
    with pytest.raises(source.UnsupportedSourceDomain):
        source.select_list_age_grid(rho, ages, 1)


@pytest.mark.parametrize("rho,depths", [
    ([[1, 2, 3]], [10, 11.5]),
    ([[1, 2, 3]], [10, 13]),
    ([[1, 2, 3]], [12, 10]),
    ([[1, 2, 3]], [10, 10]),
    ([[0, 9, 0]], [10, 12]),
    ([[1, -1, 1]], [10, 12]),
    ([[1, float("nan"), 1]], [10, 12]),
    ([[1, float("inf"), 1]], [10, 12]),
    ([1, 2, 3], [10, 12]),
    ([], [10]),
    ([[1, 2, 3]], []),
    ([[1, 2, 3]], [float("nan")]),
    ([[1e308, 1e308]], [10, 11]),
])
def test_unsupported_kernel_inputs_are_not_repaired(rho, depths):
    with pytest.raises(source.UnsupportedSourceDomain):
        source.depth_kernel_at_samples(rho, depths)


@pytest.mark.parametrize("weights,pollen", [
    ([[1, 2, 3]], [[1, 2], [3, 4]]),
    ([[1, -1]], [[1], [2]]),
    ([[1, 1]], [[1], [float("nan")]]),
    ([[1, 1]], [[1], [-2]]),
    ([[float("inf")]], [[1]]),
    ([[1e308]], [[1e308]]),
    ([[1]], []),
])
def test_unsupported_products_are_not_filled_or_normalized(weights, pollen):
    with pytest.raises(source.UnsupportedSourceDomain):
        source.pollen_at_ages(weights, pollen)


@pytest.mark.parametrize("ages", [[], [float("nan")], [float("inf")], [[1, 2]]])
def test_unsupported_ages_are_not_imputed(ages):
    with pytest.raises(source.UnsupportedSourceDomain):
        source.shift_duplicate_ages_once(ages)


def test_saved_synthetic_result_remains_reproducible():
    f = fixture()
    recorded = json.loads((PACKET / "synthetic_result.json").read_text())
    weights = source.depth_kernel_at_samples(f["rho_age_by_depth_grid"], f["depths"])
    np.testing.assert_allclose(recorded["weights"], weights, rtol=0, atol=1e-14)
    np.testing.assert_allclose(recorded["pollen_at_ages"], source.pollen_at_ages(weights, f["all_available_depth_by_taxon"]), rtol=0, atol=1e-14)
    assert recorded["actual_original_numerical_inputs_used"] is False
