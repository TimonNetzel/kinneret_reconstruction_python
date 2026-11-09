"""Historical source-fragment checks using hand-derived synthetic arithmetic."""
from fractions import Fraction
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import historical_final_pdf as source

PACKET = ROOT / "docs/thesis_research/literal_final_pdf_2026_11_02"


def fixture():
    return json.loads((PACKET / "synthetic_fixture.json").read_text())


def rational_matrix(values):
    return np.array([[float(Fraction(v)) for v in row] for row in values])


def draw_densities(f):
    return np.array([source.apply_taxon_densities(
        source.rw_coefficients(f["all_available"], weights), f["transfer_functions"])
        for weights in f["weights"]])


def test_rectangular_weighting_matches_hand_derived_rational_values():
    f = fixture()
    scaled = source.column_max_taxa(f["all_available"])
    coefficients = source.rw_coefficients(f["all_available"], f["weights"][0])
    density = source.apply_taxon_densities(coefficients, f["transfer_functions"])
    np.testing.assert_allclose(scaled, rational_matrix(f["expected_column_max_scaled"]), rtol=0, atol=1e-14)
    np.testing.assert_allclose(coefficients, rational_matrix(f["expected_rw_first_coefficients"]), rtol=0, atol=1e-14)
    np.testing.assert_allclose(density, rational_matrix(f["expected_rw_first_raw_densities"]), rtol=0, atol=1e-14)
    # These TFs intentionally have unequal masses 2, 1 and 3.
    assert density[0].sum() == pytest.approx(8 / 3)
    assert not np.allclose(density.sum(axis=1), 1)


def test_column_units_do_not_change_column_max_weighting():
    f = fixture()
    pollen = np.array(f["all_available"], dtype=float)
    original = source.rw_coefficients(pollen, f["weights"][0])
    pollen[:, 1] *= 100
    np.testing.assert_allclose(source.rw_coefficients(pollen, f["weights"][0]), original)


def test_pdf_sum_matches_hand_derived_result_and_preserves_draw_ids():
    f = fixture()
    result = source.sum_pdf_branch(draw_densities(f), f["energy"])
    assert list(result.draw_ids) == f["expected_selected_draw_ids_one_based"]
    np.testing.assert_allclose(result.density, rational_matrix(f["expected_pdf_sum_normalized"]), rtol=0, atol=1e-14)
    np.testing.assert_allclose(result.density.sum(axis=1), 1)


def test_source_does_not_normalize_each_draw_before_summing():
    f = fixture()
    draws = draw_densities(f)
    result = source.sum_pdf_branch(draws, f["energy"]).density
    # A tempting alternate method changes sample 2. It is NOT inserted in the port.
    np.testing.assert_allclose(result[1], [9 / 19, 13 / 76, 0, 27 / 76], rtol=0, atol=1e-14)
    assert not np.allclose(result[1], [17 / 36, 25 / 144, 0, 17 / 48])


def test_energy_values_do_not_become_mixture_weights():
    f = fixture()
    draws = draw_densities(f)
    a = source.sum_pdf_branch(draws, [4, 4, 9])
    b = source.sum_pdf_branch(draws, [100, 100, -1])
    np.testing.assert_array_equal(a.density, b.density)
    assert a.draw_ids == b.draw_ids == (1, 3)
    draws[1] *= 1000  # The repeated-energy draw remains unused, regardless of its mass.
    np.testing.assert_array_equal(a.density, source.sum_pdf_branch(draws, [4, 4, 9]).density)


def test_energy_deduplication_is_exact_without_rounding():
    assert source.unique_energy_ids([1, 1 + 2 ** -30, 1, 0, -0.0]) == (1, 2, 4)


def test_gibbs_indices_keep_one_based_order_and_repeated_columns():
    f = fixture()
    np.testing.assert_allclose(source.gibbs_coefficients(f["all_available"], [3, 1]),
                               [[2 / 3, 1 / 3], [1 / 3, 2 / 3], [1, 0], [0, 1]])
    repeated = source.gibbs_coefficients(f["all_available"], [1, 3, 1])
    np.testing.assert_allclose(repeated[0], [1 / 4, 1 / 2, 1 / 4])


@pytest.mark.parametrize("ids", [[0, 1], [-1, 2], [1, 4], [True, 2], [1.0, 2], [], [1]])
def test_unsupported_gibbs_indexing_is_explicit(ids):
    with pytest.raises(source.UnsupportedSourceDomain):
        source.gibbs_coefficients(fixture()["all_available"], ids)


def test_r_dimension_drop_is_not_silently_repaired():
    with pytest.raises(source.UnsupportedSourceDomain, match="dimension-drop"):
        source.gibbs_coefficients([[1, 2, 3]], [1, 2])


@pytest.mark.parametrize("pollen", [[[0, 1], [0, 2]], [[1, 0], [0, 0]], [[1, float('nan')]], [[-1, 1]]])
def test_degenerate_or_nonfinite_input_is_not_imputed(pollen):
    with pytest.raises(source.UnsupportedSourceDomain):
        source.rw_coefficients(pollen, [1, 1])


def test_mismatched_shapes_and_zero_mass_fail_without_recycling_or_smoothing():
    with pytest.raises(source.UnsupportedSourceDomain):
        source.rw_coefficients([[1, 2], [2, 1]], [1])
    with pytest.raises(source.UnsupportedSourceDomain):
        source.rw_coefficients([[1, 2], [2, 1]], [0, 0])
    with pytest.raises(source.UnsupportedSourceDomain):
        source.apply_taxon_densities([[1, 0]], [[1, 0]])
    with pytest.raises(source.UnsupportedSourceDomain):
        source.sum_pdf_branch([[[1, 0]]], [1, 2])
    with pytest.raises(source.UnsupportedSourceDomain):
        source.sum_pdf_branch([[[0, 0]]], [1])
    with pytest.raises(source.UnsupportedSourceDomain):
        source.unique_energy_ids([float('inf')])


def test_inputs_are_not_modified():
    f = fixture()
    pollen = np.array(f["all_available"], dtype=float)
    weights = np.array(f["weights"][0], dtype=float)
    functions = np.array(f["transfer_functions"], dtype=float)
    before = [x.copy() for x in (pollen, weights, functions)]
    coefficients = source.rw_coefficients(pollen, weights)
    source.gibbs_coefficients(pollen, [3, 1])
    source.apply_taxon_densities(coefficients, functions)
    draws = draw_densities(f)
    draws_before = draws.copy()
    source.sum_pdf_branch(draws, f["energy"])
    for a, b in zip((pollen, weights, functions), before):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(draws, draws_before)


def test_source_binding_matches_existing_audited_original():
    binding = json.loads((PACKET / "source_binding.json").read_text())
    prior = json.loads((ROOT / "docs/thesis_research/kinneret_postprocessing_2026_11_02/contract.json").read_text())
    assert binding["source_sha256"] == prior["sources"][binding["source"]]["sha256"]
    assert binding["source_bytes"] == prior["sources"][binding["source"]]["bytes"]
    assert binding["run_binding"] is None
    assert binding["actual_original_inputs_used_for_numerics"] is False
    assert binding["R_or_package_runtime_equivalence_claimed"] is False
