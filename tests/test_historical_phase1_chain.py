"""Cross-module synthetic contract; no historical input or production runner."""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "docs/thesis_research/literal_phase1_chain_2026_11_02"
sys.path.insert(0, str(PACKET))
from check_chain import evaluate_fixture, chronology, selection, pdf, cdf
from rational_reference import exact_expectations


def fixture():
    return json.loads((PACKET / "synthetic_fixture.json").read_text())


def expected():
    return json.loads((PACKET / "rational_expectations.json").read_text())


def numeric(value):
    return [numeric(v) for v in value] if isinstance(value, list) else float(Fraction(value))


def final_from_pollen(pollen, f):
    # Counterexamples only: keep all later operations identical to isolate timing.
    ids = pdf.unique_energy_ids(f["energy"])
    tensor = np.repeat(np.array(f["unused_draw_2_density_sentinel"], dtype=float)[None], len(f["energy"]), axis=0)
    for draw_id in ids:
        coefficients = pdf.rw_coefficients(pollen, f["rw_weights_by_original_draw_id"][str(draw_id)])
        tensor[draw_id-1] = selection.rw_selected_age_product(
            coefficients, f["transfer_functions_taxon_by_cell"], [True] * len(pollen))
    return pdf.sum_pdf_branch(tensor, f["energy"]).density


@pytest.mark.parametrize("stage", ["site_age", "age", "depth_weights", "pollen_at_ages", "column_max_scaled", "raw_sum", "pdf"])
def test_chain_intermediates_match_independent_exact_fraction_oracle(stage):
    f = fixture()
    np.testing.assert_allclose(evaluate_fixture(f)[stage], numeric(expected()[stage]), rtol=0, atol=f["tolerance"])


@pytest.mark.parametrize("draw_id", ["1", "3"])
@pytest.mark.parametrize("stage", ["coefficients", "raw_density", "raw_cumulative", "zero_reference_scores"])
def test_active_original_draw_stages_match_exact_rational_values(draw_id, stage):
    f = fixture()
    actual = evaluate_fixture(f)["draws"][draw_id][stage]
    np.testing.assert_allclose(actual, numeric(expected()["draws"][draw_id][stage]), rtol=0, atol=f["tolerance"])


def test_exact_oracle_reproduces_frozen_expectations_and_manual_kernel_values():
    f = fixture()
    assert exact_expectations(f) == expected()
    # Independently visible anchors, including TWO zero replacements in age row 3.
    np.testing.assert_array_equal(numeric(expected()["depth_weights"]),
                                  [[.25, .5, .25], [1e-9, .75, .25], [1e-9, 1e-9, 1], [.5, 1/6, 1/3]])
    assert Fraction(expected()["pollen_at_ages"][0][0]) == Fraction(4000001, 2000000)
    assert Fraction(expected()["pollen_at_ages"][0][1]) == Fraction(10000001, 4000000)


def test_axes_and_age_control_states_remain_distinct_across_modules():
    f = fixture(); result = evaluate_fixture(f)
    assert result["axis_shapes"] == {"rho": [4, 5], "weights": [4, 3], "pollen_at_ages": [4, 2], "transfer_functions": [2, 4], "pdf": [4, 4]}
    assert result["site_age"] == [-50, 0, 0, 11500]
    assert result["age"] == [-50, 0, .5, 11500]
    assert result["period_masks"] == {"hol": f["expected_hol_mask"], "lg": f["expected_lg_mask"], "eem": f["expected_eem_mask"]}
    assert result["initial_reconst_mask"] == [True, True, False, False]
    assert result["effective_reconst_mask"] == [True] * 4
    assert result["taxa_change"] is False
    assert result["historical_reproduction_ready"] is False


def test_only_retained_original_draw_ids_have_coefficients_and_scores():
    f = fixture(); actual = evaluate_fixture(f)
    assert actual["draw_ids_R_1_based"] == [1, 3]
    assert set(actual["draws"]) == {"1", "3"}
    assert "2" not in f["rw_weights_by_original_draw_id"]  # active evaluation would fail
    changed = deepcopy(f)
    changed["unused_draw_2_density_sentinel"] = [[1e6] * 4 for _ in range(4)]
    assert evaluate_fixture(changed) == actual


def test_energy_magnitudes_are_not_mixture_weights():
    f = fixture(); changed = deepcopy(f)
    changed["energy"] = [2, 2, 200]
    assert evaluate_fixture(changed) == evaluate_fixture(f)


def test_manual_cdf_and_mixture_identities_hold_with_unequal_tf_masses():
    f = fixture(); result = evaluate_fixture(f)
    c = np.array(result["draws"]["1"]["coefficients"])[:, 0]
    t = np.array(result["draws"]["3"]["coefficients"])[:, 0]
    # TF rows [2,0,0,0] and [0,0,1,2]: density mass 3-c, zero score 1+c.
    np.testing.assert_allclose(result["draws"]["1"]["zero_reference_scores"], 1+c, rtol=0, atol=1e-14)
    q = c + t
    raw = np.column_stack((2*q, np.zeros(4), 2-q, 4-2*q))
    np.testing.assert_allclose(result["raw_sum"], raw, rtol=0, atol=1e-14)
    np.testing.assert_allclose(result["pdf"], raw / (6-q)[:, None], rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.sum(result["pdf"], axis=1), np.ones(4), rtol=0, atol=1e-14)


def test_normalizing_each_draw_before_the_sum_changes_the_chain_output():
    result = evaluate_fixture(fixture())
    densities = [np.array(result["draws"][i]["raw_density"]) for i in ("1", "3")]
    premature = sum(d / d.sum(axis=1, keepdims=True) for d in densities) / 2
    assert np.max(np.abs(premature - result["pdf"])) > 1e-4


def test_zero_reference_comparison_is_before_final_mixture_normalization():
    f = fixture(); result = evaluate_fixture(f)
    raw = np.array(result["draws"]["1"]["raw_density"])
    normalized = cdf.cdf_discrepancy(raw / raw.sum(axis=1, keepdims=True), np.zeros_like(raw), dims=4).score
    assert np.max(np.abs(normalized - result["draws"]["1"]["zero_reference_scores"])) > .1


def test_extra_depth_kernel_normalization_survives_as_a_detectable_chain_difference():
    f = fixture(); result = evaluate_fixture(f)
    w = np.array(result["depth_weights"])
    assert w[2].sum() == pytest.approx(1 + 2e-9, rel=0, abs=1e-15)
    repaired = chronology.pollen_at_ages(w / w.sum(axis=1, keepdims=True), f["pollen_depth_by_taxon"])
    assert np.max(np.abs(final_from_pollen(repaired, f) - result["pdf"])) > 1e-12


def test_omitting_pollen_zero_replacement_changes_the_chain_output():
    f = fixture(); result = evaluate_fixture(f)
    without_pollen_epsilon = np.array(result["depth_weights"]) @ np.array(f["pollen_depth_by_taxon"])
    assert np.max(np.abs(final_from_pollen(without_pollen_epsilon, f) - result["pdf"])) > 1e-9


def test_silent_tf_row_normalization_or_same_shape_taxon_permutation_changes_output():
    f = fixture(); result = evaluate_fixture(f)
    normalized = deepcopy(f); tf = np.array(f["transfer_functions_taxon_by_cell"], dtype=float)
    normalized["transfer_functions_taxon_by_cell"] = (tf / tf.sum(axis=1, keepdims=True)).tolist()
    assert np.max(np.abs(np.array(evaluate_fixture(normalized)["pdf"]) - result["pdf"])) > .01
    permuted = deepcopy(f)
    permuted["transfer_functions_taxon_by_cell"] = list(reversed(f["transfer_functions_taxon_by_cell"]))
    assert np.max(np.abs(np.array(evaluate_fixture(permuted)["pdf"]) - result["pdf"])) > .01
    # Matching dimensions alone cannot detect a same-shape semantic permutation.


@pytest.mark.parametrize("broken", ["matrix_instead_of_lists", "rho_transposed", "pollen_transposed", "tf_transposed", "age_axis_shortened", "placeholder_width"])
def test_composition_rejects_inconsistent_boundary_shapes_without_repair(broken):
    f = fixture()
    if broken == "matrix_instead_of_lists":
        f["rho_grids"], f["age_grids"] = f["rho_grids"][1], f["age_grids"][1]
    elif broken == "rho_transposed":
        f["rho_grids"][1] = np.array(f["rho_grids"][1]).T.tolist()
    elif broken == "pollen_transposed":
        f["pollen_depth_by_taxon"] = np.array(f["pollen_depth_by_taxon"]).T.tolist()
    elif broken == "tf_transposed":
        f["transfer_functions_taxon_by_cell"] = np.array(f["transfer_functions_taxon_by_cell"]).T.tolist()
    elif broken == "age_axis_shortened":
        f["age_grids"][1] = f["age_grids"][1][:-1]
    else:
        f["inherited_dims"] = 5
    with pytest.raises((chronology.UnsupportedSourceDomain, selection.UnsupportedSourceDomain, cdf.UnsupportedSourceDomain)):
        evaluate_fixture(f)


def test_foreign_mask_length_is_rejected_at_the_actual_chain_product_boundary():
    f = fixture(); result = evaluate_fixture(f)
    with pytest.raises(selection.UnsupportedSourceDomain):
        selection.rw_selected_age_product(result["draws"]["1"]["coefficients"],
                                          f["transfer_functions_taxon_by_cell"], [True] * 3)


def test_fixture_and_saved_numeric_results_remain_reproducible_without_mutation():
    f = fixture(); before = deepcopy(f)
    actual = evaluate_fixture(f)
    assert f == before
    recorded = json.loads((PACKET / "synthetic_result.json").read_text())
    np.testing.assert_allclose(recorded["observed"]["pdf"], actual["pdf"], rtol=0, atol=f["tolerance"])
    assert set(recorded["observed"]["draws"]) == {"1", "3"}
    assert actual["actual_original_numerical_inputs_used"] is False
