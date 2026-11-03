"""Test/evidence-only calls on an explicit artificial fixture; no historical CLI.

No original files, models, references, automatic case selection or sampling.
The unused original draw slot is supplied sentinel data, never an active draw.
"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import historical_age_depth as chronology
import historical_age_selection as selection
import historical_final_pdf as pdf
import historical_cdf_discrepancy as cdf


def evaluate_fixture(f):
    rho, site_age = chronology.select_list_age_grid(
        f["rho_grids"], f["age_grids"], f["resolution_id_R_1_based"])
    age = chronology.shift_duplicate_ages_once(site_age)
    weights = chronology.depth_kernel_at_samples(rho, f["depths"])
    pollen = chronology.pollen_at_ages(weights, f["pollen_depth_by_taxon"])
    masks = selection.period_membership(age)
    trace = selection.preserved_selection_trace(age, taxa_merge=f["taxa_merge_control"])
    refs = cdf.prepare_reference_pair(1, 1, rho_rows=len(age), inherited_dims=f["inherited_dims"])
    # Source R 375/428 determines original IDs BEFORE active density/CDF calls.
    ids = pdf.unique_energy_ids(f["energy"])
    tensor = np.repeat(np.asarray(f["unused_draw_2_density_sentinel"], dtype=float)[None, :, :], len(f["energy"]), axis=0)
    draws = {}
    for draw_id in ids:
        coefficients = pdf.rw_coefficients(pollen, f["rw_weights_by_original_draw_id"][str(draw_id)])
        density = selection.rw_selected_age_product(
            coefficients, f["transfer_functions_taxon_by_cell"], trace.effective_reconst_ages)
        discrepancy = cdf.cdf_discrepancy(density, refs.first, dims=f["current_dims"])
        tensor[draw_id - 1] = density
        draws[str(draw_id)] = {"coefficients": coefficients.tolist(), "raw_density": density.tolist(),
                              "raw_cumulative": discrepancy.cumulative.tolist(),
                              "zero_reference_scores": discrepancy.score.tolist()}
    final = pdf.sum_pdf_branch(tensor, f["energy"])
    raw_sum = sum((tensor[i-1] for i in ids), np.zeros_like(tensor[0]))
    return {"site_age": site_age.tolist(), "age": age.tolist(),
            "period_masks": {name: mask.tolist() for name, mask in masks.items()},
            "initial_reconst_mask": trace.initial_reconst_ages.tolist(),
            "effective_reconst_mask": trace.effective_reconst_ages.tolist(),
            "taxa_change": trace.taxa_change, "depth_weights": weights.tolist(),
            "pollen_at_ages": pollen.tolist(), "column_max_scaled": pdf.column_max_taxa(pollen).tolist(),
            "draw_ids_R_1_based": list(final.draw_ids), "draws": draws,
            "raw_sum": raw_sum.tolist(), "pdf": final.density.tolist(),
            "axis_shapes": {"rho": list(rho.shape), "weights": list(weights.shape),
                            "pollen_at_ages": list(pollen.shape), "transfer_functions": list(np.shape(f["transfer_functions_taxon_by_cell"])),
                            "pdf": list(final.density.shape)},
            "actual_original_numerical_inputs_used": False, "historical_reproduction_ready": False}
