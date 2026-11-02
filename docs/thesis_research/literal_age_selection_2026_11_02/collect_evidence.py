"""Read four previously bound scripts and export synthetic age-selection checks.

No originals are written; no R, model, sampler, chronology or inventory runs.
Private source roots are read exclusively from .local/my_reconst_paths.json.
"""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tools"))
import historical_age_selection as source


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def collect():
    cfg = json.loads((ROOT / ".local/my_reconst_paths.json").read_text(encoding="utf-8"))
    original = Path(cfg["original_data_root"])
    prior = json.loads((ROOT / "docs/thesis_research/kinneret_binding_2026_11_02/summary.json").read_text())
    final = json.loads((ROOT / "docs/thesis_research/literal_final_pdf_2026_11_02/source_binding.json").read_text())
    spans = {
        "src/check_ages.R": [(175, 191), (219, 232)],
        "src/mcmc_prepare.R": [(435, 446), (820, 837), (1458, 1467), (1499, 1522)],
        "src/mcmc_functions_v1.6.R": [(366, 370), (387, 394), (433, 476), (485, 490), (1816, 1837)],
        "src/mcmc_case_studies.R": [(306, 317)],
    }
    bound = {}
    for rel, ranges in spans.items():
        label = "original:" + rel
        expected = final["source_sha256"] if label == final["source"] else prior["sources"][label]["sha256"]
        raw = (original / rel).read_bytes()
        if digest(raw) != expected:
            raise ValueError("Previously bound script changed: " + label)
        lines = raw.decode("utf-8-sig").splitlines()
        bound[label] = {"sha256": digest(raw), "bytes": len(raw),
                        "previous_binding": "literal_final_pdf_2026_11_02/source_binding.json" if label == final["source"] else "kinneret_binding_2026_11_02/summary.json",
                        "spans": [{"first_line": a, "last_line": b, "text": "\n".join(lines[a-1:b])} for a, b in ranges]}
    f = json.loads((OUT / "synthetic_fixture.json").read_text())
    masks = source.period_membership(f["age"])
    grids = source.period_grids(f["age"])
    trace = source.preserved_selection_trace(f["age"], taxa_merge=True)
    initial = source.rw_selected_age_product(f["coefficients"], f["transfer_functions"], trace.initial_reconst_ages)
    effective = source.rw_selected_age_product(f["coefficients"], f["transfer_functions"], trace.effective_reconst_ages)
    rational = lambda rows: np.array([[float(Fraction(v)) for v in row] for row in rows])
    errors = {"initial_product": float(np.abs(initial - rational(f["expected_initial_product"])).max()),
              "effective_product": float(np.abs(effective - rational(f["expected_effective_product"])).max())}
    if max(errors.values()) > f["tolerance"]:
        raise ValueError("Synthetic rational comparison failed")
    for name, mask in masks.items():
        if mask.tolist() != f["expected_masks"][name] or len(grids[name]) != f["expected_grid_counts"][name]:
            raise ValueError("Synthetic grid check failed")
    for label in bound:
        after = digest((original / label.split(":", 1)[1]).read_bytes())
        if after != bound[label]["sha256"]:
            raise ValueError("Source changed while reading: " + label)
        bound[label]["hash_before_equals_after_and_prior"] = True
    write("source_binding.json", {
        "status": "preserved_fragments_not_final_run_binding", "basis_commit": "6241f41408ed08ecc373d2dad1ace559a4effbb2",
        "sources": bound,
        "implemented": {"period_grids": "mcmc_prepare.R:435-437; bounded half-unit domain",
                        "period_membership": "mcmc_prepare.R:440-442; exact %in%",
                        "preserved_selection_trace": "mcmc_prepare.R:1462-1467,1500-1522; conditional on reaching these lines",
                        "rw_selected_age_product": "mcmc_functions_v1.6.R:461-462,470; already prepared coefficients and TFs"},
        "formula": "step = age[2]-age[1]; grids from literal period starts by step up to endpoints; masks = exact age membership; Y = C[mask,] @ TF",
        "orientation": "C: age rows x ordered taxa; mask: same age rows; TF: taxa x cells; product: selected rows x cells",
        "units": "Age-grid literals use the supplied age unit (historical bound Kinneret candidate: cal a BP), no a/ka conversion. Product retains input TF units; synthetic cells are abstract.",
        "source_control_flow": {"taxa_change_initialized": False, "change_branch_guard": "any(sitename == c(\"Birkat Ram\",\"Dead Sea\")) & FALSE", "effective_mask_if_reached": "all TRUE for both taxa.merge cases", "final_run_bound": False},
        "free_environment_variables_in_final_reconstruction": ["reconst_ages", "ages_lg", "taxa_change"],
        "supported_domain": "Finite half-unit age vectors within +/-1e9, >=2 ages, positive first difference; finite nonnegative matrices, >=2 taxa and selected rows; exact same-length boolean mask; finite product.",
        "excluded": ["general R seq floating tolerance", "NA/Inf, mask recycling, numeric indices, matrix dimension loss", "taxon matching/merging or stochastic TF preparation", "taxa_change rbind path and stochastic transition", "site.age resynchronization", "full function execution, EMD, posterior selection or historical run binding"],
        "actual_original_numerical_inputs_used": False, "R_execution": False, "historical_reproduction_ready": False,
    })
    write("synthetic_result.json", {
        "kind": "synthetic_age_selection_not_fossil_reconstruction",
        "grid_counts": {name: len(g) for name, g in grids.items()},
        "period_masks": {name: m.tolist() for name, m in masks.items()},
        "initial_merge_mask": trace.initial_reconst_ages.tolist(), "effective_mask_if_reached": trace.effective_reconst_ages.tolist(),
        "taxa_change": trace.taxa_change,
        "initial_selected_rows_R_1_based": (np.flatnonzero(trace.initial_reconst_ages) + 1).tolist(),
        "initial_product": initial.tolist(), "effective_product": effective.tolist(),
        "max_abs_errors": errors, "tolerance": f["tolerance"],
        "actual_original_numerical_inputs_used": False, "R_execution": False,
    })
    print(json.dumps({"source_hashes_match_prior_and_unchanged": len(bound), "max_abs_errors": errors}))


if __name__ == "__main__":
    collect()
