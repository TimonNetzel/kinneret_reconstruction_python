"""Collect synthetic chain results and recheck three explicitly bound scripts.

Only the synthetic JSON fixture enters the numerical functions. Original scripts
are read for hash equality only, never executed. No original arrays are loaded.
"""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from check_chain import evaluate_fixture
from rational_reference import exact_expectations


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def collect():
    reports = {}
    scripts = {}
    for packet in ("literal_age_depth", "literal_age_selection", "literal_final_pdf", "literal_cdf_discrepancy"):
        rel = f"docs/thesis_research/{packet}_2026_11_02/source_binding.json"
        raw = (ROOT / rel).read_bytes()
        report = json.loads(raw)
        reports[rel] = {"bytes": len(raw), "sha256": digest(raw)}
        if "sources" in report:
            scripts.update({label: entry["sha256"] for label, entry in report["sources"].items()})
        else:
            scripts[report["source"]] = report["source_sha256"]
    cfg = json.loads((ROOT / ".local/my_reconst_paths.json").read_text(encoding="utf-8"))
    original = Path(cfg["original_data_root"])
    checked = {}
    payloads = {}
    for rel in ("src/check_ages.R", "src/mcmc_prepare.R", "src/mcmc_functions_v1.6.R"):
        label = "original:" + rel
        before = (original / rel).read_bytes()
        if digest(before) != scripts[label]:
            raise ValueError("Previously bound script changed: " + label)
        checked[label] = {"bytes": len(before), "sha256": digest(before)}
        payloads[label] = before
    f = json.loads((OUT / "synthetic_fixture.json").read_text())
    expected = exact_expectations(f)
    observed = evaluate_fixture(f)
    errors = {}
    def compare(name, actual, rational):
        convert = lambda value: [convert(v) for v in value] if isinstance(value, list) else float(Fraction(value))
        err = float(np.abs(np.asarray(actual) - np.asarray(convert(rational))).max())
        if err > f["tolerance"]:
            raise ValueError("Synthetic exact-reference mismatch: " + name)
        errors[name] = err
    for key in ("site_age", "age", "depth_weights", "pollen_at_ages", "column_max_scaled", "raw_sum", "pdf"):
        compare(key, observed[key], expected[key])
    for draw_id in expected["draws"]:
        for key in ("coefficients", "raw_density", "raw_cumulative", "zero_reference_scores"):
            compare(f"draw_{draw_id}_{key}", observed["draws"][draw_id][key], expected["draws"][draw_id][key])
    for key in ("draw_ids_R_1_based", "period_masks", "initial_reconst_mask", "effective_reconst_mask"):
        if observed[key] != expected[key]:
            raise ValueError("Synthetic control mismatch: " + key)
    for label in checked:
        if digest((original / label.split(":", 1)[1]).read_bytes()) != checked[label]["sha256"]:
            raise ValueError("Source changed while reading: " + label)
        checked[label]["hash_before_equals_after_and_prior"] = True
    modules = {}
    for name in ("historical_age_depth", "historical_age_selection", "historical_final_pdf", "historical_cdf_discrepancy"):
        rel = "tools/" + name + ".py"
        raw = (ROOT / rel).read_bytes()
        modules[rel] = {"bytes": len(raw), "sha256": digest(raw)}
    write("source_binding.json", {
        "status": "synthetic_composition_of_preserved_fragments_not_historical_runner",
        "basis_commit": "674bf132691aadf51359ee83052987a97e7c8587",
        "checked_original_scripts": checked, "source_reports": reports, "unchanged_modules": modules,
        "source_edges": {"chronology": "check_ages.R:177,179,230-232; mcmc_prepare.R:11-20,83,91-94",
                         "age_selection": "mcmc_prepare.R:435-442,1462-1467,1500-1522; mcmc_functions_v1.6.R:461-462",
                         "draw_ids": "mcmc_functions_v1.6.R:375,428, before active per-draw products",
                         "weighting": "mcmc_functions_v1.6.R:411-425,450-455",
                         "cdf": "mcmc_functions_v1.6.R:369-370,485-489; inherited and current width explicit",
                         "mixture": "mcmc_functions_v1.6.R:543-547,711-715,755-760"},
        "TF_input_scope": "Artificial supplied TF rows with unequal masses 2 and 3; finite fragment interface probe. Historical preparation at mcmc_prepare.R:1581-1586 and its later mask is not reproduced or claimed.",
        "oracle": "rational_reference.py: Python Fraction scalar sums only; no NumPy or implementation imports; fixed synthetic fixture",
        "R_execution": False, "actual_original_numerical_inputs_used": False,
        "final_variant_selected": False, "historical_reproduction_ready": False,
    })
    prep_label = "original:src/mcmc_prepare.R"
    lines = payloads[prep_label].decode("utf-8-sig").splitlines()
    write("next_step_source_excerpt.json", {
        "status": "read_only_source_for_next_step_not_implemented_in_this_chain",
        "source": prep_label, "sha256": checked[prep_label]["sha256"],
        "spans": [{"first_line": a, "last_line": b, "text": "\n".join(lines[a-1:b])}
                  for a, b in ((1580, 1586), (1590, 1618), (1622, 1652), (1663, 1699))],
        "limits": "No sample, quantile, spline, weightedMedian, inside.range or taxon-pruning code executed. Boundary convention/package binding still unresolved; no final mask selected."
    })
    write("rational_expectations.json", expected)
    active_densities = [np.array(observed["draws"][str(i)]["raw_density"]) for i in observed["draw_ids_R_1_based"]]
    premature = sum(d / d.sum(axis=1, keepdims=True) for d in active_densities) / len(active_densities)
    from copy import deepcopy
    normalized_fixture = deepcopy(f)
    tf = np.array(f["transfer_functions_taxon_by_cell"], dtype=float)
    normalized_fixture["transfer_functions_taxon_by_cell"] = (tf / tf.sum(axis=1, keepdims=True)).tolist()
    counterexamples = {
        "normalizing_each_draw_before_sum_max_pdf_change": float(np.max(np.abs(premature - observed["pdf"]))),
        "silently_normalizing_supplied_tf_rows_max_pdf_change": float(np.max(np.abs(np.array(evaluate_fixture(normalized_fixture)["pdf"]) - observed["pdf"]))),
        "depth_row_masses_after_literal_zero_replacement": np.array(observed["depth_weights"]).sum(axis=1).tolist()
    }
    write("synthetic_result.json", {"counterexamples": counterexamples, "kind": "synthetic_module_composition_only", "observed": observed,
                                    "max_abs_errors": errors, "tolerance": f["tolerance"],
                                    "R_execution": False, "historical_reproduction_ready": False})
    print(json.dumps({"original_script_hashes_match": len(checked), "active_draw_ids": observed["draw_ids_R_1_based"],
                      "max_abs_error": max(errors.values()), "axis_shapes": observed["axis_shapes"]}))


if __name__ == "__main__":
    collect()
