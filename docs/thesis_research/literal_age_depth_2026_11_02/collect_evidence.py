"""Read three already bound scripts; export source spans and synthetic arithmetic.

Original roots are read only, from the private configuration. No R execution,
RData loading, inventory, Bacon, fit, sampling, or case reconstruction.
Run with the checkout's Python from any directory.
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
import historical_age_depth as source


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def collect():
    config = json.loads((ROOT / ".local/my_reconst_paths.json").read_text(encoding="utf-8"))
    roots = {"original": Path(config["original_data_root"]), "additional": Path(config["additional_source_root"])}
    prior = json.loads((ROOT / "docs/thesis_research/kinneret_binding_2026_11_02/summary.json").read_text(encoding="utf-8"))
    spans = {
        "original:src/check_ages.R": [(175, 208), (213, 232)],
        "original:src/mcmc_prepare.R": [(3, 34), (49, 101)],
        "additional:my_kinneret/src/bacon_kinneret.R": [(68, 100), (105, 146)],
    }
    bound = {}
    for label, ranges in spans.items():
        scope, relative = label.split(":", 1)
        raw = (roots[scope] / relative).read_bytes()
        if digest(raw) != prior["sources"][label]["sha256"]:
            raise ValueError("Previously bound script changed: " + label)
        lines = raw.decode("utf-8-sig").splitlines()
        bound[label] = {
            "sha256": digest(raw), "bytes": len(raw),
            "previous_binding": "kinneret_binding_2026_11_02/summary.json#/sources/" + label,
            "spans": [{"first_line": a, "last_line": b, "text": "\n".join(lines[a-1:b])} for a, b in ranges],
        }
    f = json.loads((OUT / "synthetic_fixture.json").read_text())
    w = source.depth_kernel_at_samples(f["rho_age_by_depth_grid"], f["depths"])
    pollen = source.pollen_at_ages(w, f["all_available_depth_by_taxon"])
    rational = lambda rows: np.array([[float(Fraction(v)) for v in row] for row in rows])
    errors = {"weights": float(np.abs(w - rational(f["expected_weights"])).max()),
              "pollen_at_ages": float(np.abs(pollen - rational(f["expected_pollen_at_ages"])).max())}
    if max(errors.values()) > f["tolerance"]:
        raise ValueError("Synthetic rational comparison failed")
    for label in bound:
        scope, relative = label.split(":", 1)
        after = digest((roots[scope] / relative).read_bytes())
        if after != bound[label]["sha256"]:
            raise ValueError("Source changed while reading: " + label)
        bound[label]["hash_before_equals_after_and_prior"] = True
    write("source_binding.json", {
        "status": "preserved_fragments_not_final_run_binding", "basis_commit": "f7d97c1fad6852ce948cb868a16b10f7b1f83b8f",
        "sources": bound,
        "implemented": {
            "select_list_age_grid": "check_ages.R:177,179; explicit synthetic position only",
            "shift_duplicate_ages_once": "check_ages.R:230-232",
            "depth_kernel_at_samples": "mcmc_prepare.R:11-20; Birkat Ram 24-34 excluded",
            "pollen_at_ages": "mcmc_prepare.R:83,91-94; upstream filtering excluded",
        },
        "orientation": {"rho": "age x full depth grid", "weights": "age x sampled depth", "pollen": "sampled depth x taxon", "product": "age x taxon"},
        "depth_index_R_1_based": "(depth[j] - min(depth)) + 1",
        "weights_formula": "q[a,j] = rho[a,k[j]] / sum_l(rho[a,k[l]]); W[a,j] = 1e-9 if q[a,j]==0 else q[a,j]",
        "pollen_formula": "P'[j,t] = 1e-6 if P[j,t]==0 else P[j,t]; Y[a,t] = sum_j(W[a,j]*P'[j,t])",
        "units": "Historical bound Kinneret depths: cm; ages: cal a BP. W discrete, dimensionless. Y inherits P units; no new calibration or conversion.",
        "supported_domain": "Finite nonnegative matrices, explicit paired resolution lists, finite age vectors, already aligned unique increasing sampled depths with exact integer in-bounds offsets, positive selected row mass, finite product.",
        "not_implemented": ["site dispatch and age.reso override/fallback", "depth membership repair, standalone sorting and duplicate-depth deletion", "Birkat Ram loess", "NA/Inf propagation, missing-output fill and fractional-index truncation", "taxon filtering", "Bacon/KDE/grid estimation", "final chronology or run selection"],
        "final_variant_selected": False, "historical_reproduction_ready": False, "R_execution": False,
    })
    write("synthetic_result.json", {
        "kind": "synthetic_arithmetic_comparison_not_fossil_reconstruction",
        "depth_columns_R_1_based": f["expected_depth_columns_R_1_based"],
        "weights": w.tolist(), "pollen_at_ages": pollen.tolist(),
        "shifted_ages": source.shift_duplicate_ages_once(f["ages"]).tolist(),
        "max_abs_errors": errors, "tolerance": f["tolerance"],
        "actual_original_numerical_inputs_used": False, "R_execution": False,
    })
    print(json.dumps({"source_hashes_match_prior_and_unchanged": len(bound), "max_abs_errors": errors}))


if __name__ == "__main__":
    collect()
