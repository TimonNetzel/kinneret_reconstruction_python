"""Read three already bound R scripts; export literal spans and synthetic scores.

No original data numerics, R execution, model runs, sampler or source inventory.
Private roots come from .local/my_reconst_paths.json; originals are read only.
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
import historical_cdf_discrepancy as source


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def collect():
    cfg = json.loads((ROOT / ".local/my_reconst_paths.json").read_text(encoding="utf-8"))
    original = Path(cfg["original_data_root"])
    prior = json.loads((ROOT / "docs/thesis_research/literal_age_selection_2026_11_02/source_binding.json").read_text())
    main_prior = json.loads((ROOT / "docs/thesis_research/kinneret_binding_2026_11_02/summary.json").read_text())
    spans = {
        "main_v1.2.R": [(65, 70), (650, 661)],
        "src/mcmc_functions_v1.6.R": [(366, 406), (485, 490), (755, 760), (846, 849)],
        "src/mcmc_case_studies.R": [(306, 317)],
    }
    bound = {}
    for rel, ranges in spans.items():
        label = "original:" + rel
        raw = (original / rel).read_bytes()
        previous = main_prior if rel == "main_v1.2.R" else prior
        if digest(raw) != previous["sources"][label]["sha256"]:
            raise ValueError("Previously bound script changed: " + label)
        lines = raw.decode("utf-8-sig").splitlines()
        bound[label] = {"sha256": digest(raw), "bytes": len(raw),
                        "previous_binding": "kinneret_binding_2026_11_02/summary.json" if rel == "main_v1.2.R" else "literal_age_selection_2026_11_02/source_binding.json",
                        "spans": [{"first_line": a, "last_line": b, "text": "\n".join(lines[a-1:b])} for a, b in ranges]}
    f = json.loads((OUT / "synthetic_fixture.json").read_text())
    pair = source.prepare_reference_pair(1, 1, rho_rows=f["rho_rows"], inherited_dims=f["inherited_dims"])
    zero = source.cdf_discrepancy(f["density_age_by_cell"], pair.first, dims=f["current_dims"])
    explicit = source.cdf_discrepancy(f["density_age_by_cell"], f["explicit_reference_cdf"], dims=f["current_dims"])
    rational = lambda values: np.array([float(Fraction(v)) for v in values])
    errors = {"zero_reference_scores": float(np.abs(zero.score - rational(f["expected_zero_reference_scores"])).max()),
              "explicit_reference_scores": float(np.abs(explicit.score - rational(f["expected_explicit_reference_scores"])).max())}
    if max(errors.values()) > f["tolerance"]:
        raise ValueError("Synthetic rational comparison failed")
    for label in bound:
        if digest((original / label.split(":", 1)[1]).read_bytes()) != bound[label]["sha256"]:
            raise ValueError("Source changed while reading: " + label)
        bound[label]["hash_before_equals_after_and_prior"] = True
    write("source_binding.json", {
        "status": "preserved_fragments_not_final_run_binding", "basis_commit": "d6e3343decbb1e74409a6c10e179da4e9da7be9f",
        "sources": bound,
        "implemented": {"prepare_reference_pair": "mcmc_functions_v1.6.R:369-370", "raw_row_cumsums": "mcmc_functions_v1.6.R:485-486", "cdf_discrepancy": "mcmc_functions_v1.6.R:487-489"},
        "formula": "C[a,k] = sum(l<=k) density[a,l]; score[a] = (1/dims) * sum_k abs(C[a,k]-reference[a,k])",
        "orientation": "density/reference/cumulative: age rows x supplied ordered climate cells; score: one value per age row, per draw",
        "scalar_placeholder": {"guard": "length(ecdf_ref1)==1", "effect": "both references become zero matrices", "shape": "nrow(rho.d.g.a) x inherited dims", "preserved_main_dims_assignments": "dims <- 300 at main_v1.2.R:68,658; no proof of final TF width", "local_dims_assignment_after_guard": "dims <- length(x1range.bck) at 389", "case_study_calls": "ecdf_ref1=1,ecdf_ref2=1 at 309 and 317"},
        "normalization_order": "No normalization in 485-489; final mixture normalization occurs later at 755-760",
        "units": "No climate coordinates or cell widths appear; score retains cumulative array-value units, dimensionless for discrete probabilities. Physical temperature/precipitation distance is not established by this formula alone.",
        "supported_domain": "Finite nonnegative nonempty 2D density; finite 2D supplied references with exactly matching comparison shape; explicit positive integer current dims equal cell count. Length-one first reference requires explicit positive integer placeholder rows/inherited width. Ignored second reference is not inspected or converted on replacement path.",
        "excluded": ["R/package runtime equivalence and NA/na.rm semantics", "R recycling or implicit alignment", "missing/mismatched environment dims repair", "CDF validity enforcement, early normalization, physical cell widths or resampling", "climate-class EMD branch, original reference recovery or final-run selection"],
        "R_execution": False, "actual_original_numerical_inputs_used": False, "historical_reproduction_ready": False,
    })
    write("synthetic_result.json", {
        "kind": "synthetic_raw_cumulative_discrepancy_not_fossil_reconstruction",
        "zero_placeholder_replaced": pair.zero_placeholder_replaced,
        "placeholder_shape": list(pair.first.shape),
        "raw_cumulative": zero.cumulative.tolist(), "zero_reference_scores": zero.score.tolist(),
        "explicit_reference_scores": explicit.score.tolist(), "max_abs_errors": errors,
        "tolerance": f["tolerance"], "R_execution": False, "actual_original_numerical_inputs_used": False,
    })
    print(json.dumps({"source_hashes_match_prior_and_unchanged": len(bound), "max_abs_errors": errors}))


if __name__ == "__main__":
    collect()
