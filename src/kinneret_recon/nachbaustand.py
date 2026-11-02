"""Read the pinned dissertation reproduction state for the app; never run models, R scripts or samplers.

Separate from the older research snapshot (``forschungsstand.py``): dated state files bind the checked historical
submodules, the unchanged Phase-1 evidence contract and strictly synthetic worked examples. The newest state is
shown; earlier states stay frozen, are still hash-checked and are listed with their pin.
"""
from __future__ import annotations
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

REPO_URL = "https://github.com/TimonNetzel/kinneret_reconstruction_python/blob/"
# Oldest first. A new state gets a new folder and pin; earlier entries are never edited.
# ``commit`` is a pushed commit containing the state file, used only for the link to a frozen state.
STATES = (
    {"path": Path("docs/thesis_research/app_reproduction_2026_11_02/reproduction_state.json"),
     "sha256": "c358b2c3382f568a94d070054ba1dee4b230c617506c9a5015e2cc4506f4e63e",
     "modules": ("age_depth", "age_selection", "final_pdf", "cdf_discrepancy", "tf_preparation"),
     "commit": "47a926c3c8a6abc0db7c947b993f5e97f7408cdd"},
    {"path": Path("docs/thesis_research/app_reproduction_2026_11_02_pruning/reproduction_state.json"),
     "sha256": "3656316de4211500184f1a1c772bf07c87e3bd4b2d210ed6316e70c5a8ca0784",
     "modules": ("age_depth", "age_selection", "final_pdf", "cdf_discrepancy", "tf_preparation", "taxon_pruning"),
     "commit": "3d013b276cb1b7af37ce4058d6c10e499a34b504"},
    {"path": Path("docs/thesis_research/app_reproduction_2026_11_02_gamma_grid/reproduction_state.json"),
     "sha256": "43edfcb67a64826ce72611c93be33490463e59566934393aeff952bd217fbf4f",
     "modules": ("age_depth", "age_selection", "final_pdf", "cdf_discrepancy", "tf_preparation", "taxon_pruning",
                 "gamma_prior_grid")},
)
STATE, STATE_SHA256, MODULE_IDS = STATES[-1]["path"], STATES[-1]["sha256"], STATES[-1]["modules"]
REQUIREMENT_IDS = ("final_base", "final_chronology", "historical_tf", "isotope_workspaces", "cru_kgc_inputs",
                   "reference_variant", "anchor_variant", "posterior_selection", "postprocessing_variant",
                   "runtime_rng", "numeric_original_output")
STATUS = {"fehlt": "missing", "widerspruechlich": "conflicting"}


def _pinned(root: Path, rel: str, sha256: str) -> bytes:
    content = (root / rel).read_bytes()
    if hashlib.sha256(content).hexdigest() != sha256:
        raise ValueError(f"Nachbaustand: hash mismatch for {rel}; review the new evidence before updating the binding")
    return content


def _close(values, expected, tol: float) -> bool:
    if isinstance(values, list):
        return len(values) == len(expected) and all(_close(v, e, tol) for v, e in zip(values, expected))
    return math.isfinite(values) and abs(values - expected) <= tol


def app_daten(root: Path) -> dict:
    """Missing evidence stays visibly unavailable; changed bytes or release flags fail the build."""
    if not (root / STATE).is_file():
        return {"schema_version": 1, "available": False, "missing": [STATE.as_posix()]}
    state = json.loads(_pinned(root, STATE.as_posix(), STATE_SHA256))
    ex, contract, pex = state["synthetic_example"], state["contract"], state.get("pruning_example")
    gex = state.get("gamma_example")
    paths = ([s["path"].as_posix() for s in STATES[:-1]] + [m["package"] for m in state["modules"]]
             + [state["chain_check"]["package"], contract["package"], contract["spec"]]
             + [e[k] for e in (ex, pex, gex) if e for k in ("fixture", "result", "rational", "arithmetic")])
    if missing := [p for p in paths if not (root / p).is_file()]:
        return {"schema_version": 1, "available": False, "missing": missing}
    earlier = []
    for frozen in STATES[:-1]:
        old = json.loads(_pinned(root, frozen["path"].as_posix(), frozen["sha256"]))
        if old["historical_reproduction_ready"] or old["final_variant_selected"]                 or tuple(m["id"] for m in old["modules"]) != frozen["modules"]:
            raise ValueError("Nachbaustand: frozen earlier state claims a release or changed modules")
        earlier.append({"date": old["date"], "state_sha256": frozen["sha256"], "modules": len(old["modules"]),
                        "tests_passed": old["tests"]["passed"],
                        "url": REPO_URL + frozen["commit"] + "/" + frozen["path"].as_posix()})
    if (state["schema_version"] != 1 or state["historical_reproduction_ready"] or state["final_variant_selected"]
            or tuple(m["id"] for m in state["modules"]) != MODULE_IDS):
        raise ValueError("Nachbaustand: unsupported state or scientific release flag")
    for m in state["modules"] + [state["chain_check"]]:
        _pinned(root, m["package"], m["package_sha256"])
    _pinned(root, contract["package"], contract["package_sha256"])
    spec = json.loads(_pinned(root, contract["spec"], contract["spec_sha256"]))
    fixture = json.loads(_pinned(root, ex["fixture"], ex["fixture_sha256"]))
    result = json.loads(_pinned(root, ex["result"], ex["result_sha256"]))
    rational = json.loads(_pinned(root, ex["rational"], ex["rational_sha256"]))
    check = json.loads(_pinned(root, ex["arithmetic"], ex["arithmetic_sha256"]))
    chain = result["prepared_chain"]
    if (spec["selected_variant"] is not None or fixture["historical_reproduction_ready"]
            or fixture["selected_final_variant"] is not None or chain["historical_reproduction_ready"]
            or chain["actual_original_numerical_inputs_used"] or check["historical_reproduction_ready"]
            or check["final_variant_selected"] or check["R_execution"]
            or check["kind"] != "synthetic_fragment_check_not_historical_reproduction"):
        raise ValueError("Nachbaustand: example or contract claims historical inputs or a final variant")
    requirements = [{"id": r["id"], "status": STATUS[r["unresolved_status"]]} for r in spec["requirements"]]
    if tuple(r["id"] for r in requirements) != REQUIREMENT_IDS:
        raise ValueError("Nachbaustand: evidence contract requirements differ from the eleven open bundles")

    # Stored values only; the sums below are consistency checks of the stored results, not a recomputation
    tol = fixture["tolerance"]
    case, stored, exact = (fixture["cases"][ex["tf_case"]], result["cases"][ex["tf_case"]],
                           rational["cases"][ex["tf_case"]])
    for key in ("normalized", "masked", "row_masses_after_mask"):
        exact_values = [[float(Fraction(v)) for v in row] for row in exact[key]] if key != "row_masses_after_mask" \
            else [float(Fraction(v)) for v in exact[key]]
        if not _close(stored[key], exact_values, tol):
            raise ValueError(f"Nachbaustand: stored {key} differs from the exact reference")
    if (not _close([sum(r) for r in stored["normalized"]], [1.0] * len(stored["normalized"]), tol)
            or not _close([sum(r) for r in stored["masked"]], stored["row_masses_after_mask"], tol)
            or not _close([sum(r) for r in chain["pdf"]], [1.0] * len(chain["pdf"]), tol)
            or not check["maximum_absolute_rational_error"] <= check["tolerance"]):
        raise ValueError("Nachbaustand: stored example is internally inconsistent")
    url = lambda rel: REPO_URL + state["source_commit"] + "/" + rel
    return {
        "schema_version": 1, "available": True, "date": state["date"], "status": state["status"],
        "state_sha256": STATE_SHA256, "source_commit": state["source_commit"],
        "tests_passed": state["tests"]["passed"], "earlier_states": earlier,
        "modules": [{"id": m["id"], "new_tests": m["new_tests"], "url": url(m["package"]),
                     "module": m["module"], "original_sources": m["original_sources"],
                     "conditional": m.get("conditional", False)} for m in state["modules"]],
        "chain_check": {"new_tests": state["chain_check"]["new_tests"], "url": url(state["chain_check"]["package"])},
        "contract": {"tests": contract["tests"], "url": url(contract["package"])},
        "requirements": requirements,
        "example": {
            "url": url(ex["result"].rsplit("/", 1)[0] + "/README.md"),
            "raw": case["raw"], "keep": case["keep"],
            "normalized": stored["normalized"], "normalized_exact": exact["normalized"],
            "masked": stored["masked"], "masked_exact": exact["masked"],
            "row_masses_after_mask": stored["row_masses_after_mask"],
            "row_masses_exact": exact["row_masses_after_mask"],
            "chain_pdf": chain["pdf"], "draw_ids_R_1_based": chain["draw_ids_R_1_based"],
            "max_rational_error": check["maximum_absolute_rational_error"], "tolerance": check["tolerance"],
            "renormalization_tf_difference": check["post_mask_renormalization_TF_max_difference"],
            "renormalization_pdf_difference": check["post_mask_renormalization_chain_PDF_max_difference"],
            "sha256": {k: ex[k + "_sha256"] for k in ("fixture", "result", "rational", "arithmetic")},
        },
        "pruning_example": _pruning_example(root, pex, url) if pex else None,
        "gamma_example": _gamma_example(root, gex, url) if gex else None,
    }


def _gamma_example(root: Path, gex: dict, url) -> dict:
    """Stored values of artificial gamma-grid cases; doubles must equal the independent reference bit for bit."""
    fixture = json.loads(_pinned(root, gex["fixture"], gex["fixture_sha256"]))
    result = json.loads(_pinned(root, gex["result"], gex["result_sha256"]))
    rational = json.loads(_pinned(root, gex["rational"], gex["rational_sha256"]))
    check = json.loads(_pinned(root, gex["arithmetic"], gex["arithmetic_sha256"]))
    if (fixture["historical_reproduction_ready"] or fixture["selected_final_variant"] is not None
            or result["historical_reproduction_ready"] or check["historical_reproduction_ready"]
            or check["final_variant_selected"] or check["R_execution"]
            or check["kind"] != "synthetic_fragment_check_not_historical_reproduction"):
        raise ValueError("Nachbaustand: gamma example claims historical inputs or a final variant")
    cases = []
    for name in gex["cases"]:
        case, stored, exact = fixture["grid_cases"][name], result["cases"][name], rational["cases"][name]
        naive = check["naive_counterexamples"][name]
        if (any(stored[k] != exact[k] for k in ("index", "shape", "rate", "rate_expectation"))
                or stored["index"] + 1 != case["hand_r_index"] or case["upper"] != 50000.0):
            raise ValueError("Nachbaustand: stored gamma example differs from the reference or the hand expectation")
        cases.append({"mean": case["mean_pann"], "sd": case["sd_pann"], "closed_form": naive["closed_form_shape"],
                      "nearest": naive["nearest_grid_shape"], "nearest_differs": naive["nearest_grid_differs"],
                      "r_index": stored["index"] + 1, "shape": stored["shape"], "rate": stored["rate"]})
    chain, chain_exact = result["cases"][gex["chain_case"]], rational["cases"][gex["chain_case"]]
    if any(chain[k] != chain_exact[k] for k in chain_exact) or not chain["variance_doubled"]:
        raise ValueError("Nachbaustand: stored gamma chain differs from the reference")
    seq = check["seq_sensitivity"]["50000"]
    return {
        "url": url(gex["result"].rsplit("/", 1)[0] + "/README.md"), "cases": cases,
        "chain": {k: chain[k] for k in ("default_shape", "sd_pann", "final_shape", "final_rate",
                                         "rate_after_recent_start")},
        "grid_length": seq["length"], "grid_last": seq["last"],
        "sha256": {k: gex[k + "_sha256"] for k in ("result", "rational")},
    }


def _pruning_example(root: Path, pex: dict, url) -> dict:
    """Stored values of one artificial pruning case; indices must agree exactly with the fraction reference."""
    fixture = json.loads(_pinned(root, pex["fixture"], pex["fixture_sha256"]))
    result = json.loads(_pinned(root, pex["result"], pex["result_sha256"]))
    rational = json.loads(_pinned(root, pex["rational"], pex["rational_sha256"]))
    check = json.loads(_pinned(root, pex["arithmetic"], pex["arithmetic_sha256"]))
    if (fixture["historical_reproduction_ready"] or fixture["selected_final_variant"] is not None
            or result["historical_reproduction_ready"] or check["historical_reproduction_ready"]
            or check["final_variant_selected"] or check["R_execution"]
            or check["kind"] != "synthetic_fragment_check_not_historical_reproduction"):
        raise ValueError("Nachbaustand: pruning example claims historical inputs or a final variant")
    case, stored, exact = (fixture["cases"][pex["case"]], result["cases"][pex["case"]],
                           rational["cases"][pex["case"]])
    exact_ratio = [float(Fraction(v)) for v in exact["null_ratio"]]
    if (stored["removed_taxa"] != exact["removed_taxa"] or stored["branch"] != exact["branch"]
            or stored["metro_hastings"] != exact["metro_hastings"] or stored["skip_site"] != exact["skip_site"]
            or not _close(stored["null_ratio"], exact_ratio, fixture["tolerance"])
            or exact_ratio != [z / len(case["pollen"]) for z in case["zero_counts"]]):
        raise ValueError("Nachbaustand: stored pruning example differs from the exact reference")
    return {
        "url": url(pex["result"].rsplit("/", 1)[0] + "/README.md"),
        "rows": len(case["pollen"]), "zero_counts": case["zero_counts"],
        "null_ratio": stored["null_ratio"], "null_ratio_exact": exact["null_ratio"],
        "removed_taxa": stored["removed_taxa"], "branch": stored["branch"],
        "metro_hastings": stored["metro_hastings"], "naive_differs": check["naive_readings_differing_from_source"][pex["case"]],
        "sha256": {k: pex[k + "_sha256"] for k in ("result", "rational")},
    }
