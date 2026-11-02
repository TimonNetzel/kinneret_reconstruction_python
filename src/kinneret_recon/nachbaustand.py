"""Read the pinned dissertation reproduction state for the app; never run models, R scripts or samplers.

Separate from the older research snapshot (``forschungsstand.py``): a dated state file binds the five checked
historical submodules, the unchanged Phase-1 evidence contract and a strictly synthetic worked example.
"""
from __future__ import annotations
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

STATE = Path("docs/thesis_research/app_reproduction_2026_11_02/reproduction_state.json")
STATE_SHA256 = "c358b2c3382f568a94d070054ba1dee4b230c617506c9a5015e2cc4506f4e63e"
REPO_URL = "https://github.com/TimonNetzel/kinneret_reconstruction_python/blob/"
MODULE_IDS = ("age_depth", "age_selection", "final_pdf", "cdf_discrepancy", "tf_preparation")
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
    ex, contract = state["synthetic_example"], state["contract"]
    paths = [m["package"] for m in state["modules"]] + [state["chain_check"]["package"], contract["package"],
             contract["spec"]] + [ex[k] for k in ("fixture", "result", "rational", "arithmetic")]
    if missing := [p for p in paths if not (root / p).is_file()]:
        return {"schema_version": 1, "available": False, "missing": missing}
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
        "tests_passed": state["tests"]["passed"],
        "modules": [{"id": m["id"], "new_tests": m["new_tests"], "url": url(m["package"]),
                     "module": m["module"], "original_sources": m["original_sources"]} for m in state["modules"]],
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
    }
