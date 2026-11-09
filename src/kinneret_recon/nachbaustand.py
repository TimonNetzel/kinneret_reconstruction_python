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
                 "gamma_prior_grid"),
     "commit": "7d3d0f6852dbdf869b6e479d4eeffe80faaf2038"},
    {"path": Path("docs/thesis_research/app_reproduction_2026_11_02_tf_missing/reproduction_state.json"),
     "sha256": "7b7bf1f449a1f9b4c71f44cff6b3b37912d1fe50263eae80a85b3ba8b1137195",
     "modules": ("age_depth", "age_selection", "final_pdf", "cdf_discrepancy", "tf_preparation", "taxon_pruning",
                 "gamma_prior_grid", "tf_missing_priors"),
     "commit": "0a3dcdb8fde7292f6b9c1596dbade5cc7c455e34"},
    {"path": Path("docs/thesis_research/app_reproduction_2026_11_02_sampler_prep/reproduction_state.json"),
     "sha256": "61c6f23142df9c40fad9d271803eb6dc0a2dce4082ee77c5058895d0f5110ccb",
     "modules": ("age_depth", "age_selection", "final_pdf", "cdf_discrepancy", "tf_preparation", "taxon_pruning",
                 "gamma_prior_grid", "tf_missing_priors", "sampler_preparation"),
     "commit": "235f79784d01f0a613a0d3f76f6fd690c245ab21"},
    {"path": Path("docs/thesis_research/app_reproduction_2026_11_03_weights_stage/reproduction_state.json"),
     "sha256": "7eb9ac5e4eb616e17db8e0ddb5e9926c9abb600094a8e88219d151d92b2dc95d",
     "modules": ("age_depth", "age_selection", "final_pdf", "cdf_discrepancy", "tf_preparation", "taxon_pruning",
                 "gamma_prior_grid", "tf_missing_priors", "sampler_preparation", "weights_stage"),
     "commit": "3104f3c9e9a7870138abcc301bdd4f108c009e73"},
    {"path": Path("docs/thesis_research/app_reproduction_2026_11_03_dirichlet_ratio/reproduction_state.json"),
     "sha256": "cf1b77f22bddb4deca27cebe9a1a98a1a9541d786fc6078e007b43f8f7d2bfea",
     "modules": ("age_depth", "age_selection", "final_pdf", "cdf_discrepancy", "tf_preparation", "taxon_pruning",
                 "gamma_prior_grid", "tf_missing_priors", "sampler_preparation", "weights_stage", "dirichlet_ratio"),
     "commit": "cfd0ad8f1c0cb3f06909f50f21cc1e9a65592a68"},
    {"path": Path("docs/thesis_research/app_reproduction_2026_11_08_weighted_median/reproduction_state.json"),
     "sha256": "4d25a3f69f73f5a75340c5e43d4934e94f3faad90609a4f40d9e98bcb4f8c9c6",
     "modules": ("age_depth", "age_selection", "final_pdf", "cdf_discrepancy", "tf_preparation", "taxon_pruning",
                 "gamma_prior_grid", "tf_missing_priors", "sampler_preparation", "weights_stage", "dirichlet_ratio",
                 "weighted_median"),
     "commit": "416d547e67ebd349eab2508f87b12977d2b6ef4d"},
    {"path": Path("docs/thesis_research/app_reproduction_2026_11_08_r_crosscheck_six/reproduction_state.json"),
     "sha256": "50fe8b0a714b4dd7326353ab86e543d3b6650c9e6ae7d21115171c64a96791c2",
     "modules": ("age_depth", "age_selection", "final_pdf", "cdf_discrepancy", "tf_preparation", "taxon_pruning",
                 "gamma_prior_grid", "tf_missing_priors", "sampler_preparation", "weights_stage", "dirichlet_ratio",
                 "weighted_median"),
     "commit": "43733532ac20f183e5ee99631fc4e9fe3d357957"},
    {"path": Path("docs/thesis_research/app_reproduction_2026_11_08_reconstruction_climate/reproduction_state.json"),
     "sha256": "329e6a8b2ada52e22be1a12272853091ca4a6c29894581167ae02b0898877f6a",
     "modules": ("age_depth", "age_selection", "final_pdf", "cdf_discrepancy", "tf_preparation", "taxon_pruning",
                 "gamma_prior_grid", "tf_missing_priors", "sampler_preparation", "weights_stage", "dirichlet_ratio",
                 "weighted_median", "reconstruction_climate")},
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
    gex, mex, sex = state.get("gamma_example"), state.get("missing_example"), state.get("sampler_example")
    wex, dex, rtc = state.get("weights_example"), state.get("dirichlet_example"), state.get("runtime_check")
    medx, six, kex = state.get("median_example"), state.get("six_check"), state.get("climate_example")
    paths = ([s["path"].as_posix() for s in STATES[:-1]] + [m["package"] for m in state["modules"]]
             + [state["chain_check"]["package"], contract["package"], contract["spec"]]
             + [e[k] for e in (ex, pex, gex, mex, sex, wex, dex, medx, kex) if e
                for k in ("fixture", "result", "rational", "arithmetic")]
             + ([medx["crosscheck"]] if medx else []) + ([kex["crosscheck"], kex["binding"]] if kex else [])
             + ([six[k] for k in ("package", "result", "causes", "provider")] if six else [])
             + ([rtc[k] for k in ("package", "crosscheck", "dirichlet_crosscheck", "site_binding")] if rtc else []))
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
        "missing_example": _missing_example(root, mex, url) if mex else None,
        "sampler_example": _sampler_example(root, sex, url) if sex else None,
        "weights_example": _weights_example(root, wex, url) if wex else None,
        "dirichlet_example": _dirichlet_example(root, dex, url) if dex else None,
        "median_example": _median_example(root, medx, url) if medx else None,
        "six_check": _six_check(root, six, url) if six else None,
        "climate_example": _climate_example(root, kex, url) if kex else None,
        "runtime_check": _runtime_check(root, rtc, url) if rtc else None,
    }


def _runtime_check(root: Path, rtc: dict, url) -> dict:
    """Stored R 3.5.0 cross checks and the Kinneret slot; numbers only from the pinned result files."""
    _pinned(root, rtc["package"], rtc["package_sha256"])
    cross = json.loads(_pinned(root, rtc["crosscheck"], rtc["crosscheck_sha256"]))
    dirichlet = json.loads(_pinned(root, rtc["dirichlet_crosscheck"], rtc["dirichlet_crosscheck_sha256"]))
    sites = json.loads(_pinned(root, rtc["site_binding"], rtc["site_binding_sha256"]))
    if (cross["historical_reproduction_ready"] or dirichlet["historical_reproduction_ready"]
            or sites["historical_reproduction_ready"] or not cross["all_unflagged_values_bit_equal"]
            or cross["packages_attached"] or cross["random_state_used"]):
        raise ValueError("Nachbaustand: R cross check claims a release, packages or random state")
    return {
        "url": url(rtc["package"]), "new_tests": rtc["new_tests"],
        "r_version": cross["R"]["version"], "checks": cross["checks"],
        "modules": len(cross["original_lines"]),
        "dirichlet_values": dirichlet["values_compared"], "dirichlet_bit_equal": dirichlet["bit_equal"],
        "dirichlet_max_ulps": dirichlet["max_ulps"], "laplacesdemon": dirichlet["runtime"]["LaplacesDemon"],
        "dirichlet_parameter_ulps": {name: row["parameter_ratio"]["ulps"][0]
                                     for name, row in dirichlet["ratio_cases"].items()},
        "kinneret_slot": sites["kinneret_slot"],
    }


def _climate_example(root: Path, kex: dict, url) -> dict:
    """Stored reconstruction_climate cases; they must equal the reference and the R record, and stay marked as dormant."""
    fixture = json.loads(_pinned(root, kex["fixture"], kex["fixture_sha256"]))
    result = json.loads(_pinned(root, kex["result"], kex["result_sha256"]))
    rational = json.loads(_pinned(root, kex["rational"], kex["rational_sha256"]))
    check = json.loads(_pinned(root, kex["arithmetic"], kex["arithmetic_sha256"]))
    cross = json.loads(_pinned(root, kex["crosscheck"], kex["crosscheck_sha256"]))
    bound = json.loads(_pinned(root, kex["binding"], kex["binding_sha256"]))
    if (fixture["historical_reproduction_ready"] or fixture["selected_final_variant"] is not None
            or result["historical_reproduction_ready"] or check["historical_reproduction_ready"]
            or check["final_variant_selected"] or check["R_execution"] or cross["historical_reproduction_ready"]
            or bound["historical_reproduction_ready"] or bound["reached_by_bound_run"]
            or check["kind"] != "synthetic_fragment_check_not_historical_reproduction"):
        raise ValueError("Nachbaustand: climate example claims historical inputs, a final variant or an active run path")
    cases = []
    for name in kex["cases"]:
        stored, exact, r, case = result["cases"][name], rational["cases"][name], cross["cases"][name], fixture["cases"][name]
        for key in ("tann", "pann"):
            if stored[key + "_kind"] != exact[key]["kind"] or (exact[key]["value"] is not None and stored[key] != exact[key]["value"]) \
                    or r[key]["module"] != stored[key]:
                raise ValueError("Nachbaustand: stored climate example differs from the reference or the R record")
        cases.append({"name": name, "recent": case["recent"], "first_row": case["core"][0], "tann": stored["tann_decimal"],
                      "tann_kind": stored["tann_kind"], "pann": stored["pann_decimal"], "pann_kind": stored["pann_kind"],
                      "ulps": max(r["tann"]["ulps"], r["pann"]["ulps"])})
    naive = check["naive_counterexamples"]
    return {"url": url(kex["result"].rsplit("/", 1)[0] + "/README.md"), "cases": cases,
            "row_mean": naive["row_mean_instead_of_first_row"]["row_mean"],
            "expectation": naive["expectation_instead_of_median"]["expectation"],
            "r": {"values": cross["values_compared"], "bit_equal": cross["bit_equal"], "max_ulps": cross["max_ulps"]},
            "caller": bound["callers"][0], "r_files_searched": bound["r_files_searched"],
            "sha256": {k: kex[k + "_sha256"] for k in ("result", "rational")}}


def _six_check(root: Path, six: dict, url) -> dict:
    """Stored R 3.5.0 results of the six older modules; differences stay visible, only release claims are rejected."""
    _pinned(root, six["package"], six["package_sha256"])
    result = json.loads(_pinned(root, six["result"], six["result_sha256"]))
    causes = json.loads(_pinned(root, six["causes"], six["causes_sha256"]))
    provider = json.loads(_pinned(root, six["provider"], six["provider_sha256"]))
    if (result["historical_reproduction_ready"] or causes["historical_reproduction_ready"]
            or provider["historical_reproduction_ready"] or result["random_state_used"]):
        raise ValueError("Nachbaustand: six-module R check claims a release or random state")
    rows = []
    for key, module_id in six["modules"].items():
        row = result["modules"][key]
        counted = [s for k, s in row["keys"].items() if not k.startswith("grid_")]
        rows.append({"id": module_id, "values": sum(s["values"] for s in counted),
                     "bit_equal": sum(s["bit_equal"] for s in counted), "max_ulps": row["max_ulps"],
                     "fixture_values": row["fixture_values"], "fixture_bit_equal": row["fixture_bit_equal"]})
    return {"url": url(six["package"]), "new_tests": six["new_tests"], "r_version": result["runtime"]["R"],
            "matrixstats": result["runtime"]["matrixStats"], "modules": rows,
            "values": sum(r["values"] for r in rows), "bit_equal": sum(r["bit_equal"] for r in rows),
            "fixture_values": sum(r["fixture_values"] for r in rows),
            "fixture_bit_equal": sum(r["fixture_bit_equal"] for r in rows), "max_ulps": result["max_ulps"],
            "special_mismatch": result["special_mismatch"], "cause_values": causes["values"],
            "cause_bit_equal": causes["emulation_bit_equal"],
            "front_exporters": provider["exporters_in_front_of_matrixStats"]}


def _median_example(root: Path, medx: dict, url) -> dict:
    """Stored weightedMedian cases and the artificial TF mask; values must equal the reference and the R record."""
    fixture = json.loads(_pinned(root, medx["fixture"], medx["fixture_sha256"]))
    result = json.loads(_pinned(root, medx["result"], medx["result_sha256"]))
    rational = json.loads(_pinned(root, medx["rational"], medx["rational_sha256"]))
    check = json.loads(_pinned(root, medx["arithmetic"], medx["arithmetic_sha256"]))
    cross = json.loads(_pinned(root, medx["crosscheck"], medx["crosscheck_sha256"]))
    if (fixture["historical_reproduction_ready"] or fixture["selected_final_variant"] is not None
            or result["historical_reproduction_ready"] or check["historical_reproduction_ready"]
            or check["final_variant_selected"] or check["R_execution"] or cross["historical_reproduction_ready"]
            or check["kind"] != "synthetic_fragment_check_not_historical_reproduction"):
        raise ValueError("Nachbaustand: median example claims historical inputs or a final variant")
    cases = []
    for name in medx["cases"]:
        stored, exact, r = result["cases"][name], rational["cases"][name], cross["cases"][name]
        if (stored["kind"] != exact["kind"] or (exact["value"] is not None and stored["value"] != exact["value"])
                or r["module"] != stored["value"]):
            raise ValueError("Nachbaustand: stored median example differs from the reference or the R record")
        case = fixture["cases"][name]
        cases.append({"name": name, "x": case["x"], "w": case["w"], "kind": stored["kind"], "value": stored["decimal"],
                      "path": stored["path"], "ulps": r["ulps"]})
    tf = fixture["tf_case"]
    which = result["tf_case"]["which_tf_missing"]
    if which != tf["hand_which_tf_missing"] or which != cross["which_tf_missing_R"] or not cross["which_tf_missing_equal"]:
        raise ValueError("Nachbaustand: stored mask differs from the hand value or the R record")
    rows = [{"taxon": name.rsplit(" ", 1)[-1], "weights": [str(v) for v in row], "x1_kind": a["kind"],
             "x1": a["decimal"], "x2_kind": b["kind"], "x2": b["decimal"], "removed": i + 1 in which}
            for i, (name, row, a, b) in enumerate(zip(tf["taxa"], tf["tf_x1"], result["tf_case"]["x1"],
                                                       result["tf_case"]["x2"]))]
    naive = check["naive_counterexamples"]
    return {"url": url(medx["result"].rsplit("/", 1)[0] + "/README.md"), "cases": cases, "rows": rows, "which": which,
            "masks": check["masks_on_tf_case"], "stable_sort": naive["ties_trace_stable_sort"]["stable_sort"],
            "true_median": naive["inf_odd_true_median"]["true_median"],
            "r": {"matrixstats": cross["runtime"]["matrixStats"], "values": cross["values_compared"],
                  "bit_equal": cross["bit_equal"], "max_ulps": cross["max_ulps"]},
            "sha256": {k: medx[k + "_sha256"] for k in ("result", "rational")}}


def _dirichlet_example(root: Path, dex: dict, url) -> dict:
    """Stored values of artificial Dirichlet-ratio cases; they must equal the independent reference exactly."""
    fixture = json.loads(_pinned(root, dex["fixture"], dex["fixture_sha256"]))
    result = json.loads(_pinned(root, dex["result"], dex["result_sha256"]))
    rational = json.loads(_pinned(root, dex["rational"], dex["rational_sha256"]))
    check = json.loads(_pinned(root, dex["arithmetic"], dex["arithmetic_sha256"]))
    if (fixture["historical_reproduction_ready"] or fixture["selected_final_variant"] is not None
            or result["historical_reproduction_ready"] or check["historical_reproduction_ready"]
            or check["final_variant_selected"] or check["R_execution"]
            or check["kind"] != "synthetic_fragment_check_not_historical_reproduction"):
        raise ValueError("Nachbaustand: Dirichlet example claims historical inputs or a final variant")
    cases = []
    for name in dex["cases"]:
        stored, exact = result["ratio_cases"][name], rational["cases"][name]
        if any(stored[k] != exact[k] for k in ("weights_ratio", "proposal_ratio", "parameter_ratio_before_na",
                                               "parameter_ratio", "densities")):
            raise ValueError("Nachbaustand: stored Dirichlet example differs from the reference")
        case = fixture["ratio_cases"][name]
        cases.append({"name": name, "num_taxa": case["num_taxa"], "weights_ratio": stored["weights_ratio"],
                      "proposal_ratio": stored["proposal_ratio"], "parameter_ratio": stored["parameter_ratio"],
                      "na_replaced": stored["na_replaced"],
                      "normalisation_skipped": any(stored["normalisation_skipped"]),
                      "spread_differs": check["naive_counterexamples"][name]["proposal_with_spread_differs"]})
    two = check["hand_expectations"]["two_taxa_closed_form"]
    return {"url": url(dex["result"].rsplit("/", 1)[0] + "/README.md"), "cases": cases,
            "closed_form_ulps": [two["weights_ratio_ulps_to_closed_form"], two["proposal_ratio_ulps_to_closed_form"],
                                 two["parameter_ratio_ulps_to_closed_form"]],
            "sha256": {k: dex[k + "_sha256"] for k in ("result", "rational")}}


def _hand_matches(text, value) -> bool:
    """A hand entry of the weights fixture against a stored value; null is not written by hand."""
    if text is None:
        return True
    if text in ("NA", "NaN", "Inf", "-Inf", "-0"):
        return value == text
    return not isinstance(value, str) and value == float(Fraction(text))


def _weights_example(root: Path, wex: dict, url) -> dict:
    """Stored values of artificial weights-stage cases; they must equal the independent reference exactly."""
    fixture = json.loads(_pinned(root, wex["fixture"], wex["fixture_sha256"]))
    result = json.loads(_pinned(root, wex["result"], wex["result_sha256"]))
    rational = json.loads(_pinned(root, wex["rational"], wex["rational_sha256"]))
    check = json.loads(_pinned(root, wex["arithmetic"], wex["arithmetic_sha256"]))
    if (fixture["historical_reproduction_ready"] or fixture["selected_final_variant"] is not None
            or fixture["dirichlet_package"] is not None or result["historical_reproduction_ready"]
            or check["historical_reproduction_ready"] or check["final_variant_selected"] or check["R_execution"]
            or check["dirichlet_package"] is not None
            or check["kind"] != "synthetic_fragment_check_not_historical_reproduction"):
        raise ValueError("Nachbaustand: weights example claims historical inputs, a final variant or a package")
    names = [wex["matrix_case"], *wex["special_cases"], wex["reset_case"], *wex["scale_cases"]]
    for name in names:
        stored, exact, hand = result["cases"][name], rational["cases"][name], fixture["cases"][name]["hand"]
        flat = sum(stored["taxa_appear_core_temp"], [])
        if (any(stored[k] != exact[k] for k in exact if k in stored)
                or not all(map(_hand_matches, hand["result"], flat))
                or not all(map(_hand_matches, hand["row_sums"], stored["row_sums"]))
                or stored["z"] != hand["z"] or stored["zero_sum_rows"] != hand["zero_rows"]):
            raise ValueError("Nachbaustand: stored weights example differs from the reference or the hand value")
    base, scaled = (result["cases"][n] for n in wex["scale_cases"])
    reset = result["cases"][wex["reset_case"]]
    if (base["taxa_appear_core_temp"] != scaled["taxa_appear_core_temp"]
            or Fraction(reset["row_sums_exact"][1]) != 1 - Fraction(1, 2**54) or reset["row_sums"][1] != 1.0
            or not check["naive_counterexamples"][wex["reset_case"]]["unweighted_differs"]):
        raise ValueError("Nachbaustand: weights example is internally inconsistent")
    m, case = result["cases"][wex["matrix_case"]], fixture["cases"][wex["matrix_case"]]
    ncol = len(m["z"])
    hand_rows = [case["hand"]["result"][i * ncol:(i + 1) * ncol] for i in range(len(m["row_sums"]))]
    special = []
    for name in wex["special_cases"]:
        stored = result["cases"][name]
        flat = sum(stored["taxa_appear_core_temp"], [])
        special.append({"nrow": len(stored["row_sums"]), "ncol": len(stored["z"]),
                        "zero_rows": len(stored["zero_sum_rows"]), "minus_inf_rows": stored["row_sums"].count("-Inf"),
                        "na": flat.count("NA"), "nan": flat.count("NaN"),
                        "inf": flat.count("Inf") + flat.count("-Inf"), "neg_zero": flat.count("-0"),
                        "sums_exact": stored["row_sums_exact"]})
    return {
        "url": url(wex["result"].rsplit("/", 1)[0] + "/README.md"),
        "state5_case": case["state5_case"], "weights_exact": case["new_weights"],
        "rows": m["taxa_appear_core_temp"], "rows_exact": hand_rows,
        "sums": m["row_sums"], "sums_exact": m["row_sums_exact"], "independent": m["rows_accumulator_independent"],
        "z": m["z"], "positions": len(m["products"]),
        "misaligned": check["naive_counterexamples"][wex["matrix_case"]]["column_major_misaligned_weight_positions"],
        "special": special,
        "reset_weights": fixture["cases"][wex["reset_case"]]["new_weights"],
        "reset_first": reset["taxa_appear_core_temp"][0][0], "reset_third": reset["taxa_appear_core_temp"][1][0],
        "scale_base": fixture["cases"][wex["scale_cases"][0]]["new_weights"],
        "scale_weights": fixture["cases"][wex["scale_cases"][1]]["new_weights"],
        "sha256": {k: wex[k + "_sha256"] for k in ("result", "rational")},
    }


def _sampler_example(root: Path, sex: dict, url) -> dict:
    """Stored values of artificial sampler-preparation cases; they must equal the independent reference exactly."""
    fixture = json.loads(_pinned(root, sex["fixture"], sex["fixture_sha256"]))
    result = json.loads(_pinned(root, sex["result"], sex["result_sha256"]))
    rational = json.loads(_pinned(root, sex["rational"], sex["rational_sha256"]))
    check = json.loads(_pinned(root, sex["arithmetic"], sex["arithmetic_sha256"]))
    if (fixture["historical_reproduction_ready"] or fixture["selected_final_variant"] is not None
            or result["historical_reproduction_ready"] or check["historical_reproduction_ready"]
            or check["final_variant_selected"] or check["R_execution"]
            or check["kind"] != "synthetic_fragment_check_not_historical_reproduction"):
        raise ValueError("Nachbaustand: sampler example claims historical inputs or a final variant")
    names = [sex["matrix_case"], *sex["special_cases"], sex["gibbs_case"], sex["weights_case"]]
    for name in names:
        stored, exact, hand = result["cases"][name], rational["cases"][name], fixture["cases"][name]["hand"]
        if (any(stored[k] != exact[k] for k in exact if k in stored)
                or (hand["values"] is not None and hand["values"] != exact["values_exact"])
                or stored["how_many_taxa"] != hand["how_many_taxa"]):
            raise ValueError("Nachbaustand: stored sampler example differs from the reference or the hand value")
    m, mx = result["cases"][sex["matrix_case"]], rational["cases"][sex["matrix_case"]]
    ncol = m["ncol_taxa_array"]
    rows = lambda v: [v[i * ncol:(i + 1) * ncol] for i in range(m["nrow_taxa_array"])]
    special = []
    for name in sex["special_cases"]:
        stored, exact = result["cases"][name], rational["cases"][name]
        special.append({"nrow": stored["nrow_taxa_array"], "ncol": stored["ncol_taxa_array"],
                        "column_max": stored["column_max"],
                        "na": stored["values"].count("NA"), "nan": stored["values"].count("NaN"),
                        "minus_inf": stored["values"].count("-Inf"), "means_exact": exact["taxa_means_exact"]})
    gibbs, weights = result["cases"][sex["gibbs_case"]], result["cases"][sex["weights_case"]]
    if gibbs["values"] != weights["values"] or gibbs["parameter_stage"] == weights["parameter_stage"]:
        raise ValueError("Nachbaustand: Gibbs and weights cases must differ only in lines 53-59")
    return {
        "url": url(sex["result"].rsplit("/", 1)[0] + "/README.md"),
        "raw": fixture["cases"][sex["matrix_case"]]["taxa_appear_core"], "column_max": m["column_max"],
        "normalized": rows(m["values"]), "normalized_exact": rows(mx["values_exact"]),
        "values_exact": mx["values_exact"], "means_exact": mx["taxa_means_exact"],
        "means_independent": m["taxa_means_accumulator_independent"],
        "misaligned": check["naive_counterexamples"][sex["matrix_case"]]["misaligned_weight_positions"],
        "special": special,
        "num_taxa": weights["ncol_taxa_array"], "how_many_weights": weights["how_many_taxa"],
        "how_many_gibbs": gibbs["how_many_taxa"], "jeffreys": weights["jeffreys_prior"][0],
        "sha256": {k: sex[k + "_sha256"] for k in ("result", "rational")},
    }


def _missing_example(root: Path, mex: dict, url) -> dict:
    """Stored values of artificial which.tf.missing cases; they must equal the independent reference exactly."""
    fixture = json.loads(_pinned(root, mex["fixture"], mex["fixture_sha256"]))
    result = json.loads(_pinned(root, mex["result"], mex["result_sha256"]))
    rational = json.loads(_pinned(root, mex["rational"], mex["rational_sha256"]))
    check = json.loads(_pinned(root, mex["arithmetic"], mex["arithmetic_sha256"]))
    if (fixture["historical_reproduction_ready"] or fixture["selected_final_variant"] is not None
            or result["historical_reproduction_ready"] or check["historical_reproduction_ready"]
            or check["final_variant_selected"] or check["R_execution"]
            or check["kind"] != "synthetic_fragment_check_not_historical_reproduction"):
        raise ValueError("Nachbaustand: missing-TF example claims historical inputs or a final variant")
    keys = ("which_tf_missing", "taxa", "num_taxa", "num_taxa_line", "nr_taxa_sd", "phi", "stale_phi1", "stale_phi2")
    cases = []
    for name in [*mex["cases"], mex["empty_case"]]:
        case, stored, exact = fixture["cases"][name], result["cases"][name], rational["cases"][name]
        naive = check["naive_counterexamples"][name]
        if any(stored[k] != exact[k] for k in keys) or stored["nr_taxa_sd"] != case["hand"]["nr_taxa_sd"]:
            raise ValueError("Nachbaustand: stored missing-TF example differs from the reference or the hand value")
        cases.append({"taxa": len(case["taxa"]), "missing": stored["which_tf_missing"],
                      "x2_only": [i + 1 for i, v in enumerate(case["x2_mean"]) if v is None
                                  and i + 1 not in stored["which_tf_missing"]],
                      "kept": [n.rsplit(" ", 1)[-1] for n in stored["taxa"]], "num_taxa": stored["num_taxa"],
                      "line": stored["num_taxa_line"], "sd": stored["nr_taxa_sd"],
                      "sd_half_up": naive["round_half_up_sd"], "half_up_differs": naive["round_half_up_differs"],
                      "phi": stored["phi"], "stale_phi1": stored["stale_phi1"], "stale_phi2": stored["stale_phi2"]})
    return {
        "url": url(mex["result"].rsplit("/", 1)[0] + "/README.md"), "cases": cases[:-1], "empty": cases[-1],
        "half_up_differs_for": check["round_ties"]["half_up_differs_for"],
        "sha256": {k: mex[k + "_sha256"] for k in ("result", "rational")},
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
