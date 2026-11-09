"""Pinned dissertation reproduction states: checked submodules, open inputs and strictly synthetic examples."""
import hashlib
import json
import shutil
from pathlib import Path
import pytest
from kinneret_recon import forschungsstand, nachbaustand
from kinneret_recon.app_bauen import baue_eine_datei, baue_ordner

ROOT = Path(__file__).resolve().parents[1]
STATE = json.loads((ROOT / nachbaustand.STATE).read_text(encoding="utf-8"))
PACKET = "docs/thesis_research/literal_tf_preparation_2026_11_02/"
PRUNING = "docs/thesis_research/literal_taxon_pruning_2026_11_02/"
GAMMA = "docs/thesis_research/literal_gamma_prior_grid_2026_11_02/"
MISSING = "docs/thesis_research/literal_tf_missing_priors_2026_11_02/"
SAMPLER = "docs/thesis_research/literal_sampler_preparation_2026_11_02/"
WEIGHTS = "docs/thesis_research/literal_weights_stage_2026_11_03/"
FROZEN = nachbaustand.STATES[0]["path"].as_posix()
FROZEN2 = nachbaustand.STATES[1]["path"].as_posix()
FROZEN3 = nachbaustand.STATES[2]["path"].as_posix()
FROZEN4 = nachbaustand.STATES[3]["path"].as_posix()
FROZEN5 = nachbaustand.STATES[4]["path"].as_posix()
FROZEN6 = nachbaustand.STATES[5]["path"].as_posix()
DIRICHLET = "docs/thesis_research/literal_dirichlet_ratio_2026_11_03/"
RUNTIME = "docs/thesis_research/r_runtime_crosscheck_2026_11_03/"
MEDIAN = "docs/thesis_research/literal_weighted_median_2026_11_08/"
FROZEN7 = nachbaustand.STATES[6]["path"].as_posix()
FROZEN8 = nachbaustand.STATES[7]["path"].as_posix()
SIX = "docs/thesis_research/r_crosscheck_six_modules_2026_11_08/"
FROZEN9 = nachbaustand.STATES[8]["path"].as_posix()
FROZEN10 = nachbaustand.STATES[9]["path"].as_posix()
COR = "docs/thesis_research/literal_local_cor_2026_11_09/"
RECONST = "docs/thesis_research/literal_reconst_predictant_2026_11_09/"
FROZEN11 = nachbaustand.STATES[10]["path"].as_posix()
WT = "docs/thesis_research/literal_wt_power_2026_11_09/"
FROZEN12 = nachbaustand.STATES[11]["path"].as_posix()
TFR = "docs/thesis_research/literal_ratios_tfs_2026_11_09/"
BIND = "docs/thesis_research/literal_rtruncnorm_binding_2026_11_09/"
FROZEN13 = nachbaustand.STATES[12]["path"].as_posix()
FROZEN14 = nachbaustand.STATES[13]["path"].as_posix()
IC = "docs/thesis_research/kinneret_input_contract_2026_11_09/"
CLIMATE = "docs/thesis_research/literal_reconstruction_climate_2026_11_08/"


def _kopie(tmp_path: Path) -> Path:
    """Copy exactly the pinned files so single bytes can be changed or removed."""
    ex, contract, pex, gex, mex, sex, wex = (STATE["synthetic_example"], STATE["contract"], STATE["pruning_example"],
                                             STATE["gamma_example"], STATE["missing_example"],
                                             STATE["sampler_example"], STATE["weights_example"])
    dex, rtc, medx, six = STATE["dirichlet_example"], STATE["runtime_check"], STATE["median_example"], STATE["six_check"]
    kex, cex, rex, wtx = STATE["climate_example"], STATE["cor_example"], STATE["reconst_example"], STATE["wt_example"]
    tfx, rtb, icx = STATE["ratios_example"], STATE["rtruncnorm_binding"], STATE["input_contract"]
    for rel in ([s["path"].as_posix() for s in nachbaustand.STATES] + [STATE["chain_check"]["package"],
                contract["package"], contract["spec"]] + [m["package"] for m in STATE["modules"]]
                + [e[k] for e in (ex, pex, gex, mex, sex, wex, dex, medx, kex, cex, rex, wtx, tfx) for k in ("fixture", "result", "rational",
                                                                                              "arithmetic")]
                + [kex["crosscheck"], kex["binding"], cex["crosscheck"], cex["binding"], rex["crosscheck"], rex["binding"], rex["measurement"],
                   wtx["crosscheck"], wtx["binding"], wtx["masking"], tfx["crosscheck"], tfx["binding"], rtb["package"], icx["package"], icx["contract"]]
                + [medx["crosscheck"]] + [six[k] for k in ("package", "result", "causes", "provider")]
                + [rtc[k] for k in ("package", "crosscheck", "dirichlet_crosscheck", "site_binding")]):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, tmp_path / rel)
    return tmp_path


def test_newest_state_binds_seventeen_modules_eleven_open_bundles_and_no_release():
    d = nachbaustand.app_daten(ROOT)
    assert d["available"] and d["status"] == "submodules_checked_original_case_open"
    assert [m["id"] for m in d["modules"]] == list(nachbaustand.MODULE_IDS)
    assert d["modules"][-1]["id"] == "ratios_tfs" and not d["modules"][-1]["conditional"]
    assert [m["id"] for m in d["modules"] if m["conditional"]] == ["taxon_pruning", "reconstruction_climate", "wt_power"]
    assert d["tests_passed"] == 2143 == (d["input_contract"]["new_tests"] + d["rtruncnorm_binding"]["new_tests"] + d["six_check"]["new_tests"] + sum(m["new_tests"] for m in d["modules"]) + d["chain_check"]["new_tests"]
                                        + d["contract"]["tests"] + d["runtime_check"]["new_tests"])
    assert [r["id"] for r in d["requirements"]] == list(nachbaustand.REQUIREMENT_IDS)
    assert {r["status"] for r in d["requirements"]} == {"missing", "conflicting"}
    assert sum(r["status"] == "conflicting" for r in d["requirements"]) == 4
    assert all("/blob/" + STATE["source_commit"] + "/" in m["url"] for m in d["modules"])
    text = json.dumps(d)
    assert ".local/" not in text and "C:\\" not in text and "historical_reproduction_ready" not in text


def test_earlier_states_stay_frozen_pinned_and_listed():
    first, second, third, fourth, fifth, sixth, seventh, eighth, ninth = nachbaustand.STATES[:9]
    assert ninth["commit"] == "43733532ac20f183e5ee99631fc4e9fe3d357957"
    assert ninth["sha256"] == "50fe8b0a714b4dd7326353ab86e543d3b6650c9e6ae7d21115171c64a96791c2"
    ninth_state = json.loads((ROOT / ninth["path"]).read_text(encoding="utf-8"))
    assert (ninth_state["tests"]["passed"], len(ninth_state["modules"])) == (975, 12)
    assert ninth_state["supersedes"] == {"state": FROZEN8, "state_sha256": eighth["sha256"]}
    tenth = nachbaustand.STATES[9]
    assert tenth["commit"] == "437f6f4baeed4a086a2d5a80c7abc9e420a353de"
    assert tenth["sha256"] == "329e6a8b2ada52e22be1a12272853091ca4a6c29894581167ae02b0898877f6a"
    tenth_state = json.loads((ROOT / tenth["path"]).read_text(encoding="utf-8"))
    assert (tenth_state["tests"]["passed"], len(tenth_state["modules"])) == (1037, 13)
    assert tenth_state["supersedes"] == {"state": FROZEN9, "state_sha256": ninth["sha256"]}
    eleventh = nachbaustand.STATES[10]
    assert eleventh["commit"] == "38775e1fc959b936fb22b7dd6dc88bef69fb7eb1"
    assert eleventh["sha256"] == "cbfaf04f6bdbf36eb8fff76dac92e2a1a4d19df15f6d607a56e510dcd71e73af"
    eleventh_state = json.loads((ROOT / eleventh["path"]).read_text(encoding="utf-8"))
    assert (eleventh_state["tests"]["passed"], len(eleventh_state["modules"])) == (1172, 14)
    assert eleventh_state["supersedes"] == {"state": FROZEN10, "state_sha256": tenth["sha256"]}
    assert STATE["modules"][:14] == eleventh_state["modules"] and STATE["cor_example"] == eleventh_state["cor_example"]
    twelfth = nachbaustand.STATES[11]
    assert twelfth["commit"] == "9afab2e968bd79aba0ae375a8e871aa889528900"
    assert twelfth["sha256"] == "ce676ea4122cf237b02d05ad15f2fb7f9663b41b06292aad4666982bbe24bf0d"
    twelfth_state = json.loads((ROOT / twelfth["path"]).read_text(encoding="utf-8"))
    assert (twelfth_state["tests"]["passed"], len(twelfth_state["modules"])) == (1402, 15)
    assert twelfth_state["supersedes"] == {"state": FROZEN11, "state_sha256": eleventh["sha256"]}
    thirteenth = nachbaustand.STATES[12]
    assert thirteenth["commit"] == "52a6accd23e075d7173aa8aa0bc494e4cd503f01"
    assert thirteenth["sha256"] == "d84fe644553c48325b53dbf29a7ac7fbe8fcfa040d53e6d8f5358ac45851f945"
    thirteenth_state = json.loads((ROOT / thirteenth["path"]).read_text(encoding="utf-8"))
    assert (thirteenth_state["tests"]["passed"], len(thirteenth_state["modules"])) == (1614, 16)
    assert thirteenth_state["supersedes"] == {"state": FROZEN12, "state_sha256": twelfth["sha256"]}
    fourteenth = nachbaustand.STATES[13]
    assert fourteenth["commit"] == "ad0882a52793edceebaaf33a3d687013bac8b89c"
    assert fourteenth["sha256"] == "a45d172c117805bc5d7e75a25ae22606650552aa0a8a73f04a4e706f76226230"
    fourteenth_state = json.loads((ROOT / fourteenth["path"]).read_text(encoding="utf-8"))
    assert (fourteenth_state["tests"]["passed"], len(fourteenth_state["modules"])) == (1858, 17)
    assert fourteenth_state["supersedes"] == {"state": FROZEN13, "state_sha256": thirteenth["sha256"]}
    assert STATE["supersedes"] == {"state": FROZEN14, "state_sha256": fourteenth["sha256"]}
    assert STATE["modules"] == fourteenth_state["modules"] and STATE["ratios_example"] == fourteenth_state["ratios_example"]
    assert STATE["modules"][:16] == thirteenth_state["modules"] and STATE["wt_example"] == thirteenth_state["wt_example"]
    assert STATE["modules"][:15] == twelfth_state["modules"] and STATE["reconst_example"] == twelfth_state["reconst_example"]
    assert STATE["modules"][:12] == ninth_state["modules"] and STATE["six_check"] == ninth_state["six_check"]
    assert STATE["modules"][:13] == tenth_state["modules"] and STATE["climate_example"] == tenth_state["climate_example"]
    assert eighth["commit"] == "416d547e67ebd349eab2508f87b12977d2b6ef4d"
    assert eighth["sha256"] == "4d25a3f69f73f5a75340c5e43d4934e94f3faad90609a4f40d9e98bcb4f8c9c6"
    eighth_state = json.loads((ROOT / eighth["path"]).read_text(encoding="utf-8"))
    assert (eighth_state["tests"]["passed"], len(eighth_state["modules"])) == (934, 12)
    assert eighth_state["supersedes"] == {"state": FROZEN7, "state_sha256": seventh["sha256"]}
    assert STATE["modules"][:12] == eighth_state["modules"] and STATE["median_example"] == eighth_state["median_example"]
    assert (first["sha256"], second["sha256"], third["sha256"], fourth["sha256"], fifth["sha256"],
            sixth["sha256"]) == (
        "c358b2c3382f568a94d070054ba1dee4b230c617506c9a5015e2cc4506f4e63e",
        "3656316de4211500184f1a1c772bf07c87e3bd4b2d210ed6316e70c5a8ca0784",
        "43edfcb67a64826ce72611c93be33490463e59566934393aeff952bd217fbf4f",
        "7b7bf1f449a1f9b4c71f44cff6b3b37912d1fe50263eae80a85b3ba8b1137195",
        "61c6f23142df9c40fad9d271803eb6dc0a2dce4082ee77c5058895d0f5110ccb",
        "7eb9ac5e4eb616e17db8e0ddb5e9926c9abb600094a8e88219d151d92b2dc95d")
    assert third["commit"] == "7d3d0f6852dbdf869b6e479d4eeffe80faaf2038"
    assert fourth["commit"] == "0a3dcdb8fde7292f6b9c1596dbade5cc7c455e34"
    assert fifth["commit"] == "235f79784d01f0a613a0d3f76f6fd690c245ab21"
    assert sixth["commit"] == "3104f3c9e9a7870138abcc301bdd4f108c009e73"
    assert seventh["commit"] == "cfd0ad8f1c0cb3f06909f50f21cc1e9a65592a68"
    assert seventh["sha256"] == "cf1b77f22bddb4deca27cebe9a1a98a1a9541d786fc6078e007b43f8f7d2bfea"
    seventh_state = json.loads((ROOT / seventh["path"]).read_text(encoding="utf-8"))
    assert (seventh_state["tests"]["passed"], len(seventh_state["modules"])) == (752, 11)
    assert seventh_state["supersedes"] == {"state": FROZEN6, "state_sha256": sixth["sha256"]}
    assert [m["package_sha256"] for m in seventh_state["modules"]] == [m["package_sha256"] for m in STATE["modules"][:11]]
    assert all(STATE[k] == seventh_state[k] for k in ("chain_check", "contract", "synthetic_example", "weights_example",
                                                      "dirichlet_example", "runtime_check"))
    for frozen in (first, second, third, fourth, fifth, sixth, seventh, eighth, ninth, tenth, eleventh, twelfth):
        assert hashlib.sha256((ROOT / frozen["path"]).read_bytes()).hexdigest() == frozen["sha256"]
    old, prev, mid, fourth_state, last, sixth_state = (json.loads((ROOT / f["path"]).read_text(encoding="utf-8"))
                                                       for f in (first, second, third, fourth, fifth, sixth))
    assert (sixth_state["tests"]["passed"], len(sixth_state["modules"])) == (689, 10)
    assert sixth_state["supersedes"] == {"state": FROZEN5, "state_sha256": fifth["sha256"]}
    assert [m["package_sha256"] for m in sixth_state["modules"]] == [m["package_sha256"] for m in STATE["modules"][:10]]
    assert all(STATE[k] == sixth_state[k] for k in ("chain_check", "contract", "synthetic_example", "weights_example",
                                                    "sampler_example"))
    assert (old["tests"]["passed"], len(old["modules"])) == (251, 5)
    assert (prev["tests"]["passed"], len(prev["modules"])) == (297, 6)
    assert (mid["tests"]["passed"], len(mid["modules"])) == (352, 7)
    assert (fourth_state["tests"]["passed"], len(fourth_state["modules"])) == (437, 8)
    assert (last["tests"]["passed"], len(last["modules"])) == (555, 9)
    assert prev["supersedes"] == {"state": FROZEN, "state_sha256": first["sha256"]}
    assert mid["supersedes"] == {"state": FROZEN2, "state_sha256": second["sha256"]}
    assert fourth_state["supersedes"] == {"state": FROZEN3, "state_sha256": third["sha256"]}
    assert last["supersedes"] == {"state": FROZEN4, "state_sha256": fourth["sha256"]}
    assert nachbaustand.app_daten(ROOT)["earlier_states"] == [
        {"date": date, "state_sha256": f["sha256"], "modules": n, "tests_passed": tests,
         "url": nachbaustand.REPO_URL + f["commit"] + "/" + f["path"].as_posix()}
        for f, n, tests, date in ((first, 5, 251, "2026-11-02"), (second, 6, 297, "2026-11-02"),
                                  (third, 7, 352, "2026-11-02"), (fourth, 8, 437, "2026-11-02"),
                                  (fifth, 9, 555, "2026-11-02"), (sixth, 10, 689, "2026-11-03"),
                                  (seventh, 11, 752, "2026-11-03"), (eighth, 12, 934, "2026-11-08"),
                                  (ninth, 12, 975, "2026-11-08"), (tenth, 13, 1037, "2026-11-08"),
                                  (eleventh, 14, 1172, "2026-11-09"), (twelfth, 15, 1402, "2026-11-09"), (nachbaustand.STATES[12], 16, 1614, "2026-11-09"),
                                  (nachbaustand.STATES[13], 17, 1858, "2026-11-09"))]
    assert [m["package_sha256"] for m in last["modules"]] == [m["package_sha256"] for m in STATE["modules"][:9]]
    assert all(STATE[k] == last[k] for k in ("chain_check", "contract", "synthetic_example", "pruning_example",
                                             "gamma_example", "missing_example", "sampler_example"))


def test_weights_example_is_taken_unchanged_from_the_pinned_packet():
    w = nachbaustand.app_daten(ROOT)["weights_example"]
    stored = json.loads((ROOT / WEIGHTS / "synthetic_result.json").read_text(encoding="utf-8"))["cases"]
    case = stored[STATE["weights_example"]["matrix_case"]]
    assert w["state5_case"] == STATE["sampler_example"]["matrix_case"] == "rowwise_flatten_three_by_four"
    assert w["weights_exact"] == ["1/8", "3/8", "1/4", "1/4"] and w["rows"] == case["taxa_appear_core_temp"]
    assert w["rows_exact"] == [[None] * 4, [None] * 4, ["1/29", "12/29", "8/29", "8/29"]]
    assert w["sums_exact"][2] == "29/32" and w["independent"] == [False, False, True]
    assert (w["z"], w["positions"], w["misaligned"]) == ([1, 2, 3, 4], 12, 8)
    assert [(x["nrow"], x["ncol"], x["zero_rows"], x["minus_inf_rows"], x["na"], x["nan"], x["inf"], x["neg_zero"])
            for x in w["special"]] == [(3, 3, 0, 0, 2, 1, 0, 0), (4, 3, 3, 0, 2, 5, 2, 0), (3, 3, 0, 1, 1, 2, 0, 2)]
    assert w["special"][0]["sums_exact"] == ["1/4", "3/8", "5/8"] and w["special"][2]["sums_exact"][1] == "-Inf"
    assert (w["reset_weights"], w["reset_first"], w["reset_third"]) == (["1/3"] * 3, 0.5714285714285715,
                                                                         0.3333333333333333)
    assert (w["scale_base"], w["scale_weights"]) == (["1/2", "1/4", "1/4"], ["1", "1/2", "1/2"])
    assert w["url"].endswith(WEIGHTS + "README.md") and "/blob/" + STATE["source_commit"] + "/" in w["url"]
    for key, sha in w["sha256"].items():
        assert hashlib.sha256((ROOT / STATE["weights_example"][key]).read_bytes()).hexdigest() == sha


def test_weights_example_rejects_a_dirichlet_package_claim(tmp_path, monkeypatch):
    root = _kopie(tmp_path)
    rel = STATE["weights_example"]["arithmetic"]
    content = json.dumps(dict(json.loads((root / rel).read_text(encoding="utf-8")), dirichlet_package="x")).encode()
    (root / rel).write_bytes(content)
    state = json.loads(json.dumps(STATE))
    state["weights_example"]["arithmetic_sha256"] = hashlib.sha256(content).hexdigest()
    state_bytes = json.dumps(state).encode()
    (root / nachbaustand.STATE).write_bytes(state_bytes)
    monkeypatch.setattr(nachbaustand, "STATE_SHA256", hashlib.sha256(state_bytes).hexdigest())
    with pytest.raises(ValueError, match="package"):
        nachbaustand.app_daten(root)


def test_dirichlet_example_is_taken_unchanged_from_the_pinned_packet():
    d = nachbaustand.app_daten(ROOT)["dirichlet_example"]
    stored = json.loads((ROOT / DIRICHLET / "synthetic_result.json").read_text(encoding="utf-8"))["ratio_cases"]
    assert [c["name"] for c in d["cases"]] == STATE["dirichlet_example"]["cases"]
    for c in d["cases"]:
        s = stored[c["name"]]
        assert (c["weights_ratio"], c["proposal_ratio"], c["parameter_ratio"]) == (
            s["weights_ratio"], s["proposal_ratio"], s["parameter_ratio"])
    assert [c["num_taxa"] for c in d["cases"]] == [2, 3, 3, 3]
    assert [c["na_replaced"] for c in d["cases"]] == [False, False, True, False]
    assert [c["normalisation_skipped"] for c in d["cases"]] == [False, True, False, False]
    assert all(c["spread_differs"] for c in d["cases"]) and d["closed_form_ulps"] == [0, 1, 0]
    assert d["cases"][2]["weights_ratio"] == "Inf" and d["cases"][2]["parameter_ratio"] == 0.0
    assert d["url"].endswith(DIRICHLET + "README.md") and "/blob/" + STATE["source_commit"] + "/" in d["url"]


def test_median_example_is_taken_unchanged_from_the_pinned_packet():
    m = nachbaustand.app_daten(ROOT)["median_example"]
    stored = json.loads((ROOT / MEDIAN / "synthetic_result.json").read_text(encoding="utf-8"))["cases"]
    assert [c["name"] for c in m["cases"]] == STATE["median_example"]["cases"]
    for c in m["cases"]:
        assert (c["kind"], c["value"], c["path"]) == (stored[c["name"]]["kind"], stored[c["name"]]["decimal"],
                                                       stored[c["name"]]["path"])
    assert [c["kind"] for c in m["cases"]] == ["number", "number", "NA", "NA", "NA", "number", "NA", "number", "number"]
    assert all(c["ulps"] == 0 for c in m["cases"]) and m["cases"][0]["value"] == 2.5
    assert m["which"] == [2, 3, 5] and [r["removed"] for r in m["rows"]] == [False, True, True, False, True]
    assert [r["taxon"] for r in m["rows"]] == list("ABCDE") and m["rows"][3]["x2_kind"] == "NA"
    assert m["masks"]["sum_is_zero_only"] == [2] and m["masks"]["x1_or_x2"] == [2, 3, 4, 5]
    assert (m["stable_sort"], m["true_median"]) == (1.4, 2.0)
    assert m["r"] == {"matrixstats": "0.56.0", "values": 439, "bit_equal": 439, "max_ulps": 0}
    assert m["url"].endswith(MEDIAN + "README.md") and "/blob/" + STATE["source_commit"] + "/" in m["url"]
    for key, sha in m["sha256"].items():
        assert hashlib.sha256((ROOT / STATE["median_example"][key]).read_bytes()).hexdigest() == sha


def test_median_example_rejects_a_stored_value_that_left_the_reference(tmp_path, monkeypatch):
    root = _kopie(tmp_path)
    rel = STATE["median_example"]["result"]
    stored = json.loads((root / rel).read_text(encoding="utf-8"))
    stored["cases"]["uniform_even"]["value"] = "0x1.8p+1"
    content = json.dumps(stored).encode()
    (root / rel).write_bytes(content)
    state = json.loads(json.dumps(STATE))
    state["median_example"]["result_sha256"] = hashlib.sha256(content).hexdigest()
    state_bytes = json.dumps(state).encode()
    (root / nachbaustand.STATE).write_bytes(state_bytes)
    monkeypatch.setattr(nachbaustand, "STATE_SHA256", hashlib.sha256(state_bytes).hexdigest())
    with pytest.raises(ValueError, match="median example differs"):
        nachbaustand.app_daten(root)


def test_median_example_rejects_a_release_claim(tmp_path, monkeypatch):
    root = _kopie(tmp_path)
    rel = STATE["median_example"]["arithmetic"]
    content = json.dumps(dict(json.loads((root / rel).read_text(encoding="utf-8")),
                              historical_reproduction_ready=True)).encode()
    (root / rel).write_bytes(content)
    state = json.loads(json.dumps(STATE))
    state["median_example"]["arithmetic_sha256"] = hashlib.sha256(content).hexdigest()
    state_bytes = json.dumps(state).encode()
    (root / nachbaustand.STATE).write_bytes(state_bytes)
    monkeypatch.setattr(nachbaustand, "STATE_SHA256", hashlib.sha256(state_bytes).hexdigest())
    with pytest.raises(ValueError, match="median example claims"):
        nachbaustand.app_daten(root)


def test_climate_example_is_taken_unchanged_and_stays_marked_as_dormant():
    k = nachbaustand.app_daten(ROOT)["climate_example"]
    stored = json.loads((ROOT / CLIMATE / "synthetic_result.json").read_text(encoding="utf-8"))["cases"]
    assert [c["name"] for c in k["cases"]] == STATE["climate_example"]["cases"] and len(k["cases"]) == 8
    for c in k["cases"]:
        assert (c["tann"], c["pann"]) == (stored[c["name"]]["tann_decimal"], stored[c["name"]]["pann_decimal"])
    assert [c["tann_kind"] for c in k["cases"]] == ["number", "number", "number", "NA", "NA", "NA", "NA", "number"]
    assert k["cases"][4]["pann_kind"] == "number" and all(c["ulps"] == 0 for c in k["cases"])
    assert k["caller"]["file"].endswith("mcmc_sampling_v1.4.R") and k["caller"]["line"] == 79 and k["r_files_searched"] == 540
    assert k["r"] == {"values": 5510, "bit_equal": 5510, "max_ulps": 0}
    assert k["url"].endswith(CLIMATE + "README.md") and "/blob/" + STATE["source_commit"] + "/" in k["url"]


def test_climate_example_rejects_an_active_run_claim(tmp_path, monkeypatch):
    root = _kopie(tmp_path)
    rel = STATE["climate_example"]["binding"]
    content = json.dumps(dict(json.loads((root / rel).read_text(encoding="utf-8")), reached_by_bound_run=True)).encode()
    (root / rel).write_bytes(content)
    state = json.loads(json.dumps(STATE))
    state["climate_example"]["binding_sha256"] = hashlib.sha256(content).hexdigest()
    state_bytes = json.dumps(state).encode()
    (root / nachbaustand.STATE).write_bytes(state_bytes)
    monkeypatch.setattr(nachbaustand, "STATE_SHA256", hashlib.sha256(state_bytes).hexdigest())
    with pytest.raises(ValueError, match="active run path"):
        nachbaustand.app_daten(root)


def test_cor_example_is_taken_unchanged_and_marks_line_1999_as_unreached():
    c = nachbaustand.app_daten(ROOT)["cor_example"]
    stored = json.loads((ROOT / COR / "synthetic_result.json").read_text(encoding="utf-8"))["cases"]
    assert [f["name"] for f in c["cases"]] == STATE["cor_example"]["cases"] and len(c["cases"]) == 8
    for f in c["cases"]:
        assert (f["kind"], f["value"], f["replaced"], f["post_value"]) == (stored[f["name"]]["kind"], stored[f["name"]]["decimal"], stored[f["name"]]["replaced"], stored[f["name"]]["post_value"])
        assert f["ulps"] == 0
    assert [f["kind"] for f in c["cases"]] == ["number", "number", "number", "number", "NA", "NaN", "number", "NaN"]
    assert [f["replaced"] for f in c["cases"]] == [False, True, False, True, True, True, False, True]
    assert c["cases"][1]["post_value"] == 1e-10 * 1e-10 and c["r_files_searched"] == 540
    assert c["r"] == {"values": 447, "bit_equal": 447, "max_ulps": 0} and c["double_sum"] == {"compared": 300, "changed": 199}
    assert c["url"].endswith(COR + "README.md") and "/blob/" + STATE["source_commit"] + "/" in c["url"]


def test_reconst_example_is_taken_unchanged_and_marks_the_mask_and_the_accumulator():
    r = nachbaustand.app_daten(ROOT)["reconst_example"]
    stored = json.loads((ROOT / RECONST / "synthetic_result.json").read_text(encoding="utf-8"))["cases"]
    assert [f["name"] for f in r["cases"]] == STATE["reconst_example"]["cases"] and len(r["cases"]) == 8
    assert all(f["ulps"] == 0 for f in r["cases"]) and r["r"] == {"values": 3009, "bit_equal": 3009, "max_ulps": 0}
    for f in r["cases"]:
        assert ([x["kind"] for x in f["tann"]], f["path"], f["stop"]) == (["number" if t not in ("NA", "NaN", "Inf", "-Inf") else t for t in stored[f["name"]]["tann"]], stored[f["name"]]["path"], stored[f["name"]].get("stop"))
    assert [f["selected_dim"] for f in r["cases"]] == [[4, 3], None, None, [0, 3], [2, 3], [2, 2], [2, 2], [2, 2]]
    assert [f["reconst_dim_R"] for f in r["cases"]] == ["4x2", "1x2", "3x2", "0x2", "2x2", "2x2", "2x2", "2x2"]
    assert [f["tann"][0]["kind"] for f in r["cases"] if f["tann"]] == ["number", "number", "number", "number", "NA", "NaN", "NaN"]
    assert r["cases"][0]["tann"][0]["value"] == 17.5 and r["cases"][3]["stop"] == "1825" and r["nan_combinations"] == 1296
    assert r["long_double"] == {"compared": 2032, "changed": 367} and r["paths"]["simple_matprod"] == 146
    assert r["url"].endswith(RECONST + "README.md") and "/blob/" + STATE["source_commit"] + "/" in r["url"]


def _reconst_changed(root, monkeypatch, key, mutate):
    rel = STATE["reconst_example"][key]
    data = json.loads((root / rel).read_text(encoding="utf-8"))
    mutate(data)
    content = json.dumps(data).encode()
    (root / rel).write_bytes(content)
    state = json.loads(json.dumps(STATE))
    state["reconst_example"][key + "_sha256"] = hashlib.sha256(content).hexdigest()
    state_bytes = json.dumps(state).encode()
    (root / nachbaustand.STATE).write_bytes(state_bytes)
    monkeypatch.setattr(nachbaustand, "STATE_SHA256", hashlib.sha256(state_bytes).hexdigest())


def test_reconst_example_rejects_a_stored_value_that_left_the_reference(tmp_path, monkeypatch):
    root = _kopie(tmp_path)
    _reconst_changed(root, monkeypatch, "result", lambda d: d["cases"]["hand_all_rows"]["tann"].__setitem__(0, "0x1.2p+4"))
    with pytest.raises(ValueError, match="reconst example differs"):
        nachbaustand.app_daten(root)


def test_reconst_example_rejects_a_long_double_claim_for_the_three_loop_path(tmp_path, monkeypatch):
    root = _kopie(tmp_path)
    _reconst_changed(root, monkeypatch, "binding", lambda d: d["r_matprod_source"]["array_c_lines"].__setitem__("simple_matprod_accumulator", "LDOUBLE"))
    with pytest.raises(ValueError, match="long double path"):
        nachbaustand.app_daten(root)


def test_reconst_example_rejects_a_claim_that_the_holocene_mask_is_effective(tmp_path, monkeypatch):
    root = _kopie(tmp_path)
    _reconst_changed(root, monkeypatch, "binding", lambda d: d["mask"].__setitem__("effect", "selects Holocene rows"))
    with pytest.raises(ValueError, match="effective mask"):
        nachbaustand.app_daten(root)


def test_reconst_example_rejects_an_na_mismatch_with_R(tmp_path, monkeypatch):
    root = _kopie(tmp_path)
    _reconst_changed(root, monkeypatch, "crosscheck", lambda d: d.__setitem__("special_mismatch", 1))
    with pytest.raises(ValueError, match="NA, NaN or stop mismatch"):
        nachbaustand.app_daten(root)


def test_reconst_example_rejects_a_release_claim(tmp_path, monkeypatch):
    root = _kopie(tmp_path)
    _reconst_changed(root, monkeypatch, "measurement", lambda d: d.__setitem__("historical_reproduction_ready", True))
    with pytest.raises(ValueError, match="reconst example claims"):
        nachbaustand.app_daten(root)


def _bound_changed(root, monkeypatch, **changes):
    rel = STATE["cor_example"]["binding"]
    bound = json.loads((root / rel).read_text(encoding="utf-8"))
    bound.update(changes)
    content = json.dumps(bound).encode()
    (root / rel).write_bytes(content)
    state = json.loads(json.dumps(STATE))
    state["cor_example"]["binding_sha256"] = hashlib.sha256(content).hexdigest()
    state_bytes = json.dumps(state).encode()
    (root / nachbaustand.STATE).write_bytes(state_bytes)
    monkeypatch.setattr(nachbaustand, "STATE_SHA256", hashlib.sha256(state_bytes).hexdigest())


def test_cor_example_rejects_a_claim_that_line_1999_is_reached(tmp_path, monkeypatch):
    root = _kopie(tmp_path)
    readers = json.loads((root / STATE["cor_example"]["binding"]).read_text(encoding="utf-8"))["readers"]
    for r in readers:
        if str(r["line"]) == "1999":
            r["kinneret"] = "reached"
    _bound_changed(root, monkeypatch, readers=readers)
    with pytest.raises(ValueError, match="reachable line 1999"):
        nachbaustand.app_daten(root)


def test_cor_example_rejects_a_multiple_flag_that_does_not_equal_sources_core(tmp_path, monkeypatch):
    root = _kopie(tmp_path)
    _bound_changed(root, monkeypatch, multiple_equals_sources_core_for_kinneret=False)
    with pytest.raises(ValueError, match="reachable line 1999"):
        nachbaustand.app_daten(root)


def test_six_check_reports_stored_r_results_and_keeps_the_differences():
    s = nachbaustand.app_daten(ROOT)["six_check"]
    assert (s["r_version"], s["matrixstats"], s["new_tests"]) == ("R version 3.5.0 (2018-04-23)", "0.56.0", 41)
    assert [r["id"] for r in s["modules"]] == ["age_depth", "age_selection", "final_pdf", "cdf_discrepancy",
                                               "tf_preparation", "taxon_pruning"]
    assert [(r["values"], r["bit_equal"], r["max_ulps"]) for r in s["modules"]] == [
        (4827, 4358, 3), (6535, 5303, 2), (15407, 10807, 4), (2827, 2797, 2), (2580, 2000, 3), (9790, 9653, 3)]
    assert (s["values"], s["bit_equal"], s["max_ulps"], s["special_mismatch"]) == (41966, 34918, 4, 0)
    assert s["fixture_values"] == s["fixture_bit_equal"] == 7064 and s["cause_values"] == s["cause_bit_equal"] == 24822
    assert s["front_exporters"] == [] and s["url"].endswith(SIX + "README.md")
    assert "/blob/" + STATE["source_commit"] + "/" in s["url"]


def test_six_check_rejects_a_release_claim(tmp_path, monkeypatch):
    root = _kopie(tmp_path)
    rel = STATE["six_check"]["causes"]
    content = json.dumps(dict(json.loads((root / rel).read_text(encoding="utf-8")), historical_reproduction_ready=True)).encode()
    (root / rel).write_bytes(content)
    state = json.loads(json.dumps(STATE))
    state["six_check"]["causes_sha256"] = hashlib.sha256(content).hexdigest()
    state_bytes = json.dumps(state).encode()
    (root / nachbaustand.STATE).write_bytes(state_bytes)
    monkeypatch.setattr(nachbaustand, "STATE_SHA256", hashlib.sha256(state_bytes).hexdigest())
    with pytest.raises(ValueError, match="six-module R check claims"):
        nachbaustand.app_daten(root)


def test_runtime_check_reports_stored_r_results_only():
    r = nachbaustand.app_daten(ROOT)["runtime_check"]
    assert (r["r_version"], r["checks"], r["modules"], r["new_tests"]) == ("R version 3.5.0 (2018-04-23)", 389, 4, 13)
    assert (r["dirichlet_values"], r["dirichlet_bit_equal"], r["dirichlet_max_ulps"], r["laplacesdemon"]) == (
        62, 59, 4, "16.1.6")
    assert r["kinneret_slot"] == 3 and r["dirichlet_parameter_ulps"]["inexact_weights"] == 4
    assert sum(r["dirichlet_parameter_ulps"].values()) == 4
    assert r["url"].endswith(RUNTIME + "README.md")


def test_sampler_example_is_taken_unchanged_from_the_pinned_packet():
    s = nachbaustand.app_daten(ROOT)["sampler_example"]
    stored = json.loads((ROOT / SAMPLER / "synthetic_result.json").read_text(encoding="utf-8"))["cases"]
    case = stored[STATE["sampler_example"]["matrix_case"]]
    assert s["raw"] == [[2, 1, 4, 3], [4, 3, 2, 6], [1, 6, 8, 9]] and s["column_max"] == [4.0, 6.0, 8.0, 9.0]
    assert s["normalized_exact"] == [["1/2", "1/6", "1/2", "1/3"], ["1", "1/2", "1/4", "2/3"], ["1/4", "1", "1", "1"]]
    assert sum(s["normalized"], []) == case["values"] and s["values_exact"][:4] == ["1/2", "1/6", "1/2", "1/3"]
    assert s["means_exact"] == ["7/3", "10/3", "14/3", "6"] and s["means_independent"] == [False, False, False, True]
    assert s["misaligned"] == {"rowwise": 0, "column_major": 8}
    assert [(x["nrow"], x["ncol"], x["na"], x["nan"], x["minus_inf"]) for x in s["special"]] == [
        (4, 3, 3, 0, 0), (3, 3, 0, 3, 0), (3, 3, 1, 1, 1)]
    assert [x["column_max"] for x in s["special"]] == [[4.0, 4.0, 8.0], [4.0, 0.0, 4.0], [0.0, -1.0, 3.0]]
    assert s["special"][0]["means_exact"] == ["8/3", "8/3", "4"]
    assert (s["num_taxa"], s["how_many_weights"], s["how_many_gibbs"], s["jeffreys"]) == (5, 5, 2, 0.5)
    assert s["url"].endswith(SAMPLER + "README.md") and "/blob/" + STATE["source_commit"] + "/" in s["url"]


def test_missing_tf_example_is_taken_unchanged_from_the_pinned_packet():
    m = nachbaustand.app_daten(ROOT)["missing_example"]
    stored = json.loads((ROOT / MISSING / "synthetic_result.json").read_text(encoding="utf-8"))["cases"]
    assert [(c["taxa"], c["missing"], c["num_taxa"], c["line"]) for c in m["cases"]] == [
        (6, [2, 5], 4, 1678), (4, [1, 4], 2, 1678), (12, [3, 9], 10, 1678), (4, [2], 3, 1678)]
    assert [c["sd"] for c in m["cases"]] == [stored[n]["nr_taxa_sd"] for n in STATE["missing_example"]["cases"]]
    assert [(c["sd"], c["sd_half_up"], c["half_up_differs"]) for c in m["cases"]] == [
        (1, 1, False), (0, 1, True), (2, 3, True), (1, 1, False)]
    assert m["cases"][0]["kept"] == ["A", "C", "D", "F"] and m["cases"][3]["x2_only"] == [3]
    assert [(c["phi"], c["stale_phi1"]) for c in m["cases"]] == [(2.0, 3.0), (1.0, 2.0), (5.0, 6.0), (1.5, 2.0)]
    assert (m["empty"]["missing"], m["empty"]["num_taxa"], m["empty"]["line"], m["empty"]["sd"]) == ([], 5, 1435, 1)
    assert m["half_up_differs_for"] == [2, 10, 18, 26, 34]
    assert m["url"].endswith(MISSING + "README.md") and "/blob/" + STATE["source_commit"] + "/" in m["url"]


def test_gamma_example_is_taken_unchanged_from_the_pinned_packet():
    g = nachbaustand.app_daten(ROOT)["gamma_example"]
    stored = json.loads((ROOT / GAMMA / "synthetic_result.json").read_text(encoding="utf-8"))["cases"]
    assert [(c["mean"], c["sd"], c["r_index"]) for c in g["cases"]] == [(500, 100, 250), (500, 200, 62),
                                                                       (315, 220, 20), (2500, 10, 1)]
    assert [c["shape"] for c in g["cases"]] == [stored[n]["shape"] for n in STATE["gamma_example"]["cases"]]
    assert g["cases"][0]["shape"] == 25.000000000000004 and g["cases"][3]["closed_form"] == 62500.0
    assert [c["nearest_differs"] for c in g["cases"]] == [False, True, True, True]
    assert g["chain"] == {"default_shape": 25.000000000000004, "sd_pann": 200.0, "final_shape": 6.2,
                          "final_rate": 0.012449899597988732, "rate_after_recent_start": 0.0124}
    assert (g["grid_length"], g["grid_last"]) == (500000, 50000.0)
    assert g["url"].endswith(GAMMA + "README.md") and "/blob/" + STATE["source_commit"] + "/" in g["url"]


def test_pruning_example_is_taken_unchanged_from_the_pinned_packet():
    p = nachbaustand.app_daten(ROOT)["pruning_example"]
    stored = json.loads((ROOT / PRUNING / "synthetic_result.json").read_text(encoding="utf-8"))["cases"]
    case = stored["short_core_descending_colon"]
    assert p["zero_counts"] == [0, 2, 5, 10, 10, 3] and p["rows"] == 20
    assert p["null_ratio_exact"] == ["0", "1/10", "1/4", "1/2", "1/2", "3/20"]
    assert p["null_ratio"] == case["null_ratio"] and p["removed_taxa"] == case["removed_taxa"] == [3, 4]
    assert p["branch"] == "rarest_15" and p["metro_hastings"] is False
    assert p["naive_differs"] == ["naive_ge095_removed", "naive_rank15_removed"]
    assert p["url"].endswith(PRUNING + "README.md") and "/blob/" + STATE["source_commit"] + "/" in p["url"]


def test_example_values_are_taken_unchanged_from_the_pinned_tf_packet():
    ex = nachbaustand.app_daten(ROOT)["example"]
    result = json.loads((ROOT / PACKET / "synthetic_result.json").read_text(encoding="utf-8"))
    check = json.loads((ROOT / PACKET / "arithmetic_check.json").read_text(encoding="utf-8"))
    assert ex["raw"] == [[2, 0, 0, 0], [0, 0, 1, 2]] and ex["keep"] == [True, True, False, True]
    assert ex["normalized"] == result["cases"]["source_brief"]["normalized"]
    assert ex["masked"] == result["cases"]["source_brief"]["masked"]
    assert ex["row_masses_exact"] == ["1", "2/3"] and ex["masked_exact"][1] == ["0", "0", "0", "2/3"]
    assert ex["chain_pdf"] == result["prepared_chain"]["pdf"] and ex["draw_ids_R_1_based"] == [1, 3]
    assert ex["max_rational_error"] == check["maximum_absolute_rational_error"] <= ex["tolerance"]
    assert ex["renormalization_pdf_difference"] == check["post_mask_renormalization_chain_PDF_max_difference"]
    for key, sha in ex["sha256"].items():
        assert hashlib.sha256((ROOT / STATE["synthetic_example"][key]).read_bytes()).hexdigest() == sha


def test_missing_state_or_evidence_is_visibly_unavailable(tmp_path):
    assert nachbaustand.app_daten(tmp_path) == {"schema_version": 1, "available": False,
                                                "missing": [nachbaustand.STATE.as_posix()]}
    root = _kopie(tmp_path)
    (root / PACKET / "arithmetic_check.json").unlink()
    d = nachbaustand.app_daten(root)
    assert not d["available"] and d["missing"] == [PACKET + "arithmetic_check.json"]
    (root / FROZEN).unlink()
    assert nachbaustand.app_daten(root)["missing"] == [FROZEN, PACKET + "arithmetic_check.json"]
    (root / GAMMA / "synthetic_fixture.json").unlink()
    assert nachbaustand.app_daten(root)["missing"][-1] == GAMMA + "synthetic_fixture.json"
    (root / MISSING / "arithmetic_check.json").unlink()
    assert nachbaustand.app_daten(root)["missing"][-1] == MISSING + "arithmetic_check.json"
    (root / SAMPLER / "rational_expectations.json").unlink()
    assert nachbaustand.app_daten(root)["missing"][-1] == SAMPLER + "rational_expectations.json"
    (root / WEIGHTS / "synthetic_fixture.json").unlink()
    assert nachbaustand.app_daten(root)["missing"][-1] == WEIGHTS + "synthetic_fixture.json"
    (root / DIRICHLET / "arithmetic_check.json").unlink()
    assert nachbaustand.app_daten(root)["missing"][-1] == DIRICHLET + "arithmetic_check.json"
    (root / RUNTIME / "site_binding.json").unlink()
    assert nachbaustand.app_daten(root)["missing"][-1] == RUNTIME + "site_binding.json"
    (root / MEDIAN / "r_crosscheck_result.json").unlink()
    assert MEDIAN + "r_crosscheck_result.json" in nachbaustand.app_daten(root)["missing"]
    (root / SIX / "difference_causes.json").unlink()
    assert SIX + "difference_causes.json" in nachbaustand.app_daten(root)["missing"]
    (root / CLIMATE / "source_binding.json").unlink()
    assert CLIMATE + "source_binding.json" in nachbaustand.app_daten(root)["missing"]
    (root / RECONST / "nan_rule_measurement.json").unlink()
    assert RECONST + "nan_rule_measurement.json" in nachbaustand.app_daten(root)["missing"]
    (root / WT / "fft_masking.json").unlink()
    assert WT + "fft_masking.json" in nachbaustand.app_daten(root)["missing"]


def test_wt_example_is_taken_unchanged_and_shows_the_deviation_from_R():
    w = nachbaustand.app_daten(ROOT)["wt_example"]
    stored = json.loads((ROOT / WT / "synthetic_result.json").read_text(encoding="utf-8"))["cases"]
    assert [c["name"] for c in w["cases"]] == STATE["wt_example"]["cases"] and len(w["cases"]) == 8
    assert [c["kind"] for c in w["cases"]] == ["number"] * 5 + ["NA", "NaN", "mixed"]
    assert [c["dim"] for c in w["cases"]][:3] == [[4, 2], None, None] and w["cases"][4]["dim"] == [13, 6] and w["cases"][4]["n"] == 32
    assert w["cases"][0]["first"] == {"kind": "number", "value": float.fromhex(stored["identity_wb_two_columns"]["power"][0])}
    assert w["cases"][5]["first"] == {"kind": "NA", "value": None} and w["cases"][5]["ulps"] is None and w["cases"][7]["tokens_R"] == ["NA", "NaN"]
    d = w["deviation"]
    assert (d["centred"], d["power_values"], d["power_bit_equal"], d["power_max_ulps"], d["post_spec_ulps"]) == (22256, 39876, 5938, 606, 12)
    assert 0 < d["power_relative"] < 5e-15 and "not bit-identical" in w["backend"] and w["masking_versions"] == 17
    assert w["url"].endswith(WT + "README.md") and "/blob/" + STATE["source_commit"] + "/" in w["url"]


def _wt_changed(root, monkeypatch, key, mutate):
    rel = STATE["wt_example"][key]
    data = json.loads((root / rel).read_text(encoding="utf-8"))
    mutate(data)
    content = json.dumps(data).encode()
    (root / rel).write_bytes(content)
    state = json.loads(json.dumps(STATE))
    state["wt_example"][key + "_sha256"] = hashlib.sha256(content).hexdigest()
    state_bytes = json.dumps(state).encode()
    (root / nachbaustand.STATE).write_bytes(state_bytes)
    monkeypatch.setattr(nachbaustand, "STATE_SHA256", hashlib.sha256(state_bytes).hexdigest())


def test_wt_example_rejects_a_claim_of_bit_identity_with_R(tmp_path, monkeypatch):
    root = _kopie(tmp_path)
    _wt_changed(root, monkeypatch, "crosscheck", lambda d: d.__setitem__("bit_identical_with_R", True))
    with pytest.raises(ValueError, match="bit identity"):
        nachbaustand.app_daten(root)


def test_wt_example_rejects_a_backend_without_the_marking(tmp_path, monkeypatch):
    root = _kopie(tmp_path)
    _wt_changed(root, monkeypatch, "crosscheck", lambda d: d.__setitem__("backend", "numpy.fft"))
    with pytest.raises(ValueError, match="bit identity"):
        nachbaustand.app_daten(root)


def test_wt_example_rejects_a_settled_kinneret_path(tmp_path, monkeypatch):
    root = _kopie(tmp_path)
    _wt_changed(root, monkeypatch, "binding", lambda d: d["kinneret"].__setitem__("spectral", "reached"))
    with pytest.raises(ValueError, match="settled Kinneret"):
        nachbaustand.app_daten(root)


def test_wt_example_rejects_a_token_mismatch_with_R(tmp_path, monkeypatch):
    root = _kopie(tmp_path)
    _wt_changed(root, monkeypatch, "crosscheck", lambda d: d.__setitem__("token_mismatch", 1))
    with pytest.raises(ValueError, match="token mismatch"):
        nachbaustand.app_daten(root)


def test_wt_example_rejects_a_masking_package(tmp_path, monkeypatch):
    root = _kopie(tmp_path)
    _wt_changed(root, monkeypatch, "masking", lambda d: d.__setitem__("versions_masking_fft_or_mvfft", ["signal_0.7-7.tar.gz"]))
    with pytest.raises(ValueError, match="masking package"):
        nachbaustand.app_daten(root)


def test_wt_example_rejects_a_stored_value_that_left_the_reference(tmp_path, monkeypatch):
    root = _kopie(tmp_path)
    _wt_changed(root, monkeypatch, "result", lambda d: d["cases"]["identity_wb_two_columns"]["power"].__setitem__(0, "0x1.8p+4"))
    with pytest.raises(ValueError, match="differs from the reference"):
        nachbaustand.app_daten(root)


def test_wt_example_rejects_a_release_claim(tmp_path, monkeypatch):
    root = _kopie(tmp_path)
    _wt_changed(root, monkeypatch, "fixture", lambda d: d.__setitem__("historical_reproduction_ready", True))
    with pytest.raises(ValueError, match="wt_power example claims"):
        nachbaustand.app_daten(root)


@pytest.mark.parametrize("rel", [nachbaustand.STATE.as_posix(), PACKET + "synthetic_result.json",
                                 PACKET + "arithmetic_check.json", PACKET + "synthetic_fixture.json",
                                 "docs/thesis_research/kinneret_phase1_spec_2026_11_02/spec.json",
                                 "docs/thesis_research/literal_age_depth_2026_11_02/README.md",
                                 FROZEN, PRUNING + "README.md", PRUNING + "synthetic_result.json",
                                 PRUNING + "rational_expectations.json", FROZEN2, GAMMA + "README.md",
                                 GAMMA + "synthetic_result.json", GAMMA + "rational_expectations.json",
                                 GAMMA + "synthetic_fixture.json", GAMMA + "arithmetic_check.json", FROZEN3,
                                 MISSING + "README.md", MISSING + "synthetic_result.json",
                                 MISSING + "rational_expectations.json", MISSING + "synthetic_fixture.json",
                                 MISSING + "arithmetic_check.json", FROZEN4, SAMPLER + "README.md",
                                 SAMPLER + "synthetic_result.json", SAMPLER + "rational_expectations.json",
                                 SAMPLER + "synthetic_fixture.json", SAMPLER + "arithmetic_check.json", FROZEN5,
                                 WEIGHTS + "README.md", WEIGHTS + "synthetic_result.json",
                                 WEIGHTS + "rational_expectations.json", WEIGHTS + "synthetic_fixture.json",
                                 WEIGHTS + "arithmetic_check.json", FROZEN6, DIRICHLET + "README.md",
                                 DIRICHLET + "synthetic_result.json", DIRICHLET + "reference_expectations.json",
                                 DIRICHLET + "synthetic_fixture.json", DIRICHLET + "arithmetic_check.json",
                                 DIRICHLET + "r_crosscheck_result.json", RUNTIME + "README.md",
                                 RUNTIME + "crosscheck_result.json", RUNTIME + "site_binding.json", FROZEN7,
                                 MEDIAN + "README.md", MEDIAN + "synthetic_result.json",
                                 MEDIAN + "reference_expectations.json", MEDIAN + "synthetic_fixture.json",
                                 MEDIAN + "arithmetic_check.json", MEDIAN + "r_crosscheck_result.json", FROZEN8,
                                 SIX + "README.md", SIX + "six_crosscheck_result.json", SIX + "difference_causes.json",
                                 SIX + "provider_rowcumsums.json", FROZEN9, CLIMATE + "README.md",
                                 CLIMATE + "synthetic_result.json", CLIMATE + "reference_expectations.json",
                                 CLIMATE + "synthetic_fixture.json", CLIMATE + "arithmetic_check.json",
                                 CLIMATE + "r_crosscheck_result.json", CLIMATE + "source_binding.json", FROZEN10,
                                 COR + "README.md", COR + "synthetic_result.json", COR + "reference_expectations.json",
                                 COR + "synthetic_fixture.json", COR + "arithmetic_check.json", COR + "r_crosscheck_result.json",
                                 COR + "source_binding.json", FROZEN11, RECONST + "README.md", RECONST + "synthetic_result.json",
                                 RECONST + "reference_expectations.json", RECONST + "synthetic_fixture.json", RECONST + "arithmetic_check.json",
                                 RECONST + "r_crosscheck_result.json", RECONST + "source_binding.json", RECONST + "nan_rule_measurement.json",
                                 FROZEN12, WT + "README.md", WT + "synthetic_result.json", WT + "reference_expectations.json", WT + "synthetic_fixture.json",
                                 WT + "arithmetic_check.json", WT + "r_crosscheck_result.json", WT + "source_binding.json", WT + "fft_masking.json",
                                 FROZEN13, TFR + "README.md", TFR + "synthetic_result.json", TFR + "reference_expectations.json", TFR + "synthetic_fixture.json",
                                 TFR + "arithmetic_check.json", TFR + "r_crosscheck_result.json", TFR + "source_binding.json", BIND + "README.md", FROZEN14, IC + "README.md", IC + "input_contract.json"])
def test_changed_bytes_are_never_shown_as_checked(tmp_path, rel):
    root = _kopie(tmp_path)
    (root / rel).write_bytes((root / rel).read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        nachbaustand.app_daten(root)


def test_release_flag_is_rejected_even_with_a_matching_pin(tmp_path, monkeypatch):
    root = _kopie(tmp_path)
    state = dict(STATE, historical_reproduction_ready=True)
    content = json.dumps(state).encode()
    (root / nachbaustand.STATE).write_bytes(content)
    monkeypatch.setattr(nachbaustand, "STATE_SHA256", hashlib.sha256(content).hexdigest())
    with pytest.raises(ValueError, match="release flag"):
        nachbaustand.app_daten(root)


def test_reproduction_state_in_both_offline_formats_separate_from_snapshot(tmp_path):
    out = tmp_path / "app"
    baue_ordner(out, ROOT)
    html = baue_eine_datei(tmp_path / "single.html", ROOT).read_text(encoding="utf-8")
    prefix = lambda name: '(window.KR_DATEN=window.KR_DATEN||{})["' + name + '"]='
    for name, reader in (("nachbau", nachbaustand), ("forschung", forschungsstand)):
        js = (out / f"daten/{name}.js").read_text(encoding="utf-8")
        assert json.loads(js[len(prefix(name)):-2]) == reader.app_daten(ROOT)
        assert prefix(name) in html
    assert 'src="daten/nachbau.js"' in (out / "index.html").read_text(encoding="utf-8")
    assert 'src="daten/nachbau.js"' not in html
    assert "daten/nachbau.js" in (out / "sw.js").read_text()
    assert "Synthetisches Rechenbeispiel" in html and "Synthetic worked example" in html
    assert "Taxon-Pruning nach Nullanteil" in html and "Taxon pruning by zero share" in html
    assert "Gamma-Prior über ein Formraster" in html and "Gamma prior from a shape grid" in html
    assert "Entfernung fehlender TF und Taxonzahl-Priors" in html and "Removal of missing TFs" in html
    assert "Sampler-Vorbereitung aus Taxonzahl und Taxonmatrix" in html and "Sampler preparation from taxon" in html
    assert "Kern der Gewichtsstufe mit vorgegebenen Gewichten" in html and "Core of the weights stage" in html
    assert "Dirichlet-Quotient der Gewichtsstufe" in html and "Dirichlet ratio of the weights stage" in html
    assert "R-Gegenprobe" in html and "R cross check" in html
    assert "Gewichteter Median und NA-Maske" in html and "Weighted median and the NA mask" in html
    assert "R-Gegenprobe der sechs älteren Teilmodule" in html and "R cross check of the six older" in html
    assert "reconstruction_climate" in html and "Dormant source path" in html
    assert "Lokales cor: {n} Kunstfälle" in html and "Local cor: {n} artificial cases" in html
    assert "Rekonstruktion und Prädiktanten: {n} Kunstfälle" in html and "Reconstruction and predictants: {n} artificial cases" in html
    assert "wt_power: {n} Kunstfälle" in html and "wt_power: {n} artificial cases" in html
    assert "Transferquotienten: {n} Kunstfälle" in html and "Transfer ratios: {n} artificial cases" in html
    assert '"earlier_states"' in html and all(s["sha256"] in html for s in nachbaustand.STATES[:13])


def test_ratios_example_is_taken_unchanged_and_shows_the_deviation_from_R():
    w = nachbaustand.app_daten(ROOT)["ratios_example"]
    stored = json.loads((ROOT / TFR / "synthetic_result.json").read_text(encoding="utf-8"))["cases"]
    assert [c["name"] for c in w["cases"]] == STATE["ratios_example"]["cases"] and len(w["cases"]) == 8
    assert [c["kind"] for c in w["cases"]] == ["number"] * 3 + ["Inf", "NaN", "0"] + ["number"] * 2
    assert w["cases"][0]["taxa"] == 3 and w["cases"][0]["grid"] == 4 and w["cases"][6]["grid"] == 1000 and w["cases"][0]["ulps"] == 2
    assert w["cases"][0]["transfer"] == {"kind": "number", "value": float.fromhex(stored["three_taxa_quarters"]["transfer_ratio"])}
    assert w["cases"][3]["transfer"] == {"kind": "Inf", "value": None} and w["cases"][3]["ulps"] is None and w["cases"][4]["transfer_R"] == {"kind": "NaN", "value": None}
    d = w["deviation"]
    assert (d["stress_cases"], d["grid_values"], d["density_values"], d["density_bit_equal"], d["density_max_ulps"], d["transfer_max_ulps"]) == (120, 3406, 3406, 2533, 15, 35)
    assert 0 < d["density_relative"] < 5e-15 and "not bit-identical" in w["backend"] and w["rcpp_versions"] == 19
    assert w["url"].endswith(TFR + "README.md") and "/blob/" + STATE["source_commit"] + "/" in w["url"]


def test_the_rtruncnorm_binding_is_linked_and_counted():
    b = nachbaustand.app_daten(ROOT)["rtruncnorm_binding"]
    assert b["new_tests"] == 32 and b["url"].endswith(BIND + "README.md") and "/blob/" + STATE["source_commit"] + "/" in b["url"]


def _tfr_changed(root, monkeypatch, key, mutate):
    rel = STATE["ratios_example"][key]
    data = json.loads((root / rel).read_text(encoding="utf-8"))
    mutate(data)
    content = json.dumps(data).encode()
    (root / rel).write_bytes(content)
    state = json.loads(json.dumps(STATE))
    state["ratios_example"][key + "_sha256"] = hashlib.sha256(content).hexdigest()
    state_bytes = json.dumps(state).encode()
    (root / nachbaustand.STATE).write_bytes(state_bytes)
    monkeypatch.setattr(nachbaustand, "STATE_SHA256", hashlib.sha256(state_bytes).hexdigest())


def test_ratios_example_rejects_a_claim_of_bit_identity_with_R(tmp_path, monkeypatch):
    root = _kopie(tmp_path)
    _tfr_changed(root, monkeypatch, "crosscheck", lambda d: d.__setitem__("bit_identical_with_R", True))
    with pytest.raises(ValueError, match="ratios example claims"):
        nachbaustand.app_daten(root)


def test_ratios_example_rejects_a_backend_without_the_marking(tmp_path, monkeypatch):
    root = _kopie(tmp_path)
    _tfr_changed(root, monkeypatch, "crosscheck", lambda d: d.__setitem__("backend", "math.erfc"))
    with pytest.raises(ValueError, match="ratios example claims"):
        nachbaustand.app_daten(root)


def test_ratios_example_rejects_a_claimed_run_of_the_cplusplus_lines(tmp_path, monkeypatch):
    root = _kopie(tmp_path)
    _tfr_changed(root, monkeypatch, "crosscheck", lambda d: d.__setitem__("cplusplus_lines_run", True))
    with pytest.raises(ValueError, match="ratios example claims"):
        nachbaustand.app_daten(root)


def test_ratios_example_rejects_a_draw(tmp_path, monkeypatch):
    root = _kopie(tmp_path)
    _tfr_changed(root, monkeypatch, "binding", lambda d: d.__setitem__("draws_run", 1))
    with pytest.raises(ValueError, match="ratios example claims"):
        nachbaustand.app_daten(root)


@pytest.mark.parametrize("group,key,field,bad", [("stress", "grid_ratio", "max_ulps", 1), ("named", "prod_from_R_factors", "bit_equal", 0), ("stress", "density_ratio", "non_finite_mismatch", 1)])
def test_ratios_example_rejects_a_change_in_the_exact_part(tmp_path, monkeypatch, group, key, field, bad):
    root = _kopie(tmp_path)
    _tfr_changed(root, monkeypatch, "crosscheck", lambda d: d[group][key].__setitem__(field, bad))
    with pytest.raises(ValueError, match="left the R record"):
        nachbaustand.app_daten(root)


def test_ratios_example_rejects_a_stored_value_that_left_the_reference(tmp_path, monkeypatch):
    root = _kopie(tmp_path)
    _tfr_changed(root, monkeypatch, "result", lambda d: d["cases"]["three_taxa_quarters"]["tann_tf_ratio"].__setitem__(0, "0x1.8p+4"))
    with pytest.raises(ValueError, match="differs from the reference"):
        nachbaustand.app_daten(root)


def test_ratios_example_rejects_a_release_claim(tmp_path, monkeypatch):
    root = _kopie(tmp_path)
    _tfr_changed(root, monkeypatch, "fixture", lambda d: d.__setitem__("historical_reproduction_ready", True))
    with pytest.raises(ValueError, match="ratios example claims"):
        nachbaustand.app_daten(root)


def test_every_language_key_is_defined_once():
    import re
    from collections import Counter
    lines = (ROOT / "src/kinneret_recon/app/i18n.js").read_text(encoding="utf-8").split("\n")
    blocks = [i for i, line in enumerate(lines) if re.match(r"\s*(de|en)\s*:\s*\{", line)]
    assert len(blocks) == 2
    for start, end in ((blocks[0], blocks[1]), (blocks[1], len(lines))):
        keys = Counter(m.group(1) for line in lines[start:end] for m in [re.match(r"\s+([A-Za-z_0-9]+)\s*:", line)] if m)
        assert [k for k, n in keys.items() if n > 1] == []


def _ic_changed(root, monkeypatch, mutate):
    rel = STATE["input_contract"]["contract"]
    data = json.loads((root / rel).read_text(encoding="utf-8"))
    mutate(data)
    content = json.dumps(data).encode()
    (root / rel).write_bytes(content)
    state = json.loads(json.dumps(STATE))
    state["input_contract"]["contract_sha256"] = hashlib.sha256(content).hexdigest()
    state_bytes = json.dumps(state).encode()
    (root / nachbaustand.STATE).write_bytes(state_bytes)
    monkeypatch.setattr(nachbaustand, "STATE_SHA256", hashlib.sha256(state_bytes).hexdigest())


def test_input_contract_is_shown_with_its_counts_and_keeps_every_input():
    c = nachbaustand.app_daten(ROOT)["input_contract"]
    stored = json.loads((ROOT / IC / "input_contract.json").read_text(encoding="utf-8"))
    assert c["counts"] == {"gebunden": 5, "teilweise_gebunden": 10, "quellkonflikt": 6, "fehlend": 27} == stored["status_counts"] and c["total"] == 48
    assert [i["id"] for i in c["inputs"]] == [i["id"] for i in stored["inputs"]] and [(x["id"], x["status"]) for x in c["conditions"]] == [(x["id"], x["status"]) for x in stored["conditions"]]
    assert c["url"].endswith(IC + "README.md") and "/blob/" + STATE["source_commit"] + "/" in c["url"] and c["new_tests"] == 285
    assert c["older_site_name"] == "Kinneret" and c["grid"] == {"lat": 34.75, "cells": 1, "lon": 24.75} and len(c["flags"]) == 8
    site = next(i for i in c["inputs"] if i["id"] == "site_name")
    assert site["status"] == "quellkonflikt" and site["lines"] == "check_ages 7; prepare 790, 881, 1145, 1209, 1248"


@pytest.mark.parametrize("key,value", [("historical_reproduction_ready", True), ("draws_run", 1), ("sampler_run", True), ("r_scripts_run", True), ("models_trained", True), ("bundles_closed", 1), ("selected_variant", "x")])
def test_input_contract_rejects_a_release_a_run_or_a_closed_bundle(tmp_path, monkeypatch, key, value):
    root = _kopie(tmp_path)
    _ic_changed(root, monkeypatch, lambda d: d.__setitem__(key, value))
    with pytest.raises(ValueError, match="input contract claims"):
        nachbaustand.app_daten(root)


def test_input_contract_rejects_a_bundle_marked_closed(tmp_path, monkeypatch):
    root = _kopie(tmp_path)
    _ic_changed(root, monkeypatch, lambda d: d["phase1_bundles"][0].__setitem__("closed", True))
    with pytest.raises(ValueError, match="input contract claims"):
        nachbaustand.app_daten(root)


def test_input_contract_rejects_an_unknown_status(tmp_path, monkeypatch):
    root = _kopie(tmp_path)
    _ic_changed(root, monkeypatch, lambda d: d["inputs"][0].__setitem__("status", "erledigt"))
    with pytest.raises(ValueError, match="unknown or inconsistent"):
        nachbaustand.app_daten(root)


def test_input_contract_rejects_counts_that_do_not_fit(tmp_path, monkeypatch):
    root = _kopie(tmp_path)
    _ic_changed(root, monkeypatch, lambda d: d["inputs"][0].__setitem__("status", "gebunden"))
    with pytest.raises(ValueError, match="unknown or inconsistent"):
        nachbaustand.app_daten(root)


def test_input_contract_keeps_the_coordinate_condition_conditional(tmp_path, monkeypatch):
    root = _kopie(tmp_path)
    _ic_changed(root, monkeypatch, lambda d: next(c for c in d["conditions"] if c["id"] == "inside_giesecke").__setitem__("status", "belegt"))
    with pytest.raises(ValueError, match="inside.giesecke"):
        nachbaustand.app_daten(root)


def test_input_contract_has_a_label_for_every_row_in_both_languages():
    import re
    c = nachbaustand.app_daten(ROOT)["input_contract"]
    lines = (ROOT / "src/kinneret_recon/app/i18n.js").read_text(encoding="utf-8").split("\n")
    blocks = [i for i, line in enumerate(lines) if re.match(r"\s*(de|en)\s*:\s*\{", line)]
    needed = {"nb_ic_i_" + i["id"] for i in c["inputs"]} | {"nb_ic_c_" + x["id"] for x in c["conditions"]} | {"nb_ic_g_" + g for g in c["groups"]}
    for start, end in ((blocks[0], blocks[1]), (blocks[1], len(lines))):
        keys = {m.group(1) for line in lines[start:end] for m in [re.match(r"\s+([A-Za-z_0-9]+)\s*:", line)] if m}
        assert needed <= keys
