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
FROZEN = nachbaustand.STATES[0]["path"].as_posix()
FROZEN2 = nachbaustand.STATES[1]["path"].as_posix()
FROZEN3 = nachbaustand.STATES[2]["path"].as_posix()


def _kopie(tmp_path: Path) -> Path:
    """Copy exactly the pinned files so single bytes can be changed or removed."""
    ex, contract, pex, gex, mex = (STATE["synthetic_example"], STATE["contract"], STATE["pruning_example"],
                                   STATE["gamma_example"], STATE["missing_example"])
    for rel in ([s["path"].as_posix() for s in nachbaustand.STATES] + [STATE["chain_check"]["package"],
                contract["package"], contract["spec"]] + [m["package"] for m in STATE["modules"]]
                + [e[k] for e in (ex, pex, gex, mex) for k in ("fixture", "result", "rational", "arithmetic")]):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, tmp_path / rel)
    return tmp_path


def test_newest_state_binds_eight_modules_eleven_open_bundles_and_no_release():
    d = nachbaustand.app_daten(ROOT)
    assert d["available"] and d["status"] == "submodules_checked_original_case_open"
    assert [m["id"] for m in d["modules"]] == list(nachbaustand.MODULE_IDS)
    assert d["modules"][-1]["id"] == "tf_missing_priors" and not d["modules"][-1]["conditional"]
    assert [m["id"] for m in d["modules"] if m["conditional"]] == ["taxon_pruning"]
    assert d["tests_passed"] == 437 == sum(m["new_tests"] for m in d["modules"]) + d["chain_check"]["new_tests"] + d["contract"]["tests"]
    assert [r["id"] for r in d["requirements"]] == list(nachbaustand.REQUIREMENT_IDS)
    assert {r["status"] for r in d["requirements"]} == {"missing", "conflicting"}
    assert sum(r["status"] == "conflicting" for r in d["requirements"]) == 4
    assert all("/blob/" + STATE["source_commit"] + "/" in m["url"] for m in d["modules"])
    text = json.dumps(d)
    assert ".local/" not in text and "C:\\" not in text and "historical_reproduction_ready" not in text


def test_earlier_states_stay_frozen_pinned_and_listed():
    first, second, third = nachbaustand.STATES[:3]
    assert (first["sha256"], second["sha256"], third["sha256"]) == (
        "c358b2c3382f568a94d070054ba1dee4b230c617506c9a5015e2cc4506f4e63e",
        "3656316de4211500184f1a1c772bf07c87e3bd4b2d210ed6316e70c5a8ca0784",
        "43edfcb67a64826ce72611c93be33490463e59566934393aeff952bd217fbf4f")
    assert third["commit"] == "7d3d0f6852dbdf869b6e479d4eeffe80faaf2038"
    for frozen in (first, second, third):
        assert hashlib.sha256((ROOT / frozen["path"]).read_bytes()).hexdigest() == frozen["sha256"]
    old, prev, last = (json.loads((ROOT / f["path"]).read_text(encoding="utf-8")) for f in (first, second, third))
    assert (old["tests"]["passed"], len(old["modules"])) == (251, 5)
    assert (prev["tests"]["passed"], len(prev["modules"])) == (297, 6)
    assert (last["tests"]["passed"], len(last["modules"])) == (352, 7)
    assert prev["supersedes"] == {"state": FROZEN, "state_sha256": first["sha256"]}
    assert last["supersedes"] == {"state": FROZEN2, "state_sha256": second["sha256"]}
    assert STATE["supersedes"] == {"state": FROZEN3, "state_sha256": third["sha256"]}
    assert nachbaustand.app_daten(ROOT)["earlier_states"] == [
        {"date": "2026-11-02", "state_sha256": f["sha256"], "modules": n, "tests_passed": tests,
         "url": nachbaustand.REPO_URL + f["commit"] + "/" + f["path"].as_posix()}
        for f, n, tests in ((first, 5, 251), (second, 6, 297), (third, 7, 352))]
    assert [m["package_sha256"] for m in last["modules"]] == [m["package_sha256"] for m in STATE["modules"][:7]]
    assert all(STATE[k] == last[k] for k in ("chain_check", "contract", "synthetic_example", "pruning_example",
                                             "gamma_example"))


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
                                 MISSING + "arithmetic_check.json"])
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
    assert '"earlier_states"' in html and all(s["sha256"] in html for s in nachbaustand.STATES[:3])
