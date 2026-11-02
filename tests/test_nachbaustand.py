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
FROZEN = nachbaustand.STATES[0]["path"].as_posix()


def _kopie(tmp_path: Path) -> Path:
    """Copy exactly the pinned files so single bytes can be changed or removed."""
    ex, contract, pex = STATE["synthetic_example"], STATE["contract"], STATE["pruning_example"]
    for rel in ([s["path"].as_posix() for s in nachbaustand.STATES] + [STATE["chain_check"]["package"],
                contract["package"], contract["spec"]] + [m["package"] for m in STATE["modules"]]
                + [e[k] for e in (ex, pex) for k in ("fixture", "result", "rational", "arithmetic")]):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, tmp_path / rel)
    return tmp_path


def test_newest_state_binds_six_modules_eleven_open_bundles_and_no_release():
    d = nachbaustand.app_daten(ROOT)
    assert d["available"] and d["status"] == "submodules_checked_original_case_open"
    assert [m["id"] for m in d["modules"]] == list(nachbaustand.MODULE_IDS)
    assert d["modules"][-1]["id"] == "taxon_pruning" and d["modules"][-1]["conditional"]
    assert not any(m["conditional"] for m in d["modules"][:-1])
    assert d["tests_passed"] == 297 == sum(m["new_tests"] for m in d["modules"]) + d["chain_check"]["new_tests"] + d["contract"]["tests"]
    assert [r["id"] for r in d["requirements"]] == list(nachbaustand.REQUIREMENT_IDS)
    assert {r["status"] for r in d["requirements"]} == {"missing", "conflicting"}
    assert sum(r["status"] == "conflicting" for r in d["requirements"]) == 4
    assert all("/blob/" + STATE["source_commit"] + "/" in m["url"] for m in d["modules"])
    text = json.dumps(d)
    assert ".local/" not in text and "C:\\" not in text and "historical_reproduction_ready" not in text


def test_earlier_state_stays_frozen_pinned_and_listed():
    frozen = nachbaustand.STATES[0]
    assert hashlib.sha256((ROOT / FROZEN).read_bytes()).hexdigest() == frozen["sha256"]
    old = json.loads((ROOT / FROZEN).read_text(encoding="utf-8"))
    assert old["tests"]["passed"] == 251 and len(old["modules"]) == 5
    assert STATE["supersedes"] == {"state": FROZEN, "state_sha256": frozen["sha256"]}
    assert nachbaustand.app_daten(ROOT)["earlier_states"] == [{
        "date": "2026-11-02", "state_sha256": frozen["sha256"], "modules": 5, "tests_passed": 251,
        "url": nachbaustand.REPO_URL + frozen["commit"] + "/" + FROZEN}]
    assert [m["package_sha256"] for m in old["modules"]] == [m["package_sha256"] for m in STATE["modules"][:5]]


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


@pytest.mark.parametrize("rel", [nachbaustand.STATE.as_posix(), PACKET + "synthetic_result.json",
                                 PACKET + "arithmetic_check.json", PACKET + "synthetic_fixture.json",
                                 "docs/thesis_research/kinneret_phase1_spec_2026_11_02/spec.json",
                                 "docs/thesis_research/literal_age_depth_2026_11_02/README.md",
                                 FROZEN, PRUNING + "README.md", PRUNING + "synthetic_result.json",
                                 PRUNING + "rational_expectations.json"])
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
    assert '"earlier_states"' in html and nachbaustand.STATES[0]["sha256"] in html
