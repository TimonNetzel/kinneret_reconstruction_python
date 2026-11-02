"""Stored species scores must remain exact and usable in both offline formats."""
import csv
import json
from pathlib import Path
import statistics

import pytest

from kinneret_recon import artenvergleich as av
from kinneret_recon.app_bauen import baue_eine_datei, baue_ordner

ROOT = Path(__file__).resolve().parents[1]


def test_all_species_scores_and_means_match_the_preserved_csv():
    data = av.app_daten(ROOT)
    with (ROOT / av.METRICS).open(encoding="utf-8", newline="") as f:
        source = list(csv.DictReader(f))
    assert data["available"] and data["status"] == "experimental"
    assert len(data["rows"]) == len(source) == 56
    assert data["blocks"] == [5, 10] and data["folds"] == 5 and data["buffer_km"] == 100
    assert data["threshold"] == 0.5
    for row, original in zip(data["rows"], source):
        assert row["taxon"] == original["taxon"]
        assert row["block_degrees"] == int(original["block_degrees"])
        assert row["scope"] == original["scope"] == "all"
        assert row["method"] == original["method"]
        for key in ("balanced_accuracy", "brier"):
            assert row[key] == float(original[key])
    assert len(data["means"]) == 8
    for row in data["means"]:
        selected = [r for r in source if int(r["block_degrees"]) == row["block_degrees"] and r["method"] == row["method"]]
        for key in ("balanced_accuracy", "brier"):
            assert row[key] == pytest.approx(statistics.mean(float(r[key]) for r in selected), abs=1e-15)
    # The UI's central claim must hold for each paired species/block, not just on average.
    for species in data["species"]:
        for block in data["blocks"]:
            pair = {r["method"]: r for r in data["rows"] if r["taxon"] == species and r["block_degrees"] == block}
            assert pair["nnet_smote"]["balanced_accuracy"] > pair["nnet_original"]["balanced_accuracy"]
            assert pair["nnet_smote"]["brier"] > pair["nnet_original"]["brier"]
    assert set(data["sources"]) == {"report", "metrics", "design", "sources", "verification", "folds"}
    assert all("/blob/" + av.SOURCE_COMMIT + "/" in u for u in data["sources"].values())
    assert ".local/" not in json.dumps(data) and "C:" not in json.dumps(data)


def test_missing_metrics_do_not_claim_available_results(tmp_path):
    assert av.app_daten(tmp_path) == {"schema_version": 1, "available": False}


def test_changed_metrics_require_review_before_building(tmp_path):
    path = tmp_path / av.METRICS
    path.parent.mkdir(parents=True)
    path.write_bytes((ROOT / av.METRICS).read_bytes() + b"\n")
    with pytest.raises(ValueError, match="metrics hash mismatch"):
        av.app_daten(tmp_path)


def test_scores_are_identical_in_both_offline_formats_and_cached(tmp_path):
    folder = tmp_path / "app"
    baue_ordner(folder, ROOT)
    single = baue_eine_datei(tmp_path / "single.html", ROOT).read_text(encoding="utf-8")
    prefix = '(window.KR_DATEN=window.KR_DATEN||{})["artenvergleich"]='
    script = (folder / "daten/artenvergleich.js").read_text(encoding="utf-8")
    assert json.loads(script[len(prefix):-2]) == json.loads(json.dumps(av.app_daten(ROOT)))
    assert script.strip() in single
    assert 'src="daten/artenvergleich.js"' not in single
    assert 'daten/artenvergleich.js' in (folder / "sw.js").read_text()
    assert 'artenvergleich.js' in (folder / "index.html").read_text(encoding="utf-8")
    assert 'id="ansicht-artenvergleich"' in single
