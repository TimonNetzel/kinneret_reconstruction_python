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


def test_fold_scores_match_all_preserved_rows_and_test_counts():
    data = av.app_daten(ROOT)
    folds = data["fold_scores"]
    assert folds["available"] and folds["scores_sha256"] == av.SCORES_SHA256
    with (ROOT / av.SCORES).open(encoding="utf-8", newline="") as f:
        source = [r for r in csv.DictReader(f) if r["scope"] != "all"]
    assert len(folds["rows"]) == len(source) == 280
    for row, original in zip(folds["rows"], source):
        assert row["taxon"] == original["taxon"] and row["method"] == original["method"]
        assert row["block_degrees"] == int(original["block_degrees"])
        assert row["fold"] == int(original["scope"])
        for key in ("cells", "positive_cells"):
            assert row[key] == int(original[key])
        for key in ("balanced_accuracy", "brier"):
            assert row[key] == float(original[key])
    assert all(0 < r["positive_cells"] < r["cells"] for r in folds["rows"])
    # The overall score is not the unweighted mean of fold scores.
    example = [r for r in folds["rows"] if r["taxon"] == "abies_alba" and r["block_degrees"] == 5 and r["method"] == "nnet_smote"]
    overall = next(r for r in data["rows"] if r["taxon"] == "abies_alba" and r["block_degrees"] == 5 and r["method"] == "nnet_smote")
    assert abs(statistics.mean(r["balanced_accuracy"] for r in example) - overall["balanced_accuracy"]) > 1e-4


def test_missing_folds_keep_overall_metrics_available(tmp_path):
    path = tmp_path / av.METRICS
    path.parent.mkdir(parents=True)
    path.write_bytes((ROOT / av.METRICS).read_bytes())
    data = av.app_daten(tmp_path)
    assert data["available"] and len(data["rows"]) == 56
    assert data["fold_scores"] == {"available": False}
    assert "folds" not in data["sources"]


def test_changed_fold_scores_require_review(tmp_path):
    path = tmp_path / av.SCORES
    path.parent.mkdir(parents=True)
    path.write_bytes((ROOT / av.SCORES).read_bytes() + b"\n")
    with pytest.raises(ValueError, match="fold scores hash mismatch"):
        av._fold_daten(tmp_path, [])


def _synthetic_fold_fixture(tmp_path, monkeypatch, summary, change):
    """Negative controls only: original score files are never edited."""
    import hashlib
    import io
    with (ROOT / av.SCORES).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    change(rows)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    content = stream.getvalue().encode()
    path = tmp_path / av.SCORES
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    monkeypatch.setattr(av, "SCORES_SHA256", hashlib.sha256(content).hexdigest())
    return av._fold_daten(tmp_path, summary)


@pytest.mark.parametrize("missing_class", ["positive", "negative"])
def test_undefined_classification_is_null_and_brier_is_preserved(tmp_path, monkeypatch, missing_class):
    # This hypothetical case is not present in the preserved 280 fold rows.
    summary = av.app_daten(ROOT)["rows"]
    def change(rows):
        subset = [r for r in rows if r["taxon"] == "abies_alba" and r["block_degrees"] == "5"]
        original = int(subset[0]["positive_cells"])
        replacement = 0 if missing_class == "positive" else int(subset[0]["cells"])
        for r in subset:
            if r["scope"] == "0":
                r["positive_cells"] = str(replacement)
                r["balanced_accuracy"] = ""
            elif r["scope"] == "all":
                r["positive_cells"] = str(int(r["positive_cells"]) + replacement - original)
    folds = _synthetic_fold_fixture(tmp_path, monkeypatch, summary, change)
    altered = [r for r in folds["rows"] if r["taxon"] == "abies_alba" and r["block_degrees"] == 5 and r["fold"] == 0]
    assert len(altered) == 4 and all(r["balanced_accuracy"] is None and r["brier"] > 0 for r in altered)


@pytest.mark.parametrize("invalid", ["duplicate", "total_mismatch", "nonfinite"])
def test_invalid_fold_contract_is_rejected(tmp_path, monkeypatch, invalid):
    summary = av.app_daten(ROOT)["rows"]
    def change(rows):
        if invalid == "duplicate":
            rows[1] = rows[0].copy()
        elif invalid == "total_mismatch":
            next(r for r in rows if r["scope"] == "all")["brier"] = "0.1"
        else:
            rows[0]["brier"] = "nan"
    with pytest.raises(ValueError, match="unsupported fold table|totals disagree|invalid fold metrics"):
        _synthetic_fold_fixture(tmp_path, monkeypatch, summary, change)
