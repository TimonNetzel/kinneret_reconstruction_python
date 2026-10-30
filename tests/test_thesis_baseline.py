"""Projektzugang zur modernen Shared37-Baseline (kinneret-recon thesis-baseline) und ihre App-Daten, ohne Netz."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from kinneret_recon import baseline as bl
from kinneret_recon.cli import main

ROOT = Path(__file__).resolve().parents[1]
MERKMALE = ("betula", "pinus", "quercus")


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture
def kohorte(tmp_path, monkeypatch):
    """Kleine synthetische Shared-Ausgabe im Format von prepare_shared2025.py, mit passender Richtlinie und Konfiguration."""
    monkeypatch.chdir(ROOT)
    rng = np.random.default_rng(1)
    ein = tmp_path / "shared"
    ein.mkdir()
    zeilen = []
    for i in range(130):
        lat, lon = -60 + (i * 7) % 130, -175 + (i * 37) % 350
        c = rng.integers(0, 80, len(MERKMALE)).astype(float)
        c[0] += 1 + (lat + 60)  # Pollen haengen vom Klima ab, damit Analoge etwas tragen
        ohne = c.sum() - c[1]
        zeilen.append({"sample_id": f"S{i:03d}", **{m: f"{v:.0f}" for m, v in zip(MERKMALE, c)},
                       "shared_count_sum": f"{c.sum():.0f}", "shared_count_without_pinus": f"{ohne:.0f}",
                       "count_threshold_pass": int(ohne >= 30), "tann_c": f"{25 - 0.4 * abs(lat):.3f}",
                       "pann_mm_per_year": f"{300 + 4 * (lat + 60):.1f}", "latitude": lat, "longitude": lon,
                       "coordinate_group_1e6deg": f"{lat:.6f}|{lon:.6f}"})
    with (ein / "modern_shared_counts.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(zeilen[0]))
        w.writeheader()
        w.writerows(zeilen)
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"variant": "published2025_shared37_v0", "minimum_shared_count_without_pinus": 30,
                                  "features": [{"feature_id": m} for m in MERKMALE]}))
    zaehl = ein / "modern_shared_counts.csv"
    (ein / "manifest.json").write_text(json.dumps({"feature_policy_sha256": _sha(policy), "outputs": {
        zaehl.name: {"byte_count": zaehl.stat().st_size, "sha256": _sha(zaehl)}}}))
    cfg = json.loads((ROOT / bl.BASELINE_CONFIG).read_text())
    cfg.update(expected_input_rows=len(zeilen), expected_features=len(MERKMALE),
               expected_selected_rows=sum(z["count_threshold_pass"] for z in zeilen))
    config = tmp_path / "config.json"
    config.write_text(json.dumps(cfg))
    return tmp_path, ["--input-dir", str(ein), "--policy", str(policy), "--config", str(config)]


def test_referenzdateien_passen_zum_evaluator():
    """Gespeichertes Manifest, Konfiguration und Metriken gehoeren zum unveraenderten Evaluator."""
    ref = json.loads((ROOT / bl.BASELINE_REFERENZ).read_text())
    assert ref["evaluation_script_sha256"] == _sha(ROOT / bl.BASELINE_EVALUATOR)
    assert ref["config_sha256"] == _sha(ROOT / bl.BASELINE_CONFIG)
    assert ref["feature_policy_sha256"] == _sha(ROOT / bl.BASELINE_POLICY)
    assert ref["outputs"]["metrics.json"]["sha256"] == _sha(ROOT / bl.BASELINE_DIR / "metrics.json")
    m = json.loads((ROOT / bl.BASELINE_DIR / "metrics.json").read_text())
    with (ROOT / bl.BASELINE_DIR / "metrics_summary.csv").open(encoding="utf-8", newline="") as f:
        tabelle = list(csv.DictReader(f))
    assert len(tabelle) == 32
    for r in tabelle:
        werte = m["scenarios"][r["block_degrees"]]["metrics"][r["model"]][r["target"]][r["weighting"]]
        assert all(float(r[k]) == pytest.approx(werte[k], rel=1e-12) for k in ("rmse", "mae", "bias", "mse_skill"))


def test_app_daten_der_gespeicherten_baseline():
    v = bl.app_daten(ROOT)
    assert (v["proben"], v["ausgeschlossen"], v["merkmale"], v["puffer_km"], v["folds"]) == (7078, 556, 37, 100, 5)
    assert v["hauptmodell"] == "analog_k5" and v["referenzmodell"] == "training_mean"
    assert v["modelle"] == ["training_mean", "analog_k1", "analog_k5", "analog_k15"]
    assert v["sensitivitaet"] == ["analog_k1", "analog_k15"]
    assert v["einheiten"] == {"tann_c": "degC", "pann_mm_per_year": "mm/year"}
    k5 = {b: v["bloecke"][b]["metriken"]["analog_k5"] for b in ("5", "10")}
    assert round(k5["5"]["tann_c"]["sample"]["rmse"], 3) == 3.765 and round(k5["10"]["tann_c"]["sample"]["rmse"], 3) == 3.926
    assert round(k5["5"]["pann_mm_per_year"]["sample"]["rmse"], 1) == 365.2
    assert v["regional"]["proben"] == 5223 and v["pruefung"]["minimum_train_test_km"] > 100
    assert bl.app_daten(Path("/nicht/vorhanden")) is None


def test_cli_lauf_und_referenzabgleich(kohorte, capsys):
    tmp, args = kohorte
    assert main(["thesis-baseline", *args, "--output-dir", str(tmp / "lauf1"), "--ohne-referenz"]) == 0
    out = capsys.readouterr().out
    manifest = json.loads((tmp / "lauf1/manifest.json").read_text())
    assert f"manifest.json SHA256: {_sha(tmp / 'lauf1/manifest.json')}" in out
    assert "analog_k5     Hauptmodell" in out and "training_mean Referenz" in out and "Referenz: nicht geprueft" in out
    assert set(manifest["outputs"]) == {"fold_assignments.csv", "metrics.json", "oof_predictions.csv", "training_roles.csv"}
    assert manifest["evaluation_script_sha256"] == _sha(ROOT / bl.BASELINE_EVALUATOR)
    # Zweiter Lauf gegen den ersten als Referenz: bytegleich
    assert main(["thesis-baseline", *args, "--output-dir", str(tmp / "lauf2"),
                 "--reference", str(tmp / "lauf1/manifest.json")]) == 0
    assert "Referenz: uebereinstimmend" in capsys.readouterr().out
    # Standard ist der gespeicherte Lauf mit den Originaldaten: synthetische Eingaben weichen ab
    with pytest.raises(SystemExit, match="Referenz abweichend .*source_counts_sha256.*outputs/metrics.json"):
        main(["thesis-baseline", *args, "--output-dir", str(tmp / "lauf3")])
    assert (tmp / "lauf3/manifest.json").exists()  # bleibt zur Pruefung


def test_cli_ueberschreibt_nicht_und_meldet_fehler(kohorte):
    tmp, args = kohorte
    (tmp / "vorhanden").mkdir()
    with pytest.raises(SystemExit, match="Baseline fehlgeschlagen: Output directory already exists"):
        main(["thesis-baseline", *args, "--output-dir", str(tmp / "vorhanden")])
    (tmp / "shared/modern_shared_counts.csv").write_text("veraendert")
    with pytest.raises(SystemExit, match="checksum mismatch"):
        main(["thesis-baseline", *args, "--output-dir", str(tmp / "neu")])
    assert not (tmp / "neu").exists()


def test_evaluator_nur_im_repository(tmp_path):
    with pytest.raises(SystemExit, match="Evaluator nicht gefunden"):
        bl.lade_evaluator(tmp_path)


def test_referenzabweichungen():
    ref = json.loads((ROOT / bl.BASELINE_REFERENZ).read_text())
    assert bl.referenz_abweichungen(ref, ref) == []
    anders = json.loads(json.dumps(ref))
    anders["numpy"] = "0"  # Umgebung allein ist keine Abweichung
    assert bl.referenz_abweichungen(anders, ref) == []
    anders["outputs"]["oof_predictions.csv"]["sha256"] = "0"
    anders["config"]["seed"] = 1
    assert bl.referenz_abweichungen(anders, ref) == ["config", "outputs/oof_predictions.csv"]
