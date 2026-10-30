"""Moderne Shared37-Baseline (docs/thesis_research/baseline2025_2026_10_29/): Projektzugang und App-Daten.

Die Rechenregeln stehen unveraendert in ``tools/thesis_import/evaluate_shared2025.py``; dieses Modul laedt den
Evaluator per Dateipfad, gleicht einen Lauf mit dem gespeicherten Referenzmanifest ab und fasst die gespeicherten
Metriken fuer die App zusammen. Es rechnet kein eigenes Modell und keine fossilen Klimakurven.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

BASELINE_EVALUATOR = Path("tools/thesis_import/evaluate_shared2025.py")
BASELINE_DIR = Path("docs/thesis_research/baseline2025_2026_10_29")
BASELINE_CONFIG = BASELINE_DIR / "baseline_config.json"
BASELINE_REFERENZ = BASELINE_DIR / "reference_manifest.json"
BASELINE_POLICY = Path("docs/thesis_research/modern2025_2026_10_29/shared_features.json")
REFERENZMODELL = "training_mean"
# Muessen zum Referenzlauf passen, sonst wurde etwas anderes gerechnet (Eingaben, Konfiguration oder Evaluator)
EINGANGS_HASHES = ("config_sha256", "feature_policy_sha256", "source_counts_sha256", "input_manifest_sha256",
                   "evaluation_script_sha256")


def sha256(pfad: Path) -> str:
    return hashlib.sha256(Path(pfad).read_bytes()).hexdigest()


def lade_evaluator(root: Path):
    """Laedt den Evaluator unveraendert aus tools/ (seine Pruefsumme steht im Manifest jedes Laufs)."""
    import importlib.util

    pfad = root / BASELINE_EVALUATOR
    if not pfad.is_file():
        raise SystemExit(f"Evaluator nicht gefunden: {pfad} (aus dem Hauptverzeichnis des Repositorys aufrufen)")
    spec = importlib.util.spec_from_file_location("thesis_evaluate_shared2025", pfad)
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


def referenz_abweichungen(manifest: dict, referenz: dict) -> list[str]:
    """Abweichungen eines Laufmanifests vom Referenzmanifest (leer: bytegleiche Ausgaben, gleiche Eingaben)."""
    out = [k for k in EINGANGS_HASHES if manifest.get(k) != referenz.get(k)]
    if manifest.get("config") != referenz.get("config"):
        out.append("config")
    ist, soll = manifest.get("outputs", {}), referenz.get("outputs", {})
    out += [f"outputs/{n}" for n in sorted(set(ist) | set(soll)) if ist.get(n) != soll.get(n)]
    return out


def kompakte_metriken(metriken: dict, gewichtung: str = "sample") -> list[dict]:
    """Je Blockgroesse und Modell RMSE und MSE-Skill von Tann/Pann (fuer die Ausgabe auf der Kommandozeile)."""
    zeilen = []
    for block, s in metriken["scenarios"].items():
        for modell, m in s["metrics"].items():
            zeilen.append(dict(block_grad=int(block), modell=modell, **{
                f"{z}_{w}": m[z][gewichtung][w] for z in metriken["target_units"] for w in ("rmse", "mse_skill")}))
    return zeilen


def app_daten(root: Path) -> dict | None:
    """Gespeicherte Baseline fuer die Ansicht "Moderne Validierung"; None, wenn die Dateien fehlen."""
    d = root / BASELINE_DIR
    if not (d / "metrics.json").is_file():
        return None
    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    cfg = json.loads((d / "baseline_config.json").read_text(encoding="utf-8"))
    pr = json.loads((d / "verification.json").read_text(encoding="utf-8"))
    haupt = f"analog_k{cfg['primary_neighbors']}"
    modelle = [REFERENZMODELL] + [f"analog_k{k}" for k in sorted({cfg["primary_neighbors"], *cfg["sensitivity_neighbors"]})]
    if m["primary_model"] != haupt or m["target_units"] != dict(zip(cfg["targets"], cfg["target_units"])):
        raise SystemExit(f"Baseline in {d}: Metriken passen nicht zur Konfiguration")
    for block, s in m["scenarios"].items():
        if sorted(s["metrics"]) != sorted(modelle) or len(s["folds"]) != cfg["folds"]:
            raise SystemExit(f"Baseline in {d}: Modelle oder Folds des {block}°-Gitters unvollstaendig")
    reg = pr["posthoc_geographic_diagnostic"]
    return {
        "quelle": BASELINE_DIR.as_posix(),
        "experiment": m["experiment"], "variante": m["variant"],
        "eingang": m["input_samples"], "proben": m["selected_samples"], "ausgeschlossen": m["threshold_excluded"],
        "merkmale": len(m["features"]), "koordinatengruppen": m["coordinate_groups"],
        "folds": cfg["folds"], "puffer_km": cfg["buffer_km"], "seed": cfg["seed"],
        "hauptmodell": haupt, "referenzmodell": REFERENZMODELL, "modelle": modelle,
        "sensitivitaet": [f"analog_k{k}" for k in cfg["sensitivity_neighbors"]],
        "einheiten": m["target_units"], "gewichtungen": cfg["metric_weighting"],
        "bloecke": {b: {"bloecke": s["blocks"], "metriken": s["metrics"],
                        "folds": [{k: f[k] for k in ("fold", "test_samples", "train_samples", "buffer_excluded",
                                                     "minimum_train_test_km")} | {"metriken": f["metrics"]}
                                  for f in s["folds"]]}
                    for b, s in m["scenarios"].items()},
        "regional": {"breite": reg["latitude_bounds"], "laenge": reg["longitude_bounds"], "proben": reg["samples"],
                     "metriken": reg["metrics"]},
        "pruefung": {k: pr[k] for k in ("minimum_train_test_km", "folds_checked", "prediction_rows_checked",
                                        "analogue_prediction_spotchecks", "all_output_hashes_verified",
                                        "all_prediction_metrics_verified", "two_full_runs_byte_identical",
                                        "design_commit")},
    }
