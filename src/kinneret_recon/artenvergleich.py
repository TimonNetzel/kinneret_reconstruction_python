"""Read the preserved seven-species scores; never fit, calibrate or predict."""
from __future__ import annotations

import csv
import hashlib
import io
import math
from pathlib import Path

DIRECTORY = Path("docs/thesis_research/smote_nnet_2026_10_30")
METRICS = DIRECTORY / "metrics_summary.csv"
METRICS_SHA256 = "93ff7c368c17f39e87e7843a423f0f36bd259cd37a4c9bca7803586f54aa320b"
SOURCE_COMMIT = "2d2fdfd019d7af09fd55fd8e60e429d1c2b2f7a0"
SPECIES = {
    "abies_alba": "Abies alba", "betula_pendula": "Betula pendula",
    "corylus_avellana": "Corylus avellana", "fagus_sylvatica": "Fagus sylvatica",
    "picea_abies": "Picea abies", "taxus_baccata": "Taxus baccata", "tilia_cordata": "Tilia cordata",
}
METHODS = ("nnet_original", "nnet_smote", "pr28_quadratic", "nnet_smote_prior_adjusted")
SOURCES = {"report": "README.md", "metrics": "metrics_summary.csv", "design": "DESIGN.md",
           "sources": "sources.json", "verification": "verification.json", "folds": "scores.csv"}


def app_daten(root: Path) -> dict:
    """A missing score file is explicit; changed bytes require a reviewed new binding."""
    path = root / METRICS
    if not path.is_file():
        return {"schema_version": 1, "available": False}
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != METRICS_SHA256:
        raise ValueError("Artenvergleich: metrics hash mismatch; review the evidence before updating the binding")
    rows = []
    for r in csv.DictReader(io.StringIO(content.decode("utf-8"))):
        rows.append({"taxon": r["taxon"], "block_degrees": int(r["block_degrees"]),
                     "scope": r["scope"], "method": r["method"],
                     "balanced_accuracy": float(r["balanced_accuracy"]), "brier": float(r["brier"])})
    expected = {(s, b, m, "all") for s in SPECIES for b in (5, 10) for m in METHODS}
    actual = {(r["taxon"], r["block_degrees"], r["method"], r["scope"]) for r in rows}
    if len(rows) != len(expected) or actual != expected or any(
        not math.isfinite(r[k]) or not 0 <= r[k] <= 1 for r in rows for k in ("balanced_accuracy", "brier")
    ):
        raise ValueError("Artenvergleich: incomplete or unsupported score table")
    # Descriptive, equally weighted species means, never pooled cells or fold uncertainty.
    means = []
    for b in (5, 10):
        for m in METHODS:
            subset = [r for r in rows if r["block_degrees"] == b and r["method"] == m]
            means.append({"block_degrees": b, "method": m, **{
                k: math.fsum(r[k] for r in subset) / len(SPECIES) for k in ("balanced_accuracy", "brier")}})
    return {
        "schema_version": 1, "available": True, "date": "2026-10-30", "status": "experimental",
        "species": SPECIES, "methods": METHODS, "blocks": [5, 10], "rows": rows, "means": means,
        "folds": 5, "buffer_km": 100, "threshold": 0.5, "metrics_sha256": METRICS_SHA256,
        "sources": {key: "https://github.com/TimonNetzel/kinneret_reconstruction_python/blob/" +
                    SOURCE_COMMIT + "/" + (DIRECTORY / file).as_posix()
                    for key, file in SOURCES.items() if (root / DIRECTORY / file).is_file()},
    }
