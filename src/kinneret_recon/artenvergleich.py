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
SCORES = DIRECTORY / "scores.csv"
SCORES_SHA256 = "69dc8e6e4da6f057f643889edb9334f94bac4f4f5751ae59a5e7134aa7f8c538"
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
        "fold_scores": _fold_daten(root, rows),
        "folds": 5, "buffer_km": 100, "threshold": 0.5, "metrics_sha256": METRICS_SHA256,
        "sources": {key: "https://github.com/TimonNetzel/kinneret_reconstruction_python/blob/" +
                    SOURCE_COMMIT + "/" + (DIRECTORY / file).as_posix()
                    for key, file in SOURCES.items() if (root / DIRECTORY / file).is_file()},
    }


def _fold_daten(root: Path, summary: list[dict]) -> dict:
    """Preserve undefined classification scores; do not average folds or refit anything."""
    path = root / SCORES
    if not path.is_file():
        return {"available": False}
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != SCORES_SHA256:
        raise ValueError("Artenvergleich: fold scores hash mismatch; review the evidence before updating the binding")
    source = list(csv.DictReader(io.StringIO(content.decode("utf-8"))))
    expected = {(s, b, m, f) for s in SPECIES for b in (5, 10) for m in METHODS
                for f in ("0", "1", "2", "3", "4", "all")}
    actual = {(r["taxon"], int(r["block_degrees"]), r["method"], r["scope"]) for r in source}
    if len(source) != len(expected) or actual != expected:
        raise ValueError("Artenvergleich: incomplete or unsupported fold table")
    rows, totals, counts = [], [], {}
    for r in source:
        cells, positive = int(r["cells"]), int(r["positive_cells"])
        ba = None if r["balanced_accuracy"] == "" else float(r["balanced_accuracy"])
        brier = float(r["brier"])
        if (cells <= 0 or not 0 <= positive <= cells or
            (ba is None) != (positive in (0, cells)) or
            any(not math.isfinite(v) or not 0 <= v <= 1 for v in (ba, brier) if v is not None)):
            raise ValueError("Artenvergleich: invalid fold metrics or test class counts")
        key = (r["taxon"], int(r["block_degrees"]), r["scope"])
        if key in counts and counts[key] != (cells, positive):
            raise ValueError("Artenvergleich: methods do not share the same test cells")
        counts[key] = (cells, positive)
        base = {"taxon": r["taxon"], "block_degrees": int(r["block_degrees"]),
                "method": r["method"], "balanced_accuracy": ba, "brier": brier}
        if r["scope"] == "all":
            totals.append({**base, "scope": "all"})
        else:
            rows.append({**base, "fold": int(r["scope"]), "cells": cells, "positive_cells": positive})
    by_key = lambda r: (r["taxon"], r["block_degrees"], r["method"])
    if sorted(totals, key=by_key) != sorted(summary, key=by_key):
        raise ValueError("Artenvergleich: fold file totals disagree with summary")
    for species in SPECIES:
        for block in (5, 10):
            summed = tuple(sum(counts[(species, block, str(f))][i] for f in range(5)) for i in (0, 1))
            if summed != counts[(species, block, "all")]:
                raise ValueError("Artenvergleich: fold cell counts disagree with totals")
    return {"available": True, "rows": rows, "scores_sha256": SCORES_SHA256}
