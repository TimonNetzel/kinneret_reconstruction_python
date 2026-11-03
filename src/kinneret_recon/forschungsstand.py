"""Read a pinned research snapshot for the app; never run models or inspect raw inputs."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

SNAPSHOT = Path("docs/thesis_research/research_snapshot_2026_11_02/summary.json")
SNAPSHOT_SHA256 = "642d7b5fdfb5984310892fc0c2c07e39c53d6f1f4ec4167ef63fa0d71862f707"
SOURCE_COMMIT = "1cec59f1c37ac59f1440969feeaf4451e1b93dcc"
SOURCES = {
    "baseline": "docs/thesis_research/baseline2025_2026_10_29/README.md",
    "species": "docs/thesis_research/smote_nnet_2026_10_30/README.md",
    "historical": "docs/thesis_research/HISTORICAL_SECOND_TF_PLOT_RESULT.md",
    "paper": "docs/thesis_research/LOCAL_REFERENCE_RESULT.md",
    "chronology": "docs/thesis_research/variable_provenance_2026_10_30/SOURCE_GAP.md",
}


def app_daten(root: Path) -> dict:
    """Missing snapshots stay visibly unavailable; changed snapshot bytes fail the build."""
    path = root / SNAPSHOT
    if not path.is_file():
        return {"schema_version": 1, "available": False}
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != SNAPSHOT_SHA256:
        raise ValueError("Forschungsstand: snapshot hash mismatch; review the new evidence before updating the binding")
    summary = json.loads(content)
    if summary["schema_version"] != 1 or summary["plots"]["numeric_TF_arrays_recovered"] or summary["gates"]["fossil_pilot_release"]:
        raise ValueError("Forschungsstand: unsupported snapshot or scientific release state")
    models = root / "docs/thesis_research/smote_nnet_2026_10_30/models.json"
    species_count = len(json.loads(models.read_text(encoding="utf-8"))) if models.is_file() else 0
    return {
        "schema_version": 1, "available": True, "date": "2026-11-02",
        "summary": summary, "species_count": species_count,
        "baseline_available": (root / "docs/thesis_research/baseline2025_2026_10_29/metrics.json").is_file(),
        "sources": {key: {"url": "https://github.com/TimonNetzel/kinneret_reconstruction_python/blob/" +
                              SOURCE_COMMIT + "/" + rel}
                    for key, rel in SOURCES.items() if (root / rel).is_file()},
        "snapshot_sha256": SNAPSHOT_SHA256,
    }
