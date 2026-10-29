"""Projektzugang zum Thesis-Importer (kinneret-recon thesis-import) mit synthetischen Quellen, ohne Netz."""

import csv
import hashlib
import json
import sys
import unittest
from pathlib import Path

import pytest

from kinneret_recon.cli import THESIS_IMPORTER, THESIS_REFERENZ, lade_thesis_importer, main

ROOT = Path(__file__).resolve().parents[1]
TOOL_DIR = ROOT / THESIS_IMPORTER.parent


@pytest.fixture
def quellen(monkeypatch):
    """Synthetische, im Register gesperrte Quellen aus den Vertragstests des Importers."""
    monkeypatch.chdir(ROOT)
    monkeypatch.syspath_prepend(str(TOOL_DIR))
    from test_import_data import ImportContractTests

    t = ImportContractTests()
    t.setUp()
    yield t
    t.doCleanups()


def _cli(q, ziel, *extra):
    return main(["thesis-import", "--raw-dir", str(q.raw), "--output-dir", str(q.root / ziel),
                 "--catalog", str(q.catalog), *extra])


def _zeilen(pfad):
    with pfad.open(encoding="utf-8", newline="") as f:
        return {r["sample_id"]: r for r in csv.DictReader(f)}


def test_vertragstests_des_importers():
    suite = unittest.defaultTestLoader.discover(str(TOOL_DIR), pattern="test_*.py", top_level_dir=str(TOOL_DIR))
    ergebnis = unittest.TextTestRunner(stream=sys.stderr, verbosity=0).run(suite)
    assert ergebnis.testsRun == 11
    assert ergebnis.wasSuccessful()


def test_referenz_passt_zum_unveraenderten_importer():
    ref = json.loads((ROOT / THESIS_REFERENZ).read_text(encoding="utf-8"))
    assert ref["manifest"]["importer_sha256"] == hashlib.sha256((ROOT / THESIS_IMPORTER).read_bytes()).hexdigest()
    flags = ref["report"]["empd"]["flags"]
    assert (flags["missing_climate"], flags["fractional_count_cells"], flags["location_X"]) == (529, 1059, 284)
    assert ref["report"]["monticchio"]["excluded_samples"] == 176


def test_cli_import_erhaelt_flags_und_rohwerte(quellen, capsys):
    assert _cli(quellen, "import", "--ohne-referenz") == 0
    out = quellen.root / "import"
    assert "Referenz: nicht geprueft" in capsys.readouterr().out
    proben = _zeilen(out / "empd_samples.csv")
    assert "fractional_counts" in proben["A"]["quality_flags"]
    assert proben["B"]["tann_c"] == "" and "location_quality_unknown" in proben["B"]["quality_flags"]
    assert "location_X" in proben["C"]["quality_flags"]
    assert {r["scientific_status"] for r in proben.values()} == {"not_assessed"}
    assert list(_zeilen(out / "monticchio_excluded.csv")) == ["LGM90D:000003"]
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["importer_sha256"] == hashlib.sha256((ROOT / THESIS_IMPORTER).read_bytes()).hexdigest()


def test_cli_referenzabgleich(quellen, capsys):
    assert _cli(quellen, "basis", "--ohne-referenz") == 0
    basis = quellen.root / "basis"
    ref = quellen.root / "ref.json"
    ref.write_text(json.dumps({k: json.loads((basis / f"{k}.json").read_text()) for k in ("report", "manifest")}))
    assert _cli(quellen, "gleich", "--reference", str(ref)) == 0
    assert "Referenz: uebereinstimmend" in capsys.readouterr().out
    # Standard ist der gespeicherte Lauf mit den Originaldaten: synthetische Quellen weichen ab
    with pytest.raises(SystemExit, match="Import fehlgeschlagen: Reference mismatch"):
        _cli(quellen, "abweichend")
    assert not (quellen.root / "abweichend").exists()


def test_cli_ueberschreibt_nicht_und_meldet_fehlende_quellen(quellen):
    (quellen.root / "vorhanden").mkdir()
    with pytest.raises(SystemExit, match="already exists"):
        _cli(quellen, "vorhanden", "--ohne-referenz")
    (quellen.raw / "monticchio_739452.tab").unlink()
    with pytest.raises(SystemExit, match="Missing source file"):
        _cli(quellen, "neu", "--ohne-referenz")
    assert not (quellen.root / "neu").exists()


def test_importer_nur_im_repository(tmp_path):
    with pytest.raises(SystemExit, match="Importer nicht gefunden"):
        lade_thesis_importer(tmp_path)
