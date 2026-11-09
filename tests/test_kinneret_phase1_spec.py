"""Regression checks of evidence integrity; no scientific model is executed."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import check_kinneret_phase1_spec as checker


@pytest.fixture
def evidence_copy(tmp_path):
    spec = checker.load_json(ROOT / checker.SPEC_PATH)
    files = [checker.SPEC_PATH] + [r["path"] for r in spec["assets"].values()]
    manifest_path = spec["assets"]["research_manifest"]["path"]
    manifest = checker.load_json(ROOT / manifest_path)
    files += [(Path(manifest_path).parent / row["published"]).as_posix()
              for row in manifest["entries"]]
    for relative in set(files):
        target = checker.safe_path(tmp_path, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checker.safe_path(ROOT, relative), target)
    return tmp_path


def test_frozen_contract_is_intact_but_unresolved():
    report = checker.check(ROOT)
    assert report["spec_valid"] is True
    assert report["verified_assets"] == 26
    assert report["verified_facts"] == 19
    assert report["verified_snapshot_files"] == 50
    assert len(report["requirements"]) == 11
    assert {r["id"] for r in report["requirements"]} == checker.REQUIREMENTS
    assert report["historical_reproduction_ready"] is False
    assert report["execution_supported"] is False
    assert report["selected_variant"] is None
    assert report["preserved_gates"]["kinneret_prerequisite"] is False
    assert report["preserved_gates"]["observed"]["monticchio_P1"] == "open"
    assert report["local_sources"] is None


@pytest.mark.parametrize("mutation", ["missing", "changed"])
def test_asset_corruption_is_rejected(evidence_copy, mutation):
    spec = checker.load_json(evidence_copy / checker.SPEC_PATH)
    path = checker.safe_path(evidence_copy, spec["assets"]["binding"]["path"])
    if mutation == "missing":
        path.unlink()
    else:
        path.write_bytes(path.read_bytes().replace(b'160', b'161', 1))
    with pytest.raises(checker.EvidenceError, match="evidence|Evidence"):
        checker.check(evidence_copy)


def test_snapshot_corruption_is_rejected(evidence_copy):
    spec = checker.load_json(evidence_copy / checker.SPEC_PATH)
    manifest_path = checker.safe_path(evidence_copy, spec["assets"]["research_manifest"]["path"])
    entry = checker.load_json(manifest_path)["entries"][0]
    checker.safe_path(manifest_path.parent, entry["published"]).write_bytes(b'{}')
    with pytest.raises(checker.EvidenceError, match="Evidence bytes differ"):
        checker.check(evidence_copy)


@pytest.mark.parametrize("mutation", ["drop_requirement", "ready_flag", "select_variant"])
def test_spec_changes_require_reviewed_release(evidence_copy, mutation):
    path = evidence_copy / checker.SPEC_PATH
    spec = checker.load_json(path)
    if mutation == "drop_requirement":
        spec["requirements"].pop()
    elif mutation == "ready_flag":
        spec["historical_reproduction_ready"] = True
    else:
        spec["selected_variant"] = "current_entry_literal"
    path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(checker.EvidenceError, match="reviewed release pin"):
        checker.check(evidence_copy)


@pytest.mark.parametrize("relative", ["../outside", "/outside", "C:/outside", "a\\b", "a/../b", "a//b"])
def test_path_escape_is_rejected(tmp_path, relative):
    with pytest.raises(checker.EvidenceError):
        checker.safe_path(tmp_path, relative)


def test_resolved_path_escape_is_rejected(tmp_path, monkeypatch):
    # Test resolution containment without requiring Windows symlink privileges.
    original = Path.resolve
    outside = tmp_path.parent / "outside"
    monkeypatch.setattr(Path, "resolve", lambda p: outside if p.name == "link" else original(p))
    with pytest.raises(checker.EvidenceError, match="escapes"):
        checker.safe_path(tmp_path, "link")


@pytest.mark.parametrize("payload", ['{"a":1,"a":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_ambiguous_json_is_rejected(tmp_path, payload):
    path = tmp_path / "input.json"
    path.write_text(payload)
    with pytest.raises(checker.EvidenceError):
        checker.load_json(path)


def test_json_pointer_and_typed_facts():
    assert checker.pointer({"a/b": {"~": [7]}}, "/a~1b/~0/0") == 7
    for address in ("/missing", "/x/01", "/x/2", "/x/~2", "x"):
        with pytest.raises(checker.EvidenceError):
            checker.pointer({"x": [1]}, address)
    assert not checker.typed_equal(True, 1)
    assert not checker.typed_equal({"x": [True]}, {"x": [1]})


def test_local_file_presence_cannot_close_historical_requirements(tmp_path):
    known = tmp_path / "known"
    known.write_bytes(b"original")
    records = {"sources": {
        "original:known": {"scope": "original", "relative_path": "known", "exists": True,
                           "bytes": 8, "sha256": hashlib.sha256(b"original").hexdigest()},
        "original:missing": {"scope": "original", "relative_path": "missing", "exists": False},
    }}
    roots = {"original": tmp_path}
    rows = checker.local_sources([records, records], roots)
    assert len(rows) == 2
    assert [r["status"] for r in rows] == ["matches_audited_bytes", "missing_at_audited_path"]
    known.write_bytes(b"modified")
    (tmp_path / "missing").write_bytes(b"new")
    rows = checker.local_sources([records], roots)
    assert [r["status"] for r in rows] == ["changed_requires_audit", "new_source_requires_audit"]
    known.unlink()
    assert checker.local_sources([records], roots)[0]["status"] == "changed_requires_audit"
    conflicting = json.loads(json.dumps(records))
    conflicting["sources"]["original:known"]["bytes"] = 9
    with pytest.raises(checker.EvidenceError, match="Conflicting"):
        checker.local_sources([records, conflicting], roots)


def test_cli_expected_exit_and_no_runner_options():
    result = subprocess.run([sys.executable, str(ROOT / "tools/check_kinneret_phase1_spec.py")],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 2
    report = json.loads(result.stdout)
    assert report["status"] == "open_prerequisites"
    assert report["historical_reproduction_ready"] is False
    assert report["execution_supported"] is False
    assert not result.stderr
