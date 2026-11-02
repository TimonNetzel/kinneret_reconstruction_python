"""Read-only integrity check of the frozen Kinneret Phase-1 evidence contract.

Exit 2 means intact evidence with unresolved original-run prerequisites; it is
expected for this release. This tool cannot authorize or start a reconstruction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

SPEC_PATH = "docs/thesis_research/kinneret_phase1_spec_2026_11_02/spec.json"
SPEC_SHA256 = "a76f15b32aa5b2c4cbbd30155147d64060b9c2990fcadb2c36164c78c9c117d4"
REQUIREMENTS = {
    "final_base", "final_chronology", "historical_tf", "isotope_workspaces",
    "cru_kgc_inputs", "reference_variant", "anchor_variant", "posterior_selection",
    "postprocessing_variant", "runtime_rng", "numeric_original_output",
}


class EvidenceError(ValueError):
    """An evidence file or its binding failed verification."""


def _object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise EvidenceError("Non-finite JSON number: " + value)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object,
                      parse_constant=_nonfinite)


def safe_path(root: Path, relative: str) -> Path:
    """Resolve a portable relative path and reject escapes, including symlinks."""
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or ":" in relative or relative.startswith("/")
            or any(part in ("", ".", "..") for part in relative.split("/"))):
        raise EvidenceError("Unsafe relative evidence path")
    root = root.resolve()
    result = root.joinpath(*PurePosixPath(relative).parts).resolve()
    if not result.is_relative_to(root):
        raise EvidenceError("Evidence path escapes its root")
    return result


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_file(path: Path, size: int, sha256: str, label: str) -> None:
    if not path.is_file():
        raise EvidenceError("Missing evidence: " + label)
    if path.stat().st_size != size or digest(path) != sha256:
        raise EvidenceError("Evidence bytes differ: " + label)


def pointer(document: Any, address: str) -> Any:
    if address == "":
        return document
    if not isinstance(address, str) or not address.startswith("/"):
        raise EvidenceError("Invalid JSON pointer")
    value = document
    for part in address[1:].split("/"):
        if re.search(r"~(?![01])", part):
            raise EvidenceError("Invalid JSON pointer escape")
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and re.fullmatch(r"0|[1-9][0-9]*", part):
            index = int(part)
            if index >= len(value):
                raise EvidenceError("JSON pointer index out of range")
            value = value[index]
        else:
            raise EvidenceError("Unbound JSON pointer: " + address)
    return value


def typed_equal(left: Any, right: Any) -> bool:
    # Python bool == int is inappropriate for scientific evidence contracts.
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(typed_equal(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(typed_equal(a, b) for a, b in zip(left, right))
    return left == right


def local_sources(documents: list[dict], roots: dict[str, Path]) -> list[dict]:
    """Only check explicitly recorded original/additional paths, never discover files."""
    sources = {}
    for document in documents:
        for label, record in document["sources"].items():
            if record["scope"] not in ("original", "additional"):
                continue
            if label in sources and not typed_equal(sources[label], record):
                raise EvidenceError("Conflicting source records: " + label)
            sources[label] = record
    rows = []
    for label, record in sorted(sources.items()):
        path = safe_path(roots[record["scope"]], record["relative_path"])
        exists = path.is_file()
        actual_hash = digest(path) if exists else None
        actual_bytes = path.stat().st_size if exists else None
        if record["exists"]:
            match = (exists and actual_hash == record["sha256"]
                     and actual_bytes == record["bytes"])
            status = "matches_audited_bytes" if match else "changed_requires_audit"
        else:
            status = "new_source_requires_audit" if exists else "missing_at_audited_path"
        rows.append({"source": label, "status": status,
                     "observed_sha256": actual_hash, "observed_bytes": actual_bytes})
    return rows


def check(repo: Path, *, with_local: bool = False) -> dict:
    spec_path = safe_path(repo, SPEC_PATH)
    if not spec_path.is_file() or digest(spec_path) != SPEC_SHA256:
        raise EvidenceError("Specification differs from the reviewed release pin")
    # The pin covers every key, value and byte, including unresolved statuses.
    # No local override, readiness flag or variant selector is accepted.
    spec = load_json(spec_path)
    documents = {}
    paths = {}
    for name, entry in spec["assets"].items():
        path = safe_path(repo, entry["path"])
        verify_file(path, entry["bytes"], entry["sha256"], name)
        paths[name] = path
        if path.suffix == ".json":
            documents[name] = load_json(path)

    def anchor(reference: dict) -> Any:
        return pointer(documents[reference["artifact"]], reference["pointer"])

    for fact in spec["facts"]:
        if not typed_equal(anchor(fact), fact["equals"]):
            raise EvidenceError("Fact does not match evidence: " + fact["id"])
    if {r["id"] for r in spec["requirements"]} != REQUIREMENTS:
        raise EvidenceError("Incomplete prerequisites")
    requirements = []
    for requirement in spec["requirements"]:
        requirements.append({**requirement, "observed_evidence": [
            anchor(reference) for reference in requirement["evidence"]]})
    variants = {key: {**value, "observed_evidence": anchor(value["evidence"])}
                for key, value in spec["variants"].items()}
    manifest_id = spec["research_snapshot"]["manifest"]
    manifest = documents[manifest_id]
    for entry in manifest["entries"]:
        path = safe_path(paths[manifest_id].parent, entry["published"])
        verify_file(path, entry["published_bytes"], entry["published_sha256"], entry["published"])
    rows = None
    if with_local:
        config = load_json(safe_path(repo, ".local/my_reconst_paths.json"))
        roots = {scope: Path(config[key]) for scope, key in (
            ("original", "original_data_root"), ("additional", "additional_source_root"))}
        if any(not root.is_absolute() or not root.is_dir() for root in roots.values()):
            raise EvidenceError("Configured source roots must be existing absolute directories")
        rows = local_sources([documents[name] for name in spec["local_source_documents"]], roots)
    audit_required = rows is not None and any("requires_audit" in row["status"] for row in rows)
    return {
        "spec_id": spec["spec_id"], "spec_sha256": SPEC_SHA256,
        "spec_valid": True,
        "status": "local_source_audit_required" if audit_required else "open_prerequisites",
        "historical_reproduction_ready": False, "execution_supported": False,
        "verified_assets": len(paths), "verified_facts": len(spec["facts"]),
        "verified_snapshot_files": len(manifest["entries"]),
        "case": spec["case"], "selected_variant": spec["selected_variant"],
        "requirements": requirements, "variants": variants,
        "preserved_gates": {**spec["preserved_gates"], "observed": anchor(spec["preserved_gates"])},
        "local_sources": rows,
        "local_check_scope": "Only original/additional paths recorded in the three source documents; not every dependency or global CRU path",
        "exclusions": spec["exclusions"],
        "interpretation": "Integrity of frozen evidence is not scientific validation or permission to run; new files require a new source audit and reviewed specification",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-local", action="store_true", help="Read explicitly recorded source paths using the private local path configuration")
    args = parser.parse_args()
    try:
        report = check(Path(__file__).resolve().parents[1], with_local=args.with_local)
    except (EvidenceError, OSError, KeyError, TypeError, ValueError) as error:
        # Do not expose private absolute paths from filesystem exceptions.
        message = str(error) if isinstance(error, EvidenceError) else type(error).__name__
        print(json.dumps({"spec_valid": False, "status": "invalid_evidence",
                          "historical_reproduction_ready": False, "execution_supported": False,
                          "error": message}, ensure_ascii=True, indent=2))
        return 1
    print(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
