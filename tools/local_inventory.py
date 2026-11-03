"""Metadatenkatalog einer unveraenderten lokalen Quelle, ohne Datenkopien.

Aufruf: python tools/local_inventory.py --config .local/paths.json
Nur die Standardbibliothek; alle Ausgaben muessen ausserhalb der Quelle liegen.
Keine Ausfuehrung von R-Code, kein Entpacken, kein Netzwerk, kein Vollhashing.
Rollen sind heuristische Hinweise und keine wissenschaftliche Zuordnung.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile


def role(path: Path) -> str:
    if path.suffix.lower() in {".tex", ".bib", ".cls", ".sty"}:
        return "document_source_candidate"
    if path.name.lower() == "diss.pdf":
        return "dissertation_candidate_version_unverified"
    if path.suffix.lower() in {".r", ".py", ".cpp", ".hpp"}:
        return "source_code"
    if path.suffix.lower() == ".zip":
        return "archive_uninspected_contents"
    if path.parts[0] == "in":
        return "input_candidate"
    if path.parts[0] == "out":
        return "output_candidate"
    if path.name.startswith("logfile"):
        return "existing_log"
    return "unassigned"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
    source = Path(cfg["original_data_root"]).resolve(strict=True)
    output = Path(cfg["inventory_dir"]).resolve()
    if not source.is_dir():
        raise ValueError("Die Quelle muss ein Verzeichnis sein")
    if output == source or source in output.parents:
        raise ValueError("Ausgaben duerfen nicht innerhalb der Originalquelle liegen")
    # Erst alle Metadaten lesen; bei Lesefehlern keinen vollstaendigen Katalog behaupten.
    rows = []
    archives = []
    extensions: Counter[str] = Counter()
    folders: Counter[str] = Counter()
    extension_bytes: Counter[str] = Counter()
    folder_bytes: Counter[str] = Counter()
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symbolische Links erfordern eine explizite Quellenentscheidung")
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        stat = path.stat()
        extension = path.suffix.lower() or "[none]"
        folder = relative.parts[0] if len(relative.parts) > 1 else "[root]"
        extensions[extension] += 1
        folders[folder] += 1
        extension_bytes[extension] += stat.st_size
        folder_bytes[folder] += stat.st_size
        rows.append(dict(relative_path=relative.as_posix(), extension=extension,
                         size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                         role_hint=role(relative), role_status="inferred_not_validated"))
        if extension == ".zip":
            with zipfile.ZipFile(path) as archive:
                members = archive.infolist()
                archives.append(dict(relative_path=relative.as_posix(), members=len(members),
                                     uncompressed_bytes=sum(m.file_size for m in members),
                                     inspection="central_directory_only_no_crc_test_no_extraction"))
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                   file_count=len(rows), total_bytes=sum(r["size_bytes"] for r in rows),
                   formats=[dict(extension=k, count=extensions[k], bytes=extension_bytes[k])
                            for k in sorted(extensions)],
                   folders=[dict(folder=k, count=folders[k], bytes=folder_bytes[k])
                            for k in sorted(folders)], archives=archives,
                   source_modified_by_tool=False, full_hashing_performed=False,
                   scientific_status="not_assessed")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "data_catalog.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["relative_path", "extension", "size_bytes",
                                                   "mtime_ns", "role_hint", "role_status"])
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                         encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("file_count", "total_bytes", "scientific_status")}))


if __name__ == "__main__":
    main()
