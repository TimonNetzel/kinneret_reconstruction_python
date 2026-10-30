#!/usr/bin/env python3
"""Read-only, hash-locked MON16 supplement intake. No age transfer or modelling.

Uses stored OOXML cell values, never evaluates formulas or modifies workbooks.
The source workbook is not a posterior ensemble. See the research README.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import posixpath
import statistics
import xml.etree.ElementTree as ET
import zipfile

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
LOCKS = {
    "1-s2.0-S0277379126003252-mmc1.xlsx": (447272, "464da840bafe2b508da20e8d78062e6928f62d41cf7b7025b43670c954feedd9"),
    "1-s2.0-S0277379126003252-mmc2.xlsx": (776950, "57bf8feac10d7b537f519bdc140fddf4f4f1319a9298c3d2cedd543a6d155d3d"),
    "1-s2.0-S0277379126003252-mmc3.docx": (1759486, "a4840acda246b1b5136b739bf086ae703f600e0020972306526d1b08138de436"),
}
CROSSWALK_SHA = "b3b9247594539bd0afd858d92fde2172bc3ff8b199c63bd106b506f82a122f83"
EPS = Decimal("0.000001")  # XML binary-float tails, not scientific uncertainty
D = Decimal


def digest(data):
    return hashlib.sha256(data).hexdigest()


def locked(data, size, sha):
    if len(data) != size or digest(data) != sha:
        raise ValueError("Source size/SHA256 mismatch")


def read_workbook(data):
    """Resolve sheet relationships; preserve decimal XML values and cell addresses."""
    result = {}
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            strings = ["".join(t.itertext()) for t in ET.fromstring(z.read("xl/sharedStrings.xml"))]
        rels = {r.attrib["Id"]: r.attrib["Target"] for r in
                ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
        for sheet in ET.fromstring(z.read("xl/workbook.xml")).findall("s:sheets/s:sheet", NS):
            target = rels[sheet.attrib[f"{{{REL}}}id"]]
            path = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
            cells, formulas, rows = {}, {}, set()
            for c in ET.fromstring(z.read(path)).findall(".//s:sheetData/s:row/s:c", NS):
                ref = c.attrib["r"]
                v, f = c.find("s:v", NS), c.find("s:f", NS)
                kind = c.get("t", "n")
                value = v.text if v is not None else ""
                if f is not None:
                    formulas[ref] = f.text or ""
                    if v is None or v.text is None:
                        raise ValueError(f"Formula without cached value: {ref}")
                if kind == "s":
                    value = strings[int(value)]
                elif kind == "inlineStr":
                    value = "".join(c.find("s:is", NS).itertext())
                elif value and kind == "n":
                    value = D(value)
                elif kind == "e":
                    raise ValueError(f"Spreadsheet error at {ref}: {value}")
                if value != "":
                    cells[ref] = value
                    rows.add(int("".join(x for x in ref if x.isdigit())))
            result[sheet.attrib["name"]] = {"cells": cells, "formulas": formulas,
                                             "nonempty_rows": len(rows)}
    return result


def data_rows(sheet, first, columns):
    cells = sheet["cells"]
    indexes = sorted({int("".join(x for x in ref if x.isdigit())) for ref in cells})
    return [(r, [cells.get(f"{c}{r}", "") for c in columns]) for r in indexes
            if r >= first and cells.get(f"{columns[0]}{r}", "") != ""]


def check_series(rows):
    """Numerical diagnostics only. Inconsistent source rows remain unchanged."""
    invalid, jumps = [], []
    decreases = {k: [] for k in ("depth", "median", "minimum", "maximum")}
    previous = None
    for row, values in rows:
        if not all(isinstance(x, D) and x.is_finite() for x in values):
            raise ValueError(f"Missing/non-numeric A-3 data at row {row}")
        depth, med, low, high = values
        if not low <= med <= high:
            invalid.append({"row": row, "depth_mm": depth, "median": med,
                            "minimum": low, "maximum": high, "median_minus_maximum": med - high})
        if previous:
            pr, pv = previous
            for key, a, b in zip(decreases, pv, values):
                if b < a - EPS:
                    decreases[key].append(row)
            if abs(depth - pv[0] - 1) > EPS:
                jumps.append({"previous_row": pr, "row": row, "previous_depth_mm": pv[0],
                              "depth_mm": depth, "step_mm": depth - pv[0]})
        previous = row, values
    return invalid, jumps, decreases


def same(a, b, tolerance=EPS):
    if isinstance(a, D) and isinstance(b, D):
        return abs(a - b) <= tolerance
    return a == b


def serial(value):
    if isinstance(value, D):
        return format(value.normalize(), "f")
    raise TypeError(type(value).__name__)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=serial) + "\n")


def save_csv(path, rows, fields):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: serial(v) if isinstance(v, D) else v for k, v in row.items()})


def run(archive, correspondence, output):
    if output.exists():
        raise ValueError("Output directory must be new")
    sources, books = [], []
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None or set(z.namelist()) != set(LOCKS):
            raise ValueError("Unexpected/corrupt archive members")
        for name, (size, sha) in LOCKS.items():
            content = z.read(name)
            locked(content, size, sha)
            sources.append({"file": name, "bytes": size, "sha256": sha})
            if name.endswith("xlsx"):
                books.append(read_workbook(content))
            else:
                with zipfile.ZipFile(io.BytesIO(content)) as doc:
                    t = ET.fromstring(doc.read("word/document.xml"))
                    text = " ".join(x.text or "" for x in t.iter()
                                    if x.tag.endswith("}t"))
                    if "Supplement Figure C-2" not in text or "Potassium" not in text:
                        raise ValueError("Unexpected Supplement C captions")
                    embedded = [x for x in doc.namelist() if x.startswith("word/embeddings/")]
    cross_bytes = correspondence.read_bytes()
    if digest(cross_bytes) != CROSSWALK_SHA:
        raise ValueError("PR #21 correspondence SHA256 mismatch")
    cross = list(csv.DictReader(io.StringIO(cross_bytes.decode())))
    if len(cross) != 69 or len({x["sample_id"] for x in cross}) != 69:
        raise ValueError("Expected 69 unique legacy samples")
    book = books[0]
    a1, a2, a3 = (book[f"Supplement Table A-{i}"] for i in (1, 2, 3))
    if a3["cells"].get("B4") != "MON16 composite depth (mm)":
        raise ValueError("A-3 depth scale/unit changed")
    if [a3["cells"].get(c + "4") for c in "CDE"] != [
            "Age_med (cal BP)", "Age_min (cal BP)", "Age_max (cal BP)"]:
        raise ValueError("A-3 age schema changed")
    series = data_rows(a3, 5, "BCDE")
    invalid, jumps, decreases = check_series(series)
    # Explicit labels, never an age/depth based match.
    a1_rows = {str(v[0]).strip(): (r, v) for r, v in data_rows(a1, 7, "ABCDEFGHIJKLMNOPQRST")}
    anchors, conflicts = [], []
    for r, v in data_rows(a2, 6, "ABCDEFGHIJKLMN"):
        label = str(v[0]).strip()
        key = {"TM-6-5a-c": "TM-6-5a-c (MON16)",
               "TM-8-1b": "TM-8-1b (new)"}.get(label, label)
        ar, av = a1_rows[key]
        anchors.append(dict(zip(("a2_row", "tephra", "core_section", "event", "published_age_cal_bp",
                                 "published_uncertainty_years", "date_type", "age_reference",
                                 "mon16_top_cm", "mon16_base_cm", "modelled_age_text",
                                 "mon12_top_cm", "mon12_base_cm", "mon12_varve_age_cal_bp",
                                 "mon12_uncertainty_5pct_years"), [r] + v)))
        for ai, bi, ac, bc, field in [(1, 1, "B", "B", "core_section"),
                (3, 7, "D", "H", "mon16_top_cm"), (2, 8, "C", "I", "mon16_base_cm"),
                (10, 11, "K", "L", "mon12_base_cm")]:
            if not same(av[ai], v[bi]):
                conflicts.append({"kind": "cross_table_value_difference", "tephra": label,
                    "field": field, "a1_cell": f"{ac}{ar}", "a1_value": av[ai],
                    "other_location": f"A-2!{bc}{r}", "other_value": v[bi]})
    # A-1 thickness arithmetic; cached formula values are not recomputed in originals.
    for label, (r, v) in a1_rows.items():
        if all(isinstance(v[i], D) for i in (2, 3, 4)):
            implied = (v[2] - v[3]) * 10
            if not same(implied, v[4], D("0.01")):
                conflicts.append({"kind": "thickness_arithmetic", "tephra": label,
                    "field": "thickness_mm", "a1_cell": f"E{r}", "a1_value": v[4],
                    "other_location": f"10*(A-1!C{r}-D{r})", "other_value": implied})
    groups = []
    for r, values in series:
        label = str(a3["cells"].get(f"A{r}", ""))
        if not label:
            continue
        # Group contiguous labelled rows with the same complete age triplet.
        if groups and groups[-1]["label"] == label and groups[-1]["last_row"] == r - 1 and all(
                same(groups[-1][k], val) for k, val in zip(("median", "minimum", "maximum"), values[1:])):
            groups[-1]["last_row"] = r
            groups[-1]["last_depth_mm"] = values[0]
            groups[-1]["rows"] += 1
        else:
            groups.append(dict(zip(("label", "first_row", "last_row", "first_depth_mm", "last_depth_mm",
                                   "rows", "median", "minimum", "maximum"),
                                  [label, r, r, values[0], values[0], 1] + values[1:])))
    for group in groups:
        key = group["label"].split()[0]
        if key == "TM-6-5a-c":
            key += " (MON16)"
        elif key in ("TM-4-1", "TM-8-1b", "TM-9-1"):
            key += " (new)"
        elif key == "TM-8-1a":
            key += " (new, rew.)"
        r, v = a1_rows[key]
        if not isinstance(v[5], D):
            continue
        for col, name, value in zip("FGH", ("median", "minimum", "maximum"), v[5:8]):
            if isinstance(value, D) and not same(value, group[name], D("1.1")):
                conflicts.append({"kind": "labelled_age_difference_gt_1p1yr", "tephra": key,
                    "field": name, "a1_cell": f"{col}{r}", "a1_value": value,
                    "other_location": f"A-3!row{group['first_row']}-{group['last_row']}",
                    "other_value": group[name]})
    samples = [{"sample_id": x["sample_id"], "legacy_pollen_depth_m": x["pollen_depth_m"],
        "legacy_chronology_depth_m": x["chronology_depth_m"], "legacy_age_bp": x["epd_cut_age_bp"],
        "pr21_correspondence_status": x["status"], "mon16_depth_mm": "", "mon16_age_cal_bp": "",
        "mon16_transfer_status": "blocked_no_documented_lgm90d_mon16_depth_crosswalk"} for x in cross]
    widths = [v[3] - v[2] for _, v in series]
    types = {}
    for a in anchors:
        types[a["date_type"]] = types.get(a["date_type"], 0) + 1
    inventory = [{"file": sources[i]["file"], "sheet": name, "nonempty_rows": sh["nonempty_rows"],
                  "nonempty_cells": len(sh["cells"]), "cached_formulas": len(sh["formulas"])}
                 for i, wb in enumerate(books) for name, sh in wb.items()]
    summary = {"schema_version": 1, "date": "2026-10-29", "source_members": sources,
        "audit_code_sha256": digest(Path(__file__).read_bytes()),
        "archive": {"bytes": archive.stat().st_size, "sha256": digest(archive.read_bytes())},
        "pr21_correspondence_sha256": CROSSWALK_SHA, "sheets": inventory,
        "a3": {"rows": len(series), "first_depth_mm": series[0][1][0], "last_depth_mm": series[-1][1][0],
            "first_median_cal_bp": series[0][1][1], "last_median_cal_bp": series[-1][1][1],
            "minimum_bound_cal_bp": min(v[2] for _, v in series),
            "maximum_bound_cal_bp": max(v[3] for _, v in series),
            "interval_width_years": {"min": min(widths), "median": statistics.median(widths), "max": max(widths)},
            "invalid_interval_rows": invalid, "non_1mm_steps": len(jumps), "decreases": decreases,
            "labelled_groups": len(groups), "labelled_unique_strings": len({x["label"] for x in groups})},
        "a2": {"anchors": len(anchors), "date_types": types}, "cross_table_flags": len(conflicts),
        "supplement_c_embedded_data_files": embedded,
        "gate": {"phase": "P1", "status": "supplements_received_depth_crosswalk_and_source_conflicts_unresolved",
            "original_supplements_received": True, "lgm90d_samples": len(samples), "mon16_age_assignments": 0,
            "joint_posterior_draws_present": False, "bacon_input_settings_present": False,
            "automatic_go_to_fossil_climate_run": False}}
    output.mkdir(parents=True)
    save_json(output / "summary.json", summary)
    for name, values, fields in [
        ("anchors.csv", anchors, list(anchors[0])),
        ("source_conflicts.csv", conflicts, list(conflicts[0])),
        ("labelled_age_groups.csv", groups, list(groups[0])),
        ("depth_steps.csv", jumps, list(jumps[0])),
        ("sample_gate.csv", samples, list(samples[0]))]:
        save_csv(output / name, values, fields)
    save_json(output / "reference_manifest.json", {p.name: digest(p.read_bytes())
              for p in sorted(output.iterdir())})
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", required=True, type=Path)
    p.add_argument("--correspondence", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()
    summary = run(args.archive, args.correspondence, args.output_dir)
    print(json.dumps(summary["gate"]))


if __name__ == "__main__":
    main()
