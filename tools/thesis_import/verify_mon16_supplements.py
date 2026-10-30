#!/usr/bin/env python3
"""Independent read-only verification using openpyxl (optional audit dependency).

Does not import the producing auditor. Compares original cells with published
diagnostics, including every A-3 value, depth gap and legacy sample gate.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import zipfile

import openpyxl


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def close(a, b):
    return abs(float(a) - float(b)) <= 1e-6


def verify(archive, correspondence, output):
    summary = json.loads((output / "summary.json").read_text())
    for name, expected in json.loads((output / "reference_manifest.json").read_text()).items():
        assert sha((output / name).read_bytes()) == expected, name
    assert sha(archive.read_bytes()) == summary["archive"]["sha256"]
    assert sha(correspondence.read_bytes()) == summary["pr21_correspondence_sha256"]
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for member in summary["source_members"]:
            content = z.read(member["file"])
            assert len(content) == member["bytes"] and sha(content) == member["sha256"]
        book = openpyxl.load_workbook(io.BytesIO(z.read(summary["source_members"][0]["file"])),
                                      read_only=True, data_only=True)
    one, two, three = book.worksheets
    a1 = {(cell.column_letter, cell.row): cell.value for row in one for cell in row
          if cell.value is not None}
    a2 = {(cell.column_letter, cell.row): cell.value for row in two for cell in row
          if cell.value is not None}
    assert three["B4"].value == "MON16 composite depth (mm)"
    rows = [(r, list(v[:5])) for r, v in enumerate(three.iter_rows(min_row=5, values_only=True), 5)
            if v[1] is not None]
    assert len(rows) == summary["a3"]["rows"]
    invalid = [(r, v) for r, v in rows if not v[3] <= v[2] <= v[4]]
    assert [r for r, _ in invalid] == [x["row"] for x in summary["a3"]["invalid_interval_rows"]]
    for (_, v), report in zip(invalid, summary["a3"]["invalid_interval_rows"]):
        assert all(close(v[i], report[key]) for i, key in
                   [(1, "depth_mm"), (2, "median"), (3, "minimum"), (4, "maximum")])
    widths = [v[4] - v[3] for _, v in rows]
    for key, actual in [("min", min(widths)), ("median", statistics.median(widths)), ("max", max(widths))]:
        assert close(actual, summary["a3"]["interval_width_years"][key])
    assert all(b[1][i] >= a[1][i] - 1e-6 for a, b in zip(rows, rows[1:]) for i in (1, 2, 3, 4))
    gaps = [(a, b) for a, b in zip(rows, rows[1:]) if abs(b[1][1] - a[1][1] - 1) > 1e-6]
    recorded_gaps = read_csv(output / "depth_steps.csv")
    assert len(gaps) == len(recorded_gaps) == summary["a3"]["non_1mm_steps"]
    for (a, b), report in zip(gaps, recorded_gaps):
        assert a[0] == int(report["previous_row"]) and b[0] == int(report["row"])
        assert close(a[1][1], report["previous_depth_mm"]) and close(b[1][1], report["depth_mm"])
        assert close(b[1][1] - a[1][1], report["step_mm"])
    groups = read_csv(output / "labelled_age_groups.csv")
    covered = set()
    lookup = dict(rows)
    for g in groups:
        first, last = int(g["first_row"]), int(g["last_row"])
        assert last - first + 1 == int(g["rows"])
        assert close(lookup[first][1], g["first_depth_mm"])
        assert close(lookup[last][1], g["last_depth_mm"])
        for r in range(first, last + 1):
            v = lookup[r]
            assert v[0] == g["label"]
            assert all(close(v[i], g[k]) for i, k in [(2, "median"), (3, "minimum"), (4, "maximum")])
            assert r not in covered
            covered.add(r)
    assert covered == {r for r, v in rows if v[0] is not None}
    # A-2 exported fields are direct source values, not recalibrated dates.
    anchors = read_csv(output / "anchors.csv")
    fields = list(anchors[0])[1:]
    assert len(anchors) == 16
    for a in anchors:
        r = int(a["a2_row"])
        for col, field in zip("ABCDEFGHIJKLMN", fields):
            actual = a2[col, r]
            if isinstance(actual, (int, float)):
                assert close(actual, a[field]), (col, r)
            else:
                assert actual == a[field], (col, r)
    conflicts = read_csv(output / "source_conflicts.csv")
    for conflict in conflicts:
        ref = conflict["a1_cell"]
        col = "".join(x for x in ref if x.isalpha())
        actual = a1[col, int(ref[len(col):])]
        expected = conflict["a1_value"]
        assert close(actual, expected) if isinstance(actual, (int, float)) else actual == expected
        loc = conflict["other_location"]
        if loc.startswith("A-2!"):
            ref = loc[4:]
            col = "".join(x for x in ref if x.isalpha())
            actual = a2[col, int(ref[len(col):])]
        elif loc.startswith("10*"):
            r = int(conflict["a1_cell"][1:])
            actual = 10 * (a1["C", r] - a1["D", r])
        else:
            r = int(loc.split("row")[1].split("-")[0])
            actual = lookup[r][{"median": 2, "minimum": 3, "maximum": 4}[conflict["field"]]]
        expected = conflict["other_value"]
        assert close(actual, expected) if isinstance(actual, (int, float)) else actual == expected
    original, gates = read_csv(correspondence), read_csv(output / "sample_gate.csv")
    assert len(gates) == len(original) == 69
    for x, g in zip(original, gates):
        assert g["sample_id"] == x["sample_id"]
        for new, old in [("legacy_pollen_depth_m", "pollen_depth_m"),
                ("legacy_chronology_depth_m", "chronology_depth_m"),
                ("legacy_age_bp", "epd_cut_age_bp"), ("pr21_correspondence_status", "status")]:
            assert g[new] == x[old]
        assert g["mon16_depth_mm"] == g["mon16_age_cal_bp"] == ""
        assert g["mon16_transfer_status"] == "blocked_no_documented_lgm90d_mon16_depth_crosswalk"
    assert summary["gate"]["automatic_go_to_fossil_climate_run"] is False
    return {"method": "Independent openpyxl reader and float arithmetic; no producer imports",
            "openpyxl_version": openpyxl.__version__, "a3_rows_checked": len(rows),
            "a3_numeric_cells_checked": len(rows) * 4, "interval_violations_confirmed": len(invalid),
            "depth_gaps_confirmed": len(gaps), "labelled_rows_checked": len(covered),
            "anchors_checked": len(anchors), "reported_source_conflicts_checked": len(conflicts),
            "legacy_sample_gates_checked": len(gates), "output_hashes_checked": 6, "passed": True}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--correspondence", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(verify(args.archive, args.correspondence, args.output_dir), indent=2))
