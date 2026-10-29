#!/usr/bin/env python3
"""Reproduce source checks; no model inputs or scientific approval are produced.

python verify_sources.py --raw-dir /path/to/downloads --output audit_results.json
Add --download to retrieve missing files from the pinned sources.json URLs.
Existing files are never overwritten. A hash mismatch stops the audit.
"""
import argparse
import collections
import csv
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import statistics
import urllib.request


def table(path):
    text = path.read_text(encoding="utf-8")
    if "*/" not in text:
        raise ValueError(f"Missing PANGAEA metadata terminator: {path}")
    rows = list(csv.reader(io.StringIO(text.split("*/", 1)[1].lstrip()), delimiter="\t"))
    header, data = rows[0], rows[1:]
    assert len(header) == len(set(header)), path
    assert all(len(r) == len(header) for r in data), path
    return header, [dict(zip(header, r)) for r in data]


def age_stats(values):
    values = sorted(v for v in values if 0 <= v <= 150)
    gaps = [b - a for a, b in zip(values, values[1:])]
    return dict(n=len(values), min_ka_bp=min(values), max_ka_bp=max(values),
                median_gap_ka=round(statistics.median(gaps), 8),
                max_gap_ka=round(max(gaps), 8))


def audit(raw):
    results = {}
    with sqlite3.connect(f"file:{(raw / 'empd2.sqlite3').resolve()}?mode=ro", uri=True) as con:
        def scalar(sql):
            return con.execute(sql).fetchone()[0]
        assert scalar("PRAGMA integrity_check") == "ok"
        results["empd_sqlite"] = {
            "integrity_check": "ok",
            "table_rows": {n: scalar(f'SELECT count(*) FROM "{n}"') for n in
                           ("metadata", "climate", "p_counts", "p_vars")},
            "measured_original_codes": scalar("SELECT count(DISTINCT var_) FROM p_counts"),
            "measured_accepted_ids": scalar("SELECT count(DISTINCT acc_var_) FROM p_vars JOIN p_counts USING(var_)"),
            "negative_counts": scalar("SELECT count(*) FROM p_counts WHERE count < 0"),
            "fractional_count_cells": scalar("SELECT count(*) FROM p_counts WHERE count != cast(count AS integer)"),
            "samples_with_fractional_counts": scalar("SELECT count(DISTINCT samplename) FROM p_counts WHERE count != cast(count AS integer)"),
            "null_tann_or_pann": scalar("SELECT count(*) FROM climate WHERE t_ann IS NULL OR p_ann IS NULL"),
            "ispercent_isdigitised_groups": con.execute("SELECT ispercent,isdigitised,count(*) FROM metadata GROUP BY 1,2").fetchall(),
            "regional_box_samples": scalar("SELECT count(*) FROM metadata WHERE latitude BETWEEN 25 AND 75 AND longitude BETWEEN -15 AND 45"),
        }
        _, climate = table(raw / "empd2_climate_909130.tab")
        tsv_ids = {r["Sample ID"] for r in climate}
        db_ids = {r[0] for r in con.execute("SELECT samplename FROM metadata")}
        results["empd_climate_tsv"] = dict(rows=len(climate), unique_ids=len(tsv_ids),
                                           tsv_only_ids=len(tsv_ids-db_ids), sqlite_only_ids=len(db_ids-tsv_ids))

    for filename in ("ohrid_892362.tab", "monticchio_739452.tab", "monticchio_739556.tab", "iznik_858056.tab"):
        header, rows = table(raw / filename)
        item = dict(rows=len(rows), columns=len(header), header=header)
        if "Age [ka BP]" in header:
            item["age_0_150ka"] = age_stats([float(r["Age [ka BP]"]) for r in rows])
        if "Depth sed [m]" in header:
            ds = [float(r["Depth sed [m]"]) for r in rows]
            item["depth_min_m"], item["depth_max_m"] = min(ds), max(ds)
        counts = [Decimal(v) for r in rows for k, v in r.items() if "[#]" in k and v]
        item["negative_count_cells"] = sum(v < 0 for v in counts)
        item["fractional_count_cells"] = sum(v != int(v) for v in counts)
        results[filename] = item

    _, pollen = table(raw / "monticchio_739452.tab")
    _, all_ages = table(raw / "epd_ages_804595.tab")
    _, cut_ages = table(raw / "epd_ages_cut_804596.tab")
    all_ages = [r for r in all_ages if r["Event"] == "LGM90D" and r["ID"] == "932"]
    cut_ages = [r for r in cut_ages if r["Event"] == "LGM90D" and r["ID"] == "932"]
    joined, used, exact = [], set(), 0
    for age in cut_ages:
        depth = Decimal(age["Depth sed [m]"])
        candidates = [r for r in pollen if abs(Decimal(r["Depth sed [m]"])-depth) <= Decimal("0.005")]
        assert len(candidates) == 1, (depth, len(candidates))
        match = candidates[0]["Depth sed [m]"]
        assert match not in used, match
        used.add(match)
        offset = Decimal(match)-depth
        exact += offset == 0
        joined.append(dict(pollen_depth_m=match, chronology_depth_m=str(depth),
                           offset_m=str(offset), age_ka_bp=age["Age dated [ka BP]"],
                           age_min_ka=age["Age min [ka]"], age_max_ka=age["Age max [ka]"],
                           uncertainty_class=age["Uncertainty class"]))
    results["monticchio_chronology"] = dict(
        event="LGM90D", entity_id="932", all_age_rows=len(all_ages), cut_age_rows=len(cut_ages),
        exact_depth_matches=exact, rounded_depth_matches=len(joined)-exact,
        excluded_pollen_rows=len(pollen)-len(joined),
        cut_age_stats=age_stats([float(r["Age dated [ka BP]"]) for r in cut_ages]),
        join_tolerance_m="0.005", join_is_candidate_not_scientific_approval=True,
        join=joined)

    header, rows = table(raw / "ohrid_892362.tab")
    taxon_cols = header[2:21]  # individual types; excludes AP and six terminal aggregates
    sums = [sum(Decimal(r[k]) for k in taxon_cols) for r in rows]
    results["ohrid_composition_check"] = dict(
        selected_columns=taxon_cols, min_sum=float(min(sums)), max_sum=float(max(sums)),
        rows_over_100=sum(x > 100 for x in sums),
        interpretation="Selected types with mixed denominator definitions: corrigendum Table 1 excludes Pinus from AP/NAP basis; archive Pinus header includes Pinus. Preserve column definitions; do not close or renormalise together.")

    with (raw / "flora_mapping.csv").open(encoding="utf-8", newline="") as stream:
        mapping = list(csv.DictReader(stream))
    geo = json.loads((raw / "flora_1.geojson").read_text())
    results["flora"] = dict(
        mapping_rows=len(mapping), unique_taxon_ids=len({r["taxon_id"] for r in mapping}),
        duplicate_synonym_rows=sum(r["duplicate_synonym"] == "TRUE" for r in mapping),
        olea_genus_rows=sum(r["name_choro_orig"].split()[0] == "Olea" for r in mapping),
        tested_taxon="Abies alba", features=len(geo["features"]), crs=geo["crs"],
        geometry_types=dict(collections.Counter(f["geometry"]["type"] for f in geo["features"])),
        feature_property_keys=sorted(geo["features"][0]["properties"]))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    catalog = json.loads(Path(__file__).with_name("sources.json").read_text())
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    verified = []
    for source in catalog["files"]:
        path = args.raw_dir / source["file"]
        if not path.exists() and args.download:
            with urllib.request.urlopen(source["download_url"], timeout=120) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != source["sha256"]:
                raise ValueError(f"Source changed; register a new version: {source['source_id']}")
            path.write_bytes(data)
        data = path.read_bytes()
        assert len(data) == source["byte_count"], path
        assert hashlib.sha256(data).hexdigest() == source["sha256"], path
        verified.append(source["source_id"])
    output = dict(schema_version=1, verified_files=verified, results=audit(args.raw_dir))
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(f"Verified {len(verified)} files; results written to {args.output}")


if __name__ == "__main__":
    main()
