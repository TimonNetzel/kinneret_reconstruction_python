#!/usr/bin/env python3
"""Import locked EMPD2 and Monticchio sources without scientific harmonisation.

Python >=3.11, standard library only. See README.md for the output contract.
"""
import argparse
import collections
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import statistics
import tempfile
import urllib.request

REQUIRED = ("empd2", "monticchio_739452", "epd_entity_804591", "epd_ages_cut_804596")
TOLERANCE_M = Decimal("0.005")


class ImportFailure(ValueError):
    """An input violates the locked source or data contract."""


def require(condition, message):
    if not condition:
        raise ImportFailure(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def number(value):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ImportFailure(f"Invalid numeric value: {value!r}") from exc
    require(result.is_finite(), f"Non-finite value: {value!r}")
    return result


def fixed(value):
    return format(number(value), "f")


def missing(value):
    return value is None or (isinstance(value, str) and value.strip() in ("", "NULL"))


def optional_numeric(value):
    return "" if missing(value) else fixed(value)


def write_csv(path, fields, rows):
    count = 0
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def read_pangaea(path):
    text = path.read_text(encoding="utf-8")
    require("*/" in text, f"Missing PANGAEA header: {path.name}")
    metadata, data = text.split("*/", 1)
    parsed = list(csv.reader(io.StringIO(data.lstrip()), delimiter="\t"))
    require(bool(parsed), f"Empty table: {path.name}")
    header, rows = parsed[0], parsed[1:]
    require(len(header) == len(set(header)), f"Duplicate columns: {path.name}")
    require(all(len(r) == len(header) for r in rows), f"Malformed rows: {path.name}")
    return metadata, header, [dict(zip(header, row)) for row in rows]


def locked_sources(raw_dir, catalog_path, download=False):
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    by_id = {}
    for source in catalog["files"]:
        require(source["source_id"] not in by_id, "Duplicate source ID")
        by_id[source["source_id"]] = source
    selected = {}
    for source_id in REQUIRED:
        require(source_id in by_id, f"Missing source ID: {source_id}")
        source = by_id[source_id]
        name = source["file"]
        require(name == Path(name).name and name not in ("", ".", ".."), "Unsafe source filename")
        path = raw_dir / name
        if not path.exists() and download:
            with urllib.request.urlopen(source["download_url"], timeout=120) as response:
                data = response.read()
            require(len(data) == source["byte_count"], f"Download size changed: {name}")
            require(hashlib.sha256(data).hexdigest() == source["sha256"], f"Download hash changed: {name}")
            raw_dir.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(data)
        require(path.is_file(), f"Missing source file: {name}")
        require(path.stat().st_size == source["byte_count"], f"Source size changed: {name}")
        require(digest(path) == source["sha256"], f"Source hash changed: {name}")
        selected[source_id] = (source, path)
    return selected


def empd_exports(path, out):
    flags = collections.Counter()
    fractional_samples = set()
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        require(con.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "SQLite integrity check failed")
        query = """SELECT c.samplename FROM p_counts c
                   LEFT JOIN metadata m USING(samplename)
                   LEFT JOIN p_vars v USING(var_)
                   WHERE m.samplename IS NULL OR v.var_ IS NULL LIMIT 1"""
        require(con.execute(query).fetchone() is None, "Orphan EMPD measurement; import would lose rows")
        query = """SELECT m.samplename FROM metadata m LEFT JOIN climate c USING(samplename)
                   WHERE c.samplename IS NULL LIMIT 1"""
        require(con.execute(query).fetchone() is None, "Missing EMPD climate row")
        source_rows = con.execute("SELECT count(*) FROM p_counts").fetchone()[0]

        def measurements():
            for row in con.execute("SELECT samplename,var_,CAST(count AS TEXT) AS value FROM p_counts ORDER BY samplename,var_"):
                value = number(row["value"])
                require(value >= 0, "Negative EMPD count")
                fractional = value != value.to_integral_value()
                if fractional:
                    flags["fractional_count_cells"] += 1
                    fractional_samples.add(row["samplename"])
                yield dict(sample_id=row["samplename"], original_taxon_code=row["var_"],
                           count=fixed(value), is_fractional=int(fractional))

        n_measurements = write_csv(out / "empd_measurements.csv",
                                   ["sample_id", "original_taxon_code", "count", "is_fractional"], measurements())
        require(n_measurements == source_rows, "EMPD measurement count changed")

        def samples():
            query = """SELECT m.*, c.t_ann,c.p_ann FROM metadata m JOIN climate c USING(samplename)
                       ORDER BY samplename"""
            for row in con.execute(query):
                issues = []
                if row["samplename"] in fractional_samples:
                    issues.append("fractional_counts")
                reliability = row["locationreliability"]
                if reliability == "X":
                    issues.append("location_X")
                elif reliability not in ("A", "B", "C", "D"):
                    issues.append("location_quality_unknown")
                for field, lower, upper in (("latitude", -90, 90), ("longitude", -180, 180)):
                    if missing(row[field]) or not lower <= number(row[field]) <= upper:
                        issues.append("invalid_" + field)
                if missing(row["t_ann"]) or missing(row["p_ann"]):
                    issues.append("missing_climate")
                if row["ispercent"] != 0:
                    issues.append("percent_or_unknown_value_type")
                flags.update(issues)
                yield dict(sample_id=row["samplename"], original_sample_name=row["originalsamplename"],
                           site_name=row["sitename"], country=row["country"], latitude=optional_numeric(row["latitude"]),
                           longitude=optional_numeric(row["longitude"]), elevation_m=optional_numeric(row["elevation"]),
                           latitude_raw=row["latitude"], longitude_raw=row["longitude"], elevation_raw=row["elevation"],
                           location_reliability=reliability, sample_type=row["sampletype"],
                           sample_context=row["samplecontext"], age_bp=optional_numeric(row["agebp"]), age_bp_raw=row["agebp"],
                           age_uncertainty_class=row["ageuncertainty"], ispercent=row["ispercent"],
                           isdigitised=row["isdigitised"], empd_version=row["empd_version"],
                           tann_c=optional_numeric(row["t_ann"]), pann_mm_per_year=optional_numeric(row["p_ann"]),
                           tann_raw=row["t_ann"], pann_raw=row["p_ann"],
                           climate_source="WorldClim2", climate_reference_period="unverified",
                           quality_flags="|".join(issues), scientific_status="not_assessed")

        sample_fields = ["sample_id", "original_sample_name", "site_name", "country", "latitude", "longitude",
                         "elevation_m", "latitude_raw", "longitude_raw", "elevation_raw", "location_reliability", "sample_type", "sample_context", "age_bp", "age_bp_raw",
                         "age_uncertainty_class", "ispercent", "isdigitised", "empd_version", "tann_c",
                         "pann_mm_per_year", "tann_raw", "pann_raw", "climate_source", "climate_reference_period", "quality_flags",
                         "scientific_status"]
        n_samples = write_csv(out / "empd_samples.csv", sample_fields, samples())
        taxa = [dict(original_taxon_code=r["var_"], accepted_taxon_id=r["acc_var_"],
                     original_taxon_name=r["original_varname"], accepted_taxon_name=r["acc_varname"],
                     group_id=r["groupid"]) for r in con.execute("SELECT * FROM p_vars ORDER BY var_")]
        write_csv(out / "empd_taxa.csv", list(taxa[0]), taxa)
        groups = [dict(r) for r in con.execute("SELECT * FROM groupid ORDER BY groupid")]
        write_csv(out / "empd_groups.csv", list(groups[0]), groups)
        return dict(samples=n_samples, measurements=n_measurements, taxon_codes=len(taxa),
                    groups=len(groups), samples_with_fractional_counts=len(fractional_samples),
                    flags=dict(sorted(flags.items())), sparse_missing_values="unresolved; not filled with zero",
                    scientific_status="not_assessed")


def prefix_column(header, prefix):
    matches = [x for x in header if x.startswith(prefix)]
    require(len(matches) == 1, f"Expected one column beginning {prefix!r}")
    return matches[0]


def monticchio_exports(pollen_path, entity_path, age_path, out):
    _, header, pollen = read_pangaea(pollen_path)
    _, entity_header, entities = read_pangaea(entity_path)
    _, _, ages = read_pangaea(age_path)
    entity_id = prefix_column(entity_header, "ID (E#,")
    top_key = prefix_column(entity_header, "Depth top")
    bottom_key = prefix_column(entity_header, "Depth bot")
    entities = [r for r in entities if r["Event"] == "LGM90D" and r[entity_id] == "932"]
    require(len(entities) == 1, "Missing or ambiguous LGM90D entity 932")
    lower, upper = number(entities[0][top_key]), number(entities[0][bottom_key])
    require(lower <= upper, "Invalid retained depth interval")
    ages = [r for r in ages if r["Event"] == "LGM90D" and r["ID"] == "932"]
    require(bool(ages), "No retained LGM90D ages")
    depths = [number(r["Depth sed [m]"]) for r in pollen]
    require(len(set(depths)) == len(depths), "Duplicate pollen depths")
    used = {}
    age_depths = set()
    for age in ages:
        depth = number(age["Depth sed [m]"])
        require(depth not in age_depths, "Duplicate chronology depth")
        age_depths.add(depth)
        require(lower <= depth <= upper, "Cut age outside retained entity interval")
        matches = [i for i, d in enumerate(depths) if abs(d-depth) <= TOLERANCE_M]
        require(len(matches) == 1, f"Ambiguous or missing pollen-depth match at {depth}")
        index = matches[0]
        require(index not in used, "Pollen sample reused by chronology")
        require(lower <= depths[index] <= upper, "Pollen sample outside retained entity interval")
        value, lo, hi = (number(age[k]) for k in ("Age dated [ka BP]", "Age min [ka]", "Age max [ka]"))
        require(lo <= value <= hi, "Invalid marginal age bounds")
        used[index] = age
    samples, values, excluded, offsets = [], [], [], []
    for i, row in enumerate(pollen):
        sample_id = f"LGM90D:{i+1:06d}"
        if i not in used:
            excluded.append(dict(sample_id=sample_id, source_row=i+1, pollen_depth_m=row["Depth sed [m]"],
                                 reason="no_match_in_EpdAgeCut"))
            continue
        age = used[i]
        offset = depths[i] - number(age["Depth sed [m]"])
        offsets.append(offset)
        samples.append(dict(sample_id=sample_id, source_row=i+1, pollen_depth_m=row["Depth sed [m]"],
                            chronology_depth_m=age["Depth sed [m]"], depth_offset_m=fixed(offset),
                            join_kind="exact" if offset == 0 else "rounded_within_5mm",
                            age_bp=fixed(number(age["Age dated [ka BP]"])*1000),
                            age_min_bp=fixed(number(age["Age min [ka]"])*1000),
                            age_max_bp=fixed(number(age["Age max [ka]"])*1000),
                            uncertainty_class=age["Uncertainty class"], chronology_entity_id="932",
                            chronology_event="LGM90D", scientific_status="candidate_depth_link"))
        for column in header[1:]:
            text = row[column]
            value = number(text) if text else None
            if "[#]" in column and value is not None:
                require(value >= 0, f"Negative count: {sample_id}, {column}")
            unit = column.rsplit("[", 1)[1].rstrip("]") if "[" in column else ""
            values.append(dict(sample_id=sample_id, original_column=column, value=text, unit=unit,
                               is_missing=int(value is None),
                               is_fractional=int(value is not None and value != value.to_integral_value())))
    write_csv(out / "monticchio_samples.csv", list(samples[0]), samples)
    write_csv(out / "monticchio_values.csv", list(values[0]), values)
    write_csv(out / "monticchio_excluded.csv", ["sample_id", "source_row", "pollen_depth_m", "reason"], excluded)
    times = sorted(number(r["age_bp"]) for r in samples)
    gaps = [b-a for a, b in zip(times, times[1:])]
    return dict(source_samples=len(pollen), retained_samples=len(samples), excluded_samples=len(excluded),
                value_rows=len(values), exact_depth_matches=sum(x == 0 for x in offsets),
                rounded_depth_matches=sum(x != 0 for x in offsets),
                max_abs_depth_offset_m=fixed(max(abs(x) for x in offsets)),
                age_min_bp=fixed(min(times)), age_max_bp=fixed(max(times)),
                median_gap_years=fixed(statistics.median(gaps)) if gaps else None,
                max_gap_years=fixed(max(gaps)) if gaps else None,
                age_bounds_are_joint_ensemble=False, scientific_status="candidate_depth_link")


def run(raw_dir, output_dir, catalog_path, download=False, reference_path=None):
    require(not output_dir.exists(), "Output directory already exists; choose a new directory")
    sources = locked_sources(raw_dir, catalog_path, download)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".netzel-import-", dir=output_dir.parent) as temp:
        stage = Path(temp) / "result"
        stage.mkdir()
        report = dict(schema_version=1,
                      empd=empd_exports(sources["empd2"][1], stage),
                      monticchio=monticchio_exports(sources["monticchio_739452"][1],
                                                   sources["epd_entity_804591"][1],
                                                   sources["epd_ages_cut_804596"][1], stage))
        (stage / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        provenance = [{k: entry[k] for k in ("source_id", "file", "doi", "license", "sha256", "byte_count", "download_url")}
                      for entry, _ in sources.values()]
        outputs = []
        for path in sorted(stage.iterdir()):
            rows = None
            if path.suffix == ".csv":
                with path.open(encoding="utf-8", newline="") as stream:
                    rows = sum(1 for _ in csv.reader(stream)) - 1
            outputs.append(dict(file=path.name, byte_count=path.stat().st_size, sha256=digest(path), rows=rows))
        manifest = dict(schema_version=1, importer_sha256=digest(Path(__file__)),
                        sources=provenance, outputs=outputs, missing_csv_value="empty field",
                        scientific_status="not_assessed", age_origin="1950 CE; BP years",
                        note="Source data licenses apply. No taxon pooling or training eligibility is inferred.")
        if reference_path is not None:
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
            require(reference["report"] == report and reference["manifest"] == manifest,
                    "Reference mismatch; do not replace the reference without investigating")
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        stage.rename(output_dir)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=Path(__file__).resolve().parents[2] /
                        "docs/thesis_research/source_audit_2026_10_29/sources.json")
    parser.add_argument("--download", action="store_true", help="Download only missing locked source files")
    parser.add_argument("--reference", type=Path, help="Require exact agreement with a saved reference run")
    args = parser.parse_args()
    try:
        report = run(args.raw_dir, args.output_dir, args.catalog, args.download, args.reference)
    except (ImportFailure, OSError, sqlite3.Error, KeyError) as exc:
        parser.exit(1, f"Import failed: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
