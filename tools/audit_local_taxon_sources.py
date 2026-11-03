"""Read-only audit of the small historical taxon block; never runs R or training.

Run in the existing checkout:
  .venv/Scripts/python tools/audit_local_taxon_sources.py --config .local/paths.json
Outputs contain local source names and belong in the ignored .local directory.
Only the small, explicitly named RData container is decoded; ZIPs are listed only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import rdata
from scipy.io import netcdf_file


SPANS = {
    "src/climate_input.R": [(6, 21), (48, 63), (77, 104)],
    "src/all_transferfunctions_v1.1.R": [(39, 69), (143, 186), (196, 210), (252, 290)],
    "src/all_transferfunctions_v1.2.R": [(41, 70), (93, 123), (167, 203), (217, 247),
                                         (263, 276), (293, 338), (361, 399)],
    "src/all_trafo.R": [(45, 53), (75, 115), (119, 152), (178, 220), (239, 281),
                        (293, 336), (349, 401), (405, 465), (517, 529)],
    "main_v1.2.R": [(431, 436), (563, 583)],
    "src/mcmc_prepare.R": [(1460, 1498), (1524, 1579), (1581, 1587)],
    "src/mcmc_prep_v1.2.R": [(1322, 1368)],
    "src/mcmc_functions_v1.6.R": [(450, 477), (1242, 1284), (1781, 1820)],
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def emit(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def metadata(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


def run(config: Path, output: Path) -> dict:
    cfg = json.loads(config.read_text(encoding="utf-8-sig"))
    source = Path(cfg["original_data_root"]).resolve(strict=True)
    output = output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Output must be separate from the original source tree")
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing audit directory")
    before = metadata(source)
    inputs = ["in/taxa_old_new_mediterr.rdata", "in/tann_cru4.nc", "in/pann_cru4.nc"]
    # No automatic expansion to the large core-data containers.
    for name in inputs:
        p = (source / name).resolve(strict=True)
        if source not in p.parents or p.stat().st_size > 10_000_000:
            raise ValueError("Unexpected source path or size; inspect before decoding")
    fingerprints = {name: sha(source / name) for name in inputs + list(SPANS)}
    raw = rdata.read_rda(source / inputs[0])["taxa_old_new_mediterr"]
    a = np.asarray(raw["taxadist"])
    names = raw["taxanames"]
    if a.ndim != 3 or names.shape != (a.shape[2], 2):
        raise ValueError("Taxon array and name table no longer have compatible dimensions")
    missing = np.isnan(a)
    grids = {}
    for filename, var in [(inputs[1], "tmp"), (inputs[2], "pre")]:
        with netcdf_file(source / filename, mmap=False) as nc:
            v = nc.variables[var]
            dims = list(v.dimensions)
            data = np.array(v[:], copy=True)
            if dims != ["time", "lat", "lon"] or data.shape[0] != 1:
                raise ValueError("Unexpected climate dimensions; no implicit reshape")
            # Named NetCDF axes -> R branch's longitude-first array axes.
            lonlat = data[0].T
            mask = ~np.isfinite(lonlat) | (lonlat == v._FillValue)
            lon = np.array(nc.variables["lon"][:], copy=True)
            lat = np.array(nc.variables["lat"][:], copy=True)
            if lonlat.shape != a.shape[:2]:
                raise ValueError("Climate and taxon grid dimensions differ")
            grids[var] = dict(data=lonlat, mask=mask, lon=lon, lat=lat,
                              units=v.units.decode(), filename=filename)
    checks = {
        "binary_finite_taxon_values": bool(np.isin(a[~missing], [0, 1]).all()),
        "same_missing_mask_for_every_taxon": bool((missing == missing[:, :, :1]).all()),
        "climate_coordinate_vectors_identical": all(
            np.array_equal(grids["tmp"][c], grids["pre"][c]) for c in ["lon", "lat"]),
        "taxon_mask_matches_both_climate_masks": all(
            np.array_equal(missing[:, :, 0], g["mask"]) for g in grids.values()),
        "all_climate_cells_used_by_taxon_block_finite": all(
            np.isfinite(g["data"][~missing[:, :, 0]]).all().item() for g in grids.values()),
    }
    orientations = {k: {"identity_lon_lat": int(np.count_nonzero(missing[:, :, 0] != g["mask"])),
                        "reverse_lon": int(np.count_nonzero(missing[:, :, 0] != g["mask"][::-1, :])),
                        "reverse_lat": int(np.count_nonzero(missing[:, :, 0] != g["mask"][:, ::-1])),
                        "reverse_both": int(np.count_nonzero(missing[:, :, 0] != g["mask"][::-1, ::-1]))}
                    for k, g in grids.items()}
    rows = []
    planes = defaultdict(list)
    for i in range(a.shape[2]):
        plane = a[:, :, i]
        encoded = np.nan_to_num(plane, nan=-1).astype(np.int8).tobytes(order="F")
        digest = hashlib.sha256(encoded).hexdigest()
        planes[digest].append(i + 1)
        rows.append(dict(array_index_R_1_based=i + 1, R_row_label=str(names.index[i]),
                         name_column_1=str(names.iloc[i, 0]), name_column_2=str(names.iloc[i, 1]),
                         presence_cells=int((plane == 1).sum()),
                         absence_cells=int((plane == 0).sum()), missing_cells=int(np.isnan(plane).sum()),
                         binary_plane_sha256=digest))
    # Preserve positional indices, R row labels, spelling and repeated names.
    name_audit = {}
    for j, col in enumerate(names.columns):
        groups = defaultdict(list)
        for i, value in enumerate(names.iloc[:, j]):
            groups[str(value)].append(i + 1)
        name_audit[str(col)] = dict(unique_names=int(names.iloc[:, j].nunique()),
                                   missing_names=int(names.iloc[:, j].isna().sum()),
                                   duplicate_groups={k: v for k, v in groups.items() if len(v) > 1})
    spans, dependencies = [], []
    pattern = re.compile(r'\b(load|nc_open|source)\s*\(\s*([\"\'])(.*?)\2')
    for name, ranges in SPANS.items():
        lines = (source / name).read_text(encoding="utf-8-sig").splitlines()
        first_stop = next((i for i, line in enumerate(lines, 1)
                           if re.match(r"^\s*stop\(\s*\)\s*$", line)), None)
        for start, end in ranges:
            spans.append(dict(source=name, first_line=start, last_line=end,
                              text="\n".join(lines[start - 1:end])))
        for line_no, line in enumerate(lines, 1):
            if line.lstrip().startswith("#"):
                continue
            for match in pattern.finditer(line):
                literal = match.group(3)
                if literal.startswith(("in/", "out/", "src/")):
                    target = (source / literal).resolve()
                    if source not in target.parents:
                        raise ValueError("Source reference escapes the selected tree")
                    status = "present" if target.is_file() else "missing_in_selected_source"
                else:
                    status = "outside_or_dynamic_not_inspected"
                dependencies.append(dict(source=name, line=line_no, function=match.group(1),
                                         literal=literal, status=status,
                                         before_first_literal_stop=first_stop is None or line_no < first_stop))
    # Search archive directories for the missing serialized model files, not their payloads.
    wanted = {Path(d["literal"]).name for d in dependencies
              if d["function"] == "load" and d["status"] == "missing_in_selected_source"}
    archives = []
    for path in sorted(source.rglob("*.zip")):
        with zipfile.ZipFile(path) as z:
            members = z.namelist()
            matches = [m for m in members if Path(m.replace("\\", "/")).name in wanted]
            archives.append(dict(archive=path.relative_to(source).as_posix(), members=len(members),
                                 matching_model_members=matches))
    after = metadata(source)
    checks["original_metadata_unchanged_during_audit"] = before == after
    checks["selected_source_hashes_unchanged_during_audit"] = all(
        sha(source / name) == value for name, value in fingerprints.items())
    catalog = Path(cfg["inventory_dir"]) / "data_catalog.csv"
    with catalog.open(encoding="utf-8-sig", newline="") as f:
        previous = {r["relative_path"]: (int(r["size_bytes"]), int(r["mtime_ns"]))
                    for r in csv.DictReader(f)}
    checks["original_metadata_matches_existing_inventory"] = before == previous
    summary = dict(
        schema_version=1, tool_sha256=sha(Path(__file__)),
        packages={k: importlib.metadata.version(k) for k in ["numpy", "scipy", "rdata", "pandas"]},
        scientific_status="local_grid_branch_supported_historical_transfer_function_link_unresolved",
        no_R_execution=True, no_training=True, no_sampler_execution=True,
        source_fingerprints=fingerprints, array_shape=list(a.shape),
        valid_cells_per_taxon=int((~missing[:, :, 0]).sum()),
        missing_cells_per_taxon=int(missing[:, :, 0].sum()),
        coordinate_mapping=dict(array_axes=["longitude", "latitude", "taxon_position"],
                                longitude=grids["tmp"]["lon"].tolist(),
                                latitude=grids["tmp"]["lat"].tolist(),
                                evidence="explicit R consumer and matching missing-cell masks; producer absent",
                                historical_global_release_equivalence=False),
        orientation_mask_mismatches=orientations, name_audit=name_audit,
        differing_name_pair_positions=[r["array_index_R_1_based"] for r in rows
                                       if r["name_column_1"] != r["name_column_2"]],
        zero_presence_positions=[r["array_index_R_1_based"] for r in rows if r["presence_cells"] == 0],
        distinct_binary_planes=len(planes),
        identical_binary_plane_groups=[v for v in planes.values() if len(v) > 1],
        dependencies=dependencies, dependency_scope="literal_first_argument_selected_scripts_not_execution_graph",
        archive_directory_search=archives, archive_payloads_decoded=False,
        original_file_count=len(before), original_total_bytes=sum(v[0] for v in before.values()),
        checks=checks, all_checks_passed=all(checks.values()),
    )
    output.mkdir(parents=True)
    emit(output / "audit.json", summary)
    emit(output / "source_spans.json", spans)
    with (output / "taxon_inventory.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path(".local/taxon_source_audit"))
    args = p.parse_args()
    summary = run(args.config, args.output)
    print(json.dumps({k: summary[k] for k in ["scientific_status", "array_shape", "valid_cells_per_taxon",
                                             "distinct_binary_planes", "all_checks_passed"]}))
    if not summary["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
