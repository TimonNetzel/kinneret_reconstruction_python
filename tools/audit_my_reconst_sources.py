"""Read-only inventory and bounded crosswalk for the user-selected my_reconst tree.

No R execution, model fitting, sampler calls or raw-array exports. Private source
paths/names stay under .local. Large global arrays are scanned one plane at a time.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import zipfile

import numpy as np
import rdata
from scipy.io import netcdf_file

from local_rdata_stream import attributes, children, read_planes, read_structure, small_value


WANTED = ["taxa_distribution_ww.rdata", "tann_test.nc", "pann_test.nc",
          "mixed_partial_arts_emd.rdata", "mixed_partial_arts_emd_info.rdata",
          "all_nnet_transfer_functions.rdata"]
MODELS = ["my_ohrid/data/out/transfer_function/best_fit_2D_v1.6.rdata",
          "my_ohrid/data/out/transfer_function/qda_2D_unsmoted_nh_unknown_relict_v2.6.rdata"]
CODE = {"my_ohrid/src/climate_input_ww.R": [(98, 176)],
        "my_kinneret/src/climate_input_ww.R": [(149, 211)],
        "my_ohrid/src/ml_competition_ww.R": [(549, 556)]}


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def emit(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def metadata(root):
    return {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


def exact_indices(axis, targets):
    matches = [np.flatnonzero(axis == x) for x in targets]
    if any(len(x) != 1 for x in matches):
        raise ValueError("Coordinates need exact unique matches; no nearest-neighbour substitution")
    return np.array([int(x[0]) for x in matches])


def run(config: Path, output: Path):
    cfg = json.loads(config.read_text(encoding="utf-8-sig"))
    source = Path(cfg["additional_source_root"]).resolve(strict=True)
    original = Path(cfg["original_data_root"]).resolve(strict=True)
    output = output.resolve()
    for root in [source, original]:
        if output == root or root in output.parents or output in root.parents:
            raise ValueError("Output must be separate from both original source trees")
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing audit")
    before = metadata(source)
    original_before = metadata(original)
    local_path = original / "in/taxa_old_new_mediterr.rdata"
    local = rdata.read_rda(local_path)["taxa_old_new_mediterr"]
    local_array = np.asarray(local["taxadist"])
    local_names = local["taxanames"].iloc[:, 0].tolist()
    with netcdf_file(original / "in/tann_cru4.nc", mmap=False) as nc:
        target_lon = nc.variables["lon"][:].copy()
        target_lat = nc.variables["lat"][:].copy()
    fingerprints = {"original/in/taxa_old_new_mediterr.rdata": sha(local_path)}
    global_results, plane_rows, name_rows = {}, [], []
    for folder in ["my_kinneret", "my_ohrid"]:
        relative = f"{folder}/data/in/climate_data/taxa_distribution_ww.rdata"
        path = source / relative
        fingerprints[relative] = sha(path)
        data, parser = read_structure(path)
        fields = children(children(data.object)["taxa.distribution.ww"])
        dims = tuple(map(int, attributes(fields["dist"])["dim"]))
        block = parser.blocks[id(fields["dist"].value)]
        lon = np.asarray(small_value(fields["lon"], parser)).ravel()
        lat = np.asarray(small_value(fields["lat"], parser)).ravel()
        names = np.asarray(small_value(fields["names"], parser)).tolist()
        if dims != (len(lon), len(lat), len(names)):
            raise ValueError("Dimensions and stored axes/names disagree")
        ix, iy = exact_indices(lon, target_lon), exact_indices(lat, target_lat)
        extra = list((Counter(names) - Counter(local_names)).elements())
        extra_set = set(extra)
        keep_positions = [i for i, name in enumerate(names) if name not in extra_set]
        ordered_names_match = [names[i] for i in keep_positions] == local_names
        if not ordered_names_match:
            raise ValueError("No exact order-preserving taxon mapping; do not guess")
        position_map = {global_i: local_i for local_i, global_i in enumerate(keep_positions)}
        equal_planes = 0
        all_mismatches = value_mismatches = mask_mismatches = 0
        all_binary = True
        for i, plane in read_planes(path, block, dims):
            finite = plane[np.isfinite(plane)]
            binary = bool(np.isin(finite, [0, 1]).all())
            all_binary &= binary
            row = dict(source=folder, global_position_R_1_based=i + 1, name=names[i],
                       canonical_binary_plane_sha256=hashlib.sha256(
                           np.nan_to_num(plane, nan=-1).astype(np.int8).tobytes(order="F")).hexdigest(),
                       global_presence=int(np.count_nonzero(plane == 1)),
                       global_absence=int(np.count_nonzero(plane == 0)),
                       global_missing=int(np.count_nonzero(np.isnan(plane))),
                       local_position_R_1_based="", cropped_mismatches="",
                       cropped_mask_mismatches="", cropped_finite_value_mismatches="")
            if i in position_map:
                j = position_map[i]
                crop = plane[np.ix_(ix, iy)]
                reference = local_array[:, :, j]
                same = (crop == reference) | (np.isnan(crop) & np.isnan(reference))
                both = np.isfinite(crop) & np.isfinite(reference)
                bad = int(np.count_nonzero(~same))
                bad_mask = int(np.count_nonzero(np.isnan(crop) != np.isnan(reference)))
                bad_values = int(np.count_nonzero(crop[both] != reference[both]))
                equal_planes += bad == 0
                all_mismatches += bad
                mask_mismatches += bad_mask
                value_mismatches += bad_values
                row.update(local_position_R_1_based=j + 1, cropped_mismatches=bad,
                           cropped_mask_mismatches=bad_mask, cropped_finite_value_mismatches=bad_values)
            plane_rows.append(row)
        removed = [8, 10, 152, 153]
        historical_indices = [i for i in range(len(names)) if i + 1 not in removed]
        historical_names = [names[i] for i in historical_indices]
        for i, name in enumerate(names):
            name_rows.append(dict(source=folder, global_position_R_1_based=i + 1, name=name,
                                  index_operation_status="referenced_by_main_v1.2" if folder == "my_ohrid"
                                  else "counterfactual_comparison_do_not_apply",
                                  local_position_R_1_based=position_map[i] + 1 if i in position_map else "",
                                  retained_after_documented_deletion=(i + 1 not in removed),
                                  position_after_index_operation_R_1_based=historical_indices.index(i) + 1
                                  if i in historical_indices else ""))
        global_results[folder] = dict(
            relative_path=relative, dimensions=list(dims), payload_bytes=block["bytes"],
            payload_xdr_sha256=block["xdr_sha256"], R_writer_integer=data.versions.serialized,
            coordinate_ranges=dict(lon=[float(lon[0]), float(lon[-1])],
                                   lat=[float(lat[0]), float(lat[-1])]),
            coordinates_strictly_increasing=bool((np.diff(lon) > 0).all() and (np.diff(lat) > 0).all()),
            local_coordinates_exactly_found=True, exact_order_preserving_local_names=ordered_names_match,
            additional_names=extra, all_finite_distribution_values_binary=all_binary,
            local_crop=dict(compared_planes=len(position_map), exactly_equal_planes=equal_planes,
                            total_cell_mismatches=all_mismatches, mask_mismatches=mask_mismatches,
                            finite_value_mismatches=value_mismatches),
            documented_deleted_positions=[dict(position=i, name=names[i-1]) for i in removed],
            index_operation_status="referenced_by_main_v1.2" if folder == "my_ohrid"
                                   else "counterfactual_comparison_do_not_apply",
            resulting_name_count=len(historical_names),
            resulting_position_110_before_main_rename=historical_names[109],
            contains_finished_transfer_functions=False,
        )
    common_hashes = {folder: {r["local_position_R_1_based"]: r["canonical_binary_plane_sha256"]
                             for r in plane_rows if r["source"] == folder and r["local_position_R_1_based"] != ""}
                     for folder in ["my_kinneret", "my_ohrid"]}
    local_copy_checks = {}
    for folder in ["my_kinneret", "my_ohrid"]:
        relative = f"{folder}/data/in/climate_data/taxa_old_new_mediterr.rdata"
        fingerprints[relative] = sha(source / relative)
        local_copy_checks[folder] = fingerprints[relative] == fingerprints["original/in/taxa_old_new_mediterr.rdata"]
    models = []
    for relative in MODELS:
        path = source / relative
        fingerprints[relative] = sha(path)
        data, parser = read_structure(path)
        fields = children(data.object)
        fit = fields["best.fit"]
        fit_fields = children(fit)
        models.append(dict(relative_path=relative, workspace_objects=list(fields),
                           fit_class=np.asarray(attributes(fit)["class"]).tolist(),
                           method=np.asarray(small_value(fit_fields["method"], parser)).tolist(),
                           class_levels=np.asarray(small_value(fit_fields["levels"], parser)).tolist(),
                           skipped_numeric_vectors=len(parser.blocks),
                           interpretation="one_saved_multiclass_caret_fit_not_a_collection_of_279_binary_functions"))
    code_spans = []
    for relative, spans in CODE.items():
        path = source / relative
        fingerprints[relative] = sha(path)
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        for first, last in spans:
            code_spans.append(dict(source=relative, first_line=first, last_line=last,
                                   text="\n".join(lines[first-1:last])))
    found = {name: [p for p in before if Path(p).name.casefold() == name.casefold()] for name in WANTED}
    archives = []
    for relative in before:
        if relative.lower().endswith(".zip"):
            with zipfile.ZipFile(source / relative) as z:
                members = z.namelist()
                matches = [m for m in members if Path(m.replace("\\", "/")).name.casefold()
                           in {name.casefold() for name in WANTED}]
                archives.append(dict(archive=relative, members=len(members), matches=matches))
    checks = dict(source_metadata_unchanged=before == metadata(source),
                  original_metadata_unchanged=original_before == metadata(original),
                  selected_hashes_unchanged=all(
                      sha((original / key.removeprefix("original/")) if key.startswith("original/")
                          else source / key) == digest for key, digest in fingerprints.items()),
                  both_global_axes_valid=all(x["coordinates_strictly_increasing"] for x in global_results.values()),
                  both_ordered_name_crosswalks_exact=all(x["exact_order_preserving_local_names"] for x in global_results.values()),
                  global_distribution_values_binary=all(x["all_finite_distribution_values_binary"] for x in global_results.values()),
                  ohrid_documented_deletion_yields_279=global_results["my_ohrid"]["resulting_name_count"] == 279,
                  common_275_global_planes_identical=common_hashes["my_kinneret"] == common_hashes["my_ohrid"]
                                                      and len(common_hashes["my_ohrid"]) == 275,
                  same_named_local_blocks_byte_identical=all(local_copy_checks.values()),
                  models_are_single_multiclass_fits=all(len(x["class_levels"]) == 7 for x in models))
    summary = dict(schema_version=1, scientific_status="global_distribution_recovered_tf_release_not_recovered",
                   no_R_execution=True, no_training=True, no_sampler=True,
                   file_count=len(before), total_bytes=sum(v[0] for v in before.values()),
                   format_counts=dict(Counter(Path(p).suffix.lower() for p in before)),
                   exact_filename_search=found, archive_directory_search=archives,
                   same_named_local_block_hash_matches=local_copy_checks,
                   fingerprints=fingerprints, global_distributions=global_results, models=models,
                   checks=checks, all_checks_passed=all(checks.values()),
                   reader_limitations=["XDR RData v2/v3 only", "no R runtime/prediction validation",
                                       "one larger caret-container probe failed at unsupported RObjectType 252; not decoded"],
                   reader_hashes={name: sha(Path(__file__).parent / name)
                                  for name in ["local_rdata_stream.py", Path(__file__).name]},
                   parser_version=importlib.metadata.version("rdata"))
    output.mkdir(parents=True)
    emit(output / "audit.json", summary)
    emit(output / "source_spans.json", code_spans)
    for name, rows in [("global_taxon_crosswalk.csv", name_rows), ("global_plane_summary.csv", plane_rows)]:
        with (output / name).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (output / "source_inventory.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["relative_path", "size_bytes", "mtime_ns"])
        writer.writerows((p, *values) for p, values in sorted(before.items()))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path(".local/my_reconst_source_audit"))
    args = parser.parse_args()
    summary = run(args.config, args.output)
    print(json.dumps({k: summary[k] for k in ["file_count", "scientific_status", "all_checks_passed"]}))
    print(json.dumps({k: v["local_crop"] for k, v in summary["global_distributions"].items()}))
    if not summary["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
