"""Read a locked public RDS snapshot, without evaluating R or workshop code.

The comparisons are exploratory provenance diagnostics, not a MON16 crosswalk.
The optional rdata dependency is imported only when reading the source archive.
"""
from __future__ import annotations

import argparse
import collections
import csv
from decimal import Decimal
import hashlib
import io
import json
import math
from pathlib import Path
import warnings

RDS_SHA256 = "03536b6f1d3974de77ff997f6d86674a78882b2ebe305bb4e135e38289a2a829"
RDS_GIT_BLOB = "eaa1e04645575ecd13ba5b2432e877ae57848db1"
PANGAEA_SHA256 = "46ab6cea3a636d1a25b8c15f4b9b79a94c8fd1bd41f2943ae5c8941f9050a4de"
KEY_FIELDS = ("taxonid", "units", "element", "context", "symmetry", "elementtype")
# These five unmodified genus fields are a separate legacy-source check.
SOURCE_TAXA = {"Abi [#]": 1, "Aln [#]": 330, "Art [#]": 25,
               "Bet [#]": 29, "Fag [#]": 107}


def atom(value):
    if value is None or type(value).__name__ in ("NAType", "MaskedConstant"):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def one(value):
    if hasattr(value, "shape") and value.shape:
        if len(value) != 1:
            raise ValueError("Expected one scalar, not a vector")
        value = value[0]
    return atom(value)


def deref(obj):
    while obj is not None and obj.info.type.name == "REF":
        obj = obj.referenced_object
    return obj


def raw_attrs(obj):
    result = {}
    node = deref(deref(obj).attributes)
    while node is not None and node.info.type.name == "LIST":
        tag = deref(node.tag)
        label = deref(tag.value).value.decode("utf-8")
        if label in result:
            raise ValueError("Duplicate S4 slot or attribute")
        result[label] = node.value[0]
        node = deref(node.value[1])
    return result


def raw_scalar(obj):
    obj = deref(obj)
    if obj.info.type.name == "STR":
        value = deref(obj.value[0]).value
        return None if value is None else value.decode("utf-8")
    return atom(obj.value[0])


def collection_values(value):
    # Duplicate R list names cannot safely be converted to dictionary keys.
    if isinstance(value, dict):
        return list(value.values())
    return list(value)


def read_site(path):
    import rdata

    raw_bytes = path.read_bytes()
    if hashlib.sha256(raw_bytes).hexdigest() != RDS_SHA256:
        raise ValueError("Wrong RDS input hash")
    blob = hashlib.sha1(b"blob " + str(len(raw_bytes)).encode() + b"\0" + raw_bytes).hexdigest()
    if blob != RDS_GIT_BLOB:
        raise ValueError("Wrong RDS Git blob")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = rdata.parser.parse_file(path)
    # The root has repeated names 'site'. Select in the raw vector BEFORE any
    # named-list-to-dict conversion, which would silently retain only one site.
    sites = deref(raw_attrs(parsed.object)["sites"])
    selected = [s for s in sites.value
                if raw_scalar(raw_attrs(s)["siteid"]) == 3295]
    if len(selected) != 1:
        raise ValueError("Expected exactly one raw Monticchio site")
    # Check every named vector in the selected subtree before using the class
    # converter. Geometry, data frames and scalar arrays are read as data only.
    seen = set()

    def visit(node):
        node = deref(node)
        if node is None or id(node) in seen:
            return
        seen.add(id(node))
        if node.info.type.name == "VEC":
            attrs = raw_attrs(node)
            if "names" in attrs:
                names = [None if deref(x).value is None else deref(x).value
                         for x in deref(attrs["names"]).value]
                if len(names) != len(set(names)):
                    raise ValueError("Selected subtree contains duplicate vector names")
        for member in (node.attributes, node.tag):
            if member is not None:
                visit(member)
        if isinstance(node.value, (list, tuple)):
            for member in node.value:
                if hasattr(member, "info"):
                    visit(member)
        elif hasattr(node.value, "info"):
            visit(node.value)

    visit(selected[0])
    with warnings.catch_warnings(record=True) as captured:
        site = rdata.conversion.convert(selected[0])
    root_names = [deref(v).value.decode("utf-8") for v in
                  deref(raw_attrs(sites)["names"]).value]
    return site, {"raw_site_count": len(sites.value),
                  "raw_root_name_count": len(root_names),
                  "raw_root_unique_names": sorted(set(root_names)),
                  "selected_subtree_duplicate_names": False,
                  "rdata_version": rdata.__version__,
                  "conversion_warning_classes": sorted(set(str(w.message) for w in captured))}


def measurement_map(records):
    result = collections.defaultdict(set)
    for record in records:
        key = tuple(None if atom(record[k]) is None else str(atom(record[k]))
                    for k in KEY_FIELDS)
        value = atom(record["value"])
        if value is None:
            raise ValueError("Missing numerical value is not a numerical zero")
        number = Decimal(str(value))
        if not number.is_finite():
            raise ValueError("Non-finite measurement")
        result[key].add(number)
    return dict(result)


def comparable_keys(old_maps, new_maps):
    old_keys = set().union(*(set(m) for m in old_maps))
    new_keys = set().union(*(set(m) for m in new_maps))
    ambiguous = {k for m in old_maps + new_maps for k, values in m.items()
                 if len(values) > 1}
    return sorted((old_keys & new_keys) - ambiguous, key=str), ambiguous, old_keys, new_keys


def fingerprint(measurements, keys):
    values = []
    for key in keys:
        present = measurements.get(key, {Decimal(0)})
        if len(present) != 1:
            raise ValueError("Ambiguous variable must not be summed or overwritten")
        values.append(next(iter(present)))
    return tuple(values)


def numeric_for_taxon(sample, taxonid):
    values = [Decimal(str(atom(r["value"]))) for r in sample.datum.to_dict("records")
              if int(r["taxonid"]) == taxonid and r["units"] == "NISP"
              and r["element"] == "pollen"]
    if not values:
        return Decimal(0)
    if len(set(values)) != 1:
        raise ValueError("Ambiguous source-check variable")
    return values[0]


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def run(rds, pangaea, prior, output):
    if output.exists():
        raise ValueError("Use a new output directory")
    if hashlib.sha256(pangaea.read_bytes()).hexdigest() != PANGAEA_SHA256:
        raise ValueError("Wrong PANGAEA input hash")
    expected_prior = "4ab39ab51bb54b76257f2ffd0460044fe15d17e962dbb49fce94d38c2ba35f03"
    if hashlib.sha256(prior.read_bytes()).hexdigest() != expected_prior:
        raise ValueError("Wrong PR25 sample table hash")
    site, reader_info = read_site(rds)
    datasets = {}
    inventory = []
    for cu in collection_values(site.collunits.collunits):
        for ds in collection_values(cu.datasets.datasets):
            samples = collection_values(ds.samples.samples)
            did = int(one(ds.datasetid))
            if did in datasets:
                raise ValueError("Duplicate dataset ID")
            datasets[did] = (cu, ds, samples)
            inventory.append({"datasetid": did, "collectionunitid": int(one(cu.collectionunitid)),
                              "handle": one(cu.handle), "samples": len(samples),
                              "analysisunit_names_present": sum(one(s.analysisunitname) is not None for s in samples),
                              "sample_names_present": sum(one(s.samplename) is not None for s in samples)})
    old_cu, old_ds, old = datasets[26608]
    new_cu, new_ds, new = datasets[47525]
    if len(old) != 245 or len(new) != 924 or one(old_cu.handle) != "LGM90D" or one(new_cu.handle) != "MONTICCHIO":
        raise ValueError("Unexpected locked dataset inventory")
    old_by_depth = {Decimal(str(one(s.depth))): s for s in old}
    if len(old_by_depth) != len(old):
        raise ValueError("Duplicate legacy depths")
    pg = list(csv.DictReader(io.StringIO(pangaea.read_text().split("*/", 1)[1].lstrip()), delimiter="\t"))
    if len(pg) != len(old):
        raise ValueError("Unexpected PANGAEA row count")
    # Preserve both source orders. Four native RDS positions differ from the
    # published table; never treat list position as a sample identity.
    pg_lookup = collections.defaultdict(list)
    for j, record in enumerate(pg):
        key = (record["Depth sed [m]"],) + tuple(Decimal(record[c]) for c in SOURCE_TAXA)
        pg_lookup[key].append(j)
    alignment = []
    aligned_pg = []
    for i, s in enumerate(old):
        key = (format(float(one(s.depth)) / 100, ".2f"),) + tuple(
            numeric_for_taxon(s, tid) for tid in SOURCE_TAXA.values())
        hits = pg_lookup[key]
        if len(hits) != 1:
            raise ValueError("Legacy source alignment is missing or ambiguous")
        j = hits[0]
        aligned_pg.append((j + 1, pg[j]))
        alignment.append({"rds_list_position": i + 1, "epd_row_number": j + 1,
                          "neotoma_sampleid": int(one(s.sampleid)),
                          "native_depth_cm": str(one(s.depth)),
                          "pangaea_export_depth_m": pg[j]["Depth sed [m]"],
                          "five_genus_check": "equal", "position_changed": i != j})
    if len({r["epd_row_number"] for r in alignment}) != 245:
        raise ValueError("Legacy source alignment is not one-to-one")
    quercus = []
    for s, (number, record) in zip(old, aligned_pg):
        source = Decimal(record["Que [#]"])
        native = numeric_for_taxon(s, 2943)
        if source != native:
            quercus.append({"epd_row_number": number, "neotoma_sampleid": int(one(s.sampleid)),
                            "depth_cm": str(one(s.depth)), "pangaea_Quercus": str(source),
                            "neotoma_Quercus_undiff": str(native),
                            "neotoma_Quercus_ilex": str(numeric_for_taxon(s, 4717))})
    old_maps = [measurement_map(s.datum.to_dict("records")) for s in old]
    new_maps = [measurement_map(s.datum.to_dict("records")) for s in new]
    shared, ambiguous, old_keys, new_keys = comparable_keys(old_maps, new_maps)
    indexes = {int(one(s.sampleid)): i for i, s in enumerate(old)}
    lookup = collections.defaultdict(list)
    new_vectors = []
    for i, m in enumerate(new_maps):
        vector = fingerprint(m, shared)
        new_vectors.append(vector)
        lookup[vector].append(i)
    pilot = list(csv.DictReader(prior.open()))
    if len(pilot) != 69:
        raise ValueError("Unexpected pilot count")
    correspondence = []
    unmatched = []
    differences = []
    for p in pilot:
        s = old_by_depth[Decimal(p["archive_midpoint_cm"])]
        index = indexes[int(one(s.sampleid))]
        # The published source row is fixed by PR25, not selected by age.
        expected = pg[int(p["epd_sample_number"]) - 1]
        for column, tid in SOURCE_TAXA.items():
            if numeric_for_taxon(s, tid) != Decimal(expected[column]):
                raise ValueError("Pilot source check differs")
        old_vector = fingerprint(old_maps[index], shared)
        hits = lookup[old_vector]
        if not hits:
            distances = [sum(a != b for a, b in zip(old_vector, v)) for v in new_vectors]
            minimum = min(distances)
            nearest = [j for j, distance in enumerate(distances) if distance == minimum]
            unmatched.append({"pilot_sample_id": p["pilot_sample_id"],
                              "legacy_sampleid": int(one(s.sampleid)),
                              "minimum_differing_fields": minimum, "nearest_count": len(nearest),
                              "nearest_sampleids": json.dumps([int(one(new[j].sampleid)) for j in nearest]),
                              "status": "diagnostic_only_not_correspondence"})
            for j in nearest:
                for key, a, b in zip(shared, old_vector, new_vectors[j]):
                    if a != b:
                        differences.append({"pilot_sample_id": p["pilot_sample_id"],
                                            "diagnostic_sampleid": int(one(new[j].sampleid)),
                                            **dict(zip(KEY_FIELDS, key)),
                                            "legacy_value": str(a), "diagnostic_value": str(b),
                                            "status": "unresolved_source_difference"})
        target = new[hits[0]] if len(hits) == 1 else None
        correspondence.append({
            "pilot_sample_id": p["pilot_sample_id"], "epd_sample_number": p["epd_sample_number"],
            "archive_midpoint_cm": p["archive_midpoint_cm"], "legacy_datasetid": 26608,
            "legacy_sampleid": int(one(s.sampleid)), "legacy_analysisunitid": int(one(s.analysisunitid)),
            "shared_numeric_keys": len(shared), "candidate_count": len(hits),
            "candidate_sampleids": json.dumps([int(one(new[i].sampleid)) for i in hits]),
            "candidate_datasetid": 47525 if target else "",
            "candidate_sampleid": int(one(target.sampleid)) if target else "",
            "candidate_analysisunitid": int(one(target.analysisunitid)) if target else "",
            "candidate_composite_depth_cm": str(one(target.depth)) if target else "",
            "candidate_analysisunitname": one(target.analysisunitname) if target else "",
            "candidate_minus_legacy_cm": str(Decimal(str(one(target.depth))) - Decimal(p["archive_midpoint_cm"])) if target else "",
            "status": "unique_shared_variable_candidate" if target else "no_shared_variable_match" if not hits else "ambiguous_shared_variable_candidates",
            "mon16_depth_cm": "", "mon16_age_cal_bp": "", "new_age_transfer_allowed": "false"})
    availability = []
    for did in (26608, 47525):
        cu, ds, samples = datasets[did]
        for field in ("samplename", "analysisunitname", "igsn", "thickness", "notes", "labnumber"):
            exists = sum(hasattr(s, field) for s in samples)
            present = sum(hasattr(s, field) and one(getattr(s, field)) is not None for s in samples)
            availability.append({"datasetid": did, "field": field, "samples": len(samples),
                                 "slot_present": exists, "nonmissing": present,
                                 "interpretation": "not_exported_in_snapshot" if not exists else "exported_value_status_only"})
    output.mkdir(parents=True)
    write_csv(output / "sample_correspondence.csv", correspondence, list(correspondence[0]))
    write_csv(output / "unmatched_diagnostics.csv", unmatched, list(unmatched[0]))
    write_csv(output / "unmatched_variable_differences.csv", differences, list(differences[0]))
    write_csv(output / "legacy_source_alignment.csv", alignment, list(alignment[0]))
    write_csv(output / "dataset_inventory.csv", sorted(inventory, key=lambda x: x["datasetid"]), list(inventory[0]))
    write_csv(output / "field_availability.csv", availability, list(availability[0]))
    write_csv(output / "legacy_quercus_differences.csv", sorted(quercus, key=lambda r: r["epd_row_number"]), list(quercus[0]))
    variables = []
    for key in sorted(old_keys | new_keys, key=str):
        variables.append(dict(zip(KEY_FIELDS, key)) | {"in_26608": key in old_keys, "in_47525": key in new_keys,
                         "ambiguous_native_values": key in ambiguous, "used_for_comparison": key in shared})
    write_csv(output / "native_variable_policy.csv", variables, list(variables[0]))
    counts = collections.Counter(r["status"] for r in correspondence)
    write_json(output / "summary.json", {
        "rds_sha256": RDS_SHA256, "rds_git_blob": RDS_GIT_BLOB, "reader": reader_info,
        "siteid": 3295, "collectionunits": len(inventory), "legacy_datasetid": 26608,
        "legacy_samples": len(old), "composite_datasetid": 47525, "composite_samples": len(new),
        "composite_collectionunit_note": one(new_cu.notes),
        "legacy_five_genus_cells_checked": len(old) * len(SOURCE_TAXA), "legacy_five_genus_differences": 0,
        "legacy_source_positions_changed": sum(r["position_changed"] for r in alignment),
        "legacy_quercus_differing_rows": len(quercus), "old_native_variable_keys": len(old_keys),
        "new_native_variable_keys": len(new_keys), "shared_unambiguous_keys": len(shared),
        "ambiguous_keys_excluded": len(ambiguous), "pilot_samples": len(pilot), "candidate_status_counts": dict(counts),
        "unmatched_diagnostic_rows": len(unmatched), "unmatched_variable_differences": len(differences),
        "unmatched_diagnostic_field_counts": dict(collections.Counter(str(r["minimum_differing_fields"]) for r in unmatched)),
        "unmatched_diagnostic_taxonids": dict(collections.Counter(r["taxonid"] for r in differences)),
        "age_used_for_matching": False, "depth_used_for_26608_to_47525_candidate_search": False,
        "same_physical_sample_proven": False, "mon16_mapping_proven": False, "p1_status": "open",
        "raw_snapshot_is_live_neotoma_state": False,
        "input_sha256": {"itDownload.RDS": RDS_SHA256, "monticchio_739452.tab": PANGAEA_SHA256,
                         "pr25_sample_identity.csv": expected_prior}})
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())}
    write_json(output / "manifest.json", hashes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rds", type=Path, required=True)
    parser.add_argument("--pangaea", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.rds, args.pangaea, args.prior, args.output)
