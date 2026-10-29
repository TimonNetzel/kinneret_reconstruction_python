#!/usr/bin/env python3
"""Audit candidate cohorts and name matches; never approve a training dataset.

Run from tools/thesis_import/ in a repository checkout. Output directory must be new.
No network access, taxon aggregation, zero filling, or model training.
"""
import argparse
import collections
import csv
from decimal import Decimal
import json
from pathlib import Path
import re
import sqlite3
from contextlib import closing
import tempfile

from import_data import locked_sources, read_pangaea, require, digest


def norm(text):
    return " ".join((text or "").strip().casefold().split())


def audit(raw_dir, catalog):
    sources = locked_sources(raw_dir, catalog)
    params_text, header, pollen = read_pangaea(sources["monticchio_739452"][1])
    _, _, ages = read_pangaea(sources["epd_ages_cut_804596"][1])
    depths = [Decimal(r["Depth sed [m]"]) for r in ages if r["Event"] == "LGM90D" and r["ID"] == "932"]
    retained = [r for r in pollen if any(abs(Decimal(r["Depth sed [m]"])-d) <= Decimal("0.005") for d in depths)]
    require(len(retained) == len(depths) == 69, "Retained cohort differs from locked reference")
    parameters = {}
    for line in params_text.splitlines():
        match = re.match(r"^\t(.+?) \[([^\]]+)\] \(([^)]+)\)", line)
        if match:
            name, unit, code = match.groups()
            comment = line.split(" * COMMENT: ", 1)[1] if " * COMMENT: " in line else ""
            parameters[f"{code} [{unit}]"] = dict(original_name=name, archive_comment=comment, unit=unit)
    require(set(header[1:]) == set(parameters), "Parameter metadata do not cover every data column")
    with closing(sqlite3.connect(sources["empd2"][1].resolve().as_uri()+"?mode=ro", uri=True)) as con:
        con.row_factory = sqlite3.Row
        con.execute("""CREATE TEMP VIEW complete AS
          SELECT m.* FROM metadata m JOIN climate k USING(samplename)
          WHERE typeof(k.t_ann) IN ('integer','real') AND typeof(k.p_ann) IN ('integer','real')
          AND typeof(m.latitude) IN ('integer','real') AND m.latitude BETWEEN -90 AND 90
          AND typeof(m.longitude) IN ('integer','real') AND m.longitude BETWEEN -180 AND 180
          AND m.ispercent=0""")
        con.execute("""CREATE TEMP VIEW strict_cohort AS SELECT * FROM complete m
          WHERE locationreliability IN ('A','B','C') AND NOT EXISTS
          (SELECT 1 FROM p_counts c WHERE c.samplename=m.samplename AND c.count!=cast(c.count AS integer))""")
        n = lambda sql: con.execute(sql).fetchone()[0]
        cohorts = dict(all_metadata=n("SELECT count(*) FROM metadata"),
                       numeric_climate_coords_nonpercent=n("SELECT count(*) FROM complete"),
                       complete_quality_ABC=n("SELECT count(*) FROM complete WHERE locationreliability IN ('A','B','C')"),
                       complete_quality_ABC_integer_only=n("SELECT count(*) FROM strict_cohort"),
                       strict_in_25_75N_15W_45E_box=n("SELECT count(*) FROM strict_cohort WHERE latitude BETWEEN 25 AND 75 AND longitude BETWEEN -15 AND 45"))
        qualities = [dict(row) for row in con.execute("""SELECT locationreliability AS code,count(*) AS n
                       FROM complete GROUP BY locationreliability ORDER BY locationreliability""")]
        groups = [dict(row) for row in con.execute("SELECT groupid,groupname,used_in_sum FROM groupid ORDER BY groupid")]
        taxa = [dict(row) for row in con.execute("""SELECT v.*,count(DISTINCT c.samplename) AS positive_samples
            FROM p_vars v LEFT JOIN p_counts c ON c.var_=v.var_ AND c.count>0
            AND c.samplename IN (SELECT samplename FROM strict_cohort)
            GROUP BY v.var_ ORDER BY v.var_""")]
        index = collections.defaultdict(list)
        for taxon in taxa:
            if taxon["positive_samples"]:
                for field in ("original_varname", "acc_varname"):
                    if norm(taxon[field]):
                        index[norm(taxon[field])].append((field, taxon))
        mapping = []
        for column in header[1:]:
            param = parameters[column]
            found = {}
            for source_field in ("original_name", "archive_comment"):
                for modern_field, taxon in index.get(norm(param[source_field]), []):
                    target_id = str(taxon["acc_var_"])
                    hit = found.setdefault(target_id, dict(accepted_taxon_id=target_id,
                         accepted_name=taxon["acc_varname"], modern_original_codes=set(),
                         evidence=set(), group_ids=set(), positive_samples_by_original_code={}))
                    hit["modern_original_codes"].add(taxon["var_"])
                    hit["group_ids"].add(taxon["groupid"])
                    hit["evidence"].add(f"fossil.{source_field} == empd.{modern_field}")
                    hit["positive_samples_by_original_code"][str(taxon["var_"])] = taxon["positive_samples"]
            candidates = []
            for hit in found.values():
                for field in ("modern_original_codes", "evidence", "group_ids"):
                    hit[field] = sorted(hit[field], key=str)
                candidates.append(hit)
            candidates.sort(key=lambda x:x["accepted_taxon_id"])
            technical = param["unit"] != "#" or "indeterminable" in norm(param["archive_comment"]) or "concentration" in norm(param["original_name"])
            status = "technical_or_indeterminate" if technical else ("no_name_candidate" if not candidates else
                      "one_name_candidate" if len(candidates)==1 else "multiple_name_candidates")
            total = sum(Decimal(r[column]) for r in retained if r[column])
            mapping.append(dict(original_column=column, **param, name_match_status=status,
                  retained_value_sum=str(total), retained_nonzero_samples=sum(bool(r[column]) and Decimal(r[column])>0 for r in retained),
                  candidates=candidates, scientific_status="unreviewed_name_matches"))
    targets = collections.defaultdict(list)
    for row in mapping:
        for target in row["candidates"]:
            targets[target["accepted_taxon_id"]].append(row["original_column"])
    for row in mapping:
        row["other_columns_sharing_targets"] = sorted({column for t in row["candidates"] for column in
                                targets[t["accepted_taxon_id"]] if column != row["original_column"]})
    summary = dict(schema_version=1, source_sha256={key:source["sha256"] for key,(source,_) in sources.items()},
                   cohort_scenarios=cohorts, quality_codes_in_complete_cohort=qualities,
                   monticchio_retained_samples=len(retained), monticchio_fields=len(mapping),
                   name_match_status_counts=dict(collections.Counter(x["name_match_status"] for x in mapping)),
                   fields_with_shared_target_candidates=sum(bool(x["other_columns_sharing_targets"]) for x in mapping),
                   quercus_identity_check=dict(source_rows=len(pollen),
                      genus_equals_robur_plus_ilex_rows=sum(Decimal(r["Que [#]"]) == Decimal(r["Que.r-T [#]"])+Decimal(r["Que.i [#]"]) for r in pollen),
                      interpretation="Equality alone does not prove the counting or aggregation convention."),
                   archive_group_definitions=groups, scientific_status="not_approved",
                   notes=["Cohorts are nested audit scenarios, not selected training sets.",
                          "Name matches do not establish equivalent taxon concepts.",
                          "Positive-sample counts are per original code, not additive across synonyms.",
                          "used_in_sum includes spore groups; it is not a terrestrial-pollen selector.",
                          "No values were pooled; no missing records were filled with zero."])
    return summary, mapping


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir",type=Path,required=True)
    parser.add_argument("--catalog",type=Path,default=Path(__file__).resolve().parents[2]/"docs/thesis_research/source_audit_2026_10_29/sources.json")
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    require(not args.output_dir.exists(), "Output directory already exists")
    summary, mapping=audit(args.raw_dir,args.catalog)
    args.output_dir.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_dir.parent) as temp:
        stage=Path(temp)/"result";stage.mkdir()
        for name,value in (("readiness.json",summary),("taxon_name_candidates.json",mapping)):
            (stage/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        fields=["original_column","original_name","archive_comment","name_match_status","candidate_ids",
                "candidate_names","candidate_groups","retained_nonzero_samples","retained_value_sum",
                "other_columns_sharing_targets","scientific_status"]
        with (stage/"taxon_mapping_review.csv").open("w",encoding="utf-8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n");writer.writeheader()
            for row in mapping:
                out={k:row[k] for k in fields if k in row}
                out["candidate_ids"]="|".join(x["accepted_taxon_id"] for x in row["candidates"])
                out["candidate_names"]="|".join(x["accepted_name"] for x in row["candidates"])
                out["candidate_groups"]="|".join('/'.join(x["group_ids"]) for x in row["candidates"])
                out["other_columns_sharing_targets"]="|".join(row["other_columns_sharing_targets"])
                writer.writerow(out)
        (stage/"provenance.json").write_text(json.dumps(dict(script_sha256=digest(Path(__file__)),
            outputs={p.name:digest(p) for p in sorted(stage.iterdir())}),indent=2)+"\n",encoding="utf-8")
        stage.rename(args.output_dir)
    print(json.dumps({k:v for k,v in summary.items() if k not in ("archive_group_definitions","source_sha256")},indent=2))


if __name__=="__main__":
    main()
