#!/usr/bin/env python3
"""Apply an explicit, conservative fossil-field policy; no training inputs.

Place beside import_data.py and audit_readiness.py in tools/thesis_import/.
Input policy and new-source evidence are in docs/thesis_research/taxon_decisions_2026_10_29/.
"""
import argparse
from contextlib import closing
import collections
import csv
from decimal import Decimal
import json
from pathlib import Path
import sqlite3
import tempfile

import import_data
import audit_readiness
from audit_readiness import audit, norm
from import_data import locked_sources, read_pangaea, require, digest


def decisions(raw_dir, catalog, policy_path, evidence_path, evidence_dir):
    sources=locked_sources(raw_dir,catalog)
    _, candidates=audit(raw_dir,catalog)
    policy=json.loads(policy_path.read_text(encoding="utf-8"))
    registry=json.loads(evidence_path.read_text(encoding="utf-8"))
    require(policy["scope"].startswith("fossil field selection"),"Unexpected policy scope")
    reference=next(x for x in registry["files"] if x["source_id"]=="neotoma_27166")
    path=evidence_dir/reference["file"]
    require(path.stat().st_size==reference["byte_count"] and digest(path)==reference["sha256"],"Neotoma snapshot changed")
    neo=json.loads(path.read_text(encoding="utf-8"))["data"][0]["site"]["collectionunit"]
    require("LGM-90D" in neo["notes"],"Wrong core context")
    ds=neo["dataset"]
    markers=[d for s in ds["samples"] for d in s["datum"] if d["variablename"]=="Eucalyptus spike"]
    require(bool(markers) and all(d["ecologicalgroup"]=="LABO" and d["element"]=="counted" for d in markers),"Marker role not established")
    _,header,pollen=read_pangaea(sources["monticchio_739452"][1])
    configured=[]
    for key in ('exclude','manual_unresolved','modern_id_selection','modern_pool_candidates','accepted_database_aliases'):
        configured.extend(policy[key])
    require(len(configured)==len(set(configured)),"Policy actions overlap")
    require(set(configured)<=set(header[1:]),"Policy mentions unknown fields")
    _,_,ages=read_pangaea(sources["epd_ages_cut_804596"][1])
    age_depths=[Decimal(r["Depth sed [m]"]) for r in ages if r["Event"]=="LGM90D" and r["ID"]=="932"]
    retained=[(i,r) for i,r in enumerate(pollen,1) if any(abs(Decimal(r["Depth sed [m]"])-d)<=Decimal("0.005") for d in age_depths)]
    require(len(retained)==69,"Retained cohort changed")
    oak=[r for r in pollen if Decimal(r["Que.r-T [#]"])+Decimal(r["Que.i [#]"])>0]
    require(len(oak)==23 and all(Decimal(r["Que [#]"])==Decimal(r["Que.r-T [#]"])+Decimal(r["Que.i [#]"]) for r in oak),"Oak subdivision evidence changed")
    corn=[r for r in pollen if Decimal(r["Cornacae [#]"])>0]
    corn_both=[r for r in corn if Decimal(r["Crn [#]"])>0]
    require(len(corn_both)==1 and all(Decimal(r["Cornacae [#]"])==Decimal(r["Crn [#]"]) for r in corn_both),"Cornus cooccurrence evidence changed")
    rows=[]
    with closing(sqlite3.connect(sources["empd2"][1].resolve().as_uri()+"?mode=ro",uri=True)) as con:
        for item in candidates:
            column=item["original_column"]
            targets=item["candidates"]
            action,reason,selected="unresolved","concept_or_resolution_not_established",[]
            if column in policy["exclude"]:
                action="exclude"
                reason={"Euc [#]":"marker_risk_same_core_LABO_evidence",
                        "conc.spk [#]":"laboratory_quantity",
                        "vol.sp.a [cm**3]":"laboratory_volume",
                        "Que.r-T [#]":"subdivision_duplicates_Quercus_total",
                        "Que.i [#]":"subdivision_duplicates_Quercus_total",
                        "Cypae [#]":"terrestrial_pilot_scope_mixed_HERB_AQUA"}[column]
            elif item["name_match_status"]=="technical_or_indeterminate":
                action,reason="exclude","technical_or_indeterminable_archive_field"
            elif targets and all(set(t["group_ids"])<=set(policy["nonpollen_groups"]) for t in targets):
                action,reason="exclude","aquatic_spore_or_nonpollen_scope"
            elif column in policy["manual_unresolved"]:
                reason="hierarchy_or_counting_convention_unresolved"
            elif item["original_name"].startswith("cf."):
                reason="uncertain_original_identification"
            elif column in policy["modern_pool_candidates"]:
                selected=policy["modern_pool_candidates"][column]
                selected_targets=[t for t in targets if t["accepted_taxon_id"] in selected]
                require(len(selected_targets)==len(selected),"Pool candidates absent")
                require(len({norm(t["accepted_name"]) for t in selected_targets})==1,"Different pool concepts")
                placeholders=','.join('?' for _ in selected)
                sql=f"""SELECT count(*) FROM (SELECT samplename FROM p_counts JOIN p_vars USING(var_)
                     WHERE count>0 AND acc_var_ IN ({placeholders}) GROUP BY samplename HAVING count(*)>1)"""
                require(con.execute(sql,selected).fetchone()[0]==0,"Potential duplicate modern observations in pool")
                action,reason="pool","candidate_modern_namespace_pool_zero_cooccurrence_not_executed"
            elif column in policy["modern_id_selection"]:
                selected=policy["modern_id_selection"][column]
                selected_targets=[t for t in targets if t["accepted_taxon_id"] in selected]
                require(len(selected_targets)==len(selected),"Selected target absent")
                require(all(norm(t["accepted_name"])==norm(item["archive_comment"]) for t in selected_targets),"Selected concept differs from EPD comment")
                action,reason="include","EPD_accepted_comment_matches_modern_accepted_name"
            elif column in policy["accepted_database_aliases"]:
                selected=[policy["accepted_database_aliases"][column]]
                require(any(t["accepted_taxon_id"]==selected[0] for t in targets),"Alias target absent")
                action,reason="include","explicit_EMPD_database_alias_operational_only"
            elif len(targets)==1 and norm(targets[0]["accepted_name"])==norm(item["archive_comment"] or item["original_name"]):
                action,reason,selected="include","exact_archive_and_modern_accepted_concept",[targets[0]["accepted_taxon_id"]]
            if action=="unresolved": selected=[]
            rows.append(dict(original_column=column,original_name=item["original_name"],archive_comment=item["archive_comment"],
                    action=action,reason=reason,candidate_modern_ids='|'.join(selected),
                    candidate_modern_names='|'.join(t["accepted_name"] for t in targets if t["accepted_taxon_id"] in selected),
                    empirical_evidence=("same_core_Neotoma_27166_and_Allen2018_analogue" if column=="Euc [#]" else
                        "all_245_source_rows" if column in ("Que.r-T [#]","Que.i [#]","Cornacae [#]","Crn [#]") else "source_metadata_and_locked_dictionary"),
                    scope="fossil_catalogue_only",scientific_status="not_training_approved"))
    by_column={r["original_column"]:r for r in rows}
    sample_rows=[]
    for index,record in retained:
        sums=collections.defaultdict(Decimal)
        for column in header[1:]:
            if "[#]" in column and record[column]:
                sums[by_column[column]["action"]]+=Decimal(record[column])
        denominator=sums['include']+sums['pool']+sums['unresolved']
        sample_rows.append(dict(sample_id=f"LGM90D:{index:06d}",depth_m=record['Depth sed [m]'],
            include_count=str(sums['include']),pool_candidate_count=str(sums['pool']),unresolved_count=str(sums['unresolved']),
            excluded_count=str(sums['exclude']),audit_count_after_exclusions=str(denominator),
            selected_count_share=str((sums['include']+sums['pool'])/denominator) if denominator else "",
            biological_pollen_sum="unresolved",scientific_status="not_training_approved"))
    summary=dict(schema_version=1,actions=dict(collections.Counter(r['action'] for r in rows)),
        fields=len(rows),retained_samples=len(retained),
        excluded_marker_count=sum(int(r['Euc [#]']) for _,r in retained),
        excluded_quercus_subdivision_count=sum(int(r['Que.r-T [#]'])+int(r['Que.i [#]']) for _,r in retained),
        oak_subtype_positive_rows_all=len(oak),oak_subtype_rows_equal_parent_all=len(oak),
        cornaceae_positive_rows_all=len(corn),
        cornaceae_cornus_copositive_rows_all=len(corn_both),
        neotoma_dataset_id=ds['datasetid'],neotoma_samples=len(ds['samples']),neotoma_age_range=ds['agerange'],
        neotoma_role="laboratory_marker_evidence_only_not_a_long_pilot",
        unresolved_fields=[r['original_column'] for r in rows if r['action']=='unresolved'],
        catalogue_includes_are_not_training_approval=True,
        modern_values_pooled=False,biological_pollen_sum=None,
        scientific_status="not_training_approved")
    return rows,sample_rows,summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    base=Path(__file__).resolve().parents[2]/"docs/thesis_research"
    parser.add_argument('--raw-dir',type=Path,required=True)
    parser.add_argument('--evidence-dir',type=Path,required=True)
    parser.add_argument('--catalog',type=Path,default=base/'source_audit_2026_10_29/sources.json')
    parser.add_argument('--policy',type=Path,default=base/'taxon_decisions_2026_10_29/taxon_policy.json')
    parser.add_argument('--evidence',type=Path,default=base/'taxon_decisions_2026_10_29/source_evidence.json')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    require(not args.output_dir.exists(),'Output directory already exists')
    rows,samples,summary=decisions(args.raw_dir,args.catalog,args.policy,args.evidence,args.evidence_dir)
    args.output_dir.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_dir.parent) as temp:
        stage=Path(temp)/'result';stage.mkdir()
        for name,data in [('fossil_decisions.csv',rows),('sample_selection_audit.csv',samples)]:
            with (stage/name).open('w',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');writer.writeheader();writer.writerows(data)
        (stage/'decision_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        (stage/'provenance.json').write_text(json.dumps(dict(script_sha256=digest(Path(__file__)),
            policy_sha256=digest(args.policy),evidence_registry_sha256=digest(args.evidence),
            source_catalog_sha256=digest(args.catalog),
            helper_sha256={'import_data.py':digest(Path(import_data.__file__)),
                           'audit_readiness.py':digest(Path(audit_readiness.__file__))},
            outputs={p.name:digest(p) for p in sorted(stage.iterdir())}),indent=2)+'\n',encoding='utf-8')
        stage.rename(args.output_dir)
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=='__main__': main()
