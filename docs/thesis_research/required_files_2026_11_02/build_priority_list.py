"""Resolve the reviewed priority list against the bounded read-only audit."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def main():
    calls = list(csv.DictReader((OUT / "dependency_calls.csv").open(encoding="utf-8")))
    curated = json.loads((OUT / "curation.json").read_text(encoding="utf-8"))
    entries = []
    for item in curated["entries"]:
        matches = [r for r in calls if r["source"] == item["source"] and int(r["line"]) == item["line"]]
        if len(matches) != 1:
            raise ValueError("Reviewed source reference no longer unique: " + item["id"])
        call = matches[0]
        entries.append({**item, "filename_or_pattern": call["path_or_pattern"], "availability": call["availability"],
                        "source_modified_utc": call["modified_utc"], "automount_origin": call["automount_origin"],
                        "same_site_path_tail_candidates": call["same_site_path_tail_candidates"],
                        "local_candidates": call["local_candidates"], "archive_candidates": call["archive_candidates"],
                        "producer_hints_not_execution_proof": call["producer_hints_not_execution_proof"]})
    with (OUT / "priority_requirements.csv").open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entries[0]), lineterminator='\n'); writer.writeheader(); writer.writerows(entries)
    # Recheck the eight explicitly documented Kinneret output variants by exact basename.
    cfg = json.loads((ROOT / ".local/my_reconst_paths.json").read_text())
    index = json.loads((ROOT / ".local/required_files_2026_11_02/data_index.json").read_text())
    posterior = list(csv.DictReader((ROOT / "docs/thesis_research/kinneret_postprocessing_2026_11_02/posterior_paths.csv").open()))
    for row in posterior:
        name = Path(row["relative_path"]).name.lower()
        row["local_filename_candidates_now"] = [e["label"] for e in index if e["basename"] == name]
        row["meaning"] = "Historical result/reference variant, not external input required before generating a new result; none selected as final"
    (OUT / "posterior_variants_recheck.json").write_text(json.dumps(posterior,indent=2)+'\n',encoding='utf-8',newline='\n')
    # Existing small distribution containers already structurally bound in the prior audit.
    prior = json.loads((ROOT / ".local/my_reconst_source_audit_v3/audit.json").read_text())
    preserved = []
    for relative in ["my_ohrid/data/in/climate_data/taxa_distribution_ww.rdata", "my_kinneret/data/in/climate_data/taxa_distribution_ww.rdata"]:
        path = Path(cfg["additional_source_root"]) / relative
        raw = path.read_bytes(); sha = hashlib.sha256(raw).hexdigest()
        assert sha == prior["fingerprints"][relative]
        preserved.append({"source": "additional:"+relative, "bytes":len(raw), "sha256":sha,
                          "matches_prior_structural_audit":True,
                          "identity_with_lost_original_run_claimed":False})
    (OUT / "preserved_copies.json").write_text(json.dumps(preserved,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({"reviewed_priority_rows":len(entries),"P0_rows":sum(e["priority"]=="P0_R_Einstieg" for e in entries),
                      "posterior_variants":len(posterior),"posterior_variants_without_local_filename_candidate":sum(not p["local_filename_candidates_now"] for p in posterior),
                      "already_bound_distribution_copies_hash_verified":len(preserved)}))


if __name__ == "__main__":
    main()
