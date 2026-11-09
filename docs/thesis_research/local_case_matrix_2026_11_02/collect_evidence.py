"""Read only the explicit case sources; never execute historical R or scan roots.

Run from the existing checkout with its private .local/my_reconst_paths.json.
The JSON export contains metadata, hashes and column labels, never pollen counts.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
import zipfile
import numpy as np
import rdata

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CONFIG = json.loads((REPO / '.local/my_reconst_paths.json').read_text(encoding='utf-8'))
ROOTS = {'original': Path(CONFIG['original_data_root']),
         'additional': Path(CONFIG['additional_source_root']), 'repository': REPO}
records = {}

def bind(scope, rel):
    key = scope + ':' + rel
    if key in records:
        return key
    p = ROOTS[scope] / rel
    item = {'scope': scope, 'relative_path': rel, 'exists': p.is_file()}
    if item['exists']:
        item.update(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    records[key] = item
    return key

def describe(value):
    if isinstance(value, dict):
        return {str(k): describe(v) for k, v in value.items()}
    if isinstance(value, list):
        return {'kind': 'list', 'items': [describe(v) for v in value]}
    a = np.asarray(value)
    result = {'shape': list(a.shape), 'dtype': str(a.dtype)}
    if a.dtype.kind in 'if' and a.size:
        result['all_finite'] = bool(np.isfinite(a).all())
        if a.ndim <= 1:
            result.update(min=float(np.nanmin(a)), max=float(np.nanmax(a)))
    return result

spec = [
    ('dead_sea_holocene','my_gedi','data/in/core/en_gedi_counts.csv','data/out/age_depth/rho_depth_given_age.rdata',14,'gedi','gedi'),
    ('dead_sea_glacial','my_gedi2','data/in/core/taxa_spectrum.csv','data/in/age_depth/rho_depth_given_age_higher_resol.rdata',19,'gedi2','dead_sea'),
    ('kinneret','my_kinneret','data/in/core/pollen_counts_epd_taxanomy.csv','data/out/age_depth/rho_depth_given_age.rdata',60,'kinneret','kinneret'),
    ('birkat_ram','my_birkat_ram','data/in/core/pollen_counts_composite.csv','data/out/age_depth/rho_depth_given_age.rdata',14,'birkat2','birkat'),
    ('van','my_van','data/in/core/taxa_spectrum.csv','data/in/age_depth/rho_depth_given_age.rdata',21,'van','van'),
    ('iznik','my_iznik','data/in/core/pollen_counts_epd_taxanomy.csv','data/out/age_depth/rho_depth_given_age.rdata',106,'iznik','iznik'),
    ('ohrid','my_ohrid','data/in/core/core_data_counts.csv','data/out/age_depth/rho_depth_given_age.rdata',46,'ohrid','ohrid'),
]
with zipfile.ZipFile(ROOTS['original']/'out/plots/mcmc/v5.0/diss_final.zip') as archive:
    cases = []
    for case_id, folder, pollen_rel, age_rel, nlabels, plotfolder, zipfolder in spec:
        pollen_key = bind('additional',folder+'/'+pollen_rel)
        p = ROOTS['additional']/folder/pollen_rel
        with p.open(encoding='utf-8-sig',newline='') as f:
            rows = list(csv.reader(f))
        # Explicit source column slices, not the final retained TF/taxon list.
        pollen = {'source': pollen_key, 'rows': len(rows)-1,
                  'header_columns': len(rows[0]), 'candidate_taxon_columns': rows[0][1:1+nlabels],
                  'taxon_list_status': 'candidate columns only; historical retained mapping unresolved',
                  'depth_first_literal': rows[1][0], 'depth_last_literal': rows[-1][0],
                  'uniform_row_width': all(len(row)==len(rows[0]) for row in rows)}
        age_key=bind('additional',folder+'/'+age_rel)
        age={'source':age_key}
        if records[age_key]['exists']:
            age['structure']=describe(rdata.read_rda(ROOTS['additional']/folder/age_rel))
        matches=[]
        for filename in ['reconstruction.pdf','reference_curves.pdf']:
            rel='out/plots/case_studies/'+plotfolder+'/'+filename
            key=bind('original',rel)
            member='figures/'+zipfolder+'/'+filename
            archive_hash=hashlib.sha256(archive.read(member)).hexdigest()
            matches.append({'source':key, 'archive_member':member,
                            'archive_member_sha256':archive_hash,
                            'byte_equal':records[key].get('sha256')==archive_hash})
        cases.append({'id':case_id,'pollen':pollen,'chronology':age,'figure_comparisons':matches})

source_ranges={
 'main_v1.2.R':[[65,79],[107,181],[185,291],[325,347],[388,433],[568,591],[600,678]],
 'src/mcmc_prepare.R':[[1,43],[109,109],[306,330],[710,725],[1010,1046],[1138,1150],[1180,1265],[1290,1386]],
 'src/mcmc_functions_v1.6.R':[[280,363],[450,477],[1242,1284],[1781,1820]],
 'src/mcmc_conduction.R':[[138,145],[427,463]],
 'src/mcmc_trimming_posterior_trafo.R':[[94,113]],
 'src/all_trafo.R':[[45,52],[106,113],[217,229],[407,411],[469,529]],
 'src/prior_climate.R':[[1,80],[108,130]],
 'src/check_ages.R':[[7,19],[103,113],[174,231]],
}
for rel, ranges in source_ranges.items():
    key=bind('original',rel);records[key]['inspected_line_ranges']=ranges
for scope,rel in [
 ('original','out/plots/mcmc/v5.0/diss_final.pdf'),
 ('original','out/plots/mcmc/v5.0/diss_final.zip'),
 ('repository','docs/paper/Phd_thesis_netzel_ch3.3.PDF'),
 ('repository','docs/paper/Phd_thesis_netzel_ch3.2.PDF'),
 ('repository','docs/paper/Phd_thesis_netzel_ch2.5.PDF'),
 ('repository','.local/inventory/data_catalog.csv'),
 ('repository','.local/my_reconst_source_audit_v3/source_inventory.csv'),
 ('additional','my_gedi2/data/in/age_depth/rho_depth_given_age.rdata'),
 ('additional','my_iznik/data/in/core/core_information.csv'),
 ('additional','my_ohrid/data/out/age_depth/core_data_ohrid.rdata'),
 ('original','in/core_data_additional_sites_v1.2.rdata'),
 ('original','in/reference_speleo_iso.rdata'),
 ('original','in/isotope_info.rdata'),
 ('original','in/isotope_speleo_other.rdata'),
 ('original','out/transfer_functions/prob_c_kgc.rdata'),
 ('original','out/transfer_functions/mixed_partial_arts_emd.rdata'),
 ('original','out/transfer_functions/mixed_partial_arts_emd_info.rdata'),
]: bind(scope,rel)

# Exact page locators for both separately named editions, not a fixed offset.
sys.path.insert(0,str(REPO/'.local/console_provenance_v1/pdf_runtime'))
import pymupdf
page_checks=[]
for scope,rel,tests in [
 ('original','out/plots/mcmc/v5.0/diss_final.pdf',[(79,'Holocene'),(87,'14.5'),(96,'3.3.2'),(104,'3.3.3'),(111,'3.3.4'),(119,'3.3.5'),(125,'3.3.6'),(125,'365')]),
 ('repository','docs/paper/Phd_thesis_netzel_ch3.3.PDF',[(3,'Holocene'),(11,'14.5'),(20,'3.3.2'),(27,'3.3.3'),(34,'3.3.4'),(42,'3.3.5'),(47,'3.3.6')]),
]:
    with pymupdf.open(ROOTS[scope]/rel) as pdf:
        for number,needle in tests:
            text=pdf[number-1].get_text()
            page_checks.append({'source':scope+':'+rel,'pdf_page':number,'needle':needle,'found':needle in text})

unchanged=[]
for key,item in records.items():
    p=ROOTS[item['scope']]/item['relative_path']
    ok = p.is_file()==item['exists']
    if item['exists']:
        ok=ok and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
    unchanged.append(ok)
result={'schema_version':1,'date':'2026-11-02','base_commit':'865d155c6f2d69f0a63e2ec7ce216c99e7954181',
        'scope':'Seven explicitly named local cases; existing catalogs reused; no traversal or R execution',
        'roots':'Resolve original/additional via private .local/my_reconst_paths.json; repository is this checkout',
        'cases':cases,'sources':records,'page_checks':page_checks,
        'checks':{'case_count':len(cases),'all_sources_unchanged_during_read':all(unchanged),
                  'all_page_needles_found':all(x['found'] for x in page_checks),
                  'figure_byte_matches':sum(m['byte_equal'] for c in cases for m in c['figure_comparisons']),
                  'figure_comparison_count':14},
        'limits':['CSV rows are not proven retained samples or numeric TF order',
                  'RData structure and ranges are not validated R predictions or a verified chronology',
                  'Figure byte equality binds archive graphics, not underlying numeric samples',
                  'Missing means absent at the explicit consumer path; prior broader negative audits remain separate']}
(OUT/'evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result['checks'],ensure_ascii=False))
print([(c['id'],[(m['archive_member'],m['byte_equal']) for m in c['figure_comparisons']]) for c in cases])
