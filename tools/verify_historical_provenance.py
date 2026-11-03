"""Independent assertions over retained historical provenance evidence.

No R execution, model fitting, data substitution, or scientific pilot approval.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from audit_my_reconst_sources import metadata, sha


def verify(config, base):
    cfg=json.loads(config.read_text(encoding='utf-8-sig'))
    roots={'original':Path(cfg['original_data_root']), 'additional':Path(cfg['additional_source_root'])}
    base=base.resolve(strict=True)
    if any(base==root.resolve() or root.resolve() in base.parents or base in root.resolve().parents for root in roots.values()):raise ValueError('Evidence directory overlaps original source tree')
    read=lambda p:json.loads((base/p).read_text(encoding='utf-8'))
    raster=read('raster_v2/raster.json'); planes=read('raster_v2/raster_planes.json')
    models=read('models_v2/models.json'); climate=read('climate/climate_headers.json')
    source_spans=read('source_evidence.json')+read('partial_model_source_evidence.json'); times=read('climate_time_bounds.json')
    distribution=read('ohrid_279_distribution.json'); nonbinary=read('eurasia_nonbinary.json'); archives=read('archive_text_search.json')
    prior=json.loads(Path('.local/my_reconst_verification.json').read_text(encoding='utf-8'))
    prior_root=Path('.local/my_reconst_source_audit_v3')
    totals=raster['totals']
    checks={
        'raster_has_275_ordered_planes':len(planes)==275 and [r['position_R'] for r in planes]==list(range(1,276)),
        'all_regional_differences_accounted_for':sum(r['global_vs_regional']['finite_value_mismatches'] for r in planes)==95006
            and totals['regional_inside_vs_eurasia.finite_value_mismatches']==95006
            and totals['global_inside_vs_eurasia.finite_value_mismatches']==0
            and totals['regional_differences_outside_eurasia']==0,
        'regional_masks_unchanged':totals['global_vs_regional.mask_mismatches']==0,
        'eurasia_RData_and_netcdf_agree':raster['eurasia_RData_and_NetCDF_arrays_equal'],
        'regional_netcdf_agrees_with_eurasia_on_shared_grid':all(x==0 for x in raster['regional_nc_vs_eurasia_on_shared_coordinates'].values()),
        'documented_edge_insertion_matches_observed_mask':all(v==0 for k,v in totals.items() if k.startswith('global_vs_drop_last_lon_observed_mask.')),
        'direct_insertion_differs_by_74_cells':totals['global_vs_direct_insertion_observed_mask.finite_value_mismatches']==74,
        'source_has_two_nonbinary_points':len(nonbinary['nonbinary_points'])==2 and all(p['source_value']==2 and p['name']=='Pinus brutia' for p in nonbinary['nonbinary_points']),
        'saved_279_snapshot_matches_all_source_planes':distribution['equal_plane_count']==279 and all(r['matches_previous_283_plane'] for r in distribution['planes'])
             and distribution['position110']=='Juglans sp.',
        'climate_header_coverage':climate['file_count']==51 and climate['parse_status_counts']=={'NetCDF3_header_read':50,'not_decoded':1},
        'climate_time_bounds_recorded':len(times)==4 and all(r['units']=='days since 1900-1-1' and r['calendar']=='gregorian' for r in times)
            and all(r['decoded_bounds'][0].startswith('1901-' if r['source_tree']=='original' else '1961-')
                and r['decoded_bounds'][1].startswith('2015-' if r['source_tree']=='original' else '1990-') for r in times),
    }
    checks['previous_v3_reference_artifact_hashes_intact']=all(sha(prior_root/name)==h for name,h in prior['artifact_sha256'].items())
    with (prior_root/'global_plane_summary.csv').open(encoding='utf-8') as f:old_planes=list(csv.DictReader(f))
    checks['global_reference_cells_all_binary_or_missing']=len(old_planes)==558 and all(sum(int(row[k]) for k in ['global_presence','global_absence','global_missing'])==720*360 for row in old_planes)
    checks['archive_search_coverage_recorded']=archives['archive_count']==18 and archives['unique_archive_contents']==18 and archives['text_members_read']==78 and len(archives['hits'])==0
    before=[read(stage+'/source_metadata_before.json') for stage in ['raster_v2','models_v2','climate']]
    now=[metadata(roots['original']),metadata(roots['additional'])]
    checks['both_original_inventories_unchanged']=all(all({k:list(v) for k,v in current.items()}==old for current,old in zip(now,stage)) for stage in before)
    inventory=list(csv.DictReader(open('.local/my_reconst_source_audit_v3/source_inventory.csv',encoding='utf-8')))
    selected={r['relative_path'] for r in inventory if r['relative_path'].lower().endswith('.rdata') and any(x in Path(r['relative_path']).parts for x in ['transfer_function','ml_competition'])}
    recorded=[p for r in models['results'] for p in r['relative_paths']]
    checks['selected_model_coverage_exact']=len(recorded)==len(set(recorded))==models['candidate_files']==109 and set(recorded)==selected
    partial=next(r for r in models['results'] if 'my_kinneret/data/out/transfer_function/qda_fit_smoted_partial.rdata' in r['relative_paths'])
    paired=next(r for r in models['results'] if 'my_kinneret/data/out/transfer_function/comb_qda_fit_smoted_partial.rdata' in r['relative_paths'])
    checks['six_partial_binary_QDA_fits_explicit']=partial.get('caret_fit_count')==partial.get('binary_caret_fit_count')==6 and all(f['method']==['qda'] and f['levels']==['taxa1','unknown'] and f['coefnames']==['t.djf.norm','p.ann.norm'] for f in partial['fits'])
    checks['fifteen_pair_QDA_fits_explicit']=paired.get('caret_fit_count')==paired.get('multiclass_caret_fit_count')==15 and all(f['method']==['qda'] and len(f['levels'])==3 and 'unknown' in f['levels'] for f in paired['fits'])
    checks['model_parse_counts_exact']=models['unique_byte_contents']==len(models['results'])==78 and models['parse_status_counts']==dict(Counter(r['parse_status'] for r in models['results']))
    checks['model_fit_counts_exact']=all(models[total]==sum(r.get(count,0) for r in models['results']) for total,count in [('complete_caret_fits','caret_fit_count'),('complete_binary_caret_fits','binary_caret_fit_count'),('complete_multiclass_caret_fits','multiclass_caret_fit_count')])
    checks['no_R_prediction_claims']=all(r.get('R_prediction_validated') is False for r in models['results'] if r['parse_status']=='complete_structure_not_R_validation')
    # Rehash every inspected model, climate file and captured source, not only its name/size.
    references={}
    for r in models['results']:
        for p in r['relative_paths']:references[roots['additional']/p]=r['sha256']
    for r in climate['results']+source_spans+times+archives['archives']: references[roots[r['source_tree']]/r['relative_path']]=r['sha256']
    known={'global':roots['additional']/'my_kinneret/data/in/climate_data/taxa_distribution_ww.rdata',
        'eurasia_nc':roots['additional']/'my_prespa/data/in/climate_data/taxa_old_new_mediterr_eurasia.nc',
        'eurasia_nc_ohrid':roots['additional']/'my_ohrid/data/in/climate_data/taxa_old_new_mediterr_eurasia.nc',
        'eurasia_RData':roots['additional']/'my_ohrid/data/in/climate_data/taxa_old_new_mediterr_eurasia.RData',
        'regional_nc':roots['additional']/'my_ohrid/data/in/climate_data/taxa_old_new_mediterr.nc',
        'regional_RData':roots['original']/'in/taxa_old_new_mediterr.rdata',
        'regional_climate':roots['original']/'in/tann_cru4.nc'}
    for k,p in known.items():references[p]=raster['fingerprints'][k]
    references[roots['additional']/distribution['relative_path']]=distribution['sha256']
    checks['all_inspected_source_hashes_unchanged']=all(sha(p)==h for p,h in references.items())
    checkout=read('checkout_before.json'); allowed={'docs/thesis_research/PROJECT_STATUS.md','docs/thesis_research/MY_RECONST_SOURCE_RESULT.md'}
    checks['existing_checkout_files_preserved']=all(Path(p).is_file() and (p in allowed or sha(Path(p))==h) for p,h in checkout['hashes'].items())
    doc_before=base/'documentation_before'
    previous_status=(doc_before/'PROJECT_STATUS.md').read_bytes();addition=(base/'project_status_addition.md').read_text(encoding='utf-8').encode('utf-8')
    status=Path('docs/thesis_research/PROJECT_STATUS.md').read_bytes()
    previous_result=(doc_before/'MY_RECONST_SOURCE_RESULT.md').read_bytes()
    checks['previous_documentation_retained']=status.count(addition)==1 and status.replace(addition,b'',1)==previous_status and Path('docs/thesis_research/MY_RECONST_SOURCE_RESULT.md').read_bytes().startswith(previous_result) and hashlib.sha256(previous_status).hexdigest()==checkout['hashes']['docs/thesis_research/PROJECT_STATUS.md'] and hashlib.sha256(previous_result).hexdigest()==checkout['hashes']['docs/thesis_research/MY_RECONST_SOURCE_RESULT.md']
    checks['same_branch_and_head']=subprocess.check_output(['git','branch','--show-current'],text=True).strip()==checkout['branch']=='dissertation-reconstruction' and subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==checkout['head']=='7f1826cffddc8893ba6354cd3cca3fa43954b9be'
    artifacts={p.relative_to(base).as_posix():sha(p) for p in base.rglob('*') if p.is_file() and p.suffix in ['.json','.md'] and p.name!='final_verification.json'}
    return {'checks':checks,'all_checks_passed':all(checks.values()),'source_hash_count':len(references),'preserved_checkout_file_count':len(checkout['hashes'])-len(allowed),'artifacts':artifacts,'verifier_sha256':sha(Path(__file__)),'documentation_hashes':{p:sha(Path('docs/thesis_research')/p) for p in ['PROJECT_STATUS.md','MY_RECONST_SOURCE_RESULT.md','HISTORICAL_PROVENANCE_RESULT.md']}}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',required=True,type=Path);ap.add_argument('--base',required=True,type=Path)
    args=ap.parse_args();target=args.base/'final_verification.json'
    if target.exists():raise FileExistsError('Refusing to overwrite verification')
    result=verify(args.config,args.base)
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='artifacts'}))
    if not result['all_checks_passed']:raise SystemExit(1)


if __name__=='__main__':main()
