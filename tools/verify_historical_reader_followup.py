"""Independent preservation and consistency checks for the targeted reader follow-up.

Rehashes source aliases and preserved files. Never runs R, fits, or samplers.
"""
from __future__ import annotations
import argparse
from collections import Counter
import io
import json
from pathlib import Path
import re
import subprocess
from audit_my_reconst_sources import sha, metadata
from local_rdata_metadata import MetadataXDR


def verify(config, base):
    repo = Path(__file__).resolve().parents[1]
    base = base.resolve(strict=True)
    if repo not in base.parents:
        raise ValueError('Evidence must remain in the existing checkout')
    cfg=json.loads(config.read_text(encoding='utf-8-sig'))
    roots={'original':Path(cfg['original_data_root']), 'additional':Path(cfg['additional_source_root'])}
    if any(base==p.resolve() or p.resolve() in base.parents for p in roots.values()):
        raise ValueError('Evidence overlaps an original source tree')
    read=lambda name:json.loads((base/name).read_text(encoding='utf-8'))
    before=read('checkout_before.json')
    current=read('model_summary.json')
    old_path=repo/'.local/provenance_continuation_v1/models_v2/models.json'
    prior=json.loads(old_path.read_text(encoding='utf-8'))
    search=read('targeted_source_search.json'); copies=read('regional_container_copies.json')
    trace=read('training_console_trace.json'); candidate=read('training_console_candidate_source.json')
    tests=read('pytest_results.json'); regression=read('regression_demonstration.json')
    results=current['results']; by_hash={r['sha256']:r for r in results}
    unresolved={r['sha256'] for r in prior['results'] if r['parse_status']!='complete_structure_not_R_validation'}
    retried=[r for r in results if r['sha256'] in unresolved]
    git=lambda *args:subprocess.check_output(['git','--no-optional-locks',*args],cwd=repo,text=True).strip()
    status=git('status','--short','--branch')
    allowed={'tools/local_rdata_metadata.py','tools/audit_historical_provenance.py',
             'docs/thesis_research/PROJECT_STATUS.md','docs/thesis_research/HISTORICAL_PROVENANCE_RESULT.md'}
    checks={
      'same_branch_and_head':git('branch','--show-current')==before['branch']=='dissertation-reconstruction' and git('rev-parse','HEAD')==before['head']=='7f1826cffddc8893ba6354cd3cca3fa43954b9be',
      'all_prior_git_status_entries_retained':set(before['git_status'].strip().splitlines()[1:])<=set(status.splitlines()[1:]),
      'all_other_existing_checkout_files_byte_identical':all((repo/name).is_file() and (name in allowed or sha(repo/name)==h) for name,h in before['hashes'].items()),
      'modified_file_original_bytes_preserved':all(sha(base/'before'/name)==before['hashes'][name] for name in allowed),
      'all_prior_provenance_evidence_byte_identical':all((repo/name).is_file() and sha(repo/name)==h for name,h in before['prior_evidence_hashes'].items()),
    }
    previous_status=(base/'before/docs/thesis_research/PROJECT_STATUS.md').read_bytes()
    cut=previous_status.index(b'\n')+1
    checks['all_previous_status_bytes_retained']=(repo/'docs/thesis_research/PROJECT_STATUS.md').read_bytes()==previous_status[:cut]+(base/'project_status_addition.md').read_bytes()+previous_status[cut:]
    checks['all_previous_historical_report_bytes_retained']=(repo/'docs/thesis_research/HISTORICAL_PROVENANCE_RESULT.md').read_bytes()==(base/'before/docs/thesis_research/HISTORICAL_PROVENANCE_RESULT.md').read_bytes()+(base/'historical_report_addition.md').read_bytes()
    old_metadata=json.loads((base/'source_metadata_before.json').read_text(encoding='utf-8'))
    checks['both_original_trees_metadata_unchanged']=all({k:list(v) for k,v in metadata(root).items()}==old_metadata[label] for label,root in roots.items())
    checks['model_scope_same_109_files_78_byte_contents']=current['candidate_files']==109 and len(results)==len(by_hash)==current['unique_byte_contents']==78 and {r['sha256']:r['relative_paths'] for r in results}=={r['sha256']:r['relative_paths'] for r in prior['results']}
    checks['only_13_unresolved_contents_retried']=len(unresolved)==current['retried_unique_contents']==len(retried)==13 and current['retried_file_aliases']==sum(len(r['relative_paths']) for r in retried)==17
    checks['65_completed_records_reused_without_change']=sum(bool(r.get('carried_forward_without_reparse')) for r in results)==65 and all(all(by_hash[r['sha256']][k]==v for k,v in r.items()) for r in prior['results'] if r['sha256'] not in unresolved)
    checks['all_78_structures_read']=dict(Counter(r['parse_status'] for r in results))==current['parse_status_counts']=={'complete_structure_not_R_validation':78}
    checks['aggregate_fit_counts_match_records']=all(current[key]==sum(r.get(field,0) for r in results) for key,field in [('complete_caret_fits','caret_fit_count'),('complete_binary_caret_fits','binary_caret_fit_count'),('complete_multiclass_caret_fits','multiclass_caret_fit_count')]) and (current['complete_caret_fits'],current['complete_binary_caret_fits'],current['complete_multiclass_caret_fits'])==(167,6,161)
    checks['new_90_fits_all_multiclass']=sum(r['caret_fit_count'] for r in retried)==current['new_caret_fits']==90 and all(r['binary_caret_fit_count']==0 and r['multiclass_caret_fit_count']==r['caret_fit_count'] for r in retried)
    checks['new_predictors_explicit_winter_and_annual_precip']=all(f['coefnames']==['t.djf.norm','p.ann.norm'] for r in retried for f in r['fits'])
    checks['no_R_prediction_or_279_function_claim']=all(r.get('R_prediction_validated') is False for r in results) and current['R_predictions_validated'] is False and current['not_a_full_search_of_all_RData'] and current['no_execution_no_training_no_prediction']
    checks['every_individual_evidence_hash_matches']=all(sha(repo/r['evidence_file'])==r['evidence_sha256'] for r in results)
    checks['every_individual_evidence_record_matches']=all(all(r[k]==v for k,v in json.loads((repo/r['evidence_file']).read_text(encoding='utf-8')).items()) for r in results)
    checks['all_retried_alias_checks_passed']=all(r['original_alias_hashes_unchanged'] for r in retried)
    mix=next(r for r in results if 'my_birkat_ram/data/out/ml_competition/all_models_smoted_2D_7.rdata' in r['relative_paths'])
    checks['gbm_extra_class_retained_without_harmonization']=len(mix['fits'])==9 and all(f['levels']==[f'taxa{i}' for i in range(1,9 if f['method']==['gbm'] else 8)]+['unknown'] for f in mix['fits'])
    big=next(r for r in retried if 'my_birkat_ram/data/out/ml_competition/all_models_smoted_2D_neumann.rdata' in r['relative_paths'])
    parser=MetadataXDR(io.BytesIO())
    checks['large_file_read_with_original_retained_budgets']=big['parsed_objects']==3816068 and big['reused_character_objects']==2762757 and big['parsed_objects']-big['reused_character_objects']<=parser.max_objects==2_000_000 and big['character_cache_entries']<=parser.max_cached_characters==65536 and big['retained_numeric_bytes']<=32_000_000 and parser.max_parsed_objects==10_000_000
    checks['largest_retry_used_current_reader_and_worker']=big['observed_reader_sha256']==sha(repo/'tools/local_rdata_metadata.py') and big['observed_worker_sha256']==sha(repo/'tools/audit_historical_provenance.py')
    checks['36_tests_passed_against_current_files']=tests['exit_code']==tests['failed']==0 and tests['passed']==36 and tests['recorded_from_observed_tool_output'] and all(sha(repo/name)==h for name,h in tests['file_sha256'].items())
    checks['old_failure_and_new_stream_alignment_demonstrated']=regression['old_reader_error']=='ValueError: 53 is not a valid RObjectType' and regression['final_nested_code']==[34] and regression['final_following_value']==[99] and regression['final_reader_sha256']==sha(repo/'tools/local_rdata_metadata.py')
    checks['targeted_source_search_385_files_no_explicit_regional_writer']=search['files_read']==len(search['sources'])==385 and not any('regional_write' in h['categories'] for h in search['hits'])
    regional_hash=next(r['sha256'] for r in copies['files'] if r['source_tree']=='original')
    checks['six_regional_copies_identical_and_three_older_copies_separate']=copies['file_count']==9 and copies['unique_byte_contents']==2 and Counter(r['sha256'] for r in copies['files'])[regional_hash]==6 and len({r['sha256'] for r in copies['files']})==2 and {tuple(v['dimensions']) for r in copies['structures'] for o in r['objects'] for v in o['fields'].values() if 'dimensions' in v}=={(149,97,275),(40,40,15)}
    log_lines=(roots['original']/trace['relative_path']).read_text(encoding='utf-8').splitlines()
    positions=[int(m.group(1)) for s in log_lines if (m:=re.fullmatch(r'\[1\] (\d+)',s))]
    labels=Counter(m.group(1) for s in log_lines if (m:=re.search(r'model label\s*:\s*(\S+)',s)))
    checks['console_trace_positions_and_four_model_counts_independent']=len(log_lines)==trace['line_count']==22262 and positions==list(range(1,280)) and dict(labels)==trace['model_label_counts']==dict.fromkeys(['nnet','mda','qda','gbm'],271)
    checks['console_eight_positions_without_model_output_retained']=trace['positions_without_model_output']==[9,16,142,149,151,162,167,206] and [b['printed_position'] for b in trace['blocks'] if not b['model_labels']]==trace['positions_without_model_output']
    checks['console_has_no_explicit_seed_or_input_manifest']=trace['session_seed_input_manifest_lines']==[] and not any(re.search(r'sessionInfo|set.seed|RNGkind|\.Random.seed|\.nc|\.rdata',s,re.I) for s in log_lines) and trace['no_taxon_assignment_from_printed_position'] and trace['no_serialized_model_binding']
    candidate_text=(roots['original']/candidate['relative_path']).read_text(encoding='utf-8-sig')
    checks['candidate_source_input_override_exposed']=any(line.strip()=='trafo.input <- titanic_imputed' for line in candidate_text.splitlines()) and candidate['candidate_only_later_titanic_input_override_prevents_exact_run_binding']
    refs={}
    for r in results:
        for name in r['relative_paths']: refs[roots['additional']/name]=r['sha256']
    for r in search['sources']+copies['files']+[trace,candidate]: refs[roots[r['source_tree']]/r['relative_path']]=r['sha256']
    checks['all_inspected_original_source_hashes_unchanged']=all(sha(path)==h for path,h in refs.items())
    checks['all_source_search_preservation_checks_passed']=read('source_search_verification.json')['all_checks_passed']
    checks['current_report_and_next_entry_present']=all(text in (repo/'docs/thesis_research/HISTORICAL_MODEL_READER_RESULT.md').read_text(encoding='utf-8') for text in ['167','95.006','titanic_imputed','Als Nächstes','Jung links, alt rechts'])
    artifacts={p.relative_to(base).as_posix():sha(p) for p in base.rglob('*') if p.is_file() and p.suffix in ('.json','.md','.py') and p.name!='final_verification.json'}
    return {'checks':checks,'all_checks_passed':all(checks.values()),'check_count':len(checks),'source_hash_count':len(refs),
      'preserved_existing_checkout_files':len(before['hashes'])-len(allowed),'preserved_prior_evidence_files':len(before['prior_evidence_hashes']),
      'artifacts':artifacts,'git_status':status,'verifier_sha256':sha(Path(__file__)),
      'documentation_hashes':{name:sha(repo/'docs/thesis_research'/name) for name in ['PROJECT_STATUS.md','HISTORICAL_PROVENANCE_RESULT.md','HISTORICAL_MODEL_READER_RESULT.md']}}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',required=True,type=Path);ap.add_argument('--base',required=True,type=Path)
    args=ap.parse_args();target=args.base/'final_verification.json'
    if target.exists():raise FileExistsError('Refusing to overwrite verification')
    result=verify(args.config,args.base)
    print(json.dumps({k:v for k,v in result.items() if k not in ('artifacts','git_status')},ensure_ascii=False,indent=2))
    if not result['all_checks_passed']:raise SystemExit(1)
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
