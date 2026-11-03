"""Export an explicitly selected portable handoff, never raw source data or runtimes."""
from pathlib import Path
import gzip,hashlib,json,re,sys
repo=Path(__file__).resolve().parents[1]
out=repo/'docs/thesis_research/research_snapshot_2026_11_02'
if out.exists():raise FileExistsError(out)
out.mkdir()
sha=lambda b:hashlib.sha256(b).hexdigest()
roots=json.loads((repo/'.local/my_reconst_paths.json').read_text())
replacements=[(str(repo),'<CHECKOUT>'),(roots['original_data_root'],'<ORIGINAL_DATA_ROOT>'),(roots['additional_source_root'],'<ADDITIONAL_SOURCE_ROOT>')]
def redact(value):
 if isinstance(value,str):
  for root,label in replacements:
   value=value.replace(root,label).replace(root.replace('\\','/'),label)
  value=re.sub(r'[A-Za-z]:[\\/]Users[\\/][^\\/\s"<>]+','<USER_HOME>',value)
  return value
 if isinstance(value,list):return [redact(v) for v in value]
 if isinstance(value,dict):return {redact(k):redact(v) for k,v in value.items()}
 return value
selected={
 'historical_provenance': ['.local/provenance_continuation_v1/final_verification.json','.local/provenance_continuation_v1/climate_time_bounds.json','.local/provenance_continuation_v1/eurasia_nonbinary.json','.local/provenance_continuation_v1/ohrid_279_distribution.json','.local/provenance_continuation_v1/source_evidence.json','.local/provenance_reader_followup_v1/final_verification.json','.local/provenance_reader_followup_v1/model_summary.json','.local/provenance_reader_followup_v1/training_console_trace.json','.local/my_reconst_source_audit_v3/audit.json'],
 'console': ['.local/console_provenance_v1/cardinality/console_cardinality.json','.local/console_provenance_v1/emd_plot_pages.json','.local/console_provenance_v1/source_bindings.json','.local/console_provenance_v1/independent_pdf_parser.json','.local/console_provenance_v1/stored_model_predictor_crosscheck.json','.local/console_provenance_v1/package_source_manifest.json','.local/console_provenance_v1/explicit_dependency_paths.json','.local/console_provenance_v1/final_verification.json','.local/console_provenance_v1/pytest_v2.xml'],
 'second_plot': ['.local/tf_plot_comparison_v1/second_plot_pages.json','.local/tf_plot_comparison_v1/vector_comparison.json','.local/tf_plot_comparison_v1/fixed_decimal_coordinates.json','.local/tf_plot_comparison_v1/source_variant_bindings.json','.local/tf_plot_comparison_v1/missing_dependency_followup.json','.local/tf_plot_comparison_v1/final_verification.json','.local/tf_plot_comparison_v1/pytest.xml','.local/tf_plot_comparison_v1/check_fixed_decimal_coordinates.py'],
 'paper_verification': ['results/local_reference_review_v2/verification.json','.local/corrected_full_replay.json','.local/original_full_replay.json','.local/corrected_posterior_diagnostics.json','.local/legacy_same_runtime_check.json','.local/final_integrity_check.json','.local/requirements-lock.txt','.local/diagnostic-original.toml','.local/corrected-reference.toml'],
 'handoff_tests': ['.local/github_handoff_v1/pytest_v2.xml']}
entries=[]
def export(source,destination):
 p=repo/source;original=p.read_bytes();suffix=p.suffix.lower();transformation='byte_identical'
 if suffix=='.json':
  value=redact(json.loads(original.decode('utf-8-sig')))
  content=(json.dumps(value,indent=2,ensure_ascii=True)+'\n').encode();transformation='JSON reserialized; only machine-specific path prefixes replaced'
 elif suffix in ['.xml','.toml','.py','.html']:
  content=redact(original.decode('utf-8-sig')).encode();transformation='Only machine-specific path prefixes replaced'
 else:content=original
 target=out/destination;target.parent.mkdir(parents=True,exist_ok=True)
 if len(content)>1000000 and suffix=='.json':
  target=target.with_suffix(target.suffix+'.gz');stored=gzip.compress(content,mtime=0);transformation+='; deterministic gzip'
 else:stored=content
 target.write_bytes(stored)
 entries.append({'source':source,'published':target.relative_to(out).as_posix(),'local_original_sha256':sha(original),'published_sha256':sha(stored),'decoded_sha256':sha(content),'published_bytes':len(stored),'transformation':transformation})
for group,paths in selected.items():
 for source in paths:
  name=Path(source).name
  if source.endswith('final_verification.json'):name=Path(source).parent.name+'_final_verification.json'
  export(source,group+'/'+name)
for group in ['local_original_full','local_corrected_full']:
 for name in ['rekonstruktion_alter.csv','rekonstruktion_tiefe.csv','diagnostik.csv','config.toml','provenance.json','meta.json']:
  export('results/'+group+'/'+name,'paper_results/'+group+'/'+name)
for name in ['comparison.png','review.html']:
 export('results/local_reference_review_v2/'+name,'paper_results/'+name)
manifest={'schema_version':1,'status':'portable_research_handoff_not_scientific_release','basis_commit':'7f1826cffddc8893ba6354cd3cca3fa43954b9be','original_local_evidence_remains_unchanged':True,'path_placeholders':['<CHECKOUT>','<ORIGINAL_DATA_ROOT>','<ADDITIONAL_SOURCE_ROOT>','<USER_HOME>'],'entries':entries,'not_included':['Original PDFs and RData/NetCDF sources','Full samplers and posterior/prior NPZ archives','Compiler, virtual environments and personal configuration','Failed and superseded exploratory evidence'],'historical_checks_note':'Saved verification files are historical results. Their source hashes refer to unredacted local files, not to the portable transformed copies. Re-run raw-source verifiers only with the original local inputs.'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
summary={'schema_version':1,'scientific_status':'experimental_taxon_method_incomplete','historical_models':{'distinct_byte_contents':78,'file_aliases':109,'caret_fits':167,'binary_fits':6,'multiclass_fits':161,'independent_279_taxon_functions_recovered':False},'console':{'matched_positions':271,'skipped_positions':8,'exact_historical_run_identified':False},'plots':{'pages_per_pdf':278,'missing_ohrid_position':151,'juglans_snapshot_label':'Juglans sp.','compared_presence_point_pairs':961568,'max_stored_x_difference_pdf_points':.02,'all_stored_y_coordinates_equal':True,'numeric_TF_arrays_recovered':False},'validation':{'handoff_targeted_tests_passed':67,'prior_console_checks':34,'second_plot_checks':31},'gates':{'monticchio_P1':'open','fossil_pilot_release':False,'smote_raw_probabilities_validated_for_fossils':False},'app_next_step':'Read-only research status/source view, reusing existing baseline and seven-species artifacts; distinguish available, experimental and missing results.'}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'exported_files':len(entries),'total_bytes':sum(e['published_bytes'] for e in entries),'largest':max(e['published_bytes'] for e in entries)},indent=2))