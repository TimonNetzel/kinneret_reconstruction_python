"""Verify the console/TF-plot evidence and preservation without re-running fits."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from audit_my_reconst_sources import metadata,sha


def verify(config,base):
    repo=Path(__file__).resolve().parents[1];base=base.resolve(strict=True)
    if repo not in base.parents:raise ValueError('Evidence must be in existing checkout')
    read=lambda name:json.loads((base/name).read_text(encoding='utf-8'))
    cfg=json.loads(config.read_text(encoding='utf-8-sig'));roots={'original':Path(cfg['original_data_root']),'additional':Path(cfg['additional_source_root'])}
    if any(base==p.resolve() or p.resolve() in base.parents for p in roots.values()):raise ValueError('Output overlaps originals')
    before=read('checkout_before.json');audit=read('cardinality/console_cardinality.json');rows=audit['rows'];plots=read('emd_plot_pages.json')
    independent=read('independent_pdf_parser.json');visual=read('emd_visual_pages.json');pdfs=read('pdf_inspection.json');bindings=read('source_bindings.json');packages=read('package_source_manifest.json')
    git=lambda *args:subprocess.check_output(['git','--no-optional-locks',*args],cwd=repo,text=True).strip()
    status=git('status','--short','--branch');old_status=(base/'before/PROJECT_STATUS.md').read_bytes();cut=old_status.index(b'\n')+1
    old_metadata=read('source_metadata_before.json');logged=[r for r in rows if r['logged_rows'] is not None]
    checks={
     'same_branch_and_head':git('branch','--show-current')==before['branch']=='dissertation-reconstruction' and git('rev-parse','HEAD')==before['head']=='7f1826cffddc8893ba6354cd3cca3fa43954b9be',
     'prior_git_status_entries_retained':set(before['git_status'].strip().splitlines()[1:])<=set(status.splitlines()[1:]),
     'all_other_710_existing_checkout_files_byte_identical':all((repo/name).is_file() and (name=='docs/thesis_research/PROJECT_STATUS.md' or sha(repo/name)==h) for name,h in before['hashes'].items()) and len(before['hashes'])-1==710,
     'all_258_prior_evidence_files_byte_identical':len(before['prior_evidence_hashes'])==258 and all(sha(repo/name)==h for name,h in before['prior_evidence_hashes'].items()),
     'all_previous_project_status_bytes_retained':(repo/'docs/thesis_research/PROJECT_STATUS.md').read_bytes()==old_status[:cut]+(base/'project_status_addition.md').read_bytes()+old_status[cut:] and sha(base/'before/PROJECT_STATUS.md')==before['hashes']['docs/thesis_research/PROJECT_STATUS.md'],
     'both_original_trees_metadata_unchanged':all({k:list(v) for k,v in metadata(root).items()}==old_metadata[label] for label,root in roots.items()),
     'all_279_snapshot_positions_read':audit['dimensions']==[720,360,279] and [r['printed_position'] for r in rows]==list(range(1,280)),
     'all_271_console_cases_and_1084_model_shapes':len(logged)==audit['positions_with_model_output']==271 and audit['all_model_shapes_count']==sum(len(r['model_labels']) for r in rows)==1084 and all(r['model_labels']==['nnet','mda','qda','gbm'] and r['logged_columns']==5 for r in logged),
     'all_northern_raster_planes_have_54025_finite_cells':audit['northern_finite_cell_counts']==[54025] and all(r['north']['presence']+r['north']['absence']==54025 for r in rows),
     'all_logged_cases_enter_actual_source_smote_branch':audit['all_logged_cases_enter_source_smote_branch'] and all(2*r['north']['train_presence']<r['north']['train_absence'] and r['north']['train_presence']>1 for r in logged),
     '271_cardinalities_independently_recalculated':all(r['logged_rows']==math.ceil(r['north']['absence']*.7)+math.ceil(r['north']['presence']*.7)*((math.ceil(r['north']['absence']*.7)-math.ceil(r['north']['presence']*.7))//math.ceil(r['north']['presence']*.7)) for r in logged),
     'eight_skips_exact_and_low_presence_only':audit['skipped_console_positions']==audit['predicted_low_presence_positions']==[9,16,142,149,151,162,167,206] and all((r['logged_rows'] is None)==(r['north']['presence']<=10) for r in rows),
     'negative_controls_recorded_without_selection_tuning':audit['control_match_counts']=={'north_synthetic_only':271,'world_synthetic_only':0,'north_original_minority_included':0,'north_no_SMOTE':7,'north_no_holdout_synthetic_only':0},
     'cardinality_collisions_exposed':audit['distinct_logged_row_counts']==len({r['logged_rows'] for r in logged})==251 and bool(audit['row_count_collision_groups']),
     'all_278_pdf_pages_have_unique_title_and_point_count_match':plots['page_count']==len(plots['pages'])==plots['pages_with_matching_title_and_presence_count']==278 and plots['all_titles_known'] and plots['titles_multiset_equals_snapshot_excluding_single_presence'] and plots['ambiguous_pages']==[] and all(len(p['same_title_and_presence_count_positions'])==1 for p in plots['pages']),
     'matched_pdf_positions_cover_all_except_151':{p['same_title_and_presence_count_positions'][0] for p in plots['pages']}==set(range(1,280))-{151},
     'pdf_counts_and_titles_match_actual_snapshot_rows':all(p['title']==rows[p['same_title_and_presence_count_positions'][0]-1]['snapshot_name_not_historical_taxon_assignment'] and p['green_point_glyph_count']==rows[p['same_title_and_presence_count_positions'][0]-1]['north']['presence'] for p in plots['pages']),
     'single_presence_picea_exclusion_explicit':rows[150]['snapshot_name_not_historical_taxon_assignment']=='Picea omorika' and rows[150]['north']['presence']==1,
     'juglans_sp_plot_before_separate_rename':plots['pages'][11]['title']=='Juglans sp.' and plots['pages'][11]['green_point_glyph_count']==2926 and plots['pages'][11]['same_title_and_presence_count_positions']==[110],
     'independent_pdf_parser_matches_fill_stroke_title_and_color':independent['all_checks_passed'] and all(independent['checks'].values()) and all(r['filled_glyph_count']==r['stroked_glyph_count']==p['green_point_glyph_count'] for r,p in zip(independent['pages'],plots['pages'])),
     'representative_complete_pages_visually_checked_with_working_renderer':visual['pages']==[1,76,278] and visual['all_three_second_renderer_previews_visually_inspected'] and visual['point_glyphs_visible_with_second_renderer'],
     'six_original_importance_figures_keep_five_labels_and_native_metadata':len(pdfs['results'])==6 and pdfs['all_six_pages_visually_inspected'] and all(r['pages']==1 and r['contains_all_five_variable_labels'] and r['metadata']['/Producer']=='R 3.5.0' and r['metadata']['/CreationDate'].startswith('D:20210609') for r in pdfs['results']),
     'package_source_files_match_recorded_hashes':len(packages)==4 and all('error' not in r and sha(base/'package_sources'/r['local_file'])==r['sha256'] for r in packages),
     'archived_package_sources_support_only_the_formula': 'Version: 1.3.1' in (base/'package_sources/DESCRIPTION').read_text(encoding='utf-8') and 'syn_data = syn_dat' in (base/'package_sources/SMOTE.R').read_text(encoding='utf-8') and 'sizeM=floor((2*size_N-size_input)/size_P)' in (base/'package_sources/n_dup_max.R').read_text(encoding='utf-8') and 'ceiling(nrow(dat) * p)' in (base/'package_sources/caret_6.0-84_createDataPartition.R').read_text(encoding='utf-8'),
    }
    sources={r['relative_path']:(roots[r['source_tree']]/r['relative_path']).read_text(encoding='utf-8-sig') for r in bindings}
    five=sources['src/all_transferfunctions_v1.R'];two=sources['src/all_trafo.R'];main=sources['main_v1.2.R']
    checks['five_variable_fit_saves_commented_and_importance_saves_active']=all(any(line.lstrip().startswith('#') and 'save('+model in line for line in five.splitlines()) for model in ['nnet.fit.smote','mda.fit.smote','qda.fit.smote','gbm.fit.smote']) and all(any(line.strip().startswith('save('+name) for line in five.splitlines()) for name in ['var.imp.logloss','var.imp.rmse','var.imp.acu'])
    checks['current_source_titanic_override_retained_as_limit']=any(line.strip()=='trafo.input <- titanic_imputed' for line in five.splitlines())
    checks['two_variable_writer_and_main_consumer_bind_same_fields_and_path']=all(f'transfer.functions[[i]]${key}' in two and f'transfer.functions[[i]]${key}' in main for key in ['prod.x1','prod.x2']) and 'save(transfer.functions, file = "out/transfer_functions/mixed_partial_arts_emd.rdata")' in two and 'load("out/transfer_functions/mixed_partial_arts_emd.rdata")' in main and 'if(length(taxa.precip[[1]]) == 1) next()' in two
    models=read('stored_model_predictor_crosscheck.json')
    checks['stored_167_fit_predictors_reused_without_new_scan']=models['fit_count']==167 and models['no_five_predictor_fit_in_prior_selected_167'] and models['source_summary_sha256']==sha(repo/'.local/provenance_reader_followup_v1/model_summary.json')
    paths=read('explicit_dependency_paths.json');checks['four_explicit_dependency_path_checks_remain_negative']=len(paths)==4 and all(not r['exists'] and not Path(r['path']).exists() for r in paths)
    suite=ET.parse(base/'pytest_v2.xml').getroot();cases=suite.findall('.//testcase')
    checks['59_targeted_tests_passed']=len(cases)==59 and not suite.findall('.//failure') and not suite.findall('.//error') and not suite.findall('.//skipped')
    checks['no_model_prediction_or_numeric_TF_recovery_claim']=audit['R_prediction_validated'] is False and audit['no_training_or_sampler'] and audit['no_historical_taxon_or_seed_or_climate_value_assignment'] and plots['no_probability_values_or_original_TF_arrays_recovered']
    refs={roots['original']/'logfile_trafo.txt':audit['source_hashes']['console'],roots['additional']/'my_ohrid/data/in/climate_data/taxa_distribution.rdata':audit['source_hashes']['distribution'],Path(plots['pdf']):plots['sha256']}
    for r in bindings+pdfs['results']:refs[roots[r['source_tree']]/r['relative_path']]=r['sha256']
    checks['all_inspected_original_hashes_unchanged']=audit['source_hashes_unchanged'] and plots['source_hash_unchanged'] and all(sha(p)==h for p,h in refs.items())
    checks['audit_tool_hashes_still_current']=audit['tool_sha256']==sha(repo/'tools/audit_console_cardinality.py') and plots['tool_sha256']==sha(repo/'tools/audit_tf_plot_pages.py')
    checks['report_preserves_limits_and_next_entry']=all(s in (repo/'docs/thesis_research/HISTORICAL_CONSOLE_TF_LINK_RESULT.md').read_text(encoding='utf-8') for s in ['271','278','Als Nächstes','59 Tests','keine neue fachliche Freigabe','Jung links, alt rechts'])
    artifacts={p.relative_to(base).as_posix():sha(p) for p in base.rglob('*') if p.is_file() and p.suffix.lower() in ('.json','.r','.py','.md','.xml','.png','.txt') and not any(part.startswith('pytest') or part=='pdf_runtime' for part in p.relative_to(base).parts[:-1]) and p.name!='final_verification.json'}
    return {'checks':checks,'all_checks_passed':all(checks.values()),'check_count':len(checks),'source_hash_count':len(refs),'preserved_existing_checkout_files':len(before['hashes'])-1,'preserved_prior_evidence_files':len(before['prior_evidence_hashes']),'artifacts':artifacts,'git_status':status,'verifier_sha256':sha(Path(__file__)),'report_sha256':sha(repo/'docs/thesis_research/HISTORICAL_CONSOLE_TF_LINK_RESULT.md'),'project_status_sha256':sha(repo/'docs/thesis_research/PROJECT_STATUS.md')}


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',required=True,type=Path);ap.add_argument('--base',required=True,type=Path);args=ap.parse_args()
    target=args.base/'final_verification.json'
    if target.exists():raise FileExistsError('Refusing to overwrite final verification')
    result=verify(args.config,args.base);print(json.dumps({k:v for k,v in result.items() if k not in ('artifacts','git_status')},indent=2))
    if not result['all_checks_passed']:raise SystemExit(1)
    target.write_text(json.dumps(result,ensure_ascii=True,indent=2)+'\n',encoding='utf-8')
