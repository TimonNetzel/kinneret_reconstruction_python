"""Verify the second TF plot audit and preserved local work, without R/model runs."""
from collections import Counter
import hashlib,json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def verify(repo,base):
    read=lambda p:json.loads((base/p).read_text(encoding='utf-8'))
    git=lambda *args:subprocess.check_output(['git','--no-optional-locks',*args],cwd=repo,text=True).strip()
    initial=read('checkout_before.json');protected={**initial['hashes'],**read('preservation_supplement.json')}
    raw=read('second_plot_pages.json');comparison=read('vector_comparison.json');decimal=read('fixed_decimal_coordinates.json')
    bindings=read('source_variant_bindings.json');missing=read('missing_dependency_followup.json');pr=read('github_pr30.json')
    priorbase=repo/'.local/console_provenance_v1'
    old=json.loads((priorbase/'emd_plot_pages.json').read_text());card=json.loads((priorbase/'cardinality/console_cardinality.json').read_text())
    roots={k:Path(v) for k,v in json.loads((repo/'.local/my_reconst_paths.json').read_text()).items() if k in ['original_data_root','additional_source_root']}
    roots={'original':roots['original_data_root'],'additional':roots['additional_source_root']}
    before_status=(base/'PROJECT_STATUS.before.md').read_bytes();cut=before_status.index(b'\n')+1
    current_status=git('status','--short','--branch')
    pages=comparison['pages'];pairs=comparison['pairs'];positions={p['same_title_and_presence_count_positions'][0] for p in raw['pages'] if len(p['same_title_and_presence_count_positions'])==1}
    stats_before=json.loads((priorbase/'source_metadata_before.json').read_text())
    metadata_failures=[];metadata_count=0
    # Revisit paths from the existing evidence list, never enumerate source trees.
    for label,records in stats_before.items():
        for rel,record in records.items():
            p=roots[label]/rel
            try:
                stat=p.stat();actual=[stat.st_size,stat.st_mtime_ns]
            except OSError:actual=None
            metadata_count+=1
            if actual!=record:metadata_failures.append({'tree':label,'path':rel,'expected':record,'actual':actual})
    unchanged={p:h for p,h in protected.items() if p!='docs/thesis_research/PROJECT_STATUS.md'}
    refs={Path(raw['pdf']):raw['sha256'],Path(old['pdf']):old['sha256']}
    refs.update({roots[r['source_tree']]/r['relative_path']:r['sha256'] for r in bindings['sources']})
    suite=ET.parse(base/'pytest.xml').getroot();cases=suite.findall('.//testcase')
    visual=read('visual_checks.json');report=(repo/'docs/thesis_research/HISTORICAL_SECOND_TF_PLOT_RESULT.md').read_text(encoding='utf-8')
    aggregate=Counter()
    for p in decimal['pages']:
        for d in p['displacement_counts']:aggregate[(d['dx_hundredth_pdf_point'],d['dy_hundredth_pdf_point'])]+=d['count']
    expected_counts={'same_point_order':23,'same_point_multiset':23,'same_axes':269,'same_contour_vectors':0,'same_contour_labels':8}
    checks={
        'same_existing_branch_and_head':git('branch','--show-current')==initial['branch']=='dissertation-reconstruction' and git('rev-parse','HEAD')==initial['head']=='7f1826cffddc8893ba6354cd3cca3fa43954b9be',
        'github_plugin_pr30_merge_matches_local_basis':pr['merged'] and pr['merge_commit_sha']==initial['head'] and pr['number']==30,
        'all_prior_git_status_entries_retained':set(initial['git_status'].splitlines()[1:])<=set(current_status.splitlines()[1:]),
        'all_existing_checkout_files_other_than_status_byte_identical':all((repo/p).is_file() and sha(repo/p)==h for p,h in unchanged.items()),
        'all_previous_status_bytes_retained_with_only_recorded_insertion':(repo/'docs/thesis_research/PROJECT_STATUS.md').read_bytes()==before_status[:cut]+(base/'project_status_addition.md').read_bytes()+before_status[cut:] and sha(base/'PROJECT_STATUS.before.md')==protected['docs/thesis_research/PROJECT_STATUS.md'],
        'all_prior_selected_evidence_byte_identical':all(sha(repo/p)==h for p,h in initial['prior_evidence_hashes'].items()),
        'all_catalogued_original_metadata_unchanged_without_new_inventory':not metadata_failures,
        'all_ten_inspected_original_file_hashes_unchanged':len(refs)==10 and all(sha(p)==h for p,h in refs.items()),
        'second_pdf_278_unique_title_and_count_bindings':raw['page_count']==raw['pages_with_matching_title_and_presence_count']==len(raw['pages'])==278 and raw['ambiguous_pages']==[] and len(positions)==278,
        'second_pdf_positions_exactly_all_except_151':positions==set(range(1,280))-{151},
        'actual_snapshot_names_and_counts_match_all_second_pages':all(len(p['same_title_and_presence_count_positions'])==1 and p['title']==card['rows'][p['same_title_and_presence_count_positions'][0]-1]['snapshot_name_not_historical_taxon_assignment'] and p['green_point_glyph_count']==card['rows'][p['same_title_and_presence_count_positions'][0]-1]['north']['presence'] for p in raw['pages']),
        'juglans_sp_not_renamed_or_reassigned':raw['pages'][11]['title']=='Juglans sp.' and raw['pages'][11]['same_title_and_presence_count_positions']==[110] and raw['pages'][11]['green_point_glyph_count']==2926,
        'independent_556_page_title_fill_stroke_color_checks_pass':comparison['independent_title_count_checks_pass'] and comparison['fill_stroke_checks_pass'] and comparison['point_color_checks_pass'] and sum(map(len,pages.values()))==556,
        'all_278_pairs_bound_by_actual_evidence':len(pairs)==278 and all(p['title']==pages['second'][p['second_page']-1]['title']==pages['emd'][p['emd_page']-1]['title'] and pages['second'][p['second_page']-1]['snapshot_positions']==pages['emd'][p['emd_page']-1]['snapshot_positions']==[p['position']] for p in pairs),
        'exact_vector_comparison_counts_recalculated':comparison['counts']==expected_counts=={k:sum(p[k] for p in pairs) for k in expected_counts},
        'all_axes_carry_actual_temperature_and_precipitation_ticks':all(r['axes']['temperature']['labels']==[-30.,-20.,-10.,0.,10.,20.,30.] and r['axes']['precipitation']['labels']==[0.,2000.,4000.,6000.,8000.,10000.] for rows in pages.values() for r in rows),
        'axis_calibration_rounding_residual_bounded':all(r['axes']['temperature']['max_tick_rounding_residual']<.002 and r['axes']['precipitation']['max_tick_rounding_residual']<.3 for rows in pages.values() for r in rows),
        'one_emd_axis_signature_and_two_second_signatures':comparison['unique_axis_signatures']=={'emd':1,'second':2} and all(len({r['axis_sha256'] for r in rows})==comparison['unique_axis_signatures'][k] for k,rows in pages.items()),
        '135_pairs_have_different_contour_path_count':sum(pages['second'][p['second_page']-1]['contour_path_count']!=pages['emd'][p['emd_page']-1]['contour_path_count'] for p in pairs)==135,
        'independent_decimal_counts_match_all_pages_and_total':decimal['all_556_page_point_counts_match_independent_trace'] and len(decimal['pages'])==278 and all(p['independent_counts_match'] and p['point_count']==raw['pages'][p['second_page']-1]['green_point_glyph_count'] for p in decimal['pages']) and decimal['point_pairs']==sum(p['point_count'] for p in decimal['pages'])==961568,
        'all_decimal_y_coordinates_equal_and_x_bounded_by_two_hundredths':all(y==0 and abs(x)<=2 for x,y in aggregate) and decimal['max_abs_dx_pdf_points']==.02 and decimal['max_abs_dy_pdf_points']==0,
        'independent_decimal_displacement_counts_recalculated':aggregate==Counter({(-1,0):16515,(0,0):922453,(1,0):22388,(2,0):212}),
        'eight_targeted_sources_and_only_one_explicit_emd_plot_writer':len(bindings['sources'])==8 and bindings['only_all_trafos_writer_in_selected_variants']==[{'source_tree':'original','relative_path':'src/all_trafo.R','lines':[493]}],
        'all_saved_source_spans_match_actual_lines':all('\n'.join((roots[r['source_tree']]/r['relative_path']).read_text(encoding='utf-8-sig').splitlines()[s['first']-1:s['last']])==s['text'] for r in bindings['sources'] for s in r['spans']),
        'missing_dependencies_remain_limited_prior_negative_results':all(not v for v in missing['original_catalog_matches'].values()) and all(missing['additional_prior_exact_results'][k]==[] for k in ['mixed_partial_arts_emd.rdata','mixed_partial_arts_emd_info.rdata','tann_test.nc','pann_test.nc']) and all(not r['exists_now'] and not Path(r['path']).exists() for r in missing['explicit_paths']),
        '31_targeted_tests_passed_including_eight_new':len(cases)==31 and len([c for c in cases if 'test_tf_plot_comparison' in c.attrib.get('classname','')])==8 and not any(suite.findall('.//'+k) for k in ['failure','error','skipped']),
        'six_complete_pages_visually_inspected_with_visible_points':visual['all_six_complete_pages_inspected'] and visual['green_points_visible'] and len(visual['images'])==6 and all(sha(base/n)==h for n,h in visual['images'].items()),
        'old_console_final_result_and_all_34_checks_retained':(lambda r:r['all_checks_passed'] and r['check_count']==34 and all(r['checks'].values()))(json.loads((priorbase/'final_verification.json').read_text())),
        'no_numeric_tf_recovery_claim':raw['no_probability_values_or_original_TF_arrays_recovered'] and comparison['no_numeric_transfer_functions_recovered'] and decimal['no_historical_climate_values_or_TF_arrays_recovered'],
        'reproducible_audit_scripts_hashes_match':comparison['tool_sha256']==sha(repo/'tools/audit_tf_plot_comparison.py') and raw['tool_sha256']==sha(repo/'tools/audit_tf_plot_pages.py') and decimal['tool_sha256']==sha(base/'check_fixed_decimal_coordinates.py'),
        'report_retains_scientific_limits_and_next_entry':all(s in report for s in ['961.568','0,02','31 gezielte Tests','keine neue fachliche Freigabe','Als Nächstes','95.006','kein Taxonindex','Jung links, alt rechts'])}
    artifacts={p.relative_to(base).as_posix():sha(p) for p in base.rglob('*') if p.is_file() and p.suffix in ['.json','.md','.py','.xml','.png'] and 'pytest' not in p.relative_to(base).parts[:-1] and p.name!='final_verification.json'}
    return {'checks':checks,'all_checks_passed':all(checks.values()),'check_count':len(checks),'preserved_existing_checkout_files_except_status':len(unchanged),'preserved_prior_evidence_files':len(initial['prior_evidence_hashes']),'catalogued_original_metadata_paths_checked':metadata_count,'metadata_failures':metadata_failures,'selected_original_hashes_checked':len(refs),'artifacts':artifacts,'verifier_sha256':sha(Path(__file__)),'report_sha256':sha(repo/'docs/thesis_research/HISTORICAL_SECOND_TF_PLOT_RESULT.md'),'project_status_sha256':sha(repo/'docs/thesis_research/PROJECT_STATUS.md'),'git_status':current_status}


if __name__=='__main__':
    repo=Path(__file__).resolve().parents[1];base=repo/'.local/tf_plot_comparison_v1';target=base/'final_verification.json'
    if target.exists():raise FileExistsError(target)
    result=verify(repo,base)
    print(json.dumps({k:v for k,v in result.items() if k not in ['artifacts','git_status']},indent=2))
    if not result['all_checks_passed']:raise SystemExit(1)
    target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')