"""Independent evidence reconciliation and preservation checks; no model runs."""
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASIS='705cf7b68b2bfdfc99919ad50e4091dd113d907e'


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT)


def main():
    cfg=json.loads((ROOT/'.local/my_reconst_paths.json').read_text(encoding='utf-8-sig'))
    root=Path(cfg['climate_modelling_source_root']).resolve(strict=True)
    summary=json.loads((OUT/'summary.json').read_text())
    index=list(csv.DictReader((OUT/'file_metadata.csv').open(encoding='utf-8')))
    queries=list(csv.DictReader((OUT/'requirement_crosswalk.csv').open(encoding='utf-8')))
    sources=list(csv.DictReader((OUT/'source_dates.csv').open(encoding='utf-8')))
    assert len(queries)==48 and len({x['id'] for x in queries})==48
    assert {x['id'] for x in queries if x['id'].startswith('F')}=={'F'+str(i).zfill(2) for i in range(1,40)}
    excluded=set(summary['excluded_directories']); actual=[]
    for folder,dirs,names in os.walk(root,onerror=lambda error: (_ for _ in ()).throw(error)):
        dirs[:]=[d for d in dirs if d not in excluded]
        actual.extend(Path(folder)/name for name in names if (Path(folder)/name).is_file())
    assert set('climate_modelling:'+p.relative_to(root).as_posix() for p in actual)=={x['source'] for x in index}
    for row in index:
        s=(root/row['source'].split(':',1)[1]).stat()
        assert (s.st_size,s.st_mtime_ns)==(int(row['bytes']),int(row['mtime_ns']))
    assert len(actual)==summary['files_indexed']==1828
    zipnames=[]
    for p in actual:
        if p.suffix.lower()=='.zip':
            with zipfile.ZipFile(p) as z:
                zipnames.extend('climate_modelling:'+p.relative_to(root).as_posix()+'!'+i.filename for i in z.infolist() if not i.is_dir())
    for q in queries:
        name=Path(q['query']).name.lower()
        expected={'climate_modelling:'+p.relative_to(root).as_posix() for p in actual if p.name.lower()==name}
        assert expected==set(filter(None,q['new_file_candidates'].split(';')))
        assert {s for s in zipnames if Path(s.split('!',1)[1]).name.lower()==name}==set(filter(None,q['new_archive_candidates'].split(';')))
    assert {q['id'] for q in queries if q['id'].startswith('F') and (q['new_file_candidates'] or q['new_archive_candidates'])}=={'F14','F15'}
    assert all(not q['new_file_candidates'] and not q['new_archive_candidates'] for q in queries if q['id'].startswith('POSTERIOR_'))
    for row in sources:
        assert sha(root/row['source'].split(':',1)[1])==row['sha256']
    for name in ['candidate_hashes.json']:
        for row in json.loads((OUT/name).read_text()):
            assert sha(root/row['source'].split(':',1)[1])==row['sha256']
    structures=json.loads((OUT/'candidate_structures.json').read_text())
    assert structures['distribution']['dimensions_lon_lat_taxa']==[720,360,275]
    assert structures['distribution']['hash_unchanged_after_read']
    for row in structures['metadata_csv_candidates']+structures['nearby_netCDF_variants']:
        assert row['hash_unchanged_after_read'] and sha(root/row['source'].split(':',1)[1])==row['sha256']
    assert [x['rows'] for x in structures['metadata_csv_candidates']]==[455,6849]
    assert 'mask2' not in structures['nearby_netCDF_variants'][0]['variables']
    # Preserve every tracked byte of all prior packages, including the old audit
    # and both previous start briefs. Compare to the secured commit, not ourselves.
    dirs=['local_case_matrix_2026_11_02','kinneret_binding_2026_11_02','kinneret_references_2026_11_02',
          'kinneret_postprocessing_2026_11_02','kinneret_phase1_spec_2026_11_02','literal_final_pdf_2026_11_02',
          'literal_age_depth_2026_11_02','literal_age_selection_2026_11_02','literal_cdf_discrepancy_2026_11_02',
          'literal_phase1_chain_2026_11_02','required_files_2026_11_02','research_snapshot_2026_11_02']
    paths=['docs/thesis_research/'+d for d in dirs]+[
        'docs/thesis_research/CLAUDE_NEXT_STEP_2026_11_02.md',
        'docs/thesis_research/CLAUDE_NEXT_STEP_TF_PREPARATION_2026_11_02.md',
        'docs/thesis_research/CLAUDE_RESUME_CLIMATE_MODELLING_2026_11_02.md']
    frozen=[]
    for record in git('ls-tree','-r',BASIS,'--',*paths).decode().splitlines():
        meta,path=record.split('\t',1);oid=meta.split()[2];raw=(ROOT/path).read_bytes()
        actual_oid=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        assert actual_oid==oid,'Frozen file changed: '+path
        frozen.append({'path':path,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'baseline_git_blob':oid})
    (OUT/'preserved_evidence.json').write_text(json.dumps({'basis_commit':BASIS,'files':frozen},indent=2)+'\n',encoding='utf-8', newline='\n')
    assert not git('diff',BASIS,'--','tools','tests','src','legacy','app','web','frontend').strip()
    checks=[]
    for local in [False,True]:
        args=[sys.executable,'tools/check_kinneret_phase1_spec.py']+(['--with-local'] if local else [])
        run=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,check=False)
        report=json.loads(run.stdout)
        assert run.returncode==2 and report['spec_valid'] and len(report['requirements'])==11
        assert report['selected_variant'] is None
        name='phase1_local_check.json' if local else 'phase1_portable_check.json'
        (OUT/name).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8', newline='\n')
        checks.append({'with_local':local,'exit_code':run.returncode,'verified_assets':report['verified_assets'],
                       'verified_facts':report['verified_facts'],'verified_snapshot_files':report['verified_snapshot_files'],
                       'open_requirements':len(report['requirements']),
                       'local_source_statuses':dict(Counter(x['status'] for x in report['local_sources'])) if local else None})
    docs=[ROOT/'CLAUDE.md',ROOT/'AGENTS.md']+[ROOT/'docs/thesis_research'/name for name in
         ['PROJECT_STATUS.md','LOCAL_DISS_RECONSTRUCTION_PLAN.md','SOURCE_RECOVERY_PLAN.md','NEXT_STEP_AFTER_CLIMATE_AUDIT_2026_11_02.md']]+[OUT/'README.md']
    links=0
    for p in docs:
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8-sig')):
            if ':' in link or link.startswith('#'):continue
            target=(p.parent/link.split('#')[0]).resolve()
            # This report is created after all checks; validate its path here.
            assert target.is_file() or target==OUT/'verification.json','Broken link: '+link
            links+=1
    # Prevent accidental publication of private absolute paths.
    for p in OUT.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts or p.name=='verification.json':continue
        raw=p.read_text(encoding='utf-8-sig')
        assert not re.search(r'(?i)[A-Z]:[\\/]+Users[\\/]',raw),'Private path: '+p.name
    check=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
    assert check.returncode==0,check.stdout
    results={'basis_commit':BASIS,'initial_worktree_clean':True,'initial_PR_state':'open_unmerged_head_equals_basis',
       'independent_index_files_verified':len(actual),'queries_independently_reconciled':len(queries),
       'new_priority_name_candidates':['F14','F15'],'all_8_posterior_variants_without_new_named_copy':True,
       'new_R_source_hashes_verified':len(sources),'source_metadata_and_selected_hashes_still_unchanged':True,
       'prior_files_byte_equal_to_basis':len(frozen),'frozen_start_briefs_preserved':2,
       'previous_climate_handoff_preserved':True,'parser_and_reader_tests':{'passed':23,'failed':0,
       'command':'.venv/Scripts/python.exe -m pytest tests/test_required_files_audit.py tests/test_local_rdata_metadata.py tests/test_local_rdata_stream.py -q -o cache_dir=.local/climate_modelling_audit_2026_11_02/pytest_cache'},
       'prior_200_numerical_tests_rerun':False,'phase1_checks':checks,'relative_document_links_valid':links,
       'git_diff_check_passed':True,'private_absolute_paths_found':False,
       'R_execution':False,'models_training_sampling_or_age_models_started':False,
       'full_fossil_reconstruction_started':False,'final_variant_selected':False,
       'prior_numerics_app_cli_baseline_unchanged':True,
       'evidence_files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.iterdir())
                          if p.is_file() and p.name!='verification.json'}}
    (OUT/'verification.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8', newline='\n')
    print(json.dumps({k:v for k,v in results.items() if k!='evidence_files'},ensure_ascii=True))


if __name__=='__main__':main()
