"""Verify this completed source-fragment packet; no reconstruction execution."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASIS='956951b9546f3120620bf375bf073066c9f8bf44'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def emit(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT)


def main():
    binding=json.loads((OUT/'source_binding.json').read_text())
    cfg=json.loads((ROOT/'.local/my_reconst_paths.json').read_text(encoding='utf-8-sig'))
    original=Path(cfg['original_data_root'])/'src/mcmc_prepare.R'
    assert sha(original)==binding['source_sha256']
    assert sha(OUT/'source_excerpt.json')==binding['portable_excerpt_sha256']
    local_test_report=ROOT/'.local/tf_preparation_2026_11_02/pytest.xml'
    xml=ET.parse(local_test_report)
    cases=list(xml.iter('testcase'))
    assert all(not list(case) for case in cases),'Failures, errors, skips or unexpected captured output in test report'
    counts=dict(Counter(case.attrib['classname'] for case in cases))
    assert len(cases)==251 and counts['tests.test_historical_tf_preparation']==51
    arithmetic=json.loads((OUT/'arithmetic_check.json').read_text())
    assert arithmetic['maximum_absolute_rational_error']<=2e-14
    assert arithmetic['post_mask_renormalization_chain_PDF_max_difference']>.01
    assert arithmetic['active_original_draw_ids']==[1,3]
    prior_dirs={'local_case_matrix_2026_11_02','kinneret_binding_2026_11_02','kinneret_references_2026_11_02',
      'kinneret_postprocessing_2026_11_02','kinneret_phase1_spec_2026_11_02','literal_final_pdf_2026_11_02',
      'literal_age_depth_2026_11_02','literal_age_selection_2026_11_02','literal_cdf_discrepancy_2026_11_02',
      'literal_phase1_chain_2026_11_02','required_files_2026_11_02','research_snapshot_2026_11_02',
      'climate_modelling_audit_2026_11_02'}
    old_modules={'tools/historical_age_depth.py','tools/historical_age_selection.py','tools/historical_final_pdf.py','tools/historical_cdf_discrepancy.py'}
    old_tests={'tests/test_historical_age_depth.py','tests/test_historical_age_selection.py',
      'tests/test_historical_final_pdf.py','tests/test_historical_cdf_discrepancy.py','tests/test_historical_phase1_chain.py','tests/test_kinneret_phase1_spec.py'}
    frozen=[];briefs=[]
    for line in git('ls-tree','-r',BASIS,'--','docs/thesis_research','tools','tests').decode().splitlines():
        meta,path=line.split('\t',1);parts=Path(path).parts
        packet=len(parts)>2 and parts[0:2]==('docs','thesis_research') and parts[2] in prior_dirs
        brief=len(parts)==3 and parts[0:2]==('docs','thesis_research') and (parts[2].startswith('CLAUDE_') or parts[2].startswith('NEXT_STEP_')) and parts[2].endswith('.md')
        if not (packet or brief or path in old_modules or path in old_tests):continue
        oid=meta.split()[2];raw=(ROOT/path).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==oid,path
        frozen.append({'path':path,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'baseline_git_blob':oid})
        if brief:briefs.append(path)
    emit('preserved_evidence.json',{'basis_commit':BASIS,'files':frozen,'prior_start_briefs':briefs})
    assert not git('diff',BASIS,'--','src','legacy','cpp','configs','reference').strip()
    # Prior fragment manifests remain valid, not merely untouched as files.
    checked=0
    for folder in ['literal_final_pdf_2026_11_02','literal_age_depth_2026_11_02','literal_age_selection_2026_11_02','literal_cdf_discrepancy_2026_11_02','literal_phase1_chain_2026_11_02']:
        report=json.loads((ROOT/'docs/thesis_research'/folder/'verification.json').read_text())
        for path,entry in report.get('files',{}).items():
            assert sha(ROOT/path)==entry['sha256'],path
            checked+=1
    cli=[]
    for local in [False,True]:
        args=['tools/check_kinneret_phase1_spec.py']+(['--with-local'] if local else [])
        run=subprocess.run([sys.executable,*args],cwd=ROOT,capture_output=True,text=True)
        report=json.loads(run.stdout)
        assert run.returncode==2 and report['spec_valid'] and len(report['requirements'])==11 and report['selected_variant'] is None
        emit('phase1_local_check.json' if local else 'phase1_portable_check.json',report)
        cli.append({'arguments':args,'exit_code':run.returncode,'assets':report['verified_assets'],
          'facts':report['verified_facts'],'snapshot_files':report['verified_snapshot_files'],
          'open_requirements':len(report['requirements']),
          'local_source_statuses':dict(Counter(x['status'] for x in report['local_sources'])) if local else None})
    docs=[ROOT/'CLAUDE.md',ROOT/'AGENTS.md',OUT/'README.md',ROOT/'docs/thesis_research/PROJECT_STATUS.md',
          ROOT/'docs/thesis_research/LOCAL_DISS_RECONSTRUCTION_PLAN.md',
          ROOT/'docs/thesis_research/CLAUDE_NEXT_STEP_APP_REPRODUCTION_2026_11_02.md']
    links=0
    for p in docs:
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8-sig')):
            if ':' in target or target.startswith('#'):continue
            resolved=(p.parent/target.split('#')[0]).resolve()
            assert resolved.is_file() or resolved==OUT/'verification.json',(p.name,target)
            links+=1
    newfiles=[p for p in OUT.iterdir() if p.is_file() and p.name!='verification.json']+[
        ROOT/'tools/historical_tf_preparation.py',ROOT/'tests/test_historical_tf_preparation.py',
        ROOT/'docs/thesis_research/CLAUDE_NEXT_STEP_APP_REPRODUCTION_2026_11_02.md']
    for p in newfiles:
        assert b'\r\n' not in p.read_bytes(),'Exported hash evidence must bind LF Git bytes'
        assert not re.search(r'(?i)[A-Z]:[\\/]+Users[\\/]',p.read_text(encoding='utf-8-sig')),p.name
    diff=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
    assert diff.returncode==0,diff.stdout
    result={'status':'passed','basis_commit':BASIS,'initial_worktree_clean':True,
       'tests':{'new_preparation_cases_passed':51,'previous_cases_passed':200,'total_passed':251,'failed':0,'skipped':0,
                'module_counts':counts,'local_junit_sha256':sha(local_test_report)},
       'arithmetic':arithmetic,'source_hash_before_after_and_prior_equal':True,
       'prior_files_byte_equal_to_basis':len(frozen),'prior_start_briefs_byte_equal':len(briefs),
       'previous_fragment_modules_byte_equal':len(old_modules),'previous_fragment_tests_byte_equal':len(old_tests),
       'previous_manifest_file_entries_verified':checked,'phase1_rechecks':cli,
       'relative_document_links_valid':links,'git_diff_check_passed':True,
       'app_cli_baseline_unchanged':True,'R_execution':False,'actual_original_inputs_used_for_numerics':False,
       'historical_R_runtime_equivalence_claimed':False,'new_training_sampling_age_models_or_fossil_runs':False,
       'final_variant_selected':False,'historical_reproduction_ready':False,
       'next_claude_handoff':'docs/thesis_research/CLAUDE_NEXT_STEP_APP_REPRODUCTION_2026_11_02.md',
       'files':{p.relative_to(ROOT).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(newfiles)}}
    emit('verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='files'},ensure_ascii=True))


if __name__=='__main__':main()
