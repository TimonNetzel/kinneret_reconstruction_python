"""Write verification.json from the test run that actually took place; refuses to write unless it passed. Logs stay unversioned in .local."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
LOCAL = ROOT / '.local/input_contract_2026_11_09'
EARLIER, NEW, FILES = 1858, 285, 23
NEW_FILE = 'tests/test_kinneret_input_contract.py'


def checker():
    return subprocess.run([sys.executable, 'tools/check_kinneret_phase1_spec.py'], cwd=ROOT, capture_output=True, stdin=subprocess.DEVNULL).returncode


def main():
    exit_code = int((LOCAL / 'hist_pytest.exit').read_text(encoding='utf-8').strip())
    text = (LOCAL / 'hist_junit.xml').read_text(encoding='utf-8')
    attributes = text[text.index('<testsuite '):text.index('>', text.index('<testsuite '))]
    get = lambda key: int(attributes.split(f'{key}="')[1].split('"')[0])
    bad = get('failures') + get('errors')
    passed = get('tests') - get('skipped') - bad
    check = checker()
    if exit_code or bad or get('skipped') or passed != EARLIER + NEW or check != 2:
        raise SystemExit(f'not passed: exit {exit_code}, passed {passed}, failures/errors {bad}, skipped {get("skipped")}, phase-1 checker {check}')
    report = json.loads((OUT / 'input_contract.json').read_text(encoding='utf-8'))
    (OUT / 'verification.json').write_text(json.dumps({
        'kind': 'recorded_test_run_not_a_case_reproduction', 'date': '2026-11-09', 'new_test_file': NEW_FILE, 'new_tests': NEW, 'earlier_tests': EARLIER, 'total_passed': passed, 'test_files': FILES, 'exit_code': exit_code,
        'local_junit_sha256': hashlib.sha256((LOCAL / 'hist_junit.xml').read_bytes()).hexdigest(), 'phase1_checker_exit_code': check, 'line_references_checked_against_diss': True,
        'inputs': len(report['inputs']), 'conditions': len(report['conditions']), 'status_counts': report['status_counts'], 'bundles_closed': report['bundles_closed'], 'draws_run': report['draws_run'], 'sampler_run': report['sampler_run'],
        'not_run': ['full pytest tests', 'verify_packet.py', 'tools/verify_*', 'R scripts', 'sampler', 'draws', 'training', 'age models', 'fossil reconstruction'],
        'historical_reproduction_ready': False}, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(passed, NEW, check)


if __name__ == '__main__':
    main()
