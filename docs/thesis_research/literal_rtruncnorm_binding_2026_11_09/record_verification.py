"""Write verification.json from the test run that actually took place; refuses to write unless it passed. Logs stay unversioned in .local."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
LOCAL = ROOT / '.local/rtruncnorm_2026_11_09'
EARLIER = 1614
NEW_FILE = 'tests/test_rtruncnorm_binding.py'


def main():
    exit_code = int((LOCAL / 'hist_pytest.exit').read_text(encoding='utf-8').strip())
    text = (LOCAL / 'hist_junit.xml').read_text(encoding='utf-8')
    attributes = text[text.index('<testsuite '):text.index('>', text.index('<testsuite '))]
    get = lambda key: int(attributes.split(f'{key}="')[1].split('"')[0])
    bad = get('failures') + get('errors')
    passed = get('tests') - get('skipped') - bad
    if exit_code or bad or get('skipped') or passed <= EARLIER:
        raise SystemExit(f'not passed: exit {exit_code}, passed {passed}, failures/errors {bad}, skipped {get("skipped")}')
    binding = json.loads((OUT / 'source_binding.json').read_text(encoding='utf-8'))
    cran = json.loads((OUT / 'rcppdist_versions.json').read_text(encoding='utf-8'))
    (OUT / 'verification.json').write_text(json.dumps({
        'kind': 'recorded_test_run_not_a_case_reproduction', 'date': '2026-11-09', 'new_test_file': NEW_FILE, 'new_tests': passed - EARLIER, 'earlier_tests': EARLIER, 'total_passed': passed,
        'test_files': 21, 'exit_code': exit_code, 'local_junit_sha256': hashlib.sha256((LOCAL / 'hist_junit.xml').read_bytes()).hexdigest(),
        'versions_in_window': [v['file'] for v in cran['versions_in_window']], 'draws_run': binding['draws_run'], 'compiled': binding['compiled'],
        'not_run': ['full pytest tests', 'verify_packet.py', 'tools/verify_*', 'app tests and browser runs (no code of the app changed)', 'compilation of rtruncnorm_cpp', 'sampler', 'draws', 'training', 'age models', 'fossil reconstruction'],
        'historical_reproduction_ready': False}, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(passed, passed - EARLIER)


if __name__ == '__main__':
    main()
