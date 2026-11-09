"""Write verification.json from the test run that actually took place; refuses to write unless it passed. Logs stay unversioned in .local."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
LOCAL = ROOT / '.local/ratios_tfs_2026_11_09'
EARLIER = 1646


def main():
    exit_code = int((LOCAL / 'hist_pytest.exit').read_text(encoding='utf-8').strip())
    text = (LOCAL / 'hist_junit.xml').read_text(encoding='utf-8')
    attributes = text[text.index('<testsuite '):text.index('>', text.index('<testsuite '))]
    get = lambda key: int(attributes.split(f'{key}="')[1].split('"')[0])
    bad = get('failures') + get('errors')
    passed = get('tests') - get('skipped') - bad
    if exit_code or bad or get('skipped') or passed <= EARLIER:
        raise SystemExit(f'not passed: exit {exit_code}, passed {passed}, failures/errors {bad}, skipped {get("skipped")}')
    cross = json.loads((OUT / 'r_crosscheck_result.json').read_text(encoding='utf-8'))
    (OUT / 'verification.json').write_text(json.dumps({
        'kind': 'recorded_test_run_not_a_case_reproduction', 'date': '2026-11-09', 'new_test_file': 'tests/test_historical_ratios_tfs.py', 'new_tests': passed - EARLIER, 'earlier_tests': EARLIER, 'total_passed': passed,
        'test_files': 22, 'exit_code': exit_code, 'local_junit_sha256': hashlib.sha256((LOCAL / 'hist_junit.xml').read_bytes()).hexdigest(),
        'r_crosscheck': {'bit_identical_with_R': cross['bit_identical_with_R'], 'cplusplus_lines_run': cross['cplusplus_lines_run'], 'stress_cases': cross['stress_cases'], 'stress': {k: cross['stress'][k] for k in ('grid_ratio', 'prod_from_R_factors', 'd_truncnorm', 'density_ratio', 'transfer_ratio')}},
        'r_runtime': cross['runtime'], 'python': cross['python'],
        'not_run': ['full pytest tests', 'verify_packet.py', 'tools/verify_*', 'compilation of the C++ lines', 'sampler', 'draws', 'training', 'age models', 'fossil reconstruction'], 'historical_reproduction_ready': False}, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(passed, passed - EARLIER)


if __name__ == '__main__':
    main()
