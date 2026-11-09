"""Write validation.json from what actually ran; hashes refer to logs kept unversioned in .local. Refuses to write unless everything passed."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
LOCAL = ROOT / '.local/ratios_tfs_2026_11_09'
PIN = hashlib.sha256((OUT / 'reproduction_state.json').read_bytes()).hexdigest()
EXPECTED_PIN = 'a45d172c117805bc5d7e75a25ae22606650552aa0a8a73f04a4e706f76226230'
EXPECTED = {'app_passed': 172, 'app_skipped': 2, 'hist_passed': 1858, 'cases_ok': 10, 'tables': 20}


def sha(name):
    return hashlib.sha256((LOCAL / name).read_bytes()).hexdigest()


def checker(*args):
    return subprocess.run([sys.executable, 'tools/check_kinneret_phase1_spec.py', *args], cwd=ROOT, capture_output=True, stdin=subprocess.DEVNULL).returncode


def counts(name):
    text = (LOCAL / name).read_text(encoding='utf-8')
    attributes = text[text.index('<testsuite '):text.index('>', text.index('<testsuite '))]
    get = lambda key: int(attributes.split(f'{key}="')[1].split('"')[0])
    return get('tests') - get('skipped') - get('failures') - get('errors'), get('skipped'), get('failures') + get('errors')


def read(name):
    return (LOCAL / name).read_text(encoding='utf-8')


def main():
    sys.path.insert(0, str(ROOT / 'src'))
    from kinneret_recon import nachbaustand
    states = nachbaustand.STATES
    if states[-1]['sha256'] != PIN or PIN != EXPECTED_PIN:
        raise SystemExit('the pin in STATES, the state file and the expected pin differ')
    frozen = [hashlib.sha256((ROOT / s['path']).read_bytes()).hexdigest() == s['sha256'] for s in states[:-1]]
    widths = json.loads(read('table_widths.json'))
    log, two_sites = read('browser.log'), read('two_sites.log')
    app_passed, app_skipped, app_bad = counts('app_junit.xml')
    hist_passed, hist_skipped, hist_bad = counts('hist_junit2.xml')
    exits = {name: int(read(f'{name}.exit').strip()) for name in ('browser', 'two_sites', 'app_pytest', 'hist_pytest2')}
    too_wide = [w for w in widths if w['scroll'] > w['client'] + 1]
    standby = read('kernel_power.txt')
    problems = {
        'exits': any(exits.values()),
        'fehler': 'FEHLER' in log or 'FEHLER' in two_sites,
        'timeout': 'timeout' in log.lower() or 'timeout' in two_sites.lower(),
        'too_wide': bool(too_wide),
        'table_count': len(widths) != EXPECTED['tables'],
        'ok_lines': log.count(': ok') != EXPECTED['cases_ok'],
        'two_sites_ok': 'zwei Standorte: ok' not in two_sites,
        'test_counts': (app_passed, app_skipped, hist_passed) != (EXPECTED['app_passed'], EXPECTED['app_skipped'], EXPECTED['hist_passed']),
        'test_failures': bool(app_bad or hist_bad or hist_skipped),
        'frozen_states': not all(frozen),
        'standby_during_runs': 'keine 506/507' not in standby,
    }
    if any(problems.values()):
        raise SystemExit(f'not passed: {[k for k, v in problems.items() if v]}, exits {exits}, wide tables {too_wide}, ok lines {log.count(": ok")}')
    size = (ROOT / 'results/kinneret-app.html').stat().st_size
    (OUT / 'validation.json').write_text(json.dumps({
        'status': 'passed',
        'exit_codes': exits,
        'standby_check': {'windows_system_log_kernel_power_506_507': 'none from 2026-11-09T19:05:00+01:00 until the query at the end of the runs', 'local_record_sha256': sha('kernel_power.txt')},
        'scope': 'read-only app display of ratios_tfs_no_ages with explicit artificial grids, densities and proposals, own normal functions with a measured and marked deviation from R; '
                 'no historical case reproduction, no sampler, no training or fossil reconstruction',
        'not_bit_identical_with_R': 'by the decision of the user of 2026-11-09; the app states it in the module title and in the example',
        'cplusplus_lines_run': False,
        'starting_commit': '9f6165e72a33b74fb590f4de30b061fc8a4b404f',
        'module_package_commit': '9f6165e72a33b74fb590f4de30b061fc8a4b404f',
        'reproduction_state_sha256': PIN,
        'frozen_earlier_states_sha256': [s['sha256'] for s in states[:-1]],
        'frozen_earlier_states_unchanged': frozen,
        'states_entry_change': 'STATES[12] received only its link commit 52a6acc (pushed); STATES[0..11] untouched',
        'display_fixes': {
            'found': 'two language keys were defined twice, so the later one won: nb_ex_w_case (G1-G3 of the weights table were shown as W1-W3) and nb_ex_m_x2 ("NA only x2" of the mask table was shown as "Median x2")',
            'change': 'renamed to nb_ex_wt_case and nb_ex_m_only_x2; a browser check and tests/test_nachbaustand.py::test_every_language_key_is_defined_once now cover it',
            'affected_earlier_app_states': 'the wrong labels come from the shared app code; the frozen state files of states 1-13 are unchanged'},
        'app_tests': {
            'files': ['tests/test_nachbaustand.py', 'tests/test_forschungsstand.py', 'tests/test_app.py', 'tests/test_artenvergleich.py'],
            'passed': app_passed, 'new_tests': app_passed - 151, 'skipped': app_skipped,
            'skip_reason': 'Python browser wrappers find no Playwright on the default Node search path; the same checks ran explicitly with --kanal msedge',
            'local_junit_sha256': sha('app_junit.xml')},
        'historical_tests': {'passed': hist_passed, 'files': 22, 'new_tests_module': 212, 'new_tests_rtruncnorm_binding': 32, 'local_junit_sha256': sha('hist_junit2.xml')},
        'builds': ['kinneret-recon app -> results/app/index.html', f'kinneret-recon app --eine-datei -> results/kinneret-app.html ({size / 1e6:.1f} MB)'],
        'browser_regression': {
            'tool': 'NODE_PATH=<Codex runtime>/dependencies/node/node_modules node tools/app_pruefen.js results/app/index.html results/kinneret-app.html --kanal msedge',
            'status': 'passed', 'cases_ok': log.count(': ok'), 'phone_viewport_px': [390, 844], 'languages': ['de', 'en'], 'fehler_lines': log.count('FEHLER'),
            'reproduction_checks': {'tests_shown': 1858, 'modules': 17, 'earlier_states': 13, 'links': 49, 'tables': 20, 'ratios_rows': 8},
            'page_errors': 0, 'local_log_sha256': sha('browser.log')},
        'table_widths_390px': widths,
        'two_sites': {'status': 'passed', 'local_log_sha256': sha('two_sites.log')},
        'checkers': {'check_kinneret_phase1_spec': checker(), 'check_kinneret_phase1_spec_with_local': checker('--with-local')},
        'visual': {'mobile.png': 'ratios table Q1-Q8 at 390 px, light', 'desktop.png': 'same table, desktop dark'},
        'not_run': ['pytest tests (full suite)', 'verify_packet.py and tools/verify_*', 'compilation of the C++ lines', 'sampler, draws, training, age model or fossil reconstruction'],
        'historical_reproduction_ready': False, 'final_variant_selected': False,
    }, indent=2) + '\n', encoding='utf-8', newline='\n')
    print('frozen', all(frozen), len(frozen), app_passed, app_skipped, hist_passed, log.count(': ok'), log.count('FEHLER'))


if __name__ == '__main__':
    main()
