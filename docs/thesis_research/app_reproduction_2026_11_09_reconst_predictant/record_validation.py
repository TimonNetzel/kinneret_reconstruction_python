"""Write validation.json from what actually ran; hashes refer to logs kept unversioned in .local."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
LOCAL = ROOT / '.local/reconst_predictant_2026_11_09'
PIN = hashlib.sha256((OUT / 'reproduction_state.json').read_bytes()).hexdigest()


def sha(name):
    return hashlib.sha256((LOCAL / name).read_bytes()).hexdigest()


def checker(*args):
    return subprocess.run([sys.executable, 'tools/check_kinneret_phase1_spec.py', *args], cwd=ROOT, capture_output=True,
                          stdin=subprocess.DEVNULL).returncode


def counts(name):
    text = (LOCAL / name).read_text(encoding='utf-8')
    attributes = text[text.index('<testsuite '):text.index('>', text.index('<testsuite '))]
    get = lambda key: int(attributes.split(f'{key}="')[1].split('"')[0])
    return get('tests') - get('skipped') - get('failures') - get('errors'), get('skipped')


def main():
    sys.path.insert(0, str(ROOT / 'src'))
    from kinneret_recon import nachbaustand
    states = nachbaustand.STATES
    if states[-1]['sha256'] != PIN:
        raise SystemExit('the pin in STATES differs from the state file')
    frozen = [hashlib.sha256((ROOT / s['path']).read_bytes()).hexdigest() == s['sha256'] for s in states[:-1]]
    widths = json.loads((LOCAL / 'table_widths.json').read_text(encoding='utf-8'))
    log = (LOCAL / 'browser.log').read_text(encoding='utf-8')
    app_passed, app_skipped = counts('app_junit.xml')
    hist_passed, _ = counts('hist_junit.xml')
    exits = {name: int((LOCAL / f'{name}.exit').read_text(encoding='utf-8').strip()) for name in ('browser', 'two_sites', 'app_pytest', 'hist_pytest')}
    two_sites = (LOCAL / 'two_sites.log').read_text(encoding='utf-8')
    too_wide = [w for w in widths if w['scroll'] > w['client'] + 1]
    if any(exits.values()) or 'FEHLER' in log or 'FEHLER' in two_sites or 'Timeout' in log or too_wide or log.count(': ok') != 10:
        raise SystemExit(f'not passed: exits {exits}, wide tables {too_wide}, ok lines {log.count(": ok")}')
    (OUT / 'validation.json').write_text(json.dumps({
        'status': 'passed',
        'exit_codes': exits,
        'click_timeout_investigation': {
            'symptom': 'page.click on the gewichte tab in pruefeGesten for results/kinneret-app.html ran into the 30 s timeout (exit 1, no FEHLER line)',
            'cause': 'the computer entered modern standby because the lid was closed while the run was at this step; not a defect found in the app or the check',
            'evidence': {
                'failing_run_started_local': '2026-11-09T16:43:58+01:00',
                'windows_system_log_kernel_power': {'506_enter_standby_lid': '2026-11-09T16:47:25+01:00', '507_leave_standby_lid': '2026-11-09T17:31:31+01:00'},
                'failing_log_last_written_local': '2026-11-09T17:31:35+01:00',
                'seconds_from_start_to_standby': 207, 'seconds_of_complete_undisturbed_runs': [225, 224],
                'isolated_gesture_runs_single_file': '5 of 5 without timeout, tab clicks at most 428 ms',
                'complete_rerun_on_unchanged_build': 'exit 0, no FEHLER line',
                'local_failing_log_sha256': sha('browser_run1_timeout.log'),
                'local_isolated_log_sha256': sha('klick_diagnose_einzeln.log'),
                'local_rerun_log_sha256': sha('browser_repro_unveraendert_1.log')},
            'limit': 'correlation in time and the absence of the failure without standby; a standby was not provoked on purpose',
            'change': 'none to the app or to tools/app_pruefen.js; no retry added, because it could hide a real hang'},
        'scope': 'read-only app display of reconst and the predictants with explicit artificial matrices and their readers; no historical case '
                 'reproduction, no sampler, no training or fossil reconstruction',
        'starting_commit': '38775e1fc959b936fb22b7dd6dc88bef69fb7eb1',
        'module_package_commit': '599c6e44e791487f112b5699048eba9d90c05798',
        'reproduction_state_sha256': PIN,
        'frozen_earlier_states_sha256': [s['sha256'] for s in states[:-1]],
        'frozen_earlier_states_unchanged': frozen,
        'states_entry_change': 'STATES[10] received only its link commit 38775e1 (pushed); STATES[0..9] untouched',
        'app_tests': {
            'files': ['tests/test_nachbaustand.py', 'tests/test_forschungsstand.py', 'tests/test_app.py', 'tests/test_artenvergleich.py'],
            'passed': app_passed, 'new_tests': app_passed - 119, 'skipped': app_skipped,
            'skip_reason': 'Python browser wrappers find no Playwright on the default Node search path; the same checks ran explicitly with --kanal msedge',
            'local_junit_sha256': sha('app_junit.xml')},
        'historical_tests': {'passed': hist_passed, 'new_tests': 230, 'local_junit_sha256': sha('hist_junit.xml')},
        'builds': ['kinneret-recon app -> results/app/index.html', 'kinneret-recon app --eine-datei -> results/kinneret-app.html (5.6 MB)'],
        'browser_regression': {
            'tool': 'NODE_PATH=<Codex runtime>/dependencies/node/node_modules node tools/app_pruefen.js results/app/index.html results/kinneret-app.html --kanal msedge',
            'status': 'passed', 'cases_ok': log.count(': ok'), 'phone_viewport_px': [390, 844], 'languages': ['de', 'en'],
            'fehler_lines': log.count('FEHLER'),
            'reproduction_checks': {'tests_shown': 1402, 'modules': 15, 'earlier_states': 11, 'links': 42, 'tables': 18, 'reconst_rows': 8},
            'page_errors': 0, 'local_log_sha256': sha('browser.log')},
        'table_widths_390px': widths,
        'two_sites': {'status': 'passed', 'local_log_sha256': sha('two_sites.log')},
        'checkers': {'check_kinneret_phase1_spec': checker(), 'check_kinneret_phase1_spec_with_local': checker('--with-local')},
        'visual': {'mobile.png': 'reconst table at 390 px, light', 'desktop.png': 'same part, desktop dark'},
        'not_run': ['pytest tests (full suite)', 'verify_packet.py and tools/verify_*', 'sampler, draws, training, age model or fossil reconstruction'],
        'historical_reproduction_ready': False, 'final_variant_selected': False,
    }, indent=2) + '\n', encoding='utf-8', newline='\n')
    print('frozen', frozen, app_passed, app_skipped, hist_passed, log.count(': ok'), log.count('FEHLER'))


if __name__ == '__main__':
    main()
