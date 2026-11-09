"""Write validation.json from what actually ran; hashes refer to logs kept unversioned in .local."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
LOCAL = ROOT / '.local/reconstruction_climate_2026_11_08'
PIN = hashlib.sha256((OUT / 'reproduction_state.json').read_bytes()).hexdigest()


def sha(name):
    return hashlib.sha256((LOCAL / name).read_bytes()).hexdigest()


def checker(*args):
    return subprocess.run([sys.executable, 'tools/check_kinneret_phase1_spec.py', *args], cwd=ROOT, capture_output=True,
                          stdin=subprocess.DEVNULL).returncode


def main():
    sys.path.insert(0, str(ROOT / 'src'))
    from kinneret_recon import nachbaustand
    states = nachbaustand.STATES
    if states[-1]['sha256'] != PIN:
        raise SystemExit('the pin in STATES differs from the state file')
    frozen = [hashlib.sha256((ROOT / s['path']).read_bytes()).hexdigest() == s['sha256'] for s in states[:-1]]
    (OUT / 'validation.json').write_text(json.dumps({
        'status': 'passed',
        'scope': 'read-only app display of the dormant reconstruction_climate, marked as conditional; no effect on the bound '
                 'run, no historical case reproduction, no sampler, no training or fossil reconstruction',
        'starting_commit': '43733532ac20f183e5ee99631fc4e9fe3d357957',
        'module_package_commit': '77ccb85220dd7dda58848a31af28c8e4c8c1ff0e',
        'reproduction_state_sha256': PIN,
        'frozen_earlier_states_sha256': [s['sha256'] for s in states[:-1]],
        'frozen_earlier_states_unchanged': frozen,
        'states_entry_change': 'STATES[8] received only its link commit 4373353 (pushed); STATES[0..7] untouched',
        'app_tests': {
            'files': ['tests/test_nachbaustand.py', 'tests/test_forschungsstand.py', 'tests/test_app.py',
                      'tests/test_artenvergleich.py'],
            'passed': 108, 'new_tests': 10, 'skipped': 2,
            'skip_reason': 'Python browser wrappers find no Playwright on the default Node search path; the same checks '
                           'ran explicitly with --kanal msedge',
            'local_junit_sha256': sha('app_junit.xml')},
        'historical_tests': {'passed': 1037, 'new_tests': 62, 'rerun_after_app_changes': 1037,
                             'local_junit_sha256': sha('hist_junit.xml')},
        'builds': ['kinneret-recon app -> results/app/index.html',
                   'kinneret-recon app --eine-datei -> results/kinneret-app.html (5.6 MB)'],
        'browser_regression': {
            'tool': 'NODE_PATH=<Codex runtime>/dependencies/node/node_modules node tools/app_pruefen.js '
                    'results/app/index.html results/kinneret-app.html --kanal msedge',
            'status': 'passed', 'cases_ok': 10, 'phone_viewport_px': [390, 844], 'languages': ['de', 'en'],
            'reproduction_checks': {'tests_shown': 1037, 'modules': 13, 'earlier_states': 9, 'links': 36, 'tables': 16, 'climate_rows': 8},
            'runs': 'first run failed only on the absent-state count still set to 12; after that one correction all ten cases passed',
            'table_widths_390px': 'no table wider than its frame according to tools/app_pruefen.js; numeric widths were not kept '
                                  'because the session was interrupted while they were printed',
            'page_errors': 0, 'local_log_sha256': sha('browser.log')},
        'two_sites': {'status': 'passed', 'local_log_sha256': sha('two_sites.log')},
        'checkers': {'check_kinneret_phase1_spec': checker(), 'check_kinneret_phase1_spec_with_local': checker('--with-local')},
        'visual': {'mobile.png': 'climate table at 390 px, light', 'desktop.png': 'same part, desktop dark'},
        'not_run': ['pytest tests (full suite)', 'verify_packet.py and tools/verify_*',
                    'sampler, draws, training, age model or fossil reconstruction'],
        'historical_reproduction_ready': False, 'final_variant_selected': False,
    }, indent=2) + '\n', encoding='utf-8', newline='\n')
    print('frozen', frozen)


if __name__ == '__main__':
    main()
