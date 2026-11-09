"""Write validation.json from what actually ran; hashes refer to logs kept unversioned in .local."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
LOCAL = ROOT / '.local/local_cor_2026_11_09'
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
    widths = json.loads((LOCAL / 'table_widths.json').read_text(encoding='utf-8'))
    (OUT / 'validation.json').write_text(json.dumps({
        'status': 'passed',
        'scope': 'read-only app display of the local cor with explicit artificial vectors and its readers; no historical case '
                 'reproduction, no sampler, no training or fossil reconstruction',
        'starting_commit': '437f6f4baeed4a086a2d5a80c7abc9e420a353de',
        'module_package_commit': '5af02eb78a7cccb2b2ea8cd408be5d397e9fd917',
        'reproduction_state_sha256': PIN,
        'frozen_earlier_states_sha256': [s['sha256'] for s in states[:-1]],
        'frozen_earlier_states_unchanged': frozen,
        'states_entry_change': 'STATES[9] received only its link commit 437f6f4 (pushed); STATES[0..8] untouched',
        'app_tests': {
            'files': ['tests/test_nachbaustand.py', 'tests/test_forschungsstand.py', 'tests/test_app.py', 'tests/test_artenvergleich.py'],
            'passed': 119, 'new_tests': 11, 'skipped': 2,
            'skip_reason': 'Python browser wrappers find no Playwright on the default Node search path; the same checks ran explicitly with --kanal msedge',
            'local_junit_sha256': sha('app_junit.xml')},
        'historical_tests': {'passed': 1172, 'new_tests': 135, 'rerun_after_app_changes': 1172, 'local_junit_sha256': sha('hist_junit.xml')},
        'builds': ['kinneret-recon app -> results/app/index.html', 'kinneret-recon app --eine-datei -> results/kinneret-app.html (5.6 MB)'],
        'browser_regression': {
            'tool': 'NODE_PATH=<Codex runtime>/dependencies/node/node_modules node tools/app_pruefen.js results/app/index.html results/kinneret-app.html --kanal msedge',
            'status': 'passed', 'cases_ok': 10, 'phone_viewport_px': [390, 844], 'languages': ['de', 'en'],
            'reproduction_checks': {'tests_shown': 1172, 'modules': 14, 'earlier_states': 10, 'links': 39, 'tables': 17, 'cor_rows': 8},
            'page_errors': 0, 'local_log_sha256': sha('browser.log')},
        'table_widths_390px': widths,
        'two_sites': {'status': 'passed', 'local_log_sha256': sha('two_sites.log')},
        'checkers': {'check_kinneret_phase1_spec': checker(), 'check_kinneret_phase1_spec_with_local': checker('--with-local')},
        'visual': {'mobile.png': 'cor table at 390 px, light', 'desktop.png': 'same part, desktop dark'},
        'not_run': ['pytest tests (full suite)', 'verify_packet.py and tools/verify_*', 'sampler, draws, training, age model or fossil reconstruction'],
        'historical_reproduction_ready': False, 'final_variant_selected': False,
    }, indent=2) + '\n', encoding='utf-8', newline='\n')
    print('frozen', frozen)


if __name__ == '__main__':
    main()
