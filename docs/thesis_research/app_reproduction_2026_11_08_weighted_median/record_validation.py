"""Write validation.json from what actually ran; hashes refer to logs kept unversioned in .local."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
LOCAL = ROOT / '.local/weighted_median_2026_11_08'
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
        'scope': 'read-only app display of the newest checked reproduction state (twelve submodules incl. the weighted '
                 'median and its NA mask) and of the stored R 3.5.0 cross checks, seven frozen earlier states and '
                 'synthetic examples; no historical case reproduction, no real transfer function, no sampler, no training '
                 'or fossil reconstruction',
        'starting_commit': 'cfd0ad8f1c0cb3f06909f50f21cc1e9a65592a68',
        'module_package_commit': 'a21c8f1ea7f6c285cbed2695a8b3750e4d0bba10',
        'reproduction_state_sha256': PIN,
        'frozen_earlier_states_sha256': [s['sha256'] for s in states[:-1]],
        'frozen_earlier_states_unchanged': frozen,
        'states_entry_change': 'STATES[6] received only its link commit cfd0ad8 (git log -1 --format=%H on its state '
                               'file; pushed); STATES[0..5] untouched',
        'app_tests': {
            'files': ['tests/test_nachbaustand.py', 'tests/test_forschungsstand.py', 'tests/test_app.py',
                      'tests/test_artenvergleich.py'],
            'passed': 91, 'new_tests': 10, 'skipped': 2,
            'skip_reason': 'Python browser wrappers find no Playwright on the default Node search path; the same checks '
                           'ran explicitly with --kanal msedge',
            'local_junit_sha256': sha('app_junit.xml')},
        'historical_tests': {'passed': 934, 'new_tests': 182, 'rerun_after_app_changes': 934,
                             'local_junit_sha256': sha('hist_junit.xml')},
        'builds': ['kinneret-recon app -> results/app/index.html',
                   'kinneret-recon app --eine-datei -> results/kinneret-app.html (5.6 MB)'],
        'browser_regression': {
            'tool': 'NODE_PATH=<Codex runtime>/dependencies/node/node_modules node tools/app_pruefen.js '
                    'results/app/index.html results/kinneret-app.html --kanal msedge',
            'status': 'passed', 'cases_ok': 10, 'phone_viewport_px': [390, 844], 'languages': ['de', 'en'],
            'reproduction_checks': {
                'tests_shown': 934, 'modules': 12, 'earlier_states': 7, 'links': 31, 'links_pinned_to_a21c8f1': 24,
                'example_tables': 14, 'runtime_median_sentence': True, 'median_rows': 9, 'median_na_cells': 4,
                'median_ulps_zero_cells': 9, 'mask_rows_removed': 3, 'mask_na_cells': 4},
            'runs': 'first run failed on the absent-state count still set to 11 and one table wider than its frame at '
                    '390 px; after white-space: normal for the median cells all later runs passed all ten cases',
            'measured_table_widths_390px': {'median': [297, 332], 'median-mask': [332, 332]},
            'case_labels': {'median': [f'M{i}' for i in range(1, 10)],
                            'note': 'new labels do not repeat 1-4, G1-G3 or D1-D4'},
            'page_errors': 0, 'local_log_sha256': sha('browser.log')},
        'two_sites': {'status': 'passed', 'local_log_sha256': sha('two_sites.log')},
        'checkers': {'check_kinneret_phase1_spec': checker(), 'check_kinneret_phase1_spec_with_local': checker('--with-local')},
        'visual': {'mobile.png': 'median and mask tables at 390 px, light', 'desktop.png': 'same part, desktop dark'},
        'not_run': ['pytest tests (full suite)', 'verify_packet.py and tools/verify_*',
                    'sampler, draws, training, age model or fossil reconstruction'],
        'historical_reproduction_ready': False, 'final_variant_selected': False,
    }, indent=2) + '\n', encoding='utf-8', newline='\n')
    print('frozen', frozen, 'checkers written')


if __name__ == '__main__':
    main()
