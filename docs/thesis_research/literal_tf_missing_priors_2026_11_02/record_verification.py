"""Record what was actually run for this packet: new-file manifest, preserved files and the pytest report.

Reads git and the JUnit report written by the targeted pytest call in README.md; runs no tests itself.
"""
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
BASIS = '7d3d0f6852dbdf869b6e479d4eeffe80faaf2038'
NEW = ['tools/historical_tf_missing_priors.py', 'tests/test_historical_tf_missing_priors.py']
FILES = ['tests/test_historical_tf_missing_priors.py', 'tests/test_historical_gamma_prior_grid.py',
         'tests/test_historical_taxon_pruning.py', 'tests/test_historical_tf_preparation.py',
         'tests/test_historical_phase1_chain.py', 'tests/test_historical_cdf_discrepancy.py',
         'tests/test_historical_age_selection.py', 'tests/test_historical_age_depth.py',
         'tests/test_historical_final_pdf.py', 'tests/test_kinneret_phase1_spec.py']


def main():
    git = lambda *a: subprocess.run(['git', *a], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    changed = [line for line in git('status', '--porcelain', '--untracked-files=all').splitlines()
               if not line.startswith('?? ')]
    if changed:
        raise ValueError(f'tracked files differ from {BASIS[:7]}: {changed}')
    files = sorted(NEW + [p.relative_to(ROOT).as_posix() for p in OUT.iterdir()
                          if p.is_file() and p.name != 'verification.json'])
    report = ET.parse(ROOT / '.local/tf_missing_priors_2026_11_02/pytest.xml').getroot()
    suite = report if report.tag == 'testsuite' else report.find('testsuite')
    counts = {k: int(suite.get(k)) for k in ('tests', 'failures', 'errors', 'skipped')}
    verification = {
        'basis_commit': BASIS,
        'tracked_files_unchanged_since_basis': True,
        'new_files_sha256': {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files},
        'targeted_pytest': {'files': FILES, **counts, 'new_tests': 85, 'previous_tests': 352},
        'phase1_checkers': {'portable_exit': 2, 'with_local_exit': 2, 'expected': 2},
        'not_run': ['pytest tests (contains sampler unit tests)',
                    'literal_tf_preparation_2026_11_02/verify_packet.py (packet-bound, not a regression checker)',
                    'tools/verify_* (time-bound checkers, not regression checkers)',
                    'R, training, sampler, age model or fossil reconstruction'],
        'historical_reproduction_ready': False,
    }
    if counts != {'tests': 437, 'failures': 0, 'errors': 0, 'skipped': 0}:
        raise ValueError(f'unexpected pytest report {counts}')
    (OUT / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(f'{len(files)} new files recorded; tracked files unchanged; {counts["tests"]} tests passed')


if __name__ == '__main__':
    main()
