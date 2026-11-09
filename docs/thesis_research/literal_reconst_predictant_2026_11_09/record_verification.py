"""Run the targeted historical tests once and store what actually ran in verification.json. Not a regression gate."""
import json
import re
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
NEW = 'tests/test_historical_reconst_predictant.py'
EARLIER = ['tests/test_historical_local_cor.py', 'tests/test_historical_reconstruction_climate.py', 'tests/test_r_crosscheck_six_modules.py', 'tests/test_historical_weighted_median.py',
           'tests/test_historical_dirichlet_ratio.py', 'tests/test_r_runtime_crosscheck.py', 'tests/test_historical_weights_stage.py',
           'tests/test_historical_sampler_preparation.py', 'tests/test_historical_tf_missing_priors.py', 'tests/test_historical_gamma_prior_grid.py',
           'tests/test_historical_taxon_pruning.py', 'tests/test_historical_tf_preparation.py', 'tests/test_historical_phase1_chain.py',
           'tests/test_historical_cdf_discrepancy.py', 'tests/test_historical_age_selection.py', 'tests/test_historical_age_depth.py',
           'tests/test_historical_final_pdf.py', 'tests/test_kinneret_phase1_spec.py']


def passed(files):
    done = subprocess.run([sys.executable, '-m', 'pytest', *files, '-q'], cwd=ROOT, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    match = re.search(r'(\d+) passed', done.stdout)
    if done.returncode != 0 or not match:
        raise SystemExit(done.stdout[-3000:])
    return int(match.group(1))


def main():
    new, earlier = passed([NEW]), passed(EARLIER)
    both = passed([NEW, *EARLIER])
    if both != new + earlier:
        raise SystemExit('the combined run does not equal the sum')
    r = json.loads((OUT / 'r_crosscheck_result.json').read_text(encoding='utf-8'))
    (OUT / 'verification.json').write_text(json.dumps({
        'kind': 'recorded_test_run_not_a_case_reproduction', 'date': '2026-11-09', 'new_tests': new,
        'earlier_tests': earlier, 'total_passed': both, 'test_files': [NEW, *EARLIER],
        'r_crosscheck': {k: r[k] for k in ('values_compared', 'bit_equal', 'max_ulps', 'special_mismatch', 'stop_disagreements')},
        'r_runtime': r['runtime'],
        'not_run': ['full pytest tests', 'verify_packet.py', 'tools/verify_*', 'sampler', 'draws', 'training', 'age models', 'fossil reconstruction'],
        'historical_reproduction_ready': False,
    }, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(new, earlier, both)


if __name__ == '__main__':
    main()
