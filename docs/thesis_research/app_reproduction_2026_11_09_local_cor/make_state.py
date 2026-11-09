"""Write reproduction_state.json: state 10 plus the local cor and its readers, with every file pinned."""
import hashlib
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PREVIOUS = 'docs/thesis_research/app_reproduction_2026_11_08_reconstruction_climate/reproduction_state.json'
PACKET = 'docs/thesis_research/literal_local_cor_2026_11_09/'
NEW_TESTS = 135


def sha(rel):
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def main():
    state = json.loads((ROOT / PREVIOUS).read_text(encoding='utf-8'))
    state.update(date='2026-11-09', label='local cor and its time-space readers',
                 supersedes={'state': PREVIOUS, 'state_sha256': sha(PREVIOUS)}, source_commit=sys.argv[1])
    state['tests'] = {'passed': state['tests']['passed'] + NEW_TESTS, 'scope': state['tests']['scope'], 'evidence': PACKET + 'README.md'}
    state['modules'].append({
        'id': 'local_cor', 'package': PACKET + 'README.md', 'package_sha256': sha(PACKET + 'README.md'),
        'module': 'tools/historical_local_cor.py', 'original_sources': ['src/mcmc_functions_v1.6.R', 'src/mcmc_prepare.R', 'main_v1.2.R'],
        'new_tests': NEW_TESTS, 'conditional': False})
    example = {'cases': ['perfect_positive', 'perfect_negative', 'zero_covariance', 'x_na_removed', 'y_na_stays_with_na_rm', 'x_nan_kept',
                         'constant_x_inexact', 'empty']}
    for key, name in (('fixture', 'synthetic_fixture.json'), ('result', 'synthetic_result.json'), ('rational', 'reference_expectations.json'),
                      ('arithmetic', 'arithmetic_check.json'), ('crosscheck', 'r_crosscheck_result.json'), ('binding', 'source_binding.json')):
        example[key], example[key + '_sha256'] = PACKET + name, sha(PACKET + name)
    state['cor_example'] = example
    (OUT / 'reproduction_state.json').write_text(json.dumps(state, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    print(sha('docs/thesis_research/app_reproduction_2026_11_09_local_cor/reproduction_state.json'))


if __name__ == '__main__':
    main()
