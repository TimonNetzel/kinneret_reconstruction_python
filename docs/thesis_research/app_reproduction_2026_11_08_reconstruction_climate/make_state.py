"""Write reproduction_state.json: state 9 plus the dormant reconstruction_climate, with every file pinned."""
import hashlib
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PREVIOUS = 'docs/thesis_research/app_reproduction_2026_11_08_r_crosscheck_six/reproduction_state.json'
PACKET = 'docs/thesis_research/literal_reconstruction_climate_2026_11_08/'
NEW_TESTS = 62


def sha(rel):
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def main():
    state = json.loads((ROOT / PREVIOUS).read_text(encoding='utf-8'))
    state.update(date='2026-11-08', label='reconstruction_climate, dormant in the bound run',
                 supersedes={'state': PREVIOUS, 'state_sha256': sha(PREVIOUS)}, source_commit=sys.argv[1])
    state['tests'] = {'passed': state['tests']['passed'] + NEW_TESTS, 'scope': state['tests']['scope'], 'evidence': PACKET + 'README.md'}
    state['modules'].append({
        'id': 'reconstruction_climate', 'package': PACKET + 'README.md', 'package_sha256': sha(PACKET + 'README.md'),
        'module': 'tools/historical_reconstruction_climate.py', 'original_sources': ['src/mcmc_functions_v1.6.R', 'matrixStats (weightedMedian)'],
        'new_tests': NEW_TESTS, 'conditional': True})
    example = {'cases': ['recent_basic', 'only_first_row_counts', 'prior_when_not_recent', 'first_row_zero', 'na_in_one_tf_cell',
                         'na_tf_at_zero_share', 'na_in_first_row_share', 'inexact_shares']}
    for key, name in (('fixture', 'synthetic_fixture.json'), ('result', 'synthetic_result.json'), ('rational', 'reference_expectations.json'),
                      ('arithmetic', 'arithmetic_check.json'), ('crosscheck', 'r_crosscheck_result.json'), ('binding', 'source_binding.json')):
        example[key], example[key + '_sha256'] = PACKET + name, sha(PACKET + name)
    state['climate_example'] = example
    (OUT / 'reproduction_state.json').write_text(json.dumps(state, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    print(sha('docs/thesis_research/app_reproduction_2026_11_08_reconstruction_climate/reproduction_state.json'))


if __name__ == '__main__':
    main()
