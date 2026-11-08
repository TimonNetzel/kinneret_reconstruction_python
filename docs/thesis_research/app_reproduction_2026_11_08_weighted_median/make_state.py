"""Write reproduction_state.json: the newest app state as state 7 plus the weighted median, with every file pinned."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PREVIOUS = 'docs/thesis_research/app_reproduction_2026_11_03_dirichlet_ratio/reproduction_state.json'
PACKET = 'docs/thesis_research/literal_weighted_median_2026_11_08/'
SOURCE_COMMIT = 'a21c8f1ea7f6c285cbed2695a8b3750e4d0bba10'
NEW_TESTS = 182


def sha(rel):
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def main():
    state = json.loads((ROOT / PREVIOUS).read_text(encoding='utf-8'))
    state.update(date='2026-11-08', label='weighted median and NA mask of which.tf.missing',
                 supersedes={'state': PREVIOUS, 'state_sha256': sha(PREVIOUS)}, source_commit=SOURCE_COMMIT)
    state['tests'] = {'passed': state['tests']['passed'] + NEW_TESTS, 'scope': state['tests']['scope'],
                      'evidence': PACKET + 'README.md'}
    state['modules'].append({
        'id': 'weighted_median', 'package': PACKET + 'README.md', 'package_sha256': sha(PACKET + 'README.md'),
        'module': 'tools/historical_weighted_median.py',
        'original_sources': ['src/mcmc_prepare.R', 'matrixStats 0.53.1–1.0.0 (weightedMedian)'], 'new_tests': NEW_TESTS})
    state['median_example'] = {
        'cases': ['uniform_even', 'dropped_nonpositive', 'all_zero', 'na_weight_last', 'na_x_at_positive_weight',
                  'inf_order_na_after', 'inf_order_na_before', 'inf_odd_is_not_the_median', 'ties_trace'],
        **{k: PACKET + f for k, f in (('fixture', 'synthetic_fixture.json'), ('result', 'synthetic_result.json'),
                                      ('rational', 'reference_expectations.json'), ('arithmetic', 'arithmetic_check.json'),
                                      ('crosscheck', 'r_crosscheck_result.json'))},
    }
    example = state['median_example']
    for key in ('fixture', 'result', 'rational', 'arithmetic', 'crosscheck'):
        example[key + '_sha256'] = sha(example[key])
    (OUT / 'reproduction_state.json').write_text(json.dumps(state, indent=2, ensure_ascii=False) + '\n', encoding='utf-8',
                                                 newline='\n')
    print(sha('docs/thesis_research/app_reproduction_2026_11_08_weighted_median/reproduction_state.json'))


if __name__ == '__main__':
    main()
