"""Write reproduction_state.json: state 8 plus the pinned R 3.5.0 check of the six older modules."""
import hashlib
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PREVIOUS = 'docs/thesis_research/app_reproduction_2026_11_08_weighted_median/reproduction_state.json'
PACKET = 'docs/thesis_research/r_crosscheck_six_modules_2026_11_08/'
NEW_TESTS = 41


def sha(rel):
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def main():
    state = json.loads((ROOT / PREVIOUS).read_text(encoding='utf-8'))
    state.update(date='2026-11-08', label='R 3.5.0 cross check of the six older modules',
                 supersedes={'state': PREVIOUS, 'state_sha256': sha(PREVIOUS)}, source_commit=sys.argv[1])
    state['tests'] = {'passed': state['tests']['passed'] + NEW_TESTS, 'scope': state['tests']['scope'],
                      'evidence': PACKET + 'README.md'}
    state['six_check'] = {
        'package': PACKET + 'README.md', 'result': PACKET + 'six_crosscheck_result.json',
        'causes': PACKET + 'difference_causes.json', 'provider': PACKET + 'provider_rowcumsums.json',
        'new_tests': NEW_TESTS,
        'modules': {'age_depth': 'age_depth', 'selection': 'age_selection', 'final_pdf': 'final_pdf',
                    'cdf': 'cdf_discrepancy', 'tf': 'tf_preparation', 'pruning': 'taxon_pruning'}}
    for key in ('package', 'result', 'causes', 'provider'):
        state['six_check'][key + '_sha256'] = sha(state['six_check'][key])
    (OUT / 'reproduction_state.json').write_text(json.dumps(state, indent=2, ensure_ascii=False) + '\n', encoding='utf-8',
                                                 newline='\n')
    print(sha('docs/thesis_research/app_reproduction_2026_11_08_r_crosscheck_six/reproduction_state.json'))


if __name__ == '__main__':
    main()
