"""Write reproduction_state.json: state 11 plus reconst and the predictants, with every file pinned."""
import hashlib
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PREVIOUS = 'docs/thesis_research/app_reproduction_2026_11_09_local_cor/reproduction_state.json'
PACKET = 'docs/thesis_research/literal_reconst_predictant_2026_11_09/'
NEW_TESTS = 230


def sha(rel):
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def main():
    state = json.loads((ROOT / PREVIOUS).read_text(encoding='utf-8'))
    state.update(date='2026-11-09', label='reconst and the predictants from a given tf_sample',
                 supersedes={'state': PREVIOUS, 'state_sha256': sha(PREVIOUS)}, source_commit=sys.argv[1])
    state['tests'] = {'passed': state['tests']['passed'] + NEW_TESTS, 'scope': state['tests']['scope'], 'evidence': PACKET + 'README.md'}
    state['modules'].append({
        'id': 'reconst_predictant', 'package': PACKET + 'README.md', 'package_sha256': sha(PACKET + 'README.md'),
        'module': 'tools/historical_reconst_predictant.py', 'original_sources': ['src/mcmc_functions_v1.6.R', 'src/mcmc_conduction.R', 'src/mcmc_prepare.R', 'main_v1.2.R'],
        'new_tests': NEW_TESTS, 'conditional': False})
    example = {'cases': ['hand_all_rows', 'one_row_selected_drops_to_vector', 'one_taxon_many_rows', 'no_row_selected', 'tie_with_nan_in_other_row',
                         'zero_times_na', 'token_x_nan_y_na', 'inf_minus_inf']}
    for key, name in (('fixture', 'synthetic_fixture.json'), ('result', 'synthetic_result.json'), ('rational', 'reference_expectations.json'),
                      ('arithmetic', 'arithmetic_check.json'), ('crosscheck', 'r_crosscheck_result.json'), ('binding', 'source_binding.json'), ('measurement', 'nan_rule_measurement.json')):
        example[key], example[key + '_sha256'] = PACKET + name, sha(PACKET + name)
    state['reconst_example'] = example
    (OUT / 'reproduction_state.json').write_text(json.dumps(state, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    print(sha('docs/thesis_research/app_reproduction_2026_11_09_reconst_predictant/reproduction_state.json'))


if __name__ == '__main__':
    main()
