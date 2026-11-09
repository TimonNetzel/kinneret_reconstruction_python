"""Write reproduction_state.json: state 12 plus wt_power with a marked NumPy FFT, with every file pinned."""
import hashlib
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PREVIOUS = 'docs/thesis_research/app_reproduction_2026_11_09_reconst_predictant/reproduction_state.json'
PACKET = 'docs/thesis_research/literal_wt_power_2026_11_09/'
NEW_TESTS = 212


def sha(rel):
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def main():
    state = json.loads((ROOT / PREVIOUS).read_text(encoding='utf-8'))
    state.update(date='2026-11-09', label='wt_power from explicit series and wb, NumPy FFT with measured deviation from R',
                 supersedes={'state': PREVIOUS, 'state_sha256': sha(PREVIOUS)}, source_commit=sys.argv[1])
    state['tests'] = {'passed': state['tests']['passed'] + NEW_TESTS, 'scope': state['tests']['scope'], 'evidence': PACKET + 'README.md'}
    state['modules'].append({
        'id': 'wt_power', 'package': PACKET + 'README.md', 'package_sha256': sha(PACKET + 'README.md'), 'module': 'tools/historical_wt_power.py',
        'original_sources': ['src/mcmc_functions_v1.6.R', 'src/mcmc_prepare.R', 'main_v1.2.R'], 'new_tests': NEW_TESTS, 'conditional': True})
    example = {'cases': ['identity_wb_two_columns', 'one_column_drops', 'l_d_differs_from_length', 'long_double_in_the_mean', 'morlet_like_n32', 'na_in_d', 'plus_inf', 'na_then_nan']}
    for key, name in (('fixture', 'synthetic_fixture.json'), ('result', 'synthetic_result.json'), ('rational', 'reference_expectations.json'),
                      ('arithmetic', 'arithmetic_check.json'), ('crosscheck', 'r_crosscheck_result.json'), ('binding', 'source_binding.json'), ('masking', 'fft_masking.json')):
        example[key], example[key + '_sha256'] = PACKET + name, sha(PACKET + name)
    state['wt_example'] = example
    (OUT / 'reproduction_state.json').write_text(json.dumps(state, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    print(sha('docs/thesis_research/app_reproduction_2026_11_09_wt_power/reproduction_state.json'))


if __name__ == '__main__':
    main()
