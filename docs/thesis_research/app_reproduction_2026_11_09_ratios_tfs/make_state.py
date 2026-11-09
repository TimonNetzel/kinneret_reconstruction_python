"""Write reproduction_state.json: state 13 plus ratios_tfs_no_ages with its own normal functions, and the binding check of rtruncnorm_cpp, with every file pinned."""
import hashlib
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PREVIOUS = 'docs/thesis_research/app_reproduction_2026_11_09_wt_power/reproduction_state.json'
PACKET = 'docs/thesis_research/literal_ratios_tfs_2026_11_09/'
BINDING = 'docs/thesis_research/literal_rtruncnorm_binding_2026_11_09/'
NEW_TESTS, BINDING_TESTS = 212, 32


def sha(rel):
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def main():
    state = json.loads((ROOT / PREVIOUS).read_text(encoding='utf-8'))
    state.update(date='2026-11-09', label='ratios_tfs_no_ages from explicit inputs, own normal functions with measured deviation from R',
                 supersedes={'state': PREVIOUS, 'state_sha256': sha(PREVIOUS)}, source_commit=sys.argv[1])
    state['tests'] = {'passed': state['tests']['passed'] + NEW_TESTS + BINDING_TESTS, 'scope': state['tests']['scope'], 'evidence': PACKET + 'README.md'}
    state['modules'].append({
        'id': 'ratios_tfs', 'package': PACKET + 'README.md', 'package_sha256': sha(PACKET + 'README.md'), 'module': 'tools/historical_ratios_tfs.py',
        'original_sources': ['src/mcmc_functions_v1.6.R', 'src/mcmc_conduction.R'], 'new_tests': NEW_TESTS, 'conditional': False})
    state['rtruncnorm_binding'] = {'package': BINDING + 'README.md', 'package_sha256': sha(BINDING + 'README.md'), 'new_tests': BINDING_TESTS}
    example = {'cases': ['three_taxa_quarters', 'tie_takes_first_minimum', 'off_grid_takes_end', 'start_density_zero', 'both_densities_zero', 'old_outside_truncation', 'live_shape_grid_1000', 'wide_proposal_small_scale']}
    for key, name in (('fixture', 'synthetic_fixture.json'), ('result', 'synthetic_result.json'), ('rational', 'reference_expectations.json'), ('arithmetic', 'arithmetic_check.json'),
                      ('crosscheck', 'r_crosscheck_result.json'), ('binding', 'source_binding.json')):
        example[key], example[key + '_sha256'] = PACKET + name, sha(PACKET + name)
    state['ratios_example'] = example
    (OUT / 'reproduction_state.json').write_text(json.dumps(state, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    print(sha('docs/thesis_research/app_reproduction_2026_11_09_ratios_tfs/reproduction_state.json'))


if __name__ == '__main__':
    main()
