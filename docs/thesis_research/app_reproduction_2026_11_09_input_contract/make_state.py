"""Write reproduction_state.json: state 14 plus the input contract of the Kinneret case, with every file pinned. No new module and no new calculation."""
import hashlib
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PREVIOUS = 'docs/thesis_research/app_reproduction_2026_11_09_ratios_tfs/reproduction_state.json'
PACKET = 'docs/thesis_research/kinneret_input_contract_2026_11_09/'
NEW_TESTS = 285


def sha(rel):
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def main():
    state = json.loads((ROOT / PREVIOUS).read_text(encoding='utf-8'))
    state.update(date='2026-11-09', label='input contract of the Kinneret case (slot 3), no new module and no calculation',
                 supersedes={'state': PREVIOUS, 'state_sha256': sha(PREVIOUS)}, source_commit=sys.argv[1])
    state['tests'] = {'passed': state['tests']['passed'] + NEW_TESTS, 'scope': state['tests']['scope'], 'evidence': PACKET + 'README.md'}
    state['input_contract'] = {'package': PACKET + 'README.md', 'package_sha256': sha(PACKET + 'README.md'), 'contract': PACKET + 'input_contract.json', 'contract_sha256': sha(PACKET + 'input_contract.json'), 'new_tests': NEW_TESTS}
    (OUT / 'reproduction_state.json').write_text(json.dumps(state, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    print(sha('docs/thesis_research/app_reproduction_2026_11_09_input_contract/reproduction_state.json'))


if __name__ == '__main__':
    main()
