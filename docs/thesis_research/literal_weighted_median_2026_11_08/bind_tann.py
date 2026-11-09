"""Collect what the top-level switch for Lake Kinneret does to ``tann``: writers and readers, without interpreting them.

Reads only; hashes must match the earlier frozen bindings. Writes tann_binding.json.
"""
import hashlib
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCES = {'src/mcmc_prepare.R': 'b897e84f8f2d72399de729592066341585974d521bf8164b185f20377ed5cc7e',
           'src/mcmc_functions_v1.6.R': 'b887144370e328fa016d1834ceebcd55622816d4b36838d33a26197110ab453c',
           'src/mcmc_conduction.R': '47c59c508e15150f7d6fda775a4f6bf6c14f0148e10cefd0014fa1ce6cfe5a9d'}
NAME = re.compile(r'(?<![\w.$])tann(?![\w.])')
PREDICTANT = re.compile(r'(?<![\w.$])predictant_current(?![\w.])')


def main():
    spec = importlib.util.spec_from_file_location('ws_bind', OUT.parent / 'literal_weights_stage_2026_11_03/bind_source.py')
    binder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(binder)
    cfg = json.loads((ROOT / '.local/my_reconst_paths.json').read_text(encoding='utf-8-sig'))
    base = Path(cfg['original_data_root'])
    text = {}
    for rel, sha in SOURCES.items():
        raw = (base / rel).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError(f'{rel} differs from the frozen binding')
        text[rel] = raw.decode('utf-8-sig').splitlines()
    prepare = text['src/mcmc_prepare.R']
    code = binder._code(prepare)
    depth = binder._depths(code)
    writers = [n for n, line in enumerate(code, 1) if re.search(r'(?<![\w.$])tann\s*<-', line)]
    if writers != [1208, 1210, 1219, 1229, 1241, 1249]:
        raise ValueError(f'writers of tann differ: {writers}')
    if prepare[1247].strip() != 'if(any(sitename == c("Lake Kinneret", ""))){' or prepare[1248].strip() != 'tann <- FALSE':
        raise ValueError('the Kinneret switch differs')
    if not (depth[1247] == 0 and depth[1248] == 1):
        raise ValueError('the Kinneret switch is not at top level')
    if 'Lake Kinneret' not in prepare[1208]:
        raise ValueError('line 1209 no longer excludes Kinneret')
    other_readers = [n for n, line in enumerate(code, 1) if NAME.search(line) and n not in writers]

    functions = text['src/mcmc_functions_v1.6.R']
    fcode = binder._code(functions)
    fdepth = binder._depths(fcode)
    definitions = [n for n, line in enumerate(fcode, 1) if re.match(r'\s*reconstruction\s*<-\s*function', line)]
    if definitions != [855, 1781]:
        raise ValueError(f'reconstruction definitions differ: {definitions}')
    active = definitions[-1]  # the file is sourced top to bottom, so the later definition is the one left in place
    end = next(n for n in range(active + 1, len(fcode) + 1) if fdepth[n - 1] == 0 and n > active + 1)
    readers = [{'line': n, 'text': functions[n - 1].strip()} for n in range(active + 1, end)
               if NAME.search(fcode[n - 1])]
    predictant = [{'line': n, 'text': functions[n - 1].strip()[:160]} for n in range(active + 1, end)
                  if PREDICTANT.search(fcode[n - 1])]
    conduction = text['src/mcmc_conduction.R'][145].strip()
    if 'reconstruction(posterior_temp=posterior_temp,tf.x1.current[posterior_temp$z,]' not in conduction \
            or ',predictors,tann,recent,' not in conduction:
        raise ValueError('the call in mcmc_conduction.R:146 differs')
    result = {
        'kind': 'tann_flag_source_collection_no_execution_no_interpretation',
        'sources_sha256': SOURCES,
        'writers_in_mcmc_prepare': writers,
        'switch_condition_line': 1248, 'switch_assignment_line': 1249,
        'switch_brace_depth_condition_line': depth[1247],
        'note_on_earlier_citation': 'earlier documents cite 1248 for tann <- FALSE; 1248 is the condition, 1249 the '
                                    'assignment. The earlier documents stay frozen.',
        'line_1209_excludes_kinneret': True,
        'other_code_lines_naming_tann_in_mcmc_prepare': other_readers,
        'caller': {'file': 'src/mcmc_conduction.R', 'line': 146},
        'reconstruction_definitions_in_v1_6': definitions, 'active_definition_line': active, 'active_definition_end': end,
        'readers_of_tann_in_active_definition': readers,
        'readers_of_predictant_current_in_active_definition': predictant,
        'effect_in_a_kinneret_run': 'tann is FALSE when reconstruction runs, so line 1939 takes predictant_pann as '
                                    'predictant_current; what that does to the correlations is not assessed here',
        'historical_reproduction_ready': False,
    }
    (OUT / 'tann_binding.json').write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n', encoding='utf-8',
                                           newline='\n')
    print('writers', writers, 'other', other_readers, 'active', active, end)
    print(json.dumps(readers, indent=0))
    print(json.dumps(predictant, indent=0)[:1500])


if __name__ == '__main__':
    main()
