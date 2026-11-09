"""Bind reconstruction_climate and establish whether the bound run can reach it, or lines 1408-1413 of mcmc_prepare.R.

Reads only. Hashes must match the frozen bindings. Every .R file under the diss folder is searched for the function name.
Writes source_binding.json, source_excerpt.json and determinism_audit.json.
"""
import hashlib
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASIS = '43733532ac20f183e5ee99631fc4e9fe3d357957'
SOURCES = {'src/mcmc_functions_v1.6.R': 'b887144370e328fa016d1834ceebcd55622816d4b36838d33a26197110ab453c',
           'src/mcmc_prepare.R': 'b897e84f8f2d72399de729592066341585974d521bf8164b185f20377ed5cc7e'}
LITERAL = {'src/mcmc_functions_v1.6.R': [
    (1188, 'reconstruction_climate <- function(taxa.appear.core,which.taxa,tf.x1.current,tf.x2.current,recent){'),
    (1190, 'if(recent){'),
    (1191, 'tann.recent <- weightedMedian(x1range.bck,(taxa.appear.core[1,] %*% tf.x1.current)[1,])'),
    (1192, 'pann.recent <- weightedMedian(x2range.bck,(taxa.appear.core[1,] %*% tf.x2.current)[1,])'),
    (1194, 'tann.recent <- prior.recent$mean.tmp'), (1195, 'pann.recent <- prior.recent$mean.pann'),
    (1198, 'return(list(tann = tann.recent, pann = pann.recent))'),
    (1850, 'posterior_temp$post_tann <- predictant_tann[1]'), (1851, 'posterior_temp$post_pann <- predictant_pann[1]')],
    'src/mcmc_prepare.R': [(1380, 'if(recent & !add.sites){'),
                           (1408, 'tann.recon.age <- (pollen_percent %*% tf.x1.current)[1,]'),
                           (1412, 'tann_recent[count] <- weightedMedian(x1range.bck,tann.recon.age)')]}
AUDIT = {
    'kind': 'determinism_audit_from_source_text_and_R_cross_check',
    'question': 'Is reconstruction_climate deterministic, and does the bound run execute it?',
    'candidates': [
        {'id': 'reconstruction_climate', 'lines': 'mcmc_functions_v1.6.R:1188-1199', 'random_state': False,
         'package_semantics_needed': True, 'package_semantics_bound': 'matrixStats::weightedMedian within 2018-04-23..2023-07-07',
         'reached_by_bound_run': False, 'decision': 'implemented as tools/historical_reconstruction_climate.py on user request; dormant'},
        {'id': 'bound_recent_climate', 'lines': 'mcmc_functions_v1.6.R:1812-1820,1850-1851', 'random_state': True,
         'evidence': 'post_tann = predictant_tann[1], reconst from rtruncnorm_cpp draws of tf_sample', 'decision': 'documented only'},
        {'id': 'candidate_loop_1394_1417', 'lines': 'mcmc_prepare.R:1380,1394-1417', 'random_state': True,
         'evidence': 'guarded by recent & !add.sites; add.sites <- T in main_v1.2.R:73; uses sample()', 'reached_by_bound_run': False,
         'decision': 'documented only'},
    ],
    'selected_for_implementation': 'reconstruction_climate', 'final_variant_selected': False, 'R_execution_for_binding': False,
}


def main():
    spec = importlib.util.spec_from_file_location('ws_bind', OUT.parent / 'literal_weights_stage_2026_11_03/bind_source.py')
    binder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(binder)
    cfg = json.loads((ROOT / '.local/my_reconst_paths.json').read_text(encoding='utf-8-sig'))
    base = Path(cfg['original_data_root'])
    diss = base.parents[1]
    lines = {}
    for rel, sha in SOURCES.items():
        raw = (base / rel).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError(f'{rel} differs from the frozen binding')
        lines[rel] = raw.decode('utf-8-sig').splitlines()
        for n, text in LITERAL[rel]:
            if lines[rel][n - 1].strip() != text:
                raise ValueError(f'{rel}:{n} no longer reads {text!r}')
    main_lines = (base / 'main_v1.2.R').read_text(encoding='utf-8-sig', errors='replace').splitlines()
    if main_lines[72].strip() != 'add.sites <- T':
        raise ValueError('main_v1.2.R:73 differs')
    fun = binder._code(lines['src/mcmc_functions_v1.6.R'])
    depth = binder._depths(fun)
    if not (depth[1187] == 0 and depth[1849] == 2):
        raise ValueError(f'brace depths differ: {depth[1187]} {depth[1849]}')
    files, calls, sampling_refs, count = [], [], [], 0
    for path in sorted(diss.rglob('*')):
        if not path.is_file() or path.suffix.lower() != '.r':
            continue
        count += 1
        text = path.read_text(encoding='utf-8', errors='replace')
        rel = path.relative_to(diss).as_posix()
        if 'mcmc_sampling_v1.4' in text:
            sampling_refs.append(rel)
        if 'reconstruction_climate' not in text:
            continue
        files.append(rel)
        for n, line in enumerate(text.splitlines(), 1):
            code = line.split('#', 1)[0]
            if re.search(r'reconstruction_climate\s*\(', code) and not re.search(r'reconstruction_climate\s*(<-|=)\s*function', code):
                calls.append({'file': rel, 'line': n, 'text': line.strip()[:140]})
    expected = [{'file': 'src/local_reconstr_all_epd/src/mcmc_sampling_v1.4.R', 'line': 79,
                 'text': 'recent_climate <- reconstruction_climate(taxa.appear.core.temp,which.taxa,tf.x1.current,tf.x2.current,recent)'}]
    if calls != expected or sampling_refs:
        raise ValueError(f'callers differ: {calls}; references to the caller file: {sampling_refs}')
    source_calls = sorted({m.group(1) for line in main_lines for m in [re.match(r'\s*source\("src/([^"]+)"\)', line)] if m})
    span = lines['src/mcmc_functions_v1.6.R'][1187:1199]
    (OUT / 'source_excerpt.json').write_text(json.dumps({'status': 'read_only_spans', 'spans': [
        {'source': 'original:src/mcmc_functions_v1.6.R', 'first_line': 1188, 'last_line': 1199, 'role': 'implemented', 'text': '\n'.join(span)}]},
        indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    (OUT / 'source_binding.json').write_text(json.dumps({
        'status': 'bounded_historical_source_fragment_not_case_reproduction', 'basis_commit': BASIS, 'sources_sha256': SOURCES,
        'implemented_span': ['src/mcmc_functions_v1.6.R', 1188, 1199], 'r_files_searched': count,
        'files_naming_the_function': files, 'callers': calls, 'files_referencing_the_caller_file': sampling_refs,
        'caller_file_sourced_by_main_v1_2': 'mcmc_sampling_v1.4.R' in source_calls, 'reached_by_bound_run': False,
        'bound_run_path': 'mcmc_functions_v1.6.R:1850-1851 (post_tann from predictant_tann[1], sampled TF values)',
        'add_sites_line': 'main_v1.2.R:73', 'guard_1380_needs_not_add_sites': True, 'lines_1408_1413_reached_by_bound_run': False,
        'globals_made_explicit': ['x1range.bck', 'x2range.bck', 'prior.recent$mean.tmp', 'prior.recent$mean.pann'],
        'which_taxa_read': False, 'portable_excerpt': 'source_excerpt.json',
        'portable_excerpt_sha256': hashlib.sha256((OUT / 'source_excerpt.json').read_bytes()).hexdigest(),
        'actual_original_inputs_used_for_numerics': False, 'run_binding': None, 'historical_reproduction_ready': False,
    }, indent=2) + '\n', encoding='utf-8', newline='\n')
    (OUT / 'determinism_audit.json').write_text(json.dumps(AUDIT, indent=2) + '\n', encoding='utf-8', newline='\n')
    print('files searched', count, 'naming', files, 'calls', calls)


if __name__ == '__main__':
    main()
