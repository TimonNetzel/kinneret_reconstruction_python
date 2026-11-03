"""Bind the ddirichlet ratio of parameter_stage_weights (mcmc_functions_v1.6.R 1268-1274) and its provider text.

Reads only. mcmc_functions_v1.6.R and mcmc_conduction.R must match the SHA-256 of the earlier frozen binding; the
LaplacesDemon text must equal the one recorded in r_runtime_crosscheck_2026_11_03/dirichlet_versions.json (identical
in all four window versions). The brace-depth and comment stripping rules are imported unchanged from the frozen
weights-stage binder. Writes source_binding.json, source_excerpt.json and determinism_audit.json.
"""
import hashlib
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASIS = 'c0cdf81444d1dbbabe3282fb8b1ee7ad6cae6783'
SOURCES = {'src/mcmc_functions_v1.6.R': 'b887144370e328fa016d1834ceebcd55622816d4b36838d33a26197110ab453c',
           'src/mcmc_conduction.R': '47c59c508e15150f7d6fda775a4f6bf6c14f0148e10cefd0014fa1ce6cfe5a9d'}
PROVIDER = ROOT / 'docs/thesis_research/r_runtime_crosscheck_2026_11_03/dirichlet_versions.json'
LITERAL = [
    (1243, 'old_weights <- posterior_temp$taxa_weights'),
    (1248, 'new_weights <- rdirichlet(1,old_weights*posterior_temp$dirichlet_spread)[1,]'),
    (1249, 'if (min(new_weights) > 0) {'),
    (1254, 'new_weights <- rep(1/((num_taxa)),num_taxa)'),
    (1268, '# dirichlet is not a symmetric distribution'),
    (1269, 'weights_ratio <- ddirichlet(new_weights,rep(0.5,num_taxa)) / ddirichlet(old_weights,rep(0.5,num_taxa))'),
    (1270, 'proposal_ratio <- ddirichlet(old_weights,new_weights)/ddirichlet(new_weights,old_weights)'),
    (1272, 'parameter_ratio <- proposal_ratio * weights_ratio'),
    (1273, 'if(is.na(parameter_ratio)) parameter_ratio <- 0'),
    (1274, 'posterior_temp$parameter_ratio <- parameter_ratio'),
]
AUDIT = {
    'kind': 'determinism_audit_from_source_text_no_R_execution',
    'question': 'Is the ratio 1269-1274 deterministic once the provider of ddirichlet is bound?',
    'candidates': [
        {'id': 'ddirichlet_ratio_1269_1274', 'lines': 'mcmc_functions_v1.6.R:1269-1274', 'random_state': False,
         'package_semantics_needed': True,
         'package_semantics_bound': 'LaplacesDemon::ddirichlet, first on the search path within 2018-04-23..'
                                    '2023-07-07, text identical in 16.1.0-16.1.6 (r_runtime_crosscheck_2026_11_03)',
         'functions': ['ddirichlet (LaplacesDemon): rbind, matrix(byrow=TRUE), rowSums, lgamma, log, exp', 'rep',
                       '/', '*', 'is.na'],
         'inputs': 'old_weights (1243) and new_weights (1248 or the reset 1254) as EXPLICIT inputs; num_taxa global',
         'readers': 'posterior$parameter_ratio[i] <- posterior_temp$parameter_ratio (mcmc_conduction.R:151), then '
                    'acc_prob (302), compared with runif(1) in 304',
         'precision_note': 'lgamma (R nmath), log and exp (C library) need not be correctly rounded and rowSums may '
                           'accumulate in long double; the module rounds every elementary function correctly and '
                           'measures the distance to R 3.5.0 instead of assuming bit equality',
         'decision': 'implemented as tools/historical_dirichlet_ratio.py; inputs only synthetic'},
        {'id': 'rdirichlet_1248', 'lines': 'mcmc_functions_v1.6.R:1248', 'random_state': True,
         'evidence': 'LaplacesDemon::rdirichlet draws with rgamma', 'decision': 'documented only, no seed'},
        {'id': 'reset_1253_1257', 'lines': 'mcmc_functions_v1.6.R:1253-1257', 'random_state': True,
         'evidence': 'reached after 1000 draws without min > 0', 'decision': 'documented only'},
        {'id': 'acceptance_runif_304', 'lines': 'mcmc_conduction.R:304', 'random_state': True,
         'evidence': 'runif(1) without a local seed', 'decision': 'documented only'},
    ],
    'selected_for_implementation': 'ddirichlet_ratio_1269_1274',
    'final_variant_selected': False, 'R_execution_for_binding': False,
}


def main():
    spec = importlib.util.spec_from_file_location('ws_bind', OUT.parent / 'literal_weights_stage_2026_11_03/bind_source.py')
    binder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(binder)
    cfg = json.loads((ROOT / '.local/my_reconst_paths.json').read_text(encoding='utf-8-sig'))
    base = Path(cfg['original_data_root'])
    lines = {}
    for rel, sha in SOURCES.items():
        raw = (base / rel).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError(f'{rel} differs from the frozen binding')
        lines[rel] = raw.decode('utf-8-sig').splitlines()
    fun = lines['src/mcmc_functions_v1.6.R']
    for n, text in LITERAL:
        if fun[n - 1].strip() != text:
            raise ValueError(f'mcmc_functions_v1.6.R:{n} no longer reads {text!r}')
    code = binder._code(fun)
    depth = binder._depths(code)
    expected_depth = {1269: 1, 1270: 1, 1272: 1, 1273: 1, 1274: 1}
    if {n: depth[n - 1] for n in expected_depth} != expected_depth:
        raise ValueError('brace depths of 1269-1274 differ')
    commented = [n for n in range(1262, 1267) if 'ddirichlet' in fun[n - 1] and fun[n - 1].lstrip().startswith('#')]
    if commented != [1263, 1264, 1265]:
        raise ValueError(f'commented ddirichlet alternatives differ: {commented}')
    if 'dirichlet_spread' in code[1269]:
        raise ValueError('the proposal ratio 1270 now contains dirichlet_spread')
    reader = [n for n, line in enumerate(binder._code(lines['src/mcmc_conduction.R']), 1)
              if re.search(r'parameter_ratio', line)]
    if reader != [97, 151, 302]:
        raise ValueError(f'parameter_ratio readers differ: {reader}')
    provider = json.loads(PROVIDER.read_text(encoding='utf-8'))
    if provider['distinct'] != {'rdirichlet': 1, 'ddirichlet': 1}:
        raise ValueError('LaplacesDemon text is no longer identical across the window')
    (ddirichlet_text,) = [t['text'] for t in provider['texts']['ddirichlet'].values()]
    must = ['if(!is.matrix(x)) x <- rbind(x)', 'alpha <- matrix(alpha, nrow(x), length(alpha), byrow=TRUE)',
            'if(any(rowSums(x) != 1)) x / rowSums(x)', 'if(any(x < 0)) stop("x must be non-negative.")',
            'if(any(alpha <= 0)) stop("alpha must be positive.")',
            'dens <- as.vector(lgamma(rowSums(alpha)) - rowSums(lgamma(alpha)) +',
            'rowSums((alpha-1)*log(x)))', 'if(log == FALSE) dens <- exp(dens)']
    if not all(m in ddirichlet_text for m in must):
        raise ValueError('ddirichlet text differs from the audited statements')
    spans = [{'source': 'original:src/mcmc_functions_v1.6.R', 'first_line': 1242, 'last_line': 1283,
              'role': 'parameter_stage_weights; implemented here: 1269-1274 with explicit weights',
              'text': '\n'.join(fun[1241:1283])},
             {'source': 'original:src/mcmc_conduction.R', 'first_line': 149, 'last_line': 151,
              'role': 'reader of parameter_ratio', 'text': '\n'.join(lines['src/mcmc_conduction.R'][148:151])},
             {'source': 'CRAN:LaplacesDemon 16.1.0-16.1.6 R/distributions.R', 'first_line': None, 'last_line': None,
              'role': 'ddirichlet as bound in the window (MIT licence)', 'text': ddirichlet_text}]
    (OUT / 'source_excerpt.json').write_text(json.dumps({'status': 'read_only_spans', 'spans': spans}, indent=2,
                                                        ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    result = {
        'status': 'bounded_historical_source_fragment_not_case_reproduction', 'basis_commit': BASIS,
        'sources_sha256': SOURCES, 'provider_evidence': 'r_runtime_crosscheck_2026_11_03/provider_analysis.json and '
        'dirichlet_versions.json', 'implemented_span': ['src/mcmc_functions_v1.6.R', 1269, 1274],
        'brace_depth_at_line_start': {str(k): v for k, v in expected_depth.items()},
        'commented_alternatives': [1263, 1264, 1265],
        'proposal_ratio_contains_dirichlet_spread': False,
        'num_taxa_is_global': True,
        'parameter_ratio_readers_mcmc_conduction': reader,
        'ddirichlet_normalisation_line_assigns': False,
        'portable_excerpt': 'source_excerpt.json',
        'portable_excerpt_sha256': hashlib.sha256((OUT / 'source_excerpt.json').read_bytes()).hexdigest(),
        'supported_domain': 'old and new weights: equal length num_taxa >= 2, finite doubles > 0 (the loop 1249 '
                            'needs min > 0; start weights 1/num_taxa); no NA',
        'actual_original_inputs_used_for_numerics': False, 'run_binding': None,
        'historical_reproduction_ready': False,
    }
    (OUT / 'source_binding.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    (OUT / 'determinism_audit.json').write_text(json.dumps(AUDIT, indent=2) + '\n', encoding='utf-8', newline='\n')
    print('bound 1269-1274, readers', reader, '; LaplacesDemon text checked')


if __name__ == '__main__':
    main()
