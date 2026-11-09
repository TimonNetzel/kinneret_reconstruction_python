"""Bind the sampler preparation (mcmc_conduction.R 53-79) and the determinism audit to already pinned originals.

Reads only. Each original must match the SHA-256 recorded by an earlier frozen packet before any span is copied;
the hash is checked again after reading. On the source text with comments and string literals stripped (also the
multi-line embedded Rcpp code) the script checks brace depths, every assignment of gibbs, F, T, num_taxa,
taxa.appear.core and taxa_information, that no base function used in 53-79 is redefined, and every occurrence of
the prepared fields, combs and tf_x12_mean_current, i.e. their readers. No R is executed, no variant is selected.
"""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASIS = '0a3dcdb8fde7292f6b9c1596dbade5cc7c455e34'

# source -> (sha256 from an earlier frozen binding, that binding)
SOURCES = {
    'main_v1.2.R': ('82486f99bafff08af2b51475ed225501d26cec1a0322a286b97a248ebd2f0520',
                    'kinneret_binding_2026_11_02/summary.json'),
    'src/mcmc_prepare.R': ('b897e84f8f2d72399de729592066341585974d521bf8164b185f20377ed5cc7e',
                           'literal_tf_preparation_2026_11_02/source_binding.json'),
    'src/mcmc_conduction.R': ('47c59c508e15150f7d6fda775a4f6bf6c14f0148e10cefd0014fa1ce6cfe5a9d',
                              'kinneret_references_2026_11_02/contract.json'),
    'src/mcmc_functions_v1.6.R': ('b887144370e328fa016d1834ceebcd55622816d4b36838d33a26197110ab453c',
                                  'kinneret_postprocessing_2026_11_02/contract.json'),
    'src/mcmc_trimming_posterior_trafo.R': ('5d9bc9d06ae172e64e398cff84b2a66866d9f7c4559721b7f5ebe903777a42c4',
                                            'kinneret_postprocessing_2026_11_02/contract.json'),
    # sourced by mcmc_prepare.R (3, 43) and main_v1.2.R (43) before mcmc_conduction.R runs
    'src/check_ages.R': ('c7f92e6030d83540c14f29a2e9d0ad07c74104dfe86befebda921c1cffbd2729',
                         'kinneret_binding_2026_11_02/summary.json'),
    'src/prior_climate.R': ('77cfe4e2297abf5f0dd8f7179721702bb1a15e7f9989c9c62374a2fdda642ebd',
                            'kinneret_references_2026_11_02/contract.json'),
    'src/loess_functions.R': ('371e2761eb532644da9bd7c1e9dc09c930c41ea3026fa616680a9d3000071ead',
                              'required_files_2026_11_02/source_dates.csv'),
}

# (source, first, last, role)
SPANS = [
    ('main_v1.2.R', 622, 633, 'gibbs <- F at depth 0, the only assignment of gibbs in the eight files'),
    ('main_v1.2.R', 673, 677, 'bound loop sources mcmc_prepare.R, then mcmc_conduction.R'),
    ('src/mcmc_prepare.R', 91, 100, 'all.avail.age is a numeric array; NA rows filled from the row before'),
    ('src/mcmc_prepare.R', 1450, 1454, 'taxa.appear.core <- all.avail.age[,which_taxa] (drops at one row/column)'),
    ('src/mcmc_prepare.R', 1664, 1679, 'removal: taxa.appear.core[,-which.tf.missing]; num_taxa from nrow (1678)'),
    ('src/mcmc_prepare.R', 1692, 1706, 'documented only: tf_x12_mean_current, combs with choose, posterior <- list()'),
    ('src/mcmc_conduction.R', 53, 79, 'implemented: how_many_taxa, taxa_information, column max norm, jeffreys_prior'),
    ('src/mcmc_conduction.R', 101, 101, 'posterior_temp <- posterior copies how_many_taxa and jeffreys_prior'),
    ('src/mcmc_conduction.R', 112, 114, 'reader (sampler, not run): z starts as 1:posterior$how_many_taxa'),
    ('src/mcmc_conduction.R', 130, 133, 'documented only: rmvnorm_rcpp draws (random state)'),
    ('src/mcmc_conduction.R', 138, 146, 'reader (sampler, not run): parameter_stage gets taxa_information'),
    ('src/mcmc_functions_v1.6.R', 1207, 1238, 'reader in the dead Gibbs branch: jeffreys_prior, how_many_taxa'),
    ('src/mcmc_functions_v1.6.R', 1242, 1283, 'active reader: values, ncol_taxa_array, nrow_taxa_array (1276)'),
]

AUDIT = {
    'kind': 'determinism_audit_from_source_text_no_R_execution',
    'question': 'Which next step is deterministic without random state and without unbound package semantics?',
    'candidates': [
        {'id': 'sampler_preparation_53_79', 'lines': 'mcmc_conduction.R:53-79 (+main_v1.2.R:624)',
         'random_state': False,
         'functions': ['if', 'floor', '/', 'list', 'c', 't', 'ncol', 'nrow', 'colMeans(na.rm = T)', 'for', ':',
                       '[ and [<- on a numeric matrix', 'max(na.rm = T)', 'rep'],
         'package_semantics_needed': False,
         'active_in_bound_reconstruction_loop': True,
         'evidence': 'main_v1.2.R:676 sources mcmc_conduction.R; 53, 63-79 lie at brace depth 0; gibbs is '
                     'assigned only in main_v1.2.R:624 (gibbs <- F, depth 0); F and T are never assigned; no base '
                     'function used here is redefined in the eight files',
         'precision_note': 'colMeans may accumulate in extended precision (R version and platform unbound); '
                           'taxa_means has no reader, the module flags means that are not accumulator independent',
         'readers': 'taxa_information$values, ncol_taxa_array, nrow_taxa_array: parameter_stage_weights '
                    '(mcmc_functions_v1.6.R:1276); taxa_information$taxa_means: none; posterior$jeffreys_prior and '
                    'floor(num_taxa/2): only parameter_stage_gibbs (1216, 1228), dead with gibbs <- F; '
                    'posterior$how_many_taxa: z start 1:how_many_taxa (mcmc_conduction.R:114), reset to '
                    '1:ncol in every weights stage (1280); taxa.appear.core.temp is passed in 142 but '
                    'parameter_stage_weights never reads its taxa.appear.core argument',
         'masking_note': 'like all earlier packets: the base functions are assumed not to be masked by the 50 '
                         'attached packages; their namespaces are unbound and R was not run',
         'decision': 'implemented as tools/historical_sampler_preparation.py; inputs only synthetic'},
        {'id': 'tf_x12_mean_current_1692_1700', 'lines': 'mcmc_prepare.R:1692-1700',
         'random_state': False, 'package_semantics_needed': False,
         'evidence': 'pure column binding of the two mean vectors (initialised in 1461); read by the sampler '
                     '(mcmc_conduction.R:146, argument of reconstruction) and in mcmc_functions_v1.6.R:445, 470 '
                     '(final_reconstruction) and 858 (reconstruction)',
         'decision': 'documented only; its inputs depend on weightedMedian (open)'},
        {'id': 'combs_1702_1703', 'lines': 'mcmc_prepare.R:1702-1703',
         'random_state': False, 'package_semantics_needed': False,
         'evidence': 'choose(num_taxa, c) for c in 1:num_taxa; combs has no reader in the eight files; the loop '
                     'variable c becomes an integer but R skips non-function bindings when calling c(...)',
         'decision': 'documented only, no reader'},
        {'id': 'weightedMedian', 'lines': 'mcmc_prepare.R:1631,1632,1651,1652',
         'random_state': False, 'package_semantics_needed': True,
         'evidence': 'main_v1.2.R:27 attaches matrixStats without version',
         'decision': 'documented only; the NA mask of the previous packet stays an explicit input'},
        {'id': 'rmvnorm_rcpp_133', 'lines': 'mcmc_conduction.R:133',
         'random_state': True, 'evidence': 'multivariate normal draws for the whole run, no local seed',
         'decision': 'documented only'},
        {'id': 'prior_samples_seed_2022', 'lines': 'mcmc_prepare.R:1297-1303',
         'random_state': True, 'evidence': 'set.seed(2022) before rnorm and rgamma; RNG kind unbound',
         'decision': 'documented only'},
        {'id': 'start_values_rbeta', 'lines': 'mcmc_prepare.R:1709-1738', 'random_state': True,
         'evidence': 'rbeta in the start values', 'decision': 'documented only'},
        {'id': 'tf_variance_sample', 'lines': 'mcmc_prepare.R:1591-1592', 'random_state': True,
         'evidence': 'sample(..., replace = T) without a local seed', 'decision': 'documented only'},
    ],
    'selected_for_implementation': 'sampler_preparation_53_79',
    'final_variant_selected': False,
    'R_execution': False,
}


def _code(lines):
    """Code per line without comments and string literals; a literal may span lines (embedded Rcpp code)."""
    out, quote = [], None
    for line in lines:
        code, i = [], 0
        while i < len(line):
            ch = line[i]
            if quote:
                if ch == '\\':
                    i += 1
                elif ch == quote:
                    quote = None
            elif ch in '"\'':
                quote = ch
            elif ch == '#':
                break
            else:
                code.append(ch)
            i += 1
        out.append(''.join(code))
    if quote:
        raise ValueError('unterminated string literal')
    return out


def _depths(code, strays=None):
    """Brace depth at the start of each line. With a list for strays, a closing brace below depth 0 is recorded
    there and the depth stays 0; without it, any imbalance is an error."""
    depth, out = 0, []
    for n, line in enumerate(code, 1):
        out.append(depth)
        depth += line.count('{') - line.count('}')
        if depth < 0:
            if strays is None:
                raise ValueError('unbalanced braces')
            strays.append(n)
            depth = 0
    if depth:
        raise ValueError('unbalanced braces at end of file')
    return out


def _name(name):
    """A whole R symbol: not part of a longer name, not a $ field."""
    return r'(?<![\w.$])' + re.escape(name) + r'(?![\w.])'


def _lines(code, pattern):
    return [n for n, line in enumerate(code, 1) if re.search(pattern, line)]


def main():
    cfg = json.loads((ROOT / '.local/my_reconst_paths.json').read_text(encoding='utf-8-sig'))
    base = Path(cfg['original_data_root'])
    lines, binding = {}, {}
    for rel, (sha, prior) in SOURCES.items():
        raw = (base / rel).read_bytes()
        before = hashlib.sha256(raw).hexdigest()
        if before != sha or sha not in (ROOT / 'docs/thesis_research' / prior).read_text(encoding='utf-8'):
            raise ValueError(f'{rel} differs from the earlier frozen binding {prior}')
        lines[rel] = raw.decode('utf-8-sig').splitlines()
        if hashlib.sha256((base / rel).read_bytes()).hexdigest() != before:
            raise ValueError(f'{rel} changed while reading')
        binding['original:' + rel] = {'sha256': sha, 'bytes': len(raw), 'previous_binding': prior}
    code = {rel: _code(text) for rel, text in lines.items()}
    # main_v1.2.R has three closing braces whose openings are commented out (740, 788, 829, all after the bound
    # loop 651-684); a whole-file source() would stop with a parse error, so the file was run in parts.
    # Only main_v1.2.R may contain such braces.
    strays = []
    depth = {rel: _depths(c, strays if rel == 'main_v1.2.R' else None) for rel, c in code.items()}
    if strays != [740, 788, 829]:
        raise ValueError(f'stray closing braces in main_v1.2.R differ: {strays}')

    literal = [('main_v1.2.R', 624, 'gibbs <- F'), ('main_v1.2.R', 625, 'if(gibbs){'),
               ('main_v1.2.R', 675, 'source("src/mcmc_prepare.R")'),
               ('main_v1.2.R', 676, 'source("src/mcmc_conduction.R")'),
               ('src/mcmc_prepare.R', 1454, 'taxa.appear.core <- all.avail.age[,which_taxa]'),
               ('src/mcmc_prepare.R', 1666, 'taxa.appear.core <- taxa.appear.core[,-which.tf.missing]'),
               ('src/mcmc_prepare.R', 1703, 'for(c in 1:num_taxa) combs[c] <- choose(num_taxa,c)'),
               ('src/mcmc_conduction.R', 53, 'if(gibbs){'),
               ('src/mcmc_conduction.R', 54, 'posterior$how_many_taxa <- floor(num_taxa/2)'),
               ('src/mcmc_conduction.R', 57, 'posterior$how_many_taxa <- num_taxa'),
               ('src/mcmc_conduction.R', 58, 'parameter_stage <- parameter_stage_weights'),
               ('src/mcmc_conduction.R', 64, 'taxa_information$values <- c(t(taxa.appear.core))'),
               ('src/mcmc_conduction.R', 67, 'taxa_information$taxa_means <- colMeans(taxa.appear.core, na.rm=T)'),
               ('src/mcmc_conduction.R', 70, 'for(j in 1:num_taxa){'),
               ('src/mcmc_conduction.R', 71,
                'taxa.appear.core.temp[,j] <- taxa.appear.core[,j]/max(taxa.appear.core[,j], na.rm=T)'),
               ('src/mcmc_conduction.R', 75, 'taxa_information$values <- c(t(taxa.appear.core.temp))'),
               ('src/mcmc_conduction.R', 79, 'posterior$jeffreys_prior <- rep(0.5,num_taxa)'),
               ('src/mcmc_functions_v1.6.R', 1276,
                'taxa.appear.core.temp <- matrix(taxa_information$values * new_weights,ncol = '
                'taxa_information$ncol_taxa_array,nrow=taxa_information$nrow_taxa_array,byrow=T)')]
    for rel, number, text in literal:
        if lines[rel][number - 1].strip() != text:
            raise ValueError(f'{rel}:{number} no longer reads {text!r}')
    expected_depth = {'main_v1.2.R': {624: 0, 625: 0, 676: 1},
                      'src/mcmc_conduction.R': {53: 0, 54: 1, 57: 1, 63: 0, 64: 0, 67: 0, 69: 0, 70: 0, 71: 1,
                                                75: 0, 79: 0, 101: 0, 114: 0, 142: 1},
                      'src/mcmc_functions_v1.6.R': {1216: 2, 1228: 1, 1243: 1, 1276: 1, 1280: 1}}
    for rel, wanted in expected_depth.items():
        if any(depth[rel][n - 1] != d for n, d in wanted.items()):
            raise ValueError(f'{rel}: brace depths differ from the audited control flow')

    assign = lambda name: r'(' + _name(name) + r'(\$\w+|\[[^\]]*\])*\s*<<?-|->\s*' + _name(name) + \
        r'|for\s*\(\s*' + re.escape(name) + r'\s+in\b)'
    sites = {name: {rel: _lines(c, assign(name)) for rel, c in code.items() if _lines(c, assign(name))}
             for name in ('gibbs', 'F', 'T', 'num_taxa', 'taxa_information', 'c', 't', 'max', 'colMeans', 'rep',
                          'floor', 'ncol', 'nrow', 'list')}
    expected_sites = {'gibbs': {'main_v1.2.R': [624]}, 'F': {}, 'T': {},
                      'num_taxa': {'src/mcmc_prepare.R': [1435, 1678]},
                      'taxa_information': {'src/mcmc_conduction.R': [63, 64, 65, 66, 67, 75]},
                      'c': {'src/mcmc_prepare.R': [747, 1703]}, 't': {}, 'max': {}, 'colMeans': {}, 'rep': {},
                      'floor': {}, 'ncol': {}, 'nrow': {}, 'list': {}}
    if sites != expected_sites:
        raise ValueError(f'assignment sites differ: {sites}')
    if depth['src/mcmc_prepare.R'][746] < 1 or depth['src/mcmc_prepare.R'][1702] != 0:
        raise ValueError('c in mcmc_prepare.R:747 is no longer local to a function or 1703 not at top level')
    core_top = {rel: [n for n in _lines(c, assign('taxa.appear.core')) if depth[rel][n - 1] == 0]
                for rel, c in code.items()}
    if {k: v for k, v in core_top.items() if v} != {'src/mcmc_prepare.R': [1454]}:
        raise ValueError(f'taxa.appear.core assigned at top level elsewhere: {core_top}')
    if _lines(code['src/mcmc_prepare.R'], assign('taxa.appear.core')) != [1454, 1666]:
        raise ValueError('taxa.appear.core in mcmc_prepare.R is assigned outside 1454 and 1666')

    occurrences = {}
    for label, pattern in [('gibbs', _name('gibbs')), ('taxa_information', _name('taxa_information')),
                           ('$taxa_means', r'\$taxa_means(?![\w.])'), ('jeffreys_prior', r'jeffreys_prior'),
                           ('how_many_taxa', r'how_many_taxa'), ('combs', _name('combs')),
                           ('tf_x12_mean_current', _name('tf_x12_mean_current'))]:
        occurrences[label] = {rel: _lines(c, pattern) for rel, c in code.items() if _lines(c, pattern)}
    expected_occ = {
        'gibbs': {'main_v1.2.R': [624, 625], 'src/mcmc_conduction.R': [53],
                  'src/mcmc_functions_v1.6.R': [366, 433, 2051, 2077]},
        'taxa_information': {'src/mcmc_conduction.R': [63, 64, 65, 66, 67, 75, 142],
                             'src/mcmc_functions_v1.6.R': [1207, 1242, 1276]},
        '$taxa_means': {'src/mcmc_conduction.R': [67]},
        'jeffreys_prior': {'src/mcmc_conduction.R': [79], 'src/mcmc_functions_v1.6.R': [1216]},
        'how_many_taxa': {'src/mcmc_conduction.R': [54, 57, 114], 'src/mcmc_functions_v1.6.R': [1228]},
        'combs': {'src/mcmc_prepare.R': [1702, 1703]},
    }
    tf12 = occurrences.pop('tf_x12_mean_current')
    if occurrences != expected_occ:
        raise ValueError(f'occurrences differ: {occurrences}')
    # 1461 initialises c(), 1692-1694 build it; read by the sampler call 146 and by functions 445/470/858
    if tf12 != {'src/mcmc_prepare.R': [1461, 1692, 1693, 1694], 'src/mcmc_conduction.R': [146],
                'src/mcmc_functions_v1.6.R': [445, 470, 855, 858]}:
        raise ValueError(f'tf_x12_mean_current readers differ: {tf12}')
    weights_body = code['src/mcmc_functions_v1.6.R'][1242:1283]  # lines 1243-1283
    if any(re.search(_name('taxa.appear.core'), line) for line in weights_body):
        raise ValueError('parameter_stage_weights reads its taxa.appear.core argument')
    redefined = {rel: _lines(c, r'(?<![\w.$])(c|t|max|colMeans|rep|floor|ncol|nrow|list)\s*<<?-\s*function')
                 for rel, c in code.items()}
    if any(redefined.values()):
        raise ValueError(f'base function redefined: {redefined}')

    spans = [{'source': 'original:' + rel, 'first_line': a, 'last_line': b, 'role': role,
              'text': '\n'.join(lines[rel][a - 1:b])} for rel, a, b, role in SPANS]
    (OUT / 'source_excerpt.json').write_text(json.dumps({'status': 'read_only_original_spans', 'spans': spans},
                                                        indent=2, ensure_ascii=False) + '\n',
                                             encoding='utf-8', newline='\n')
    result = {
        'status': 'bounded_historical_source_fragment_not_case_reproduction',
        'basis_commit': BASIS, 'sources': binding,
        'source_hashes_before_equal_after_and_prior': True,
        'portable_excerpt': 'source_excerpt.json',
        'portable_excerpt_sha256': hashlib.sha256((OUT / 'source_excerpt.json').read_bytes()).hexdigest(),
        'implemented_spans': {'sampler_preparation': ['src/mcmc_conduction.R', 53, 79],
                              'gibbs_switch': ['main_v1.2.R', 624, 624]},
        'brace_depth_at_line_start': {rel: {str(k): v for k, v in d.items()} for rel, d in expected_depth.items()},
        'assignment_sites_without_comments': sites,
        'taxa_appear_core_assignments_mcmc_prepare': [1454, 1666],
        'occurrences_without_comments': {**occurrences, 'tf_x12_mean_current': tf12},
        'parameter_stage_weights_reads_taxa_appear_core_argument': False,
        'base_functions_redefined': False,
        'gibbs': 'gibbs <- F at main_v1.2.R:624 (depth 0) is the only assignment; F is never reassigned: '
                 'how_many_taxa <- num_taxa (57) and parameter_stage_weights (58) are active',
        'main_stray_closing_braces': [740, 788, 829],
        'main_stray_note': 'closing braces whose openings are commented out, all after the bound loop 651-684; '
                           'a whole-file source() of main_v1.2.R would fail to parse, so it was run in parts; '
                           'this stays a run-binding question (no correction)',
        'c_loop_variable': 'mcmc_prepare.R:747 binds c locally inside a function; 1703 binds c to an integer; R skips non-function bindings when it '
                           'looks up the function in c(...) (documented R semantics, not executed)',
        'col_means': 'colMeans may accumulate in extended precision; taxa_means has no reader; means flagged',
        'excluded_context': 'sampler loop from 82 on, rmvnorm_rcpp (133), parameter stages, reconstruction, '
                            'tf_x12_mean_current, combs, weightedMedian, seeded and unseeded draws',
        'supported_domain': 'finite numeric matrix with NA, at least two rows and two columns, num_taxa equal to '
                            'ncol, no column of NA only, no negative zero',
        'actual_original_inputs_used_for_numerics': False,
        'R_or_package_runtime_equivalence_claimed': False,
        'run_binding': None, 'historical_reproduction_ready': False,
    }
    (OUT / 'source_binding.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    (OUT / 'determinism_audit.json').write_text(json.dumps(AUDIT, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(f'{len(SOURCES)} source hashes and {len(SPANS)} spans verified; depths, assignments and readers checked; '
          'originals read only')


if __name__ == '__main__':
    main()
