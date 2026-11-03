"""Bind the deterministic core of parameter_stage_weights (mcmc_functions_v1.6.R 1276-1280) to pinned originals.

Reads only. Each original must match the SHA-256 recorded by an earlier frozen packet before any span is copied; the
hash is checked again after reading. On the source text with comments and string literals stripped (also the
multi-line embedded Rcpp code) the script checks brace depths, that parameter_stage_weights is defined once and
selected by gibbs <- F, every assignment and occurrence of rdirichlet, ddirichlet, new_weights, dirichlet_spread,
parameter_stage and of the base functions used in 1276-1280, the readers of the stage outputs and the order of all
library() calls. It also searches the raw text (including Rcpp string literals) for a definition of rdirichlet or
ddirichlet. No R is executed, no variant and no Dirichlet package is selected.
"""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASIS = '235f79784d01f0a613a0d3f76f6fd690c245ab21'
PRIOR = 'literal_sampler_preparation_2026_11_02/source_binding.json'

# source -> sha256 from the earlier frozen binding PRIOR (which itself names the first binding of each file)
SOURCES = {
    'main_v1.2.R': '82486f99bafff08af2b51475ed225501d26cec1a0322a286b97a248ebd2f0520',
    'src/mcmc_prepare.R': 'b897e84f8f2d72399de729592066341585974d521bf8164b185f20377ed5cc7e',
    'src/mcmc_conduction.R': '47c59c508e15150f7d6fda775a4f6bf6c14f0148e10cefd0014fa1ce6cfe5a9d',
    'src/mcmc_functions_v1.6.R': 'b887144370e328fa016d1834ceebcd55622816d4b36838d33a26197110ab453c',
    'src/mcmc_trimming_posterior_trafo.R': '5d9bc9d06ae172e64e398cff84b2a66866d9f7c4559721b7f5ebe903777a42c4',
    'src/check_ages.R': 'c7f92e6030d83540c14f29a2e9d0ad07c74104dfe86befebda921c1cffbd2729',
    'src/prior_climate.R': '77cfe4e2297abf5f0dd8f7179721702bb1a15e7f9989c9c62374a2fdda642ebd',
    'src/loess_functions.R': '371e2761eb532644da9bd7c1e9dc09c930c41ea3026fa616680a9d3000071ead',
}

# (source, first, last, role)
SPANS = [
    ('main_v1.2.R', 1, 62, 'library() calls before the bound loop (attach order); no version is recorded'),
    ('main_v1.2.R', 622, 633, 'gibbs <- F at depth 0; sample_length and burn_in'),
    ('src/mcmc_prepare.R', 1865, 1868, 'posterior$dirichlet_spread <- num_taxa^2 (start of the proposal spread)'),
    ('src/mcmc_prepare.R', 1884, 1887, 'library(npreg), library(truncnorm) attached inside the bound loop'),
    ('src/mcmc_conduction.R', 53, 79, 'parameter_stage <- parameter_stage_weights; taxa_information prepared (state 5)'),
    ('src/mcmc_conduction.R', 109, 114, 'start weights rep(1/num_taxa, num_taxa) and z = 1:how_many_taxa'),
    ('src/mcmc_conduction.R', 138, 146, 'call of the stage in every iteration; reader of z (146)'),
    ('src/mcmc_conduction.R', 301, 306, 'acceptance with runif (random state) reads taxa_weights and z'),
    ('src/mcmc_conduction.R', 328, 330, 'rejection restores taxa_weights and z of the previous iteration'),
    ('src/mcmc_conduction.R', 353, 355, 'stored taxa_weights and z'),
    ('src/mcmc_functions_v1.6.R', 1207, 1238, 'dead Gibbs stage: SBMSplitMerge::rdirichlet, rcat (gibbs <- F)'),
    ('src/mcmc_functions_v1.6.R', 1242, 1283, 'parameter_stage_weights; implemented: 1259 and 1276-1280'),
    ('src/mcmc_functions_v1.6.R', 1781, 1781, 'last definition of reconstruction (active)'),
    ('src/mcmc_functions_v1.6.R', 1816, 1820, 'reader: reconst <- taxa.appear.core[reconst_ages,] %*% tf_sample'),
    ('src/mcmc_trimming_posterior_trafo.R', 414, 417, 'library(colorRamps) after the sampler, inside the bound loop'),
]

AUDIT = {
    'kind': 'determinism_audit_from_source_text_no_R_execution',
    'question': 'Which next step is deterministic without random state and without unbound package semantics?',
    'candidates': [
        {'id': 'weights_stage_core_1276_1280', 'lines': 'mcmc_functions_v1.6.R:1259, 1276-1280',
         'random_state': False,
         'functions': ['$ on a list', '*', 'matrix(ncol, nrow, byrow = T)', 'rowSums(na.rm = T)', '/', 'ncol', ':',
                       '$<- on a list'],
         'package_semantics_needed': False,
         'active_in_bound_reconstruction_loop': True,
         'inputs': 'taxa_information$values, ncol_taxa_array, nrow_taxa_array (prepared in mcmc_conduction.R 63-75, '
                   'app state 5) and new_weights from line 1248 or 1254, taken here as an EXPLICIT input',
         'evidence': 'parameter_stage_weights is defined once (1242); gibbs <- F (main_v1.2.R:624) selects it in '
                     'mcmc_conduction.R:58; it is called in every iteration (142); 1259 and 1276-1280 lie at brace '
                     'depth 1 of the function body, i.e. run unconditionally after the while loop; no base function '
                     'used here is redefined in the eight files',
         'precision_note': 'rowSums may accumulate in extended precision (R version and platform unbound); the '
                           'module flags every row whose sum is not accumulator independent; the products and the '
                           'quotients are single IEEE double operations',
         'readers': 'posterior_temp$taxa.appear.core.temp: reconstruction (last definition 1781) reads it in 1817 '
                    '(reconst <- taxa.appear.core[reconst_ages,] %*% posterior_temp$tf_sample); posterior_temp$z: '
                    'tf.x1.current[z,], tf.x2.current[z,], tf_x12_mean_current[z,] in mcmc_conduction.R:146, stored '
                    'in 306/355; posterior_temp$taxa_weights: acceptance 305/354, previous value restored in 329',
         'masking_note': 'like all earlier packets: base functions are assumed not to be masked by the attached '
                         'packages; their namespaces are unbound and R was not run',
         'decision': 'implemented as tools/historical_weights_stage.py with explicit weights; inputs only synthetic'},
        {'id': 'rdirichlet_1248', 'lines': 'mcmc_functions_v1.6.R:1246-1251', 'random_state': True,
         'package_semantics_needed': True,
         'evidence': 'unqualified rdirichlet(1, old_weights*dirichlet_spread)[1,]; no definition in the eight files '
                     '(also none in the Rcpp string literals); the provider is the first attached package on the '
                     'search path that exports rdirichlet, which the bound sources cannot show (no version, no '
                     'namespace file); SBMSplitMerge::rdirichlet in 1216 is qualified and lies in the dead Gibbs '
                     'stage, so it is no evidence for 1248',
         'decision': 'documented only, no package chosen, no seed'},
        {'id': 'reset_after_1000_draws_1253_1257', 'lines': 'mcmc_functions_v1.6.R:1253-1257',
         'random_state': True, 'package_semantics_needed': True,
         'evidence': 'reached only after 1000 draws of rdirichlet without min(new_weights) > 0; then '
                     'new_weights <- rep(1/num_taxa, num_taxa) with the global num_taxa and dirichlet_spread doubled',
         'decision': 'documented only; the reset weights can be given to the core as explicit weights'},
        {'id': 'ddirichlet_ratio_1269_1274', 'lines': 'mcmc_functions_v1.6.R:1269-1274',
         'random_state': False, 'package_semantics_needed': True,
         'evidence': 'ddirichlet is unqualified and not defined in the eight files; is.na in 1273 maps NA and NaN '
                     'ratios to 0; the proposal ratio 1270 omits dirichlet_spread (already recorded in '
                     'LOCAL_SOURCE_CROSSWALK.md)',
         'decision': 'documented only'},
        {'id': 'sampler_initialisation_109_135', 'lines': 'mcmc_conduction.R:109-135 without 118 and 133',
         'random_state': False, 'package_semantics_needed': False,
         'evidence': 'start weights, z, fill ids and checking_intervals; 118 needs prior.recent (fitdistr) and 133 '
                     'draws rmvnorm_rcpp',
         'decision': 'alternative for a later packet'},
        {'id': 'tf_x12_mean_current_1692_1700', 'lines': 'mcmc_prepare.R:1692-1700',
         'random_state': False, 'package_semantics_needed': False,
         'evidence': 'pure column binding; inputs depend on weightedMedian', 'decision': 'documented only'},
        {'id': 'weightedMedian', 'lines': 'mcmc_prepare.R:1631,1632,1651,1652',
         'random_state': False, 'package_semantics_needed': True,
         'evidence': 'main_v1.2.R:27 attaches matrixStats without version', 'decision': 'open'},
        {'id': 'acceptance_runif_304', 'lines': 'mcmc_conduction.R:304', 'random_state': True,
         'evidence': 'runif(1) without a local seed', 'decision': 'documented only'},
        {'id': 'rmvnorm_rcpp_133', 'lines': 'mcmc_conduction.R:133', 'random_state': True,
         'evidence': 'multivariate normal draws for the whole run', 'decision': 'documented only'},
        {'id': 'prior_samples_seed_2022', 'lines': 'mcmc_prepare.R:1297-1303', 'random_state': True,
         'evidence': 'set.seed(2022) before rnorm and rgamma; RNG kind unbound', 'decision': 'documented only'},
        {'id': 'start_values_rbeta', 'lines': 'mcmc_prepare.R:1709-1738', 'random_state': True,
         'evidence': 'rbeta in the start values', 'decision': 'documented only'},
        {'id': 'tf_variance_sample', 'lines': 'mcmc_prepare.R:1591-1592', 'random_state': True,
         'evidence': 'sample(..., replace = T) without a local seed', 'decision': 'documented only'},
    ],
    'selected_for_implementation': 'weights_stage_core_1276_1280',
    'dirichlet_provider_bound': False,
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
    """Brace depth at the start of each line; stray closing braces only where a list is given."""
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


def _assign(name):
    return (r'(' + _name(name) + r'(\$\w+|\[[^\]]*\])*\s*<<?-|->\s*' + _name(name) + r'|for\s*\(\s*'
            + re.escape(name) + r'\s+in\b)')


def main():
    cfg = json.loads((ROOT / '.local/my_reconst_paths.json').read_text(encoding='utf-8-sig'))
    base = Path(cfg['original_data_root'])
    prior_text = (ROOT / 'docs/thesis_research' / PRIOR).read_text(encoding='utf-8')
    lines, raw_lines, binding = {}, {}, {}
    for rel, sha in SOURCES.items():
        raw = (base / rel).read_bytes()
        before = hashlib.sha256(raw).hexdigest()
        if before != sha or sha not in prior_text:
            raise ValueError(f'{rel} differs from the earlier frozen binding {PRIOR}')
        lines[rel] = raw.decode('utf-8-sig').splitlines()
        if hashlib.sha256((base / rel).read_bytes()).hexdigest() != before:
            raise ValueError(f'{rel} changed while reading')
        binding['original:' + rel] = {'sha256': sha, 'bytes': len(raw), 'previous_binding': PRIOR}
    code = {rel: _code(text) for rel, text in lines.items()}
    strays = []
    depth = {rel: _depths(c, strays if rel == 'main_v1.2.R' else None) for rel, c in code.items()}
    if strays != [740, 788, 829]:
        raise ValueError(f'stray closing braces in main_v1.2.R differ: {strays}')

    literal = [
        ('main_v1.2.R', 624, 'gibbs <- F'),
        ('src/mcmc_prepare.R', 1867, 'posterior$dirichlet_spread <- num_taxa^2'),
        ('src/mcmc_conduction.R', 58, 'parameter_stage <- parameter_stage_weights'),
        ('src/mcmc_conduction.R', 113,
         'posterior_temp$taxa_weights <- posterior$taxa_weights[[1]] <-  rep(1/((num_taxa)),num_taxa)'),
        ('src/mcmc_conduction.R', 142, 'posterior_temp <- parameter_stage(posterior_temp=posterior_temp,'
         'taxa.appear.core=taxa.appear.core.temp,taxa_information=taxa_information)'),
        ('src/mcmc_functions_v1.6.R', 1242, 'parameter_stage_weights <- function(posterior_temp=posterior_temp,'
         'taxa.appear.core=taxa.appear.core,taxa_information=taxa_information){'),
        ('src/mcmc_functions_v1.6.R', 1243, 'old_weights <- posterior_temp$taxa_weights'),
        ('src/mcmc_functions_v1.6.R', 1248,
         'new_weights <- rdirichlet(1,old_weights*posterior_temp$dirichlet_spread)[1,]'),
        ('src/mcmc_functions_v1.6.R', 1249, 'if (min(new_weights) > 0) {'),
        ('src/mcmc_functions_v1.6.R', 1253, 'if(counts >= 1000){'),
        ('src/mcmc_functions_v1.6.R', 1254, 'new_weights <- rep(1/((num_taxa)),num_taxa)'),
        ('src/mcmc_functions_v1.6.R', 1255,
         'posterior_temp$dirichlet_spread <- posterior_temp$dirichlet_spread*2'),
        ('src/mcmc_functions_v1.6.R', 1259, 'posterior_temp$taxa_weights <- new_weights'),
        ('src/mcmc_functions_v1.6.R', 1269,
         'weights_ratio <- ddirichlet(new_weights,rep(0.5,num_taxa)) / ddirichlet(old_weights,rep(0.5,num_taxa))'),
        ('src/mcmc_functions_v1.6.R', 1270,
         'proposal_ratio <- ddirichlet(old_weights,new_weights)/ddirichlet(new_weights,old_weights)'),
        ('src/mcmc_functions_v1.6.R', 1273, 'if(is.na(parameter_ratio)) parameter_ratio <- 0'),
        ('src/mcmc_functions_v1.6.R', 1276,
         'taxa.appear.core.temp <- matrix(taxa_information$values * new_weights,ncol = '
         'taxa_information$ncol_taxa_array,nrow=taxa_information$nrow_taxa_array,byrow=T)'),
        ('src/mcmc_functions_v1.6.R', 1277,
         'posterior_temp$taxa.appear.core.temp <- taxa.appear.core.temp / rowSums(taxa.appear.core.temp, na.rm=T)'),
        ('src/mcmc_functions_v1.6.R', 1280, 'posterior_temp$z <- 1:ncol(taxa.appear.core.temp)'),
        ('src/mcmc_functions_v1.6.R', 1282, 'return(posterior_temp)'),
        ('src/mcmc_functions_v1.6.R', 1817, 'taxa.appear.core <- posterior_temp$taxa.appear.core.temp'),
        ('src/mcmc_functions_v1.6.R', 1818, 'reconst <- taxa.appear.core[reconst_ages,] %*% posterior_temp$tf_sample'),
    ]
    for rel, number, text in literal:
        if lines[rel][number - 1].strip() != text:
            raise ValueError(f'{rel}:{number} no longer reads {text!r}')
    expected_depth = {'main_v1.2.R': {624: 0, 676: 1},
                      'src/mcmc_prepare.R': {1867: 0, 1885: 0, 1886: 0},
                      'src/mcmc_conduction.R': {58: 1, 113: 0, 142: 1, 146: 1, 304: 1, 306: 2, 330: 2, 355: 1},
                      'src/mcmc_functions_v1.6.R': {1242: 0, 1243: 1, 1248: 2, 1254: 3, 1259: 1, 1269: 1, 1273: 1,
                                                    1276: 1, 1277: 1, 1280: 1, 1282: 1, 1283: 1, 1781: 0, 1817: 1},
                      'src/mcmc_trimming_posterior_trafo.R': {416: 0}}
    found_depth = {rel: {n: depth[rel][n - 1] for n in wanted} for rel, wanted in expected_depth.items()}
    if found_depth != expected_depth:
        raise ValueError(f'brace depths differ from the audited control flow: {found_depth}')

    names = ('parameter_stage_weights', 'parameter_stage', 'reconstruction', 'rdirichlet', 'ddirichlet',
             'new_weights', 'old_weights', 'matrix', 'rowSums', 'ncol', 'nrow', 'min', 'rep', 'is.na', 'gibbs', 'F',
             'T', 'num_taxa')
    sites = {name: {rel: _lines(c, _assign(name)) for rel, c in code.items() if _lines(c, _assign(name))}
             for name in names}
    expected_sites = {
        'parameter_stage_weights': {'src/mcmc_functions_v1.6.R': [1242]},
        'parameter_stage': {'src/mcmc_conduction.R': [55, 58]},
        'reconstruction': {'src/mcmc_functions_v1.6.R': [855, 1781]},
        'rdirichlet': {}, 'ddirichlet': {},
        'new_weights': {'src/mcmc_functions_v1.6.R': [1248, 1254]},
        'old_weights': {'src/mcmc_functions_v1.6.R': [1243]},
        'matrix': {}, 'rowSums': {}, 'ncol': {}, 'nrow': {}, 'min': {}, 'rep': {}, 'is.na': {},
        'gibbs': {'main_v1.2.R': [624]}, 'F': {}, 'T': {},
        'num_taxa': {'src/mcmc_prepare.R': [1435, 1678]},
    }
    if sites != expected_sites:
        raise ValueError(f'assignment sites differ: {sites}')
    # a definition hidden in a string literal (Rcpp) or via assign() would escape the stripped code
    raw_defs = {rel: _lines(text, r'(rdirichlet|ddirichlet)\s*(<<?-|=)\s*function|'
                                  r'\b\w+\s+(rdirichlet|ddirichlet)\s*\(|assign\s*\(\s*["\'](rdirichlet|ddirichlet)')
                for rel, text in lines.items()}
    if any(raw_defs.values()):
        raise ValueError(f'rdirichlet or ddirichlet is defined in the sources: {raw_defs}')

    occurrences = {}
    for label, pattern in [('rdirichlet', r'(?<![\w.$])rdirichlet(?![\w.])'),
                           ('ddirichlet', r'(?<![\w.$])ddirichlet(?![\w.])'),
                           ('SBMSplitMerge::', r'SBMSplitMerge::'),
                           ('dirichlet_spread', r'dirichlet_spread'),
                           ('$taxa.appear.core.temp', r'\$taxa\.appear\.core\.temp(?![\w.])'),
                           ('$z', r'\$z(?![\w.])'),
                           ('$taxa_weights', r'\$taxa_weights(?![\w.])'),
                           ('parameter_stage(', _name('parameter_stage') + r'\s*\(')]:
        occurrences[label] = {rel: _lines(c, pattern) for rel, c in code.items() if _lines(c, pattern)}
    expected_occ = {
        'rdirichlet': {'src/mcmc_functions_v1.6.R': [1216, 1248]},
        'ddirichlet': {'src/mcmc_functions_v1.6.R': [1269, 1270]},
        'SBMSplitMerge::': {'src/mcmc_functions_v1.6.R': [1216, 1228]},
        'dirichlet_spread': {'src/mcmc_prepare.R': [1867], 'src/mcmc_functions_v1.6.R': [1248, 1255]},
        '$taxa.appear.core.temp': {'src/mcmc_functions_v1.6.R': [1232, 1277, 1817]},
        '$z': {'src/mcmc_conduction.R': [112, 114, 146, 306, 330, 355],
               'src/mcmc_functions_v1.6.R': [1211, 1228, 1229, 1231, 1280]},
        '$taxa_weights': {'src/mcmc_conduction.R': [112, 113, 305, 329, 354],
                          'src/mcmc_trimming_posterior_trafo.R': [190, 199, 550],
                          'src/mcmc_functions_v1.6.R': [1221, 1228, 1243, 1259]},
        'parameter_stage(': {'src/mcmc_conduction.R': [142]},
    }
    if occurrences != expected_occ:
        raise ValueError(f'occurrences differ: {occurrences}')
    redefined = {rel: _lines(c, r'(?<![\w.$])(matrix|rowSums|ncol|nrow|min|rep|is\.na)\s*<<?-\s*function')
                 for rel, c in code.items()}
    if any(redefined.values()):
        raise ValueError(f'base function redefined: {redefined}')
    # body 1243-1283 reads num_taxa as a free (global) variable and never its taxa.appear.core argument
    body = code['src/mcmc_functions_v1.6.R'][1242:1283]
    free_num_taxa = [1243 + k for k, line in enumerate(body) if re.search(_name('num_taxa'), line)]
    if free_num_taxa != [1254, 1269]:
        raise ValueError(f'num_taxa in the weights stage differs: {free_num_taxa}')
    if any(re.search(_name('taxa.appear.core'), line) for line in body):
        raise ValueError('parameter_stage_weights reads its taxa.appear.core argument')

    attach = []
    order = [('main_v1.2.R', range(1, 651)), ('src/mcmc_prepare.R', range(1, len(code['src/mcmc_prepare.R']) + 1)),
             ('src/mcmc_trimming_posterior_trafo.R',
              range(1, len(code['src/mcmc_trimming_posterior_trafo.R']) + 1))]
    for rel, numbers in order:
        for n in numbers:
            for package in re.findall(r'(?<![\w.])library\(\s*([\w.]+)\s*\)', code[rel][n - 1]):
                attach.append({'source': rel, 'line': n, 'package': package, 'depth': depth[rel][n - 1]})
    other = {rel: _lines(c, r'(?<![\w.])(library|require|requireNamespace|attachNamespace)\s*\(')
             for rel, c in code.items() if rel not in ('main_v1.2.R', 'src/mcmc_prepare.R',
                                                       'src/mcmc_trimming_posterior_trafo.R')}
    if any(other.values()):
        raise ValueError(f'unexpected attach calls: {other}')
    packages = [a['package'] for a in attach]
    if len(packages) != len(set(packages)):
        raise ValueError(f'a package is attached twice: {packages}')

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
        'implemented_spans': {'store_weights': ['src/mcmc_functions_v1.6.R', 1259, 1259],
                              'weights_stage_core': ['src/mcmc_functions_v1.6.R', 1276, 1280]},
        'brace_depth_at_line_start': {rel: {str(k): v for k, v in d.items()} for rel, d in expected_depth.items()},
        'assignment_sites_without_comments': sites,
        'rdirichlet_or_ddirichlet_defined_in_sources_incl_string_literals': False,
        'occurrences_without_comments': occurrences,
        'weights_stage_free_num_taxa_lines': free_num_taxa,
        'parameter_stage_weights_reads_taxa_appear_core_argument': False,
        'base_functions_redefined': False,
        'library_calls_in_run_order': attach,
        'library_note': 'main_v1.2.R 1-62 run before the bound loop 651-684; mcmc_prepare.R 1885-1886 and '
                        'mcmc_trimming_posterior_trafo.R 416 run inside it (the latter only after the sampler of '
                        'the first site). R searches the most recently attached package first; which of these '
                        'packages exports rdirichlet or ddirichlet is not recorded in the bound sources (no '
                        'version, no namespace file), so the provider of 1248/1269-1270 stays unbound',
        'dirichlet_provider_bound': False,
        'main_stray_closing_braces': [740, 788, 829],
        'rejection_note': 'on rejection 329-330 restore taxa_weights and z, not taxa.appear.core.temp; the next '
                          'iteration overwrites it in 1277 before reconstruction reads it in 1817',
        'row_sums': 'rowSums may accumulate in extended precision; rows whose sum is not accumulator independent '
                    'are flagged',
        'excluded_context': 'rdirichlet (1248), the while loop and reset (1246-1258), ddirichlet and the ratio '
                            '(1269-1274), acceptance (runif 304), reconstruction, sampler initialisation, '
                            'weightedMedian, seeded and unseeded draws',
        'supported_domain': 'values from the prepared taxa_information (finite, NA, NaN, -Inf; no +Inf, no '
                            'negative zero), at least two rows and two columns, length nrow*ncol, explicit weights: '
                            'one finite positive double per column',
        'actual_original_inputs_used_for_numerics': False,
        'R_or_package_runtime_equivalence_claimed': False,
        'run_binding': None, 'historical_reproduction_ready': False,
    }
    (OUT / 'source_binding.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    (OUT / 'determinism_audit.json').write_text(json.dumps(AUDIT, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(f'{len(SOURCES)} source hashes and {len(SPANS)} spans verified; depths, assignments, readers and '
          f'{len(attach)} library calls checked; originals read only')


if __name__ == '__main__':
    main()
