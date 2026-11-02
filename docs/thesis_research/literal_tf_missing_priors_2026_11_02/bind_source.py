"""Bind which.tf.missing, the num_taxa priors and the determinism audit to already pinned originals.

Reads only. Each original must match the SHA-256 recorded by an earlier frozen packet before any span is
copied; the hash is checked again after reading. Brace depths, the assignment sites of the objects involved and
the absence of active readers of the prior fields are checked programmatically on the source text (comments and
string literals stripped). No R is executed and no candidate variant is selected.
"""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASIS = '7d3d0f6852dbdf869b6e479d4eeffe80faaf2038'

# source -> (sha256 from an earlier frozen binding, that binding)
SOURCES = {
    'src/mcmc_prepare.R': ('b897e84f8f2d72399de729592066341585974d521bf8164b185f20377ed5cc7e',
                           'literal_tf_preparation_2026_11_02/source_binding.json'),
    'main_v1.2.R': ('82486f99bafff08af2b51475ed225501d26cec1a0322a286b97a248ebd2f0520',
                    'kinneret_binding_2026_11_02/summary.json'),
    'src/mcmc_conduction.R': ('47c59c508e15150f7d6fda775a4f6bf6c14f0148e10cefd0014fa1ce6cfe5a9d',
                              'kinneret_references_2026_11_02/contract.json'),
    'src/mcmc_functions_v1.6.R': ('b887144370e328fa016d1834ceebcd55622816d4b36838d33a26197110ab453c',
                                  'kinneret_postprocessing_2026_11_02/contract.json'),
    'src/mcmc_trimming_posterior_trafo.R': ('5d9bc9d06ae172e64e398cff84b2a66866d9f7c4559721b7f5ebe903777a42c4',
                                            'kinneret_postprocessing_2026_11_02/contract.json'),
}
LOOP_FILES = ['src/mcmc_prepare.R', 'src/mcmc_conduction.R', 'src/mcmc_functions_v1.6.R',
              'src/mcmc_trimming_posterior_trafo.R', 'main_v1.2.R']

# (source, first, last, role)
SPANS = [
    ('main_v1.2.R', 673, 677, 'bound reconstruction loop sources mcmc_prepare.R, then conduction and postprocessing'),
    ('src/mcmc_prepare.R', 1433, 1442, 'num_taxa <- ncol(all.avail) at depth 0 and the first priors incl. phi1/phi2'),
    ('src/mcmc_prepare.R', 1450, 1454, 'which_taxa <- c(1:num_taxa); taxa.appear.core <- all.avail.age[,which_taxa]'),
    ('src/mcmc_prepare.R', 1460, 1467, 'taxa_change <- FALSE at depth 0'),
    ('src/mcmc_prepare.R', 1499, 1502, 'the only taxa_change <- TRUE, behind if(any(...) & FALSE): inactive'),
    ('src/mcmc_prepare.R', 1527, 1532, 'renames colnames(all.avail) only; the column count is unchanged'),
    ('src/mcmc_prepare.R', 1572, 1582, 'else branch: TF rows follow taxa.sampled = colnames(all.avail); row norm'),
    ('src/mcmc_prepare.R', 1626, 1632, 'documented only: weightedMedian (matrixStats) produces the NA mask'),
    ('src/mcmc_prepare.R', 1663, 1686, 'implemented: which.tf.missing removal and the num_taxa priors'),
    ('src/mcmc_prepare.R', 1702, 1706, 'downstream: combs over 1:num_taxa; posterior <- list() discards 1684'),
    ('src/mcmc_prepare.R', 129, 137, 'audit candidate: default thresholds mean +/- sd'),
    ('src/mcmc_prepare.R', 186, 208, 'audit: period branch overwrites thresholds from fitdistr inputs and quantile'),
    ('src/mcmc_prepare.R', 1295, 1303, 'documented only: set.seed(2022) + rnorm/rgamma + quantile thresholds'),
    ('src/mcmc_prepare.R', 1729, 1735, 'documented only: rbeta in the recent start values'),
    ('src/mcmc_functions_v1.6.R', 319, 328, 'audit: thresholds are read through inside.range (semantics unbound)'),
    ('src/mcmc_functions_v1.6.R', 1210, 1216, 'prior$phi1 appears only in a comment; sampler uses jeffreys_prior'),
    ('src/mcmc_conduction.R', 52, 58, 'reader (sampler, not run): num_taxa'),
    ('src/mcmc_conduction.R', 110, 115, 'reader (sampler, not run): taxa weights rebuilt from num_taxa'),
]

AUDIT = {
    'kind': 'determinism_audit_from_source_text_no_R_execution',
    'question': 'Which next step is deterministic without random state and without unbound package semantics?',
    'candidates': [
        {'id': 'which_tf_missing_num_taxa_priors', 'lines': 'mcmc_prepare.R:1664-1685 (+1435-1442)',
         'random_state': False,
         'functions': ['which', 'is.na', 'length', '!=', '[ with negative index', 'nrow', 'c', ':', 'rep', '/',
                       'round'],
         'package_semantics_needed': False,
         'active_in_bound_reconstruction_loop': True,
         'evidence': 'main_v1.2.R:675 sources mcmc_prepare.R; 1664-1665 and 1681-1685 lie at brace depth 0, the '
                     'removal 1666-1678 inside the data-dependent guard length(which.tf.missing) != 0; '
                     'taxa_change is FALSE (1462) and only set TRUE behind & FALSE (1500-1501)',
         'mask': 'explicit input is.na(tf.x1.current.mean); in the source it comes from weightedMedian (1631), '
                 'which stays open; only the x1 mean is tested',
         'round_note': 'round(num_taxa/4): num_taxa/4 is an exact double; ties occur for num_taxa %% 4 == 2. '
                       'R documents IEC 60559 rounding to the even digit (?Round); R version unbound, not executed. '
                       'Half-up would differ for num_taxa %% 8 == 2. prior$nr.taxa.sd has no active reader in the '
                       'bound loop files.',
         'readers': 'num_taxa is read downstream (1703 combs, mcmc_conduction.R, mcmc_functions_v1.6.R, '
                    'postprocessing); prior$phi, phi1, phi2, taxa.weights, nr.taxa.mean and nr.taxa.sd have no '
                    'non-comment reader in the five bound loop files; posterior <- list() in 1706 discards '
                    'posterior$nr.taxa.mean[1]',
         'decision': 'implemented as tools/historical_tf_missing_priors.py; inputs only synthetic'},
        {'id': 'default_thresholds_131_137', 'lines': 'mcmc_prepare.R:131-137',
         'random_state': False, 'package_semantics_needed': True,
         'evidence': 'mean +/- sd is base arithmetic, but the period branches (200-208, 268-276, 356-364) overwrite '
                     'the same variables from fitdistr (MASS) inputs and quantile; tann_thresh/pann_thresh come '
                     'from seeded samples (1297-1303); readers use inside.range (mcmc_functions_v1.6.R:319-328), '
                     'whose semantics are unbound',
         'decision': 'not selected; weaker candidate, documented'},
        {'id': 'weightedMedian', 'lines': 'mcmc_prepare.R:1631,1632,1651,1652',
         'random_state': False, 'package_semantics_needed': True,
         'evidence': 'main_v1.2.R:27 attaches matrixStats without version',
         'decision': 'documented only; stays open in the runtime_rng bundle; the NA mask is therefore an input'},
        {'id': 'prior_samples_seed_2022', 'lines': 'mcmc_prepare.R:1297-1303',
         'random_state': True, 'seed_in_bound_sources': True,
         'evidence': 'set.seed(2022) before rnorm and rgamma; RNG kind, sample.kind and R version unbound',
         'decision': 'documented only'},
        {'id': 'start_values_rbeta', 'lines': 'mcmc_prepare.R:1709-1738',
         'random_state': True,
         'evidence': 'rbeta(1, 0.1, 20) in the start values, after posterior <- list() (1706)',
         'decision': 'documented only'},
        {'id': 'tf_variance_sample', 'lines': 'mcmc_prepare.R:1591-1592',
         'random_state': True,
         'evidence': 'sample(..., replace = T) without a local seed; overwritten by range() in 1607-1608',
         'decision': 'documented only'},
    ],
    'selected_for_implementation': 'which_tf_missing_num_taxa_priors',
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


def _depths(code):
    """Brace depth at the start of each line."""
    depth, out = 0, []
    for line in code:
        out.append(depth)
        depth += line.count('{') - line.count('}')
        if depth < 0:
            raise ValueError('unbalanced braces')
    if depth:
        raise ValueError('unbalanced braces at end of file')
    return out


def _assigned(code, name):
    pattern = re.compile(r'(^|<-|\))\s*' + re.escape(name) + r'\s*(\[[^\]]*\])?\s*<-')
    return [n for n, line in enumerate(code, 1) if pattern.search(line)]


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
    prepare = lines['src/mcmc_prepare.R']
    code = _code(prepare)
    depth = _depths(code)
    for rel, number, text in [('src/mcmc_prepare.R', 1435, 'num_taxa <- ncol(all.avail)'),
                              ('src/mcmc_prepare.R', 1462, 'taxa_change <- FALSE'),
                              ('src/mcmc_prepare.R', 1500, 'if(any(sitename == c("Birkat Ram","Dead Sea")) & FALSE){'),
                              ('src/mcmc_prepare.R', 1664, 'which.tf.missing <- which(is.na(tf.x1.current.mean))'),
                              ('src/mcmc_prepare.R', 1665, 'if(length(which.tf.missing) != 0){'),
                              ('src/mcmc_prepare.R', 1678, 'num_taxa <- nrow(tf.x1.current)'),
                              ('src/mcmc_prepare.R', 1685, 'prior$nr.taxa.sd <- round(num_taxa/4)'),
                              ('src/mcmc_prepare.R', 1706, 'posterior <- list()'),
                              ('main_v1.2.R', 675, 'source("src/mcmc_prepare.R")')]:
        if lines[rel][number - 1].strip() != text:
            raise ValueError(f'{rel}:{number} no longer reads {text!r}')
    expected_depth = {1435: 0, 1462: 0, 1500: 1, 1501: 2, 1664: 0, 1665: 0, 1666: 1, 1671: 1, 1672: 2,
                      1676: 1, 1678: 1, 1680: 0, 1681: 0, 1685: 0, 1706: 0}
    if any(depth[n - 1] != d for n, d in expected_depth.items()):
        raise ValueError('brace depths differ from the audited control flow')
    expected_assignments = {'num_taxa': [1435, 1678], 'taxa_change': [1462, 1501],
                            'all.avail': [70, 76, 83, 1676], 'taxa.appear.core': [1454, 1666],
                            'all.avail.age': [91, 93, 99, 1677], 'posterior': [1434, 1706]}
    # 70/76 prune columns before 1435, 83/93/99 assign elements; nothing resizes them between 1435 and 1664
    found = {name: _assigned(code, name) for name in expected_assignments}
    if found != expected_assignments:
        raise ValueError(f'assignment sites differ: {found}')
    if [n for n, line in enumerate(code, 1) if re.search(r'colnames\(all\.avail\)\s*\[[^\]]*\]\s*<-', line)] != [1370, 1530]:
        raise ValueError('colnames(all.avail) is renamed outside 1370 and 1530 (names only)')
    readers = {}
    for rel in LOOP_FILES:
        rel_code = _code(lines[rel])
        hits = [n for n, line in enumerate(rel_code, 1)
                if re.search(r'\bphi[12]?\b|nr\.taxa\.(mean|sd)|prior\$taxa\.weights', line)]
        readers[rel] = hits
    if readers != {'src/mcmc_prepare.R': [1437, 1438, 1439, 1440, 1441, 1442, 1682, 1683, 1684, 1685],
                   'src/mcmc_conduction.R': [], 'src/mcmc_functions_v1.6.R': [],
                   'src/mcmc_trimming_posterior_trafo.R': [], 'main_v1.2.R': []}:
        raise ValueError(f'prior fields occur outside their assignments: {readers}')
    spans = [{'source': 'original:' + rel, 'first_line': a, 'last_line': b, 'role': role,
              'text': '\n'.join(lines[rel][a - 1:b])} for rel, a, b, role in SPANS]
    excerpt = {'status': 'read_only_original_spans', 'spans': spans}
    (OUT / 'source_excerpt.json').write_text(json.dumps(excerpt, indent=2, ensure_ascii=False) + '\n',
                                             encoding='utf-8', newline='\n')
    result = {
        'status': 'bounded_historical_source_fragment_not_case_reproduction',
        'basis_commit': BASIS, 'sources': binding,
        'source_hashes_before_equal_after_and_prior': True,
        'portable_excerpt': 'source_excerpt.json',
        'portable_excerpt_sha256': hashlib.sha256((OUT / 'source_excerpt.json').read_bytes()).hexdigest(),
        'implemented_spans': {'removal_and_priors': ['src/mcmc_prepare.R', 1664, 1685],
                              'num_taxa_without_removal': ['src/mcmc_prepare.R', 1435, 1435]},
        'brace_depth_at_line_start': {str(k): v for k, v in expected_depth.items()},
        'assignment_sites_mcmc_prepare': found,
        'prior_field_occurrences_without_comments': readers,
        'taxa_change': 'FALSE at 1462; the only TRUE assignment 1501 lies behind if(... & FALSE) at 1500',
        'mask': 'explicit boolean input is.na(tf.x1.current.mean); weightedMedian (1631) is not evaluated',
        'round': 'R digits = 0 rounding to the even digit as documented (?Round, IEC 60559); not executed',
        'excluded_context': 'weightedMedian, spline expansion 1627/1647, tf_x12_mean_current 1692-1700, combs 1703, '
                            'start values with rbeta, seeded samples 1297-1303, readers in the sampler',
        'supported_domain': 'one shared taxon order, at least two remaining taxa after a removal, '
                            'no matrix with extent 1 on the other axis when a removal happens',
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
