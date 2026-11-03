"""Bind the gamma prior grid and the determinism audit to already pinned originals.

Reads only. Each original must match the SHA-256 recorded by an earlier frozen packet before any span is
copied; the hash is checked again after reading. The four active grid blocks of mcmc_prepare.R and the
mcmc_prep_v1.2.R copy are compared programmatically. No R is executed and no candidate variant is selected.
"""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASIS = '94f81cc31e2ea14aaa7800c8b557200b0d3a9319'

# source -> (sha256 from an earlier frozen binding, that binding)
SOURCES = {
    'src/mcmc_prepare.R': ('b897e84f8f2d72399de729592066341585974d521bf8164b185f20377ed5cc7e',
                           'literal_tf_preparation_2026_11_02/source_binding.json'),
    'src/mcmc_prep_v1.2.R': ('badabd80978a074f42c25e06f1d254fd4ac8deef4e8f31bc591af6de65123949',
                             'kinneret_references_2026_11_02/contract.json'),
    'main_v1.2.R': ('82486f99bafff08af2b51475ed225501d26cec1a0322a286b97a248ebd2f0520',
                    'kinneret_binding_2026_11_02/summary.json'),
    'src/prior_climate.R': ('77cfe4e2297abf5f0dd8f7179721702bb1a15e7f9989c9c62374a2fdda642ebd',
                            'kinneret_references_2026_11_02/contract.json'),
    'src/mcmc_conduction.R': ('47c59c508e15150f7d6fda775a4f6bf6c14f0148e10cefd0014fa1ce6cfe5a9d',
                              'kinneret_references_2026_11_02/contract.json'),
    'src/mcmc_functions_v1.6.R': ('b887144370e328fa016d1834ceebcd55622816d4b36838d33a26197110ab453c',
                                  'kinneret_postprocessing_2026_11_02/contract.json'),
    'src/mcmc_trimming_posterior_trafo.R': ('5d9bc9d06ae172e64e398cff84b2a66866d9f7c4559721b7f5ebe903777a42c4',
                                            'kinneret_postprocessing_2026_11_02/contract.json'),
}

# (source, first, last, role)
SPANS = [
    ('main_v1.2.R', 651, 677, 'bound reconstruction loop: source("src/mcmc_prepare.R") without a grid switch'),
    ('src/mcmc_prepare.R', 43, 43, 'prior_climate.R sourced at depth 0; it creates prior.recent'),
    ('src/prior_climate.R', 128, 130, 'prior.recent from fitdistr estimates; floors sd.tmp >= 2 and sd.pann >= 150'),
    ('src/mcmc_prepare.R', 114, 128, 'implemented: default grid at depth 0 and the four period shape/rate copies'),
    ('src/mcmc_prepare.R', 213, 219, 'commented copy (also 281-287, 369-375): inactive'),
    ('src/mcmc_prepare.R', 393, 413, 'conditional copy for Birkat Ram inside periods_check[4]; same inputs as 114-121'),
    ('src/mcmc_prepare.R', 466, 470, 'stad_interstad <- FALSE at depth 0, so its else branch 524-545 is active'),
    ('src/mcmc_prepare.R', 524, 545, 'implemented: grid copy 526-531 and diff_rate <- shape/10'),
    ('src/mcmc_prepare.R', 1261, 1281, 'implemented: variance doubling if(recent & min(age) >= 2000), then grid at depth 0'),
    ('src/mcmc_prepare.R', 1295, 1303, 'documented only: set.seed(2022) + rnorm/rgamma + quantile read shape/rate'),
    ('src/mcmc_prepare.R', 1729, 1735, 'implemented: if(recent) rate <- shape/prior.recent$mean.pann'),
    ('src/mcmc_prepare.R', 1622, 1660, 'audit: weightedMedian fills tf.x1/x2.current.mean (matrixStats, version unbound)'),
    ('src/mcmc_prepare.R', 1663, 1686, 'audit: which.tf.missing removal and the num_taxa-derived priors that follow'),
    ('src/mcmc_prep_v1.2.R', 1102, 1116, 'source copy with upper bound 5000 instead of 50000; not sourced by main_v1.2.R'),
    ('src/mcmc_conduction.R', 153, 171, 'reader (sampler, not run): dgamma(..., shape=shape, rate=rate) and period copies'),
    ('src/mcmc_functions_v1.6.R', 227, 248, 'reader (energy, not run): dgamma with shape/rate, period copies and diff_rate'),
    ('src/mcmc_trimming_posterior_trafo.R', 635, 638, 'reader (postprocessing, not run): dgamma prior density'),
]

AUDIT = {
    'kind': 'determinism_audit_from_source_text_no_R_execution',
    'question': 'Which next step is deterministic without random state and without unbound package semantics?',
    'candidates': [
        {'id': 'gamma_prior_grid', 'lines': 'mcmc_prepare.R:114-121,1264-1281 (+526-536, 1733)',
         'random_state': False,
         'functions': ['seq', '/', 'sqrt', '-', 'abs', 'which.min', '[', '*', '&', 'min'],
         'package_semantics_needed': False,
         'active_in_bound_reconstruction_loop': True,
         'evidence': 'main_v1.2.R:675 sources mcmc_prepare.R; 115-121, 526-531 (else of stad_interstad <- FALSE, 468) '
                     'and 1270-1281 lie at brace depth 0; readers in mcmc_conduction.R:156-170, '
                     'mcmc_functions_v1.6.R:229-246 and mcmc_trimming_posterior_trafo.R:637',
         'version_note': 'seq.default with upper 50000 is invariant to the +1e-10 fuzz and pmin(x, to): '
                         '(50000-0.1)/0.1 is exactly 499999 and the last value exactly 50000. '
                         'The mcmc_prep_v1.2.R copy (upper 5000) depends on both; R version unbound.',
         'decision': 'implemented as tools/historical_gamma_prior_grid.py; inputs only synthetic'},
        {'id': 'which_tf_missing', 'lines': 'mcmc_prepare.R:1664-1679',
         'random_state': False, 'package_semantics_needed': False,
         'active_in_bound_reconstruction_loop': True,
         'evidence': 'Mechanics are base R (which, is.na, negative indexing with drop), but the NA mask is produced '
                     'by weightedMedian in 1631/1651, the column drop to a vector for one remaining taxon and '
                     'taxa_change are further R semantics',
         'decision': 'deterministic with an explicit mask; not selected in this packet, because its only trigger '
                     'is the open weightedMedian output; documented for a later packet'},
        {'id': 'weightedMedian', 'lines': 'mcmc_prepare.R:1631,1632,1651,1652',
         'random_state': False, 'package_semantics_needed': True,
         'evidence': 'main_v1.2.R:27 attaches matrixStats without version',
         'decision': 'documented only; stays open in the runtime_rng bundle'},
        {'id': 'prior_samples_seed_2022', 'lines': 'mcmc_prepare.R:1297-1303',
         'random_state': True, 'seed_in_bound_sources': True,
         'evidence': 'set.seed(2022) before rnorm and rgamma(shape, rate); RNG kind, sample.kind and R version unbound',
         'decision': 'documented only; module reports the rate passed to rgamma, draws nothing'},
        {'id': 'recent_start_value', 'lines': 'mcmc_prepare.R:1733-1734',
         'random_state': True,
         'evidence': '1733 rate <- shape/mean.pann is deterministic; 1734 adds rbeta(1, 0.1, 20) to the start value',
         'decision': '1733 implemented behind the explicit recent flag; 1734 documented only'},
    ],
    'selected_for_implementation': 'gamma_prior_grid',
    'final_variant_selected': False,
    'R_execution': False,
}


def _block(lines, first, last):
    """Grid statements without indentation, comments, blank lines and variable names."""
    names = {'shape_temp': 'S', 'shape1': 'S', 'rate_temp1': 'R1', 'rate1': 'R1', 'rate_temp2': 'R2', 'rate2': 'R2'}
    out = []
    for line in lines[first - 1:last]:
        code = line.split('#', 1)[0].strip()
        if code:
            out.append(re.sub(r'\b(shape_temp|shape1|rate_temp1|rate1|rate_temp2|rate2)\b', lambda m: names[m[1]], code))
    return out


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
    default = _block(prepare, 115, 121)
    if _block(prepare, 1270, 1281) != default or _block(prepare, 402, 408) != default:
        raise ValueError('grid blocks 1270-1281 or 402-408 no longer equal 115-121')
    if _block(prepare, 526, 531) != default[:-1]:
        raise ValueError('grid block 526-531 no longer equals 115-120')
    copy = _block(lines['src/mcmc_prep_v1.2.R'], 1103, 1114)
    if [c.replace('5000,', '50000,') for c in copy] != default or copy[0] != 'S <- seq(0.1,5000,by=0.1)':
        raise ValueError('mcmc_prep_v1.2.R grid copy differs beyond the upper bound 5000')
    for rel, number, text in [('src/mcmc_prepare.R', 468, 'stad_interstad <- FALSE'),
                              ('src/mcmc_prepare.R', 1264, 'if( recent & min(age) >= 2000 ){'),
                              ('src/mcmc_prepare.R', 1733, 'rate <- shape/prior.recent$mean.pann'),
                              ('src/mcmc_prepare.R', 536, 'diff_rate <- shape/10'),
                              ('main_v1.2.R', 675, 'source("src/mcmc_prepare.R")')]:
        if lines[rel][number - 1].strip() != text:
            raise ValueError(f'{rel}:{number} no longer reads {text!r}')
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
        'implemented_spans': {'default_grid': ['src/mcmc_prepare.R', 114, 128],
                              'diff_rate': ['src/mcmc_prepare.R', 526, 536],
                              'doubling_and_final_grid': ['src/mcmc_prepare.R', 1264, 1281],
                              'recent_start_rate': ['src/mcmc_prepare.R', 1733, 1733]},
        'block_identity': {'mcmc_prepare.R:1270-1281': 'equal to 115-121 up to variable names',
                           'mcmc_prepare.R:402-408': 'equal to 115-121 (conditional, Birkat Ram)',
                           'mcmc_prepare.R:526-531': 'equal to 115-120 (rate not assigned)',
                           'mcmc_prep_v1.2.R:1103-1114': 'equal up to variable names except upper bound 5000'},
        'site_conditions': 'recent and min(age) >= 2000 are explicit booleans; age/site.age is not evaluated',
        'grid_upper_bound': 'explicit input; 50000 for mcmc_prepare.R (sourced by main_v1.2.R:675), 5000 for mcmc_prep_v1.2.R',
        'excluded_context': 'prior.recent fit (fitdistr, MASS), seeded rnorm/rgamma 1297-1303, rbeta in 1734, '
                            'dgamma readers in sampler and postprocessing, weightedMedian, which.tf.missing',
        'supported_domain': 'finite positive scalar mean.pann, sd.pann and sd.tmp; no vector recycling or NA',
        'IEEE_double_bit_equality_with_independent_reference': True,
        'R_seq_version_independence': 'claimed only for upper 50000 (fuzz and pmin inactive); not for 5000',
        'actual_original_inputs_used_for_numerics': False,
        'R_or_package_runtime_equivalence_claimed': False,
        'run_binding': None, 'historical_reproduction_ready': False,
    }
    (OUT / 'source_binding.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    (OUT / 'determinism_audit.json').write_text(json.dumps(AUDIT, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(f'{len(SOURCES)} source hashes and {len(SPANS)} spans verified; four grid blocks compared; originals read only')


if __name__ == '__main__':
    main()
