"""Bind the taxon-pruning source and the determinism audit to already pinned originals.

Reads only. Each original must match the SHA-256 recorded by an earlier frozen packet before any span is
copied; the hash is checked again after reading. No R is executed and no candidate variant is selected.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASIS = '47a926c3c8a6abc0db7c947b993f5e97f7408cdd'

# source -> (sha256 from an earlier frozen binding, that binding)
SOURCES = {
    'src/mcmc_prepare.R': ('b897e84f8f2d72399de729592066341585974d521bf8164b185f20377ed5cc7e',
                           'literal_tf_preparation_2026_11_02/source_binding.json'),
    'src/mcmc_prep_v1.2.R': ('badabd80978a074f42c25e06f1d254fd4ac8deef4e8f31bc591af6de65123949',
                             'kinneret_references_2026_11_02/contract.json'),
    'src/mcmc_prep.R': ('e8d3db1554582af5326ba00051256dd4f40343936575bbc6cf23774d4909aac2',
                        'required_files_2026_11_02/source_dates.csv'),
    'main_v1.2.R': ('82486f99bafff08af2b51475ed225501d26cec1a0322a286b97a248ebd2f0520',
                    'kinneret_binding_2026_11_02/summary.json'),
}

# (source, first, last, role)
SPANS = [
    ('src/mcmc_prepare.R', 45, 83, 'implemented: taxon pruning 51-80; 83 follows and is already in literal_age_depth'),
    ('src/mcmc_prep_v1.2.R', 147, 176, 'copy of 51-80, compared byte for byte'),
    ('src/mcmc_prep.R', 144, 173, 'copy of 51-80 except the relict condition: if(ohrid)'),
    ('main_v1.2.R', 651, 676, 'bound reconstruction loop: null.ratio.check <- FALSE before source(mcmc_prepare.R)'),
    ('main_v1.2.R', 895, 895, 'stop() before the plant-map loop; reachable only by interactive execution'),
    ('main_v1.2.R', 904, 913, 'plant-map loop: null.ratio.check <- TRUE before source(mcmc_prep.R)'),
    ('main_v1.2.R', 27, 27, 'matrixStats attached; version not recorded'),
    ('main_v1.2.R', 50, 50, 'spatstat.utils attached (inside.range); version not recorded'),
    ('src/mcmc_prepare.R', 1589, 1608, 'audit: unseeded sample/quantile, then overwritten by range()'),
    ('src/mcmc_prepare.R', 1622, 1660, 'audit: expectation overwritten, spline only used in comments, weightedMedian active'),
    ('src/mcmc_prepare.R', 1663, 1679, 'audit: NA-median taxon removal depends on weightedMedian'),
    ('src/mcmc_prepare.R', 1691, 1694, 'audit: weightedMedian results stored in tf_x12_mean_current'),
    ('src/mcmc_prepare.R', 1394, 1413, 'audit: unseeded sample() inside loop, then weightedMedian'),
    ('src/mcmc_prepare.R', 1890, 1914, 'audit: set.seed(2023) + sample/density/quantile; sample algorithm depends on R RNG settings'),
]

AUDIT = {
    'kind': 'determinism_audit_from_source_text_no_R_execution',
    'question': 'Which step around the TF preparation is deterministic without random state and fully defined by bound sources?',
    'candidates': [
        {'id': 'taxon_pruning', 'lines': 'mcmc_prepare.R:51-80',
         'random_state': False, 'functions': ['length', 'which', '==', '>=', '%in%', 'sort', ':', 'ncol', 'nrow', 'colMeans', 'next'],
         'package_semantics_needed': False,
         'active_in_bound_reconstruction_loop': False,
         'evidence': 'main_v1.2.R:659 sets null.ratio.check <- FALSE before main_v1.2.R:675 source("src/mcmc_prepare.R"); '
                     'main_v1.2.R:909 sets TRUE only for the plant-map loop with mcmc_prep.R, after stop() in line 895',
         'decision': 'implemented as conditional module; no run activation claimed'},
        {'id': 'weightedMedian', 'lines': 'mcmc_prepare.R:1631,1632,1651,1652',
         'random_state': False, 'package_semantics_needed': True,
         'active_in_bound_reconstruction_loop': True,
         'evidence': 'Not defined in the bound sources; main_v1.2.R:27 attaches matrixStats without version. '
                     'Interpolation/tie and zero-weight rules depend on the package version.',
         'decision': 'documented only; implementing would select an unbound package variant (runtime_rng bundle)'},
        {'id': 'spline_expand', 'lines': 'mcmc_prepare.R:1627,1647',
         'random_state': False, 'package_semantics_needed': False,
         'active_in_bound_reconstruction_loop': False,
         'evidence': 'tann_expand/pann_expand are read only in commented lines 1637 and 1657',
         'decision': 'documented only; no effect on the preserved active flow'},
        {'id': 'expectation_value', 'lines': 'mcmc_prepare.R:1623-1624',
         'random_state': False, 'package_semantics_needed': False,
         'active_in_bound_reconstruction_loop': False,
         'evidence': 'tf.x1.current.mean/tf.x2.current.mean are reset to c() in 1628/1648 and refilled by weightedMedian',
         'decision': 'documented only; overwritten'},
        {'id': 'tf_variance_sample_quantile', 'lines': 'mcmc_prepare.R:1591-1603',
         'random_state': True, 'seed_in_bound_sources': False,
         'active_in_bound_reconstruction_loop': False,
         'evidence': 'tann_tf_range/pann_tf_range are overwritten by range(x1range.bck)/range(x2range.bck) in 1607-1608',
         'decision': 'documented only; random and overwritten'},
        {'id': 'recent_taxa_sample', 'lines': 'mcmc_prepare.R:1399',
         'random_state': True, 'seed_in_bound_sources': False,
         'decision': 'documented only; no seed or RNG state bound'},
        {'id': 'tf_quantiles_seed_2023', 'lines': 'mcmc_prepare.R:1896-1914',
         'random_state': True, 'seed_in_bound_sources': True,
         'evidence': 'set.seed(2023) is present, but RNG kind, sample.kind and R version are not bound',
         'decision': 'documented only; no RNG emulation or substitute seed'},
    ],
    'selected_for_implementation': 'taxon_pruning',
    'final_variant_selected': False,
    'R_execution': False,
}


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
    spans = [{'source': 'original:' + rel, 'first_line': a, 'last_line': b, 'role': role,
              'text': '\n'.join(lines[rel][a - 1:b])} for rel, a, b, role in SPANS]
    prepare = lines['src/mcmc_prepare.R'][50:80]
    if lines['src/mcmc_prep_v1.2.R'][146:176] != prepare:
        raise ValueError('mcmc_prep_v1.2.R pruning block is no longer identical')
    prep = lines['src/mcmc_prep.R'][143:173]
    differing = [i + 51 for i, (a, b) in enumerate(zip(prepare, prep)) if a != b]
    if differing != [57] or prep[6].strip() != 'if(ohrid){':
        raise ValueError('mcmc_prep.R pruning copy differs beyond the relict condition')
    excerpt = {'status': 'read_only_original_spans', 'spans': spans}
    (OUT / 'source_excerpt.json').write_text(json.dumps(excerpt, indent=2, ensure_ascii=False) + '\n',
                                             encoding='utf-8', newline='\n')
    result = {
        'status': 'bounded_conditional_historical_source_fragment_not_case_reproduction',
        'basis_commit': BASIS, 'sources': binding,
        'source_hashes_before_equal_after_and_prior': True,
        'portable_excerpt': 'source_excerpt.json',
        'portable_excerpt_sha256': hashlib.sha256((OUT / 'source_excerpt.json').read_bytes()).hexdigest(),
        'implemented_span': {'prune_taxa': ['src/mcmc_prepare.R', 51, 80]},
        'copies': {'src/mcmc_prep_v1.2.R:147-176': 'identical',
                   'src/mcmc_prep.R:144-173': 'identical except line 57 relict condition if(ohrid)'},
        'relict_condition': 'explicit boolean input; mcmc_prepare.R uses sitename == "Lake Ohrid", mcmc_prep.R a free variable ohrid',
        'activation': 'main_v1.2.R:659 null.ratio.check <- FALSE in the bound reconstruction loop; no run activation claimed',
        'excluded_context': '83 zero replacement (in literal_age_depth), weightedMedian, spline, sample/quantile, inside.range',
        'next_semantics': 'reported as skip_site; R next() across source() is not emulated',
        'supported_domain': 'nonempty real numeric depth-by-taxon matrix, finite nonnegative, at most 10**6 rows; '
                            'relict override needs at least 25 taxa; at least two remaining taxa',
        'NA_vector_extension_or_dimension_drop_semantics_reproduced': False,
        'colMeans_long_double_bit_equality_claimed': False,
        'actual_original_inputs_used_for_numerics': False,
        'R_or_package_runtime_equivalence_claimed': False,
        'run_binding': None, 'historical_reproduction_ready': False,
    }
    (OUT / 'source_binding.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    (OUT / 'determinism_audit.json').write_text(json.dumps(AUDIT, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(f'{len(SOURCES)} source hashes and {len(SPANS)} spans verified; originals read only')


if __name__ == '__main__':
    main()
