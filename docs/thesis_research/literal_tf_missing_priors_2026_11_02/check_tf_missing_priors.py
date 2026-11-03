"""Compare the which.tf.missing module with the independent reference on the artificial fixture; write result files.

Naive readings serve as counterexamples: round half up, a mask that also counts NA x2 means, phi1/phi2 recomputed
after the removal, and negative indexing without the length guard (in R -integer(0) is integer(0), which selects
nothing). None of them is a candidate variant; they show which source details the module keeps.
"""
import importlib.util
import json
import math
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import historical_tf_missing_priors as missing_module  # noqa: E402

spec = importlib.util.spec_from_file_location('rational_reference', OUT / 'rational_reference.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)

ARGS = ('taxa', 'all_avail', 'all_avail_age', 'taxa_appear_core', 'tf_x1', 'tf_x2', 'x1_mean', 'x2_mean',
        'taxa_change', 'x1_mean_change', 'x2_mean_change')


def _nan(value):
    """Fixture null (NA) -> NaN for the module, recursively; a missing TF row becomes a row of NaN."""
    if isinstance(value, list):
        return [_nan(v) for v in value]
    return math.nan if value is None else value


def module_inputs(case):
    kwargs = {k: case[k] for k in ARGS}
    cells = len(next(row for row in case['tf_x1'] if row is not None))
    for key in ('tf_x1', 'tf_x2'):
        kwargs[key] = [[math.nan] * cells if row is None else row for row in case[key]]
    for key in ('x1_mean', 'x2_mean', 'x1_mean_change', 'x2_mean_change'):
        kwargs[key] = None if case[key] is None else _nan(case[key])
    return case['x1_mean_is_na'], kwargs


def _null(value):
    """NaN -> null, recursively, so results compare with the fixture notation."""
    if isinstance(value, (list, tuple)):
        return [_null(v) for v in value]
    return None if isinstance(value, float) and math.isnan(value) else value


def as_record(got):
    rows = lambda m: [None if all(math.isnan(v) for v in row) else row for row in m.tolist()]  # NaN TF row -> null
    return {
        'which_tf_missing': list(got.which_tf_missing), 'removal_applied': got.removal_applied,
        'kept': list(got.kept), 'taxa': list(got.taxa),
        'all_avail': _null(got.all_avail.tolist()), 'all_avail_age': _null(got.all_avail_age.tolist()),
        'taxa_appear_core': _null(got.taxa_appear_core.tolist()),
        'tf_x1': rows(got.tf_x1), 'tf_x2': rows(got.tf_x2),
        'x1_mean': _null(list(got.x1_mean)), 'x2_mean': _null(list(got.x2_mean)),
        'x1_mean_change': None if got.x1_mean_change is None else _null(list(got.x1_mean_change)),
        'x2_mean_change': None if got.x2_mean_change is None else _null(list(got.x2_mean_change)),
        'num_taxa': got.priors.num_taxa, 'num_taxa_line': got.num_taxa_line,
        'which_taxa': list(got.priors.which_taxa), 'taxa_weights': list(got.priors.taxa_weights),
        'phi': got.priors.phi, 'nr_taxa_mean': got.priors.nr_taxa_mean, 'nr_taxa_sd': got.priors.nr_taxa_sd,
        'nr_taxa_sd_is_tie': got.priors.nr_taxa_sd_is_tie,
        'stale_phi1': got.stale_phi1, 'stale_phi2': got.stale_phi2,
        'posterior_nr_taxa_mean_survives': got.posterior_nr_taxa_mean_survives,
    }


def _write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')


def main():
    fixture = json.loads((OUT / 'synthetic_fixture.json').read_text(encoding='utf-8'))
    exact, results, hand, naive = {}, {}, {}, {}
    for name, case in fixture['cases'].items():
        ref = reference.reference(case)
        mask, kwargs = module_inputs(case)
        got = as_record(missing_module.remove_missing_tfs(mask, **kwargs))
        for key, value in got.items():
            if key in ref and value != ref[key]:  # exact: integers, names and correctly rounded doubles
                raise AssertionError(f'{name}: {key} differs from the independent reference')
        exact[name], results[name] = ref, got
        h = case['hand']
        hand[name] = {**{k: h[k] for k in ('num_taxa', 'num_taxa_line', 'nr_taxa_sd')},
                      'agrees': (h['which_tf_missing'], h['kept_taxa'], h['num_taxa'], h['num_taxa_line'],
                                 h['nr_taxa_sd'], h['nr_taxa_sd_half_up'], h['phi'], h['stale_phi1'],
                                 h['stale_phi2'])
                      == (ref['which_tf_missing'], ref['taxa'], ref['num_taxa'], ref['num_taxa_line'],
                          ref['nr_taxa_sd'], ref['nr_taxa_sd_half_up'], ref['phi'], ref['stale_phi1'],
                          ref['stale_phi2'])}
        x2_na = [i + 1 for i, v in enumerate(case['x2_mean']) if v is None]
        either = sorted(set(ref['which_tf_missing']) | set(x2_na))
        naive[name] = {
            'round_half_up_sd': ref['nr_taxa_sd_half_up'], 'source_sd': ref['nr_taxa_sd'],
            'round_half_up_differs': ref['nr_taxa_sd_half_up'] != ref['nr_taxa_sd'],
            'mask_x1_or_x2_removes': either, 'mask_x1_or_x2_differs': either != ref['which_tf_missing'],
            'recomputed_phi1_phi2': [ref['current_phi1'], ref['current_phi2']],
            'stale_phi1_phi2': [ref['stale_phi1'], ref['stale_phi2']],
            'phi1_phi2_stale_differs': [ref['current_phi1'], ref['current_phi2']] != [ref['stale_phi1'],
                                                                                        ref['stale_phi2']],
            'unguarded_empty_index_num_taxa': 0 if not ref['removal_applied'] else None,
        }
    if not all(h['agrees'] for h in hand.values()):
        raise AssertionError(f'hand expectation differs: {hand}')
    ties = {n: (reference.round_even_quarter(n), reference.round_half_up_quarter(n)) for n in range(1, 41)}
    _write('rational_expectations.json', {'kind': 'independent_list_and_fraction_reference', 'cases': exact})
    _write('synthetic_result.json', {'kind': 'module_result_on_synthetic_fixture_not_historical',
                                     'historical_reproduction_ready': False, 'cases': results})
    _write('arithmetic_check.json', {
        'kind': 'synthetic_fragment_check_not_historical_reproduction',
        'cases': len(results), 'agreement': 'exact equality of positions, names, values and doubles',
        'hand_expectations': hand, 'naive_counterexamples': naive,
        'round_ties': {'source_rule': 'R round, digits = 0: ties to the even digit (documented, not executed)',
                       'tie_counts': [n for n in ties if n % 4 == 2],
                       'half_up_differs_for': [n for n, (a, b) in ties.items() if a != b],
                       'values_1_to_40': {str(n): {'even': a, 'half_up': b} for n, (a, b) in ties.items()}},
        'unguarded_negative_index': 'R: -integer(0) is integer(0); x[integer(0)] and m[, integer(0)] select '
                                    'nothing, so the guard 1665 prevents emptying every object',
        'historical_reproduction_ready': False, 'final_variant_selected': False, 'R_execution': False})
    print(f'{len(results)} cases agree exactly with the independent reference and all hand expectations')
    print('half-up differs for n =', [n for n, (a, b) in ties.items() if a != b])
    print('counterexamples:', {n: [k for k, v in d.items() if k.endswith('differs') and v] for n, d in naive.items()})


if __name__ == '__main__':
    main()
