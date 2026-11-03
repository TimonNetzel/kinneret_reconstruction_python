"""Compare the sampler-preparation module with the independent reference on the artificial fixture; write results.

Naive readings serve as counterexamples: column-major flattening c(.) instead of c(t(.)) (it misaligns the weights
that parameter_stage_weights recycles over taxa_information$values in mcmc_functions_v1.6.R:1276), the raw values of
line 64 instead of the overwrite in 75, one global maximum instead of the column maxima, the commented alternative
in line 72, colMeans without na.rm, and the Gibbs count floor(num_taxa/2). None of them is a candidate variant.
"""
import importlib.util
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import historical_sampler_preparation as prep_module  # noqa: E402

spec = importlib.util.spec_from_file_location('rational_reference', OUT / 'rational_reference.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def module_inputs(case):
    """Fixture null (NA) -> NaN for the module."""
    matrix = [[math.nan if v is None else v for v in row] for row in case['taxa_appear_core']]
    return matrix, case['num_taxa'], case['gibbs']


def _special(v):
    if v is None:
        return 'NA'
    if math.isnan(v):
        return 'NaN'
    if math.isinf(v):
        return '-Inf' if v < 0 else 'Inf'
    return v


def as_record(got):
    return {
        'how_many_taxa': got.how_many_taxa, 'how_many_taxa_line': got.how_many_taxa_line,
        'parameter_stage': got.parameter_stage,
        'nrow_taxa_array': got.nrow_taxa_array, 'ncol_taxa_array': got.ncol_taxa_array,
        'values_line64': [_special(v) for v in got.values_line64],
        'values': [_special(v) for v in got.values],
        'column_max': list(got.column_max),
        'taxa_means': [_special(m.value) for m in got.taxa_means],
        'taxa_means_exact': ['NaN' if m.exact is None else str(m.exact) for m in got.taxa_means],
        'taxa_means_accumulator_independent': [m.accumulator_independent for m in got.taxa_means],
        'jeffreys_prior': list(got.jeffreys_prior),
        'taxa_appear_core_temp': [[_special(v) for v in row] for row in got.taxa_appear_core_temp],
    }


def _write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')


def _hand(case, ref):
    h = case['hand']
    same = [h[k] == ref[k] for k in ('how_many_taxa', 'how_many_taxa_line', 'parameter_stage', 'nrow_taxa_array',
                                     'ncol_taxa_array', 'taxa_means_accumulator_independent')]
    same.append([float(Fraction(m)) for m in h['column_max']] == ref['column_max'])
    same.append([float(Fraction(p)) for p in h['jeffreys_prior']] == ref['jeffreys_prior'])
    if h['values'] is not None:
        same.append(h['values'] == ref['values_exact'])
    if h['taxa_means'] is not None:
        same.append(h['taxa_means'] == ref['taxa_means_exact'])
    return all(same)


def _naive(case, got):
    rows = got.taxa_appear_core_temp
    nrow, ncol = len(rows), len(rows[0])
    column_major = [_special(rows[i][j]) for j in range(ncol) for i in range(nrow)]
    # 1276 multiplies values by new_weights (length ncol) with recycling: position k gets weight k %% ncol;
    # count the positions whose taxon (column j) is not that weight's taxon
    order = {'rowwise': [(i, j) for i in range(nrow) for j in range(ncol)],
             'column_major': [(i, j) for j in range(ncol) for i in range(nrow)]}
    misaligned = {key: sum(1 for k, (_, j) in enumerate(cells) if j != k % ncol) for key, cells in order.items()}
    raw = case['taxa_appear_core']
    finite = [v for row in raw for v in row if v is not None]
    top = max(finite)
    global_max = ['NA' if v is None else ('NaN' if top == 0 and v == 0 else float(Fraction(v) / Fraction(top)))
                  for row in raw for v in row] if top != 0 else None
    line72 = []
    for j in range(ncol):
        column = [raw[i][j] for i in range(nrow)]
        if any(v is None for v in column):
            line72.append('NA column (mean without na.rm)')
        else:
            mean = sum(Fraction(v) for v in column) / nrow
            line72.append(None if mean == 0 else [float(Fraction(v) / (got.num_taxa * mean)) for v in column])
    no_na_rm = ['NA' if any(raw[i][j] is None for i in range(nrow)) else 'value' for j in range(ncol)]
    values = [_special(v) for v in got.values]
    return {
        'column_major_flatten_differs': column_major != values,
        'misaligned_weight_positions': misaligned,
        'line64_values_differ_from_75': [_special(v) for v in got.values_line64] != values,
        'global_max_differs': global_max is not None and global_max != values,
        'line72_commented_alternative': line72,
        'colmeans_without_na_rm': no_na_rm,
        'gibbs_floor_half': got.num_taxa // 2, 'weights_count': got.num_taxa,
    }


def main():
    fixture = json.loads((OUT / 'synthetic_fixture.json').read_text(encoding='utf-8'))
    exact, results, hand, naive, roundtrip = {}, {}, {}, {}, {}
    for name, case in fixture['cases'].items():
        ref = reference.reference(case)
        got_raw = prep_module.prepare_sampler(*module_inputs(case)[:2], gibbs=case['gibbs'])
        got = as_record(got_raw)
        for key, value in got.items():
            if key in ref and value != ref[key]:  # exact: integers, strings and correctly rounded doubles
                raise AssertionError(f'{name}: {key} differs from the independent reference')
        exact[name], results[name] = ref, got
        hand[name] = {'how_many_taxa': ref['how_many_taxa'], 'agrees': _hand(case, ref)}
        naive[name] = _naive(case, got_raw)
        nrow, ncol = got['nrow_taxa_array'], got['ncol_taxa_array']
        rebuilt = [got['values'][i * ncol:(i + 1) * ncol] for i in range(nrow)]  # matrix(..., byrow = T)
        roundtrip[name] = rebuilt == got['taxa_appear_core_temp']
    if not all(h['agrees'] for h in hand.values()):
        raise AssertionError(f'hand expectation differs: {hand}')
    if not all(roundtrip.values()):
        raise AssertionError(f'byrow round trip differs: {roundtrip}')
    _write('rational_expectations.json', {'kind': 'independent_list_and_fraction_reference', 'cases': exact})
    _write('synthetic_result.json', {'kind': 'module_result_on_synthetic_fixture_not_historical',
                                     'historical_reproduction_ready': False, 'cases': results})
    _write('arithmetic_check.json', {
        'kind': 'synthetic_fragment_check_not_historical_reproduction',
        'cases': len(results), 'agreement': 'exact equality of counts, R specials and correctly rounded doubles',
        'hand_expectations': hand, 'naive_counterexamples': naive,
        'byrow_round_trip_1276': {'rule': 'matrix(values, ncol, nrow, byrow = T) restores the normalised matrix',
                                  'cases': roundtrip},
        'col_means_precision': 'module and reference give the correctly rounded exact mean; R may accumulate in '
                               'extended precision, so only means flagged accumulator independent are R-exact; '
                               'taxa_means has no reader in the bound loop files',
        'historical_reproduction_ready': False, 'final_variant_selected': False, 'R_execution': False})
    print(f'{len(results)} cases agree exactly with the independent reference and all hand expectations')
    print('counterexamples:', {n: [k for k, v in d.items() if k.endswith('differ') or k.endswith('differs')
                                   if v] for n, d in naive.items()})


if __name__ == '__main__':
    main()
