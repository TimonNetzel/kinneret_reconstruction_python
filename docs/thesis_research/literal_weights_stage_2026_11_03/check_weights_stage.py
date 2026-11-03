"""Compare the weights-stage module with the independent reference on the artificial fixture; write results.

Also chains the four state5 cases through the sampler-preparation module (its values must equal the fixture values)
and compares their finite rows with rw_coefficients of the final-PDF packet (the same pattern in
final_reconstruction 450-455). Naive readings serve as counterexamples: column-major values c(.) with the same
recycling, matrix(byrow = F), the row-sum vector recycled along the rows, rowSums without na.rm, no weights, and
column sums instead of row sums. None of them is a candidate variant.
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
import historical_final_pdf as final_pdf  # noqa: E402
import historical_sampler_preparation as prep_module  # noqa: E402
import historical_weights_stage as ws  # noqa: E402

spec = importlib.util.spec_from_file_location('weights_stage_reference', OUT / 'rational_reference.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
STATE5 = ROOT / 'docs/thesis_research/literal_sampler_preparation_2026_11_02/synthetic_fixture.json'


def _input(text):
    if text == 'NA':
        return None
    if text == 'NaN':
        return math.nan
    if text == '-Inf':
        return -math.inf
    return float(Fraction(text))


def module_inputs(case):
    """Fixture strings -> module values (None is NA) and weights."""
    return ([_input(v) for v in case['values']], case['ncol_taxa_array'], case['nrow_taxa_array'],
            [float(Fraction(w)) for w in case['new_weights']])


def _special(v):
    if v is None:
        return 'NA'
    if math.isnan(v):
        return 'NaN'
    if math.isinf(v):
        return '-Inf' if v < 0 else 'Inf'
    if v == 0 and math.copysign(1.0, v) < 0:
        return '-0'
    return v


def as_record(got):
    return {
        'taxa_weights': list(got.taxa_weights),
        'recycled_weight': list(got.recycled_weight),
        'products': [_special(v) for v in got.products],
        'row_sums': [_special(s.value) for s in got.row_sums],
        'row_sums_exact': ['-Inf' if s.exact is None else str(s.exact) for s in got.row_sums],
        'row_counts': [s.count for s in got.row_sums],
        'rows_accumulator_independent': [s.accumulator_independent for s in got.row_sums],
        'unnormalised': [[_special(v) for v in row] for row in got.unnormalised],
        'taxa_appear_core_temp': [[_special(v) for v in row] for row in got.taxa_appear_core_temp],
        'zero_sum_rows': list(got.zero_sum_rows),
        'z': list(got.z),
    }


def _write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')


def _matches(text, value):
    """A hand entry against a record entry; null is not written by hand."""
    if text is None:
        return True
    if text in ('NA', 'NaN', 'Inf', '-Inf', '-0'):
        return value == text
    return value == float(Fraction(text)) and not isinstance(value, str)


def _hand(case, ref):
    h = case['hand']
    flat = [x for row in ref['taxa_appear_core_temp'] for x in row]
    same = [h['z'] == ref['z'], h['recycled_weight'] == ref['recycled_weight'],
            [float(Fraction(w)) for w in h['taxa_weights']] == ref['taxa_weights'],
            h['counts'] == ref['row_counts'], h['independent'] == ref['rows_accumulator_independent'],
            h['zero_rows'] == ref['zero_sum_rows']]
    same += [_matches(t, v) for t, v in zip(h['products'], ref['products'])]
    same += [_matches(t, v) for t, v in zip(h['row_sums'], ref['row_sums'])]
    same += [_matches(t, v) for t, v in zip(h['result'], flat)]
    return all(same)


def _records(stage):
    return [[_special(v) for v in row] for row in stage]


def _normalise(rows, sums_of):
    return [[_special(ws._divide(x, sums_of(i, j))) for j, x in enumerate(row)] for i, row in enumerate(rows)]


def _naive(case, got):
    values, ncol, nrow, weights = module_inputs(case)
    correct = _records(got.taxa_appear_core_temp)
    rows = [list(r) for r in got.unnormalised]
    # values as if flattened column-major c(.), then the same recycling and byrow = T
    matrix = [values[i * ncol:(i + 1) * ncol] for i in range(nrow)]
    column_major = [matrix[i][j] for j in range(ncol) for i in range(nrow)]
    misaligned = sum(1 for k in range(nrow * ncol) if (k // nrow) != k % ncol)
    cm = _records(ws.weights_stage(column_major, ncol, nrow, weights).taxa_appear_core_temp)
    cm_back = [[cm[(j * nrow + i) // ncol][(j * nrow + i) % ncol] for j in range(ncol)] for i in range(nrow)]
    # matrix(products, byrow = F): column after column
    products = list(got.products)
    byrow_false = [[products[i + j * nrow] for j in range(ncol)] for i in range(nrow)]
    sums_f = [ws.row_sum_na_rm(r).value for r in byrow_false]
    sums = [s.value for s in got.row_sums]
    with_na = [i + 1 for i, r in enumerate(rows) if any(x is None or math.isnan(x) for x in r)]
    plain_sums = [ws.row_sum_na_rm(r).value for r in matrix]
    col_sums = [ws.row_sum_na_rm([rows[i][j] for i in range(nrow)]).value for j in range(ncol)]
    return {
        'column_major_values_differ': cm_back != correct,
        'column_major_misaligned_weight_positions': misaligned,
        'byrow_false_differs': _normalise(byrow_false, lambda i, j: sums_f[i]) != correct,
        'sum_recycled_along_rows_differs': _normalise(rows, lambda i, j: sums[(i * ncol + j) % nrow]) != correct,
        'rows_without_na_rm_lost': with_na,
        'unweighted_differs': _normalise(matrix, lambda i, j: plain_sums[i]) != correct,
        'column_sums_differ': _normalise(rows, lambda i, j: col_sums[j]) != correct,
    }


def _state5_chain(fixture):
    state5 = json.loads(STATE5.read_text(encoding='utf-8'))['cases']
    chain, cross = {}, {}
    for name, case in fixture['cases'].items():
        source = case['state5_case']
        if source is None:
            continue
        raw = [[math.nan if v is None else v for v in row] for row in state5[source]['taxa_appear_core']]
        prepared = prep_module.prepare_sampler(raw, state5[source]['num_taxa'], gibbs=False)
        values, ncol, nrow, weights = module_inputs(case)
        chain[name] = {'state5_case': source,
                       'values_equal': [_special(v) for v in prepared.values] == [_special(v) for v in values],
                       'dimensions_equal': (prepared.ncol_taxa_array, prepared.nrow_taxa_array) == (ncol, nrow)}
        finite = all(v is not None and math.isfinite(v) and v > 0 for row in raw for v in row)
        if finite:
            got = ws.weights_stage(prepared.values, ncol, nrow, weights)
            other = final_pdf.rw_coefficients(raw, weights)
            cross[name] = [{'row': i + 1, 'accumulator_independent': s.accumulator_independent,
                            'equal': [float(v) for v in other[i]] == list(got.taxa_appear_core_temp[i])}
                           for i, s in enumerate(got.row_sums)]
    return chain, cross


def main():
    fixture = json.loads((OUT / 'synthetic_fixture.json').read_text(encoding='utf-8'))
    exact, results, hand, naive = {}, {}, {}, {}
    for name, case in fixture['cases'].items():
        ref = reference.reference(case)
        got_raw = ws.weights_stage(*module_inputs(case))
        got = as_record(got_raw)
        for key, value in got.items():
            if key in ref and value != ref[key]:  # exact: integers, strings and correctly rounded doubles
                raise AssertionError(f'{name}: {key} differs from the independent reference')
        exact[name], results[name] = ref, got
        hand[name] = {'zero_sum_rows': ref['zero_sum_rows'], 'agrees': _hand(case, ref)}
        naive[name] = _naive(case, got_raw)
    if not all(h['agrees'] for h in hand.values()):
        raise AssertionError(f'hand expectation differs: {hand}')
    chain, cross = _state5_chain(fixture)
    if not all(c['values_equal'] and c['dimensions_equal'] for c in chain.values()):
        raise AssertionError(f'state-5 chain differs: {chain}')
    if not all(r['equal'] for rows in cross.values() for r in rows if r['accumulator_independent']):
        raise AssertionError(f'final-PDF cross check differs on an exact row: {cross}')
    _write('rational_expectations.json', {'kind': 'independent_list_and_fraction_reference', 'cases': exact})
    _write('synthetic_result.json', {'kind': 'module_result_on_synthetic_fixture_not_historical',
                                     'historical_reproduction_ready': False, 'cases': results})
    _write('arithmetic_check.json', {
        'kind': 'synthetic_fragment_check_not_historical_reproduction',
        'cases': len(results), 'agreement': 'exact equality of counts, R specials, signed zeros and correctly '
                                            'rounded doubles',
        'hand_expectations': hand, 'naive_counterexamples': naive,
        'state5_chain': {'rule': 'values of the sampler-preparation module for the named state-5 case equal the '
                                 'fixture values', 'cases': chain},
        'final_pdf_rw_coefficients_cross_check': {
            'rule': 'finite positive raw matrices: rows flagged accumulator independent equal rw_coefficients of '
                    'literal_final_pdf_2026_11_02 (final_reconstruction 450-455) exactly; flagged rows are only '
                    'reported', 'cases': cross},
        'row_sum_precision': 'module and reference give the correctly rounded exact row sum; R may accumulate in '
                             'extended precision, so only rows flagged accumulator independent are R-exact; the '
                             'module flag is a sufficient power-of-two bound, the reference enumerates subset sums',
        'dirichlet_package': None, 'historical_reproduction_ready': False, 'final_variant_selected': False,
        'R_execution': False})
    print(f'{len(results)} cases agree exactly with the independent reference and all hand expectations')
    print('counterexamples:', {n: [k for k, v in d.items() if k.endswith('differ') or k.endswith('differs')
                                   if v] for n, d in naive.items()})
    print('state-5 chain:', {n: c['state5_case'] for n, c in chain.items()}, 'cross check:', cross)


if __name__ == '__main__':
    main()
