"""The local cor (mcmc_functions_v1.6.R 2355-2366) and its time-space readers: reference, hand values, NA rules, stored R record, reachability."""
from fractions import Fraction
import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / 'docs/thesis_research/literal_local_cor_2026_11_09'
sys.path.insert(0, str(ROOT / 'tools'))
import historical_local_cor as lc  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, PACKET / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference = _load('local_cor_reference_t', 'precise_reference.py')
check = _load('local_cor_check_t', 'check_local_cor.py')


def _json(name):
    return json.loads((PACKET / name).read_text(encoding='utf-8'))


FIX = _json('synthetic_fixture.json')
CASES = FIX['cases']
READERS = FIX['readers']
EXPECTED = _json('reference_expectations.json')['cases']
RESULT, CROSS, ARITH, BOUND, AUDIT = (_json(n) for n in ('synthetic_result.json', 'r_crosscheck_result.json', 'arithmetic_check.json', 'source_binding.json', 'determinism_audit.json'))
NAN = float('nan')


@pytest.mark.parametrize('name', sorted(CASES))
def test_named_case_matches_the_independent_reference(name):
    got = check.run_cor(CASES[name])
    assert got.kind == EXPECTED[name]['kind']
    if got.kind == 'number':
        assert got.value.hex() == EXPECTED[name]['value']


@pytest.mark.parametrize('name', sorted(CASES))
def test_named_case_matches_stored_R(name):
    row = CROSS['cases'][name]
    assert row['ulps'] == 0 and row['module'] == row['R'] and row['dim'] == '1x1'


@pytest.mark.parametrize('name', sorted(READERS))
def test_reader_matches_stored_R(name):
    row = CROSS['readers'][name]
    for key in ('post_pollen', 'post_iso', 'post_iso_hi'):
        assert row[key]['ulps'] == 0 and row[key]['module'] == row[key]['R']


def test_stored_r_record_is_complete_and_bit_equal():
    assert CROSS['runtime']['R'] == 'R version 3.5.0 (2018-04-23)'
    assert (CROSS['values_compared'], CROSS['bit_equal'], CROSS['max_ulps'], CROSS['special_mismatch']) == (447, 447, 0, 0)
    assert CROSS['ulp_histogram'] == {'0': 447}
    assert CROSS['stress']['count'] == FIX['stress']['cases'] == 400 and CROSS['stress']['not_bit_equal'] == []
    assert CROSS['constants_match_python_literals'] is True and CROSS['historical_reproduction_ready'] is False


def test_stress_cases_match_the_reference_and_the_stored_module_result():
    stored = RESULT['stress']
    for i, case in enumerate(check.stress_cases(FIX['stress'])):
        got = lc.local_cor(case['x'], case['y'], na_rm=case['na_rm'])
        x, y = [reference.NA if v is None else v for v in case['x']], [reference.NA if v is None else v for v in case['y']]
        if case['na_rm']:
            pairs = [(a, b) for a, b in zip(x, y) if not reference._is_missing(a)]
            x, y = [a for a, _ in pairs], [b for _, b in pairs]
        kind, value = reference.cor(x, y)
        assert got.kind == kind, i
        if kind == 'number':
            assert got.value.hex() == value.hex(), i
        assert check.show(got) == stored[i]


def test_stress_set_uses_every_style_and_missing_values():
    cases = check.stress_cases(FIX['stress'])
    assert len(cases) == 400
    assert sum(any(v is None for v in c['x'] + c['y']) for c in cases) > 20
    assert sum(any(v is not None and math.isnan(v) for v in c['x'] + c['y']) for c in cases) > 20
    assert sum(c['na_rm'] for c in cases) > 100


def test_hand_values():
    assert check.run_cor(CASES['perfect_positive']).value == 1.0
    assert check.run_cor(CASES['perfect_negative']).value == -1.0
    assert check.run_cor(CASES['zero_covariance']).value == 0.0
    r = check.run_cor(CASES['hand_inexact']).value
    assert abs(Fraction(r) ** 2 - Fraction(1, 9545)) < Fraction(1, 10 ** 15)


def test_result_details_of_a_hand_case():
    got = check.run_cor(CASES['hand_inexact'])
    assert (got.n, got.mean_x, got.mean_y, got.numerator) == (4, 15 / 4, 13 / 4, 1 / 12)
    assert got.sd_x == math.sqrt(115 / 12) and got.sd_y == math.sqrt(83 / 12)


def test_na_in_x_is_removed_from_both_vectors():
    got = lc.local_cor([1, None, 4, 8], [3, 99, 7, 2], na_rm=True)
    assert got.n == 3 and got.y_used == (3.0, 7.0, 2.0) and got.x_used == (1.0, 4.0, 8.0)


def test_nan_in_x_is_removed_like_na():
    a, b = lc.local_cor([1, NAN, 4, 8], [3, 1, 7, 2], na_rm=True), lc.local_cor([1, None, 4, 8], [3, 1, 7, 2], na_rm=True)
    assert a.value.hex() == b.value.hex()


@pytest.mark.parametrize('bad, kind', [(None, 'NA'), (NAN, 'NaN')])
def test_only_x_is_screened(bad, kind):
    got = lc.local_cor([1, 2, 4, 8], [3, bad, 7, 2], na_rm=True)
    assert got.kind == kind and math.isnan(got.value) and got.n == 4


@pytest.mark.parametrize('name, kind', [('x_na_kept', 'NA'), ('x_nan_kept', 'NaN'), ('x_na_before_y_nan', 'NA'), ('x_nan_before_y_na', 'NaN'),
                                        ('x_na_then_nan', 'NA'), ('x_nan_then_na', 'NaN')])
def test_first_missing_value_decides_the_token(name, kind):
    assert check.run_cor(CASES[name]).kind == kind


@pytest.mark.parametrize('name', ['empty', 'single_value', 'constant_x_exact', 'x_all_missing_removed'])
def test_degenerate_input_is_nan_not_na(name):
    got = check.run_cor(CASES[name])
    assert got.kind == 'NaN' and math.isnan(got.value)


def test_constant_vector_with_inexact_mean_is_a_tiny_number_as_in_R():
    got = check.run_cor(CASES['constant_x_inexact'])
    assert got.kind == 'number' and 0 < got.value < 1e-15
    assert CROSS['cases']['constant_x_inexact']['R'] == got.value.hex()


def test_long_double_sum_keeps_small_terms():
    values = [1.0, 2.0 ** -53, 2.0 ** -53]
    assert lc.long_double_sum(values) == 1.0 + 2.0 ** -52
    assert check.plain_sum(values) == 1.0


def test_long_double_sum_rounds_only_once_to_double_at_the_end():
    values = [2.0 ** 60, 1.0, 1.0, -2.0 ** 60]
    assert lc.long_double_sum(values) == 2.0
    assert check.plain_sum(values) == 0.0


def test_sixty_four_bit_steps_differ_from_an_exact_sum():
    values = [1.0, 2.0 ** -53, 2.0 ** -65]
    assert lc.long_double_sum(values) == 1.0
    assert float(sum(Fraction(v) for v in values)) == 1.0 + 2.0 ** -52
    got, row = check.run_cor(CASES['x87_rounding_step']), CROSS['cases']['x87_rounding_step']
    assert row['module'] == row['R'] == got.value.hex()


@pytest.mark.parametrize('values', [[], [0.0], [-0.0], [0.0, -0.0], [1.5, -1.5]])
def test_long_double_sum_of_nothing_or_cancelling_terms_is_positive_zero(values):
    got = lc.long_double_sum(values)
    assert got == 0.0 and math.copysign(1.0, got) == 1.0


def test_long_double_sum_agrees_with_the_fraction_reference_on_hard_values():
    values = [0.1 * i * (-1) ** i for i in range(1, 60)] + [2.0 ** -70, 3.0 ** 20, 1e-300]
    assert lc.long_double_sum(values) == reference.long_double_sum(values)


def test_long_double_sum_past_dbl_max_is_rejected():
    with pytest.raises(lc.UnsupportedSourceDomain):
        lc.long_double_sum([1.7976931348623157e308, 1.7976931348623157e308])


def test_subnormal_values_round_like_the_reference():
    values = [5e-324, 5e-324, 1e-320, -2e-322]
    assert lc.long_double_sum(values) == reference.long_double_sum(values)


@pytest.mark.parametrize('values, mask, expected', [
    ([1, 2, 3, 4], [True, False], [1, 3]),
    ([1, 2], [True, True, True, False, True], [1, 2, lc.NA, lc.NA]),
    ([1, 2, 3], [True, None, False], [1, lc.NA]),
    ([1, 2, 3], [], []),
    ([], [True, False], [lc.NA]),
])
def test_logical_index_recycles_extends_and_reads_na(values, mask, expected):
    assert lc.logical_index(values, mask) == expected


def test_logical_index_rejects_non_boolean_flags():
    with pytest.raises(lc.UnsupportedSourceDomain):
        lc.logical_index([1, 2], [1, 0])


def _weight(value, kind='number'):
    return lc.post_treatment(lc.LocalCor(value, kind, 3, None, None, None, None, None, (), ()))


def test_positive_correlation_is_only_squared():
    w = _weight(0.5)
    assert (w.replaced, w.before_square, w.value) == (False, 0.5, 0.25)


def test_negative_correlation_becomes_1e_minus_20():
    w = _weight(-0.5)
    assert (w.replaced, w.before_square, w.value) == (True, 1e-10, 1e-10 * 1e-10)
    assert w.value.hex() == CROSS['constants']['squared']


@pytest.mark.parametrize('kind', ['NA', 'NaN'])
def test_missing_correlation_becomes_1e_minus_20_either_way(kind):
    w = _weight(NAN, kind)
    assert (w.replaced, w.value) == (True, 1e-10 * 1e-10)


def test_zero_and_negative_zero_are_not_replaced():
    assert _weight(0.0).replaced is False and _weight(0.0).value == 0.0
    w = _weight(-0.0)
    assert w.replaced is False and w.value == 0.0 and math.copysign(1.0, w.value) == 1.0


def test_values_above_one_are_kept():
    assert _weight(1.0000000000000002).value == 1.0000000000000002 ** 2


def _reader(name):
    return check.run_reader(READERS[name])


def test_multiple_branch_reads_pann_for_pollen_and_tann_for_iso():
    got = _reader('multiple_pann')
    assert (got.branch, got.tann, got.predictant_read, got.post_iso_hi) == ('1964+1966', False, 'predictant_pann', 0.95)
    x = lc.logical_index(lc.logical_index([100.0, 300.0, 200.0, 400.0, 500.0], [True] * 5), READERS['multiple_pann']['hi_ages_time'])
    expected = lc.post_treatment(lc.local_cor(x, [2.0, 3.0, 5.0, 4.0], na_rm=True))
    assert got.post_pollen == expected.value
    iso = lc.post_treatment(lc.local_cor([5.0, 6.0, 9.0, 8.0], [-1.0, 2.0, 3.0, 4.0], na_rm=True))
    assert got.post_iso == iso.value


def test_counterexample_line_1966_with_the_current_predictant_differs():
    assert ARITH['checks']['line_1966_reads_current_predictant']['differs'] is True


def test_multiple_branch_with_tann_reads_tann_for_both():
    got = _reader('multiple_tann')
    assert got.predictant_read == 'predictant_tann'


def test_mask_longer_than_selection_gives_na_that_na_rm_drops():
    long_mask = _reader('mask_longer_than_selection')
    short = check.run_reader({**READERS['mask_longer_than_selection'], 'hi_ages_time': [True, True, True]})
    assert long_mask.pollen.correlation == short.pollen.correlation
    assert CROSS['readers']['mask_longer_than_selection']['post_pollen']['ulps'] == 0


def test_mask_shorter_is_recycled():
    got = _reader('mask_shorter_recycled')
    expected = lc.post_treatment(lc.local_cor([1.0, 2.0], [1.0, 4.0], na_rm=True))
    assert got.post_pollen == expected.value


def test_negative_and_missing_are_replaced_in_the_reader():
    negative, missing = _reader('negative_replaced'), _reader('missing_replaced')
    assert negative.pollen.replaced and negative.post_pollen == 1e-10 * 1e-10
    assert missing.pollen.kind == 'NA' and missing.post_pollen == 1e-10 * 1e-10


def test_single_reference_branches():
    iso, pollen = _reader('single_iso_1993'), _reader('single_pollen_1999')
    assert (iso.branch, iso.post_pollen, iso.pollen) == ('1993', 0.5, None)
    assert (pollen.branch, pollen.post_iso, pollen.iso) == ('1999', 0.5, None)


def test_hi_time_true_is_rejected_because_add_sites_sets_it_false():
    r = READERS['multiple_pann']
    with pytest.raises(lc.UnsupportedSourceDomain, match='add.sites'):
        lc.time_space_correlations(predictant_tann=check.vector(r['predictant_tann']), predictant_pann=check.vector(r['predictant_pann']), tann=False, multiple=True,
                                   hi_ages_time=r['hi_ages_time'], hi_time=True, ages_pollen=r['ages_pollen'], predictors1=check.vector(r['predictors1']),
                                   ages_iso=r['ages_iso'], predictors2=check.vector(r['predictors2']))


@pytest.mark.parametrize('kwargs', [{'tann': None}, {'multiple': 1}, {'hi_time': 'no'}])
def test_reader_flags_must_be_booleans(kwargs):
    base = {'predictant_tann': [1.0], 'predictant_pann': [1.0], 'tann': False, 'multiple': False, 'hi_ages_time': [True], 'sources_marine': 1}
    with pytest.raises(lc.UnsupportedSourceDomain):
        lc.time_space_correlations(**{**base, **kwargs})


@pytest.mark.parametrize('value', [None, float('nan'), True, 'x'])
def test_sources_marine_must_be_a_number(value):
    with pytest.raises(lc.UnsupportedSourceDomain):
        lc.time_space_correlations(predictant_tann=[1.0], predictant_pann=[1.0], tann=False, multiple=False, hi_ages_time=[True], sources_marine=value)


@pytest.mark.parametrize('x, y, kwargs', [
    ([1, float('inf')], [1, 2], {}), ([1, 2], [float('-inf'), 2], {}), ('ab', [1, 2], {}), ([True, 2], [1, 2], {}), (['a', 2], [1, 2], {}),
    ([1, 2], [1, 2], {'na_rm': None}), ([1, 2], [1, 2], {'na_rm': 1}), ([1, 2, 3], [1, 2], {}), ([1, 2], [1, 2, 3], {'na_rm': True}),
])
def test_invalid_inputs_are_rejected(x, y, kwargs):
    with pytest.raises(lc.UnsupportedSourceDomain):
        lc.local_cor(x, y, **kwargs)


def test_overflow_is_rejected_not_emulated():
    with pytest.raises(lc.UnsupportedSourceDomain):
        lc.local_cor([1e300, -1e300, 1e300], [1e300, 1e300, -1e300])


def test_a_short_y_gets_na_under_na_rm_as_in_R():
    got = lc.local_cor([1, 2, 4, 8], [3, 1], na_rm=True)
    assert got.kind == 'NA' and got.n == 4


def test_length_mismatch_is_rejected_where_R_stops():
    for row in CROSS['length_mismatch'].values():
        assert row == {'module': 'rejected', 'R': 'ERROR'}


@pytest.mark.parametrize('check_name', sorted(ARITH['checks']))
def test_counterexamples_differ_from_the_module(check_name):
    assert ARITH['checks'][check_name]['differs'] is True


def test_double_sum_changes_most_finite_stress_results():
    row = ARITH['checks']['double_sum_for_the_mean']
    assert row['stress_compared'] == 300 and row['stress_changed'] == 199


def test_stored_module_results_match_a_rerun():
    assert RESULT['cases']['perfect_positive']['value'] == check.run_cor(CASES['perfect_positive']).value.hex()
    assert all(v['matches_reference'] for v in RESULT['cases'].values())


def test_source_binding_names_one_definition_and_the_live_calls():
    assert BOUND['status'] == 'bounded_historical_source_fragment_not_case_reproduction' and BOUND['historical_reproduction_ready'] is False
    assert BOUND['r_files_searched'] == 540 and BOUND['files_defining_cor'] == ['src/local_reconstr_all_epd/src/mcmc_functions_v1.6.R']
    assert BOUND['live_cor_call_sites']['src/mcmc_prepare.R'] == [1084]
    assert 1964 in BOUND['live_cor_call_sites']['src/mcmc_functions_v1.6.R'] and 1999 in BOUND['live_cor_call_sites']['src/mcmc_functions_v1.6.R']
    assert BOUND['correlation_decomposition_callers'] == [] and BOUND['packages_needed_by_cor'] == []


def test_multiple_is_exactly_sources_core_for_kinneret():
    table = BOUND['multiple_table_kinneret']
    assert BOUND['multiple_equals_sources_core_for_kinneret'] is True and len(table) == 4
    for row in table:
        assert row['multiple'] == (row['sources_core'] == 1)
        assert row['time_reader'] == ('1964+1966' if row['multiple'] else '1993')


def test_lines_1999_1982_2009_2032_are_marked_unreached_for_kinneret():
    readers = {str(r['line']): r['kinneret'] for r in BOUND['readers']}
    for line in ('1999', '1982', '2009', '2032'):
        assert readers[line].startswith('not reached')
    assert readers['1964, 1966'].startswith('reached when')


def test_flags_and_open_conditions_are_recorded():
    flags = BOUND['kinneret_flags']
    assert (flags['add.sites'], flags['case_studies'], flags['tann'], flags['hi_time'], flags['hi_spec']) == (True, False, False, False, False)
    assert flags['human_impact'] == [-60, 6500]
    assert len(BOUND['open_conditions']) == 3 and BOUND['giesecke_space_test']['inside_giesecke_space'] is False
    assert BOUND['site_coordinates_used_for_space_test']['older_than_v1_2'] is True


def test_audit_selects_no_variant_and_keeps_draws_out():
    assert AUDIT['final_variant_selected'] is False
    ids = {c['id']: c for c in AUDIT['candidates']}
    assert ids['local_cor']['random_state'] is False and ids['input_vectors']['random_state'] is True
    assert ids['reader_single_pollen']['reached_by_bound_run'] is False
