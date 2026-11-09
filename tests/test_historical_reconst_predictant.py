"""reconst and the predictants (mcmc_functions_v1.6.R 1817-1820, 1825, 1829-1857, 1939-1943): reference, hand values, NA rules, stored R record, reachability."""
from fractions import Fraction
import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / 'docs/thesis_research/literal_reconst_predictant_2026_11_09'
sys.path.insert(0, str(ROOT / 'tools'))
import historical_reconst_predictant as rp  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, PACKET / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference = _load('reconst_reference_t', 'precise_reference.py')
check = _load('reconst_check_t', 'check_reconst_predictant.py')


def _json(name):
    return json.loads((PACKET / name).read_text(encoding='utf-8'))


FIX = _json('synthetic_fixture.json')
CASES, CHAINS = FIX['cases'], FIX['chains']
EXPECTED = _json('reference_expectations.json')['cases']
RESULT, CROSS, ARITH, BOUND, AUDIT = (_json(n) for n in ('synthetic_result.json', 'r_crosscheck_result.json', 'arithmetic_check.json', 'source_binding.json', 'determinism_audit.json'))
NAN, INF = float('nan'), float('inf')
TF = [[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]]
CORE = [[0.5, 0.25, 0.25], [0.0, 0.5, 0.5], [0.25, 0.25, 0.5], [0.125, 0.375, 0.5]]
HAND = sorted(n for n, c in CASES.items() if c['hand'])
RUNNABLE = sorted(n for n, r in EXPECTED.items() if r.get('stop') != '1818')


def _values(tokens):
    return [check.value(t) for t in tokens]


@pytest.mark.parametrize('name', sorted(CASES))
def test_named_case_matches_the_independent_reference(name):
    got, ref = check.run_case(CASES[name]), EXPECTED[name]
    assert got.get('stop') == ref.get('stop')
    if ref.get('stop') != '1818':
        assert (got['tann'], got['pann']) == (ref['tann'], ref['pann'])


@pytest.mark.parametrize('name', RUNNABLE)
def test_named_case_matches_stored_R_bit_for_bit(name):
    row = CROSS['cases'][name]
    for key in ('tann_ulps', 'pann_ulps'):
        assert all(u == 0 for u in row[key])
    assert row['stops_agree'] is True and row['reconst_dimnames_R'] == 'none'


@pytest.mark.parametrize('name', RUNNABLE)
def test_dimensions_after_selection_match_R(name):
    row, got = CROSS['cases'][name], check.run_case(CASES[name])
    if got['selected_dim'] is None:
        assert row['selected_R'].startswith('vector:') and row['selected_R'] == 'vector:' + str(len(rp.select_rows(check.matrix(CASES[name]['core']), CASES[name]['mask']).values))
    else:
        assert row['selected_R'] == 'x'.join(map(str, got['selected_dim']))
    assert row['reconst_dim_R'] == f'{got["nrow"]}x2'


@pytest.mark.parametrize('name', sorted(n for n, r in EXPECTED.items() if r.get('stop') == '1818'))
def test_stops_at_line_1818_agree_with_R(name):
    row = CROSS['cases'][name]
    assert row['module_stop'] == '1818' and row['R_stop'] == '1818' and row['stops_agree'] is True


@pytest.mark.parametrize('name', ['no_row_selected', 'no_row_selected_one_taxon', 'empty_mask'])
def test_zero_rows_give_an_empty_result_and_stop_at_line_1825(name):
    got = check.run_case(CASES[name])
    assert (got['stop'], got['nrow'], got['tann'], got['pann'], got['path']) == ('1825', 0, [], [], 'zero_extent')
    assert CROSS['cases'][name]['R_stop'] == '1825' and CROSS['cases'][name]['reconst_dim_R'] == '0x2'


@pytest.mark.parametrize('name', sorted(CHAINS))
def test_chain_through_the_local_cor_matches_R(name):
    row = CROSS['chains'][name]
    for key in ('post_pollen', 'post_iso', 'post_iso_hi'):
        assert row[key]['ulps'] == 0 and float.fromhex(row[key]['module']) == float.fromhex(row[key]['R'])
    assert row['R_stop'] is None


def test_stored_r_record_is_complete_and_bit_equal():
    assert CROSS['runtime'] == {'R': 'R version 3.5.0 (2018-04-23)', 'matprod': 'default', 'long_double': 'TRUE', 'rng_kind': 'Mersenne-Twister', 'blas': ''}
    assert CROSS['values_compared'] == CROSS['bit_equal'] and CROSS['values_compared'] > 2900
    assert (CROSS['max_ulps'], CROSS['special_mismatch'], CROSS['stop_disagreements']) == (0, 0, 0) and set(CROSS['ulp_histogram']) == {'0'}
    assert CROSS['stress']['count'] == FIX['stress']['cases'] == 400 and CROSS['stress']['not_bit_equal'] == []
    assert CROSS['historical_reproduction_ready'] is False


def test_every_matrix_product_path_was_exercised_against_R():
    paths = CROSS['stress']['paths']
    assert min(paths['dgemm'], paths['dgemv_t'], paths['simple_matprod'], paths['zero_extent']) > 0 and sum(paths.values()) == 400


def test_stress_cases_match_the_reference_and_the_stored_module_result():
    stored = RESULT['stress']
    for i, case in enumerate(check.stress_cases(FIX['stress'])):
        got = check.run_stress(case)
        tann, pann = check.stress_reference(case)
        assert (got['tann'], got['pann']) == (tann, pann), i
        assert got == stored[i]


def test_stress_set_uses_missing_values_and_masks():
    cases = check.stress_cases(FIX['stress'])
    flat = [v for c in cases for row in c['core'] for v in row]
    assert sum(v is None for v in flat) > 20 and sum(v is not None and math.isnan(v) for v in flat) > 20
    assert sum(v is not None and math.isinf(v) for v in flat) > 10 and sum(not all(c['mask']) for c in cases) > 20


@pytest.mark.parametrize('name', HAND)
def test_hand_values(name):
    got, hand = check.run_case(CASES[name]), CASES[name]['hand']
    for key in ('tann', 'pann'):
        assert [float.fromhex(t) for t in got[key]] == [float(Fraction(v)) for v in hand[key]]
    if 'youngest' in hand:
        assert [float.fromhex(t) for t in got['youngest']] == [float(Fraction(v)) for v in hand['youngest']]


def test_reconst_is_a_matrix_of_two_columns_tann_first():
    got = rp.reconst_predictant(CORE, [True] * 4, TF)
    assert (got.path, got.nrow, got.tann[0], got.pann[0]) == ('dgemm', 4, 17.5, 175.0)
    assert got.selected.dim == (4, 3)


def test_line_1464_and_1522_both_mean_all_rows():
    assert rp.reconst_predictant(CORE, [True] * 4, TF).nrow == 4
    assert BOUND['mask']['writes'] == [1464, 1466, 1522] and BOUND['mask']['reads_in_active_reconstruction'] == [1818]
    assert 'always all TRUE' in BOUND['mask']['effect']


def test_masks_recycle_when_short():
    assert rp.reconst_predictant(CORE, [True, False], TF).tann == (17.5, 22.5)


def test_too_long_masks_stop_as_in_R():
    with pytest.raises(rp.UnsupportedSourceDomain, match='too long'):
        rp.select_rows(CORE, [True] * 5)


def test_na_in_the_mask_selects_a_row_of_na():
    selected = rp.select_rows(CORE, [True, None, True, False])
    assert selected.dim == (3, 3) and selected.values[1] == (rp.NA, rp.NA, rp.NA)
    got = rp.matrix_product(selected, TF)
    assert (got.tann[1], got.pann[1]) == (rp.NA, rp.NA) and got.path == 'simple_matprod'


def test_one_selected_row_drops_to_a_vector_and_acts_as_a_row():
    selected = rp.select_rows(CORE, [False, True, False, False])
    assert selected.dim is None and selected.values == tuple(CORE[1])
    got = rp.matrix_product(selected, TF)
    assert (got.nrow, got.tann, got.pann, got.path) == (1, (25.0,), (250.0,), 'dgemv_t')


def test_one_taxon_drops_to_a_vector_and_acts_as_a_column():
    selected = rp.select_rows([[0.5], [0.25], [1.0]], [True] * 3)
    assert selected.dim is None and selected.values == (0.5, 0.25, 1.0)
    got = rp.matrix_product(selected, [[8.0, 16.0]])
    assert (got.tann, got.pann) == ((4.0, 2.0, 8.0), (8.0, 4.0, 16.0))


def test_one_row_vector_against_a_wrong_tf_is_not_conformable():
    with pytest.raises(rp.UnsupportedSourceDomain, match='non-conformable'):
        rp.reconst_predictant(CORE, [True, False, False, False], TF[:2])


def test_matrix_against_a_wrong_tf_is_not_conformable():
    with pytest.raises(rp.UnsupportedSourceDomain, match='non-conformable'):
        rp.reconst_predictant(CORE, [True] * 4, TF[:2])


@pytest.mark.parametrize('name', ['tie_finite', 'tie_one_row_dgemv', 'tie_with_nan_in_other_row', 'tie_with_na_in_other_row', 'tie_with_inf_in_other_row'])
def test_a_finite_row_keeps_its_double_accumulation_whatever_stands_elsewhere(name):
    got = check.run_case(CASES[name])
    assert got['tann'][0] == (1.0).hex() and got['pann'][0] == (1.0).hex()
    assert CROSS['cases'][name]['tann_R'].split()[0] == '0x1p+0'


def test_three_loop_path_is_double_not_long_double():
    row = ARITH['checks']['long_double_in_three_loop_path']
    assert row['differs'] is True and row['module'] == (1.0).hex() and row['variant'] == (1.0 + 2.0 ** -52).hex()
    facts = BOUND['r_matprod_source']
    assert (facts['array_c_lines']['simple_matprod_accumulator'], facts['array_c_lines']['internal_matprod_accumulator']) == ('double', 'LDOUBLE')
    assert facts['options_rd_states_double_accumulator_for_three_loop_path'] is True and 'long double' in facts['correction_of_the_handover']


def test_a_nan_in_one_tf_column_leaves_the_other_column_finite():
    got = check.run_case(CASES['tie_with_nan_in_tf_column_1'])
    assert got['tann'] == ['NaN', 'NaN'] and got['pann'] == [(1.0).hex(), (1.0).hex()] and got['path'] == 'simple_matprod'


def test_python_builtin_sum_is_not_a_plain_double_accumulation_here():
    row = ARITH['checks']['python_builtin_sum']
    assert row['differs'] is True and row['stress_changed'] > 100 and row['python'].startswith('3.12')
    terms = [1.0, 2.0 ** -53, 2.0 ** -53]
    total = 0.0
    for v in terms:
        total += v
    assert total == 1.0 and sum(terms) == 1.0 + 2.0 ** -52


def test_exact_sum_differs_where_taxon_order_matters():
    got = check.run_case(CASES['many_taxa_order_matters'])
    assert float.fromhex(got['tann'][0]) == 1.0 and ARITH['checks']['exact_sum']['differs'] is True
    assert float(sum(Fraction(v) for v in [2.0 ** 60, 1.0, -2.0 ** 60, 1.0])) == 2.0


@pytest.mark.parametrize('x, y, token', [(None, NAN, 'NA'), (NAN, None, 'NaN'), (None, None, 'NA'), (NAN, NAN, 'NaN'), (None, 2.0, 'NA'), (2.0, None, 'NA'), (NAN, 2.0, 'NaN'), (2.0, NAN, 'NaN')])
def test_product_token_takes_the_first_missing_operand(x, y, token):
    assert check.show(rp.reconst_predictant([[x, 1.0], [1.0, 1.0]], [True, True], [[y, 1.0], [1.0, 1.0]]).tann[0]) == token


@pytest.mark.parametrize('a, b, token', [(None, NAN, 'NA'), (NAN, None, 'NaN'), (None, 1.0, 'NA'), (1.0, None, 'NA'), (NAN, 1.0, 'NaN'), (1.0, NAN, 'NaN')])
def test_a_missing_accumulator_stays_before_a_missing_product(a, b, token):
    assert check.show(rp.reconst_predictant([[a, b], [1.0, 1.0]], [True, True], [[1.0, 1.0], [1.0, 1.0]]).tann[0]) == token


def test_zero_times_na_stays_na_and_zero_times_inf_is_nan():
    assert check.run_case(CASES['zero_times_na'])['tann'][0] == 'NA' and check.run_case(CASES['zero_times_inf'])['tann'][0] == 'NaN'


def test_inf_minus_inf_and_nan_from_cancellation():
    assert check.run_case(CASES['inf_minus_inf'])['tann'][0] == 'NaN' and check.run_case(CASES['token_sum_inf_cancel_then_na'])['tann'][0] == 'NaN'


def test_overflowing_products_follow_ieee():
    got = check.run_case(CASES['product_overflow_finite_inputs'])
    assert got['tann'] == ['Inf', 'NaN'] and got['pann'] == ['Inf', '0']
    assert check.run_case(CASES['product_overflow_blas_path'])['path'] == 'dgemm'


def test_false_positive_of_the_nan_test_changes_only_the_path():
    got = check.run_case(CASES['false_positive_nan_test'])
    assert got['path'] == 'simple_matprod' and all(math.isfinite(float.fromhex(t)) for t in got['tann'] + got['pann'])


def test_accumulator_starts_at_positive_zero():
    got = check.run_case(CASES['negative_zero_terms'])
    assert got['tann'] == ['0', '0'] and got['pann'][1] == '0'


def test_subnormal_terms_match_the_reference():
    assert check.run_case(CASES['subnormal_terms'])['tann'] == EXPECTED['subnormal_terms']['tann']


def test_nan_test_is_pairwise_and_treats_na_as_missing():
    assert rp._nan_or_inf_in([rp.NA, 1.0]) is True
    assert rp._nan_or_inf_in([1e308, 1e308]) is True and rp._nan_or_inf_in([1e308, -1e308, 1.0]) is False
    assert rp._nan_or_inf_in([INF]) is True and rp._nan_or_inf_in([1.0, 2.0]) is False


def test_youngest_values_are_the_first_selected_row():
    assert rp.youngest_values(rp.reconst_predictant(CORE, [False, True, True, False], TF)) == (25.0, 250.0)


def test_youngest_values_stop_on_zero_rows():
    empty = rp.reconst_predictant(CORE, [False] * 4, TF)
    with pytest.raises(rp.UnsupportedSourceDomain, match='1825'):
        rp.youngest_values(empty)


def test_recent_true_reads_the_first_predictant():
    r = rp.reconst_predictant(CORE, [True] * 4, TF)
    assert rp.recent_climate(r, True, 5.5, 600.0) == (17.5, 175.0)


def test_recent_false_returns_the_prior_means():
    r = rp.reconst_predictant(CORE, [True] * 4, TF)
    assert rp.recent_climate(r, False, 5.5, 600.0) == (5.5, 600.0)
    row = RESULT['cases']['recent_false_uses_prior']
    assert (row['post_tann'], row['post_pann']) == ((5.5).hex(), (600.0).hex())


def test_recent_on_an_empty_reconst_is_na_as_in_R_indexing():
    assert rp.recent_climate(rp.reconst_predictant(CORE, [False] * 4, TF), True, 1.0, 2.0) == (rp.NA, rp.NA)


@pytest.mark.parametrize('tann, name, column', [(True, 'predictant_tann', 0), (False, 'predictant_pann', 1)])
def test_predictant_current_follows_tann(tann, name, column):
    r = rp.reconst_predictant(CORE, [True] * 4, TF)
    assert rp.predictant_current(r, tann) == (name, (r.tann, r.pann)[column])


def test_line_1942_is_the_kinneret_branch():
    assert RESULT['cases']['hand_all_rows']['current_name'] == 'predictant_pann' and CROSS['cases']['hand_all_rows']['current_is_tann_R'] == 'FALSE'
    assert RESULT['cases']['tann_is_read']['current_name'] == 'predictant_tann' and CROSS['cases']['tann_is_read']['current_is_tann_R'] == 'TRUE'


def test_block_rejects_taxa_change():
    with pytest.raises(rp.UnsupportedSourceDomain, match='taxa_change'):
        rp.reconstruction_block(taxa_appear_core=CORE, reconst_ages=[True] * 4, tf_sample=TF, tann=False, recent=True, taxa_change=True)


@pytest.mark.parametrize('kwargs', [{'tann': None}, {'tann': 1}, {'recent': 'yes'}, {'recent': None}, {'taxa_change': None}])
def test_flags_must_be_booleans(kwargs):
    base = {'taxa_appear_core': CORE, 'reconst_ages': [True] * 4, 'tf_sample': TF, 'tann': False, 'recent': True}
    with pytest.raises(rp.UnsupportedSourceDomain):
        rp.reconstruction_block(**{**base, **kwargs})


@pytest.mark.parametrize('core', ['ab', [], [[]], [[1.0, 2.0], [1.0]], [['a', 1.0]], [[True, 1.0]]])
def test_invalid_core_is_rejected(core):
    with pytest.raises(rp.UnsupportedSourceDomain):
        rp.select_rows(core, [True])


@pytest.mark.parametrize('tf', ['ab', [], [[1.0]], [[1.0, 2.0, 3.0]], [['a', 2.0]], [[True, 2.0]]])
def test_invalid_tf_is_rejected(tf):
    with pytest.raises(rp.UnsupportedSourceDomain):
        rp.reconst_predictant([[1.0, 1.0]], [True], tf)


@pytest.mark.parametrize('mask', [[1], ['x'], [0, 1]])
def test_mask_must_hold_booleans(mask):
    with pytest.raises(rp.UnsupportedSourceDomain):
        rp.select_rows([[1.0, 1.0], [2.0, 2.0]], mask)


def test_predictants_feed_the_local_cor_module():
    got = RESULT['chains']['kinneret_shape_multiple']
    assert (got['branch'], got['predictant_read'], got['post_iso_hi']) == ('1964+1966', 'predictant_pann', (0.95).hex())
    assert CROSS['chains']['kinneret_shape_multiple']['post_iso']['ulps'] == 0


def test_na_predictant_is_screened_by_na_rm_in_the_chain():
    assert CROSS['chains']['na_row_removed_by_cor']['post_pollen']['ulps'] == 0


@pytest.mark.parametrize('check_name', ['long_double_in_three_loop_path', 'python_builtin_sum', 'exact_sum', 'na_treated_as_zero', 'column_wise_reading', 'line_1466_holocene_mask_taken_as_effective'])
def test_counterexamples_differ_from_the_module(check_name):
    assert ARITH['checks'][check_name]['differs'] is True


def test_stored_module_results_match_a_rerun():
    assert RESULT['cases']['hand_all_rows']['tann'] == check.run_case(CASES['hand_all_rows'])['tann']
    assert all(v['matches_reference'] for v in RESULT['cases'].values())


def test_source_binding_names_the_active_definition_and_the_mask_writes():
    assert BOUND['status'] == 'bounded_historical_source_fragment_not_case_reproduction' and BOUND['historical_reproduction_ready'] is False
    assert BOUND['active_definition'] == {'line': 1781, 'replaces': 855, 'called_at': 'mcmc_conduction.R:146', 'body_last_line': 2046}
    assert BOUND['occurrences']['reconst_ages']['src/mcmc_prepare.R'] == [1464, 1466, 1522]
    assert BOUND['occurrences']['reconst_ages']['src/mcmc_functions_v1.6.R'] == [461, 462, 470, 1818]
    assert BOUND['packages_needed'] == [] and BOUND['actual_original_inputs_used_for_numerics'] is False


def test_formal_tf_sample_is_never_read():
    assert BOUND['formal_tf_sample_read_in_body'] is False and 'posterior_temp$tf_sample' in BOUND['formal_tf_sample_note']


def test_global_reconst_is_shadowed_and_never_read():
    assert BOUND['occurrences']['reconst']['src/mcmc_conduction.R'] == [109] and 'never read' in BOUND['global_shadowed']['mcmc_conduction.R:109-110']


def test_final_reconstruction_has_no_caller_in_the_loaded_files():
    assert BOUND['final_reconstruction_callers_in_loaded_files'] == {}


def test_every_predictant_read_of_the_active_function_is_in_the_table():
    reads = BOUND['active_body_reads']
    assert reads['predictant_tann'] == [1819, 1831, 1834, 1837, 1850, 1867, 1879, 1901, 1940, 1966, 1993]
    assert reads['predictant_pann'] == [1820, 1832, 1835, 1837, 1851, 1868, 1880, 1892, 1902, 1942, 1982, 2009]
    assert BOUND['predictant_current_occurrences'] == [1940, 1942, 1964, 1999, 2023, 2031]


@pytest.mark.parametrize('line', ['1831-1837', '1867-1902', '1923-1929', '1940', '1999', '1982, 2009', '2031'])
def test_unreached_lines_are_marked_for_kinneret(line):
    assert {str(r['line']): r['kinneret'] for r in BOUND['readers']}[line].startswith('not reached')


@pytest.mark.parametrize('line, start', [('1825', 'reached'), ('1942', 'reached'), ('1850, 1851', 'reached when recent'), ('1964, 1966', 'reached when sources$core'), ('1993', 'reached only if')])
def test_reached_lines_are_marked_with_their_condition(line, start):
    assert {str(r['line']): r['kinneret'] for r in BOUND['readers']}[line].startswith(start)


def test_flags_and_open_conditions_are_recorded():
    flags = BOUND['kinneret_flags']
    assert (flags['past_climate_module'], flags['stad_interstad'], flags['taxa_change'], flags['tann']) == (False, False, False, False)
    assert len(BOUND['open_conditions']) == 3 and 'recent' in BOUND['open_conditions'][0]


def test_audit_selects_no_variant_and_keeps_draws_out():
    assert AUDIT['final_variant_selected'] is False
    ids = {c['id']: c for c in AUDIT['candidates']}
    assert ids['proposal_draws']['random_state'] is True and ids['selection_and_product']['random_state'] is False and ids['taxa_change_block']['reached_by_bound_run'] is False


def test_measured_nan_rule_fits_exactly_one_of_four_rules():
    m = _json('nan_rule_measurement.json')
    assert m['combinations'] == 1296 and m['mismatches_per_rule']['x_first_in_product__sum_first_in_addition'] == 0
    assert sorted(v > 0 for v in m['mismatches_per_rule'].values()) == [False, True, True, True]
    assert (m['three_loop_accumulator']['tie_with_nan_in_other_row'], m['three_loop_accumulator']['tie_finite']) == ('0x1p+0', '0x1p+0')


def test_no_gpl_text_is_stored_only_numbers():
    text = json.dumps(BOUND['r_matprod_source'])
    assert 'mayHaveNaNOrInf(x' not in text and 'for (int' not in text and BOUND['r_matprod_source']['tarball_sha256'] == 'fd1725535e21797d3d9fea8963d99be0ba4c3aecadcf081b43e261458b416870'
