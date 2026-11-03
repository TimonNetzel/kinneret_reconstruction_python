"""Deterministic core of parameter_stage_weights (mcmc_functions_v1.6.R 1259, 1276-1280) with explicit weights:
reference, R semantics, domain; no rdirichlet, no sampler is run."""
from fractions import Fraction
import copy
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / 'docs/thesis_research/literal_weights_stage_2026_11_03'
STATE5 = ROOT / 'docs/thesis_research/literal_sampler_preparation_2026_11_02'
sys.path.insert(0, str(ROOT / 'tools'))
import historical_weights_stage as ws  # noqa: E402
import historical_sampler_preparation as sp  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, PACKET / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference = _load('weights_stage_reference', 'rational_reference.py')
check = _load('weights_stage_check', 'check_weights_stage.py')


def _json(name, packet=PACKET):
    return json.loads((packet / name).read_text(encoding='utf-8'))


CASES = _json('synthetic_fixture.json')['cases']
NAN, INF = math.nan, math.inf
SCALES = [2.0, 0.25, 1024.0]


def _run(name, scale=1.0):
    values, ncol, nrow, weights = check.module_inputs(CASES[name])
    return ws.weights_stage(values, ncol, nrow, [w * scale for w in weights])


def _rec(rows):
    return [[check._special(v) for v in row] for row in rows]


# --- independent reference and stored files ---------------------------------------------------------------------

@pytest.mark.parametrize('name', list(CASES))
def test_module_matches_independent_reference_exactly(name):
    got, ref = check.as_record(_run(name)), reference.reference(CASES[name])
    assert {k: v for k, v in got.items() if k in ref} == {k: ref[k] for k in got if k in ref}


@pytest.mark.parametrize('name', list(CASES))
def test_stored_files_match_a_fresh_module_run(name):
    stored = _json('synthetic_result.json')['cases'][name]
    expected = _json('rational_expectations.json')['cases'][name]
    assert stored == check.as_record(_run(name))
    assert stored['taxa_appear_core_temp'] == expected['taxa_appear_core_temp']


@pytest.mark.parametrize('name', list(CASES))
def test_hand_expectations(name):
    assert check._hand(CASES[name], reference.reference(CASES[name]))
    assert _json('arithmetic_check.json')['hand_expectations'][name]['agrees'] is True


@pytest.mark.parametrize('name', list(CASES))
def test_byrow_refills_the_rows_of_the_product_vector(name):
    got = _run(name)
    ncol = len(got.z)
    assert [list(r) for r in got.unnormalised] == [list(got.products[i * ncol:(i + 1) * ncol])
                                                   for i in range(len(got.row_sums))]


@pytest.mark.parametrize('name', list(CASES))
def test_z_is_one_to_ncol_and_weights_are_stored_unchanged(name):
    got = _run(name)
    values, ncol, nrow, weights = check.module_inputs(CASES[name])
    assert got.z == tuple(range(1, ncol + 1)) and all(isinstance(k, int) for k in got.z)
    assert got.taxa_weights == tuple(weights)  # 1259, no renormalisation


@pytest.mark.parametrize('name', list(CASES))
@pytest.mark.parametrize('scale', SCALES)
def test_power_of_two_weight_scale_cancels_exactly(name, scale):
    base, scaled = _run(name), _run(name, scale)
    assert _rec(scaled.taxa_appear_core_temp) == _rec(base.taxa_appear_core_temp)
    assert [s.accumulator_independent for s in scaled.row_sums] == [s.accumulator_independent for s in base.row_sums]


# --- recycling, byrow and division per row ---------------------------------------------------------------------

def test_each_position_gets_the_weight_of_its_own_taxon():
    got = _run('chain_three_by_four_inexact')
    assert got.recycled_weight == (1, 2, 3, 4) * 3
    assert _json('arithmetic_check.json')['naive_counterexamples']['chain_three_by_four_inexact'][
        'column_major_misaligned_weight_positions'] == 8


def test_recycling_with_more_rows_than_columns():
    got = ws.weights_stage([1.0, 1.0] * 5, 2, 5, [0.25, 0.75])
    assert got.recycled_weight == (1, 2) * 5 and _rec(got.taxa_appear_core_temp) == [[0.25, 0.75]] * 5


def test_row_sums_and_division_per_row():
    got = _run('rowwise_recycling_two_by_three')
    assert [s.exact for s in got.row_sums] == [Fraction(11, 16), Fraction(3, 4)]
    assert got.taxa_appear_core_temp[0] == (float(Fraction(8, 11)), float(Fraction(2, 11)), float(Fraction(1, 11)))
    assert got.taxa_appear_core_temp[1] == (float(Fraction(1, 3)),) * 3


@pytest.mark.parametrize('key', ['column_major_values_differ', 'byrow_false_differs', 'sum_recycled_along_rows_differs',
                                 'column_sums_differ'])
def test_naive_readings_are_counterexamples_in_every_case(key):
    assert all(v[key] for v in _json('arithmetic_check.json')['naive_counterexamples'].values())


def test_weights_matter_even_when_uniform_in_floating_point():
    # rep(1/3, 3) as in the reset 1254: one ulp from the unweighted row normalisation in row 1 (flagged row)
    got = _run('reset_weights_uniform_third')
    assert got.taxa_appear_core_temp[0][0] == 0.5714285714285715 != float(Fraction(4, 7))
    assert not got.row_sums[0].accumulator_independent
    assert _json('arithmetic_check.json')['naive_counterexamples']['reset_weights_uniform_third']['unweighted_differs']


def test_uniform_third_row_sum_ties_to_even_one():
    got = _run('reset_weights_uniform_third')
    d = Fraction(float(Fraction(1, 3)))
    assert got.row_sums[1].exact == 3 * d == 1 - Fraction(1, 2**54)
    assert got.row_sums[1].value == 1.0 and got.taxa_appear_core_temp[1] == (float(d),) * 3


# --- rowSums(na.rm = T) ----------------------------------------------------------------------------------------

def test_row_sums_skip_na_and_nan_and_keep_them_in_the_result():
    got = _run('na_and_nan_skipped_in_row_sum')
    assert [s.count for s in got.row_sums] == [2, 2, 2]
    assert _rec(got.taxa_appear_core_temp) == [[0.5, 'NA', 0.5], ['NaN', float(Fraction(2, 3)), float(Fraction(1, 3))],
                                               [0.8, 0.2, 'NA']]
    assert _json('arithmetic_check.json')['naive_counterexamples']['na_and_nan_skipped_in_row_sum'][
        'rows_without_na_rm_lost'] == [1, 2, 3]


def test_nan_column_from_state5_is_skipped():
    got = _run('chain_zero_column_nan')
    assert all(math.isnan(row[1]) for row in got.taxa_appear_core_temp)
    assert [s.exact for s in got.row_sums] == [Fraction(1, 4), Fraction(1, 2), Fraction(9, 16)]


def test_row_without_values_sums_to_zero():
    s = ws.row_sum_na_rm([None, NAN, None])
    assert (s.value, s.exact, s.count, s.accumulator_independent) == (0.0, 0, 0, True)


def test_minus_inf_row_sum_and_signed_zeros():
    got = _run('chain_minus_inf')
    assert got.row_sums[1].value == -INF and got.row_sums[1].exact is None and got.row_sums[1].count == 3
    assert _rec(got.taxa_appear_core_temp)[1] == ['NaN', '-0', '-0']


def test_negative_cell_over_minus_inf_is_positive_zero():
    got = ws.weights_stage([-INF, -1.0, 2.0, 3.0], 2, 2, [0.5, 0.5])
    assert _rec(got.taxa_appear_core_temp) == [['NaN', 0.0], [0.4, 0.6]]


# --- zero row sums ----------------------------------------------------------------------------------------------

def test_zero_row_sums_give_nan_and_signed_inf():
    got = _run('zero_row_sums')
    assert got.zero_sum_rows == (1, 2, 3)
    assert _rec(got.taxa_appear_core_temp) == [['NaN'] * 3, ['NA', 'NaN', 'NA'], ['Inf', '-Inf', 'NaN'],
                                               [0.25, 0.25, 0.5]]


def test_minus_inf_and_nan_row_and_negative_row_sum():
    got = ws.weights_stage([-INF, NAN, -2.0, 1.0], 2, 2, [1.0, 1.0])
    assert _rec(got.taxa_appear_core_temp) == [['NaN', 'NaN'], [2.0, -1.0]]


# --- accumulator flag ---------------------------------------------------------------------------------------------

DYADIC = [0.5, 0.25, 1.0, -0.125, 3.0, 2.0**-40, 2.0**30]
INEXACT = [0.1, float(Fraction(1, 3)), 2.0**53, 1.0 + 2.0**-52]


def test_flag_is_sufficient_for_exact_subset_sums():
    rows = list(itertools.product(DYADIC + INEXACT, repeat=3))
    flags = [ws.row_sum_na_rm(list(row)).accumulator_independent for row in rows]
    assert all(reference._independent([Fraction(v) for v in row]) for row, flag in zip(rows, flags) if flag)
    assert 0 < sum(flags) < len(rows)


def test_flag_is_conservative():
    # every subset sum of (2^53, 1 - 2^53) is exact, but the power-of-two bound does not see the cancellation
    row = [2.0**53, 1.0 - 2.0**53]
    assert reference._independent([Fraction(v) for v in row])
    assert not ws.row_sum_na_rm(row).accumulator_independent


def test_flags_agree_with_the_reference_on_the_fixture():
    for name in CASES:
        assert [s.accumulator_independent for s in _run(name).row_sums] == \
            reference.reference(CASES[name])['rows_accumulator_independent']


# --- chain from state 5 and cross check with the final-PDF packet ---------------------------------------------------

CHAIN = {n: c['state5_case'] for n, c in CASES.items() if c['state5_case']}


@pytest.mark.parametrize('name', list(CHAIN))
def test_chain_from_the_sampler_preparation_module(name):
    raw = _json('synthetic_fixture.json', STATE5)['cases'][CHAIN[name]]
    prepared = sp.prepare_sampler([[NAN if v is None else v for v in r] for r in raw['taxa_appear_core']],
                                  raw['num_taxa'], gibbs=False)
    values, ncol, nrow, weights = check.module_inputs(CASES[name])
    got = ws.weights_stage(prepared.values, prepared.ncol_taxa_array, prepared.nrow_taxa_array, weights)
    assert _rec(got.taxa_appear_core_temp) == _rec(_run(name).taxa_appear_core_temp)
    assert _json('arithmetic_check.json')['state5_chain']['cases'][name]['values_equal'] is True


def test_cross_check_with_final_pdf_rw_coefficients():
    cross = _json('arithmetic_check.json')['final_pdf_rw_coefficients_cross_check']['cases']
    assert set(cross) == {'chain_minimal_two_by_two', 'chain_three_by_four_inexact'}
    assert all(r['equal'] for rows in cross.values() for r in rows if r['accumulator_independent'])


def test_earlier_modules_are_byte_identical_to_their_packets():
    for packet, files in [(STATE5, ['tools/historical_sampler_preparation.py',
                                    'tests/test_historical_sampler_preparation.py'])]:
        recorded = _json('verification.json', packet)['new_files_sha256']
        for f in files:
            assert hashlib.sha256((ROOT / f).read_bytes()).hexdigest() == recorded[f]


def test_input_is_not_modified():
    values, ncol, nrow, weights = check.module_inputs(CASES['na_and_nan_skipped_in_row_sum'])
    before, wbefore = copy.deepcopy(values), list(weights)
    array = np.array(weights)
    ws.weights_stage(values, ncol, nrow, weights)
    ws.weights_stage(values, ncol, nrow, array)
    assert [check._special(v) for v in values] == [check._special(v) for v in before] and weights == wbefore
    assert np.array_equal(array, np.array(wbefore))


# --- rejected R semantics ---------------------------------------------------------------------------------------

GOOD = ([1.0, 0.5, 0.25, 1.0], 2, 2, [0.5, 0.5])


@pytest.mark.parametrize('values', [[1.0, INF, 0.5, 1.0], [1.0, -0.0, 0.5, 1.0], [1.0, 2.0, 3.0],
                                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, True, 0.5, 1.0], [1.0, '2', 0.5, 1.0],
                                    [[1.0, 2.0], [3.0, 4.0]], np.array([[1.0, 2.0], [3.0, 4.0]]), '1234'])
def test_values_outside_the_prepared_domain_are_rejected(values):
    with pytest.raises(ws.UnsupportedSourceDomain):
        ws.weights_stage(values, 2, 2, [0.5, 0.5])


@pytest.mark.parametrize('weights', [[0.5], [0.5, 0.25, 0.25], [0.0, 1.0], [-0.5, 1.5], [NAN, 1.0], [INF, 1.0],
                                     [True, 1.0], ['0.5', 0.5], '05'])
def test_weights_must_be_one_finite_positive_double_per_taxon(weights):
    with pytest.raises(ws.UnsupportedSourceDomain):
        ws.weights_stage(GOOD[0], 2, 2, weights)


@pytest.mark.parametrize('ncol,nrow', [(1, 4), (4, 1), (2.0, 2), (2, '2'), (True, 2), (2, None), (0, 0)])
def test_dimensions_must_be_integer_counts_of_at_least_two(ncol, nrow):
    with pytest.raises(ws.UnsupportedSourceDomain):
        ws.weights_stage(GOOD[0], ncol, nrow, [0.5] * (ncol if isinstance(ncol, int) and ncol > 0 else 2))


@pytest.mark.parametrize('values,ncol,weights,match', [
    ([1e308, 1.0, 1.0, 1.0], 2, [10.0, 1.0], 'product'),
    ([2.0**-1074, 1.0, 1.0, 1.0], 2, [0.25, 1.0], 'product'),
    ([2.0**1000, -(2.0**1000), 2.0**-1000, 1.0, 1.0, 1.0], 3, [1.0, 1.0, 1.0], 'quotient'),
    ([2.0**-1074, 2.0**1000, 1.0, 1.0], 2, [1.0, 1.0], 'quotient')])
def test_overflow_and_underflow_are_rejected(values, ncol, weights, match):
    with pytest.raises(ws.UnsupportedSourceDomain, match=match):
        ws.weights_stage(values, ncol, 2, weights)


def test_numpy_inputs_are_accepted():
    got = ws.weights_stage(np.array([0.5, 0.5, 1.0, 1.0]), np.int64(2), np.int64(2), np.array([0.25, 0.75]))
    assert _rec(got.taxa_appear_core_temp) == [[0.25, 0.75], [0.25, 0.75]]


# --- packet binding -----------------------------------------------------------------------------------------------

def test_source_excerpt_matches_binding_and_contains_the_literal_lines():
    binding = _json('source_binding.json')
    raw = (PACKET / 'source_excerpt.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == binding['portable_excerpt_sha256']
    text = '\n'.join(s['text'] for s in json.loads(raw)['spans'])
    for line in ('new_weights <- rdirichlet(1,old_weights*posterior_temp$dirichlet_spread)[1,]',
                 'new_weights <- rep(1/((num_taxa)),num_taxa)', 'posterior_temp$taxa_weights <- new_weights',
                 'if(is.na(parameter_ratio)) parameter_ratio <- 0',
                 'taxa.appear.core.temp <- matrix(taxa_information$values * new_weights,ncol = '
                 'taxa_information$ncol_taxa_array,nrow=taxa_information$nrow_taxa_array,byrow=T)',
                 'posterior_temp$taxa.appear.core.temp <- taxa.appear.core.temp / rowSums(taxa.appear.core.temp, '
                 'na.rm=T)', 'posterior_temp$z <- 1:ncol(taxa.appear.core.temp)',
                 'taxa.appear.core <- posterior_temp$taxa.appear.core.temp', 'posterior$dirichlet_spread <- num_taxa^2'):
        assert line in text
    assert binding['implemented_spans'] == {'store_weights': ['src/mcmc_functions_v1.6.R', 1259, 1259],
                                            'weights_stage_core': ['src/mcmc_functions_v1.6.R', 1276, 1280]}
    assert all(binding['brace_depth_at_line_start']['src/mcmc_functions_v1.6.R'][str(n)] == 1
               for n in (1259, 1276, 1277, 1280))
    assert binding['historical_reproduction_ready'] is False and binding['run_binding'] is None


def test_dirichlet_functions_are_not_defined_and_the_provider_stays_unbound():
    binding = _json('source_binding.json')
    sites = binding['assignment_sites_without_comments']
    assert sites['rdirichlet'] == {} == sites['ddirichlet']
    assert binding['rdirichlet_or_ddirichlet_defined_in_sources_incl_string_literals'] is False
    assert binding['occurrences_without_comments']['rdirichlet'] == {'src/mcmc_functions_v1.6.R': [1216, 1248]}
    assert binding['occurrences_without_comments']['SBMSplitMerge::'] == {'src/mcmc_functions_v1.6.R': [1216, 1228]}
    assert binding['dirichlet_provider_bound'] is False and _json('determinism_audit.json')[
        'dirichlet_provider_bound'] is False


def test_library_calls_are_recorded_in_run_order():
    calls = _json('source_binding.json')['library_calls_in_run_order']
    before = [c for c in calls if c['source'] == 'main_v1.2.R']
    assert len(calls) == 55 and len(before) == 52 and all(c['line'] < 651 and c['depth'] == 0 for c in before)
    assert [c['package'] for c in calls[-3:]] == ['npreg', 'truncnorm', 'colorRamps']
    order = [c['package'] for c in calls]
    assert order.index('SBMSplitMerge') < order.index('gtools') < order.index('extraDistr') < \
        order.index('LaplacesDemon')


def test_readers_and_controls_are_documented():
    binding = _json('source_binding.json')
    occ = binding['occurrences_without_comments']
    assert occ['$taxa.appear.core.temp'] == {'src/mcmc_functions_v1.6.R': [1232, 1277, 1817]}
    assert 146 in occ['$z']['src/mcmc_conduction.R'] and occ['parameter_stage('] == {'src/mcmc_conduction.R': [142]}
    assert binding['assignment_sites_without_comments']['parameter_stage_weights'] == \
        {'src/mcmc_functions_v1.6.R': [1242]}
    assert binding['assignment_sites_without_comments']['gibbs'] == {'main_v1.2.R': [624]}
    assert binding['weights_stage_free_num_taxa_lines'] == [1254, 1269]
    assert binding['main_stray_closing_braces'] == [740, 788, 829]


def test_audit_selects_only_the_core_and_keeps_random_and_package_steps_documented():
    audit = _json('determinism_audit.json')
    by_id = {c['id']: c for c in audit['candidates']}
    chosen = by_id['weights_stage_core_1276_1280']
    assert audit['selected_for_implementation'] == chosen['id'] and not audit['final_variant_selected']
    assert not chosen['random_state'] and not chosen['package_semantics_needed'] and not audit['R_execution']
    assert by_id['rdirichlet_1248']['random_state'] and by_id['rdirichlet_1248']['package_semantics_needed']
    assert by_id['ddirichlet_ratio_1269_1274']['package_semantics_needed']
    assert all(by_id[k]['random_state'] for k in ('reset_after_1000_draws_1253_1257', 'acceptance_runif_304',
                                                    'rmvnorm_rcpp_133', 'prior_samples_seed_2022',
                                                    'start_values_rbeta', 'tf_variance_sample'))
    assert by_id['weightedMedian']['decision'] == 'open'


def test_fixture_and_results_are_marked_synthetic():
    fixture = _json('synthetic_fixture.json')
    assert fixture['historical_reproduction_ready'] is False and fixture['selected_final_variant'] is None
    assert fixture['dirichlet_package'] is None
    arithmetic = _json('arithmetic_check.json')
    assert arithmetic['kind'] == 'synthetic_fragment_check_not_historical_reproduction'
    assert not (arithmetic['historical_reproduction_ready'] or arithmetic['final_variant_selected']
                or arithmetic['R_execution'] or arithmetic['dirichlet_package'])
