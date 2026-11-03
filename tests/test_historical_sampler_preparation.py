"""Historical sampler preparation (mcmc_conduction.R 53-79): reference, R semantics, domain; no sampler is run."""
from fractions import Fraction
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / 'docs/thesis_research/literal_sampler_preparation_2026_11_02'
sys.path.insert(0, str(ROOT / 'tools'))
import historical_sampler_preparation as sp  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, PACKET / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference = _load('sampler_prep_reference', 'rational_reference.py')
check = _load('sampler_prep_check', 'check_sampler_preparation.py')


def _json(name):
    return json.loads((PACKET / name).read_text(encoding='utf-8'))


CASES = _json('synthetic_fixture.json')['cases']
NAN = math.nan


def _run(name):
    matrix, num_taxa, gibbs = check.module_inputs(CASES[name])
    return sp.prepare_sampler(matrix, num_taxa, gibbs=gibbs)


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
    assert (stored['values'], stored['how_many_taxa']) == (expected['values'], expected['how_many_taxa'])


@pytest.mark.parametrize('name', list(CASES))
def test_hand_expectations(name):
    h, got = CASES[name]['hand'], _run(name)
    assert (got.how_many_taxa, got.how_many_taxa_line, got.parameter_stage) == \
        (h['how_many_taxa'], h['how_many_taxa_line'], h['parameter_stage'])
    assert (got.nrow_taxa_array, got.ncol_taxa_array) == (h['nrow_taxa_array'], h['ncol_taxa_array'])
    assert list(got.column_max) == [float(Fraction(m)) for m in h['column_max']]
    assert [m.accumulator_independent for m in got.taxa_means] == h['taxa_means_accumulator_independent']
    if h['values'] is not None:
        specials = {'NA': None}
        for v, text in zip(got.values, h['values']):
            if text in specials:
                assert v is None
            elif text == 'NaN':
                assert math.isnan(v)
            elif text == '-Inf':
                assert v == -math.inf
            else:
                assert v == float(Fraction(text))
    if h['taxa_means'] is not None:
        assert [str(m.exact) for m in got.taxa_means] == h['taxa_means']


@pytest.mark.parametrize('name', list(CASES))
def test_byrow_round_trip_restores_the_normalised_matrix(name):
    got = _run(name)
    ncol = got.ncol_taxa_array
    rebuilt = [got.values[i * ncol:(i + 1) * ncol] for i in range(got.nrow_taxa_array)]
    assert [[check._special(v) for v in row] for row in rebuilt] == \
        [[check._special(v) for v in row] for row in got.taxa_appear_core_temp]
    assert _json('arithmetic_check.json')['byrow_round_trip_1276']['cases'][name] is True


# --- gibbs switch ---------------------------------------------------------------------------------------------

def test_bound_branch_is_weights_with_num_taxa():
    got = _run('weights_branch_odd_num_taxa')
    assert (got.gibbs, got.how_many_taxa, got.how_many_taxa_line, got.parameter_stage) == \
        (False, 5, 57, 'parameter_stage_weights')


def test_gibbs_branch_is_floor_half_and_reported_only():
    got = _run('gibbs_true_branch_odd_num_taxa')
    assert (got.how_many_taxa, got.how_many_taxa_line, got.parameter_stage) == (2, 54, 'parameter_stage_gibbs')
    same = _run('weights_branch_odd_num_taxa')
    assert got.values == same.values and got.jeffreys_prior == same.jeffreys_prior  # only 53-59 differ


@pytest.mark.parametrize('n', range(2, 40))
def test_how_many_taxa_against_integer_division(n):
    assert sp.how_many_taxa(n, False) == (n, 57, 'parameter_stage_weights')
    assert sp.how_many_taxa(n, True)[0] == math.floor(Fraction(n, 2)) == n // 2


@pytest.mark.parametrize('bad', [1, 0, None, 'F', 'FALSE'])
def test_gibbs_must_be_an_explicit_boolean(bad):
    with pytest.raises(sp.UnsupportedSourceDomain):
        sp.how_many_taxa(4, bad)


def test_numpy_integer_count_is_accepted_and_bool_count_rejected():
    assert sp.how_many_taxa(np.int64(6), True)[0] == 3
    with pytest.raises(sp.UnsupportedSourceDomain):
        sp.how_many_taxa(True, False)


# --- flattening, overwrite and normalisation --------------------------------------------------------------------

def test_flattening_runs_along_the_rows():
    got = _run('rowwise_flatten_three_by_four')
    assert got.values_line64[:4] == (2.0, 1.0, 4.0, 3.0)  # first row, not first column
    assert got.values[:4] == (0.5, 1 / 6, 0.5, 1 / 3)
    naive = _json('arithmetic_check.json')['naive_counterexamples']['rowwise_flatten_three_by_four']
    assert naive['column_major_flatten_differs']
    assert naive['misaligned_weight_positions'] == {'rowwise': 0, 'column_major': 8}


def test_line_75_overwrites_the_raw_values_of_line_64():
    got = _run('minimal_two_by_two')
    assert got.values_line64 == (1.0, 3.0, 2.0, 6.0) and got.values == (0.5, 0.5, 1.0, 1.0)
    assert all(v['line64_values_differ_from_75'] for v in _json('arithmetic_check.json')
               ['naive_counterexamples'].values())


def test_dimensions_come_from_the_raw_matrix():
    got = _run('na_skipped_in_mean_and_max')
    assert (got.nrow_taxa_array, got.ncol_taxa_array, len(got.values)) == (4, 3, 12)


def test_column_max_ignores_na_and_na_stays_na():
    got = _run('na_skipped_in_mean_and_max')
    assert got.column_max == (4.0, 4.0, 8.0)
    assert got.taxa_appear_core_temp == ((0.25, None, 0.25), (0.75, 1.0, None), (None, 0.5, 1.0),
                                         (1.0, 0.5, 0.25))


def test_zero_column_gives_nan_everywhere():
    got = _run('zero_column_gives_nan')
    assert got.column_max[1] == 0.0 and all(math.isnan(row[1]) for row in got.taxa_appear_core_temp)
    assert got.taxa_means[1].value == 0.0


def test_zero_maximum_with_negative_value_gives_minus_inf_and_negative_maximum_flips_sign():
    got = _run('zero_max_negative_and_na')
    first = [row[0] for row in got.taxa_appear_core_temp]
    assert math.isnan(first[0]) and first[1] == -math.inf and first[2] is None
    assert [row[1] for row in got.taxa_appear_core_temp] == [2.0, 4.0, 1.0]


def test_division_is_one_correctly_rounded_ieee_quotient():
    got = _run('inexact_partial_sums_flagged')
    assert [row[0] for row in got.taxa_appear_core_temp] == [float(Fraction(v) / Fraction(0.3))
                                                              for v in (0.1, 0.2, 0.3)]


def test_global_maximum_is_a_counterexample():
    assert all(v['global_max_differs'] for v in _json('arithmetic_check.json')['naive_counterexamples'].values())


def test_jeffreys_prior_is_one_half_per_taxon():
    for name in CASES:
        got = _run(name)
        assert got.jeffreys_prior == (0.5,) * got.num_taxa


def test_input_is_not_modified():
    matrix, n, gibbs = check.module_inputs(CASES['na_skipped_in_mean_and_max'])
    before = copy.deepcopy(matrix)
    array = np.array(matrix)
    sp.prepare_sampler(matrix, n, gibbs=gibbs)
    sp.prepare_sampler(array, n, gibbs=gibbs)
    assert json.dumps(matrix) == json.dumps(before) and np.array_equal(array, np.array(before), equal_nan=True)


# --- colMeans(na.rm = T) ----------------------------------------------------------------------------------------

def test_col_means_skip_na_and_count_only_values():
    got = _run('na_skipped_in_mean_and_max')
    assert [(str(m.exact), m.count) for m in got.taxa_means] == [('8/3', 3), ('8/3', 3), ('4', 3)]
    assert _json('arithmetic_check.json')['naive_counterexamples']['na_skipped_in_mean_and_max'][
        'colmeans_without_na_rm'] == ['NA', 'NA', 'NA']


def test_col_means_column_of_na_only_is_nan():
    means = sp.col_means_na_rm([[NAN, 1.0], [NAN, 3.0]])
    assert math.isnan(means[0].value) and means[0].exact is None and means[0].count == 0
    assert means[1].value == 2.0 and means[1].accumulator_independent


def test_col_means_flag_accumulator_dependent_means():
    got = _run('inexact_partial_sums_flagged')
    first = got.taxa_means[0]
    assert not first.accumulator_independent and first.exact == (Fraction(0.1) + Fraction(0.2) + Fraction(0.3)) / 3
    assert first.value == float(first.exact)  # correctly rounded, not claimed as the R result
    assert got.taxa_means[1].accumulator_independent and got.taxa_means[1].value == 2.0


def test_col_means_use_the_raw_not_the_normalised_matrix():
    got = _run('rowwise_flatten_three_by_four')
    assert got.taxa_means[3].value == 6.0  # raw column 3, 6, 9; the normalised column would give 2/3


# --- rejected R semantics ---------------------------------------------------------------------------------------

def test_column_of_na_only_is_rejected_because_max_gives_minus_inf():
    with pytest.raises(sp.UnsupportedSourceDomain, match='-Inf'):
        sp.prepare_sampler([[NAN, 1.0], [NAN, 2.0]], 2, gibbs=False)


@pytest.mark.parametrize('matrix', [[[1.0, 2.0, 3.0]], [[1.0], [2.0]], [[1.0]]])
def test_one_row_or_one_column_is_rejected_because_r_drops_to_a_vector(matrix):
    with pytest.raises(sp.UnsupportedSourceDomain, match='drops to a vector'):
        sp.prepare_sampler(matrix, len(matrix[0]), gibbs=False)


@pytest.mark.parametrize('num_taxa', [2, 4, 1, 0])
def test_num_taxa_must_equal_ncol(num_taxa):
    with pytest.raises(sp.UnsupportedSourceDomain):
        sp.prepare_sampler([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], num_taxa, gibbs=False)


@pytest.mark.parametrize('matrix', [
    [[1.0, math.inf], [2.0, 3.0]], [[1.0, -math.inf], [2.0, 3.0]], [[-0.0, 1.0], [0.0, 2.0]],
    [[True, False], [False, True]], [['1', '2'], ['3', '4']], [[1.0, 2.0], [3.0]], [[1.0, None], [2.0, 3.0]],
    [1.0, 2.0], [[[1.0]]]])
def test_outside_the_checked_domain_is_rejected(matrix):
    with pytest.raises(sp.UnsupportedSourceDomain):
        sp.prepare_sampler(matrix, 2, gibbs=False)


@pytest.mark.parametrize('num_taxa', [2.0, '2', None, True])
def test_num_taxa_must_be_an_integer(num_taxa):
    with pytest.raises(sp.UnsupportedSourceDomain):
        sp.prepare_sampler([[1.0, 2.0], [3.0, 4.0]], num_taxa, gibbs=False)


def test_integer_matrix_is_accepted_as_numeric():
    got = sp.prepare_sampler(np.array([[1, 3], [2, 6]]), 2, gibbs=False)
    assert got.values == (0.5, 0.5, 1.0, 1.0)


# --- packet binding -----------------------------------------------------------------------------------------------

def test_source_excerpt_matches_binding_and_contains_the_literal_lines():
    binding = _json('source_binding.json')
    raw = (PACKET / 'source_excerpt.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == binding['portable_excerpt_sha256']
    text = '\n'.join(s['text'] for s in json.loads(raw)['spans'])
    for line in ('gibbs <- F', 'posterior$how_many_taxa <- floor(num_taxa/2)', 'posterior$how_many_taxa <- num_taxa',
                 'taxa_information$values <- c(t(taxa.appear.core))',
                 'taxa_information$taxa_means <- colMeans(taxa.appear.core, na.rm=T)',
                 'taxa.appear.core.temp[,j] <- taxa.appear.core[,j]/max(taxa.appear.core[,j], na.rm=T)',
                 'taxa_information$values <- c(t(taxa.appear.core.temp))',
                 'posterior$jeffreys_prior <- rep(0.5,num_taxa)', 'for(c in 1:num_taxa) combs[c] <- choose(num_taxa,c)',
                 'parameter_stage_weights <- function('):
        assert line in text
    assert binding['assignment_sites_without_comments']['gibbs'] == {'main_v1.2.R': [624]}
    assert binding['assignment_sites_without_comments']['F'] == {} == binding['assignment_sites_without_comments']['T']
    assert binding['brace_depth_at_line_start']['main_v1.2.R']['624'] == 0
    assert binding['brace_depth_at_line_start']['src/mcmc_conduction.R']['75'] == 0
    assert binding['main_stray_closing_braces'] == [740, 788, 829]
    assert binding['historical_reproduction_ready'] is False and binding['run_binding'] is None


def test_readers_are_documented_and_taxa_means_has_none():
    occ = _json('source_binding.json')['occurrences_without_comments']
    assert occ['$taxa_means'] == {'src/mcmc_conduction.R': [67]}
    assert occ['jeffreys_prior'] == {'src/mcmc_conduction.R': [79], 'src/mcmc_functions_v1.6.R': [1216]}
    assert occ['combs'] == {'src/mcmc_prepare.R': [1702, 1703]}
    assert 1276 in occ['taxa_information']['src/mcmc_functions_v1.6.R']
    assert _json('source_binding.json')['parameter_stage_weights_reads_taxa_appear_core_argument'] is False


def test_audit_selects_only_this_step_and_keeps_random_steps_documented():
    audit = _json('determinism_audit.json')
    by_id = {c['id']: c for c in audit['candidates']}
    chosen = by_id['sampler_preparation_53_79']
    assert audit['selected_for_implementation'] == chosen['id'] and not audit['final_variant_selected']
    assert not chosen['random_state'] and not chosen['package_semantics_needed'] and not audit['R_execution']
    assert by_id['weightedMedian']['package_semantics_needed']
    assert all(by_id[k]['random_state'] for k in ('rmvnorm_rcpp_133', 'prior_samples_seed_2022',
                                                    'start_values_rbeta', 'tf_variance_sample'))
    assert by_id['combs_1702_1703']['decision'].startswith('documented only')


def test_fixture_and_results_are_marked_synthetic():
    fixture = _json('synthetic_fixture.json')
    assert fixture['historical_reproduction_ready'] is False and fixture['selected_final_variant'] is None
    arithmetic = _json('arithmetic_check.json')
    assert arithmetic['kind'] == 'synthetic_fragment_check_not_historical_reproduction'
    assert not (arithmetic['historical_reproduction_ready'] or arithmetic['final_variant_selected']
                or arithmetic['R_execution'])
    assert all(h['agrees'] for h in arithmetic['hand_expectations'].values())
