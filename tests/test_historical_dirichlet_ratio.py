"""Dirichlet ratio of parameter_stage_weights (mcmc_functions_v1.6.R 1269-1274): reference, hand values, R record,
domain; explicit weights only, no rdirichlet draw."""
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / 'docs/thesis_research/literal_dirichlet_ratio_2026_11_03'
sys.path.insert(0, str(ROOT / 'tools'))
import historical_dirichlet_ratio as dr  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, PACKET / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference = _load('dirichlet_reference', 'precise_reference.py')
check = _load('dirichlet_check', 'check_dirichlet_ratio.py')


def _json(name):
    return json.loads((PACKET / name).read_text(encoding='utf-8'))


FIX = _json('synthetic_fixture.json')
RATIO, DENS = FIX['ratio_cases'], FIX['density_cases']
val = lambda t: float(Fraction(t))


def _run(name):
    c = RATIO[name]
    return dr.dirichlet_ratio([val(v) for v in c['old_weights']], [val(v) for v in c['new_weights']], c['num_taxa'])


# --- independent reference, stored files, hand values --------------------------------------------------------------

@pytest.mark.parametrize('name', list(RATIO))
def test_module_equals_the_independent_pipeline_reference(name):
    c = RATIO[name]
    old, new = [val(v) for v in c['old_weights']], [val(v) for v in c['new_weights']]
    got, ref = check.as_record(_run(name)), reference.ratio(old, new, c['num_taxa'])
    assert got['densities'] == [check.special(v) for v in ref['densities']]
    for key in ('weights_ratio', 'proposal_ratio', 'parameter_ratio_before_na', 'parameter_ratio'):
        assert got[key] == check.special(ref[key])


@pytest.mark.parametrize('name', list(RATIO))
def test_stored_result_matches_a_fresh_run(name):
    assert _json('synthetic_result.json')['ratio_cases'][name] == check.as_record(_run(name))


@pytest.mark.parametrize('name', list(DENS))
def test_density_closed_forms(name):
    c = DENS[name]
    got = dr.ddirichlet([val(v) for v in c['x']], [val(v) for v in c['alpha']])
    assert got.value == float(reference.closed_form(c['hand_closed_form']))
    assert got.value == reference.density_pipeline([val(v) for v in c['x']], [val(v) for v in c['alpha']])


def test_identical_weights_give_exactly_one():
    got = _run('identical_weights')
    assert (got.weights_ratio, got.proposal_ratio, got.parameter_ratio) == (1.0, 1.0, 1.0)


def test_two_taxa_closed_forms_within_one_ulp():
    got = _run('two_taxa_closed_form')
    for v, form in ((got.weights_ratio, '2/sqrt(3)'), (got.proposal_ratio, 'sqrt(6)/4'),
                    (got.parameter_ratio, '1/sqrt(2)')):
        assert check.ulps(v, float(reference.closed_form(form))) <= 1


def test_hand_records_agree():
    hand = _json('arithmetic_check.json')['hand_expectations']
    assert all(v is True or (isinstance(v, int) and not isinstance(v, bool) and v <= 1)
               for case in hand.values() for v in case.values())


# --- literal semantics ----------------------------------------------------------------------------------------------

def test_overflow_gives_nan_and_line_1273_sets_zero():
    got = _run('overflow_nan_becomes_zero')
    assert got.weights_ratio == math.inf and got.proposal_ratio == 0.0
    assert math.isnan(got.parameter_ratio_before_na) and got.parameter_ratio == 0.0 and got.na_replaced


def test_infinite_ratio_is_not_replaced():
    # is.na(Inf) is FALSE in R: an infinite parameter ratio survives line 1273
    got = dr.dirichlet_ratio([0.5, 0.5], [1e-310, 1.0], 2)
    assert got.parameter_ratio == got.parameter_ratio_before_na and not got.na_replaced


def test_unnormalised_weights_are_not_renormalised():
    got = _run('unnormalised_weights_kept')
    assert got.weights_numerator.normalisation_skipped and got.weights_numerator.row_sum_x == 2.0
    naive = _json('arithmetic_check.json')['naive_counterexamples']['unnormalised_weights_kept']
    assert naive['renormalisation_differs']


def test_proposal_without_dirichlet_spread_is_what_the_source_computes():
    naive = _json('arithmetic_check.json')['naive_counterexamples']
    assert all(v['proposal_with_spread_differs'] for k, v in naive.items() if k != 'identical_weights')
    assert all(v['symmetric_proposal_differs'] for k, v in naive.items() if k != 'identical_weights')


def test_correctly_rounded_lgamma_matches_known_values():
    assert dr.lgamma_cr(3.0) == math.log(2.0) and dr.lgamma_cr(1.0) == 0.0 == dr.lgamma_cr(2.0)
    assert dr.lgamma_cr(0.5) == float(reference.closed_form('sqrt(pi)').ln())
    assert sum(v['cpython_lgamma_densities_differing'] for v in
               _json('arithmetic_check.json')['naive_counterexamples'].values()) > 0  # math.lgamma is not enough


def test_accumulator_flags():
    # alpha = (1, 1): every row sum is exact (0, 2, 1), the density is exactly Gamma(2) = 1
    exact = dr.ddirichlet([0.5, 0.5], [1.0, 1.0])
    assert exact.accumulator_independent and exact.value == 1.0
    # sums of transcendental doubles are flagged; R 3.5.0 still agreed in every such flagged case except lgamma(0.4)
    assert not _run('identical_weights').weights_numerator.accumulator_independent
    assert not _run('inexact_weights').proposal_numerator.accumulator_independent


def test_thirty_three_taxa_runs_and_is_finite():
    got = _run('thirty_three_taxa')
    assert got.num_taxa == 33 and math.isfinite(got.parameter_ratio) and got.parameter_ratio > 0


# --- R 3.5.0 record ---------------------------------------------------------------------------------------------

def test_r_record_lists_runtime_and_measured_ulps():
    r = _json('r_crosscheck_result.json')
    assert r['runtime'] == {'R': 'R version 3.5.0 (2018-04-23)', 'LaplacesDemon': '16.1.6',
                            'ddirichlet_namespace': 'LaplacesDemon'}
    assert (r['values_compared'], r['bit_equal'], r['max_ulps'], r['special_mismatch']) == (62, 59, 4, 0)
    off = {n for n, row in r['ratio_cases'].items() if any(u for v in row.values() for u in v['ulps'])}
    assert off == {'inexact_weights'}
    assert all(not any(v['ulps'].values()) for v in r['density_cases'].values())


def test_r_matches_the_specials_and_the_1273_replacement():
    row = _json('r_crosscheck_result.json')['ratio_cases']['overflow_nan_becomes_zero']
    assert row['weights_ratio']['R'] == ['Inf'] and row['parameter_ratio']['R'] == [0.0]


# --- rejected domain ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize('old,new,n', [([0.5, 0.5], [0.5, 0.5], 3), ([0.5, 0.5], [0.2, 0.3, 0.5], 2),
                                       ([0.5], [0.5], 1), ([0.0, 1.0], [0.5, 0.5], 2), ([-0.5, 1.5], [0.5, 0.5], 2),
                                       ([math.nan, 1.0], [0.5, 0.5], 2), ([math.inf, 1.0], [0.5, 0.5], 2),
                                       ([True, 1.0], [0.5, 0.5], 2), (['0.5', 0.5], [0.5, 0.5], 2),
                                       ([0.5, 0.5], [0.5, 0.5], 2.0), ([0.5, 0.5], [0.5, 0.5], True)])
def test_outside_the_checked_domain_is_rejected(old, new, n):
    with pytest.raises(dr.UnsupportedSourceDomain):
        dr.dirichlet_ratio(old, new, n)


@pytest.mark.parametrize('x,alpha', [([0.5, 0.5], [0.0, 1.0]), ([0.5, 0.5], [-1.0, 1.0]), ([0.0, 1.0], [0.5, 0.5]),
                                     ([0.5, 0.5], [0.5]), ([np.nan, 0.5], [0.5, 0.5])])
def test_ddirichlet_stops_are_rejected(x, alpha):
    with pytest.raises(dr.UnsupportedSourceDomain):
        dr.ddirichlet(x, alpha)


def test_numpy_inputs_are_accepted():
    got = dr.dirichlet_ratio(np.array([0.5, 0.5]), np.array([0.25, 0.75]), np.int64(2))
    assert got.parameter_ratio == _run('two_taxa_closed_form').parameter_ratio


# --- packet binding -----------------------------------------------------------------------------------------------

def test_binding_records_literal_lines_and_provider():
    b = _json('source_binding.json')
    raw = (PACKET / 'source_excerpt.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == b['portable_excerpt_sha256']
    text = '\n'.join(s['text'] for s in json.loads(raw)['spans'])
    for line in ('weights_ratio <- ddirichlet(new_weights,rep(0.5,num_taxa)) / ddirichlet(old_weights,rep(0.5,num_taxa))',
                 'proposal_ratio <- ddirichlet(old_weights,new_weights)/ddirichlet(new_weights,old_weights)',
                 'if(is.na(parameter_ratio)) parameter_ratio <- 0', 'if(any(rowSums(x) != 1)) x / rowSums(x)'):
        assert line in text
    assert b['proposal_ratio_contains_dirichlet_spread'] is False and b['ddirichlet_normalisation_line_assigns'] is False
    assert b['parameter_ratio_readers_mcmc_conduction'] == [97, 151, 302]
    assert b['historical_reproduction_ready'] is False and b['run_binding'] is None


def test_audit_keeps_random_steps_documented():
    audit = _json('determinism_audit.json')
    by_id = {c['id']: c for c in audit['candidates']}
    assert audit['selected_for_implementation'] == 'ddirichlet_ratio_1269_1274'
    assert not by_id['ddirichlet_ratio_1269_1274']['random_state']
    assert all(by_id[k]['random_state'] for k in ('rdirichlet_1248', 'reset_1253_1257', 'acceptance_runif_304'))
    assert FIX['historical_reproduction_ready'] is False and FIX['selected_final_variant'] is None
