"""Historical gamma prior grid (mcmc_prepare.R 114-121, 1264-1281): independent reference, R quirks, domain."""
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
PACKET = ROOT / 'docs/thesis_research/literal_gamma_prior_grid_2026_11_02'
sys.path.insert(0, str(ROOT / 'tools'))
import historical_gamma_prior_grid as gpg  # noqa: E402

_spec = importlib.util.spec_from_file_location('gamma_grid_reference', PACKET / 'rational_reference.py')
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)


def _json(name):
    return json.loads((PACKET / name).read_text(encoding='utf-8'))


FIXTURE = _json('synthetic_fixture.json')
GRID_CASES, CHAIN_CASES = FIXTURE['grid_cases'], FIXTURE['chain_cases']
GRIDS = {upper: reference.r_seq(upper) for upper in (50000.0, 5000.0)}


def _grid(name):
    case = GRID_CASES[name]
    return gpg.gamma_grid_intersection(case['mean_pann'], case['sd_pann'], upper=case['upper'])


def _chain(name):
    c = CHAIN_CASES[name]
    return gpg.prior_gamma_chain(c['mean_pann'], c['sd_pann'], c['sd_tmp'], recent=c['recent'],
                                 min_age_at_least_2000=c['min_age_at_least_2000'])


@pytest.mark.parametrize('name', list(GRID_CASES))
def test_grid_matches_independent_reference_bitwise(name):
    case = GRID_CASES[name]
    exact = reference.grid_reference(case['mean_pann'], case['sd_pann'], case['upper'], GRIDS[case['upper']])
    got = _grid(name)
    assert (got.grid_length, got.index, got.shape, got.rate, got.rate_expectation) == \
        (exact['grid_length'], exact['index'], exact['shape'], exact['rate'], exact['rate_expectation'])
    assert got.index + 1 == case['hand_r_index'] and got.shape == pytest.approx(case['hand_shape_approx'], abs=1e-9)


@pytest.mark.parametrize('name', list(CHAIN_CASES))
def test_chain_matches_independent_reference(name):
    c, got = CHAIN_CASES[name], _chain(name)
    exact = reference.chain_reference(c['mean_pann'], c['sd_pann'], c['sd_tmp'], recent=c['recent'],
                                      min_age_at_least_2000=c['min_age_at_least_2000'], grid=GRIDS[50000.0])
    assert (got.default.index, got.default.shape, got.default.rate, got.diff_rate, got.variance_doubled,
            got.sd_pann, got.sd_tmp, got.final.index, got.final.shape, got.final.rate, got.rate_after_recent_start) \
        == tuple(exact[k] for k in ('default_index', 'default_shape', 'default_rate', 'diff_rate', 'variance_doubled',
                                    'sd_pann', 'sd_tmp', 'final_index', 'final_shape', 'final_rate',
                                    'rate_after_recent_start'))


@pytest.mark.parametrize('name', list(GRID_CASES) + list(CHAIN_CASES))
def test_stored_files_match_a_fresh_module_run(name):
    stored, expected = _json('synthetic_result.json')['cases'][name], _json('rational_expectations.json')['cases'][name]
    if name in GRID_CASES:
        got = _grid(name)
        assert (stored['index'], stored['shape'], stored['rate']) == (got.index, got.shape, got.rate)
        assert expected['shape_hex'] == got.shape.hex()
    else:
        got = _chain(name)
        assert (stored['final_index'], stored['final_rate'], stored['rate_after_recent_start']) == \
            (got.final.index, got.final.rate, got.rate_after_recent_start) == \
            (expected['final_index'], expected['final_rate'], expected['rate_after_recent_start'])


# --- seq(0.1, upper, by = 0.1) ---------------------------------------------------------------------------------

def test_seq_50000_has_500000_values_ending_exactly_at_the_bound():
    grid = gpg.r_seq_by(0.1, 50000.0, 0.1)
    assert len(grid) == 500000 and grid[0] == 0.1 and grid[-1] == 50000.0
    assert (50000.0 - 0.1) / 0.1 == 499999.0  # fuzz cannot change the count
    assert grid.tolist() == GRIDS[50000.0]


def test_seq_50000_is_invariant_to_fuzz_and_pmin():
    for kw in (dict(fuzz=False), dict(clamp=False), dict(fuzz=False, clamp=False)):
        assert reference.r_seq(50000.0, **kw) == GRIDS[50000.0]


def test_seq_5000_copy_depends_on_fuzz_and_pmin():
    grid = gpg.r_seq_by(0.1, 5000.0, 0.1)
    assert len(grid) == 50000 and grid[-1] == 5000.0 and grid.tolist() == GRIDS[5000.0]
    assert (5000.0 - 0.1) / 0.1 < 49999  # without the 1e-10 fuzz one value is lost
    assert len(reference.r_seq(5000.0, fuzz=False)) == 49999
    assert reference.r_seq(5000.0, clamp=False)[-1] == 5000.000000000001  # pmin removes the overshoot


def test_grid_values_are_two_rounded_double_operations_not_decimals():
    grid = gpg.r_seq_by(0.1, 50000.0, 0.1)
    for k in (0, 1, 2, 29, 249, 61, 123456, 499998, 499999):
        assert grid[k] == float(Fraction(0.1) + Fraction(float(k * Fraction(0.1))))
    assert grid[249] == 25.000000000000004 != 25.0 and grid[2] == 0.30000000000000004


def test_fused_rounding_would_change_grid_values_but_no_fixture_decision():
    check = _json('arithmetic_check.json')['seq_sensitivity']
    assert check['50000']['variants']['fused_rounding']['values_differing'] > 0
    assert check['50000']['variants']['fused_rounding']['case_indices_changed'] == []
    assert check['5000']['variants']['no_fuzz']['length'] == 49999


# --- which.min -------------------------------------------------------------------------------------------------

def test_which_min_takes_the_first_of_equal_minima():
    assert gpg.r_which_min([3.0, 1.0, 1.0]) == 1
    assert gpg.r_which_min([2.0, 2.0, 2.0]) == 0
    assert gpg.r_which_min([math.inf, math.inf]) == 0


def test_which_min_rejects_na_instead_of_skipping():
    with pytest.raises(gpg.UnsupportedSourceDomain):
        gpg.r_which_min([1.0, math.nan])
    with pytest.raises(gpg.UnsupportedSourceDomain):
        gpg.r_which_min([])


# --- hand anchors and counterexamples --------------------------------------------------------------------------

def test_hand_anchor_centre_25():
    got = _grid('centre_25')
    assert got.index == 249 and got.shape == 25.000000000000004 and got.rate == 0.05
    assert got.rate_expectation == 0.05000000000000001 != got.rate  # rate1 and rate2 differ in the last bit


def test_midpoint_goes_to_the_lower_grid_point():
    got = _grid('midpoint_6_25')
    assert got.closed_form_shape == 6.25 and got.shape == 6.2
    naive = _json('arithmetic_check.json')['naive_counterexamples']['midpoint_6_25']
    assert naive['nearest_grid_shape'] == 6.3 and naive['nearest_grid_differs']


def test_nearest_grid_point_to_the_closed_form_is_not_the_source_rule():
    got = _grid('asymmetric_nearest_fails')
    assert got.closed_form_shape == pytest.approx(2.0501033, abs=1e-6) and got.shape == 2.0
    grid = GRIDS[50000.0]
    f = lambda k: abs(math.sqrt(grid[k]) / 220 - grid[k] / 315)
    assert abs(grid[20] - got.closed_form_shape) < abs(grid[19] - got.closed_form_shape)  # 2.1 is nearer
    assert f(19) < f(20)  # but 2.0 has the smaller objective


def test_closed_form_beyond_the_grid_jumps_to_the_first_point():
    got = _grid('beyond_upper_jumps_to_first')
    assert got.closed_form_shape == 62500.0 and got.index == 0 and got.shape == 0.1 and got.at_lower_boundary
    grid = GRIDS[50000.0]
    assert abs(math.sqrt(grid[0]) / 10 - grid[0] / 2500) < abs(math.sqrt(grid[-1]) / 10 - grid[-1] / 2500)


def test_just_beyond_the_5000_bound_hits_the_clamped_last_point():
    got = _grid('just_beyond_5000_last_point')
    assert got.at_upper_boundary and got.index == 49999 and got.shape == 5000.0
    assert _grid('same_input_upper_50000').shape == 5012.6  # same inputs, mcmc_prepare.R bound


def test_grid_rate_is_the_sqrt_branch_not_shape_over_mean():
    for name in GRID_CASES:
        got = _grid(name)
        assert got.rate == math.sqrt(got.shape) / GRID_CASES[name]['sd_pann']
    assert _json('arithmetic_check.json')['naive_counterexamples']['midpoint_6_25']['rate_differs']


# --- chain over mcmc_prepare.R --------------------------------------------------------------------------------

def test_doubling_needs_recent_and_min_age():
    assert _chain('chain_recent_after_2000').variance_doubled
    assert not _chain('chain_recent_older').variance_doubled
    assert not _chain('chain_not_recent').variance_doubled


def test_doubling_changes_final_but_not_default_or_period_copies():
    got = _chain('chain_recent_after_2000')
    assert (got.sd_pann, got.sd_tmp) == (200.0, 3.0)
    assert got.default.shape == got.pann_period_shape == 25.000000000000004 and got.pann_period_rate == 0.05
    assert got.final.shape == 6.2 and got.rgamma_1302_rate == got.final.rate == 0.012449899597988732


def test_recent_start_overwrites_rate_with_shape_over_mean():
    assert _chain('chain_recent_after_2000').rate_after_recent_start == 6.2 / 500 == 0.0124
    assert _chain('chain_recent_older').rate_after_recent_start == 0.05000000000000001
    assert _chain('chain_not_recent').rate_after_recent_start == 0.05


def test_diff_rate_uses_the_undoubled_default_shape():
    got = _chain('chain_recent_after_2000')
    assert got.diff_rate == got.default.shape / 10 == 2.5000000000000004


def test_doubling_by_two_is_exact():
    for sd in (150.0, 0.1, 123.456789, 1e-300):
        assert Fraction(sd * 2) == 2 * Fraction(sd)


# --- domain ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('bad', [0, -1.0, math.nan, math.inf, True, '500', [500.0, 600.0], None, np.array([1.0])])
def test_rejects_non_scalar_or_non_positive_priors(bad):
    with pytest.raises(gpg.UnsupportedSourceDomain):
        gpg.gamma_grid_intersection(bad, 100, upper=50000.0)
    with pytest.raises(gpg.UnsupportedSourceDomain):
        gpg.gamma_grid_intersection(500, bad, upper=50000.0)


def test_rejects_an_upper_bound_without_source_copy():
    with pytest.raises(gpg.UnsupportedSourceDomain):
        gpg.gamma_grid_intersection(500, 100, upper=10000.0)


def test_rejects_non_boolean_site_conditions():
    with pytest.raises(gpg.UnsupportedSourceDomain):
        gpg.prior_gamma_chain(500, 100, 1.5, recent=1, min_age_at_least_2000=True)
    with pytest.raises(gpg.UnsupportedSourceDomain):
        gpg.prior_gamma_chain(500, 100, 1.5, recent=True, min_age_at_least_2000=None)


def test_numpy_scalars_are_accepted_like_floats():
    got = gpg.gamma_grid_intersection(np.float64(500), np.int64(100), upper=50000.0)
    assert got.index == 249 and got.shape == _grid('centre_25').shape


# --- packet binding --------------------------------------------------------------------------------------------

def test_source_excerpt_matches_binding_and_contains_the_literal_lines():
    binding = _json('source_binding.json')
    raw = (PACKET / 'source_excerpt.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == binding['portable_excerpt_sha256']
    text = '\n'.join(s['text'] for s in json.loads(raw)['spans'])
    for line in ('shape_temp <- seq(0.1,50000,by=0.1)', 'shape1 <- seq(0.1,5000,by=0.1)',
                 'intersect_id <- which.min(abs((rate2-rate1) - 0))', 'if( recent & min(age) >= 2000 ){',
                 'rate <- shape/prior.recent$mean.pann', 'diff_rate <- shape/10', 'stad_interstad <- FALSE',
                 'prior_pann_sample <- rgamma(10000,shape=shape,rate=rate)'):
        assert line in text
    assert binding['historical_reproduction_ready'] is False and binding['run_binding'] is None


def test_audit_selects_only_the_grid_and_keeps_weighted_median_open():
    audit = _json('determinism_audit.json')
    by_id = {c['id']: c for c in audit['candidates']}
    assert audit['selected_for_implementation'] == 'gamma_prior_grid' and not audit['final_variant_selected']
    assert not by_id['gamma_prior_grid']['random_state'] and not by_id['gamma_prior_grid']['package_semantics_needed']
    assert by_id['weightedMedian']['package_semantics_needed'] and by_id['prior_samples_seed_2022']['random_state']
    assert 'not selected' in by_id['which_tf_missing']['decision']


def test_fixture_and_results_are_marked_synthetic():
    assert FIXTURE['historical_reproduction_ready'] is False and FIXTURE['selected_final_variant'] is None
    check = _json('arithmetic_check.json')
    assert check['kind'] == 'synthetic_fragment_check_not_historical_reproduction'
    assert not (check['historical_reproduction_ready'] or check['final_variant_selected'] or check['R_execution'])
    assert all(h['agrees'] for h in check['hand_expectations'].values())
