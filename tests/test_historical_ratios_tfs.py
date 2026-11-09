"""ratios_tfs_no_ages from explicit inputs. Artificial numbers only; no draw, no sampler, no compilation. Not bit-identical with R, and the tests keep that marked."""
import json
import math
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import historical_ratios_tfs as rt  # noqa: E402

PACKET = ROOT / 'docs/thesis_research/literal_ratios_tfs_2026_11_09'


def _json(name):
    return json.loads((PACKET / name).read_text(encoding='utf-8'))


FIXTURE, RESULT, REFERENCE, CROSS, BINDING, ARITHMETIC = (_json(n) for n in ('synthetic_fixture.json', 'synthetic_result.json', 'reference_expectations.json', 'r_crosscheck_result.json', 'source_binding.json', 'arithmetic_check.json'))
NAMES = list(FIXTURE['cases'])
ARGS = ('tann_range', 'pann_range', 'new_tann', 'old_tann', 'tann_dists', 'new_pann', 'old_pann', 'pann_dists', 'lower_tann', 'upper_tann', 'sds_tann_trunc', 'lower_pann', 'upper_pann', 'sds_pann_trunc')
GRID_T, GRID_P = [0.0, 1.0, 2.0, 3.0], [0.0, 100.0, 200.0, 300.0]
BASE = dict(tann_range=GRID_T, pann_range=GRID_P, new_tann=[1.0], old_tann=[2.0], tann_dists=[[0.1, 0.4, 0.3, 0.2]], new_pann=[100.0], old_pann=[200.0], pann_dists=[[0.5, 0.25, 0.125, 0.125]],
            lower_tann=[0.0], upper_tann=[3.0], sds_tann_trunc=[0.5], lower_pann=[0.0], upper_pann=[300.0], sds_pann_trunc=[40.0])


def value(t):
    return 0.0 if t == '0' else float.fromhex(t)


def arguments(c):
    return {k: ([[value(t) for t in row] for row in c[k]] if k.endswith('dists') else [value(t) for t in c[k]]) for k in ARGS}


def run(name):
    c = FIXTURE['cases'][name]
    return rt.ratios_tfs_no_ages(c['num_taxa'], **arguments(c))


def token(v):
    if math.isnan(v):
        return 'NaN'
    if math.isinf(v):
        return 'Inf' if v > 0 else '-Inf'
    if v == 0:
        return '-0' if math.copysign(1.0, v) < 0 else '0'
    return v.hex()


def with_changes(**changes):
    return {**BASE, **changes}


def test_the_backend_says_it_is_not_bit_identical():
    assert 'not bit-identical' in rt.BACKEND and rt.ratios_tfs_no_ages(1, **BASE).backend == rt.BACKEND


@pytest.mark.parametrize('x,mu,sigma,expected', [(0.0, 0.0, 1.0, '0.398942280401432677939946'), (1.0, 0.0, 1.0, '0.241970724519143349797830'), (-1.0, 0.0, 2.0, '0.176032663382149970')])
def test_own_density_matches_known_values(x, mu, sigma, expected):
    assert rt.dnorm_own(x, mu, sigma) == pytest.approx(float(expected), rel=4e-16)


@pytest.mark.parametrize('x,mu,sigma,expected', [(0.0, 0.0, 1.0, '0.5'), (1.0, 0.0, 1.0, '0.841344746068542948585232'), (-1.96, 0.0, 1.0, '0.0249978951482204')])
def test_own_distribution_function_matches_known_values(x, mu, sigma, expected):
    assert rt.pnorm_own(x, mu, sigma) == pytest.approx(float(expected), rel=3e-15)


def test_density_is_scaled_by_the_interval_mass():
    p = rt.density_parts(1.0, 1.0, 0.5, 0.0, 3.0)
    assert p.scale == p.p_b - p.p_a and p.value == p.density / p.scale


@pytest.mark.parametrize('x', [-0.1, 3.1])
def test_a_value_outside_the_interval_has_density_zero(x):
    assert rt.d_truncnorm(x, 1.0, 0.5, 0.0, 3.0) == 0.0


@pytest.mark.parametrize('x', [0.0, 3.0])
def test_the_interval_is_closed(x):
    assert rt.d_truncnorm(x, 1.0, 0.5, 0.0, 3.0) > 0


def test_equal_bounds_divide_by_a_zero_scale():
    assert rt.d_truncnorm(1.0, 1.0, 0.5, 1.0, 1.0) == math.inf


@pytest.mark.parametrize('value,index', [(0.5, 0), (1.5, 1), (2.5, 2), (-5.0, 0), (9.0, 3), (2.0, 2), (0.49, 0), (0.51, 1)])
def test_grid_index_takes_the_first_minimum(value, index):
    assert rt.grid_index(GRID_T, value) == index


def test_grid_index_first_of_equal_distances_in_a_repeated_grid():
    assert rt.grid_index([1.0, 1.0, 1.0], 1.0) == 0


def test_tann_grid_ratio_divides_the_row_at_the_two_indices():
    assert rt.ratios_tfs_no_ages(1, **BASE).tann_tf_ratio == (0.4 / 0.3,)


def test_pann_grid_ratio_divides_the_row_at_the_two_indices():
    assert rt.ratios_tfs_no_ages(1, **BASE).pann_tf_ratio == (0.25 / 0.125,)


def test_the_density_ratio_takes_old_over_new_on_top():
    r = rt.ratios_tfs_no_ages(1, **BASE)
    top = rt.d_truncnorm(2.0, 1.0, 0.5, 0.0, 3.0)
    assert r.tann_dists_tf_ratio == (top / rt.d_truncnorm(1.0, 2.0, 0.5, 0.0, 3.0),)


def test_grid_ratio_by_zero_is_infinite():
    assert rt.ratios_tfs_no_ages(1, **with_changes(tann_dists=[[0.1, 0.4, 0.0, 0.5]])).tann_tf_ratio == (math.inf,)


def test_grid_ratio_zero_over_zero_is_nan():
    assert math.isnan(rt.ratios_tfs_no_ages(1, **with_changes(tann_dists=[[0.1, 0.0, 0.0, 0.9]])).tann_tf_ratio[0])


def test_density_ratio_zero_when_old_is_outside():
    assert rt.ratios_tfs_no_ages(1, **with_changes(old_tann=[3.5], upper_tann=[3.0])).tann_dists_tf_ratio == (0.0,)


def test_density_ratio_infinite_when_new_is_outside():
    assert rt.ratios_tfs_no_ages(1, **with_changes(new_tann=[3.5], upper_tann=[3.0])).tann_dists_tf_ratio == (math.inf,)


def test_underflow_of_both_densities_is_nan():
    assert math.isnan(rt.ratios_tfs_no_ages(1, **with_changes(sds_tann_trunc=[1e-3])).tann_dists_tf_ratio[0])


def test_unlist_order():
    r = rt.ratios_tfs_no_ages(1, **BASE)
    assert rt.unlisted(r) == [r.tann_tf_ratio[0], r.pann_tf_ratio[0], r.tann_dists_tf_ratio[0], r.pann_dists_tf_ratio[0]]


def test_indices_are_kept_per_taxon():
    assert rt.ratios_tfs_no_ages(1, **BASE).indices == ((1, 2, 1, 2),)


def test_parts_are_in_call_order():
    r = rt.ratios_tfs_no_ages(1, **BASE)
    assert r.parts[0][0].value == rt.d_truncnorm(2.0, 1.0, 0.5, 0.0, 3.0) and r.parts[0][1].value == rt.d_truncnorm(1.0, 2.0, 0.5, 0.0, 3.0)
    assert r.parts[0][2].value == rt.d_truncnorm(200.0, 100.0, 40.0, 0.0, 300.0) and r.parts[0][3].value == rt.d_truncnorm(100.0, 200.0, 40.0, 0.0, 300.0)


@pytest.mark.parametrize('factors', [[2.0, 3.0], [0.5, 0.25, 8.0], [1.5] * 5, [1e10, 1e-10]])
def test_long_double_product_equals_the_exact_product_when_it_fits(factors):
    exact = Fraction(1)
    for f in factors:
        exact *= Fraction(f)
    if exact.numerator.bit_length() <= 64:
        assert rt.long_double_prod(factors) == float(exact)


def test_long_double_keeps_bits_a_double_loses():
    factors = [1.0 + 2.0 ** -40, 1.0 + 2.0 ** -40]
    assert rt.long_double_prod(factors) == float(Fraction(factors[0]) * Fraction(factors[1]))


def test_long_double_rounds_twice_and_can_differ_from_one_correct_rounding():
    a, b = float(2 ** 53 - 1), float(2 ** 12 + 1)
    exact = int(a) * int(b)
    excess = exact.bit_length() - 64
    low, half, top = exact & ((1 << excess) - 1), 1 << (excess - 1), exact >> excess
    top += low > half or (low == half and top & 1)
    assert rt.long_double_prod([a, b]) == float(top << excess)
    assert rt.long_double_prod([a, b]) != float(Fraction(exact))


@pytest.mark.parametrize('factors,expected', [([1.0, math.inf], math.inf), ([-1.0, math.inf], -math.inf), ([0.0, 2.0], 0.0)])
def test_long_double_special_values(factors, expected):
    assert rt.long_double_prod(factors) == expected


@pytest.mark.parametrize('factors', [[0.0, math.inf], [math.inf, 0.0], [math.nan, 1.0], [1.0, math.nan], [math.inf, math.nan, 0.0]])
def test_long_double_nan_cases(factors):
    assert math.isnan(rt.long_double_prod(factors))


def test_long_double_negative_zero():
    assert math.copysign(1.0, rt.long_double_prod([-1.0, 0.0])) == -1.0


def test_long_double_product_above_dbl_max_is_infinite():
    assert rt.long_double_prod([1e308, 10.0]) == math.inf


def test_long_double_intermediate_above_dbl_max_can_come_back():
    assert rt.long_double_prod([1e308, 10.0, 1e-10]) == pytest.approx(1e299, rel=1e-15)


def test_long_double_range_is_rejected_rather_than_emulated():
    with pytest.raises(rt.UnsupportedSourceDomain, match='80-bit'):
        rt.long_double_prod([1e308] * 20)


@pytest.mark.parametrize('kwarg,bad', [('new_tann', [math.nan]), ('old_tann', [math.inf]), ('new_pann', [None]), ('old_pann', [math.nan]), ('tann_range', [0.0, math.nan, 2.0, 3.0]), ('pann_range', [0.0, math.inf, 2.0, 3.0]),
                                       ('lower_tann', [math.nan]), ('upper_tann', [math.inf]), ('lower_pann', [None]), ('upper_pann', [math.nan]), ('sds_tann_trunc', [math.nan]), ('sds_pann_trunc', [math.inf]),
                                       ('tann_dists', [[0.1, math.nan, 0.3, 0.2]]), ('pann_dists', [[0.5, 0.25, math.inf, 0.125]])])
def test_non_finite_inputs_are_rejected(kwarg, bad):
    with pytest.raises(rt.UnsupportedSourceDomain):
        rt.ratios_tfs_no_ages(1, **with_changes(**{kwarg: bad}))


@pytest.mark.parametrize('kwarg,bad', [('new_tann', [1.0, 2.0]), ('old_tann', []), ('new_pann', [1.0, 2.0]), ('old_pann', [1.0, 2.0]), ('lower_tann', [1.0, 2.0]), ('upper_tann', [1.0, 2.0]), ('sds_tann_trunc', [1.0, 2.0]),
                                       ('lower_pann', [1.0, 2.0]), ('upper_pann', [1.0, 2.0]), ('sds_pann_trunc', [1.0, 2.0]), ('tann_dists', [[0.1, 0.4, 0.3]]), ('pann_dists', [[0.5, 0.25, 0.125]]), ('tann_dists', [[0.1, 0.4, 0.3, 0.2], [0.1, 0.4, 0.3, 0.2]])])
def test_shapes_are_rejected(kwarg, bad):
    with pytest.raises(rt.UnsupportedSourceDomain):
        rt.ratios_tfs_no_ages(1, **with_changes(**{kwarg: bad}))


@pytest.mark.parametrize('n', [0, -1, 1.0, True, None])
def test_bad_num_taxa_is_rejected(n):
    with pytest.raises(rt.UnsupportedSourceDomain):
        rt.ratios_tfs_no_ages(n, **BASE)


@pytest.mark.parametrize('sigma', [0.0, -1.0])
def test_non_positive_sigma_is_rejected(sigma):
    with pytest.raises(rt.UnsupportedSourceDomain, match='positive'):
        rt.d_truncnorm(1.0, 1.0, sigma, 0.0, 3.0)


def test_inverted_bounds_are_rejected():
    with pytest.raises(rt.UnsupportedSourceDomain, match='exceeds'):
        rt.d_truncnorm(1.0, 1.0, 0.5, 3.0, 0.0)


def test_numpy_scalars_are_accepted():
    assert rt.d_truncnorm(np.float64(1.0), np.float64(1.0), np.float64(0.5), np.float64(0.0), np.float64(3.0)) > 0


def test_a_string_vector_is_rejected():
    with pytest.raises(rt.UnsupportedSourceDomain):
        rt.ratios_tfs_no_ages(1, **with_changes(new_tann='1'))


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('key', ['tann_tf_ratio', 'pann_tf_ratio', 'tann_dists_tf_ratio', 'pann_dists_tf_ratio'])
def test_stored_result_is_what_the_module_returns(name, key):
    assert RESULT['cases'][name][key] == [token(v) for v in getattr(run(name), key)]


@pytest.mark.parametrize('name', NAMES)
def test_stored_indices_and_product_are_what_the_module_returns(name):
    r = run(name)
    assert RESULT['cases'][name]['indices'] == [list(row) for row in r.indices] and RESULT['cases'][name]['transfer_ratio'] == token(rt.transfer_ratio(r))


@pytest.mark.parametrize('name', list(REFERENCE['cases']))
def test_grid_part_equals_the_independent_reference(name):
    r, ref = run(name), REFERENCE['cases'][name]
    assert [list(row) for row in r.indices] == ref['indices']
    for key in ('tann_tf_ratio', 'pann_tf_ratio'):
        for mine, true in zip(getattr(r, key), ref[key]):
            assert (math.isnan(mine) if true == 'NaN' else mine == math.inf if true == 'Inf' else mine == pytest.approx(float(Decimal(true)), rel=4e-16))


@pytest.mark.parametrize('name', [n for n in REFERENCE['cases'] if n not in ('tiny_sigma_saturates', 'wide_proposal_small_scale')])
def test_density_part_is_close_to_the_independent_reference(name):
    r, ref = run(name), REFERENCE['cases'][name]
    for key in ('tann_dists_tf_ratio', 'pann_dists_tf_ratio'):
        for mine, true in zip(getattr(r, key), ref[key]):
            assert (math.isnan(mine) if true == 'NaN' else mine == math.inf if true == 'Inf' else mine == pytest.approx(float(Decimal(true)), rel=1e-13))


def test_tiny_sigma_is_nan_in_module_and_r_although_the_true_ratio_is_finite():
    assert all(math.isnan(v) for v in run('tiny_sigma_saturates').tann_dists_tf_ratio) and REFERENCE['cases']['tiny_sigma_saturates']['tann_dists_tf_ratio'][0] != 'NaN'
    assert CROSS['cases']['tiny_sigma_saturates']['transfer_ratio_R'] == 'NaN'


def test_the_record_keeps_the_marking():
    assert CROSS['bit_identical_with_R'] is False and CROSS['backend'] == rt.BACKEND and CROSS['historical_reproduction_ready'] is False


def test_the_cplusplus_lines_were_not_run():
    assert CROSS['cplusplus_lines_run'] is False and 'R twin' in CROSS['source'] and '228, 230' in CROSS['source']


@pytest.mark.parametrize('group', ['named', 'stress'])
@pytest.mark.parametrize('key', ['grid_ratio', 'prod_from_R_factors'])
def test_the_parts_without_a_normal_function_are_bit_identical(group, key):
    t = CROSS[group][key]
    assert t['max_ulps'] == 0 and t['bit_equal'] == t['values'] and t['non_finite_mismatch'] == 0


@pytest.mark.parametrize('group', ['named', 'stress'])
@pytest.mark.parametrize('key', ['dnorm', 'p_a', 'p_b', 'd_truncnorm', 'density_ratio', 'transfer_ratio'])
def test_non_finite_values_agree_with_r(group, key):
    assert CROSS[group][key]['non_finite_mismatch'] == 0


def test_indices_agree_with_r_everywhere():
    assert CROSS['indices_equal_everywhere'] is True


def test_stress_measurement_is_recorded():
    s = CROSS['stress']
    assert CROSS['stress_cases'] == 120
    assert (s['density_ratio']['values'], s['density_ratio']['bit_equal'], s['density_ratio']['max_ulps']) == (3406, 2533, 15)
    assert (s['transfer_ratio']['values'], s['transfer_ratio']['bit_equal'], s['transfer_ratio']['max_ulps']) == (120, 39, 35)
    assert s['density_ratio']['max_relative'] < 3.4e-15 and s['transfer_ratio']['max_relative'] < 4.5e-15


def test_deviation_is_not_zero_so_the_marking_is_needed():
    assert CROSS['stress']['density_ratio']['bit_equal'] < CROSS['stress']['density_ratio']['values']


def test_r_runtime_is_the_bound_version():
    assert CROSS['runtime']['R'] == 'R version 3.5.0 (2018-04-23)' and CROSS['runtime']['pnorm_namespace'] == CROSS['runtime']['dnorm_namespace'] == 'stats' and CROSS['runtime']['long_double'] == 'TRUE'


def test_counterexamples_change_something():
    c = ARITHMETIC['checks']
    assert c['double_accumulator_instead_of_long_double']['changed'] > 0 and c['density_ratio_without_the_truncation_scale']['changed'] > 0
    assert c['last_minimum_instead_of_first'] == {'index_first': 0, 'index_last': 1}


def test_active_definition_and_reach_are_recorded():
    assert BINDING['active_definition'] == 'src/mcmc_functions_v1.6.R:1485' and 'every iteration' in BINDING['reach']['call'] and '302' in BINDING['reach']['reader']
    assert {1437, 1485, 1516, 1529, 1552} <= set(BINDING['pinned_lines']['src/mcmc_functions_v1.6.R']) and {228, 230, 302} <= set(BINDING['pinned_lines']['src/mcmc_conduction.R'])


def test_rcpp_binding_is_recorded():
    r = BINDING['rcpp']
    assert r['versions_read'] == 19 and r['which_min_h_identical_in_all'] is True and r['abs_lines'] == [42] and 'unknown' in r['not_bound']


def test_r_locations_are_recorded():
    assert BINDING['r_3_5_0']['locations']['src/main/summary.c']['lines'] == [367, 369, 376, 381, 383]


def test_nothing_was_drawn_or_released():
    assert BINDING['random_state_used'] is False and BINDING['draws_run'] == 0 and BINDING['compiled'] is False and BINDING['historical_reproduction_ready'] is False and FIXTURE['historical_reproduction_ready'] is False


@pytest.mark.parametrize('marker', ['Copyright', 'GNU General Public', 'WhichMin', 'R_xlen_t'])
def test_no_gpl_text_is_stored(marker):
    assert marker not in (PACKET / 'source_binding.json').read_text(encoding='utf-8')
