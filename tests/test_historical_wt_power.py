"""wt_power (mcmc_functions_v1.6.R 1003-1012, 2023-2026): hand values, reference, measured deviation from R 3.5.0, rejections, reachability."""
from fractions import Fraction
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / 'docs/thesis_research/literal_wt_power_2026_11_09'
sys.path.insert(0, str(ROOT / 'tools'))
import historical_wt_power as wp  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, PACKET / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check = _load('wt_check_t', 'check_wt_power.py')


def _json(name):
    return json.loads((PACKET / name).read_text(encoding='utf-8'))


FIX = _json('synthetic_fixture.json')
CASES, CHAINS = FIX['cases'], FIX['chains']
EXPECTED = _json('reference_expectations.json')['cases']
RESULT, CROSS, ARITH, BOUND, AUDIT, MASK = (_json(n) for n in ('synthetic_result.json', 'r_crosscheck_result.json', 'arithmetic_check.json', 'source_binding.json', 'determinism_audit.json', 'fft_masking.json'))
HAND = sorted(n for n, c in CASES.items() if c['hand'])
FINITE = sorted(n for n, r in CROSS['cases'].items() if r['kind'] == 'number')
MISSING = sorted(n for n, r in CROSS['cases'].items() if r['kind'] != 'number')
NAN, INF = float('nan'), float('inf')
ONES = [[1.0]] * 8
ARGS = {'n_obs': 4, 'zeros': [0.0] * 4, 'invflen': 1 / 8, 'wb': ONES}
D = [1.0, 2.0, 3.0, 5.0]


@pytest.mark.parametrize('name', HAND)
def test_hand_values(name):
    got, r = check.run_case(CASES[name])
    want = [float(Fraction(t)) for t in CASES[name]['hand']['power']]
    assert list(r.values) == pytest.approx(want, rel=1e-13, abs=1e-13)
    assert got['dim'] == CASES[name]['hand']['dim']


@pytest.mark.parametrize('name', sorted(EXPECTED))
def test_centred_values_equal_the_independent_reference_exactly(name):
    assert [t.replace('-0', '0') if t == '-0' else t for t in check.run_case(CASES[name])[0]['x']] == EXPECTED[name]['x']


@pytest.mark.parametrize('name', sorted(EXPECTED))
def test_power_is_within_a_few_ulp_of_the_true_value(name):
    _, r = check.run_case(CASES[name])
    truth = [float(t) for t in EXPECTED[name]['power_true']]
    assert list(r.values) == pytest.approx(truth, rel=2e-14, abs=1e-300)


@pytest.mark.parametrize('name', FINITE)
def test_centred_values_are_bit_identical_with_R(name):
    assert CROSS['cases'][name]['x']['max_ulps'] == 0


@pytest.mark.parametrize('name', sorted(CROSS['cases']))
def test_dimensions_after_the_row_selection_match_R(name):
    row = CROSS['cases'][name]
    assert row['dim_R'] == row['dim_module']


@pytest.mark.parametrize('name', FINITE)
def test_power_stays_close_to_R_relative_to_its_column(name):
    assert CROSS['cases'][name]['power']['max_error_relative_to_largest_of_column'] < 5e-15


@pytest.mark.parametrize('name', MISSING)
def test_missing_series_gives_missing_power_everywhere_in_R(name):
    assert set(CROSS['cases'][name]['power_tokens_R']) <= {'NA', 'NaN'}


@pytest.mark.parametrize('name', sorted(n for n in MISSING if CROSS['cases'][n]['kind'] != 'mixed'))
def test_pure_missing_kind_equals_the_R_token(name):
    assert CROSS['cases'][name]['power_tokens_R'] == [CROSS['cases'][name]['kind']]


@pytest.mark.parametrize('name', MISSING)
def test_centred_tokens_equal_R(name):
    assert CROSS['cases'][name]['x_tokens_equal'] is True


def test_mixed_missing_claims_only_missing():
    r = wp.wt_power([wp.NA, NAN, 3.0, 5.0], 4, **ARGS)
    assert r.kind == 'mixed' and all(isinstance(v, float) and math.isnan(v) for v in r.values)
    assert r.f is None and r.wave is None


@pytest.mark.parametrize('values, want', [([1.0, wp.NA, 3.0], 'NA'), ([1.0, NAN, 3.0], 'NaN'), ([wp.NA, NAN], 'NA'), ([NAN, wp.NA], 'NaN'), ([INF, wp.NA], 'NA'), ([INF, -INF, wp.NA], 'NaN')])
def test_sum_keeps_the_first_missing_token(values, want):
    got = wp.r_sum(values)
    assert ('NA' if got is wp.NA else 'NaN' if math.isnan(got) else 'number') == want


@pytest.mark.parametrize('values, want', [([INF, 1.0], INF), ([1.0, -INF], -INF), ([INF, INF], INF)])
def test_sum_of_infinities(values, want):
    assert wp.r_sum(values) == want


def test_opposite_infinities_give_nan():
    assert math.isnan(wp.r_sum([INF, -INF]))


def test_sum_is_long_double():
    e = 2.0 ** -53
    assert wp.r_sum([1.0, e, e]) == 1.0 + 2.0 ** -52


def test_the_mean_divides_the_cast_sum_by_l_d_not_by_the_length():
    assert wp.centred([1.0, 3.0], 4, []) == [0.0, 2.0]


def test_a_missing_left_operand_keeps_its_token():
    got = wp.centred([NAN, wp.NA, 1.0], 3, [])
    assert math.isnan(got[0]) and got[1] is wp.NA and math.isnan(got[2])


def test_padding_is_appended_after_centring():
    assert wp.centred([1.0, 3.0], 2, [0.0, 0.0]) == [-1.0, 1.0, 0.0, 0.0]


def test_inverse_is_unscaled_so_invflen_does_the_scaling():
    r = wp.wt_power(D, 4, **{**ARGS, 'invflen': 1.0})
    assert r.values[0] == pytest.approx(64 * (1 - 11 / 4) ** 2)


def test_every_result_names_the_numpy_backend():
    r = wp.wt_power(D, 4, **ARGS)
    assert r.backend == wp.BACKEND and 'not bit-identical' in r.backend


@pytest.mark.parametrize('n_obs, columns, dim', [(4, 2, (4, 2)), (4, 1, None), (1, 2, None), (1, 1, None)])
def test_dimension_drop(n_obs, columns, dim):
    r = wp.wt_power(D, 4, n_obs=n_obs, zeros=[0.0] * 4, invflen=1 / 8, wb=[[1.0] * columns] * 8)
    assert r.dim == dim and len(r.values) == n_obs * columns


def test_values_are_column_major():
    r = wp.wt_power(D, 4, n_obs=4, zeros=[0.0] * 4, invflen=1 / 8, wb=[[1.0, 2.0]] * 8)
    assert np.allclose(r.values[4:], 4 * np.array(r.values[:4]))


def test_selection_is_column_major():
    r = wp.wt_power(D, 4, n_obs=4, zeros=[0.0] * 4, invflen=1 / 8, wb=[[1.0, 2.0]] * 8)
    got = wp.select_coi(r, [[1, 0], [0, 0], [0, 1], [0, 0]])
    assert got == (r.values[0], r.values[6])


def test_selection_works_on_a_dropped_vector():
    r = wp.wt_power(D, 4, n_obs=4, zeros=[0.0] * 4, invflen=1 / 8, wb=ONES)
    assert wp.select_coi(r, [[0], [1], [0], [1]]) == (r.values[1], r.values[3])


@pytest.mark.parametrize('kwargs, message', [
    ({'wb': [[1.0 + 0j]] * 8}, 'real'), ({'wb': [[NAN]] * 8}, 'non-finite'), ({'wb': [[INF]] * 8}, 'non-finite'), ({'wb': [[1.0]] * 7}, 'nrow'), ({'wb': [1.0] * 8}, 'matrix'),
    ({'wb': [[]] * 8}, 'matrix'), ({'n_obs': 9}, 'n.obs'), ({'n_obs': 0}, 'n_obs'), ({'n_obs': 1.5}, 'n_obs'), ({'n_obs': True}, 'n_obs'), ({'invflen': wp.NA}, 'invflen'), ({'invflen': INF}, 'invflen')])
def test_rejections(kwargs, message):
    with pytest.raises(wp.UnsupportedSourceDomain, match=message):
        wp.wt_power(D, 4, **{**ARGS, **kwargs})


@pytest.mark.parametrize('l_d', [0, -1, 2.5, True])
def test_l_d_must_be_a_positive_integer(l_d):
    with pytest.raises(wp.UnsupportedSourceDomain, match='l_d'):
        wp.wt_power(D, l_d, **ARGS)


@pytest.mark.parametrize('bad', ['1', True, object()])
def test_series_must_be_numeric(bad):
    with pytest.raises(wp.UnsupportedSourceDomain):
        wp.wt_power([1.0, bad, 3.0, 5.0], 4, **ARGS)


def test_series_must_be_flat():
    with pytest.raises(wp.UnsupportedSourceDomain):
        wp.centred(np.ones((2, 2)), 4, [])


def test_empty_series_is_rejected():
    with pytest.raises(wp.UnsupportedSourceDomain, match='empty'):
        wp.wt_power([], 1, n_obs=1, zeros=[], invflen=1.0, wb=np.ones((0, 1)))


def test_overflow_is_rejected():
    with pytest.raises(wp.UnsupportedSourceDomain, match='overflow'):
        wp.wt_power([1e300, -1e300, 2e300, -2e300], 4, n_obs=4, zeros=[0.0] * 4, invflen=1.0, wb=[[1e300]] * 8)


def test_formals_have_no_default():
    with pytest.raises(TypeError):
        wp.wt_power(D, 4)  # R stops on the self-referencing default as well


@pytest.mark.parametrize('coi, message', [([[1, 1]], 'n.obs'), ([[1, 2]] * 4, 'only 0 and 1'), ([[1, NAN]] * 4, 'only 0 and 1')])
def test_coi_rejections(coi, message):
    r = wp.wt_power(D, 4, n_obs=4, zeros=[0.0] * 4, invflen=1 / 8, wb=[[1.0, 2.0]] * 8)
    with pytest.raises(wp.UnsupportedSourceDomain, match=message):
        wp.select_coi(r, coi)


@pytest.mark.parametrize('name', sorted(CHAINS))
def test_chain_post_spec_stays_close_to_R(name):
    row = CROSS['chains'][name]
    assert row['absolute'] < 1e-15 and row['ulps'] <= 64


@pytest.mark.parametrize('name', sorted(CHAINS))
def test_length_ages_spec2_equals_R(name):
    row = CROSS['chains'][name]
    assert row['l_d_R'] == row['l_d_module']


@pytest.mark.parametrize('name, replaced', [('negative_correlation_is_replaced', True), ('missing_series_is_replaced', True), ('kinneret_shape_all_true', False)])
def test_line_2025_replacement(name, replaced):
    got, r = check.run_chain(CHAINS[name])
    assert got['replaced'] is replaced
    if replaced:
        assert r.post_spec == 1e-10 * 1e-10


def test_kinneret_shape_keeps_l_d_equal_to_the_series():
    ch = CHAINS['kinneret_shape_all_true']
    assert wp.length_ages_spec2(ch['ages_spec'], ch['hi_ages_spec']) == len(ch['predictant_current'])


def test_short_hi_mask_is_recycled_over_the_full_ages_vector():
    # Marked, not corrected: mcmc_prepare.R:1817 recycles hi_ages_spec over ages_spec, so the count can differ from the selected series
    assert wp.length_ages_spec2([True, False, True, True], [True, False]) == 2
    assert wp.length_ages_spec2([True, True, False, True], [False, True]) == 2


def test_stored_result_equals_a_fresh_run():
    for name, case in CASES.items():
        assert check.run_case(case)[0] == RESULT['cases'][name]
    for name, chain in CHAINS.items():
        assert check.run_chain(chain)[0] == RESULT['chains'][name]


def test_the_record_does_not_claim_bit_identity():
    assert CROSS['bit_identical_with_R'] is False and AUDIT['bit_identical_with_R'] is False
    assert CROSS['stress']['power']['bit_equal'] < CROSS['stress']['power']['values']


def test_centred_values_are_bit_identical_in_all_stress_cases():
    assert CROSS['stress']['x']['bit_equal'] == CROSS['stress']['x']['values'] == 22256 and CROSS['stress']['x']['max_ulps'] == 0


def test_deviation_is_recorded_with_its_size():
    power = CROSS['stress']['power']
    assert power['values'] == 39876 and 0 < power['max_error_relative_to_largest_of_column'] < 5e-15 and power['max_ulps'] == 606


def test_no_token_or_missing_disagreement_with_R():
    assert CROSS['token_mismatch'] == 0 and CROSS['missing_mismatch'] == 0 and CROSS['function_equals_steps_everywhere'] is True


def test_r_uses_a_real_wb_and_the_stats_fft():
    assert CROSS['wb_types_in_R'] == ['double'] and CROSS['runtime']['fft_namespace'] == 'stats' and CROSS['runtime']['R'].startswith('R version 3.5.0')


def test_module_is_closer_to_the_truth_than_R_on_the_reference_cases():
    # A measurement on 17 artificial cases, not a statement about quality in general
    assert CROSS['distance_to_true_value_ulps']['module'] <= CROSS['distance_to_true_value_ulps']['R']


@pytest.mark.parametrize('key', sorted(ARITH['checks']))
def test_counterexample_has_a_stored_size(key):
    assert 0 <= ARITH['checks'][key]['changed'] <= ARITH['checks'][key]['compared'] > 0


def test_counterexamples_are_not_variants():
    assert ARITH['kind'] == 'counterexamples_not_candidate_variants'


def test_callers_are_bound():
    callers = BOUND['callers']
    assert set(callers['mcmc_functions_v1.6.R']) == {'991', '2023', '2031'} and set(callers['mcmc_prepare.R']) == {'947', '966'} and callers['wt_cpp_only_in_mcmc_prepare'] == [950, 971]


def test_kinneret_reachability_stays_open():
    k = BOUND['kinneret']
    assert k['spectral'].startswith('open') and k['line_1145_reached'] is False and k['case_studies'] is False and k['hi_spec'] is False and k['line_2031_reached'] is False


def test_nothing_claims_a_historical_reproduction():
    for record in (FIX, CROSS, BOUND, AUDIT, ARITH, RESULT):
        assert record['historical_reproduction_ready'] is False
    assert FIX['selected_final_variant'] is None


def test_masking_was_read_for_the_four_packages_and_found_nothing():
    assert MASK['versions_masking_fft_or_mvfft'] == [] and MASK['versions_read'] == 17 and set(MASK['packages']) == {'signal', 'pracma', 'WaveletComp', 'biwavelet'}
    assert 'not_checked' in MASK and MASK['window'] == ['2018-04-23', '2023-07-07']


def test_biwavelet_daughter_is_real_in_every_version_read():
    assert {v['daughter_type'] for v in MASK['packages']['biwavelet']['versions']} == {'NumericVector'}


def test_module_contains_no_translation_of_r_fft():
    text = (ROOT / 'tools/historical_wt_power.py').read_text(encoding='utf-8')
    assert 'np.fft.fft' in text and 'np.fft.ifft' in text and 'r_sort_gpl' not in text and 'fft_work' not in text and 'fftmx' not in text
