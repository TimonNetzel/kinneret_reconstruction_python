"""weightedMedian of matrixStats for mcmc_prepare.R 1631/1651 and the NA mask of 1664: reference, hand values, stored R
record, sort routines, domain. Artificial inputs only; R itself is not needed here."""
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
PACKET = ROOT / 'docs/thesis_research/literal_weighted_median_2026_11_08'
sys.path.insert(0, str(ROOT / 'tools'))
import historical_tf_missing_priors as missing  # noqa: E402
import historical_weighted_median as wm  # noqa: E402
import r_sort_gpl  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, PACKET / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference = _load('wm_reference', 'precise_reference.py')
check = _load('wm_check', 'check_weighted_median.py')


def _json(name):
    return json.loads((PACKET / name).read_text(encoding='utf-8'))


FIX = _json('synthetic_fixture.json')
CASES = FIX['cases']
RESULT, ARITH, CROSS = _json('synthetic_result.json'), _json('arithmetic_check.json'), _json('r_crosscheck_result.json')


def _run(name):
    return check.run_case(CASES[name])


# --- independent reference, stored files, hand values --------------------------------------------------------------

@pytest.mark.parametrize('name', list(CASES))
def test_module_equals_the_independent_reference(name):
    got, ref = _run(name), reference.expected(CASES[name])
    assert got.kind == ref['kind']
    if ref['value'] is not None and got.kind == 'number':
        assert got.value.hex() == ref['value'].hex()


@pytest.mark.parametrize('name', list(CASES))
def test_stored_result_matches_a_fresh_run(name):
    assert RESULT['cases'][name] == check.as_record(_run(name))


@pytest.mark.parametrize('name', [n for n, c in CASES.items() if c['hand'] is not None])
def test_hand_values_from_the_c_text(name):
    got, hand = _run(name), CASES[name]['hand']
    assert (got.kind, got.path, got.half) == (hand['kind'], hand['path'], hand['half'])
    if hand['kind'] == 'number':
        target = reference.parse(hand['value'])
        assert got.value == target or abs(got.value - target) <= math.ulp(target)


def test_hand_values_are_met_exactly():
    assert ARITH['maximum_hand_ulps'] == 0


# --- when the result is NA ------------------------------------------------------------------------------------------

@pytest.mark.parametrize('x,w,path', [
    ([1, 2, 3], [0, 0, 0], 'no_positive_weight'),
    ([1, 2, 3], [-1, 0, -2], 'no_positive_weight'),
    ([], [], 'no_positive_weight'),
    ([1, 2, 3], [1, None, 1], 'na_weight'),
    ([None, 2, 3], [None, 1, 1], 'na_weight'),
    ([1, 2, 3], [float('nan'), 1, 1], 'na_weight'),
    ([1, None, 3], [1, 1, 1], 'na_x_at_positive_weight'),
])
def test_na_paths(x, w, path):
    got = wm.weighted_median(x, w)
    assert (got.kind, got.is_na, got.path) == ('NA', True, path) and math.isnan(got.value)


def test_na_weight_anywhere_wins_over_a_plus_inf_weight():
    assert wm.weighted_median([1, 2, 3], [math.inf, 1, None]).path == 'na_weight_after_inf'
    assert wm.weighted_median([1, 2, 3], [None, 1, math.inf]).path == 'na_weight'


def test_na_x_at_nonpositive_weight_is_not_read():
    got = wm.weighted_median([None, 2.0, 3.0], [0, 1, 1])
    assert got.kind == 'number' and got.value == 2.5
    assert wm.weighted_median([None, 2.0, 3.0], [-5, 1, 1]).value == 2.5


def test_the_order_decides_with_a_plus_inf_weight():
    assert wm.weighted_median([None, 2.0], [1, math.inf]).kind == 'NA'
    assert wm.weighted_median([2.0, None], [math.inf, 1]).value == 2.0


def test_na_x_at_minus_inf_stops_only_in_the_plus_inf_branch():
    assert wm.weighted_median([None, 2.0, 3.0], [-math.inf, math.inf, 1]).path == 'na_x_at_infinite_weight'
    assert wm.weighted_median([None, 2.0, 3.0], [-math.inf, 1, 1]).value == 2.5


def test_stress_set_obeys_the_na_rule():
    for x, w in check.stress_cases(FIX['stress']):
        xs = [math.nan if v is None else v for v in x]
        ws = [math.nan if v is None else v for v in w]
        assert (wm.weighted_median(x, w).kind == 'NA') == reference.na_rule(xs, ws)


def test_nan_from_infinite_x_is_not_na():
    got = wm.weighted_median([-math.inf, math.inf], [1, 1])
    assert (got.kind, got.is_na) == ('NaN', True) and got.path == 'interpolated'
    assert wm.weighted_median([-math.inf, 1.0], [3, 1]).value == -math.inf


# --- source facts that a casual reimplementation would change --------------------------------------------------------

def test_equal_weight_branch_is_not_the_median_for_odd_n():
    assert wm.weighted_median([3.0, 1.0, 2.0], [math.inf] * 3).value == 1.0
    assert ARITH['naive_counterexamples']['inf_odd_true_median'] == {'module': 1.0, 'true_median': 2.0}


def test_minus_inf_counts_as_infinite_once_a_plus_inf_exists():
    assert wm.weighted_median([1.0, 2.0, 3.0], [-math.inf, math.inf, 1]).value == 1.5


def test_tie_order_is_the_unstable_one_of_r():
    tied = ARITH['naive_counterexamples']['ties_trace_stable_sort']
    assert tied['module'] != tied['stable_sort'] and tied['stable_sort'] == 1.4
    assert _run('ties_trace').value == tied['module']


def test_pair_order_does_not_matter_without_ties():
    assert _run('skewed').value == _run('unsorted_pairs').value


def test_rounding_tie_at_one_half_selects_the_middle_value():
    assert _run('uniform_odd_rounding_tie').value == 2.0


def test_dominant_first_weight_returns_the_first_sorted_value():
    got = _run('first_sorted_value')
    assert (got.value, got.path, got.half) == (1.0, 'first_sorted_value', 0)


def test_expectation_lines_are_not_the_result():
    row = ARITH['naive_counterexamples']['expectation_line_1623_is_overwritten']
    assert row['expectation'] != row['weighted_median']


def test_every_shortcut_mask_differs_from_the_source():
    assert ARITH['masks_on_tf_case']['source_1664'] == [2, 3, 5]
    assert all(ARITH['masks_differ_from_source'].values())


# --- the R sort translation ------------------------------------------------------------------------------------------

@pytest.mark.parametrize('values', [[], [1.0], [2.0, 1.0], [3.0, 1.0, 2.0], [5.0, 5.0, 5.0, 5.0],
                                    [float(i % 7) for i in range(100)], [float(50 - i) for i in range(50)]])
def test_r_qsort_i_sorts_and_keeps_a_permutation(values):
    v, idx = list(values), list(range(len(values)))
    r_sort_gpl.r_qsort_i(v, idx)
    assert v == sorted(values) and sorted(idx) == list(range(len(values)))
    assert [values[i] for i in idx] == v


def test_r_qsort_i_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        r_sort_gpl.r_qsort_i([1.0], [])


@pytest.mark.parametrize('k', range(9))
def test_r_psort_places_the_k_th_value(k):
    x = [5.0, 3.0, 8.0, 1.0, 9.0, 2.0, 7.0, 4.0, 6.0]
    r_sort_gpl.r_psort(x, len(x), k)
    assert x[k] == k + 1.0 and all(a <= x[k] for a in x[:k]) and all(a >= x[k] for a in x[k + 1:])


# --- domain and rejected paths ---------------------------------------------------------------------------------------

def test_overflowing_total_weight_is_rejected():
    with pytest.raises(wm.UnsupportedSourceDomain, match='outside its arrays'):
        wm.weighted_median([1.0, 2.0], [2.0 ** 1023, 2.0 ** 1023])


@pytest.mark.parametrize('x,w', [([1, 2], [1]), ('ab', [1, 2]), ([1, 2], 'ab'), ([True, 2], [1, 1]), ([1, 'a'], [1, 1]),
                                 (np.ones((2, 2)), [1, 1])])
def test_bad_shapes_and_types_are_rejected(x, w):
    with pytest.raises(wm.UnsupportedSourceDomain):
        wm.weighted_median(x, w)


def test_inputs_stay_unchanged():
    x, w = [3.0, 1.0, 2.0], [1.0, 2.0, 3.0]
    wm.weighted_median(x, w)
    assert x == [3.0, 1.0, 2.0] and w == [1.0, 2.0, 3.0]


def test_numpy_floats_and_integers_are_accepted():
    assert wm.weighted_median(np.array([1.0, 2.0, 3.0, 4.0]), [1, 1, 1, 1]).value == 2.5


# --- TF matrix, mask and the existing removal module ---------------------------------------------------------------

def _tf():
    tf = FIX['tf_case']
    rows = lambda m: [[check.input_value(str(v)) for v in r] for r in m]
    return tf, rows


def test_tf_loop_and_mask_on_the_artificial_matrix():
    tf, rows = _tf()
    x1 = wm.tf_row_medians([float(v) for v in tf['x1range']], rows(tf['tf_x1']))
    x2 = wm.tf_row_medians([float(v) for v in tf['x2range']], rows(tf['tf_x2']))
    assert [m.kind for m in x1] == tf['hand_x1_kinds'] and [m.kind for m in x2] == tf['hand_x2_kinds']
    assert list(wm.which_tf_missing(x1)) == tf['hand_which_tf_missing'] == RESULT['tf_case']['which_tf_missing']


def test_mask_feeds_the_existing_removal_module():
    tf, rows = _tf()
    x1 = wm.tf_row_medians([float(v) for v in tf['x1range']], rows(tf['tf_x1']))
    x2 = wm.tf_row_medians([float(v) for v in tf['x2range']], rows(tf['tf_x2']))
    n = len(tf['taxa'])
    got = missing.remove_missing_tfs(
        [m.is_na for m in x1], taxa=tf['taxa'], all_avail=np.ones((3, n)), all_avail_age=np.ones((3, n)),
        taxa_appear_core=np.ones((3, n)), tf_x1=np.ones((n, 4)), tf_x2=np.ones((n, 4)),
        x1_mean=[m.value for m in x1], x2_mean=[m.value for m in x2], taxa_change=False)
    assert got.which_tf_missing == (2, 3, 5) and got.taxa == ('Kunsttaxon A', 'Kunsttaxon D')
    assert got.num_taxa_line == 1678
    assert math.isnan(got.x2_mean[1])  # taxon D keeps its NA x2 median: only x1 is tested in 1664


def test_tf_loop_rejects_rows_of_another_length():
    with pytest.raises(wm.UnsupportedSourceDomain):
        wm.tf_row_medians([1.0, 2.0], [[1.0]])
    with pytest.raises(wm.UnsupportedSourceDomain):
        wm.tf_row_medians([1.0, 2.0], [])


# --- stored R 3.5.0 + matrixStats 0.56.0 record ------------------------------------------------------------------

def test_r_runtime_is_the_recorded_binary():
    info = CROSS['runtime']
    assert info['R'] == 'R version 3.5.0 (2018-04-23)' and info['matrixStats'] == '0.56.0'
    assert info['namespace'] == info['lookup_namespace'] == 'matrixStats'
    assert info['built'].startswith('R 3.5.3; x86_64-w64-mingw32')
    assert info['formals'] == 'x,w,idxs,na.rm,interpolate,ties,...'


def test_every_r_value_is_bit_equal_and_the_mask_agrees():
    assert CROSS['values_compared'] == CROSS['bit_equal'] == 439 and CROSS['max_ulps'] == 0
    assert CROSS['special_mismatch'] == 0 and CROSS['which_tf_missing_equal'] is True
    assert CROSS['which_tf_missing_R'] == [2, 3, 5]


@pytest.mark.parametrize('name', list(CASES))
def test_stored_r_value_equals_a_fresh_module_run(name):
    got = _run(name)
    assert CROSS['cases'][name]['R'] == check.show(got.value, got.kind)


def test_stored_r_stress_values_equal_fresh_module_runs():
    r_values = CROSS['stress']['R_values']
    runs = [wm.weighted_median(x, w) for x, w in check.stress_cases(FIX['stress'])]
    assert r_values == [check.show(m.value, m.kind) for m in runs] and len(r_values) == 400


def test_stress_set_reaches_every_path_family():
    assert set(RESULT['stress']['paths']) >= {'interpolated', 'na_weight', 'na_x_at_positive_weight',
                                              'equal_weights_odd', 'equal_weights_even', 'single_value',
                                              'no_positive_weight'}


# --- bindings ----------------------------------------------------------------------------------------------------

def test_helper_comparison_covers_13_versions_with_one_kernel():
    helper = _json('helper_comparison.json')
    assert len(helper['versions']) == 13 and len(helper['normalised_kernel_groups']) == 1
    assert helper['tarball_hashes_equal_recorded'] is True
    assert len(next(iter(helper['normalised_kernel_groups'].values()))) == 13


def test_r_index_get_only_loses_its_na_check_for_null_indices():
    macros = _json('helper_comparison.json')['macros']['R_INDEX_GET']
    assert [len(m['versions']) for m in macros] == [8, 4, 1]
    assert 'check_i_NA' in macros[-1]['definitions'][0]


def test_binary_package_was_recorded_as_a_window_version():
    binary = _json('binary_package.json')
    assert binary['url'].endswith('/bin/windows/contrib/3.5/matrixStats_0.56.0.zip')
    assert binary['description']['Version'] == '0.56.0' and binary['source_group_in_window'] == '2440873576cb'
    assert binary['sha256'] == '71740915432553e0f88ae493b937c2c4cceab97aa203da5749d21ed5aba382cb'


def test_binary_zip_matches_the_record_when_present():
    local = ROOT / '.local/weighted_median_2026_11_08/dl_bin/matrixStats_0.56.0.zip'
    if not local.exists():
        pytest.skip('local download folder not present')
    assert hashlib.sha256(local.read_bytes()).hexdigest() == _json('binary_package.json')['sha256']


def test_r_sort_code_did_not_change_for_this_call_inside_the_window():
    sort = _json('r_sort_source.json')
    assert sort['psort_body_identical_at_window_end'] and sort['rcmp_identical_at_window_end']
    assert sort['qsort_body_only_change'] == ['    --I;', '    if(I) --I; // precaution for use in package bnlearn']
    assert sort['qsort_body_change_affects_this_call'] is False


def test_source_binding_claims_no_run_and_no_original_inputs():
    bound = _json('source_binding.json')
    assert bound['historical_reproduction_ready'] is False and bound['run_binding'] is None
    assert bound['actual_original_inputs_used_for_numerics'] is False and bound['only_x1_tested_in_1664'] is True
    assert bound['provider_record'] == 'matrixStats'
    assert bound['mean_writers'] == [1623, 1624, 1628, 1631, 1648, 1651, 1669, 1670]
    assert bound['spline_only_written'] == [1627, 1647]


def test_determinism_audit_selects_only_the_weighted_median():
    audit = _json('determinism_audit.json')
    assert audit['selected_for_implementation'] == 'weighted_median_1631_1651'
    assert audit['final_variant_selected'] is False


def test_tann_switch_is_at_top_level_and_assigned_in_1249():
    tann = _json('tann_binding.json')
    assert tann['writers_in_mcmc_prepare'] == [1208, 1210, 1219, 1229, 1241, 1249]
    assert (tann['switch_condition_line'], tann['switch_assignment_line'], tann['switch_brace_depth_condition_line']) == (
        1248, 1249, 0)
    assert tann['line_1209_excludes_kinneret'] is True and tann['other_code_lines_naming_tann_in_mcmc_prepare'] == []


def test_active_reconstruction_is_the_later_definition():
    tann = _json('tann_binding.json')
    assert tann['reconstruction_definitions_in_v1_6'] == [855, 1781] and tann['active_definition_line'] == 1781


def test_tann_has_exactly_one_reader_in_the_active_definition():
    tann = _json('tann_binding.json')
    assert tann['readers_of_tann_in_active_definition'] == [{'line': 1939, 'text': 'if(tann){'}]


def test_predictant_current_readers_are_collected_not_interpreted():
    tann = _json('tann_binding.json')
    assert [r['line'] for r in tann['readers_of_predictant_current_in_active_definition']] == [1940, 1942, 1964, 1999,
                                                                                                 2023, 2031]
    assert tann['historical_reproduction_ready'] is False and 'not assessed' in tann['effect_in_a_kinneret_run']


def test_fixture_is_marked_synthetic():
    assert FIX['historical_reproduction_ready'] is False and FIX['selected_final_variant'] is None
    assert FIX['kind'].startswith('synthetic')


def test_reference_does_not_import_the_module():
    text = (PACKET / 'precise_reference.py').read_text(encoding='utf-8')
    assert 'historical_weighted_median' not in text and 'r_sort_gpl' not in text


def test_fractions_parse_exactly():
    assert reference.parse('1/3') == float(Fraction(1, 3)) and math.isnan(reference.parse('NA'))
