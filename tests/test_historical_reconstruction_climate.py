"""reconstruction_climate (mcmc_functions_v1.6.R 1188-1199): reference, hand values, laziness, stored R record, reachability."""
from fractions import Fraction
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / 'docs/thesis_research/literal_reconstruction_climate_2026_11_08'
sys.path.insert(0, str(ROOT / 'tools'))
import historical_reconstruction_climate as rc  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, PACKET / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference = _load('climate_reference_t', 'precise_reference.py')
check = _load('climate_check_t', 'check_reconstruction_climate.py')


def _json(name):
    return json.loads((PACKET / name).read_text(encoding='utf-8'))


FIX = _json('synthetic_fixture.json')
CASES = FIX['cases']
RESULT, CROSS, ARITH, BOUND = _json('synthetic_result.json'), _json('r_crosscheck_result.json'), _json('arithmetic_check.json'), _json('source_binding.json')
X1, X2 = FIX['x1range'], FIX['x2range']


def _run(name):
    return check.run(CASES[name], FIX)


def call(core, tf1, tf2, x1=X1, x2=X2, recent=True):
    return rc.reconstruction_climate(core, tf1, tf2, x1, x2, recent=recent, prior_mean_tmp=18.5, prior_mean_pann=640.0)


@pytest.mark.parametrize('name', list(CASES))
def test_module_equals_the_independent_reference(name):
    got, ref = _run(name), reference.expected(CASES[name], X1, X2, FIX['prior'])
    for key, value, kind in (('tann', got.tann, got.tann_kind), ('pann', got.pann, got.pann_kind)):
        assert kind == ref[key][0]
        if kind == 'number':
            assert value.hex() == ref[key][1].hex()


@pytest.mark.parametrize('name', [n for n, c in CASES.items() if c['recent']])
def test_products_equal_the_reference(name):
    got, ref = _run(name), reference.expected(CASES[name], X1, X2, FIX['prior'])
    for mine, theirs in ((got.tann_weights, ref['tann_weights']), (got.pann_weights, ref['pann_weights'])):
        assert [reference.show(v) for v in mine] == [reference.show(v) for v in theirs]


@pytest.mark.parametrize('name', list(CASES))
def test_stored_result_matches_a_fresh_run(name):
    assert RESULT['cases'][name] == check.record(_run(name))


@pytest.mark.parametrize('name', [n for n, c in CASES.items() if c['hand'] is not None])
def test_hand_values(name):
    got, hand = _run(name), CASES[name]['hand']
    for key, value, kind in (('tann', got.tann, got.tann_kind), ('pann', got.pann, got.pann_kind)):
        if hand[key] is None:
            continue
        if hand[key] == 'NA':
            assert kind == 'NA'
        else:
            target = float(Fraction(hand[key]))
            assert kind == 'number' and abs(value - target) <= math.ulp(target)


def test_hand_weights_are_exact():
    got, hand = _run('recent_basic'), CASES['recent_basic']['hand']
    assert list(got.tann_weights) == [float(v) for v in hand['tann_weights']]
    assert list(got.pann_weights) == [float(v) for v in hand['pann_weights']]


def test_only_the_first_row_counts():
    a, b = _run('recent_basic'), _run('only_first_row_counts')
    assert a.tann == b.tann and a.pann == b.pann


def test_not_recent_reads_nothing_but_the_prior():
    got = call('anything', None, object(), x1='not a grid', x2=7, recent=False)
    assert (got.tann, got.pann, got.source) == (18.5, 640.0, 'prior_recent')
    assert got.tann_weights is None and got.tann_median is None


def test_not_recent_passes_na_prior_through():
    got = rc.reconstruction_climate(None, None, None, None, None, recent=False, prior_mean_tmp=None, prior_mean_pann=3.0)
    assert (got.tann_kind, got.pann_kind, got.pann) == ('NA', 'number', 3.0)


def test_zero_first_row_gives_na_for_both():
    got = _run('first_row_zero')
    assert (got.tann_kind, got.pann_kind) == ('NA', 'NA') and got.tann_median.path == 'no_positive_weight'


def test_one_na_tf_cell_only_touches_its_own_median():
    got = _run('na_in_one_tf_cell')
    assert (got.tann_kind, got.pann_kind) == ('NA', 'number') and got.tann_median.path == 'na_weight'


def test_zero_share_does_not_protect_against_na():
    assert _run('na_tf_at_zero_share').tann_kind == 'NA'
    assert ARITH['naive_counterexamples']['skip_na_at_zero_share']['skipping_kind'] == 'number'


@pytest.mark.parametrize('key', ['row_mean_instead_of_first_row', 'expectation_instead_of_median', 'skip_na_at_zero_share'])
def test_counterexamples_differ(key):
    assert ARITH['naive_counterexamples'][key]['differs'] is True


@pytest.mark.parametrize('core,tf1', [([[1.0]], [[1, 0, 0, 0], [0, 0, 1, 3]]), ([[1.0, 2.0]], [[1, 0, 0, 0]])])
def test_row_and_tf_must_conform(core, tf1):
    with pytest.raises(rc.UnsupportedSourceDomain):
        call(core, tf1, [[1, 1, 1, 1], [1, 1, 1, 1]])


@pytest.mark.parametrize('core,tf1,tf2,x1', [
    ([[1.0, 2.0]], [[1, 0, 0], [0, 1, 1]], [[1, 1, 1, 1], [1, 1, 1, 1]], X1),
    ([[math.inf, 2.0]], [[1, 0, 0, 0], [0, 1, 1, 1]], [[1, 1, 1, 1], [1, 1, 1, 1]], X1),
    ([[True, 2.0]], [[1, 0, 0, 0], [0, 1, 1, 1]], [[1, 1, 1, 1], [1, 1, 1, 1]], X1),
    ([[1.0, 2.0]], [[1, 0], [0]], [[1, 1, 1, 1], [1, 1, 1, 1]], X1),
    ([], [[1, 0, 0, 0], [0, 1, 1, 1]], [[1, 1, 1, 1], [1, 1, 1, 1]], X1),
])
def test_bad_inputs_are_rejected(core, tf1, tf2, x1):
    with pytest.raises(rc.UnsupportedSourceDomain):
        call(core, tf1, tf2, x1=x1)


@pytest.mark.parametrize('recent', [None, 1, 'TRUE', float('nan')])
def test_recent_must_be_an_explicit_boolean(recent):
    with pytest.raises(rc.UnsupportedSourceDomain):
        call([[1.0, 3.0]], [[1, 0, 0, 0], [0, 0, 1, 3]], [[1, 1, 0, 0], [0, 2, 1, 0]], recent=recent)


def test_numpy_inputs_are_accepted_and_stay_unchanged():
    core = np.array([[1.0, 3.0]])
    tf1 = np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 3.0]])
    before = core.copy(), tf1.copy()
    got = call(core, tf1, [[1, 1, 0, 0], [0, 2, 1, 0]])
    assert got.tann == _run('recent_basic').tann
    assert (core == before[0]).all() and (tf1 == before[1]).all()


def test_row_times_matrix_is_a_plain_sum_in_taxon_order():
    row, tf = [0.1, 0.2, 0.3], [[1.0], [1.0], [1.0]]
    assert rc.row_times_matrix(row, tf) == [(0.1 + 0.2) + 0.3]


# --- stored R 3.5.0 + matrixStats record --------------------------------------------------------------------------

def test_r_runtime():
    assert CROSS['runtime']['R'] == 'R version 3.5.0 (2018-04-23)' and CROSS['runtime']['matrixStats'] == '0.56.0'
    assert CROSS['runtime']['weightedMedian_namespace'] == 'matrixStats' and CROSS['lazy_arguments_not_forced'] is True


def test_every_r_value_is_bit_equal():
    assert CROSS['values_compared'] == CROSS['bit_equal'] and CROSS['max_ulps'] == 0 and CROSS['special_mismatch'] == 0


def test_stress_set_is_stored_and_reproduced():
    cases = check.stress_cases(FIX['stress'])
    runs = [check.record(check.run_stress(c)) for c in cases]
    assert [r['tann'] for r in runs] == CROSS['stress']['R_tann'] and [r['pann'] for r in runs] == CROSS['stress']['R_pann']
    assert len(runs) == 200 and RESULT['stress']['prior_cases'] == 20 and RESULT['stress']['kinds'] == ['NA', 'number']


# --- reachability, stated plainly ---------------------------------------------------------------------------------

def test_the_only_caller_is_an_unsourced_older_file():
    assert BOUND['reached_by_bound_run'] is False and BOUND['caller_file_sourced_by_main_v1_2'] is False
    assert [c['file'] for c in BOUND['callers']] == ['src/local_reconstr_all_epd/src/mcmc_sampling_v1.4.R']
    assert BOUND['callers'][0]['line'] == 79 and BOUND['files_referencing_the_caller_file'] == [] and BOUND['r_files_searched'] > 500


def test_mcmc_prepare_loop_is_not_reached_either():
    assert BOUND['guard_1380_needs_not_add_sites'] is True and BOUND['lines_1408_1413_reached_by_bound_run'] is False


def test_binding_claims_no_run_and_no_original_inputs():
    assert BOUND['historical_reproduction_ready'] is False and BOUND['run_binding'] is None
    assert BOUND['actual_original_inputs_used_for_numerics'] is False and BOUND['which_taxa_read'] is False


def test_audit_records_the_dormant_status():
    audit = _json('determinism_audit.json')
    assert audit['selected_for_implementation'] == 'reconstruction_climate' and audit['final_variant_selected'] is False
    assert audit['candidates'][0]['reached_by_bound_run'] is False and audit['candidates'][1]['random_state'] is True


def test_fixture_is_marked_synthetic():
    assert FIX['historical_reproduction_ready'] is False and FIX['kind'].startswith('synthetic')


def test_reference_does_not_import_the_module():
    text = (PACKET / 'precise_reference.py').read_text(encoding='utf-8')
    assert 'historical_reconstruction_climate' not in text and 'historical_weighted_median' not in text
