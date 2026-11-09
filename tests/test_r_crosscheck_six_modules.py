"""Stored R 3.5.0 cross check of the six older modules and the attribution of its ulp differences; R is not needed."""
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import random

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / 'docs/thesis_research/r_crosscheck_six_modules_2026_11_08'


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, PACKET / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


explain = _load('six_explain', 'explain_differences.py')
stress = _load('six_stress', 'make_stress.py')


def _json(name):
    return json.loads((PACKET / name).read_text(encoding='utf-8'))


RESULT, CAUSES, PROVIDER = _json('six_crosscheck_result.json'), _json('difference_causes.json'), _json('provider_rowcumsums.json')
MODULES = ('age_depth', 'selection', 'final_pdf', 'cdf', 'tf', 'pruning')
FLAGGED = {'age_depth': {'weights', 'pollen_at_ages'},
           'selection': {'product_initial', 'product_effective', 'product_effective_pann', 'product_effective_mean'},
           'final_pdf': {'coefficients', 'density', 'pdf_sum_depth', 'pdf_sum_age', 'gibbs_coefficients', 'gibbs_density'},
           'cdf': {'score'}, 'tf': {'normalized', 'masked'}, 'pruning': {'spectrum'}}


def test_runtime_is_r_350_with_matrixstats_for_rowcumsums():
    assert RESULT['runtime'] == {'R': 'R version 3.5.0 (2018-04-23)', 'sizeof_longdouble': '16', 'matrixStats': '0.56.0',
                                 'rowCumsums_namespace': 'matrixStats'}
    assert RESULT['random_state_used'] is False and RESULT['historical_reproduction_ready'] is False


def test_every_module_was_compared_on_literal_lines():
    assert tuple(RESULT['modules']) == MODULES
    assert 'check_ages.R:177-179,230-232' in RESULT['modules']['age_depth']['original_lines']
    assert RESULT['modules']['cdf']['original_lines'].endswith('369-371,485-490')
    assert RESULT['modules']['pruning']['original_lines'] == 'src/mcmc_prepare.R:51-80'


@pytest.mark.parametrize('module', MODULES)
def test_earlier_fixture_values_are_all_bit_equal(module):
    row = RESULT['modules'][module]
    assert row['fixture_values'] > 0 and row['fixture_values'] == row['fixture_bit_equal']


def test_no_special_value_differs():
    assert RESULT['special_mismatch'] == 0 and RESULT['max_ulps'] == 4


@pytest.mark.parametrize('module', MODULES)
def test_only_sums_and_matrix_products_differ(module):
    keys = RESULT['modules'][module]['keys']
    assert {k for k, s in keys.items() if s['values'] != s['bit_equal']} == FLAGGED[module]
    assert all(s['special_mismatch'] == 0 for s in keys.values())


@pytest.mark.parametrize('module', MODULES)
def test_differences_stay_within_a_few_ulps(module):
    assert 0 < RESULT['modules'][module]['max_ulps'] <= 4


def test_grids_masks_and_integer_results_are_exact():
    selection = RESULT['modules']['selection']['keys']
    for key in ('grid_hol', 'grid_lg', 'grid_eem', 'mask_hol', 'mask_lg', 'mask_eem', 'initial_mask', 'effective_mask'):
        assert selection[key]['values'] == selection[key]['bit_equal']
    pruning = RESULT['modules']['pruning']['keys']
    for key in ('skipped', 'metro_hastings', 'pollen', 'pollen_dim', 'null_ratio', 'removed'):
        assert pruning[key]['values'] == pruning[key]['bit_equal']


def test_rowcumsums_is_bit_equal_although_it_uses_long_double():
    cumulative = RESULT['modules']['cdf']['keys']['cumulative']
    assert cumulative['values'] == cumulative['bit_equal'] and cumulative['values'] > 800


def test_recorded_differences_are_real_and_counted():
    row = RESULT['modules']['final_pdf']
    assert row['differences_first_400'] and all(d['ulps'] >= 1 for d in row['differences_first_400'])


# --- attribution of the differences ------------------------------------------------------------------------------

def test_emulating_r_arithmetic_explains_every_flagged_value():
    assert CAUSES['emulation_bit_equal_everywhere'] is True and CAUSES['values'] == CAUSES['emulation_bit_equal'] == 24822
    assert CAUSES['module_bit_equal'] < CAUSES['values']
    assert all(s['max_emulation_ulps'] == 0 for r in CAUSES['modules'].values() for s in r.values())


@pytest.fixture(scope='module')
def emulated():
    return explain.main(None)


@pytest.mark.parametrize('module', MODULES)
def test_stored_emulation_digests_are_reproduced_without_r(module, emulated):
    for key, slot in emulated[module].items():
        text = ' '.join(t if isinstance(t, str) else t.hex() for t in slot['tokens'])
        assert hashlib.sha256(text.encode()).hexdigest() == CAUSES['modules'][module][key]['emulated_digest']


def test_round_bits_keeps_ties_to_even():
    assert explain.round_bits(Fraction(1) + Fraction(1, 2 ** 64), 64) == 1
    assert explain.round_bits(Fraction(1) + Fraction(3, 2 ** 64), 64) == 1 + Fraction(1, 2 ** 62)
    assert explain.round_bits(Fraction(-3, 2), 64) == Fraction(-3, 2) and explain.round_bits(Fraction(0), 64) == 0


def test_long_double_sum_differs_from_a_double_sum_by_design():
    values = [1.0, 2.0 ** -53, 2.0 ** -53]
    assert float(np.sum(values)) == 1.0  # Python 3.12's builtin sum is compensated, so NumPy is the reference here
    assert explain.ld_sum(values) == 1.0 + 2.0 ** -52


def test_long_double_mean_rounds_twice():
    assert explain.ld_mean([1.0, 2.0, 4.0]) == 7.0 / 3.0


def test_reference_blas_is_a_plain_loop():
    assert explain.dot_plain([0.1, 0.2, 0.3], [1.0, 1.0, 1.0]) == (0.1 + 0.2) + 0.3


# --- provider ------------------------------------------------------------------------------------------------------

def test_nothing_in_front_of_matrixstats_exports_rowcumsums():
    assert PROVIDER['exporters_in_front_of_matrixStats'] == [] and list(PROVIDER['rowCumsums_exported_by']) == ['matrixStats']
    assert len(PROVIDER['rowCumsums_exported_by']['matrixStats']) == 13


def test_na_rm_is_swallowed_by_the_dots():
    assert PROVIDER['na_rm_is_a_formal'] is False
    assert all('...' in s and 'na.rm' not in s for s in PROVIDER['signatures'])


def test_arithmetic_statements_of_rowcumsums_are_one_text():
    assert len(PROVIDER['arithmetic_statement_groups']) == 1
    assert set(PROVIDER['c_template_uses_long_double'].values()) == {True}


def test_only_as_matrix_is_reexported_and_only_by_proxy():
    assert PROVIDER['base_names_exported_by_attached_packages'] == {'as.matrix': {'proxy': 6}}
    assert all(r['definitions'] == ['as.matrix <- function(x, ...) base::as.matrix(x, ...)'] for r in PROVIDER['proxy_as_matrix'])


# --- stress inputs -------------------------------------------------------------------------------------------------

def test_stress_fixture_is_regenerated_exactly():
    stored = _json('stress_fixture.json')
    rng = random.Random(stored['seed'])
    for name, make in (('age_depth', stress.age_depth), ('selection', stress.selection), ('final_pdf', stress.final_pdf),
                       ('cdf', stress.cdf), ('tf', stress.tf), ('pruning', stress.pruning)):
        assert stored[name] == [make(rng) for _ in range(stored['per_module'])]


def test_stress_fixture_is_marked_synthetic():
    stored = _json('stress_fixture.json')
    assert stored['kind'].startswith('synthetic') and stored['historical_reproduction_ready'] is False
