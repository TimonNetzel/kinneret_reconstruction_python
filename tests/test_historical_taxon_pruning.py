"""Historical taxon pruning (mcmc_prepare.R 51-80): exact reference, R index quirks and domain limits."""
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / 'docs/thesis_research/literal_taxon_pruning_2026_11_02'
sys.path.insert(0, str(ROOT / 'tools'))
import historical_taxon_pruning as pruning  # noqa: E402

_spec = importlib.util.spec_from_file_location('pruning_rational_reference', PACKET / 'rational_reference.py')
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)


def _json(name):
    return json.loads((PACKET / name).read_text(encoding='utf-8'))


FIXTURE = _json('synthetic_fixture.json')
CASES = FIXTURE['cases']
FLAGS = ('null_ratio_check', 'metro_hastings', 'relict_override')


def _run(name):
    case = CASES[name]
    return pruning.prune_taxa(case['pollen'], **{k: case[k] for k in FLAGS})


def _zeros(rows, zeros):
    return [[0 if i < z else 1 + j + i % 3 for j, z in enumerate(zeros)] for i in range(rows)]


@pytest.mark.parametrize('name', list(CASES))
def test_module_matches_independent_exact_reference(name):
    case = CASES[name]
    exact = reference.exact_pruning(case['pollen'], **{k: case[k] for k in FLAGS})
    got = _run(name)
    assert (got.executed, got.branch, got.metro_hastings, got.skip_site) == \
        (exact['executed'], exact['branch'], exact['metro_hastings'], exact['skip_site'])
    assert list(got.removed_taxa) == exact['removed_taxa'] and list(got.kept_taxa) == exact['kept_taxa']
    for field in ('null_ratio', 'sorting_available_spectrum'):
        value = getattr(got, field)
        assert (value is None) == (exact[field] is None)
        if value is not None:
            np.testing.assert_allclose(value, [float(Fraction(v)) for v in exact[field]], rtol=0, atol=FIXTURE['tolerance'])


@pytest.mark.parametrize('name', list(CASES))
def test_stored_result_file_matches_a_fresh_module_run(name):
    stored, got = _json('synthetic_result.json')['cases'][name], _run(name)
    assert stored['removed_taxa'] == list(got.removed_taxa) and stored['branch'] == got.branch
    assert stored['pruned_shape'] == (None if got.pollen is None else list(got.pollen.shape))


def test_hand_anchor_tie_extension_keeps_a_rare_taxon():
    # 17 taxa, ratios 1, 19/20, 1, 1 and 13 below: 17 - 4 <= 15, sort()[16:17] = {1}
    got = _run('rarest_15_tie_keeps_rare')
    assert got.branch == 'rarest_15' and not got.metro_hastings
    assert got.removed_taxa == (0, 2, 3) and 1 in got.kept_taxa and got.null_ratio[1] == 0.95


def test_hand_anchor_descending_colon_removes_the_maximum_below_095():
    # six taxa: 16:6 = 16, 15, ..., 6; only position 6 exists and holds the maximum 1/2
    got = _run('short_core_descending_colon')
    assert got.null_ratio == (0.0, 0.1, 0.25, 0.5, 0.5, 0.15)
    assert got.removed_taxa == (3, 4) and got.pollen.shape == (20, 4)


def test_exact_095_is_removed_and_090_kept_in_the_ratio_branch():
    got = _run('ratio_095_branch')
    assert got.branch == 'ratio_095' and got.metro_hastings
    assert got.removed_taxa == (0, 1) and 2 in got.kept_taxa and got.null_ratio[:3] == (0.95, 1.0, 0.9)


def test_bound_loop_flag_false_leaves_everything_unchanged():
    case = CASES['flag_false_bound_loop']
    got = _run('flag_false_bound_loop')
    assert not got.executed and got.branch == 'not_executed' and got.null_ratio is None
    assert got.sorting_available_spectrum is None and got.metro_hastings is False
    np.testing.assert_array_equal(got.pollen, case['pollen'])
    got.pollen[0, 0] = 99
    assert case['pollen'][0][0] != 99


def test_equal_ratios_signal_next_instead_of_returning_an_empty_matrix():
    got = _run('all_equal_skip_site')
    assert got.skip_site and got.pollen is None and got.sorting_available_spectrum is None
    assert got.removed_taxa == (0, 1, 2, 3) and got.kept_taxa == ()


def test_metro_hastings_is_only_ever_set_true():
    preset = _run('metro_preset_true')
    assert preset.metro_hastings and preset.branch == 'ratio_095' and preset.removed_taxa == ()
    unset = _run('short_core_descending_colon')
    assert not unset.metro_hastings


@pytest.mark.parametrize('n_taxa, removed', [(15, (14,)), (16, ())])
def test_fifteen_taxa_boundary(n_taxa, removed):
    # no rare taxa: 15 taxa keep metro_hastings FALSE and lose the maximum, 16 taxa switch to the 0.95 rule
    got = pruning.prune_taxa(_zeros(20, list(range(n_taxa))), null_ratio_check=True, metro_hastings=False,
                             relict_override=False)
    assert got.removed_taxa == removed and got.metro_hastings == (n_taxa == 16)


def test_relict_override_forces_ratios_before_the_threshold():
    got = _run('relict_override_26')
    assert got.null_ratio[0] == 1.0 and all(got.null_ratio[i - 1] == 0.0 for i in pruning.RELICT_IDS_R_1_BASED)
    assert got.removed_taxa == (0, 1)
    plain = pruning.prune_taxa(CASES['relict_override_26']['pollen'], null_ratio_check=True,
                               metro_hastings=False, relict_override=False)
    assert plain.removed_taxa == (1, 2, 7, 22)


def test_relict_override_with_fewer_than_25_taxa_is_rejected():
    with pytest.raises(pruning.UnsupportedSourceDomain, match='25 taxa'):
        pruning.prune_taxa(_zeros(20, [0] * 24), null_ratio_check=True, metro_hastings=False, relict_override=True)


def test_single_remaining_taxon_would_drop_dimension_and_is_rejected():
    with pytest.raises(pruning.UnsupportedSourceDomain, match='drops to a vector'):
        pruning.prune_taxa(_zeros(4, [0, 2]), null_ratio_check=True, metro_hastings=False, relict_override=False)


@pytest.mark.parametrize('bad', [
    [[1, np.nan]], [[1, np.inf]], [[1, -1]], [['1', '2']], [[True, False]], [[1 + 0j, 2]],
    [1, 2, 3], [[]], [[[1]]], 'abc'])
def test_unsupported_input_is_rejected_without_repairs(bad):
    with pytest.raises(pruning.UnsupportedSourceDomain):
        pruning.prune_taxa(bad, null_ratio_check=True, metro_hastings=False, relict_override=False)


@pytest.mark.parametrize('flag', [1, 0, 'TRUE', None, np.bool_(True)])
def test_control_flags_must_be_explicit_python_booleans(flag):
    with pytest.raises(pruning.UnsupportedSourceDomain, match='booleans'):
        pruning.prune_taxa([[1, 2]], null_ratio_check=flag, metro_hastings=False, relict_override=False)


def test_input_and_column_order_are_preserved():
    case = CASES['short_core_descending_colon']
    before = json.dumps(case['pollen'])
    got = _run('short_core_descending_colon')
    assert json.dumps(case['pollen']) == before
    np.testing.assert_array_equal(got.pollen, np.asarray(case['pollen'])[:, [0, 1, 2, 5]])


def test_r_colon_counts_down_and_marks_positions_beyond_the_end():
    assert pruning._r_index([1, 2, 3], 16, 3) == [None] * 13 + [3]
    assert pruning._r_index(list(range(1, 19)), 16, 18) == [16, 17, 18]


def test_double_threshold_equals_exact_19_20_in_the_supported_row_range():
    for n in range(1, 2001):
        k = np.arange(n + 1)
        assert np.array_equal(k / n >= 0.95, 20 * k >= 19 * n)
    n = np.arange(20, pruning.MAX_ROWS + 1, 20)
    assert ((19 * n // 20) / n >= 0.95).all() and not ((19 * n // 20 - 1) / n >= 0.95).any()


def test_frozen_counterexamples_differ_from_both_naive_readings():
    check = _json('arithmetic_check.json')
    assert check['maximum_absolute_rational_error'] <= check['tolerance'] and not check['R_execution']
    differs = check['naive_readings_differing_from_source']
    for name in ('rarest_15_tie_keeps_rare', 'short_core_descending_colon', 'all_equal_skip_site'):
        assert differs[name] == ['naive_ge095_removed', 'naive_rank15_removed']
    assert differs['ratio_095_branch'] == ['naive_rank15_removed']


def test_source_binding_and_determinism_audit_claim_no_run_or_variant():
    binding, audit = _json('source_binding.json'), _json('determinism_audit.json')
    assert binding['historical_reproduction_ready'] is False and binding['run_binding'] is None
    assert binding['implemented_span'] == {'prune_taxa': ['src/mcmc_prepare.R', 51, 80]}
    excerpt = (PACKET / 'source_excerpt.json').read_bytes()
    assert hashlib.sha256(excerpt).hexdigest() == binding['portable_excerpt_sha256']
    assert audit['selected_for_implementation'] == 'taxon_pruning' and audit['final_variant_selected'] is False
    by_id = {c['id']: c for c in audit['candidates']}
    assert by_id['taxon_pruning']['random_state'] is False and not by_id['taxon_pruning']['package_semantics_needed']
    assert by_id['taxon_pruning']['active_in_bound_reconstruction_loop'] is False
    assert by_id['weightedMedian']['package_semantics_needed'] is True
    assert all(c['decision'].startswith('documented only') for c in audit['candidates'] if c['id'] != 'taxon_pruning')


def test_excerpt_contains_the_literal_source_lines():
    spans = {(s['source'], s['first_line']): s['text'] for s in _json('source_excerpt.json')['spans']}
    block = spans[('original:src/mcmc_prepare.R', 45)]
    assert 'rm.id <- which(null.ratio %in% sort(null.ratio)[16:ncol(all.avail)])' in block
    assert 'if(ncol(all.avail) - length(rm.id1) > 15) metro_hastings <- TRUE' in block
    assert 'null.ratio.check <- FALSE' in spans[('original:main_v1.2.R', 651)]
    assert 'weightedMedian(x1range.bck,tf.x1.current[j,] )' in spans[('original:src/mcmc_prepare.R', 1622)]
