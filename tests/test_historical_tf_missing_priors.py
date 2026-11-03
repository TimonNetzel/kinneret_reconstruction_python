"""Historical which.tf.missing removal and num_taxa priors (mcmc_prepare.R 1664-1685): reference, R quirks, domain."""
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
PACKET = ROOT / 'docs/thesis_research/literal_tf_missing_priors_2026_11_02'
sys.path.insert(0, str(ROOT / 'tools'))
import historical_tf_missing_priors as tmp  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, PACKET / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference = _load('tf_missing_reference', 'rational_reference.py')
check = _load('tf_missing_check', 'check_tf_missing_priors.py')


def _json(name):
    return json.loads((PACKET / name).read_text(encoding='utf-8'))


CASES = _json('synthetic_fixture.json')['cases']


def _run(name):
    mask, kwargs = check.module_inputs(CASES[name])
    return tmp.remove_missing_tfs(mask, **kwargs)


def _plain(n_taxa=4, samples=3, ages=2, cells=3, **override):
    """Small finite inputs; override replaces single objects."""
    kwargs = dict(taxa=[f'T{i}' for i in range(n_taxa)],
                  all_avail=np.arange(samples * n_taxa, dtype=float).reshape(samples, n_taxa) + 1,
                  all_avail_age=np.ones((ages, n_taxa)), taxa_appear_core=np.ones((ages, n_taxa)),
                  tf_x1=np.full((n_taxa, cells), 0.5), tf_x2=np.full((n_taxa, cells), 0.25),
                  x1_mean=[1.0] * n_taxa, x2_mean=[2.0] * n_taxa, taxa_change=False)
    kwargs.update(override)
    return kwargs


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
    assert (stored['num_taxa'], stored['nr_taxa_sd'], stored['taxa']) == \
        (expected['num_taxa'], expected['nr_taxa_sd'], expected['taxa'])


@pytest.mark.parametrize('name', list(CASES))
def test_hand_expectations(name):
    h, got = CASES[name]['hand'], _run(name)
    assert list(got.which_tf_missing) == h['which_tf_missing'] and list(got.taxa) == h['kept_taxa']
    assert (got.priors.num_taxa, got.num_taxa_line, got.priors.nr_taxa_sd, got.priors.phi) == \
        (h['num_taxa'], h['num_taxa_line'], h['nr_taxa_sd'], h['phi'])
    assert (got.stale_phi1, got.stale_phi2) == (h['stale_phi1'], h['stale_phi2'])


# --- guard, order and num_taxa source ---------------------------------------------------------------------------

def test_empty_mask_keeps_everything_and_num_taxa_from_1435():
    got = _run('empty_mask_keeps_1435')
    assert not got.removal_applied and got.which_tf_missing == () and got.num_taxa_line == 1435
    assert got.priors.num_taxa == 5 and got.priors.which_taxa == (1, 2, 3, 4, 5)
    assert _json('arithmetic_check.json')['naive_counterexamples']['empty_mask_keeps_1435'][
        'unguarded_empty_index_num_taxa'] == 0


def test_removal_keeps_remaining_order_in_every_object():
    got = _run('remove_two_keeps_order')
    case = CASES['remove_two_keeps_order']
    assert got.which_tf_missing == (2, 5) and got.kept == (0, 2, 3, 5)
    assert got.taxa == ('Kunsttaxon A', 'Kunsttaxon C', 'Kunsttaxon D', 'Kunsttaxon F')
    assert got.all_avail.tolist() == [[row[i] for i in (0, 2, 3, 5)] for row in case['all_avail']]
    assert got.tf_x1.tolist() == [case['tf_x1'][i] for i in (0, 2, 3, 5)]
    assert got.x1_mean == (10.0, 12.0, 13.0, 15.0) and got.x2_mean == (300.0, 320.0, 330.0, 350.0)


def test_num_taxa_after_removal_is_nrow_of_tf():
    got = _run('tie_n10_to_even_two')
    assert got.num_taxa_line == 1678 and got.priors.num_taxa == got.tf_x1.shape[0] == 10
    assert got.all_avail.shape == (3, 10) and got.all_avail_age.shape == got.taxa_appear_core.shape == (2, 10)


def test_only_the_x1_mean_mask_counts():
    got = _run('x2_only_na_is_kept')
    assert got.which_tf_missing == (2,) and 'Kunsttaxon C' in got.taxa and math.isnan(got.x2_mean[1])
    naive = _json('arithmetic_check.json')['naive_counterexamples']['x2_only_na_is_kept']
    assert naive['mask_x1_or_x2_removes'] == [2, 3] and naive['mask_x1_or_x2_differs']


def test_mask_is_explicit_not_derived_from_values():
    kwargs = _plain(x1_mean=[math.nan, 1.0, 1.0, 1.0])
    assert tmp.remove_missing_tfs([False] * 4, **kwargs).removal_applied is False  # NaN alone removes nothing
    assert tmp.remove_missing_tfs([False, True, False, False], **_plain()).which_tf_missing == (2,)


def test_taxa_change_reduces_only_the_change_means():
    got = _run('taxa_change_true_branch')
    assert got.x1_mean_change == (20.0, 22.0, 23.0, 24.0) and got.x2_mean_change == (400.0, 402.0, 403.0, 404.0)
    assert _run('remove_two_keeps_order').x1_mean_change is None


def test_stale_phi1_phi2_and_discarded_posterior():
    got = _run('tie_n2_to_even_zero')
    assert (got.priors.phi, got.stale_phi1, got.stale_phi2) == (1.0, 2.0, 1.0)
    assert all(not _run(n).posterior_nr_taxa_mean_survives for n in CASES)
    assert [n for n, v in _json('arithmetic_check.json')['naive_counterexamples'].items()
            if not v['phi1_phi2_stale_differs']] == ['empty_mask_keeps_1435']


def test_inputs_are_not_modified():
    mask, kwargs = check.module_inputs(CASES['remove_two_keeps_order'])
    before = copy.deepcopy(kwargs)
    tmp.remove_missing_tfs(mask, **kwargs)
    assert json.dumps(kwargs, default=str) == json.dumps(before, default=str)


# --- R round(num_taxa/4) and the other priors -------------------------------------------------------------------

@pytest.mark.parametrize('x, expected', [(0.0, 0), (0.25, 0), (0.5, 0), (0.75, 1), (1.25, 1), (1.5, 2), (2.5, 2),
                                         (3.5, 4), (4.5, 4), (5.5, 6), (2.0, 2), (7.49, 7)])
def test_round_digits0_goes_to_the_even_digit_on_ties(x, expected):
    assert tmp.r_round_digits0(x) == expected


def test_round_matches_the_integer_reference_and_half_up_differs_for_n_mod_8_eq_2():
    for n in range(0, 401):
        assert tmp.r_round_digits0(n / 4) == reference.round_even_quarter(n)
        assert (n / 4) * 4 == n  # exact double
    differs = [n for n in range(1, 41) if reference.round_even_quarter(n) != reference.round_half_up_quarter(n)]
    assert differs == [2, 10, 18, 26, 34] == _json('arithmetic_check.json')['round_ties']['half_up_differs_for']


@pytest.mark.parametrize('bad', [-0.5, math.nan, math.inf, True, '1.5', 2.0**52, None])
def test_round_rejects_outside_the_checked_domain(bad):
    with pytest.raises(tmp.UnsupportedSourceDomain):
        tmp.r_round_digits0(bad)


def test_priors_for_one_taxon_and_weights_correctly_rounded():
    p = tmp.num_taxa_priors(1)
    assert (p.which_taxa, p.taxa_weights, p.phi, p.nr_taxa_mean, p.nr_taxa_sd) == ((1,), (1.0,), 0.5, 1, 0)
    for n in range(1, 60):
        q = tmp.num_taxa_priors(n)
        assert q.taxa_weights == (float(Fraction(1, n)),) * n and q.phi == float(Fraction(n, 2))
        assert q.nr_taxa_sd_is_tie == (n % 4 == 2)
    assert tmp.num_taxa_priors(np.int64(6)).nr_taxa_sd == 2


@pytest.mark.parametrize('bad', [0, -3, True, 2.0, '4', None])
def test_priors_reject_non_positive_or_non_integer_counts(bad):
    with pytest.raises(tmp.UnsupportedSourceDomain):
        tmp.num_taxa_priors(bad)


# --- rejected R semantics -----------------------------------------------------------------------------------------

def test_one_remaining_taxon_is_rejected_instead_of_dropping_to_a_vector():
    with pytest.raises(tmp.UnsupportedSourceDomain, match='drops to a vector'):
        tmp.remove_missing_tfs([True, True, False, True], **_plain())


def test_every_tf_missing_is_rejected():
    with pytest.raises(tmp.UnsupportedSourceDomain, match='c\\(1:0\\)'):
        tmp.remove_missing_tfs([True] * 4, **_plain())


@pytest.mark.parametrize('override', [dict(samples=1), dict(ages=1), dict(cells=1)])
def test_extent_one_on_the_other_axis_is_rejected_when_removing(override):
    with pytest.raises(tmp.UnsupportedSourceDomain, match='drop'):
        tmp.remove_missing_tfs([False, True, False, False], **_plain(**override))
    assert tmp.remove_missing_tfs([False] * 4, **_plain(**override)).priors.num_taxa == 4  # nothing indexed


@pytest.mark.parametrize('field, value', [
    ('taxa', ['a', 'b', 'c']), ('tf_x1', np.ones((5, 3))), ('x2_mean', [1.0] * 3),
    ('all_avail_age', np.ones((2, 5))), ('taxa_appear_core', np.ones((2, 3))), ('taxa', [1, 2, 3, 4]),
    ('all_avail', [[True, False, True, True]] * 3), ('tf_x2', [['a'] * 3] * 4), ('all_avail', [[1, 2], [3]]),
    ('x1_mean', [1.0, True, 1.0, 1.0]), ('x1_mean', '1234'), ('all_avail', np.ones((0, 4)))])
def test_misaligned_or_non_numeric_objects_are_rejected(field, value):
    with pytest.raises(tmp.UnsupportedSourceDomain):
        tmp.remove_missing_tfs([False, True, False, False], **_plain(**{field: value}))


@pytest.mark.parametrize('mask', [[0, 1, 0, 0], [False, None, False, False], 'FTFF', [False] * 3])
def test_mask_must_be_explicit_booleans_of_full_length(mask):
    with pytest.raises(tmp.UnsupportedSourceDomain):
        tmp.remove_missing_tfs(mask, **_plain())


def test_numpy_boolean_mask_is_accepted():
    assert tmp.remove_missing_tfs(np.array([False, True, False, False]), **_plain()).which_tf_missing == (2,)


def test_taxa_change_must_be_boolean_and_match_the_change_means():
    with pytest.raises(tmp.UnsupportedSourceDomain):
        tmp.remove_missing_tfs([False] * 4, **_plain(taxa_change=1))
    with pytest.raises(tmp.UnsupportedSourceDomain):
        tmp.remove_missing_tfs([False] * 4, **_plain(taxa_change=True))
    with pytest.raises(tmp.UnsupportedSourceDomain):
        tmp.remove_missing_tfs([False] * 4, **_plain(x1_mean_change=[1.0] * 4, x2_mean_change=[1.0] * 4))


# --- packet binding -----------------------------------------------------------------------------------------------

def test_source_excerpt_matches_binding_and_contains_the_literal_lines():
    binding = _json('source_binding.json')
    raw = (PACKET / 'source_excerpt.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == binding['portable_excerpt_sha256']
    text = '\n'.join(s['text'] for s in json.loads(raw)['spans'])
    for line in ('which.tf.missing <- which(is.na(tf.x1.current.mean))', 'if(length(which.tf.missing) != 0){',
                 'num_taxa <- nrow(tf.x1.current)', 'prior$nr.taxa.sd <- round(num_taxa/4)',
                 'num_taxa <- ncol(all.avail)', 'prior$phi2 <- (num_taxa-2)/2', 'taxa_change <- FALSE',
                 'if(any(sitename == c("Birkat Ram","Dead Sea")) & FALSE){', 'posterior <- list()',
                 'tf.x1.current.mean[j] <- weightedMedian(x1range.bck,tf.x1.current[j,] )'):
        assert line in text
    assert binding['brace_depth_at_line_start']['1681'] == 0 and binding['brace_depth_at_line_start']['1678'] == 1
    assert binding['assignment_sites_mcmc_prepare']['num_taxa'] == [1435, 1678]
    assert all(v == [] for k, v in binding['prior_field_occurrences_without_comments'].items()
               if k != 'src/mcmc_prepare.R')
    assert binding['historical_reproduction_ready'] is False and binding['run_binding'] is None


def test_audit_selects_only_this_step_and_keeps_weighted_median_open():
    audit = _json('determinism_audit.json')
    by_id = {c['id']: c for c in audit['candidates']}
    chosen = by_id['which_tf_missing_num_taxa_priors']
    assert audit['selected_for_implementation'] == chosen['id'] and not audit['final_variant_selected']
    assert not chosen['random_state'] and not chosen['package_semantics_needed'] and not audit['R_execution']
    assert by_id['weightedMedian']['package_semantics_needed'] and by_id['prior_samples_seed_2022']['random_state']
    assert by_id['default_thresholds_131_137']['decision'].startswith('not selected')


def test_fixture_and_results_are_marked_synthetic():
    fixture = _json('synthetic_fixture.json')
    assert fixture['historical_reproduction_ready'] is False and fixture['selected_final_variant'] is None
    arithmetic = _json('arithmetic_check.json')
    assert arithmetic['kind'] == 'synthetic_fragment_check_not_historical_reproduction'
    assert not (arithmetic['historical_reproduction_ready'] or arithmetic['final_variant_selected']
                or arithmetic['R_execution'])
    assert all(h['agrees'] for h in arithmetic['hand_expectations'].values())
