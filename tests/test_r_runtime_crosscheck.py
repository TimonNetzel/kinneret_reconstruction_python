"""Stored evidence of the R 3.5.0 cross check, the site slots and the package providers; R itself is not needed."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / 'docs/thesis_research/r_runtime_crosscheck_2026_11_03'


def _json(name):
    return json.loads((PACKET / name).read_text(encoding='utf-8'))


CROSS = _json('crosscheck_result.json')


def test_crosscheck_ran_literal_lines_in_r_350_without_packages_or_random_state():
    assert CROSS['R'] == {'version': 'R version 3.5.0 (2018-04-23)', 'sizeof_longdouble': 16}
    assert CROSS['packages_attached'] == [] and CROSS['random_state_used'] is False
    assert CROSS['original_lines'] == {'sampler_preparation': 'src/mcmc_conduction.R:53-79',
                                       'weights_stage_core': 'src/mcmc_functions_v1.6.R:1276-1280',
                                       'gamma_grid': 'src/mcmc_prepare.R:1270-1281',
                                       'num_taxa_priors': 'src/mcmc_prepare.R:1681-1685'}
    assert CROSS['historical_reproduction_ready'] is False


def test_every_unflagged_value_is_bit_equal():
    assert CROSS['all_unflagged_values_bit_equal'] is True and CROSS['checks'] == 389
    for section in ('sampler_preparation', 'weights_stage', 'gamma_grid', 'num_taxa_priors'):
        for row in CROSS[section].values():
            assert all(v for k, v in row.items() if k.endswith('equal'))


@pytest.mark.parametrize('section,count', [('sampler_preparation', 8), ('weights_stage', 9), ('gamma_grid', 6),
                                           ('num_taxa_priors', 59)])
def test_all_fixture_cases_were_compared(section, count):
    assert len(CROSS[section]) == count


def test_flagged_values_are_reported_and_equal_on_this_platform():
    means = [x for row in CROSS['sampler_preparation'].values() for x in row['means_flagged_dependent']]
    rows = [x for row in CROSS['weights_stage'].values() for x in row['rows_flagged_dependent']]
    assert len(means) + len(rows) == 14
    assert all(x['equal'] for x in means) and all(x['row_equal'] for x in rows)


def test_kinneret_is_slot_3_of_the_add_sites_branch():
    sites = _json('site_binding.json')
    assert sites['kinneret_slot'] == 3 and sites['slot_sources']['3'] == 'my_kinneret'
    assert sites['kinneret_switch_brace_depth']['1248'] == 0
    assert set(sites['kinneret_switches_mcmc_prepare']) == {'790', '881', '1145', '1209', '1248'}
    assert sites['historical_reproduction_ready'] is False


def test_provider_analysis_names_no_runtime_binding():
    providers = _json('provider_analysis.json')
    assert providers['kind'] == 'search_path_provider_analysis_from_CRAN_namespaces_not_runtime_binding'
    assert set(providers['providers']) == {'rdirichlet', 'ddirichlet', 'weightedMedian'}
    assert providers['window'] == ['2018-04-23', '2023-07-07']


def test_providers_within_the_window_and_nothing_in_front_of_them():
    providers = _json('provider_analysis.json')
    assert providers['providers_bound_within_window'] == {'rdirichlet': 'LaplacesDemon', 'ddirichlet': 'LaplacesDemon',
                                                          'weightedMedian': 'matrixStats'}
    assert providers['tarball_resolution']['blocking_providers_in_front'] == {
        'rdirichlet': [], 'ddirichlet': [], 'weightedMedian': []}
    exporting = [h['package'] for h in providers['providers']['rdirichlet']['all_exporting']]
    assert exporting == ['LaplacesDemon', 'extraDistr', 'gtools', 'SBMSplitMerge']
    path = providers['search_path_front_to_back']
    assert path.index('LaplacesDemon') < path.index('extraDistr') < path.index('gtools') < path.index('SBMSplitMerge')


def test_laplacesdemon_dirichlet_text_is_identical_in_every_window_version():
    d = _json('dirichlet_versions.json')
    assert [r['version'] for r in d['versions']] == ['16.1.0', '16.1.1', '16.1.4', '16.1.6']
    assert d['distinct'] == {'rdirichlet': 1, 'ddirichlet': 1}
    (ddir,) = d['texts']['ddirichlet'].values()
    assert 'if(any(rowSums(x) != 1)) x / rowSums(x)' in ddir['text']  # result not assigned: no renormalisation
    (rdir,) = d['texts']['rdirichlet'].values()
    assert 'rgamma(' in rdir['text']  # random state: stays documented only


def test_weighted_median_signature_is_stable_but_sources_differ():
    w = _json('weighted_median_versions.json')
    assert len(w['versions']) == 13 and w['distinct_sources'] == 6
    assert {r['signature'] for r in w['versions']} == {
        'x, w = NULL, idxs = NULL, na.rm = FALSE, interpolate = is.null(ties), ties = NULL, ...'}


def test_run_inputs_were_written_by_r_350():
    r = _json('rdata_writers.json')['run_inputs_and_kinneret']
    assert r['writer_counts'] == {'3.5.0': 44, '2.11.0': 1}
    assert all(row['writer_R'] == '3.5.0' for row in r['rows'] if row['path'].startswith('src/local_reconstr_all_epd/'))
