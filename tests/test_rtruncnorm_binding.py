"""Stored evidence of the source and binding check of rtruncnorm_cpp; no draw, no compilation, no network and no access to diss needed."""
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / 'docs/thesis_research/literal_rtruncnorm_binding_2026_11_09'
F, P, M, C = 'src/mcmc_functions_v1.6.R', 'src/mcmc_prepare.R', 'main_v1.2.R', 'src/mcmc_conduction.R'
HEADER_SHA = 'c1fdd6889a30'
RANDOM = {'R::runif', 'R::rnorm', 'R::rexp', 'R::rgamma', 'unif_rand', 'norm_rand', 'exp_rand', 'Rcpp::runif', 'Rcpp::rnorm'}


def _json(name):
    return json.loads((PACKET / name).read_text(encoding='utf-8'))


CRAN = _json('rcppdist_versions.json')
BINDING = _json('source_binding.json')
VERSION = CRAN['versions_in_window'][0]
FUNCTIONS = VERSION['functions']


def test_exactly_one_cran_version_was_current_inside_the_window():
    assert CRAN['window'] == ['2018-04-23', '2023-07-07']
    assert [v['file'] for v in CRAN['versions_in_window']] == ['RcppDist_0.1.1.tar.gz']
    assert CRAN['last_version_before_window'] is None


def test_the_gap_at_the_start_of_the_window_stays_visible():
    assert VERSION['date_publication'] == '2018-10-28 22:50:09 UTC' and CRAN['first_publication_on_cran'] == '2018-10-28'
    assert CRAN['window_days_without_any_cran_version'] is True
    assert 'unknown' in CRAN['not_bound'] and 'not excluded' in CRAN['not_bound']


def test_the_later_version_has_the_same_header_and_is_only_a_comparison():
    assert [v['file'] for v in CRAN['versions_outside_window_for_comparison']] == ['RcppDist_0.1.1.1.tar.gz']
    assert CRAN['header_identical_in_all_cran_versions'] is True
    assert CRAN['versions_outside_window_for_comparison'][0]['header_sha256'] == VERSION['header_sha256']
    assert VERSION['header_sha256'].startswith(HEADER_SHA)


def test_one_header_defines_r_truncnorm():
    assert VERSION['members_named_truncnorm_h'] == ['RcppDist/inst/include/truncnorm.h'] == VERSION['members_defining_r_truncnorm']
    assert VERSION['license'] == 'GPL (>= 2.0)'


def test_r_truncnorm_is_one_uniform_through_the_quantile_function():
    assert (FUNCTIONS['r_truncnorm']['first_line'], FUNCTIONS['r_truncnorm']['last_line']) == (225, 228)
    assert FUNCTIONS['r_truncnorm']['calls'] == {'R::runif': 1, 'q_truncnorm(': 1}


@pytest.mark.parametrize('name,span,calls', [('q_truncnorm', (209, 223), {'R::pnorm': 2, 'R::qnorm': 1}), ('d_truncnorm', (142, 152), {'R::pnorm': 2, 'R::dnorm': 2})])
def test_the_partner_functions_use_no_random_numbers(name, span, calls):
    assert (FUNCTIONS[name]['first_line'], FUNCTIONS[name]['last_line']) == span
    assert not RANDOM & set(FUNCTIONS[name]['calls'])
    assert all(FUNCTIONS[name]['calls'][key] == value for key, value in calls.items())


def test_the_pinned_originals_are_the_frozen_ones():
    assert BINDING['sources_sha256'] == {M: '82486f99bafff08af2b51475ed225501d26cec1a0322a286b97a248ebd2f0520', F: 'b887144370e328fa016d1834ceebcd55622816d4b36838d33a26197110ab453c',
                                         P: 'b897e84f8f2d72399de729592066341585974d521bf8164b185f20377ed5cc7e', C: '47c59c508e15150f7d6fda775a4f6bf6c14f0148e10cefd0014fa1ce6cfe5a9d'}
    assert {1556, 1558, 1560, 1563, 1566, 1812, 1813, 1485, 1516, 1521} <= set(BINDING['pinned_lines'][F])


def test_the_run_has_one_definition_and_two_callers():
    assert BINDING['definitions_of_rtruncnorm_cpp_in_source_tree'] == [f'{F}:1563'] and BINDING['r_files_scanned_for_definitions'] == 95
    assert BINDING['calls']['rtruncnorm_cpp'] == {F: [1812, 1813], C: [], P: []}


def test_propose_cpp_has_no_active_caller():
    assert BINDING['calls']['propose_cpp_uncommented'] == {F: [], C: []}


def test_the_active_density_reader_is_the_second_definition():
    assert BINDING['calls']['ratios_tfs_no_ages_uncommented'] == {C: [228]}
    assert {1437, 1485} <= set(BINDING['pinned_lines'][F])


def test_the_r_package_truncnorm_is_a_separate_open_provider():
    assert BINDING['calls']['r_package_truncnorm_rtruncnorm_uncommented'] == {P: [1957, 1958]} and 1886 in BINDING['pinned_lines'][P]


def test_seeds_and_generator_calls_stay_listed():
    assert BINDING['seeds_uncommented'] == {M: [1055], F: [], P: [1297, 1301, 1332, 1341, 1354, 1896, 1898], C: []}
    assert BINDING['rngkind_or_rngversion_uncommented'] == {M: [], F: [], P: [], C: []}


@pytest.mark.parametrize('name,lines', [('src/nmath/runif.c', [26, 30, 36]), ('src/main/RNG.c', [97, 100, 101, 134, 31, 46]), ('src/nmath/qnorm.c', [52]), ('src/nmath/pnorm.c', [62]), ('src/nmath/dnorm.c', [34])])
def test_r_350_locations(name, lines):
    assert BINDING['r_3_5_0']['locations'][name]['lines'] == lines


def test_nothing_was_drawn_compiled_or_released():
    assert BINDING['random_state_used'] is False and BINDING['draws_run'] == 0 and BINDING['compiled'] is False and BINDING['historical_reproduction_ready'] is False


@pytest.mark.parametrize('name', ['rcppdist_versions.json', 'source_binding.json'])
@pytest.mark.parametrize('marker', ['Copyright', 'GNU General Public', 'inline double', 'std::max', 'R_NegInf', 'do {u'])
def test_no_gpl_text_is_stored(name, marker):
    assert marker not in (PACKET / name).read_text(encoding='utf-8')


def test_local_tarball_matches_when_present():
    tarball = ROOT / '.local/rtruncnorm_2026_11_09/cran/RcppDist/RcppDist_0.1.1.tar.gz'
    if not tarball.exists():
        pytest.skip('local CRAN cache not present')
    assert hashlib.sha256(tarball.read_bytes()).hexdigest() == VERSION['tarball_sha256']
