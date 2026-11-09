"""Stored input contract of the Kinneret case (slot 3): line references and the status of every input. No calculation, no draw; diss is only touched by the one test that skips without it."""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / 'docs/thesis_research/kinneret_input_contract_2026_11_09'
SPEC = ROOT / 'docs/thesis_research/kinneret_phase1_spec_2026_11_02/spec.json'
REFERENCES = ROOT / 'docs/thesis_research/kinneret_references_2026_11_02/contract.json'
M, P, C, F, A, K = 'main_v1.2.R', 'src/mcmc_prepare.R', 'src/mcmc_conduction.R', 'src/mcmc_functions_v1.6.R', 'src/check_ages.R', 'src/prior_climate.R'
REPORT = json.loads((PACKET / 'input_contract.json').read_text(encoding='utf-8'))
INPUTS = {i['id']: i for i in REPORT['inputs']}
CONDITIONS = {c['id']: c for c in REPORT['conditions']}
TABLE = {(r['site_name'], r['sources_core'], r['sources_speleothem']): r for r in REPORT['source_flag_table']}
STATUS = {
    'core_container': 'fehlend', 'site_name': 'quellkonflikt', 'taxa_spectrum': 'quellkonflikt', 'taxon_names': 'quellkonflikt', 'depth': 'teilweise_gebunden', 'age_grid': 'quellkonflikt',
    'rho_depth_given_age': 'quellkonflikt', 'coordinates': 'teilweise_gebunden', 'country_elevation': 'teilweise_gebunden', 'ap_reference': 'gebunden', 'human_impact': 'gebunden', 'tf_container': 'fehlend',
    'tf_axes': 'fehlend', 'taxa_name_pool': 'teilweise_gebunden', 'tf_current': 'fehlend', 'cru_pann_ww': 'fehlend', 'cru_tann_ww': 'fehlend', 'cru4_pann': 'fehlend', 'cru4_tann': 'fehlend', 'climatezones': 'fehlend',
    'reference_speleo_iso': 'fehlend', 'sisal_speleo_info': 'teilweise_gebunden', 'sisal_ages_info': 'teilweise_gebunden', 'sisal_d18_d13': 'fehlend', 'isotope_info': 'fehlend', 'isotope_speleo_other': 'fehlend', 'd18o_age': 'fehlend',
    'soreq_index': 'quellkonflikt', 'medstack_series': 'fehlend', 'giesecke_grid': 'gebunden', 'insolation': 'fehlend', 'station_ids': 'gebunden', 'kgc_rdata': 'gebunden', 'prob_c_kgc': 'fehlend', 'bond2001': 'fehlend', 'pmip3_mask': 'fehlend',
    'r_version': 'teilweise_gebunden', 'pkg_laplacesdemon': 'teilweise_gebunden', 'pkg_matrixstats': 'teilweise_gebunden', 'pkg_rcppdist': 'teilweise_gebunden', 'pkg_truncnorm': 'fehlend', 'pkg_spatstat_utils': 'fehlend', 'pkg_rpref': 'fehlend',
    'pkg_biwavelet': 'fehlend', 'pkg_signal': 'fehlend', 'pkg_mass': 'fehlend', 'pkg_kgc': 'fehlend', 'pkg_armadillo': 'fehlend'}
CONDITION_STATUS = {
    'sources_giesecke': 'belegt', 'inside_giesecke': 'bedingt', 'sources_core': 'bedingt', 'sources_speleothem': 'offen', 'sources_marine': 'bedingt', 'multiple_spectral': 'offen', 'wt_power_reachable': 'offen', 'site_age_override': 'bedingt', 'recent': 'bedingt',
    'anchor_branch': 'bedingt', 'tann_flag': 'bedingt', 'hi_time': 'belegt', 'hi_spec': 'bedingt', 'stad_interstad': 'belegt', 'taxa_merge': 'offen', 'tf_missing': 'offen', 'active_stop': 'belegt', 'rng_start': 'offen'}
READS = [
    ('core_container', M, 107), ('site_name', A, 7), ('site_name', P, 790), ('site_name', P, 881), ('site_name', P, 1209), ('site_name', P, 1248), ('taxa_spectrum', A, 14), ('taxa_spectrum', P, 93), ('taxon_names', M, 148), ('taxon_names', M, 149), ('taxon_names', M, 150),
    ('depth', A, 178), ('age_grid', M, 656), ('age_grid', A, 179), ('rho_depth_given_age', A, 177), ('rho_depth_given_age', P, 14), ('coordinates', K, 9), ('coordinates', K, 10), ('country_elevation', P, 1036), ('country_elevation', P, 1080), ('ap_reference', M, 143), ('ap_reference', M, 144), ('ap_reference', P, 639),
    ('human_impact', M, 154), ('human_impact', P, 716), ('tf_container', M, 569), ('tf_container', M, 577), ('tf_axes', M, 570), ('tf_axes', P, 1445), ('taxa_name_pool', M, 433), ('taxa_name_pool', P, 1460), ('tf_current', P, 1574), ('tf_current', P, 1581), ('tf_current', P, 1897), ('tf_current', C, 146),
    ('cru_pann_ww', M, 325), ('cru_pann_ww', K, 6), ('cru_tann_ww', M, 326), ('cru4_pann', M, 343), ('cru4_pann', K, 71), ('cru4_tann', M, 344), ('cru4_tann', K, 70), ('climatezones', M, 360), ('climatezones', K, 33), ('reference_speleo_iso', M, 388), ('sisal_speleo_info', M, 391), ('sisal_ages_info', M, 392), ('sisal_d18_d13', M, 394), ('isotope_info', M, 397), ('isotope_speleo_other', M, 401),
    ('d18o_age', P, 751), ('d18o_age', P, 766), ('d18o_age', P, 828), ('d18o_age', P, 1038), ('soreq_index', P, 790), ('soreq_index', P, 828), ('soreq_index', 'src/mcmc_prep_v1.2.R', 773), ('medstack_series', P, 1038), ('giesecke_grid', M, 407), ('giesecke_grid', P, 565), ('insolation', M, 367), ('station_ids', M, 431), ('kgc_rdata', M, 359), ('prob_c_kgc', M, 432), ('bond2001', A, 238), ('pmip3_mask', P, 451), ('pmip3_mask', P, 468),
    ('pkg_truncnorm', P, 1957), ('pkg_rcppdist', F, 1566), ('pkg_matrixstats', P, 1631), ('pkg_laplacesdemon', F, 1248), ('pkg_spatstat_utils', P, 716), ('pkg_rpref', P, 779), ('pkg_biwavelet', P, 934), ('pkg_signal', F, 2325), ('pkg_mass', K, 106), ('pkg_kgc', K, 37), ('pkg_armadillo', F, 1720)]
CONDITION_READS = [('sources_core', P, 636), ('sources_core', P, 661), ('sources_core', P, 662), ('sources_speleothem', P, 759), ('sources_speleothem', P, 772), ('sources_speleothem', P, 773), ('sources_speleothem', P, 813), ('sources_speleothem', P, 814), ('wt_power_reachable', P, 947), ('wt_power_reachable', P, 966), ('wt_power_reachable', F, 2023),
                   ('recent', P, 728), ('recent', P, 1183), ('recent', P, 1184), ('recent', F, 1844), ('site_age_override', P, 825), ('site_age_override', P, 826), ('tann_flag', P, 1210), ('tann_flag', P, 1249), ('hi_time', P, 718), ('stad_interstad', P, 468), ('active_stop', P, 2057), ('rng_start', P, 1898), ('anchor_branch', K, 69), ('taxa_merge', P, 1460), ('tf_missing', P, 1664)]
SETTINGS = {'add.sites': (M, 73, 'TRUE'), 'percent': (M, 80, 'FALSE'), 'gibbs': (M, 624, 'FALSE'), 'sample_length': (M, 630, '25000'), 'burn_in': (M, 631, '5000'), 'thin_size': (M, 632, '5'), 'age.reso': (M, 656, '2'), 'sim_nr': (M, 657, '1'), 'dims': (M, 658, '300'), 'null.ratio.check': (M, 659, 'FALSE'), 'giesecke_prespa': (M, 669, 'FALSE'), 'past_climate_module': (M, 670, 'FALSE'), 'case_studies': (M, 672, 'FALSE')}


def has_read(entry, file, line):
    return any(ref['file'] == file and ref['line'] == line and ref['starts_with'] for ref in entry['code_reads'])


def test_the_pinned_originals_are_the_frozen_ones():
    assert REPORT['sources_sha256'] == {M: '82486f99bafff08af2b51475ed225501d26cec1a0322a286b97a248ebd2f0520', P: 'b897e84f8f2d72399de729592066341585974d521bf8164b185f20377ed5cc7e', C: '47c59c508e15150f7d6fda775a4f6bf6c14f0148e10cefd0014fa1ce6cfe5a9d',
                                        F: 'b887144370e328fa016d1834ceebcd55622816d4b36838d33a26197110ab453c', A: 'c7f92e6030d83540c14f29a2e9d0ad07c74104dfe86befebda921c1cffbd2729', K: '77cfe4e2297abf5f0dd8f7179721702bb1a15e7f9989c9c62374a2fdda642ebd'}


def test_the_list_of_inputs_is_complete():
    assert set(INPUTS) == set(STATUS) and len(REPORT['inputs']) == len(INPUTS) == 48


@pytest.mark.parametrize('ident,status', sorted(STATUS.items()))
def test_status_of_every_input(ident, status):
    assert INPUTS[ident]['status'] == status and INPUTS[ident]['status_reason_de']


def test_status_counts():
    assert REPORT['status_counts'] == {'gebunden': 5, 'teilweise_gebunden': 10, 'quellkonflikt': 6, 'fehlend': 27}
    assert set(REPORT['status_values']) == set(REPORT['status_counts'])


@pytest.mark.parametrize('ident', sorted(STATUS))
def test_every_input_has_a_code_reference(ident):
    assert INPUTS[ident]['code_reads'] and all(ref['line'] > 0 and ref['starts_with'] for ref in INPUTS[ident]['code_reads'])


@pytest.mark.parametrize('ident,file,line', READS)
def test_line_reference_of_an_input(ident, file, line):
    assert has_read(INPUTS[ident], file, line)


@pytest.mark.parametrize('ident,file,line', CONDITION_READS)
def test_line_reference_of_a_condition(ident, file, line):
    assert has_read(CONDITIONS[ident], file, line)


@pytest.mark.parametrize('name', sorted(SETTINGS))
def test_literal_settings(name):
    row = next(s for s in REPORT['settings_literal'] if s['name'] == name)
    assert (row['file'], row['line'], row['value']) == SETTINGS[name]


@pytest.mark.parametrize('ident,status', sorted(CONDITION_STATUS.items()))
def test_status_of_every_condition(ident, status):
    assert CONDITIONS[ident]['status'] == status and CONDITIONS[ident]['reason_de']


def test_the_list_of_conditions_is_complete():
    assert set(CONDITIONS) == set(CONDITION_STATUS) and len(REPORT['conditions']) == 18


def test_conditions_only_name_known_inputs():
    assert all(set(c['depends_on_inputs']) <= set(INPUTS) for c in REPORT['conditions'])


def test_the_site_name_conflict_stays_visible():
    older = REPORT['older_container_slot_3']
    assert older['sitename'] == 'Kinneret' and older['all_sitenames'] == ['Iznik', 'Birkat Ram', 'Kinneret', 'En Gedi']
    assert all('Lake Kinneret' in ref['starts_with'] for ref in INPUTS['site_name']['code_reads'][1:])


def test_the_older_container_lacks_the_fields_the_code_writes_itself():
    older = REPORT['older_container_slot_3']
    assert older['has_field_qualitativ_data'] is False and older['has_field_human_impact'] is False


def test_the_name_decides_whether_the_levant_branch_runs():
    assert TABLE[('Lake Kinneret', 1, 1)]['multiple'] is True and TABLE[('Lake Kinneret', 1, 1)]['marine_ice_core_after_1212'] == 1
    assert TABLE[('Kinneret', 1, 1)]['multiple'] is False and TABLE[('Kinneret', 1, 1)]['marine_ice_core_after_1212'] == 0
    assert all(row['tann'] is False for row in REPORT['source_flag_table'])


@pytest.mark.parametrize('name', ['Lake Kinneret', 'Kinneret'])
@pytest.mark.parametrize('core', [1, 0])
@pytest.mark.parametrize('speleo', [1, 0])
def test_wt_power_is_reachable_exactly_with_the_speleothem_flag(name, core, speleo):
    row = TABLE[(name, core, speleo)]
    assert row['spectral'] is bool(speleo) and row['wt_power_reachable'] is bool(speleo)


def test_the_source_table_is_read_before_line_1212():
    assert TABLE[('Kinneret', 1, 1)]['active_source_indices_at_1189'] == [2, 3, 4]


@pytest.mark.parametrize('scenario,key', [('main_three_channels', ('Lake Kinneret', 1, 1)), ('without_speleo', ('Lake Kinneret', 1, 0))])
def test_the_table_agrees_with_the_frozen_reference_traces(scenario, key):
    traces = json.loads(REFERENCES.read_text(encoding='utf-8'))['control_flow_traces']
    trace = next(t for t in traces if t['scenario'] == scenario)
    row = TABLE[key]
    assert (trace['active_source_indices'], trace['multiple'], trace['spectral']) == (row['active_source_indices_at_1189'], row['multiple'], row['spectral'])


def test_the_coordinate_condition_is_conditional_on_the_older_values():
    condition = CONDITIONS['inside_giesecke']
    evaluation = condition['evaluation']
    assert condition['status'] == 'bedingt' and condition['value_under_older_coordinates'] is False
    assert (evaluation['site_lat_used'], evaluation['site_lon_used']) == (32.79, 35.53)
    assert (evaluation['nearest_lat'], evaluation['cells_on_nearest_lat'], evaluation['lon_range_on_nearest_lat']) == (34.75, 1, [24.75, 24.75]) and evaluation['inside_giesecke_space'] is False
    assert 'not a model step' in evaluation['scope']


def test_the_grid_read_is_the_first_listed_file_as_in_the_code():
    evaluation = CONDITIONS['inside_giesecke']['evaluation']
    assert (evaluation['first_listed_file'], evaluation['last_listed_file'], evaluation['files_listed']) == ('epdQuant_Abi.csv', 'readme.txt', 17)


def test_sources_core_keeps_its_variance_test_open():
    condition = CONDITIONS['sources_core']
    assert condition['branch_reached_under_older_coordinates'] is True and condition['variance_test_open'] is True and condition['status'] == 'bedingt'


def test_speleothem_flag_and_wt_power_stay_open_on_the_missing_object():
    assert CONDITIONS['sources_speleothem']['status'] == CONDITIONS['wt_power_reachable']['status'] == 'offen'
    assert 'd18o_age' in CONDITIONS['sources_speleothem']['depends_on_inputs'] and 'd18o_age' in CONDITIONS['wt_power_reachable']['depends_on_inputs']


def test_no_writer_of_the_missing_objects_is_in_the_code_tree():
    search = REPORT['writer_search']
    assert search['d18o_age_assignments_or_saves'] == [] == search['climatezones_assignments_or_saves'] and search['files_scanned'] == 95
    assert 'not a search of all of diss' in search['scope']


def test_climatezones_is_not_in_the_file_loaded_next_to_it():
    assert REPORT['container_objects']['in/kgc_distribution_ww.rdata'] == ['kgc.distribution.ww'] and REPORT['container_objects']['in/station_ids_ref.rdata'] == ['station_ids_ref']
    assert INPUTS['climatezones']['status'] == 'fehlend' and INPUTS['kgc_rdata']['affects_kinneret_result'] is False


@pytest.mark.parametrize('ident', ['insolation', 'station_ids', 'kgc_rdata', 'bond2001', 'pmip3_mask', 'sisal_speleo_info', 'sisal_ages_info', 'country_elevation'])
def test_inputs_without_influence_are_marked(ident):
    assert INPUTS[ident]['affects_kinneret_result'] is False


def test_the_unknown_influence_is_not_turned_into_none():
    assert INPUTS['prob_c_kgc']['affects_kinneret_result'] == 'unbekannt' and INPUTS['pkg_biwavelet']['affects_kinneret_result'] == 'bedingt'


def test_the_transfer_functions_stay_missing():
    assert all(INPUTS[i]['status'] == 'fehlend' for i in ('tf_container', 'tf_axes', 'tf_current')) and INPUTS['tf_container']['missing_list_ids'] == ['F02']


def test_paths_marked_missing_are_absent_and_present_ones_are_hashed():
    for entry in REPORT['inputs']:
        for source in entry['diss_sources']:
            if source['role'] == 'fehlt_am_verbraucherpfad':
                assert source['exists'] is False, source
            else:
                assert source['exists'] is True and re.fullmatch(r'[0-9a-f]{64}', source['sha256']), source


@pytest.mark.parametrize('line,rel,exists', [(161, 'my_iznik/data/in/core/core_information.csv', False), (221, 'my_gedi2/data/in/age_depth/rho_depth_given_age_higher_resol.rdata', False), (288, 'my_van/data/in/age_depth/rho_depth_given_age.rdata', False), (112, 'my_prespa/data/in/core/core_data_counts.csv', True)])
def test_other_slots_are_listed_apart(line, rel, exists):
    row = next(o for o in REPORT['other_slot_paths_not_affecting_kinneret'] if o['line'] == line)
    assert (row['relative_path'], row['exists']) == (rel, exists)


def lines_of(kind):
    return [(row['file'], row['line']) for row in REPORT['rng_chain'] if row['kind'] == kind]


def test_the_last_reset_before_the_stop_is_seed_2023():
    assert [row['line'] for row in REPORT['rng_chain'] if row['kind'] == 'seed' and row['file'] == P][-2:] == [1896, 1898]
    assert lines_of('stop') == [(P, 2057)] and REPORT['rng_chain'][-1]['file'] == C and REPORT['rng_chain'][-1]['line'] == 133


def test_the_start_values_are_drawn_before_seed_2023():
    assert {1709, 1715, 1716, 1723, 1724, 1731, 1734, 1762, 1763} <= {line for file, line in lines_of('draw') if file == P and 1354 < line < 1896}


def test_draws_follow_the_last_reset():
    assert {1899, 1957, 1958, 1979, 1980} <= {line for file, line in lines_of('draw') if file == P}


def test_no_generator_is_chosen_and_seed_lines_stay_listed():
    facts = REPORT['rng_facts']
    assert all(lines == [] for lines in facts['rngkind_or_rngversion_uncommented'].values())
    assert facts['set_seed_uncommented'] == {M: [1055], P: [1297, 1301, 1332, 1341, 1354, 1896, 1898], C: [], F: [], A: [], K: []}
    assert facts['main_line_1055_is_after_the_station_loop'] is True and facts['station_loop_lines'] == [651, 684] and CONDITIONS['rng_start']['rngkind_or_rngversion_calls_in_active_sources'] == 0


def test_the_eleven_bundles_stay_open_and_keep_their_order():
    spec = json.loads(SPEC.read_text(encoding='utf-8'))
    assert [(b['id'], b['phase1_status']) for b in REPORT['phase1_bundles']] == [(q['id'], q['unresolved_status']) for q in spec['requirements']] and len(REPORT['phase1_bundles']) == 11
    assert all(b['closed'] is False for b in REPORT['phase1_bundles']) and REPORT['bundles_closed'] == 0 and REPORT['selected_variant'] is None


def test_every_bundle_has_inputs_or_is_marked_as_output_side():
    for bundle in REPORT['phase1_bundles']:
        assert bundle['inputs'] or 'Ausgabeseite' in bundle['note_de'], bundle
    assert all(set(b['inputs']) <= set(INPUTS) for b in REPORT['phase1_bundles'])


def test_the_frozen_phase1_contract_is_the_one_bound():
    assert hashlib.sha256(SPEC.read_bytes()).hexdigest() == REPORT['phase1_spec_sha256']


def test_the_phase1_checker_still_reports_open_requirements():
    done = subprocess.run([sys.executable, 'tools/check_kinneret_phase1_spec.py'], cwd=ROOT, capture_output=True, stdin=subprocess.DEVNULL, timeout=120)
    assert done.returncode == 2


def test_nothing_was_drawn_run_or_released():
    assert REPORT['draws_run'] == 0 and REPORT['sampler_run'] is False and REPORT['r_scripts_run'] is False and REPORT['models_trained'] is False and REPORT['historical_reproduction_ready'] is False


@pytest.mark.parametrize('marker', ['tnetz', 'OneDrive', 'C:\\\\', '/automount/agh/Projects/s6tinetz/dissertation/palaeo_sfb/cmip'])
def test_no_private_root_is_stored(marker):
    assert marker not in (PACKET / 'input_contract.json').read_text(encoding='utf-8')


def test_line_references_match_the_originals_when_present():
    config = ROOT / '.local/my_reconst_paths.json'
    if not config.exists():
        pytest.skip('private source configuration not present')
    base = Path(json.loads(config.read_text(encoding='utf-8-sig'))['original_data_root'])
    if not (base / M).is_file():
        pytest.skip('diss is not mounted')
    lines = {}
    for rel, expected in REPORT['sources_sha256'].items():
        raw = (base / rel).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected
        lines[rel] = raw.decode('utf-8', 'replace').split('\n')
    lines['src/mcmc_prep_v1.2.R'] = (base / 'src/mcmc_prep_v1.2.R').read_bytes().decode('utf-8', 'replace').split('\n')
    refs = [ref for group in (REPORT['inputs'], REPORT['conditions']) for entry in group for ref in entry['code_reads']] + REPORT['settings_literal'] + REPORT['rng_chain']
    assert len(refs) > 200
    for ref in refs:
        assert lines[ref['file']][ref['line'] - 1].strip().startswith(ref['starts_with']), ref
