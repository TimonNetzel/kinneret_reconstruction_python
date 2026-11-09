"""Bind the local cor and establish which of its call sites the Kinneret run (add.sites <- T, slot 3) can reach.

Reads only. Hashes must match. Every .R file under the diss folder is searched for a redefinition of cor. Comments and strings are
stripped before searching, so a commented call is not counted. Writes source_binding.json, source_excerpt.json, determinism_audit.json.
"""
import csv
import hashlib
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASIS = '437f6f4baeed4a086a2d5a80c7abc9e420a353de'
SOURCES = {'main_v1.2.R': '82486f99bafff08af2b51475ed225501d26cec1a0322a286b97a248ebd2f0520',
           'src/mcmc_functions_v1.6.R': 'b887144370e328fa016d1834ceebcd55622816d4b36838d33a26197110ab453c',
           'src/mcmc_prepare.R': 'b897e84f8f2d72399de729592066341585974d521bf8164b185f20377ed5cc7e',
           'src/mcmc_conduction.R': '47c59c508e15150f7d6fda775a4f6bf6c14f0148e10cefd0014fa1ce6cfe5a9d'}
ALSO_LOADED = ['src/loess_functions.R', 'src/check_ages.R', 'src/prior_climate.R', 'src/mcmc_trimming_posterior_trafo.R']
F, P, M, C = 'src/mcmc_functions_v1.6.R', 'src/mcmc_prepare.R', 'main_v1.2.R', 'src/mcmc_conduction.R'
LITERAL = {
    M: [(43, 'source("src/loess_functions.R")'), (73, 'add.sites <- T'), (154, 'core.data[[3]]$human_impact <- c(-60,6500)'),
        (644, 'source("src/mcmc_functions_v1.6.R")'), (651, 'for(station_id in stations ){'), (669, 'giesecke_prespa <- F'),
        (672, 'case_studies <- F'), (675, 'source("src/mcmc_prepare.R")'), (676, 'source("src/mcmc_conduction.R")'),
        (677, 'source("src/mcmc_trimming_posterior_trafo.R")'), (684, '}')],
    C: [(146, 'posterior_temp <- reconstruction(posterior_temp=posterior_temp,tf.x1.current[posterior_temp$z,],tf.x2.current[posterior_temp$z,],tf_x12_mean_current[posterior_temp$z,],predictors,tann,recent,multiple,spectral,stad_interstad)')],
    F: [(855, 'reconstruction <- function(taxa.appear.core,which.taxa,age,tf.x1.current,tf.x2.current,tf_x12_mean_current,predictors,tann,recent,multiple,spectral,stad_interstad){'),
        (1781, 'reconstruction <- function(posterior_temp,tf.x1.current,tf.x2.current,tf_sample,predictors,tann,recent,multiple,spectral,stad_interstad){'),
        (1939, 'if(tann){'), (1948, 'if(multiple){'),
        (1964, 'posterior_temp$post_pollen <- cor(predictant_current[ages_pollen][hi_ages_time], predictors1[hi_ages_time],na.rm=TRUE)'),
        (1965, 'if(posterior_temp$post_pollen < 0 | is.na(posterior_temp$post_pollen)) posterior_temp$post_pollen <- 1e-10'),
        (1966, 'posterior_temp$post_iso <- cor( predictant_tann[ages_iso],predictors2,na.rm=TRUE )'),
        (1969, 'posterior_temp$post_pollen <- posterior_temp$post_pollen^2'),
        (1979, 'if(hi_time){'), (1987, 'posterior_temp$post_iso_hi <- 0.95'), (1992, 'if(sources[,4] == 1){'),
        (1993, 'posterior_temp$post_iso <- cor(predictant_tann[ages_iso1],predictors$iso_time[ages_iso2],na.rm=TRUE)'),
        (1999, 'posterior_temp$post_pollen <- cor(predictant_current[ages_pollen1][hi_ages_time],predictors$pollen[ages_pollen2][hi_ages_time],na.rm=TRUE)'),
        (2355, 'cor <- function(x, y,na.rm=FALSE){'), (2364, 'correlation <- ( (x-mean_x) %*% (y-mean_y) /(N-1)) / ( sqrt(sum((x-mean_x)^2)/(N-1)) * sqrt(sum((y-mean_y)^2)/(N-1)))'),
        (2366, '}')],
    P: [(555, 'sources <- data.frame(giesecke = 0, core = 0, speleothem = 0, marine_ice_core = 0)'),
        (591, 'if( inside.giesecke & !add.sites | giesecke_prespa ){'),
        (636, '} else if( !inside.giesecke &  add.sites | sitename %in% c("Lake Prespa", "Lake Ohrid", "Lake Van") ){'),
        (654, 'pollen.age <- site.age'), (661, 'sources$core <- 1'),
        (662, 'if( var(predictors$pollen[pollen.age %in% site.age], na.rm=T)== 0 ) sources$core <- 0'),
        (713, 'hi_time <- FALSE'), (716, 'hi_ages_time <- !inside.range(pollen.age,human_impact) # !inside.range(pollen.age,c(-60,2300))'),
        (718, 'if(add.sites) hi_time <- FALSE'),
        (881, 'if(length(human_impact) != 0 & !any(sitename == c("Lake Kinneret","En Gedi","Birkat Ram")) ){'),
        (1072, 'sources$marine_ice_core <- 1'), (1096, 'if(FALSE ){ # sources$marine_ice_core == 1 & sources$giesecke == 1 | sources$core == 1'),
        (1192, 'multiple <- spectral <- FALSE'), (1200, 'if( all(which_sources == c(2,3,4)) )  multiple <- TRUE'),
        (1209, 'if( !inside.giesecke & add.sites & lat <= 36 & !any(sitename == c("Dead Sea","Birkat Ram","Lake Kinneret")) ){'),
        (1249, 'tann <- FALSE'), (1250, '#     multiple <- FALSE'), (1788, 'predictors1 <- predictors$pollen[pollen.age %in% site.age]'),
        (1790, 'ages_pollen <- site.age %in% pollen.age')],
}
COR_CALLS = {F: [136, 137, 138, 965, 967, 976, 981, 992, 1964, 1966, 1982, 1993, 1999, 2009, 2024, 2032], P: [1084]}
SITE_NOT_BOUND = {'source': 'in/core_data_all_andrea_taxa_additional_sites.rdata, slot 3, read in R 3.5.0', 'older_than_v1_2': True,
                  'sitename': 'Kinneret', 'lat': 32.79, 'lon': 35.53, 'country': 'Israel'}
READERS = [
    {'line': 1084, 'file': 'mcmc_prepare.R', 'guard': 'country != "Israel"', 'kinneret': 'not reached under the older record (Israel); cor23 has no live reader anyway'},
    {'line': '136-138', 'file': 'mcmc_functions_v1.6.R', 'guard': 'correlation_decomposition has no live caller', 'kinneret': 'not reached'},
    {'line': '965-992', 'file': 'mcmc_functions_v1.6.R', 'guard': 'first reconstruction definition, replaced at 1781', 'kinneret': 'not reached'},
    {'line': '1964, 1966', 'guard': 'multiple', 'kinneret': 'reached when sources$core is 1 (see multiple_table)'},
    {'line': 1982, 'guard': 'multiple & hi_time', 'kinneret': 'not reached: add.sites <- T sets hi_time FALSE (mcmc_prepare.R:718)'},
    {'line': 1993, 'guard': '!multiple & sources[,4]==1', 'kinneret': 'reached only if sources$core is 0'},
    {'line': 1999, 'guard': '!multiple & sources[,4]!=1', 'kinneret': 'not reached: sources[,4] stays 1 for Kinneret'},
    {'line': 2009, 'guard': '!multiple & hi_time', 'kinneret': 'not reached: hi_time FALSE'},
    {'line': 2024, 'guard': 'spectral', 'kinneret': 'depends on sources$speleothem; needs wt_power, not part of this module'},
    {'line': 2032, 'guard': 'spectral & hi_spec', 'kinneret': 'not reached: mcmc_prepare.R:881 excludes Kinneret from hi_spec'},
]


def multiple_table():
    """Lines 1192-1200 for Kinneret. sources[1] is 0 (591 is false) and sources[4] is 1, so only core and speleothem vary."""
    rows = []
    for core in (0, 1):
        for speleothem in (0, 1):
            which = [i for i, s in zip((1, 2, 3, 4), (0, core, speleothem, 1)) if s == 1]
            multiple = which in ([1, 4], [2, 4], [1, 3, 4], [2, 3, 4])
            rows.append({'sources_core': core, 'sources_speleothem': speleothem, 'which_sources': which, 'multiple': multiple,
                         'spectral': 3 in which, 'time_reader': '1964+1966' if multiple else '1993'})
    return rows


def giesecke_space(base):
    path = base / 'in/epd_giesecke_2016/quant_files/epdQuant_Abi.csv'
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    lat, lon = [float(r['Lat']) for r in rows], [float(r['Lon']) for r in rows]
    distances = [abs(v - SITE_NOT_BOUND['lat']) for v in lat]
    nearest = lat[distances.index(min(distances))]
    row_lons = [o for a, o in zip(lat, lon) if a == nearest]
    inside = min(row_lons) - 1 < SITE_NOT_BOUND['lon'] < max(row_lons) + 1
    return {'file': 'in/epd_giesecke_2016/quant_files/epdQuant_Abi.csv', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'grid_points': len(rows),
            'lat_range': [min(lat), max(lat)], 'nearest_grid_lat': nearest, 'lon_range_at_that_lat': [min(row_lons), max(row_lons)],
            'inside_giesecke_space': inside,
            'holds_for_any_site_below_lat_35': 'nearest row 34.75 has one point at 24.75, so every longitude above 25.75 is outside'}


def main():
    spec = importlib.util.spec_from_file_location('ws_bind', OUT.parent / 'literal_weights_stage_2026_11_03/bind_source.py')
    binder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(binder)
    cfg = json.loads((ROOT / '.local/my_reconst_paths.json').read_text(encoding='utf-8-sig'))
    base = Path(cfg['original_data_root'])
    diss = base.parents[1]
    lines, code = {}, {}
    for rel in [*SOURCES, *ALSO_LOADED]:
        raw = (base / rel).read_bytes()
        if rel in SOURCES and hashlib.sha256(raw).hexdigest() != SOURCES[rel]:
            raise ValueError(f'{rel} differs from the frozen binding')
        lines[rel] = raw.decode('utf-8-sig', errors='replace').splitlines()
        code[rel] = binder._code(lines[rel])
    for rel, expected in LITERAL.items():
        for n, text in expected:
            if lines[rel][n - 1].strip() != text:
                raise ValueError(f'{rel}:{n} no longer reads {text!r}')
    pattern = binder._name('cor') + r'\s*\('
    calls = {}
    for rel, stripped in code.items():
        found = [n for n, line in enumerate(stripped, 1) if re.search(pattern, line)]
        if found:
            calls[rel] = found
    if calls != COR_CALLS:
        raise ValueError(f'cor call sites differ: {calls}')
    depth = binder._depths(code[F])
    if depth[2354] != 0 or depth[854] != 0 or depth[1780] != 0:
        raise ValueError('cor or a reconstruction definition is not at top level')
    decomposition = [(rel, n) for rel, stripped in code.items() for n, line in enumerate(stripped, 1)
                     if re.search(binder._name('correlation_decomposition') + r'\s*\(', line)]
    if decomposition:
        raise ValueError(f'correlation_decomposition has a call: {decomposition}')
    strays = []
    binder._depths(code[M], strays)
    assign = re.compile(r'(?<![\w.$])(?:multiple|spectral|hi_time|hi_spec)(?![\w.])\s*(<-|=[^=])')
    assignments = {rel: [(n, lines[rel][n - 1].strip()[:90]) for n, line in enumerate(stripped, 1) if assign.search(line) and 'function' not in line]
                   for rel, stripped in code.items()}
    assignments = {rel: found for rel, found in assignments.items() if found}
    if set(assignments) != {P}:
        raise ValueError(f'unexpected assignment files: {list(assignments)}')
    numbers = [n for n, _ in assignments[P]]
    expected = [713, 715, 718, 879, 882, 1192, 1194, 1195, 1196, 1199, 1200, 1203, 1211, 1221, 1231, 1233, 1235, 1242]
    if numbers != expected:
        raise ValueError(f'assignment lines differ: {numbers}')
    sources_writes = [n for n, line in enumerate(code[P], 1) if re.search(r'(?<![\w.$])sources(\$marine_ice_core|\[4\])\s*<-|<-\s*sources\$marine_ice_core\s*<-', line)]
    other_writes = [(rel, n) for rel, stripped in code.items() if rel != P for n, line in enumerate(stripped, 1) if re.search(r'(?<![\w.$])sources[$\[].*<-', line)]
    if other_writes or sources_writes != [1072, 1123, 1143, 1145, 1147, 1150, 1153, 1154, 1155, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1212]:
        raise ValueError(f'writes to sources[4] differ: {sources_writes}, {other_writes}')
    redefinitions, count = [], 0
    for path in sorted(diss.rglob('*')):
        if not path.is_file() or path.suffix.lower() != '.r':
            continue
        count += 1
        text = path.read_text(encoding='utf-8', errors='replace')
        if re.search(binder._name('cor') + r'\s*(<-|=)\s*function', text):
            redefinitions.append(path.relative_to(diss).as_posix())
    space = giesecke_space(base)
    if space['inside_giesecke_space']:
        raise ValueError('Kinneret would lie inside the Giesecke row')
    table = multiple_table()
    span = lines[F][2354:2366]
    (OUT / 'source_excerpt.json').write_text(json.dumps({'status': 'read_only_spans', 'spans': [
        {'source': 'original:src/mcmc_functions_v1.6.R', 'first_line': 2355, 'last_line': 2366, 'role': 'implemented', 'text': '\n'.join(span)},
        {'source': 'original:src/mcmc_functions_v1.6.R', 'first_line': 1939, 'last_line': 2018, 'role': 'implemented_readers', 'text': '\n'.join(lines[F][1938:2018])}]},
        indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    (OUT / 'source_binding.json').write_text(json.dumps({
        'status': 'bounded_historical_source_fragment_not_case_reproduction', 'basis_commit': BASIS, 'sources_sha256': SOURCES,
        'implemented_spans': [[F, 2355, 2366], [F, 1939, 2018]], 'r_files_searched': count, 'files_defining_cor': redefinitions,
        'cor_definition_in_bound_file': {'line': 2355, 'top_level': True, 'sourced_by': 'main_v1.2.R:644', 'masks': 'stats::cor for code evaluated in the global environment'},
        'live_cor_call_sites': calls, 'readers': READERS, 'correlation_decomposition_callers': decomposition,
        'assignments_of_multiple_spectral_hi_time_hi_spec': assignments, 'writes_to_sources_4': sources_writes,
        'writes_to_sources_4_reached_for_kinneret': {'1072': 'unconditional, sets 1', 'others': 'if(FALSE) block, case_studies F, or excluded by sitename at 1209'},
        'multiple_table_kinneret': table, 'multiple_equals_sources_core_for_kinneret': all(r['multiple'] == (r['sources_core'] == 1) for r in table),
        'kinneret_flags': {'add.sites': True, 'case_studies': False, 'giesecke_prespa': False, 'tann': False, 'hi_time': False, 'hi_spec': False,
                           'human_impact': [-60, 6500], 'hi_ages_time': '!inside.range(pollen.age, human_impact), mcmc_prepare.R:716; kept although hi_time is FALSE'},
        'site_coordinates_used_for_space_test': SITE_NOT_BOUND, 'giesecke_space_test': space,
        'open_conditions': ['sources$core needs var(AP reference) != 0 (mcmc_prepare.R:662); not evaluated without the final v1.2 inputs',
                            'sources$speleothem depends on d18o.age and does not change multiple',
                            'length and alignment of hi_ages_time against the selected vectors need the final ages'],
        'main_v1_2_stray_closing_braces': strays, 'main_v1_2_note': 'the file does not parse as a whole; the loop 651-684 is the binding used here',
        'packages_needed_by_cor': [], 'base_functions_used': ['is.na', 'length', 'sum', 'sqrt', '%*%', '^', '/', '-', '['], 'globals_made_explicit': [],
        'portable_excerpt': 'source_excerpt.json', 'portable_excerpt_sha256': hashlib.sha256((OUT / 'source_excerpt.json').read_bytes()).hexdigest(),
        'actual_original_inputs_used_for_numerics': False, 'run_binding': None, 'historical_reproduction_ready': False,
    }, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    (OUT / 'determinism_audit.json').write_text(json.dumps({
        'kind': 'determinism_audit_from_source_text_and_R_cross_check',
        'question': 'Is the local cor deterministic, which call sites does Kinneret reach, and which branch applies?',
        'candidates': [{'id': 'local_cor', 'lines': 'mcmc_functions_v1.6.R:2355-2366', 'random_state': False, 'package_semantics_needed': False,
                        'reached_by_bound_run': True, 'decision': 'implemented with explicit vectors'},
                       {'id': 'reader_multiple', 'lines': '1964-1970', 'random_state': False, 'reached_by_bound_run': 'when sources$core is 1', 'decision': 'implemented'},
                       {'id': 'reader_single_iso', 'lines': '1993-1996', 'random_state': False, 'reached_by_bound_run': 'only when sources$core is 0', 'decision': 'implemented, marked conditional'},
                       {'id': 'reader_single_pollen', 'lines': '1999-2002', 'random_state': False, 'reached_by_bound_run': False, 'decision': 'implemented, marked unreachable for Kinneret'},
                       {'id': 'reader_spectral', 'lines': '2024-2026', 'random_state': False, 'package_semantics_needed': True, 'decision': 'not implemented here; needs wt_power'},
                       {'id': 'input_vectors', 'lines': '1812-1820', 'random_state': True, 'decision': 'given explicitly; no draws'}],
        'selected_for_implementation': 'local_cor and time-space readers', 'final_variant_selected': False, 'R_execution_for_binding': 'older site record only',
    }, indent=2) + '\n', encoding='utf-8', newline='\n')
    print('files searched', count, 'defining cor', redefinitions)
    print('stray braces in main', strays, 'space', space['inside_giesecke_space'])


if __name__ == '__main__':
    main()
