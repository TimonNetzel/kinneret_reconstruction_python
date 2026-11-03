"""Bind the site slots of the add.sites branch of main_v1.2.R and the Kinneret switches in mcmc_prepare.R.

Reads only; both originals must match the SHA-256 of the earlier frozen binding. Checks the literal lines that set
add.sites, load the missing additional-sites container, fill slot 3 from the my_kinneret pollen file, name slots 7-9,
exclude slot 4 and station 469, and every top-level sitename == "Lake Kinneret" switch. Writes site_binding.json.
"""
import hashlib
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCES = {'main_v1.2.R': '82486f99bafff08af2b51475ed225501d26cec1a0322a286b97a248ebd2f0520',
           'src/mcmc_prepare.R': 'b897e84f8f2d72399de729592066341585974d521bf8164b185f20377ed5cc7e'}
LITERAL = [
    ('main_v1.2.R', 73, 'add.sites <- T'),
    ('main_v1.2.R', 107, 'load(paste0("in/core_data_additional_sites_v1.2.rdata"))'),
    ('main_v1.2.R', 112, 'prespa_core <- read.csv(file="../../reconstructions/my_reconst/my_prespa/data/in/core/'
                         'core_data_counts.csv",header=T)'),
    ('main_v1.2.R', 143, 'kinneret_core <- read.csv(file="../../reconstructions/my_reconst/my_kinneret/data/in/core/'
                         'pollen_counts_epd_taxanomy.csv",header=T)'),
    ('main_v1.2.R', 144, 'core.data[[3]]$qualitativ_data$qualitativ_reconstruction <- '
                         'kinneret_core$tree_percentage[!is.na(kinneret_core$tree_percentage)]'),
    ('main_v1.2.R', 212, 'core.data[[7]]$sitename <- "Dead Sea"'),
    ('main_v1.2.R', 247, 'core.data[[8]]$sitename <- "Birkat Ram"'),
    ('main_v1.2.R', 275, 'core.data[[9]]$sitename <- "Lake Van"'),
    ('main_v1.2.R', 607, 'if(par.iter == 1) stations <- c(1:100) # c(26,79,80,81,83,84,86,90,91,92,94,95,98) #c(1:100)'),
    ('main_v1.2.R', 615, 'if(add.sites ) stations <- c(1:length(core.data))[-4]'),
    ('main_v1.2.R', 651, 'for(station_id in stations ){'),
    ('main_v1.2.R', 654, 'if( is.null(core.data[[station_id]]) | station_id == 469 ) next()'),
    ('src/mcmc_prepare.R', 790, 'if(sitename == "Lake Kinneret") which.site <- 105'),
    ('src/mcmc_prepare.R', 881, 'if(length(human_impact) != 0 & !any(sitename == c("Lake Kinneret","En Gedi",'
                                '"Birkat Ram")) ){'),
    ('src/mcmc_prepare.R', 1145, 'if(sitename == "Lake Kinneret")  sources$speleothem <- sources$marine_ice_core <- 0'),
    ('src/mcmc_prepare.R', 1208, 'tann <- TRUE'),
    ('src/mcmc_prepare.R', 1209, 'if( !inside.giesecke & add.sites & lat <= 36 & !any(sitename == c("Dead Sea",'
                                 '"Birkat Ram","Lake Kinneret")) ){'),
    ('src/mcmc_prepare.R', 1248, 'if(any(sitename == c("Lake Kinneret", ""))){'),
    ('src/mcmc_prepare.R', 1249, 'tann <- FALSE'),
]
# slot -> data path in main_v1.2.R (my_reconst folder of the read.csv / load that fills it)
SLOT_SOURCES = {1: 'my_prespa', 2: 'my_gedi', 3: 'my_kinneret', 4: 'my_birkat_ram', 5: 'my_iznik', 6: 'my_ohrid',
                7: 'my_gedi2', 8: 'my_birkat_ram', 9: 'my_van'}


def main():
    cfg = json.loads((ROOT / '.local/my_reconst_paths.json').read_text(encoding='utf-8-sig'))
    base = Path(cfg['original_data_root'])
    lines = {}
    for rel, sha in SOURCES.items():
        raw = (base / rel).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError(f'{rel} differs from the frozen binding')
        lines[rel] = raw.decode('utf-8-sig').splitlines()
    for rel, n, text in LITERAL:
        if lines[rel][n - 1].strip() != text:
            raise ValueError(f'{rel}:{n} no longer reads {text!r}')
    main_lines = lines['main_v1.2.R']
    slot_lines = {}
    for n, line in enumerate(main_lines[106:298], 107):
        code = line.split('#', 1)[0]
        for slot in re.findall(r'core\.data\[\[(\d)\]\]', code):
            slot_lines.setdefault(int(slot), []).append(n)
    kinneret = [n for n, line in enumerate(lines['src/mcmc_prepare.R'], 1)
                if 'Lake Kinneret' in line.split('#', 1)[0]]
    if kinneret != [790, 881, 1145, 1209, 1248]:
        raise ValueError(f'Kinneret switches differ: {kinneret}')
    # brace depth with the stripping rules of the frozen weights-stage binder (imported, not changed)
    spec = importlib.util.spec_from_file_location('ws_bind', OUT.parent / 'literal_weights_stage_2026_11_03/bind_source.py')
    binder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(binder)
    depth = binder._depths(binder._code(lines['src/mcmc_prepare.R']))
    switch_depth = {str(n): depth[n - 1] for n in (790, 881, 1145, 1208, 1209, 1248, 1249)}
    if switch_depth != {'790': 1, '881': 1, '1145': 1, '1208': 0, '1209': 0, '1248': 0, '1249': 1}:
        raise ValueError(f'brace depths of the switches differ: {switch_depth}')
    result = {
        'kinneret_switch_brace_depth': switch_depth,
        'depth_note': '790, 881 and 1145 lie inside enclosing blocks (their conditions are not resolved here); 1208, '
                      '1209 and 1248 are top level, so tann <- FALSE applies to every run with sitename '
                      '"Lake Kinneret"',
        'status': 'site_slot_binding_from_source_text_not_case_reproduction',
        'sources_sha256': SOURCES,
        'add_sites': 'main_v1.2.R:73 add.sites <- T is the only setting; the add.sites branch (107-298) loads the '
                     'missing in/core_data_additional_sites_v1.2.rdata and overwrites slot contents from my_reconst',
        'slot_sources': {str(k): v for k, v in SLOT_SOURCES.items()},
        'slot_lines_in_main': {str(k): v for k, v in sorted(slot_lines.items())},
        'kinneret_slot': 3,
        'kinneret_sitename_in_prepare': 'Lake Kinneret (sitename comes from core.data[[3]]$sitename in the missing '
                                        'container; the older container of the kinneret_binding packet names '
                                        'Kinneret in slot 3 as well)',
        'kinneret_switches_mcmc_prepare': {'790': 'which.site <- 105 (speleothem site)',
                                           '881': 'no human-impact handling for Kinneret',
                                           '1145': 'speleothem and marine/ice-core sources set to 0',
                                           '1209': 'Kinneret excluded from the Levante branch (tann <- FALSE, multiple <- FALSE, sources[4] <- 0 for other sites at lat <= 36)',
                                           '1248': 'tann <- FALSE for Lake Kinneret (after tann <- TRUE in 1208); meaning of tann in the run not interpreted here'},
        'stations_add_sites': 'c(1:length(core.data))[-4]: slot 4 (first Birkat Ram) skipped; the composite Birkat '
                              'Ram is slot 8; length(core.data) depends on the missing container (at least 9)',
        'epd_branch': 'with add.sites <- F the stations are EPD entities 1..n from in/core_data_all_giesecke_counts'
                      '.rdata (par.iter blocks of 100); the bound file runs the add.sites branch',
        'historical_reproduction_ready': False,
    }
    (OUT / 'site_binding.json').write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n', encoding='utf-8',
                                           newline='\n')
    print('slots', {k: len(v) for k, v in sorted(slot_lines.items())}, 'Kinneret switches', kinneret)


if __name__ == '__main__':
    main()
