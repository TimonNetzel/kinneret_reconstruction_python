"""Bit-exact cross check of six modules against their literal original lines in R 3.5.0, on artificial inputs only.

Originals are read only and must match the SHA-256 of the frozen bindings. Doubles travel as C99 hex, NA/NaN/Inf and the
sign of zero stay distinct. Every value is compared bit for bit; differences are counted in ulps and reported, not
hidden. Usage: run_six_crosscheck.py <Rscript.exe> <library folder with matrixStats>
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import historical_age_depth as ad  # noqa: E402
import historical_age_selection as sel  # noqa: E402
import historical_cdf_discrepancy as cd  # noqa: E402
import historical_final_pdf as fp  # noqa: E402
import historical_taxon_pruning as tp  # noqa: E402
import historical_tf_preparation as tf  # noqa: E402

DOCS = ROOT / 'docs/thesis_research'
LOCAL = ROOT / '.local/r_crosscheck_six_modules_2026_11_08'
SOURCES = {'prepare': ('src/mcmc_prepare.R', 'b897e84f8f2d72399de729592066341585974d521bf8164b185f20377ed5cc7e'),
           'functions': ('src/mcmc_functions_v1.6.R', 'b887144370e328fa016d1834ceebcd55622816d4b36838d33a26197110ab453c'),
           'ages': ('src/check_ages.R', None)}
AGES_BINDING = DOCS / 'literal_age_depth_2026_11_02/source_binding.json'
LINES = {'age_depth': 'src/check_ages.R:177-179,230-232; src/mcmc_prepare.R:11-20,83,91-94',
         'selection': 'src/mcmc_prepare.R:435-437,440-442,1462-1467,1522; src/mcmc_functions_v1.6.R:461-462,470',
         'final_pdf': 'src/mcmc_functions_v1.6.R:375,411-419,422-424,434-436,439-443,450-451,454-455,458-462,543-547,'
                      '712-715,756-760',
         'cdf': 'src/mcmc_functions_v1.6.R:369-371,485-490', 'tf': 'src/mcmc_prepare.R:1581-1586,1613-1618',
         'pruning': 'src/mcmc_prepare.R:51-80'}


def val(text):
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return float(text)
    if isinstance(text, str) and text.startswith(('0x', '-0x')):
        return float.fromhex(text)
    return float(Fraction(text))


def matrix(rows):
    return [[val(v) for v in row] for row in rows]


def r_num(v):
    return '0' if v == 0 else float(v).hex()


def r_vec(values):
    return 'c(' + ', '.join(r_num(v) for v in values) + ')' if len(values) else 'numeric(0)'


def flat(rows):
    return [v for row in rows for v in row]


def r_bool(b):
    return 'TRUE' if b else 'FALSE'


def token(v):
    v = float(v)
    if math.isnan(v):
        return 'NaN'
    if math.isinf(v):
        return 'Inf' if v > 0 else '-Inf'
    if v == 0:
        return '-0' if math.copysign(1.0, v) < 0 else '0'
    return v


def from_r(t):
    return t if t in ('NA', 'NaN', 'Inf', '-Inf', '-0', '0') else float.fromhex(t)


def ulps(a, b):
    if isinstance(a, str) or isinstance(b, str):
        return 0 if a == b else None
    ia, ib = (struct.unpack('<q', struct.pack('<d', v))[0] for v in (a, b))
    return abs(ia - ib)


def as_list(x):
    return [token(v) for v in np.asarray(x, dtype=np.float64).ravel()]


# --- cases: fixtures of the earlier packets first, then the stress set ---------------------------------------------

def load(rel):
    return json.loads((DOCS / rel).read_text(encoding='utf-8'))


def cases():
    out = {k: {} for k in LINES}
    stress = json.loads((OUT / 'stress_fixture.json').read_text(encoding='utf-8'))
    f = load('literal_age_depth_2026_11_02/synthetic_fixture.json')
    out['age_depth']['fixture'] = dict(rho=matrix(f['rho_age_by_depth_grid']), age=[val(v) for v in f['ages']],
                                       depth=[val(v) for v in f['depths']], pollen=matrix(f['all_available_depth_by_taxon']))
    f = load('literal_age_selection_2026_11_02/synthetic_fixture.json')
    for merge in (True, False):
        out['selection'][f'fixture_merge_{merge}'] = dict(age=[val(v) for v in f['age']], taxa_merge=merge,
                                                          coefficients=matrix(f['coefficients']),
                                                          tf=matrix(f['transfer_functions']))
    f = load('literal_final_pdf_2026_11_02/synthetic_fixture.json')
    out['final_pdf']['fixture'] = dict(available=matrix(f['all_available']), weights=matrix(f['weights']),
                                       tf=matrix(f['transfer_functions']), energy=[val(v) for v in f['energy']],
                                       gibbs_ids=[3, 1])
    f = load('literal_cdf_discrepancy_2026_11_02/synthetic_fixture.json')
    out['cdf']['fixture_placeholder'] = dict(density=matrix(f['density_age_by_cell']), reference=None, placeholder=True)
    out['cdf']['fixture_explicit'] = dict(density=matrix(f['density_age_by_cell']),
                                          reference=matrix(f['explicit_reference_cdf']), placeholder=False)
    f = load('literal_tf_preparation_2026_11_02/synthetic_fixture.json')
    for name, c in f['cases'].items():
        out['tf'][f'fixture_{name}'] = dict(raw=matrix(c['raw']), keep=c['keep'])
    f = load('literal_taxon_pruning_2026_11_02/synthetic_fixture.json')
    for name, c in f['cases'].items():
        out['pruning'][f'fixture_{name}'] = dict(pollen=matrix(c['pollen']), check=c['null_ratio_check'],
                                                 metro=c['metro_hastings'], relict=c['relict_override'])
    for i, c in enumerate(stress['age_depth']):
        out['age_depth'][f'stress_{i}'] = dict(rho=matrix(c['rho']), age=[val(v) for v in c['age']],
                                               depth=[val(v) for v in c['depth']], pollen=matrix(c['pollen']))
    for i, c in enumerate(stress['selection']):
        out['selection'][f'stress_{i}'] = dict(age=[val(v) for v in c['age']], taxa_merge=c['taxa_merge'],
                                               coefficients=matrix(c['coefficients']), tf=matrix(c['tf']))
    for i, c in enumerate(stress['final_pdf']):
        out['final_pdf'][f'stress_{i}'] = dict(available=matrix(c['available']), weights=matrix(c['weights']),
                                               tf=matrix(c['tf']), energy=[val(v) for v in c['energy']],
                                               gibbs_ids=c['gibbs_ids'])
    for i, c in enumerate(stress['cdf']):
        out['cdf'][f'stress_{i}'] = dict(density=matrix(c['density']), reference=matrix(c['reference']),
                                         placeholder=c['placeholder'])
    for i, c in enumerate(stress['tf']):
        out['tf'][f'stress_{i}'] = dict(raw=matrix(c['raw']), keep=c['keep'])
    for i, c in enumerate(stress['pruning']):
        out['pruning'][f'stress_{i}'] = dict(pollen=matrix(c['pollen']), check=c['check'], metro=c['metro'], relict=c['relict'])
    return out


def r_input(all_cases, paths):
    lines = ['sources <- list(' + ', '.join(f'{k} = "{v}"' for k, v in paths.items()) + ')',
             'cases <- list(age_depth = list(), selection = list(), final_pdf = list(), cdf = list(), tf = list(), pruning = list())']
    for n, c in all_cases['age_depth'].items():
        lines.append(f'cases$age_depth[["{n}"]] <- list(rho = {r_vec(flat(c["rho"]))}, n_age = {len(c["rho"])}L, '
                     f'age = {r_vec(c["age"])}, depth = {r_vec(c["depth"])}, pollen = {r_vec(flat(c["pollen"]))})')
    for n, c in all_cases['selection'].items():
        lines.append(f'cases$selection[["{n}"]] <- list(age = {r_vec(c["age"])}, taxa_merge = {r_bool(c["taxa_merge"])}, '
                     f'coefficients = {r_vec(flat(c["coefficients"]))}, n_row = {len(c["coefficients"])}L, '
                     f'tf = {r_vec(flat(c["tf"]))}, n_tf = {len(c["tf"])}L)')
    for n, c in all_cases['final_pdf'].items():
        lines.append(f'cases$final_pdf[["{n}"]] <- list(available = {r_vec(flat(c["available"]))}, n_row = {len(c["available"])}L, '
                     f'n_taxa = {len(c["available"][0])}L, weights = {r_vec(flat(c["weights"]))}, n_draw = {len(c["weights"])}L, '
                     f'tf = {r_vec(flat(c["tf"]))}, energy = {r_vec(c["energy"])}, gibbs_ids = c({", ".join(f"{i}L" for i in c["gibbs_ids"])}))')
    for n, c in all_cases['cdf'].items():
        ref = r_vec(flat(c['reference'])) if c['reference'] is not None else 'NULL'
        lines.append(f'cases$cdf[["{n}"]] <- list(density = {r_vec(flat(c["density"]))}, n_row = {len(c["density"])}L, reference = {ref}, '
                     f'placeholder = {r_bool(c["placeholder"])}, inherited_dims = {len(c["density"][0])}L, current_dims = {len(c["density"][0])}L)')
    for n, c in all_cases['tf'].items():
        lines.append(f'cases$tf[["{n}"]] <- list(raw = {r_vec(flat(c["raw"]))}, n_row = {len(c["raw"])}L, '
                     f'keep = c({", ".join(r_bool(k) for k in c["keep"])}))')
    for n, c in all_cases['pruning'].items():
        lines.append(f'cases$pruning[["{n}"]] <- list(pollen = {r_vec(flat(c["pollen"]))}, n_row = {len(c["pollen"])}L, '
                     f'check = {r_bool(c["check"])}, metro = {r_bool(c["metro"])}, relict = {r_bool(c["relict"])})')
    return '\n'.join(lines) + '\n'


# --- module side ------------------------------------------------------------------------------------------------

def module_age_depth(c):
    rho, age = ad.select_list_age_grid([np.array(c['rho']) * 2, c['rho']], [np.array(c['age']) + 1, c['age']], 2)
    shifted = ad.shift_duplicate_ages_once(age)
    weights = ad.depth_kernel_at_samples(rho, c['depth'])
    pollen = np.array(c['pollen'], dtype=np.float64)
    zero_replaced = pollen.copy()
    zero_replaced[zero_replaced == 0] = 1e-6
    return {'picked_rho': as_list(rho), 'picked_age': as_list(age), 'shifted_age': as_list(shifted),
            'site_age_untouched': as_list(age), 'weights': as_list(weights), 'pollen_zero_replaced': as_list(zero_replaced),
            'pollen_at_ages': as_list(ad.pollen_at_ages(weights, c['pollen']))}


def module_selection(c):
    grids, masks = sel.period_grids(c['age']), sel.period_membership(c['age'])
    trace = sel.preserved_selection_trace(c['age'], taxa_merge=c['taxa_merge'])
    out = {}
    for key in ('hol', 'lg', 'eem'):
        out[f'grid_{key}'] = as_list(grids[key])
        out[f'mask_{key}'] = as_list(masks[key])
    out['initial_mask'] = as_list(trace.initial_reconst_ages)
    out['effective_mask'] = as_list(trace.effective_reconst_ages)
    if trace.initial_reconst_ages.sum() >= 2:
        out['product_initial'] = as_list(sel.rw_selected_age_product(c['coefficients'], c['tf'], trace.initial_reconst_ages))
    effective = as_list(sel.rw_selected_age_product(c['coefficients'], c['tf'], trace.effective_reconst_ages))
    out.update(product_effective=effective, product_effective_pann=effective, product_effective_mean=effective)
    return out


def module_final_pdf(c):
    out = {'scaled': as_list(fp.column_max_taxa(c['available'])), 'unique_ids': as_list(fp.unique_energy_ids(c['energy']))}
    draws = []
    for i, weights in enumerate(c['weights'], 1):
        coefficients = fp.rw_coefficients(c['available'], weights)
        density = fp.apply_taxon_densities(coefficients, c['tf'])
        out[f'coefficients_{i}'], out[f'density_{i}'] = as_list(coefficients), as_list(density)
        draws.append(density)
    summed = fp.sum_pdf_branch(np.array(draws), c['energy']).density
    out['pdf_sum_depth'] = out['pdf_sum_age'] = as_list(summed)
    if len(c['gibbs_ids']) >= 2:
        coefficients = fp.gibbs_coefficients(c['available'], c['gibbs_ids'])
        out['gibbs_coefficients'] = as_list(coefficients)
        out['gibbs_density'] = as_list(fp.apply_taxon_densities(coefficients, np.array(c['tf'])[[i - 1 for i in c['gibbs_ids']]]))
    return out


def module_cdf(c):
    rows, width = len(c['density']), len(c['density'][0])
    first = 0.0 if c['placeholder'] else c['reference']
    pair = cd.prepare_reference_pair([first] if c['placeholder'] else first, None if c['placeholder'] else c['reference'],
                                     rho_rows=rows, inherited_dims=width)
    got = cd.cdf_discrepancy(c['density'], pair.first, dims=width)
    return {'reference_first': as_list(pair.first), 'reference_second': as_list(pair.second),
            'cumulative': as_list(got.cumulative), 'score': as_list(got.score)}


def module_tf(c):
    normalized = tf.normalize_tf_rows(c['raw'])
    return {'normalized': as_list(normalized), 'masked': as_list(tf.mask_tf_cells(normalized, c['keep']))}


def module_pruning(c):
    got = tp.prune_taxa(c['pollen'], null_ratio_check=c['check'], metro_hastings=c['metro'], relict_override=c['relict'])
    out = {'skipped': [token(float(got.skip_site))], 'metro_hastings': [token(float(got.metro_hastings))]}
    if got.skip_site or not got.executed:
        pollen = np.array(c['pollen'], dtype=np.float64)
    else:
        pollen = got.pollen
    out['pollen'] = as_list(pollen)
    out['pollen_dim'] = as_list(pollen.shape)
    if got.executed:
        out['null_ratio'] = as_list(got.null_ratio)
        out['removed'] = as_list([i + 1 for i in got.removed_taxa])
        if not got.skip_site:
            out['spectrum'] = as_list(got.sorting_available_spectrum)
    return out


MODULES = {'age_depth': module_age_depth, 'selection': module_selection, 'final_pdf': module_final_pdf, 'cdf': module_cdf,
           'tf': module_tf, 'pruning': module_pruning}


def main():
    rscript, library = sys.argv[1], sys.argv[2]
    cfg = json.loads((ROOT / '.local/my_reconst_paths.json').read_text(encoding='utf-8-sig'))
    base = Path(cfg['original_data_root'])
    paths, shas = {}, {}
    ages_binding = json.dumps(json.loads(AGES_BINDING.read_text(encoding='utf-8')))
    for key, (rel, sha) in SOURCES.items():
        digest = hashlib.sha256((base / rel).read_bytes()).hexdigest()
        if sha is None:
            if digest not in ages_binding:
                raise ValueError(f'{rel}: hash not found in the frozen age-depth binding')
        elif digest != sha:
            raise ValueError(f'{rel} differs from the frozen binding')
        paths[key], shas[rel] = (base / rel).as_posix(), digest
    all_cases = {module: {f'{module}.{name}': case for name, case in group.items()} for module, group in cases().items()}
    LOCAL.mkdir(parents=True, exist_ok=True)
    (LOCAL / 'input.R').write_text(r_input(all_cases, paths), encoding='utf-8', newline='\n')
    subprocess.run([rscript, '--vanilla', str(OUT / 'six_crosscheck.R'), library, str(LOCAL / 'input.R'), str(LOCAL / 'output.tsv')],
                   check=True, stdin=subprocess.DEVNULL)
    r_out, info = {}, None
    for line in (LOCAL / 'output.tsv').read_text(encoding='utf-8').splitlines():
        parts = line.split('\t')
        if parts[0] == 'info':
            info = dict(zip(('R', 'sizeof_longdouble', 'matrixStats', 'rowCumsums_namespace'), parts[1:]))
            continue
        r_out.setdefault(parts[0], {})[parts[1]] = [from_r(t) for t in parts[2].split()] if len(parts) > 2 and parts[2] else []
    report = {}
    total = bit = 0
    worst = 0
    mismatches = 0
    for module, group in all_cases.items():
        keys, diffs, fixture_values, fixture_equal = {}, [], 0, 0
        for name, case in group.items():
            mine = MODULES[module](case)
            theirs = r_out[name]
            if set(mine) != set(theirs):
                raise ValueError(f'{module}/{name}: keys differ {set(mine) ^ set(theirs)}')
            for key, values in mine.items():
                short = key.split('_')[0] if key.startswith(('coefficients_', 'density_')) and key[-1].isdigit() else key
                if len(values) != len(theirs[key]):
                    raise ValueError(f'{module}/{name}/{key}: lengths differ {len(values)} {len(theirs[key])}')
                slot = keys.setdefault(short, {'values': 0, 'bit_equal': 0, 'max_ulps': 0, 'special_mismatch': 0})
                for k, (a, b) in enumerate(zip(values, theirs[key])):
                    u = ulps(a, b)
                    slot['values'] += 1
                    total += 1
                    if '.fixture' in name:
                        fixture_values += 1
                    if u == 0:
                        slot['bit_equal'] += 1
                        bit += 1
                        if '.fixture' in name:
                            fixture_equal += 1
                    elif u is None:
                        slot['special_mismatch'] += 1
                        mismatches += 1
                        diffs.append({'case': name, 'key': key, 'position': k, 'module': a if isinstance(a, str) else a.hex(),
                                      'R': b if isinstance(b, str) else b.hex(), 'ulps': None})
                    else:
                        slot['max_ulps'] = max(slot['max_ulps'], u)
                        worst = max(worst, u)
                        if len(diffs) < 400:
                            diffs.append({'case': name, 'key': key, 'position': k, 'module': a.hex(), 'R': b.hex(), 'ulps': u})
        report[module] = {'original_lines': LINES[module], 'cases': len(group), 'keys': keys,
                          'values': sum(s['values'] for s in keys.values()),
                          'bit_equal': sum(s['bit_equal'] for s in keys.values()),
                          'max_ulps': max(s['max_ulps'] for s in keys.values()),
                          'fixture_values': fixture_values, 'fixture_bit_equal': fixture_equal,
                          'differences_first_400': diffs}
    result = {'kind': 'R_cross_check_six_modules_literal_lines_on_synthetic_inputs_not_case_reproduction',
              'runtime': info, 'sources_sha256': shas, 'random_state_used': False,
              'values_compared': total, 'bit_equal': bit, 'max_ulps': worst, 'special_mismatch': mismatches,
              'modules': report, 'historical_reproduction_ready': False}
    (OUT / 'six_crosscheck_result.json').write_text(json.dumps(result, indent=0, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    print(info)
    print(total, 'values,', bit, 'bit-equal, max ulps', worst, ', special mismatches', mismatches)
    for module, row in report.items():
        print(module, row['cases'], 'cases', row['values'], 'values', row['bit_equal'], 'equal, max', row['max_ulps'],
              'fixture', row['fixture_bit_equal'], '/', row['fixture_values'],
              {k: (s['values'] - s['bit_equal'], s['max_ulps']) for k, s in row['keys'].items() if s['values'] != s['bit_equal']})


if __name__ == '__main__':
    main()
