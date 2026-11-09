"""R 3.5.0 against the module; lines 1817-1820, 1825, 1829-1857, 1939-1943 verbatim on artificial matrices.

The original is read only after its SHA-256 matches. Doubles travel as C99 hex; NA and NaN stay distinct; every value is compared bit for
bit and the difference counted in ulp. Usage: run_r_crosscheck.py <Rscript.exe>
"""
import hashlib
import importlib.util
import json
import math
import struct
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
_spec = importlib.util.spec_from_file_location('reconst_check', OUT / 'check_reconst_predictant.py')
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)

SOURCE = ('src/mcmc_functions_v1.6.R', 'b887144370e328fa016d1834ceebcd55622816d4b36838d33a26197110ab453c')
LOCAL = ROOT / '.local/reconst_predictant_2026_11_09'


def r_num(v):
    if v is None or v is check.rp.NA:
        return 'NA_real_'
    if isinstance(v, float) and math.isnan(v):
        return 'NaN'
    if v in (math.inf, -math.inf):
        return 'Inf' if v > 0 else '-Inf'
    if v == 0:
        return '-0' if math.copysign(1.0, v) < 0 else '0'
    return float(v).hex()


def r_matrix(rows):
    flat = ', '.join(r_num(v) for row in rows for v in row)
    return f'matrix(as.double(c({flat})), nrow = {len(rows)}, ncol = {len(rows[0])}, byrow = TRUE)'


def r_lgl(values):
    return 'c(' + ', '.join('NA' if v is None else 'TRUE' if v else 'FALSE' for v in values) + ')' if values else 'logical(0)'


def r_vec(values):
    return 'as.double(c(' + ', '.join(r_num(v) for v in values) + '))' if values else 'numeric(0)'


def r_case(name, core, mask, tf, extra=''):
    return f'[["{name}"]] <- list(core = {r_matrix(core)}, mask = {r_lgl(mask)}, tf = {r_matrix(tf)}{extra})'


def named_extra(c):
    return (f', recent = {"TRUE" if c["recent"] else "FALSE"}, tann = {"TRUE" if c["tann"] else "FALSE"}'
            f', prior.recent = list(mean.tmp = {r_num(check.value(c["prior"][0]))}, mean.pann = {r_num(check.value(c["prior"][1]))})')


def chain_extra(ch):
    v = check.vector
    items = [f'tann = {"TRUE" if ch["tann"] else "FALSE"}', f'multiple = {"TRUE" if ch["multiple"] else "FALSE"}', 'hi_time = FALSE', 'spectral = FALSE', f'hi_ages_time = {r_lgl(ch["hi_ages_time"])}']
    if ch['multiple']:
        items += [f'ages_pollen = {r_lgl(ch["ages_pollen"])}', f'predictors1 = {r_vec(v(ch["predictors1"]))}', f'ages_iso = {r_lgl(ch["ages_iso"])}', f'predictors2 = {r_vec(v(ch["predictors2"]))}']
    else:
        marine = ch['sources_marine']
        items.append(f'sources = data.frame(giesecke = 0, core = 0, speleothem = 0, marine_ice_core = {marine})')
        if marine == 1:
            items += [f'ages_iso1 = {r_lgl(ch["ages_iso1"])}', f'ages_iso2 = {r_lgl(ch["ages_iso2"])}', f'predictors = list(iso_time = {r_vec(v(ch["iso_time"]))})']
        else:
            items += [f'ages_pollen1 = {r_lgl(ch["ages_pollen1"])}', f'ages_pollen2 = {r_lgl(ch["ages_pollen2"])}', f'predictors = list(pollen = {r_vec(v(ch["pollen"]))})']
    return ', ' + ', '.join(items)


def from_r(text):
    return text if text in ('NA', 'NaN', 'Inf', '-Inf', '-0', '0') else float.fromhex(text)


def split(text):
    return [from_r(t) for t in text.split()] if text else []


def ulps(a, b):
    if isinstance(a, str) or isinstance(b, str):
        return 0 if a == b else None
    ia, ib = (struct.unpack('<q', struct.pack('<d', v))[0] for v in (a, b))
    return abs(ia - ib)


def module_values(tokens):
    return [from_r(t) for t in tokens]


def compare(mine, theirs, every):
    """Length mismatches count as a single unmatched value."""
    if len(mine) != len(theirs):
        every.append(None)
        return [None]
    out = []
    for a, b in zip(mine, theirs):
        u = ulps(a, b)
        every.append(u)
        out.append(u)
    return out



def main():
    rscript = str(Path(sys.argv[1]).resolve())
    cfg = json.loads((ROOT / '.local/my_reconst_paths.json').read_text(encoding='utf-8-sig'))
    path = Path(cfg['original_data_root']) / SOURCE[0]
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE[1]:
        raise ValueError('mcmc_functions_v1.6.R differs from the frozen binding')
    fixture = json.loads((OUT / 'synthetic_fixture.json').read_text(encoding='utf-8'))
    named = {n: (check.matrix(c['core']), c['mask'], check.matrix(c['tf']), c) for n, c in fixture['cases'].items()}
    stress = check.stress_cases(fixture['stress'])
    lines = [f'source_file <- "{path.as_posix()}"', 'cases <- list()', 'stress <- list()', 'chains <- list()']
    lines += ['cases' + r_case(n, core, mask, tf, named_extra(c)) for n, (core, mask, tf, c) in named.items()]
    lines += ['stress' + r_case(f'stress_{i}', c['core'], c['mask'], c['tf']) for i, c in enumerate(stress)]
    lines += ['chains' + r_case(n, check.matrix(ch['core']), ch['mask'], check.matrix(ch['tf']), chain_extra(ch)) for n, ch in fixture['chains'].items()]
    LOCAL.mkdir(parents=True, exist_ok=True)
    (LOCAL / 'input.R').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    subprocess.run([rscript, '--vanilla', str(OUT / 'r_crosscheck.R'), str(LOCAL / 'input.R'), str(LOCAL / 'output.tsv')], check=True, stdin=subprocess.DEVNULL)
    r_out, info = {}, None
    for line in (LOCAL / 'output.tsv').read_text(encoding='utf-8').splitlines():
        parts = line.split('\t')
        if parts[0] == 'info':
            info = dict(zip(('R', 'matprod', 'long_double', 'rng_kind', 'blas'), parts[1:]))
        else:
            r_out.setdefault(parts[0], {})[parts[1]] = parts[2] if len(parts) > 2 else ''
    every = []
    rows = {}
    for name, (core, mask, tf, case) in named.items():
        got = check.run_case(case)
        r = r_out[name]
        row = {'R_stop': (r.get('stop') or '').split(':')[0] or None, 'module_stop': got.get('stop')}
        if got.get('stop') == '1818' or (r.get('stop') or '').startswith('1818'):
            row['stops_agree'] = (got.get('stop') == '1818') == (r.get('stop') or '').startswith('1818')
            rows[name] = row
            continue
        row['selected_R'] = r['selected']
        row['selected_dim_module'] = got['selected_dim']
        row['reconst_dim_R'] = r['reconst_dim']
        row['reconst_dimnames_R'] = r['reconst_dimnames']
        row['tann_ulps'] = compare(module_values(got['tann']), split(r['predictant_tann']), every)
        row['pann_ulps'] = compare(module_values(got['pann']), split(r['predictant_pann']), every)
        row['tann_R'] = r['predictant_tann']
        row['pann_R'] = r['predictant_pann']
        row['stops_agree'] = (got.get('stop') == '1825') == (r.get('stop') or '').startswith('1825')
        if 'youngest' in got:
            row['youngest_ulps'] = compare(module_values(got['youngest']), split(r['youngest']), every)
            for key in ('post_tann', 'post_pann'):
                row[key + '_ulps'] = compare([from_r(got[key])], split(r[key]), every)
            row['current_ulps'] = compare(module_values(got['current']), split(r['predictant_current']), every)
            row['current_is_tann_R'] = r['current_is_tann']
            row['current_name_module'] = got['current_name']
        rows[name] = row
    stress_rows = []
    for i, c in enumerate(stress):
        name = f'stress_{i}'
        got, r = check.run_stress(c), r_out[name]
        if got.get('stop') == '1818' or (r.get('stop') or '').startswith('1818'):
            stress_rows.append({'stops_agree': (got.get('stop') == '1818') == (r.get('stop') or '').startswith('1818')})
            continue
        tann, pann = compare(module_values(got['tann']), split(r['predictant_tann']), every), compare(module_values(got['pann']), split(r['predictant_pann']), every)
        stress_rows.append({'path': got['path'], 'selected_R': r['selected'], 'reconst_dim_R': r['reconst_dim'], 'tann_ulps': tann, 'pann_ulps': pann, 'R': r['predictant_tann'] + ' | ' + r['predictant_pann']})
    chains = {}
    for name, ch in fixture['chains'].items():
        got, r = check.run_chain(ch), r_out[name]
        row = {'branch_module': got['branch'], 'predictant_read': got['predictant_read'], 'R_stop': (r.get('stop') or '').split(':')[0] or None}
        for key in ('post_pollen', 'post_iso', 'post_iso_hi'):
            mine, theirs = from_r(got[key]), from_r(r[key])
            u = ulps(mine, theirs)
            every.append(u)
            row[key] = {'module': got[key], 'R': r[key], 'ulps': u}
        chains[name] = row
    hist = {}
    for u in every:
        hist[str(u)] = hist.get(str(u), 0) + 1
    report = {'kind': 'R_cross_check_reconst_predictant_on_synthetic_fixture', 'runtime': info, 'source': SOURCE[0] + ':1817-1820, 1825, 1829-1857, 1939-1943 and 1939-2018 with 2355-2366', 'source_sha256': SOURCE[1],
              'values_compared': len(every), 'bit_equal': sum(1 for u in every if u == 0), 'max_ulps': max(u for u in every if u is not None), 'ulp_histogram': hist,
              'special_mismatch': sum(u is None for u in every), 'stop_disagreements': sum(1 for r in [*rows.values(), *stress_rows] if r.get('stops_agree') is False),
              'cases': rows, 'stress': {'count': len(stress_rows), 'not_bit_equal': [dict(r, index=i) for i, r in enumerate(stress_rows) if any(u != 0 for u in r.get('tann_ulps', []) + r.get('pann_ulps', []))],
                                        'paths': {p: sum(1 for r in stress_rows if r.get('path') == p) for p in ('dgemm', 'dgemv_t', 'simple_matprod', 'zero_extent')}, 'R': [r.get('R') for r in stress_rows]},
              'chains': chains, 'historical_reproduction_ready': False}
    (OUT / 'r_crosscheck_result.json').write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(info)
    print(report['values_compared'], 'values,', report['bit_equal'], 'bit-equal, histogram', hist, ', token mismatches', report['special_mismatch'], ', stop disagreements', report['stop_disagreements'])
    print('paths', report['stress']['paths'], 'stress not bit equal', len(report['stress']['not_bit_equal']))


if __name__ == '__main__':
    main()
