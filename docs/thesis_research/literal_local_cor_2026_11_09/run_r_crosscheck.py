"""R 3.5.0 against the module; cor (2355-2366) and the time-space block (1939-2018) verbatim on artificial inputs.

The original is read only after its SHA-256 matches. Doubles travel as C99 hex; NA and NaN stay distinct; every value is compared
bit for bit and the difference counted in ulp. Usage: run_r_crosscheck.py <Rscript.exe>
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
_spec = importlib.util.spec_from_file_location('cor_check', OUT / 'check_local_cor.py')
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)

SOURCE = ('src/mcmc_functions_v1.6.R', 'b887144370e328fa016d1834ceebcd55622816d4b36838d33a26197110ab453c')
LOCAL = ROOT / '.local/local_cor_2026_11_09'


def r_num(v):
    if v is None:
        return 'NA_real_'
    if isinstance(v, float) and math.isnan(v):
        return 'NaN'
    if v == 0:
        return '-0' if math.copysign(1.0, v) < 0 else '0'
    return float(v).hex()


def r_vec(values):
    return 'as.double(c(' + ', '.join(r_num(v) for v in values) + '))' if values else 'numeric(0)'


def r_lgl(values):
    return 'c(' + ', '.join('TRUE' if v else 'FALSE' for v in values) + ')'


def r_cor_case(name, x, y, na_rm):
    return f'cases[["{name}"]] <- list(x = {r_vec(x)}, y = {r_vec(y)}, na_rm = {"TRUE" if na_rm else "FALSE"})'


def r_reader(name, r):
    items = [f'tann = {"TRUE" if r["tann"] else "FALSE"}', f'multiple = {"TRUE" if r["multiple"] else "FALSE"}', 'hi_time = FALSE', 'spectral = FALSE',
             f'predictant_tann = {r_vec(check.vector(r["predictant_tann"]))}', f'predictant_pann = {r_vec(check.vector(r["predictant_pann"]))}',
             f'hi_ages_time = {r_lgl(r["hi_ages_time"])}']
    if r['multiple']:
        items += [f'ages_pollen = {r_lgl(r["ages_pollen"])}', f'predictors1 = {r_vec(check.vector(r["predictors1"]))}',
                  f'ages_iso = {r_lgl(r["ages_iso"])}', f'predictors2 = {r_vec(check.vector(r["predictors2"]))}']
    else:
        marine = r['sources_marine']
        items.append(f'sources = data.frame(giesecke = 0, core = 0, speleothem = 0, marine_ice_core = {marine})')
        if marine == 1:
            items += [f'ages_iso1 = {r_lgl(r["ages_iso1"])}', f'ages_iso2 = {r_lgl(r["ages_iso2"])}', f'predictors = list(iso_time = {r_vec(check.vector(r["iso_time"]))})']
        else:
            items += [f'ages_pollen1 = {r_lgl(r["ages_pollen1"])}', f'ages_pollen2 = {r_lgl(r["ages_pollen2"])}', f'predictors = list(pollen = {r_vec(check.vector(r["pollen"]))})']
    return f'readers[["{name}"]] <- list(' + ', '.join(items) + ')'


def from_r(t):
    return t if t in ('NA', 'NaN', 'Inf', '-Inf', '-0', '0', 'ERROR') else float.fromhex(t)


def ulps(a, b):
    if isinstance(a, str) or isinstance(b, str):
        return 0 if a == b else None
    ia, ib = (struct.unpack('<q', struct.pack('<d', v))[0] for v in (a, b))
    return abs(ia - ib)


def module_token(result):
    if result.kind != 'number':
        return result.kind
    if result.value == 0:
        return '-0' if math.copysign(1.0, result.value) < 0 else '0'
    return result.value


def main():
    rscript = str(Path(sys.argv[1]).resolve())
    cfg = json.loads((ROOT / '.local/my_reconst_paths.json').read_text(encoding='utf-8-sig'))
    path = Path(cfg['original_data_root']) / SOURCE[0]
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE[1]:
        raise ValueError('mcmc_functions_v1.6.R differs from the frozen binding')
    fixture = json.loads((OUT / 'synthetic_fixture.json').read_text(encoding='utf-8'))
    named = {n: (check.vector(c['x']), check.vector(c['y']), c['na_rm']) for n, c in fixture['cases'].items()}
    stress = {f'stress_{i}': (c['x'], c['y'], c['na_rm']) for i, c in enumerate(check.stress_cases(fixture['stress']))}
    mismatch = {f'mismatch_{n}': ([float(v) for v in c['x']], [float(v) for v in c['y']], c['na_rm']) for n, c in fixture['length_mismatch'].items()}
    lines = [f'source_file <- "{path.as_posix()}"', 'cases <- list()', 'readers <- list()']
    lines += [r_cor_case(n, *v) for n, v in {**named, **stress, **mismatch}.items()]
    lines += [r_reader(n, r) for n, r in fixture['readers'].items()]
    LOCAL.mkdir(parents=True, exist_ok=True)
    (LOCAL / 'input.R').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    subprocess.run([rscript, '--vanilla', str(OUT / 'r_crosscheck.R'), str(LOCAL / 'input.R'), str(LOCAL / 'output.tsv')],
                   check=True, stdin=subprocess.DEVNULL)
    r_out, info = {}, None
    for line in (LOCAL / 'output.tsv').read_text(encoding='utf-8').splitlines():
        parts = line.split('\t')
        if parts[0] == 'info':
            info = dict(zip(('R', 'cor_environment', 'label', 'cor_found', 'rng_kind'), parts[1:]))
        else:
            r_out.setdefault(parts[0], {})[parts[1]] = parts[2]
    every, rows = [], {}

    def compare(name, x, y, na_rm):
        got = check.lc.local_cor(x, y, na_rm=na_rm)
        a, b = module_token(got), from_r(r_out[name]['cor'])
        u = ulps(a, b)
        every.append(u)
        return {'module': a if isinstance(a, str) else a.hex(), 'R': b if isinstance(b, str) else b.hex(), 'ulps': u, 'dim': r_out[name].get('dim')}

    rows = {n: compare(n, *v) for n, v in named.items()}
    stress_rows = [compare(n, *v) for n, v in stress.items()]
    readers = {}
    for name, r in fixture['readers'].items():
        got = check.run_reader(r)
        row = {}
        for key, value in (('post_pollen', got.post_pollen), ('post_iso', got.post_iso), ('post_iso_hi', got.post_iso_hi)):
            mine = '0' if value == 0 else value
            theirs = from_r(r_out[name][key])
            u = ulps(mine, theirs)
            every.append(u)
            row[key] = {'module': mine if isinstance(mine, str) else mine.hex(), 'R': theirs if isinstance(theirs, str) else theirs.hex(), 'ulps': u}
        row['post_pollen_dim_in_R'] = r_out[name]['post_pollen_dim']
        readers[name] = row
    length_rows = {}
    for name, (x, y, na_rm) in mismatch.items():
        try:
            check.lc.local_cor(x, y, na_rm=na_rm)
            module = 'accepted'
        except check.lc.UnsupportedSourceDomain:
            module = 'rejected'
        length_rows[name] = {'module': module, 'R': r_out[name]['cor']}
    constants = {key: r_out['constants'][key] for key in r_out['constants']}
    consts_ok = all(float.fromhex(constants[k]) == v for k, v in (('one_e_minus_10', 1e-10), ('squared', 1e-10 * 1e-10), ('ninety_five', 0.95), ('half', 0.5)))
    hist = {}
    for u in every:
        hist[str(u)] = hist.get(str(u), 0) + 1
    report = {'kind': 'R_cross_check_local_cor_on_synthetic_fixture', 'runtime': info, 'source': SOURCE[0] + ':2355-2366 and 1939-2018', 'source_sha256': SOURCE[1],
              'values_compared': len(every), 'bit_equal': sum(1 for u in every if u == 0), 'max_ulps': max(u for u in every if u is not None),
              'ulp_histogram': hist, 'special_mismatch': sum(u is None for u in every), 'cases': rows,
              'stress': {'count': len(stress_rows), 'not_bit_equal': [dict(r, index=i) for i, r in enumerate(stress_rows) if r['ulps'] != 0], 'R': [r['R'] for r in stress_rows]},
              'readers': readers, 'length_mismatch': length_rows, 'constants': constants, 'constants_match_python_literals': consts_ok, 'historical_reproduction_ready': False}
    (OUT / 'r_crosscheck_result.json').write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(info)
    print(report['values_compared'], 'values,', report['bit_equal'], 'bit-equal, histogram', hist, ', token mismatches', report['special_mismatch'], ', constants', consts_ok)
    print(length_rows)


if __name__ == '__main__':
    main()
