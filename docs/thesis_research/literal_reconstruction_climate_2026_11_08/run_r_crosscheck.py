"""R 3.5.0 + matrixStats 0.56.0 against the module; mcmc_functions_v1.6.R 1188-1199 verbatim on artificial inputs.

The original is read only after its SHA-256 matches the frozen binding. Doubles travel as C99 hex, NA and NaN stay
distinct; every value is compared bit for bit. Usage: run_r_crosscheck.py <Rscript.exe> <library folder with matrixStats>
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
_spec = importlib.util.spec_from_file_location('climate_check', OUT / 'check_reconstruction_climate.py')
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)

SOURCE = ('src/mcmc_functions_v1.6.R', 'b887144370e328fa016d1834ceebcd55622816d4b36838d33a26197110ab453c')
LOCAL = ROOT / '.local/reconstruction_climate_2026_11_08'


def r_num(v):
    if v is None:
        return 'NA_real_'
    if isinstance(v, float) and math.isnan(v):
        return 'NaN'
    return '0' if v == 0 else float(v).hex()


def r_vec(values):
    return 'as.double(c(' + ', '.join(r_num(v) for v in values) + '))'


def flat(rows):
    return [v for row in rows for v in row]


def from_r(t):
    return t if t in ('NA', 'NaN', 'Inf', '-Inf', '-0', '0') else float.fromhex(t)


def token(v, kind):
    if kind != 'number':
        return kind
    return '0' if v == 0 else v


def ulps(a, b):
    if isinstance(a, str) or isinstance(b, str):
        return 0 if a == b else None
    ia, ib = (struct.unpack('<q', struct.pack('<d', v))[0] for v in (a, b))
    return abs(ia - ib)


def r_case(name, c):
    head = (f'cases[["{name}"]] <- list(x1range = {r_vec(c["x1range"])}, x2range = {r_vec(c["x2range"])}, '
            f'prior_tmp = {r_num(c["prior_tmp"])}, prior_pann = {r_num(c["prior_pann"])}, recent = {"TRUE" if c["recent"] else "FALSE"}')
    if not c['recent']:
        return head + ')'
    return head + (f', core = {r_vec(flat(c["core"]))}, n_row = {len(c["core"])}L, n_taxa = {len(c["tf_x1"])}L, '
                   f'tf1 = {r_vec(flat(c["tf_x1"]))}, tf2 = {r_vec(flat(c["tf_x2"]))})')


def fixture_case(c, fixture):
    prior = fixture['prior']
    return {'core': check.matrix(c['core']), 'tf_x1': check.matrix(c['tf_x1']), 'tf_x2': check.matrix(c['tf_x2']),
            'x1range': [float(v) for v in fixture['x1range']], 'x2range': [float(v) for v in fixture['x2range']],
            'recent': c['recent'], 'prior_tmp': check.reference.parse(prior['mean_tmp']), 'prior_pann': check.reference.parse(prior['mean_pann'])}


def main():
    rscript, library = sys.argv[1], sys.argv[2]
    cfg = json.loads((ROOT / '.local/my_reconst_paths.json').read_text(encoding='utf-8-sig'))
    path = Path(cfg['original_data_root']) / SOURCE[0]
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE[1]:
        raise ValueError('mcmc_functions_v1.6.R differs from the frozen binding')
    fixture = json.loads((OUT / 'synthetic_fixture.json').read_text(encoding='utf-8'))
    inputs = {name: fixture_case(c, fixture) for name, c in fixture['cases'].items()}
    stress = {f'stress_{i}': c for i, c in enumerate(check.stress_cases(fixture['stress']))}
    lines = [f'source_file <- "{path.as_posix()}"', 'cases <- list()'] + [r_case(n, c) for n, c in {**inputs, **stress}.items()]
    LOCAL.mkdir(parents=True, exist_ok=True)
    (LOCAL / 'input.R').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    subprocess.run([rscript, '--vanilla', str(OUT / 'r_crosscheck.R'), library, str(LOCAL / 'input.R'), str(LOCAL / 'output.tsv')],
                   check=True, stdin=subprocess.DEVNULL)
    r_out, info = {}, None
    for line in (LOCAL / 'output.tsv').read_text(encoding='utf-8').splitlines():
        parts = line.split('\t')
        if parts[0] == 'info':
            info = dict(zip(('R', 'matrixStats', 'weightedMedian_namespace', 'rng_kind'), parts[1:]))
        else:
            r_out.setdefault(parts[0], {})[parts[1]] = [from_r(t) for t in parts[2].split()]
    rows, every = {}, []

    def compare(name, c):
        got = check.run_stress(c)
        row = {}
        for key, value, kind in (('tann', got.tann, got.tann_kind), ('pann', got.pann, got.pann_kind)):
            a, b = token(value, kind), r_out[name][key][0]
            row[key] = {'module': a if isinstance(a, str) else a.hex(), 'R': b if isinstance(b, str) else b.hex(), 'ulps': ulps(a, b)}
            every.append(row[key]['ulps'])
        if c['recent']:
            for key, weights in (('tann_weights', got.tann_weights), ('pann_weights', got.pann_weights)):
                mine = ['NaN' if math.isnan(v) else ('0' if v == 0 else v) for v in weights]
                theirs = r_out[name][key]
                # R's NA and NaN in a product both count as missing here; the median above already separates NA from a number
                d = [ulps(a if a != 'NaN' else 'x', 'x' if b in ('NA', 'NaN') else b) for a, b in zip(mine, theirs)]
                row[key] = {'values': len(d), 'bit_equal': sum(u == 0 for u in d), 'max_ulps': max(u for u in d if u is not None),
                            'mismatch': sum(u is None for u in d)}
                every.extend(d)
        return row

    case_rows = {name: compare(name, c) for name, c in inputs.items()}
    stress_rows = [compare(name, c) for name, c in stress.items()]
    report = {'kind': 'R_cross_check_reconstruction_climate_on_synthetic_fixture', 'runtime': info,
              'source': SOURCE[0] + ':1188-1199', 'source_sha256': SOURCE[1], 'values_compared': len(every),
              'bit_equal': sum(1 for u in every if u == 0), 'max_ulps': max(u for u in every if u is not None),
              'special_mismatch': sum(u is None for u in every), 'cases': case_rows,
              'stress': {'count': len(stress_rows), 'R_tann': [r['tann']['R'] for r in stress_rows], 'R_pann': [r['pann']['R'] for r in stress_rows]},
              'lazy_arguments_not_forced': True, 'historical_reproduction_ready': False}
    (OUT / 'r_crosscheck_result.json').write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(info)
    print(report['values_compared'], 'values,', report['bit_equal'], 'bit-equal, max ulps', report['max_ulps'], ', mismatches', report['special_mismatch'])


if __name__ == '__main__':
    main()
