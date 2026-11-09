"""R 3.5.0 + matrixStats 0.56.0 against the module; mcmc_prepare.R lines verbatim for one artificial TF matrix.

mcmc_prepare.R is read only after its SHA-256 matches the frozen binding; the installed package comes from the CRAN
archive zip recorded in binary_package.json. Doubles travel as C99 hex, NA and NaN stay distinct. Every value is
compared bit for bit, differences are reported in ulps. Usage: run_r_crosscheck.py <Rscript.exe> <library folder>
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
sys.path.insert(0, str(ROOT / 'tools'))
_spec = importlib.util.spec_from_file_location('wm_check', OUT / 'check_weighted_median.py')
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)
wm = check.wm

SOURCE = ('src/mcmc_prepare.R', 'b897e84f8f2d72399de729592066341585974d521bf8164b185f20377ed5cc7e')
LOCAL = ROOT / '.local/weighted_median_2026_11_08'


def r_num(v):
    if v is None:
        return 'NA_real_'
    if math.isnan(v):
        return 'NaN'
    if math.isinf(v):
        return 'Inf' if v > 0 else '-Inf'
    return float(v).hex()


def r_vec(values):
    return 'as.double(c(' + ', '.join(map(r_num, values)) + '))' if values else 'numeric(0)'


def from_r(t):
    return t if t in ('NA', 'NaN', 'Inf', '-Inf') else ('-0' if t == '-0' else float.fromhex('0x0p+0' if t == '0' else t))


def mine(m):
    if m.kind != 'number':
        return m.kind
    if math.isinf(m.value):
        return 'Inf' if m.value > 0 else '-Inf'
    return '-0' if m.value == 0 and math.copysign(1, m.value) < 0 else m.value


def ulps(a, b):
    if isinstance(a, str) or isinstance(b, str):
        return 0 if a == b else None
    ia, ib = (struct.unpack('<q', struct.pack('<d', v))[0] for v in (a, b))
    return abs(ia - ib)


def text(v):
    return v if isinstance(v, str) else v.hex()


def main():
    rscript, library = sys.argv[1], sys.argv[2]
    cfg = json.loads((ROOT / '.local/my_reconst_paths.json').read_text(encoding='utf-8-sig'))
    path = Path(cfg['original_data_root']) / SOURCE[0]
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE[1]:
        raise ValueError('mcmc_prepare.R differs from the frozen binding')
    fixture = json.loads((OUT / 'synthetic_fixture.json').read_text(encoding='utf-8'))
    lines = [f'source_file <- "{path.as_posix()}"', 'cases <- list()', 'stress <- list()']
    cases = {n: ([check.input_value(v) for v in c['x']], [check.input_value(v) for v in c['w']])
             for n, c in fixture['cases'].items()}
    for name, (x, w) in cases.items():
        lines.append(f'cases[["{name}"]] <- list(x = {r_vec(x)}, w = {r_vec(w)})')
    stress = check.stress_cases(fixture['stress'])
    for i, (x, w) in enumerate(stress, 1):
        lines.append(f'stress[[{i}]] <- list(x = {r_vec(x)}, w = {r_vec(w)})')
    tf = fixture['tf_case']
    rows = lambda m: [[check.input_value(str(v)) for v in r] for r in m]
    def matrix(m):
        return f'matrix(c({", ".join(r_num(v) for r in m for v in r)}), nrow = {len(m)}, byrow = TRUE)'
    lines.append(f'tf_case <- list(x1range = {r_vec([float(v) for v in tf["x1range"]])}, '
                 f'x2range = {r_vec([float(v) for v in tf["x2range"]])}, tf_x1 = {matrix(rows(tf["tf_x1"]))}, '
                 f'tf_x2 = {matrix(rows(tf["tf_x2"]))})')
    LOCAL.mkdir(parents=True, exist_ok=True)
    (LOCAL / 'input.R').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    subprocess.run([rscript, '--vanilla', str(OUT / 'r_crosscheck.R'), library, str(LOCAL / 'input.R'),
                    str(LOCAL / 'output.tsv')], check=True, stdin=subprocess.DEVNULL)
    r_cases, r_stress, r_tf, info = {}, {}, {}, None
    for line in (LOCAL / 'output.tsv').read_text(encoding='utf-8').splitlines():
        parts = line.split('\t')
        if parts[0] == 'info':
            info = dict(zip(('R', 'matrixStats', 'namespace', 'built', 'lookup_namespace', 'formals'), parts[1:]))
        elif parts[0] == 'case':
            r_cases[parts[1]] = from_r(parts[2])
        elif parts[0] == 'stress':
            r_stress[int(parts[1])] = from_r(parts[2])
        elif parts[1] == 'which':
            r_tf['which'] = [int(v) for v in parts[2].split()]
        else:
            r_tf[parts[1]] = [from_r(v) for v in parts[2].split()]

    def compare(got, r):
        return {'module': text(got), 'R': text(r), 'ulps': ulps(got, r)}

    case_rows = {name: compare(mine(check.run_case(c)), r_cases[name]) for name, c in fixture['cases'].items()}
    stress_rows = [compare(mine(wm.weighted_median(x, w)), r_stress[i]) for i, (x, w) in enumerate(stress, 1)]
    grid = lambda key: [float(v) for v in tf[key]]
    x1 = wm.tf_row_medians(grid('x1range'), rows(tf['tf_x1']))
    x2 = wm.tf_row_medians(grid('x2range'), rows(tf['tf_x2']))
    tf_rows = {'x1': [compare(mine(m), r) for m, r in zip(x1, r_tf['x1'])],
               'x2': [compare(mine(m), r) for m, r in zip(x2, r_tf['x2'])]}
    which_equal = list(wm.which_tf_missing(x1)) == r_tf['which']
    every = [row['ulps'] for row in [*case_rows.values(), *stress_rows, *tf_rows['x1'], *tf_rows['x2']]]
    report = {
        'kind': 'R_cross_check_weightedMedian_on_synthetic_fixture',
        'runtime': info, 'source': SOURCE[0] + ':1628-1640,1648-1660,1664', 'source_sha256': SOURCE[1],
        'values_compared': len(every), 'bit_equal': sum(1 for u in every if u == 0),
        'max_ulps': max((u for u in every if u is not None), default=None),
        'special_mismatch': sum(u is None for u in every),
        'which_tf_missing_R': r_tf['which'], 'which_tf_missing_equal': which_equal,
        'cases': case_rows, 'tf_case': tf_rows,
        'stress': {'count': len(stress_rows), 'bit_equal': sum(1 for r in stress_rows if r['ulps'] == 0),
                   'R_values': [r['R'] for r in stress_rows]},
        'note': 'Only the binary built for R 3.5 (matrixStats 0.56.0) was run; the other 12 window versions are '
                'compared as source text only. Plain IEEE double arithmetic, x86_64.',
        'historical_reproduction_ready': False,
    }
    (OUT / 'r_crosscheck_result.json').write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(info)
    print(report['values_compared'], 'values,', report['bit_equal'], 'bit-equal, max ulps', report['max_ulps'],
          ', special mismatches', report['special_mismatch'], ', which equal', which_equal, r_tf['which'])
    for name, row in case_rows.items():
        if row['ulps'] != 0:
            print('DIFF', name, row)
    for i, row in enumerate(stress_rows):
        if row['ulps'] != 0:
            print('DIFF stress', i, row)


if __name__ == '__main__':
    main()
