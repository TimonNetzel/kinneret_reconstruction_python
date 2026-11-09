"""R 3.5.0 against the module; measures the deviation of the own normal functions, it does not hide it.

The C++ of lines 1485-1525 cannot be run here (no compiler, no Rcpp). R runs the verbatim R twin of the function (comment lines 1529-1552) and mcmc_conduction.R:228 and 230.
The grid ratios and the long double product must be bit-identical once R's own factors go in. Usage: run_r_crosscheck.py <Rscript.exe>
"""
import hashlib
import importlib.util
import json
import math
import platform
import struct
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
_spec = importlib.util.spec_from_file_location('ratios_check', OUT / 'check_ratios.py')
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)
rt = check.rt

SOURCES = {'src/mcmc_functions_v1.6.R': 'b887144370e328fa016d1834ceebcd55622816d4b36838d33a26197110ab453c', 'src/mcmc_conduction.R': '47c59c508e15150f7d6fda775a4f6bf6c14f0148e10cefd0014fa1ce6cfe5a9d'}
LOCAL = ROOT / '.local/ratios_tfs_2026_11_09'
BINS = ((0, 0, '0'), (1, 1, '1'), (2, 2, '2'), (3, 4, '3-4'), (5, 16, '5-16'), (17, 256, '17-256'), (257, 2 ** 70, '>256'))
TALLIES = ('dnorm', 'p_a', 'p_b', 'd_truncnorm', 'grid_ratio', 'density_ratio', 'transfer_ratio', 'prod_from_R_factors')


def r_num(v):
    if v == 0:
        return '-0' if math.copysign(1.0, v) < 0 else '0'
    return float(v).hex()


def r_vec(values):
    return 'as.double(c(' + ', '.join(r_num(v) for v in values) + '))'


def r_rows(rows):
    return f'matrix({r_vec([v for row in rows for v in row])}, nrow = {len(rows)}, ncol = {len(rows[0])}, byrow = TRUE)'


def r_case(c):
    a = check.arguments(c)
    fields = [f'num_taxa = {c["num_taxa"]}L'] + [f'{k} = {r_rows(a[k]) if k.endswith("dists") else r_vec(a[k])}' for k in check.ARGS]
    return 'list(' + ', '.join(fields) + ')'


def from_r(text):
    if text in ('NA', 'NaN'):
        return math.nan
    if text in ('Inf', '-Inf'):
        return math.inf if text == 'Inf' else -math.inf
    return -0.0 if text == '-0' else 0.0 if text == '0' else float.fromhex(text)


def split(text):
    return [from_r(t) for t in text.split()] if text else []


def ulps(a, b):
    ia, ib = (struct.unpack('<q', struct.pack('<d', v))[0] for v in (a, b))
    if a == 0 and b == 0:
        return 0
    ia = ia if ia >= 0 else -(ia & 0x7fffffffffffffff)
    ib = ib if ib >= 0 else -(ib & 0x7fffffffffffffff)
    return abs(ia - ib)


class Tally:
    def __init__(self):
        self.values, self.bit_equal, self.max_ulps, self.max_rel, self.special_mismatch, self.special = 0, 0, 0, 0.0, 0, 0
        self.hist = {label: 0 for _, _, label in BINS}

    def add(self, mine, theirs):
        if len(mine) != len(theirs):
            raise ValueError('different lengths')
        for a, b in zip(mine, theirs):
            self.values += 1
            if not (math.isfinite(a) and math.isfinite(b)):
                self.special += 1
                same = (math.isnan(a) and math.isnan(b)) or a == b
                self.special_mismatch += not same
                self.bit_equal += same
                continue
            u = ulps(a, b)
            self.bit_equal += u == 0
            self.max_ulps = max(self.max_ulps, u)
            for low, high, label in BINS:
                if low <= u <= high:
                    self.hist[label] += 1
                    break
            if b != 0:
                self.max_rel = max(self.max_rel, abs(a - b) / abs(b))

    def report(self):
        return {'values': self.values, 'bit_equal': self.bit_equal, 'max_ulps': self.max_ulps, 'ulp_histogram': self.hist, 'max_relative': self.max_rel, 'non_finite': self.special, 'non_finite_mismatch': self.special_mismatch}


def compare(result, r, tallies, indices_ok):
    flat = [p for row in result.parts for p in row]
    tallies['dnorm'].add([p.density for p in flat], split(r['dnorm']))
    tallies['p_a'].add([p.p_a for p in flat], split(r['p_a']))
    tallies['p_b'].add([p.p_b for p in flat], split(r['p_b']))
    tallies['d_truncnorm'].add([p.value for p in flat], split(r['d_truncnorm']))
    tallies['grid_ratio'].add([*result.tann_tf_ratio, *result.pann_tf_ratio], [*split(r['tann_tf_ratio']), *split(r['pann_tf_ratio'])])
    tallies['density_ratio'].add([*result.tann_dists_tf_ratio, *result.pann_dists_tf_ratio], [*split(r['tann_dists_tf_ratio']), *split(r['pann_dists_tf_ratio'])])
    tallies['transfer_ratio'].add([rt.transfer_ratio(result)], split(r['transfer_ratio']))
    factors = [*split(r['tann_tf_ratio']), *split(r['pann_tf_ratio']), *split(r['tann_dists_tf_ratio']), *split(r['pann_dists_tf_ratio'])]
    tallies['prod_from_R_factors'].add([rt.long_double_prod(factors)], split(r['transfer_ratio']))
    mine = [v for row in result.indices for v in row]
    indices_ok[0] &= mine == [int(t) for t in r['indices'].split()]


def main():
    rscript = str(Path(sys.argv[1]).resolve())
    cfg = json.loads((ROOT / '.local/my_reconst_paths.json').read_text(encoding='utf-8-sig'))
    root = Path(cfg['original_data_root'])
    for rel, expected in SOURCES.items():
        if hashlib.sha256((root / rel).read_bytes()).hexdigest() != expected:
            raise ValueError(f'{rel} differs from the frozen binding')
    fixture = json.loads((OUT / 'synthetic_fixture.json').read_text(encoding='utf-8'))
    lines = [f'source_file <- "{(root / "src/mcmc_functions_v1.6.R").as_posix()}"', f'conduction_file <- "{(root / "src/mcmc_conduction.R").as_posix()}"', 'cases <- list()', 'stress <- list()']
    lines += [f'cases[["{n}"]] <- {r_case(c)}' for n, c in fixture['cases'].items()]
    lines += [f'stress[["stress_{i}"]] <- {r_case(c)}' for i, c in enumerate(fixture['stress'])]
    LOCAL.mkdir(parents=True, exist_ok=True)
    (LOCAL / 'input.R').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    subprocess.run([rscript, '--vanilla', str(OUT / 'r_crosscheck.R'), str(LOCAL / 'input.R'), str(LOCAL / 'output.tsv')], check=True, stdin=subprocess.DEVNULL)
    r_out, info = {}, None
    for row in (LOCAL / 'output.tsv').read_text(encoding='utf-8').splitlines():
        parts = row.split('\t')
        if parts[0] == 'info':
            info = dict(zip(('R', 'pnorm_namespace', 'dnorm_namespace', 'long_double'), parts[1:]))
        else:
            r_out.setdefault(parts[0], {})[parts[1]] = parts[2] if len(parts) > 2 else ''
    groups = {g: {k: Tally() for k in TALLIES} for g in ('named', 'stress')}
    indices_ok = [True]
    named = {}
    for name, c in fixture['cases'].items():
        result = check.run(c)
        single = {k: Tally() for k in TALLIES}
        compare(result, r_out[name], single, indices_ok)
        compare(result, r_out[name], groups['named'], indices_ok)
        named[name] = {'kind': check.summary(result)['kind'], 'transfer_ratio_module': check.token(rt.transfer_ratio(result)), 'transfer_ratio_R': r_out[name]['transfer_ratio'], **{k: single[k].report() for k in ('grid_ratio', 'density_ratio', 'transfer_ratio')}}
    for i, c in enumerate(fixture['stress']):
        compare(check.run(c), r_out[f'stress_{i}'], groups['stress'], indices_ok)
    reference = json.loads((OUT / 'reference_expectations.json').read_text(encoding='utf-8'))['cases']
    distance = {'module': 0, 'R': 0, 'compared': 0}
    for name, ref in reference.items():
        result = check.run(fixture['cases'][name])
        for key, mine in (('tann_dists_tf_ratio', result.tann_dists_tf_ratio), ('pann_dists_tf_ratio', result.pann_dists_tf_ratio)):
            for a, b, t in zip(mine, split(r_out[name][key]), ref[key]):
                if t in ('NaN', 'Inf') or not (math.isfinite(a) and math.isfinite(b)):
                    continue
                truth = float(t)
                distance['compared'] += 1
                distance['module'] = max(distance['module'], ulps(a, truth))
                distance['R'] = max(distance['R'], ulps(b, truth))
    report = {'kind': 'R_cross_check_ratios_tfs_own_normal_functions_deviation_measured', 'runtime': info, 'python': platform.python_version(), 'backend': rt.BACKEND,
              'source': 'mcmc_functions_v1.6.R:1529-1552 (R twin of 1485-1525) and mcmc_conduction.R:228, 230, verbatim', 'cplusplus_lines_run': False, 'sources_sha256': SOURCES, 'bit_identical_with_R': False,
              'named': {k: t.report() for k, t in groups['named'].items()}, 'stress': {k: t.report() for k, t in groups['stress'].items()}, 'stress_cases': len(fixture['stress']), 'indices_equal_everywhere': indices_ok[0],
              'distance_to_true_value_ulps': distance, 'cases': named, 'historical_reproduction_ready': False}
    (OUT / 'r_crosscheck_result.json').write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(info, 'indices', indices_ok[0], distance)
    for group in ('named', 'stress'):
        for key in TALLIES:
            t = report[group][key]
            print(group, key, t['values'], 'equal', t['bit_equal'], 'max ulp', t['max_ulps'], 'rel', f"{t['max_relative']:.3g}", 'special', t['non_finite'], t['non_finite_mismatch'], t['ulp_histogram'])
    print({n: (c['kind'], c['transfer_ratio']['max_ulps'], c['transfer_ratio_module'], c['transfer_ratio_R']) for n, c in named.items()})


if __name__ == '__main__':
    main()
