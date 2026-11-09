"""R 3.5.0 against the module; wt_power and its reader verbatim on artificial series. Measures the deviation of the NumPy FFT, it does not hide it.

Lines 1004-1005 must be bit-identical. The steps from ``fft`` on are compared value by value in ulp and relative to the largest value of the
column, because a value near zero can be many ulp away while being tiny on the scale of the column. Usage: run_r_crosscheck.py <Rscript.exe>
"""
import hashlib
import importlib.util
import json
import math
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
_spec = importlib.util.spec_from_file_location('wt_check', OUT / 'check_wt_power.py')
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)

SOURCES = (('src/mcmc_functions_v1.6.R', 'b887144370e328fa016d1834ceebcd55622816d4b36838d33a26197110ab453c'),)
LOCAL = ROOT / '.local/wt_power_2026_11_09'
BINS = ((0, 0, '0'), (1, 1, '1'), (2, 2, '2'), (3, 4, '3-4'), (5, 16, '5-16'), (17, 256, '17-256'), (257, 2 ** 70, '>256'))


def r_num(v):
    if v is check.wp.NA:
        return 'NA_real_'
    if math.isnan(v):
        return 'NaN'
    if math.isinf(v):
        return 'Inf' if v > 0 else '-Inf'
    if v == 0:
        return '-0' if math.copysign(1.0, v) < 0 else '0'
    return float(v).hex()


def r_vec(values):
    return 'as.double(c(' + ', '.join(r_num(v) for v in values) + '))' if len(values) else 'numeric(0)'


def r_rows(rows):
    return f'matrix({r_vec([v for row in rows for v in row])}, nrow = {len(rows)}, ncol = {len(rows[0])}, byrow = TRUE)'


def r_lgl(values):
    return 'c(' + ', '.join('TRUE' if v else 'FALSE' for v in values) + ')'


def r_case(c):
    return f'list(d = {r_vec(c["d"])}, l_d = {c["l_d"]}L, n_obs = {c["n_obs"]}L, zeros = {c["zeros"]}L, invflen = {r_num(c["invflen"])}, wb = {r_rows(c["wb"])})'


def named(c):
    return {'d': check.vector(c['d']), 'l_d': c['l_d'], 'n_obs': c['n_obs'], 'zeros': c['zeros'], 'invflen': check.value(c['invflen']), 'wb': check.wb_rows(c['wb'])}


def r_chain(ch):
    return (f'list(current = {r_vec(check.vector(ch["predictant_current"]))}, ages_spec = {r_lgl(ch["ages_spec"])}, hi_ages_spec = {r_lgl(ch["hi_ages_spec"])}, n_obs = {ch["n_obs"]}L, zeros = {ch["zeros"]}L, '
            f'invflen = {r_num(check.value(ch["invflen"]))}, wb = {r_rows(check.wb_rows(ch["wb"]))}, coi = {r_rows(ch["coi"])}, iso = {r_vec(check.vector(ch["iso_spec"]))})')


def from_r(text):
    return text if text in ('NA', 'NaN', 'Inf', '-Inf') else (-0.0 if text == '-0' else 0.0 if text == '0' else float.fromhex(text))


def split(text):
    return [from_r(t) for t in text.split()] if text else []


def ulps(a, b):
    ia, ib = (struct.unpack('<q', struct.pack('<d', v))[0] for v in (a, b))
    # distances are measured over the sign change; +0 and -0 count as equal
    if a == 0 and b == 0:
        return 0
    ia = ia if ia >= 0 else -(ia & 0x7fffffffffffffff)
    ib = ib if ib >= 0 else -(ib & 0x7fffffffffffffff)
    return abs(ia - ib)


class Tally:
    def __init__(self):
        self.values, self.bit_equal, self.max_ulps, self.max_rel = 0, 0, 0, 0.0
        self.hist = {label: 0 for _, _, label in BINS}

    def add(self, mine, theirs, scale=None):
        for a, b in zip(mine, theirs):
            u = ulps(a, b)
            self.values += 1
            self.bit_equal += u == 0
            self.max_ulps = max(self.max_ulps, u)
            for low, high, label in BINS:
                if low <= u <= high:
                    self.hist[label] += 1
                    break
            if scale:
                self.max_rel = max(self.max_rel, abs(a - b) / scale)

    def report(self):
        return {'values': self.values, 'bit_equal': self.bit_equal, 'max_ulps': self.max_ulps, 'ulp_histogram': self.hist, 'max_error_relative_to_largest_of_column': self.max_rel}


def pieces(values, columns):
    """Split column-major values into columns."""
    size = len(values) // columns
    return [values[i * size:(i + 1) * size] for i in range(columns)]


def compare_finite(result, r, tallies):
    tallies['x'].add([float(v) for v in result.x], split(r['x']))
    fr, fi = split(r['f_re']), split(r['f_im'])
    scale = max(max(map(abs, fr)), max(map(abs, fi))) or 1.0
    tallies['f'].add(list(result.f.real.astype(float)), fr, scale)
    tallies['f'].add(list(result.f.imag.astype(float)), fi, scale)
    wr, wi, power = split(r['wave_re']), split(r['wave_im']), split(r['power'])
    wave = result.wave.flatten(order='F')
    mine_r, mine_i = [float(v) for v in wave.real], [float(v) for v in wave.imag]
    for cols in zip(pieces(mine_r, result.columns), pieces(wr, result.columns), pieces(mine_i, result.columns), pieces(wi, result.columns)):
        tallies['wave'].add(cols[0], cols[1], max(max(map(abs, cols[1])), max(map(abs, cols[3]))) or 1.0)
        tallies['wave'].add(cols[2], cols[3], max(max(map(abs, cols[1])), max(map(abs, cols[3]))) or 1.0)
    for a, b in zip(pieces(list(result.values), result.columns), pieces(power, result.columns)):
        tallies['power'].add(a, b, max(map(abs, b)) or 1.0)
    # the modulus alone, on R's own wave: separates the FFT from hypot
    on_r_wave = np.abs(np.array(wr) + 1j * np.array(wi))
    tallies['power_from_R_wave'].add([float(v) for v in on_r_wave * on_r_wave], power)


def main():
    rscript = str(Path(sys.argv[1]).resolve())
    cfg = json.loads((ROOT / '.local/my_reconst_paths.json').read_text(encoding='utf-8-sig'))
    root = Path(cfg['original_data_root'])
    path, prepare = root / SOURCES[0][0], root / 'src/mcmc_prepare.R'
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCES[0][1]:
        raise ValueError('mcmc_functions_v1.6.R differs from the frozen binding')
    fixture = json.loads((OUT / 'synthetic_fixture.json').read_text(encoding='utf-8'))
    cases = {n: named(c) for n, c in fixture['cases'].items()}
    stress = check.stress_cases(fixture['stress'])
    lines = [f'source_file <- "{path.as_posix()}"', f'prepare_file <- "{prepare.as_posix()}"', 'cases <- list()', 'stress <- list()', 'chains <- list()']
    lines += [f'cases[["{n}"]] <- {r_case(c)}' for n, c in cases.items()]
    lines += [f'stress[["stress_{i}"]] <- {r_case(c)}' for i, c in enumerate(stress)]
    lines += [f'chains[["{n}"]] <- {r_chain(c)}' for n, c in fixture['chains'].items()]
    LOCAL.mkdir(parents=True, exist_ok=True)
    (LOCAL / 'input.R').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    subprocess.run([rscript, '--vanilla', str(OUT / 'r_crosscheck.R'), str(LOCAL / 'input.R'), str(LOCAL / 'output.tsv')], check=True, stdin=subprocess.DEVNULL)
    r_out, info = {}, None
    for row in (LOCAL / 'output.tsv').read_text(encoding='utf-8').splitlines():
        parts = row.split('\t')
        if parts[0] == 'info':
            info = dict(zip(('R', 'fft_namespace', 'long_double'), parts[1:]))
        else:
            r_out.setdefault(parts[0], {})[parts[1]] = parts[2] if len(parts) > 2 else ''
    groups = {g: {k: Tally() for k in ('x', 'f', 'wave', 'power', 'power_from_R_wave')} for g in ('named', 'stress')}
    named_rows, token_mismatch, missing_mismatch, functions_agree = {}, 0, 0, True
    wb_types = set()
    for name, c in fixture['cases'].items():
        got, result = check.run_case(c)
        r = r_out[name]
        functions_agree &= r['function_equals_steps'] == 'TRUE'
        wb_types.add(r['wb_type'])
        row = {'dim_R': r['dim'], 'dim_module': 'x'.join(map(str, got['dim'])) if got['dim'] else 'none', 'kind': got['kind'], 'x_R': r['x']}
        if got['kind'] == 'number':
            single = {k: Tally() for k in ('x', 'f', 'wave', 'power', 'power_from_R_wave')}
            compare_finite(result, r, single)
            compare_finite(result, r, groups['named'])
            row.update({k: single[k].report() for k in ('x', 'power')})
            row['power_R'] = r['power']
        else:
            tokens = r['power'].split()
            row['power_tokens_R'] = sorted(set(tokens))
            row['x_tokens_equal'] = r['x'].split() == got['x']
            if any(t not in ('NA', 'NaN') for t in tokens):
                missing_mismatch += 1
            if got['kind'] != 'mixed' and tokens != got['power']:
                token_mismatch += 1
            if r['x'].split() != got['x']:
                token_mismatch += 1
        named_rows[name] = row
    for i, c in enumerate(stress):
        result = check.run_stress(c)
        r = r_out[f'stress_{i}']
        functions_agree &= r['function_equals_steps'] == 'TRUE'
        compare_finite(result, r, groups['stress'])
    chains = {}
    post_max_ulps, post_max_abs = 0, 0.0
    for name, ch in fixture['chains'].items():
        got, _ = check.run_chain(ch)
        r = r_out[name]
        mine, theirs = from_r(got['post_spec']), from_r(r['post_spec'])
        u = ulps(mine, theirs)
        post_max_ulps = max(post_max_ulps, u)
        post_max_abs = max(post_max_abs, abs(mine - theirs))
        chains[name] = {'post_spec_module': got['post_spec'], 'post_spec_R': r['post_spec'], 'ulps': u, 'absolute': abs(mine - theirs), 'l_d_R': int(r['length_ages_spec2']), 'l_d_module': got['l_d'],
                        'replaced': got['replaced']}
    reference = json.loads((OUT / 'reference_expectations.json').read_text(encoding='utf-8'))['cases']
    from_true = {'module': 0, 'R': 0}
    for name, ref in reference.items():
        got = named_rows[name]
        mine = check.run_case(fixture['cases'][name])[1].values
        truth = [float(v) for v in (t for t in ref['power_true'])]
        for a, b, t in zip(mine, split(r_out[name]['power']), truth):
            from_true['module'] = max(from_true['module'], ulps(a, t))
            from_true['R'] = max(from_true['R'], ulps(b, t))
    report = {'kind': 'R_cross_check_wt_power_numpy_fft_deviation_measured', 'runtime': info, 'backend': check.wp.BACKEND, 'numpy': np.__version__, 'source': SOURCES[0][0] + ':1003-1012, 2023-2026, 2355-2366; mcmc_prepare.R:1816-1817, verbatim',
              'source_sha256': SOURCES[0][1], 'bit_identical_with_R': False, 'named': {k: t.report() for k, t in groups['named'].items()}, 'stress': {k: t.report() for k, t in groups['stress'].items()},
              'stress_cases': len(stress), 'token_mismatch': token_mismatch, 'missing_mismatch': missing_mismatch, 'function_equals_steps_everywhere': functions_agree, 'wb_types_in_R': sorted(wb_types),
              'chains': chains, 'post_spec_max_ulps': post_max_ulps, 'post_spec_max_absolute': post_max_abs, 'distance_to_true_value_ulps': from_true, 'cases': named_rows, 'historical_reproduction_ready': False}
    (OUT / 'r_crosscheck_result.json').write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(info)
    for group in ('named', 'stress'):
        for key in ('x', 'f', 'wave', 'power', 'power_from_R_wave'):
            t = report[group][key]
            print(group, key, t['values'], 'equal', t['bit_equal'], 'max ulp', t['max_ulps'], 'rel', f"{t['max_error_relative_to_largest_of_column']:.3g}", t['ulp_histogram'])
    print('token', token_mismatch, 'missing', missing_mismatch, 'function=steps', functions_agree, 'wb', wb_types, 'post ulps', post_max_ulps, post_max_abs, 'to truth', from_true)
    print({n: (c['ulps'], c['replaced'], c['l_d_R'], c['l_d_module']) for n, c in chains.items()})


if __name__ == '__main__':
    main()
