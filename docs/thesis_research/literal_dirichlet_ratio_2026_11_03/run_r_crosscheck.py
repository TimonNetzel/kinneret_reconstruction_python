"""R 3.5.0 + LaplacesDemon 16.1.6 against the module: mcmc_functions_v1.6.R 1269-1274 verbatim on the fixture.

The original is read only after its SHA-256 matches the frozen binding; LaplacesDemon comes from the CRAN archive
tarball whose SHA-256 is recorded in r_runtime_crosscheck_2026_11_03/dirichlet_versions.json, installed into a local
library. Doubles travel as C99 hex. Every value is compared bit for bit; differences are reported in ulps, not hidden.
Usage: run_r_crosscheck.py <Rscript.exe> <library folder with LaplacesDemon>
"""
import hashlib
import json
import math
import struct
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import historical_dirichlet_ratio as dr  # noqa: E402

SOURCE = ('src/mcmc_functions_v1.6.R', 'b887144370e328fa016d1834ceebcd55622816d4b36838d33a26197110ab453c')
LOCAL = ROOT / '.local/dirichlet_ratio_2026_11_03'


def r_vec(values):
    return 'c(' + ', '.join(float(v).hex() for v in values) + ')'


def from_hex(t):
    return {'NA': 'NA', 'NaN': 'NaN', 'Inf': 'Inf', '-Inf': '-Inf', '-0': '-0', '0': 0.0}.get(t) \
        if t in ('NA', 'NaN', 'Inf', '-Inf', '-0', '0') else float.fromhex(t)


def special(v):
    if math.isnan(v):
        return 'NaN'
    if math.isinf(v):
        return 'Inf' if v > 0 else '-Inf'
    return v


def ulps(a, b):
    if isinstance(a, str) or isinstance(b, str):
        return 0 if a == b else None
    ia, ib = (struct.unpack('<q', struct.pack('<d', v))[0] for v in (a, b))
    return abs(ia - ib)


def main():
    rscript, library = sys.argv[1], sys.argv[2]
    cfg = json.loads((ROOT / '.local/my_reconst_paths.json').read_text(encoding='utf-8-sig'))
    path = Path(cfg['original_data_root']) / SOURCE[0]
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE[1]:
        raise ValueError('mcmc_functions_v1.6.R differs from the frozen binding')
    fixture = json.loads((OUT / 'synthetic_fixture.json').read_text(encoding='utf-8'))
    val = lambda t: float(Fraction(t))
    lines = [f'source_file <- "{path.as_posix()}"', 'ratio_cases <- list()', 'density_cases <- list()']
    for name, c in fixture['ratio_cases'].items():
        lines.append(f'ratio_cases[["{name}"]] <- list(old = {r_vec(map(val, c["old_weights"]))}, '
                     f'new = {r_vec(map(val, c["new_weights"]))}, num_taxa = {c["num_taxa"]})')
    for name, c in fixture['density_cases'].items():
        lines.append(f'density_cases[["{name}"]] <- list(x = {r_vec(map(val, c["x"]))}, '
                     f'alpha = {r_vec(map(val, c["alpha"]))})')
    LOCAL.mkdir(parents=True, exist_ok=True)
    (LOCAL / 'input.R').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    subprocess.run([rscript, '--vanilla', str(OUT / 'r_crosscheck.R'), library, str(LOCAL / 'input.R'),
                    str(LOCAL / 'output.tsv')], check=True, stdin=subprocess.DEVNULL)
    r, info = {}, None
    for line in (LOCAL / 'output.tsv').read_text(encoding='utf-8').splitlines():
        parts = line.split('\t')
        if parts[0] == 'R':
            info = {'R': parts[1], 'LaplacesDemon': parts[2], 'ddirichlet_namespace': parts[3]}
            continue
        r.setdefault(parts[0], {})[parts[1]] = [from_hex(t) for t in parts[2].split()]
    rows, all_ulps = {}, []
    for name, c in fixture['ratio_cases'].items():
        got = dr.dirichlet_ratio([val(v) for v in c['old_weights']], [val(v) for v in c['new_weights']], c['num_taxa'])
        mine = {'weights_ratio': [special(got.weights_ratio)], 'proposal_ratio': [special(got.proposal_ratio)],
                'parameter_ratio': [special(got.parameter_ratio)],
                'densities': [special(d.value) for d in (got.weights_numerator, got.weights_denominator,
                                                         got.proposal_numerator, got.proposal_denominator)]}
        row = {}
        for key, values in mine.items():
            d = [ulps(a, b) for a, b in zip(values, r[name][key])]
            row[key] = {'module': values, 'R': r[name][key], 'ulps': d}
            all_ulps += d
        rows[name] = row
    dens = {}
    for name, c in fixture['density_cases'].items():
        got = dr.ddirichlet([val(v) for v in c['x']], [val(v) for v in c['alpha']])
        d = {'density': ulps(got.value, r[name]['density'][0]), 'log_density': ulps(got.log_density,
                                                                                    r[name]['log_density'][0])}
        dens[name] = {'module': [got.value, got.log_density], 'R': [r[name]['density'][0], r[name]['log_density'][0]],
                      'ulps': d}
        all_ulps += list(d.values())
    report = {
        'kind': 'R_cross_check_literal_lines_1269_1274_with_LaplacesDemon_on_synthetic_fixture',
        'runtime': info, 'source': SOURCE[0] + ':1269-1274', 'source_sha256': SOURCE[1],
        'values_compared': len(all_ulps), 'bit_equal': sum(1 for u in all_ulps if u == 0),
        'max_ulps': max(u for u in all_ulps if u is not None), 'special_mismatch': sum(u is None for u in all_ulps),
        'ratio_cases': rows, 'density_cases': dens,
        'note': 'R uses nmath lgamma, C-library log/exp and long-double rowSums; the module rounds each function '
                'correctly. Non-zero ulps are reported as measured, not treated as errors.',
        'historical_reproduction_ready': False,
    }
    (OUT / 'r_crosscheck_result.json').write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(info, report['values_compared'], 'values,', report['bit_equal'], 'bit-equal, max', report['max_ulps'],
          'ulps, special mismatches', report['special_mismatch'])


if __name__ == '__main__':
    main()
