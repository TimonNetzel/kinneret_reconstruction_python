"""Run tools/historical_ratios_tfs.py on the fixture and write synthetic_result.json."""
import json
import math
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import historical_ratios_tfs as rt  # noqa: E402

ARGS = ('tann_range', 'pann_range', 'new_tann', 'old_tann', 'tann_dists', 'new_pann', 'old_pann', 'pann_dists', 'lower_tann', 'upper_tann', 'sds_tann_trunc', 'lower_pann', 'upper_pann', 'sds_pann_trunc')


def value(t):
    return 0.0 if t == '0' else float.fromhex(t)


def arguments(c):
    out = {}
    for key in ARGS:
        out[key] = [[value(t) for t in row] for row in c[key]] if key.endswith('dists') else [value(t) for t in c[key]]
    return out


def token(v):
    if math.isnan(v):
        return 'NaN'
    if math.isinf(v):
        return 'Inf' if v > 0 else '-Inf'
    if v == 0:
        return '-0' if math.copysign(1.0, v) < 0 else '0'
    return float(v).hex()


def run(c):
    return rt.ratios_tfs_no_ages(c['num_taxa'], **arguments(c))


def summary(result):
    total = rt.transfer_ratio(result)
    return {'num_taxa': result.num_taxa, 'tann_tf_ratio': [token(v) for v in result.tann_tf_ratio], 'pann_tf_ratio': [token(v) for v in result.pann_tf_ratio],
            'tann_dists_tf_ratio': [token(v) for v in result.tann_dists_tf_ratio], 'pann_dists_tf_ratio': [token(v) for v in result.pann_dists_tf_ratio],
            'indices': [list(row) for row in result.indices], 'transfer_ratio': token(total), 'kind': 'number' if math.isfinite(total) and total != 0 else token(total)}


def main():
    fixture = json.loads((OUT / 'synthetic_fixture.json').read_text(encoding='utf-8'))
    cases = {name: summary(run(c)) for name, c in fixture['cases'].items()}
    (OUT / 'synthetic_result.json').write_text(json.dumps({'kind': 'module_result_on_synthetic_fixture', 'backend': rt.BACKEND, 'cases': cases, 'historical_reproduction_ready': False}, indent=1) + '\n', encoding='utf-8', newline='\n')
    print({name: (c['kind'], c['transfer_ratio']) for name, c in cases.items()})


if __name__ == '__main__':
    main()
