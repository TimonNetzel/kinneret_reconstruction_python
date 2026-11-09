"""Independent reference for reconstruction_climate, without importing the module.

The product is written as exact fractions with one rounding per IEEE operation. The weighted median comes from the
independent rule-and-fraction reference of the weighted-median packet, which does not import the module either.
"""
from __future__ import annotations

import importlib.util
import json
import math
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('wm_reference_for_climate', HERE.parent / 'literal_weighted_median_2026_11_08/precise_reference.py')
wm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wm)
SPECIAL = {'NA': math.nan, 'NaN': math.nan}


def parse(text):
    return SPECIAL[text] if text in SPECIAL else float(Fraction(text))


def product(row, matrix):
    out = []
    for j in range(len(matrix[0])):
        total = 0.0
        for share, tf_row in zip(row, matrix):
            if math.isnan(share) or math.isnan(tf_row[j]) or math.isnan(total):
                total = math.nan
                continue
            total = float(Fraction(total) + Fraction(float(Fraction(tf_row[j]) * Fraction(share))))
        out.append(total)
    return out


def median(grid, weights):
    if wm.na_rule(grid, weights):
        return 'NA', None
    return 'number', wm.interpolated(grid, weights)['value']


def expected(case, x1range, x2range, prior):
    if not case['recent']:
        return {'tann': ('number', parse(prior['mean_tmp'])), 'pann': ('number', parse(prior['mean_pann']))}
    first = [parse(v) for v in case['core'][0]]
    w1 = product(first, [[parse(v) for v in r] for r in case['tf_x1']])
    w2 = product(first, [[parse(v) for v in r] for r in case['tf_x2']])
    return {'tann': median([float(v) for v in x1range], w1), 'pann': median([float(v) for v in x2range], w2),
            'tann_weights': w1, 'pann_weights': w2}


def show(v):
    return 'NaN' if math.isnan(v) else v.hex()


def main():
    fixture = json.loads((HERE / 'synthetic_fixture.json').read_text(encoding='utf-8'))
    out = {}
    for name, case in fixture['cases'].items():
        r = expected(case, fixture['x1range'], fixture['x2range'], fixture['prior'])
        out[name] = {k: ({'kind': r[k][0], 'value': None if r[k][1] is None else r[k][1].hex()}) for k in ('tann', 'pann')}
        for k in ('tann_weights', 'pann_weights'):
            if k in r:
                out[name][k] = [show(v) for v in r[k]]
    (HERE / 'reference_expectations.json').write_text(json.dumps({'kind': 'independent_reconstruction_climate_reference', 'cases': out},
                                                                 indent=1) + '\n', encoding='utf-8', newline='\n')
    print(len(out), 'reference cases')


if __name__ == '__main__':
    main()
