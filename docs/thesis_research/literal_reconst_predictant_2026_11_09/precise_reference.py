"""Independent reference for reconst and the predictants, without importing the module.

Written in exact fractions with one explicit rounding step per IEEE operation, and accumulated column by column as dgemm is, while the
module walks row by row. Shapes come from a small table of R's rules. Which token a missing value carries is the rule measured in R 3.5.0.
"""
from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
NA, NAN, ZERO = 'NA', 'NaN', 0
TOKENS = {'NA': NA, 'NaN': NAN, 'Inf': math.inf, '-Inf': -math.inf}


def parse(text):
    if text in TOKENS:
        return TOKENS[text]
    if text == '-0':
        return -0.0
    if text.startswith('0x') or text.startswith('-0x'):
        return float.fromhex(text)
    return float(Fraction(text))


def _is_na_or_nan(v):
    return v is NA or v is NAN


def _exact_to_double(value: Fraction) -> float:
    try:
        return float(value)  # int / int in CPython is correctly rounded, nearest even, also for subnormals
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def multiply(a, b):
    if _is_na_or_nan(a):
        return a
    if _is_na_or_nan(b):
        return b
    if math.isinf(a) or math.isinf(b):
        if a == 0 or b == 0:
            return NAN
        return math.copysign(math.inf, a) * math.copysign(1.0, b)
    if a == 0 or b == 0:
        return math.copysign(0.0, a) * math.copysign(1.0, b)
    return _exact_to_double(Fraction(a) * Fraction(b))


def add(a, b):
    if _is_na_or_nan(a):
        return a
    if _is_na_or_nan(b):
        return b
    if math.isinf(a) or math.isinf(b):
        if math.isinf(a) and math.isinf(b) and a != b:
            return NAN
        return a if math.isinf(a) else b
    exact = Fraction(a) + Fraction(b)
    if exact == 0:
        return -0.0 if math.copysign(1.0, a) < 0 and math.copysign(1.0, b) < 0 else 0.0
    return _exact_to_double(exact)


def shape_after_selection(nrow, ncol, chosen):
    """(row count, col count) of the selection, or None when R drops to a plain vector."""
    if chosen == 1 or (chosen == 0 and ncol == 1) or ncol == 1:
        return None
    return chosen, ncol


def select(core, mask):
    nrow, ncol = len(core), len(core[0])
    if len(mask) > nrow:
        return 'STOP_1818', None
    rows = []
    for i in range(nrow if mask else 0):
        flag = mask[i % len(mask)]
        if flag is None:
            rows.append([NA] * ncol)
        elif flag:
            rows.append(core[i])
    return rows, shape_after_selection(nrow, ncol, len(rows))


def product(rows, shape, tf):
    """Accumulation per result column, with the taxon index running outermost per column, as in dgemm."""
    nry = len(tf)
    if shape is None:
        length = len(rows[0]) if len(rows) == 1 else len(rows)
        if len(rows) == 1 and length == nry:
            matrix = rows
        elif len(rows) != 1 and nry == 1:
            matrix = rows
        elif len(rows) == 1 and nry == 1:
            matrix = [[v] for v in rows[0]]
        elif not rows and nry == 1:
            matrix = []
        else:
            return 'STOP_1818'
    else:
        if shape[1] != nry:
            return 'STOP_1818'
        matrix = rows
    result = [[0.0 for _ in matrix] for _ in range(2)]
    for k in range(2):
        for j in range(nry):
            for i in range(len(matrix)):
                result[k][i] = add(result[k][i], multiply(matrix[i][j], tf[j][k]))
    return result


def token(v):
    if v is NA or v is NAN:
        return v
    if math.isinf(v):
        return 'Inf' if v > 0 else '-Inf'
    if v == 0:
        return '-0' if math.copysign(1.0, v) < 0 else '0'
    return v.hex()


def compute(case):
    core = [[parse(v) for v in row] for row in case['core']]
    tf = [[parse(v) for v in row] for row in case['tf']]
    rows, shape = select(core, case['mask'])
    if rows == 'STOP_1818':
        return {'stop': '1818'}
    columns = product(rows, shape, tf)
    if columns == 'STOP_1818':
        return {'stop': '1818'}
    out = {'tann': [token(v) for v in columns[0]], 'pann': [token(v) for v in columns[1]]}
    if not columns[0]:
        out['stop'] = '1825'
    return out


def main():
    fixture = json.loads((HERE / 'synthetic_fixture.json').read_text(encoding='utf-8'))
    out = {name: compute(case) for name, case in fixture['cases'].items()}
    (HERE / 'reference_expectations.json').write_text(json.dumps({'kind': 'independent_reconst_reference', 'cases': out}, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(len(out), 'reference cases')


if __name__ == '__main__':
    main()
