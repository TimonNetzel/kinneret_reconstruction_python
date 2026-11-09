"""Independent reference for the local cor, without importing the module.

Every step is written as exact fractions with an explicit rounding rule, not as integer alignment as in the module: 64-bit
nearest-even rounding for each long double addition, nearest-even double rounding for each IEEE operation, and a square root
with a sticky bit. Which NA token a missing result carries comes from the rule measured in R 3.5.0.
"""
from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
NA = object()


def parse(text):
    if text == 'NA':
        return NA
    if text == 'NaN':
        return math.nan
    if text.startswith('0x') or text.startswith('-0x'):
        return float.fromhex(text)
    return float(Fraction(text))


def _is_missing(v):
    return v is NA or (isinstance(v, float) and math.isnan(v))


def round_bits(value: Fraction, bits: int) -> Fraction:
    """Nearest-even rounding of a nonzero fraction to ``bits`` significant binary digits."""
    if value == 0:
        return value
    sign = 1 if value > 0 else -1
    magnitude = abs(value)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    while Fraction(2) ** exponent > magnitude:
        exponent -= 1
    while Fraction(2) ** (exponent + 1) <= magnitude:
        exponent += 1
    unit = Fraction(2) ** (exponent - bits + 1)
    quotient = magnitude / unit
    whole = quotient.numerator // quotient.denominator
    rest = quotient - whole
    if rest > Fraction(1, 2) or (rest == Fraction(1, 2) and whole % 2 == 1):
        whole += 1
    return sign * whole * unit


def long_double_sum(values):
    total = Fraction(0)
    for v in values:
        total = round_bits(total + Fraction(v), 64)
    return float(total)


def divide(a, b):
    if b == 0:
        return math.nan if a == 0 or math.isnan(a) else math.copysign(math.inf, a) * math.copysign(1.0, b)
    if math.isnan(a) or math.isnan(b):
        return math.nan
    if a == 0:
        return math.copysign(0.0, a) * math.copysign(1.0, b)
    return float(Fraction(a) / Fraction(b))


def sqrt(v):
    if math.isnan(v) or v == 0:
        return v
    fraction = Fraction(v)
    k = max(fraction.denominator.bit_length(), 700)
    scaled = fraction * 2 ** (2 * k)
    if scaled.denominator != 1:
        raise ValueError('scale too small')
    root = math.isqrt(scaled.numerator)
    sticky = 0 if root * root == scaled.numerator else 1
    return float(Fraction(2 * root + sticky, 2 ** (k + 1)))


def cor(x, y):
    """Lines 2361-2365 for equal lengths."""
    n = len(y)
    missing = next((v for v in x + y if _is_missing(v)), None)
    if missing is not None:
        return 'NA' if missing is NA else 'NaN', None
    mean_x, mean_y = divide(long_double_sum(x), float(n)), divide(long_double_sum(y), float(n))
    dx = [float(Fraction(v) - Fraction(mean_x)) for v in x]
    dy = [float(Fraction(v) - Fraction(mean_y)) for v in y]
    total = 0.0
    for a, b in zip(dx, dy):
        total = float(Fraction(total) + Fraction(float(Fraction(b) * Fraction(a))))
    divisor = float(n - 1)
    numerator = divide(total, divisor)
    var_x = divide(long_double_sum([float(Fraction(v) ** 2) for v in dx]), divisor)
    var_y = divide(long_double_sum([float(Fraction(v) ** 2) for v in dy]), divisor)
    root_x, root_y = sqrt(var_x), sqrt(var_y)
    if math.isnan(root_x) or math.isnan(root_y):
        return 'NaN', None
    product = root_x * root_y if root_x == 0 or root_y == 0 else float(Fraction(root_x) * Fraction(root_y))
    value = divide(numerator, product)
    return ('NaN', None) if math.isnan(value) else ('number', value)


def apply_cor(case):
    x, y = [parse(v) for v in case['x']], [parse(v) for v in case['y']]
    if case['na_rm']:
        pairs = [(a, b) for a, b in zip(x, y) if not _is_missing(a)]
        x, y = [a for a, _ in pairs], [b for _, b in pairs]
    return cor(x, y)


def main():
    fixture = json.loads((HERE / 'synthetic_fixture.json').read_text(encoding='utf-8'))
    out = {}
    for name, case in fixture['cases'].items():
        kind, value = apply_cor(case)
        out[name] = {'kind': kind, 'value': None if value is None else value.hex()}
    (HERE / 'reference_expectations.json').write_text(json.dumps({'kind': 'independent_local_cor_reference', 'cases': out}, indent=1) + '\n',
                                                      encoding='utf-8', newline='\n')
    print(len(out), 'reference cases')


if __name__ == '__main__':
    main()
