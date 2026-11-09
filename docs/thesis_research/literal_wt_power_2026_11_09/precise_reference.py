"""Independent reference for wt_power: a direct DFT in 90-digit decimals, without NumPy and without the module.

Inputs are exact binary rationals. The mean follows R: an 80-bit accumulator, one cast to double, a double division. Transforms, products and
the modulus are not rounded, so the stored power is the correctly rounded true value of the formula. Only finite cases with n <= 32 are run.
"""
import json
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path

OUT = Path(__file__).resolve().parent
getcontext().prec = 90
ZERO, ONE = Decimal(0), Decimal(1)


def exact(token):
    if '/' in token:
        return float(Fraction(token))
    return float.fromhex(token) if token.lower().startswith(('0x', '-0x')) else float(Fraction(token))


def pi():
    def atan_inv(k):
        x, total, term, sign, n = ONE / k, ONE / k, ONE / k, -1, 3
        while abs(term) > Decimal(10) ** -95:
            term = term / (k * k)
            total += sign * term / n
            sign, n = -sign, n + 2
        return total
    return 4 * (4 * atan_inv(Decimal(5)) - atan_inv(Decimal(239)))


PI = pi()


def cos_sin(angle):
    c, s, term, k = ONE, angle, angle, 1
    term_c = ONE
    while True:
        term_c = -term_c * angle * angle / ((2 * k - 1) * (2 * k))
        term = -term * angle * angle / ((2 * k) * (2 * k + 1))
        c += term_c
        s += term
        k += 1
        if abs(term_c) + abs(term) < Decimal(10) ** -95:
            return c, s


def round_64(value):
    """Round a Fraction to 64 significant bits, nearest even."""
    if value == 0:
        return value
    sign, value = (-1, -value) if value < 0 else (1, value)
    shift = 64 - (value.numerator.bit_length() - value.denominator.bit_length())
    scaled = value * (Fraction(2) ** shift)
    while scaled >= 2 ** 64:
        shift -= 1
        scaled = value * (Fraction(2) ** shift)
    while scaled < 2 ** 63:
        shift += 1
        scaled = value * (Fraction(2) ** shift)
    whole = scaled.numerator // scaled.denominator
    rest = scaled - whole
    if rest > Fraction(1, 2) or (rest == Fraction(1, 2) and whole & 1):
        whole += 1
    return sign * Fraction(whole) / (Fraction(2) ** shift)


def centred(d, l_d, zeros):
    total = Fraction(0)
    for v in d:
        total = round_64(total + Fraction(v))
    mean = float(total) / float(l_d)
    return [v - mean for v in d] + [0.0] * zeros


def to_dec(v):
    return Decimal(v.as_integer_ratio()[0]) / Decimal(v.as_integer_ratio()[1])


def tok(v):
    return '0' if v == 0 else v.hex()


def power(case):
    d = [exact(t) for t in case['d']]
    x = centred(d, case['l_d'], case['zeros'])
    n = len(x)
    wb = [[exact(t) for t in col] for col in case['wb']]
    invflen = exact(case['invflen'])
    twiddle = [cos_sin(2 * PI * Decimal(j) / Decimal(n)) for j in range(n)]
    f = []
    for k in range(n):
        re = sum((to_dec(x[j]) * twiddle[j * k % n][0] for j in range(n)), ZERO)
        im = sum((-to_dec(x[j]) * twiddle[j * k % n][1] for j in range(n)), ZERO)
        f.append((re, im))
    out = []
    for col in wb:
        g = [(f[k][0] * to_dec(col[k]), f[k][1] * to_dec(col[k])) for k in range(n)]
        for j in range(case['n_obs']):
            re = sum((g[k][0] * twiddle[j * k % n][0] - g[k][1] * twiddle[j * k % n][1] for k in range(n)), ZERO)
            im = sum((g[k][0] * twiddle[j * k % n][1] + g[k][1] * twiddle[j * k % n][0] for k in range(n)), ZERO)
            out.append((re * re + im * im) * to_dec(invflen) * to_dec(invflen))
    return {'x': [tok(v) for v in x], 'power_true': [str(v) for v in out], 'power': [tok(float(v)) for v in out]}


def finite(case):
    return all(t not in ('NA', 'NaN', 'Inf', '-Inf') for t in case['d']) and case['zeros'] + len(case['d']) <= 32


def main():
    fixture = json.loads((OUT / 'synthetic_fixture.json').read_text(encoding='utf-8'))
    cases = {n: power(c) for n, c in fixture['cases'].items() if finite(c)}
    (OUT / 'reference_expectations.json').write_text(json.dumps({'kind': 'independent_direct_dft_90_digits', 'cases': cases, 'historical_reproduction_ready': False}, indent=1) + '\n',
                                                     encoding='utf-8', newline='\n')
    print(len(cases), 'reference cases')


if __name__ == '__main__':
    main()
