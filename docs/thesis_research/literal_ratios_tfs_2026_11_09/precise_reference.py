"""Independent reference for d_truncnorm and the ratios: 150-digit decimals, without NumPy, without math.erfc and without the module.

Inputs are exact binary rationals. The density is exp(-z*z/2)/(sqrt(2*pi)*sigma) and the distribution function the series of positive terms
1/2 + phi(z) * sum z^(2k+1)/(2k+1)!!. Nothing is rounded before the stored value, so the stored ratios are true values of the formulas.
"""
import json
from decimal import MAX_EMAX, MIN_EMIN, Decimal, getcontext
from pathlib import Path

OUT = Path(__file__).resolve().parent
getcontext().prec = 150
getcontext().Emax, getcontext().Emin = MAX_EMAX, MIN_EMIN
ZERO, ONE = Decimal(0), Decimal(1)
LIMIT = Decimal(10) ** -140


def to_dec(token):
    if token == '0':
        return ZERO
    n, d = float.fromhex(token).as_integer_ratio()
    return Decimal(n) / Decimal(d)


def pi():
    def atan_inv(k):
        x, total, sign, n = ONE / k, ONE / k, -1, 3
        while x > Decimal(10) ** -155:
            x = x / (k * k)
            total += sign * x / n
            sign, n = -sign, n + 2
        return total
    return 4 * (4 * atan_inv(Decimal(5)) - atan_inv(Decimal(239)))


ROOT_2PI = (2 * pi()).sqrt()


def phi(z):
    return (-z * z / 2).exp() / ROOT_2PI


def cdf(z):
    if abs(z) > 40:  # the tail is below 1e-340, far under the 150 digits
        return ONE if z > 0 else ZERO
    term, total, k = z, z, 1
    while abs(term) > LIMIT * max(abs(total), ONE) or k < 3:
        term = term * z * z / (2 * k + 1)
        total += term
        k += 1
    return Decimal('0.5') + phi(z) * total


def density(x, mu, sigma, a, b):
    if x < a or x > b:
        return ZERO
    return phi((x - mu) / sigma) / sigma / (cdf((b - mu) / sigma) - cdf((a - mu) / sigma))


def ratio(top, bottom):
    if bottom == 0:
        return 'NaN' if top == 0 else 'Inf'
    return str(top / bottom)


def nearest(grid, value):
    best, index = abs(grid[0] - value), 0
    for i, g in enumerate(grid):
        if abs(g - value) < best:
            best, index = abs(g - value), i
    return index


def compute(c):
    d = {k: ([[to_dec(t) for t in row] for row in c[k]] if k.endswith('dists') else [to_dec(t) for t in c[k]]) for k in c if k != 'num_taxa'}
    out = {'tann_tf_ratio': [], 'pann_tf_ratio': [], 'tann_dists_tf_ratio': [], 'pann_dists_tf_ratio': [], 'indices': []}
    for i in range(c['num_taxa']):
        idx = [nearest(d['tann_range'], d['new_tann'][i]), nearest(d['tann_range'], d['old_tann'][i]), nearest(d['pann_range'], d['new_pann'][i]), nearest(d['pann_range'], d['old_pann'][i])]
        out['indices'].append(idx)
        out['tann_tf_ratio'].append(ratio(d['tann_dists'][i][idx[0]], d['tann_dists'][i][idx[1]]))
        out['pann_tf_ratio'].append(ratio(d['pann_dists'][i][idx[2]], d['pann_dists'][i][idx[3]]))
        for name, new, old, sd, low, up in (('tann_dists_tf_ratio', d['new_tann'], d['old_tann'], d['sds_tann_trunc'], d['lower_tann'], d['upper_tann']), ('pann_dists_tf_ratio', d['new_pann'], d['old_pann'], d['sds_pann_trunc'], d['lower_pann'], d['upper_pann'])):
            out[name].append(ratio(density(old[i], new[i], sd[i], low[i], up[i]), density(new[i], old[i], sd[i], low[i], up[i])))
    return out


def main():
    fixture = json.loads((OUT / 'synthetic_fixture.json').read_text(encoding='utf-8'))
    skip = {'equal_bounds', 'live_shape_grid_1000'}  # equal bounds divide 0 by 0 in the scale; the 1000-point case is left to the R record
    cases = {n: compute(c) for n, c in fixture['cases'].items() if n not in skip}
    (OUT / 'reference_expectations.json').write_text(json.dumps({'kind': 'independent_decimal_reference_150_digits', 'cases': cases, 'historical_reproduction_ready': False}, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(len(cases), 'reference cases')


if __name__ == '__main__':
    main()
