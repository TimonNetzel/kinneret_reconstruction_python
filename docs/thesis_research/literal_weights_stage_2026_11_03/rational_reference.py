"""Independent reference for mcmc_functions_v1.6.R 1259 and 1276-1280: plain lists and Fraction; no NumPy, no module
import.

Every stored double becomes an exact Fraction. A product or quotient is exact first and rounded once to the nearest
double (float(Fraction) rounds correctly), which is what one IEEE operation does. The row sum is the exact sum of the
kept values (neither NA nor NaN) rounded once; -Inf among them gives -Inf. A row sum is accumulator independent when
EVERY subset sum of the kept values is an exact double, because every partial sum of any summation order is such a
subset sum (checked here by enumeration, unlike the sufficient power-of-two bound in the module). Cells are divided
by the sum of their own row: 0/0 and -Inf/-Inf are "NaN", x/0 is "Inf" or "-Inf" by the sign of x, x/-Inf is a zero
with the opposite sign of x ("-0" for x >= 0). Doubles are written as numbers, the R specials and negative zero as
the strings "NA", "NaN", "Inf", "-Inf", "-0".
"""
from fractions import Fraction
from itertools import combinations

SPECIAL = ('NA', 'NaN', 'Inf', '-Inf')


def _double(text):
    """A fixture fraction -> the exact value of its correctly rounded double."""
    return Fraction(float(Fraction(text)))


def _parse(text):
    return text if text in SPECIAL else _double(text)


def _number(q):
    value = float(q)
    return '-0' if value == 0 and q < 0 else value


def _exact(q):
    return Fraction(float(q)) == q


def _independent(kept):
    return all(_exact(sum(subset, Fraction(0))) for r in range(1, len(kept) + 1)
               for subset in combinations(kept, r))


def _quotient(x, s):
    if x in ('NA', 'NaN'):
        return x
    if s == '-Inf':
        if x == '-Inf':
            return 'NaN'
        return '-0' if x >= 0 else 0.0
    if s == 0:
        if x == '-Inf':
            return '-Inf'
        return 'NaN' if x == 0 else ('Inf' if x > 0 else '-Inf')
    if x == '-Inf':
        return '-Inf' if s > 0 else 'Inf'
    return _number(x / s)


def reference(case):
    nrow, ncol = case['nrow_taxa_array'], case['ncol_taxa_array']
    values = [_parse(v) for v in case['values']]
    weights = [_double(w) for w in case['new_weights']]
    products = []
    for k, v in enumerate(values):
        w = weights[k % ncol]  # recycling: position k gets weight k mod ncol
        products.append(v if v in SPECIAL else Fraction(float(v * w)))
    rows = [products[i * ncol:(i + 1) * ncol] for i in range(nrow)]  # matrix(..., byrow = T)
    sums, sums_exact, counts, independent = [], [], [], []
    for row in rows:
        kept = [x for x in row if x not in ('NA', 'NaN')]
        counts.append(len(kept))
        if '-Inf' in kept:
            sums.append('-Inf')
            sums_exact.append('-Inf')
            independent.append(True)
            continue
        total = sum(kept, Fraction(0))
        sums.append(Fraction(float(total)))
        sums_exact.append(str(total))
        independent.append(_independent(kept))
    result = [[_quotient(x, sums[i]) for x in row] for i, row in enumerate(rows)]
    return {
        'taxa_weights': [float(w) for w in weights],
        'recycled_weight': [k % ncol + 1 for k in range(nrow * ncol)],
        'products': [x if x in SPECIAL else _number(x) for x in products],
        'row_sums': [s if s == '-Inf' else float(s) for s in sums],
        'row_sums_exact': sums_exact,
        'row_counts': counts,
        'rows_accumulator_independent': independent,
        'taxa_appear_core_temp': result,
        'zero_sum_rows': [i + 1 for i, s in enumerate(sums) if s == 0],
        'z': list(range(1, ncol + 1)),
    }
