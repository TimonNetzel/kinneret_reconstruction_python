"""Independent reference for mcmc_conduction.R 53-79: plain lists, integers and Fraction; no NumPy, no module import.

Every stored double becomes an exact Fraction. A quotient is exact first and rounded once to the nearest double
(float(Fraction) rounds correctly), which is what one IEEE division does. At a zero column maximum the sign of the
numerator decides: 0/0 is "NaN", a negative value gives "-Inf". NA (null) stays "NA". A column mean is the exact
sum of the non-NA values over their count; it is called accumulator independent when every running sum and the
mean itself are exact doubles. Doubles are written as numbers, the R specials as the strings "NA", "NaN", "-Inf".
"""
from fractions import Fraction


def _exact_double(q):
    return Fraction(float(q)) == q


def _quotient(x, m):
    if x is None:
        return 'NA', 'NA'
    if m == 0:
        return ('NaN', 'NaN') if x == 0 else ('-Inf', '-Inf')  # a positive value cannot have maximum 0
    q = Fraction(x) / Fraction(m)
    return str(q), float(q)


def reference(case):
    rows = case['taxa_appear_core']
    nrow, ncol = len(rows), len(rows[0])
    columns = [[rows[i][j] for i in range(nrow)] for j in range(ncol)]
    maxima = []
    for column in columns:
        best = None
        for v in column:
            if v is not None and (best is None or v > best):
                best = v
        maxima.append(best)
    exact, doubles, raw = [], [], []
    for i in range(nrow):  # c(t(.)): row after row
        for j in range(ncol):
            text, value = _quotient(rows[i][j], maxima[j])
            exact.append(text)
            doubles.append(value)
            raw.append('NA' if rows[i][j] is None else float(rows[i][j]))
    means, means_exact, independent = [], [], []
    for column in columns:
        total, count, sums_exact = Fraction(0), 0, True
        for v in column:
            if v is not None:
                total += Fraction(v)
                count += 1
                sums_exact = sums_exact and _exact_double(total)
        if count == 0:
            means.append('NaN')
            means_exact.append('NaN')
            independent.append(True)
            continue
        mean = total / count
        means.append(float(mean))
        means_exact.append(str(mean))
        independent.append(sums_exact and _exact_double(mean))
    n, gibbs = case['num_taxa'], case['gibbs']
    return {
        'how_many_taxa': n // 2 if gibbs else n, 'how_many_taxa_line': 54 if gibbs else 57,
        'parameter_stage': 'parameter_stage_gibbs' if gibbs else 'parameter_stage_weights',
        'nrow_taxa_array': nrow, 'ncol_taxa_array': ncol,
        'values_line64': raw, 'values': doubles, 'values_exact': exact,
        'column_max': [float(m) for m in maxima],
        'taxa_means': means, 'taxa_means_exact': means_exact,
        'taxa_means_accumulator_independent': independent,
        'jeffreys_prior': [float(Fraction(1, 2))] * n,
    }
