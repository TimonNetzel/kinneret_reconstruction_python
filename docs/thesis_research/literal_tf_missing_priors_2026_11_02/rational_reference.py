"""Independent reference for mcmc_prepare.R 1664-1685: plain lists, integers and Fraction; no NumPy, no module import.

Rounding is derived from integer division: n = 4q + r, so n/4 = q + r/4 and only r = 2 is a tie, which goes to the
even one of q and q + 1. Each double (1/n, n/2, stale phi1/phi2) is the correctly rounded value of the exact
fraction. null stands for NA/NaN in the fixture and stays null here.
"""
from fractions import Fraction


def round_even_quarter(n):
    """round(n/4) for an integer n >= 0, ties to the even digit, via divmod only."""
    q, r = divmod(n, 4)
    if r == 2:
        return q if q % 2 == 0 else q + 1
    return q + (r == 3)


def round_half_up_quarter(n):
    """Counterexample reading: floor(n/4 + 1/2)."""
    return (n + 2) // 4


def reference(case):
    mask, n_taxa = case['x1_mean_is_na'], len(case['taxa'])
    which = [i + 1 for i, missing in enumerate(mask) if missing]
    if which:
        keep = [i for i, missing in enumerate(mask) if not missing]
        line = 1678
    else:
        keep = list(range(n_taxa))
        line = 1435
    cols = lambda m: [[row[i] for i in keep] for row in m]
    rows = lambda m: [m[i] for i in keep]
    vec = lambda v: None if v is None else [v[i] for i in keep]
    n = len(keep)
    return {
        'which_tf_missing': which, 'removal_applied': bool(which), 'kept': keep,
        'taxa': [case['taxa'][i] for i in keep],
        'all_avail': cols(case['all_avail']), 'all_avail_age': cols(case['all_avail_age']),
        'taxa_appear_core': cols(case['taxa_appear_core']),
        'tf_x1': rows(case['tf_x1']), 'tf_x2': rows(case['tf_x2']),
        'x1_mean': vec(case['x1_mean']), 'x2_mean': vec(case['x2_mean']),
        'x1_mean_change': vec(case['x1_mean_change']) if case['taxa_change'] else None,
        'x2_mean_change': vec(case['x2_mean_change']) if case['taxa_change'] else None,
        'num_taxa': n, 'num_taxa_line': line,
        'which_taxa': list(range(1, n + 1)),
        'taxa_weights': [float(Fraction(1, n))] * n, 'taxa_weights_exact': f'1/{n}',
        'phi': float(Fraction(n, 2)), 'nr_taxa_mean': n,
        'nr_taxa_sd': round_even_quarter(n), 'nr_taxa_sd_is_tie': n % 4 == 2,
        'nr_taxa_sd_half_up': round_half_up_quarter(n),
        'stale_phi1': float(Fraction(n_taxa, 2)), 'stale_phi2': float(Fraction(n_taxa - 2, 2)),
        'current_phi1': float(Fraction(n, 2)), 'current_phi2': float(Fraction(n - 2, 2)),
    }
