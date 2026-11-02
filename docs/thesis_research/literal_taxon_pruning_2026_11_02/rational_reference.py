"""Exact scalar reference for the artificial pruning fixture; no NumPy and no module imports.

Written from mcmc_prepare.R 51-80 directly: ratios as Fractions against 19/20, explicit R colon and
out-of-range NA positions, membership by equality, and column means as exact fractions.
"""
from fractions import Fraction


def r_positions(first, last):
    step = 1 if last >= first else -1
    return list(range(first, last + step, step))


def exact_pruning(pollen, null_ratio_check, metro_hastings, relict_override):
    n_rows, n_taxa = len(pollen), len(pollen[0])
    if not null_ratio_check:
        return {'executed': False, 'branch': 'not_executed', 'metro_hastings': metro_hastings, 'skip_site': False,
                'null_ratio': None, 'removed_taxa': [], 'kept_taxa': list(range(n_taxa)),
                'sorting_available_spectrum': None}
    ratio = []
    for j in range(n_taxa):
        zeros = 0
        for i in range(n_rows):
            if Fraction(str(pollen[i][j])) == 0:
                zeros += 1
        ratio.append(Fraction(zeros, n_rows))
    if relict_override:
        for r_id in (3, 8, 23, 24, 25):
            ratio[r_id - 1] = Fraction(0)
        ratio[0] = Fraction(1)
    rare = [j for j in range(n_taxa) if ratio[j] >= Fraction(19, 20)]
    if n_taxa - len(rare) > 15:
        metro_hastings = True
    if not metro_hastings:
        ordered = sorted(ratio)
        table = [ordered[p - 1] for p in r_positions(16, n_taxa) if 1 <= p <= n_taxa]
        removed = [j for j in range(n_taxa) if ratio[j] in table]
        branch = 'rarest_15'
    else:
        removed = rare
        branch = 'ratio_095'
    kept = [j for j in range(n_taxa) if j not in removed]
    out = {'executed': True, 'branch': branch, 'metro_hastings': metro_hastings,
           'skip_site': len(removed) == n_taxa, 'null_ratio': [str(r) for r in ratio],
           'removed_taxa': removed, 'kept_taxa': kept, 'sorting_available_spectrum': None}
    if not out['skip_site']:
        out['sorting_available_spectrum'] = [
            str(sum((Fraction(str(pollen[i][j])) for i in range(n_rows)), Fraction(0)) / n_rows) for j in kept]
    return out
