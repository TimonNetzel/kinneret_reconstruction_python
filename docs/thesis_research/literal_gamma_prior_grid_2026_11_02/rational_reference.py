"""Independent reference for the gamma grid: no NumPy and no import of the module under test.

The full scan uses plain Python floats (IEEE binary64, one rounding per operation). Every rounding the result
depends on is then re-derived from exact fractions: grid values as RN(from + RN(k * by)), quotients, the
square root (bracketed by the squares of its rounding midpoints) and the difference, around the selected
index and at both grid ends.
"""
import math
from fractions import Fraction

FROM, BY, FUZZ = 0.1, 0.1, 1e-10


def rnd(q: Fraction) -> float:
    """Round an exact rational to the nearest double, ties to even."""
    return float(q)


def r_seq(to: float, *, fuzz: bool = True, clamp: bool = True, fused: bool = False) -> list:
    """seq(0.1, to, by = 0.1) as written in seq.default; the switches give the sensitivity variants."""
    delta = rnd(Fraction(to) - Fraction(FROM))
    n = rnd(Fraction(delta) / Fraction(BY))
    count = math.floor(rnd(Fraction(n) + Fraction(FUZZ)) if fuzz else n)
    if fused:
        values = [rnd(k * Fraction(BY) + Fraction(FROM)) for k in range(count + 1)]
    else:
        values = [FROM + k * BY for k in range(count + 1)]
    return [min(v, to) for v in values] if clamp else values


def _sqrt_is_correctly_rounded(x: float, y: float) -> bool:
    lower = (Fraction(math.nextafter(y, 0.0)) + Fraction(y)) / 2
    upper = (Fraction(y) + Fraction(math.nextafter(y, math.inf))) / 2
    return lower * lower <= Fraction(x) <= upper * upper


def _exact_checks(grid, k, mean, sd, to, two_step_grid):
    x = grid[k]
    two_step = rnd(Fraction(FROM) + Fraction(rnd(k * Fraction(BY))))
    root = math.sqrt(x)
    rate1, rate2 = x / mean, root / sd
    return ((x == min(two_step, to) or not two_step_grid) and _sqrt_is_correctly_rounded(x, root)
            and rate1 == rnd(Fraction(x) / Fraction(mean)) and rate2 == rnd(Fraction(root) / Fraction(sd))
            and rate2 - rate1 == rnd(Fraction(rate2) - Fraction(rate1)))


def grid_reference(mean, sd, upper, grid=None, *, two_step_grid: bool = True) -> dict:
    """``two_step_grid=False`` only for the sensitivity variants, whose values are not seq.default's."""
    mean, sd, upper = float(mean), float(sd), float(upper)
    grid = grid if grid is not None else r_seq(upper)
    best_index, best = 0, None
    for index, s in enumerate(grid):
        value = abs((math.sqrt(s) / sd - s / mean) - 0)
        if best is None or value < best:  # strict: which.min keeps the first minimum
            best, best_index = value, index
    window = sorted({0, len(grid) - 1, *range(max(0, best_index - 3), min(len(grid), best_index + 4))})
    if not all(_exact_checks(grid, k, mean, sd, upper, two_step_grid) for k in window):
        raise AssertionError('a float result disagrees with its exact rounding')
    x = grid[best_index]
    return {
        'grid_length': len(grid), 'index': best_index, 'r_index': best_index + 1,
        'shape': x, 'shape_hex': x.hex(), 'shape_exact': str(Fraction(x)),
        'rate': math.sqrt(x) / sd, 'rate_expectation': x / mean, 'objective': best,
        'neighbour_objectives': {str(k): abs(math.sqrt(grid[k]) / sd - grid[k] / mean) for k in window},
        'closed_form_shape_exact': str((Fraction(mean) / Fraction(sd)) ** 2),
        'exactly_checked_indices': window,
    }


def chain_reference(mean, sd_pann, sd_tmp, *, recent, min_age_at_least_2000, grid=None) -> dict:
    grid = grid if grid is not None else r_seq(50000.0)
    default = grid_reference(mean, sd_pann, 50000.0, grid)
    doubled = recent and min_age_at_least_2000
    sd_pann_final = sd_pann * 2 if doubled else sd_pann
    final = grid_reference(mean, sd_pann_final, 50000.0, grid)
    return {
        'default_index': default['index'], 'default_shape': default['shape'], 'default_rate': default['rate'],
        'diff_rate': default['shape'] / 10, 'variance_doubled': doubled,
        'sd_pann': float(sd_pann_final), 'sd_tmp': float(sd_tmp * 2 if doubled else sd_tmp),
        'final_index': final['index'], 'final_shape': final['shape'], 'final_rate': final['rate'],
        'rate_after_recent_start': final['shape'] / mean if recent else final['rate'],
    }
