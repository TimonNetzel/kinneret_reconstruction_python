"""Independent reference for weightedMedian: stated as rules and exact arithmetic, without importing the module.

The NA decision is the closed rule from the module's header, evaluated declaratively over the whole vectors instead of
walking the C loops. Finite values use exact fractions and round each IEEE operation once via float(Fraction). The sort
order is taken from a plain stable sort when x has no ties; for ties, and for +Inf weights or infinite x, only the
hand values in the fixture are used, because reproducing them here would repeat the module's own translation.
"""
from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPECIAL = {'NA': math.nan, 'NaN': math.nan, 'Inf': math.inf, '-Inf': -math.inf}


def parse(text: str) -> float:
    return SPECIAL[text] if text in SPECIAL else float(Fraction(text))


def na_rule(x: list[float], w: list[float]) -> bool:
    """True iff the kernel returns NA_REAL."""
    if any(math.isnan(v) for v in w):
        return True
    n = len(x)
    plus = [i for i in range(n) if w[i] == math.inf]
    if not plus:
        return any(math.isnan(x[i]) and w[i] > 0 for i in range(n)) or not any(w[i] > 0 for i in range(n))
    first = plus[0]
    return (any(math.isnan(x[i]) and 0 < w[i] < math.inf for i in range(first))
            or any(math.isnan(x[i]) and math.isinf(w[i]) for i in range(n)))


def add(a: float, b: float) -> float:
    return float(Fraction(a) + Fraction(b))


def sub(a: float, b: float) -> float:
    return float(Fraction(a) - Fraction(b))


def mul(a: float, b: float) -> float:
    return float(Fraction(a) * Fraction(b))


def div(a: float, b: float) -> float:
    return float(Fraction(a) / Fraction(b))


def interpolated(x: list[float], w: list[float], order: list[int] | None = None) -> dict:
    """Finite x, finite weights: returns value and path; order = None means a stable sort on x."""
    pairs = [(a, b) for a, b in zip(x, w) if b > 0]
    if len(pairs) == 1:
        return {'value': pairs[0][0], 'path': 'single_value', 'half': None}
    total = 0.0
    for _, b in pairs:
        total = add(total, b)
    positions = sorted(range(len(pairs)), key=lambda i: pairs[i][0]) if order is None else list(order)
    xs = [pairs[i][0] for i in positions]
    ws = [pairs[i][1] for i in positions]
    running, cums = 0.0, []
    for k, b in enumerate(ws):
        part = div(b, total)
        running = add(running, part)
        cums.append(sub(running, part / 2))
        if cums[-1] >= 0.5:
            if k == 0:
                return {'value': xs[0], 'path': 'first_sorted_value', 'half': 0}
            step = mul(div(sub(0.5, cums[k]), sub(cums[k], cums[k - 1])), sub(xs[k], xs[k - 1]))
            return {'value': add(step, xs[k]), 'path': 'interpolated', 'half': k}
    raise ValueError('cumulative weight never reached 1/2')


def expected(case: dict) -> dict:
    """Reference outcome of one fixture case: kind, value (float or None)."""
    x, w = [parse(v) for v in case['x']], [parse(v) for v in case['w']]
    if na_rule(x, w):
        return {'kind': 'NA', 'value': None}
    finite = (all(math.isfinite(v) or not case['w'][i] for i, v in enumerate(x)) and all(math.isfinite(v) for v in w)
              and all(math.isfinite(a) for a, b in zip(x, w) if b > 0))
    tied = len({a for a, b in zip(x, w) if b > 0}) < sum(1 for b in w if b > 0)
    if finite and (not tied or case.get('hand_order') is not None):
        out = interpolated(x, w, case.get('hand_order'))
        return {'kind': 'number', 'value': out['value'], 'path': out['path'], 'half': out['half']}
    hand = case['hand']  # +Inf weights or infinite x: only the hand value is independent of the translation
    if hand is None:
        raise ValueError('non-finite case without hand value')
    return {'kind': hand['kind'], 'value': parse(hand['value']) if hand['value'] is not None else None,
            'path': hand['path'], 'half': hand['half']}


def main():
    fixture = json.loads((HERE / 'synthetic_fixture.json').read_text(encoding='utf-8'))
    out = {'kind': 'independent_weighted_median_reference',
           'cases': {name: {**(r := expected(c)), 'value': None if r['value'] is None else
                            (r['value'].hex() if math.isfinite(r['value']) else str(r['value']))}
                     for name, c in fixture['cases'].items()}}
    (HERE / 'reference_expectations.json').write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(len(out['cases']), 'reference cases')


if __name__ == '__main__':
    main()
