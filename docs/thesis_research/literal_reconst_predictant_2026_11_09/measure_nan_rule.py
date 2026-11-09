"""Fit the four candidate NA/NaN rules to a measurement in R 3.5.0; write nan_rule_measurement.json. Usage: measure_nan_rule.py <Rscript.exe>"""
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
LOCAL = ROOT / '.local/reconst_predictant_2026_11_09'
NA, NAN = 'NA', 'NaN'
VALUES = {'one': 1.0, 'NA_': NA, 'NaN_': NAN, 'Inf_': math.inf, 'MInf': -math.inf, 'Zero': 0.0}


def missing(v):
    return isinstance(v, str)


def combine(a, b, a_first, operate):
    if missing(a) and missing(b):
        return a if a_first else b
    if missing(a):
        return a
    if missing(b):
        return b
    result = operate(a, b)
    return NAN if math.isnan(result) else result


def token(v):
    if isinstance(v, str):
        return v
    if math.isinf(v):
        return 'Inf' if v > 0 else '-Inf'
    return v.hex()


def main():
    rscript = str(Path(sys.argv[1]).resolve())
    LOCAL.mkdir(parents=True, exist_ok=True)
    subprocess.run([rscript, '--vanilla', str(OUT / 'measure_nan_rule.R'), str(LOCAL / 'nan_rule.tsv')], check=True, stdin=subprocess.DEVNULL)
    lines = [line.split('\t') for line in (LOCAL / 'nan_rule.tsv').read_text(encoding='utf-8').splitlines()]
    extra, grid = {r[0]: r[1] for r in lines if len(r) == 2}, [r for r in lines if len(r) == 5]
    fits = {}
    for x_first, sum_first in itertools.product((True, False), (True, False)):
        bad = 0
        for a, b, c, d, got in grid:
            total = 0.0
            for x, y in ((VALUES[a], VALUES[c]), (VALUES[b], VALUES[d])):
                total = combine(total, combine(x, y, x_first, lambda p, q: p * q), sum_first, lambda p, q: p + q)
            expected = got if got in ('NA', 'NaN', 'Inf', '-Inf') else float.fromhex(got).hex()
            bad += token(total) != expected
        fits[f'{"x" if x_first else "y"}_first_in_product__{"sum" if sum_first else "product"}_first_in_addition'] = bad
    report = {'kind': 'measurement_in_R_3_5_0_not_a_derivation', 'combinations': len(grid), 'mismatches_per_rule': fits,
              'three_loop_accumulator': {'tie_with_nan_in_other_row': extra['tie_with_nan_in_other_row'], 'tie_finite': extra['tie_finite'], 'one_means': 'double'},
              'historical_reproduction_ready': False}
    (OUT / 'nan_rule_measurement.json').write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(report)


if __name__ == '__main__':
    main()
