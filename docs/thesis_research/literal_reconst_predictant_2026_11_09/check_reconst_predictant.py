"""Run the module on the fixture: synthetic_result.json and arithmetic_check.json. No R, no original data."""
from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import historical_local_cor as lc  # noqa: E402
import historical_reconst_predictant as rp  # noqa: E402

_spec = importlib.util.spec_from_file_location('reconst_reference', HERE / 'precise_reference.py')
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)


def value(text):
    parsed = reference.parse(text)
    if parsed is reference.NA:
        return rp.NA
    return math.nan if parsed is reference.NAN else parsed


def show(v):
    if v is rp.NA:
        return 'NA'
    if math.isnan(v):
        return 'NaN'
    if math.isinf(v):
        return 'Inf' if v > 0 else '-Inf'
    if v == 0:
        return '-0' if math.copysign(1.0, v) < 0 else '0'
    return v.hex()


def matrix(rows):
    return [[value(v) for v in row] for row in rows]


def run_case(case):
    """Each source step in order; a stop in R is reported at the line where it happens."""
    core, tf, mask = matrix(case['core']), matrix(case['tf']), case['mask']
    try:
        r = rp.reconst_predictant(core, mask, tf)
    except rp.UnsupportedSourceDomain as error:
        return {'stop': '1818', 'message': str(error)}
    out = {'path': r.path, 'selected_dim': None if r.selected.dim is None else list(r.selected.dim), 'nrow': r.nrow,
           'tann': [show(v) for v in r.tann], 'pann': [show(v) for v in r.pann]}
    try:
        block = rp.reconstruction_block(taxa_appear_core=core, reconst_ages=mask, tf_sample=tf, tann=case['tann'], recent=case['recent'],
                                        prior_mean_tmp=value(case['prior'][0]), prior_mean_pann=value(case['prior'][1]))
    except rp.UnsupportedSourceDomain as error:
        out.update(stop='1825', message=str(error))
        return out
    out.update(youngest=[show(v) for v in block.youngest_values], post_tann=show(block.post_tann), post_pann=show(block.post_pann),
               current_name=block.current_name, current=[show(v) for v in block.predictant_current])
    return out


def vector(items):
    return [value(v) for v in items]


def run_chain(ch):
    block = rp.reconst_predictant(matrix(ch['core']), ch['mask'], matrix(ch['tf']))
    args = {'predictant_tann': list(block.tann), 'predictant_pann': list(block.pann), 'tann': ch['tann'], 'multiple': ch['multiple'], 'hi_ages_time': ch['hi_ages_time']}
    if ch['multiple']:
        args.update(ages_pollen=ch['ages_pollen'], predictors1=vector(ch['predictors1']), ages_iso=ch['ages_iso'], predictors2=vector(ch['predictors2']))
    else:
        args['sources_marine'] = ch['sources_marine']
        if ch['sources_marine'] == 1:
            args.update(ages_iso1=ch['ages_iso1'], ages_iso2=ch['ages_iso2'], iso_time=vector(ch['iso_time']))
        else:
            args.update(ages_pollen1=ch['ages_pollen1'], ages_pollen2=ch['ages_pollen2'], pollen=vector(ch['pollen']))
    got = lc.time_space_correlations(**args)
    return {'branch': got.branch, 'predictant_read': got.predictant_read, 'post_pollen': got.post_pollen.hex(), 'post_iso': got.post_iso.hex(), 'post_iso_hi': got.post_iso_hi.hex()}


def stress_cases(cfg):
    """Deterministic artificial inputs; only rng.random() is used. A quarter carries NA or NaN, some carry Inf, some select fewer rows."""
    rng = random.Random(cfg['seed'])
    out = []
    for number in range(cfg['cases']):
        nrow, ncol = 1 + int(rng.random() * 6), 1 + int(rng.random() * 7)
        style = number % 5
        draw = {0: lambda: rng.random(), 1: lambda: (rng.random() - 0.5) * 10 ** int(rng.random() * 9), 2: lambda: 1.0 + rng.random() * 2 ** -40,
                3: lambda: 0.1 * int(rng.random() * 5), 4: lambda: rng.random() * 2.0 ** int(rng.random() * 60 - 30)}[style]
        core = [[draw() for _ in range(ncol)] for _ in range(nrow)]
        if style in (0, 3):
            for row in core:
                total = 0.0
                for v in row:
                    total += v
                if total:
                    row[:] = [v / total for v in row]
        tf = [[(rng.random() - 0.25) * 40, rng.random() * 3000] if style != 2 else [1.0 + rng.random() * 2 ** -45, 1.0] for _ in range(ncol)]
        if number % 4 == 1:
            for _ in range(1 + int(rng.random() * 3)):
                core[int(rng.random() * nrow)][int(rng.random() * ncol)] = None if rng.random() < 0.5 else math.nan
            if rng.random() < 0.3:
                tf[int(rng.random() * ncol)][int(rng.random() * 2)] = None if rng.random() < 0.5 else math.nan
        if number % 8 == 3:
            core[int(rng.random() * nrow)][int(rng.random() * ncol)] = math.inf if rng.random() < 0.5 else -math.inf
        mask = [True] * nrow
        if number % 7 == 0:
            mask = [rng.random() < 0.5 for _ in range(nrow)]
        out.append({'core': core, 'mask': mask, 'tf': tf})
    return out


def run_stress(c):
    try:
        r = rp.reconst_predictant(c['core'], c['mask'], c['tf'])
    except rp.UnsupportedSourceDomain:
        return {'stop': '1818'}
    return {'path': r.path, 'tann': [show(v) for v in r.tann], 'pann': [show(v) for v in r.pann]}


def stress_reference(c):
    core = [[reference.NA if v is None else (reference.NAN if isinstance(v, float) and math.isnan(v) else v) for v in row] for row in c['core']]
    tf = [[reference.NA if v is None else (reference.NAN if isinstance(v, float) and math.isnan(v) else v) for v in row] for row in c['tf']]
    rows, shape = reference.select(core, c['mask'])
    columns = reference.product(rows, shape, tf)
    return [reference.token(v) for v in columns[0]], [reference.token(v) for v in columns[1]]


def _x87_round(value: Fraction) -> Fraction:
    if value == 0:
        return value
    sign, magnitude = (1 if value > 0 else -1), abs(value)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    while Fraction(2) ** exponent > magnitude:
        exponent -= 1
    while Fraction(2) ** (exponent + 1) <= magnitude:
        exponent += 1
    unit = Fraction(2) ** (exponent - 63)
    q = magnitude / unit
    whole, rest = q.numerator // q.denominator, q - q.numerator // q.denominator
    if rest > Fraction(1, 2) or (rest == Fraction(1, 2) and whole & 1):
        whole += 1
    return sign * whole * unit


def long_double_variant(row, column):
    """internal_matprod for a finite row: not what R 3.5.0 runs by default."""
    total = Fraction(0)
    for a, b in zip(row, column):
        total = _x87_round(total + Fraction(float(Fraction(a) * Fraction(b))))
    return float(total)


def python_sum_variant(row, column):
    return sum(a * b for a, b in zip(row, column))  # compensated for floats from Python 3.12 on


def exact_sum_variant(row, column):
    return float(sum(Fraction(float(Fraction(a) * Fraction(b))) for a, b in zip(row, column)))


def finite_rows(cases):
    for c in cases:
        if not all(c['mask']):
            continue
        if any(v is None or not math.isfinite(v) for row in c['tf'] for v in row):
            continue
        for row in c['core']:
            if any(v is None or not math.isfinite(v) for v in row):
                continue
            yield c, row


def counterexamples(fixture):
    """Plausible misreadings, each differing from the module on a named case or on stress cases. Not candidate variants."""
    cases = fixture['cases']
    tie = cases['tie_with_nan_in_other_row']
    row, tf = matrix(tie['core'])[0], matrix(tie['tf'])
    column = [t[0] for t in tf]
    got = run_case(tie)
    module_value = float.fromhex(got['tann'][0])
    stress = stress_cases(fixture['stress'])
    finite, long_double_changed, python_changed, exact_changed = 0, 0, 0, 0
    numpy_changed = 0
    import numpy as np
    for c, r in finite_rows(stress):
        for k in range(2):
            col = [t[k] for t in c['tf']]
            total = 0.0
            for a, b in zip(r, col):
                total += a * b
            if not math.isfinite(total):
                continue
            finite += 1
            long_double_changed += long_double_variant(r, col) != total
            python_changed += python_sum_variant(r, col) != total
            exact_changed += exact_sum_variant(r, col) != total
            numpy_changed += float(np.asarray(r) @ np.asarray(col)) != total
    with_missing = [c for c in stress if any(v is None or not math.isfinite(v) for row in c['core'] for v in row)]
    hand = cases['hand_all_rows']
    columnwise = [[Fraction(a) * Fraction(b) for a, b in zip(col, [t[0] for t in hand['tf']])] for col in zip(*hand['core'])]
    na = cases['zero_times_na']
    prelim = run_case(cases['preliminary_holocene_mask'])
    reset = run_case(hand)
    return {
        'long_double_in_three_loop_path': {'module': got['tann'][0], 'variant': long_double_variant(row, column).hex(), 'differs': long_double_variant(row, column) != module_value,
                                           'source': 'R-3.5.0 array.c simple_matprod uses double; the handover said long double',
                                           'stress_finite_values': finite, 'stress_changed': long_double_changed},
        'python_builtin_sum': {'differs': python_changed > 0, 'stress_finite_values': finite, 'stress_changed': python_changed, 'python': sys.version.split()[0]},
        'exact_sum': {'differs': exact_changed > 0, 'stress_finite_values': finite, 'stress_changed': exact_changed},
        'numpy_matmul_not_used': {'stress_finite_values': finite, 'stress_changed': numpy_changed, 'note': 'NumPy BLAS is not the reference BLAS; informative only'},
        'na_treated_as_zero': {'module': run_case(na)['tann'][0], 'variant': (1 * 2.0).hex(), 'differs': run_case(na)['tann'][0] == 'NA'},
        'column_wise_reading': {'module': reset['tann'], 'variant': [str(sum(v)) for v in columnwise], 'differs': len(columnwise) != len(reset['tann'])},
        'line_1466_holocene_mask_taken_as_effective': {'module_rows': reset['nrow'], 'variant_rows': prelim['nrow'], 'differs': prelim['nrow'] != reset['nrow'], 'note': 'line 1522 resets the mask to all rows'},
        'missing_value_rows_in_stress': len(with_missing),
    }


def main():
    fixture = json.loads((HERE / 'synthetic_fixture.json').read_text(encoding='utf-8'))
    expected = json.loads((HERE / 'reference_expectations.json').read_text(encoding='utf-8'))['cases']
    cases = {}
    for name, case in fixture['cases'].items():
        got = run_case(case)
        ref = expected[name]
        got['matches_reference'] = got.get('stop') == ref.get('stop') and got.get('tann', ref.get('tann')) == ref.get('tann') and got.get('pann', ref.get('pann')) == ref.get('pann')
        got['tann_decimal'] = [None if t in ('NA', 'NaN', 'Inf', '-Inf') else float.fromhex(t) if t not in ('0', '-0') else 0.0 for t in got.get('tann', [])]
        got['pann_decimal'] = [None if t in ('NA', 'NaN', 'Inf', '-Inf') else float.fromhex(t) if t not in ('0', '-0') else 0.0 for t in got.get('pann', [])]
        cases[name] = got
    chains = {name: run_chain(ch) for name, ch in fixture['chains'].items()}
    stress = [run_stress(c) for c in stress_cases(fixture['stress'])]
    (HERE / 'synthetic_result.json').write_text(json.dumps({'kind': 'module_on_synthetic_fixture_not_historical', 'cases': cases, 'chains': chains, 'stress': stress},
                                                           indent=1, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    arithmetic = counterexamples(fixture)
    (HERE / 'arithmetic_check.json').write_text(json.dumps({'kind': 'counterexamples_not_candidate_variants', 'checks': arithmetic}, indent=1) + '\n', encoding='utf-8', newline='\n')
    bad = [k for k, v in cases.items() if not v['matches_reference']]
    print(len(cases), 'cases;', len(bad), 'differ from the reference', bad)
    print({k: (v['differs'] if isinstance(v, dict) and 'differs' in v else v) for k, v in arithmetic.items()})
    print({k: v.get('stress_changed') for k, v in arithmetic.items() if isinstance(v, dict) and 'stress_changed' in v}, arithmetic['long_double_in_three_loop_path']['stress_finite_values'])
    print(sum(1 for s in stress if s.get('path') == 'simple_matprod'), 'stress cases on simple_matprod')


if __name__ == '__main__':
    main()
