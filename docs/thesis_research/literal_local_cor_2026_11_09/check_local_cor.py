"""Run the module on the fixture: synthetic_result.json and arithmetic_check.json. No R, no original data."""
from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import historical_local_cor as lc  # noqa: E402

_spec = importlib.util.spec_from_file_location('local_cor_reference', HERE / 'precise_reference.py')
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)


def value(text):
    parsed = reference.parse(text)
    return None if parsed is reference.NA else parsed


def vector(items):
    return [value(v) for v in items]


def show(result):
    return result.kind if result.kind != 'number' else result.value.hex()


def run_cor(case):
    return lc.local_cor(vector(case['x']), vector(case['y']), na_rm=case['na_rm'])


def run_reader(r):
    args = {'predictant_tann': vector(r['predictant_tann']), 'predictant_pann': vector(r['predictant_pann']), 'tann': r['tann'], 'multiple': r['multiple'],
            'hi_ages_time': r['hi_ages_time']}
    if r['multiple']:
        args.update(ages_pollen=r['ages_pollen'], predictors1=vector(r['predictors1']), ages_iso=r['ages_iso'], predictors2=vector(r['predictors2']))
    else:
        args['sources_marine'] = r['sources_marine']
        if r['sources_marine'] == 1:
            args.update(ages_iso1=r['ages_iso1'], ages_iso2=r['ages_iso2'], iso_time=vector(r['iso_time']))
        else:
            args.update(ages_pollen1=r['ages_pollen1'], ages_pollen2=r['ages_pollen2'], pollen=vector(r['pollen']))
    return lc.time_space_correlations(**args)


def record_reader(got):
    return {'branch': got.branch, 'predictant_read': got.predictant_read, 'post_pollen': got.post_pollen.hex(), 'post_iso': got.post_iso.hex(),
            'post_iso_hi': got.post_iso_hi.hex(),
            'pollen_replaced': None if got.pollen is None else got.pollen.replaced, 'iso_replaced': None if got.iso is None else got.iso.replaced}


def stress_cases(cfg):
    """Deterministic artificial inputs; only rng.random() is used. A quarter carries NA or NaN, positions and tokens vary."""
    rng = random.Random(cfg['seed'])
    out = []
    for number in range(cfg['cases']):
        n = 2 + int(rng.random() * 39)
        style = number % 4
        draw = {0: lambda: rng.random(), 1: lambda: (rng.random() - 0.5) * 10 ** int(rng.random() * 9),
                2: lambda: 1.0 + rng.random() * 2 ** -40, 3: lambda: 0.1 * int(rng.random() * 5)}[style]
        x, y = [draw() for _ in range(n)], [draw() for _ in range(n)]
        if number % 4 == 1:
            for vec in (x, y):
                for _ in range(1 + int(rng.random() * 3)):
                    vec[int(rng.random() * n)] = None if rng.random() < 0.5 else math.nan
        out.append({'x': x, 'y': y, 'na_rm': number % 3 == 0})
    return out


def run_stress(c):
    return lc.local_cor(c['x'], c['y'], na_rm=c['na_rm'])


def plain_sum(values):
    """A plain double accumulation. Not Python's built-in sum, which is compensated for floats from 3.12 on."""
    total = 0.0
    for v in values:
        total += v
    return total


def double_sum_variant(x, y):
    """The same formula with every sum in plain double arithmetic: what NumPy-style code would do."""
    n = len(y)
    mean_x, mean_y = plain_sum(x) / n, plain_sum(y) / n
    dx, dy = [v - mean_x for v in x], [v - mean_y for v in y]
    total = 0.0
    for a, b in zip(dx, dy):
        total += b * a
    denominator = math.sqrt(plain_sum([v * v for v in dx]) / (n - 1)) * math.sqrt(plain_sum([v * v for v in dy]) / (n - 1))
    return (total / (n - 1)) / denominator if denominator else math.nan


def counterexamples(fixture):
    """Plausible misreadings, each differing from the module on a named case. Not candidate variants."""
    named = fixture['cases']['long_double_mean']
    x, y = vector(named['x']), vector(named['y'])
    plain_mean = plain_sum(x) / len(x)
    module_mean = run_cor(named).mean_x
    finite = [c for c in stress_cases(fixture['stress']) if not any(v is None or math.isnan(v) for v in c['x'] + c['y'])]
    compared = [(c, run_stress(c)) for c in finite]
    compared = [(c, r) for c, r in compared if r.kind == 'number']
    changed = sum(1 for c, r in compared if double_sum_variant(c['x'], c['y']) != r.value)
    y_na = fixture['cases']['y_na_stays_with_na_rm']
    pairs = [(a, b) for a, b in zip(vector(y_na['x']), vector(y_na['y'])) if a is not None and b is not None]
    both_filtered = lc.local_cor([a for a, _ in pairs], [b for _, b in pairs], na_rm=True)
    negative = lc.post_treatment(run_cor(fixture['cases']['perfect_negative']))
    reader = fixture['readers']['multiple_pann']
    got = run_reader(reader)
    wrong_x = lc.logical_index(vector(reader['predictant_pann']), reader['ages_iso'])
    wrong = lc.post_treatment(lc.local_cor(wrong_x, vector(reader['predictors2']), na_rm=True))
    return {'double_sum_for_the_mean': {'module': module_mean.hex(), 'double_sum': plain_mean.hex(), 'differs': plain_mean != module_mean, 'stress_compared': len(compared), 'stress_changed': changed},
            'filter_y_as_well': {'module_kind': run_cor(y_na).kind, 'variant_kind': both_filtered.kind, 'differs': run_cor(y_na).kind != both_filtered.kind},
            'square_before_replacing': {'module': negative.value.hex(), 'variant': (1.0).hex(), 'differs': negative.value != 1.0},
            'line_1966_reads_current_predictant': {'module': got.iso.value.hex(), 'variant': wrong.value.hex(), 'differs': got.iso.value != wrong.value}}


def main():
    fixture = json.loads((HERE / 'synthetic_fixture.json').read_text(encoding='utf-8'))
    expected = json.loads((HERE / 'reference_expectations.json').read_text(encoding='utf-8'))['cases']
    cases = {}
    for name, case in fixture['cases'].items():
        got = run_cor(case)
        ref = expected[name]
        cases[name] = {'kind': got.kind, 'value': show(got), 'n': got.n, 'matches_reference': got.kind == ref['kind'] and (got.kind != 'number' or got.value.hex() == ref['value']),
                       'decimal': None if got.kind != 'number' else got.value}
    readers = {name: record_reader(run_reader(r)) for name, r in fixture['readers'].items()}
    stress = [show(run_stress(c)) for c in stress_cases(fixture['stress'])]
    (HERE / 'synthetic_result.json').write_text(json.dumps({'kind': 'module_on_synthetic_fixture_not_historical', 'cases': cases, 'readers': readers, 'stress': stress},
                                                           indent=1, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    arithmetic = counterexamples(fixture)
    (HERE / 'arithmetic_check.json').write_text(json.dumps({'kind': 'counterexamples_not_candidate_variants', 'checks': arithmetic}, indent=1) + '\n', encoding='utf-8', newline='\n')
    bad = [k for k, v in cases.items() if not v['matches_reference']]
    print(len(cases), 'cases;', len(bad), 'differ from the reference', bad)
    print({k: v['differs'] for k, v in arithmetic.items()})


if __name__ == '__main__':
    main()
