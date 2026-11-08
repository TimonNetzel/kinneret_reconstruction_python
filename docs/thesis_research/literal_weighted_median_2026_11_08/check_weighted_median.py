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
import historical_weighted_median as wm  # noqa: E402

_spec = importlib.util.spec_from_file_location('wm_reference', HERE / 'precise_reference.py')
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)


def input_value(text):
    return None if text == 'NA' else reference.parse(text)


def show(value, kind):
    if kind != 'number':
        return kind
    if math.isinf(value):
        return 'Inf' if value > 0 else '-Inf'
    return value.hex()


def as_record(m):
    return {'value': show(m.value, m.kind), 'kind': m.kind, 'is_na': m.is_na, 'path': m.path, 'n_used': m.n_used,
            'half': m.half, 'decimal': None if m.kind != 'number' or math.isinf(m.value) else m.value}


def run_case(case):
    return wm.weighted_median([input_value(v) for v in case['x']], [input_value(v) for v in case['w']])


def stress_cases(cfg):
    """Deterministic artificial inputs: only rng.random() is used, so the sequence is stable across Python versions."""
    rng = random.Random(cfg['seed'])
    out = []
    for number in range(cfg['cases']):
        n = 1 + int(rng.random() * cfg['max_n'])
        style = number % 5
        if style == 0:  # distinct-ish grid, positive weights
            x = [-20.0 + 0.25 * i + rng.random() for i in range(n)]
            w = [rng.random() for _ in range(n)]
        elif style == 1:  # many ties
            x = [float(int(rng.random() * 6)) for _ in range(n)]
            w = [rng.random() for _ in range(n)]
        elif style == 2:  # zeros and negatives
            x = [rng.random() * 100 for _ in range(n)]
            w = [0.0 if rng.random() < 0.5 else (rng.random() - 0.3) for _ in range(n)]
        elif style == 3:  # NA sprinkled in x and w
            x = [None if rng.random() < 0.04 else rng.random() for _ in range(n)]
            w = [None if rng.random() < 0.02 else rng.random() for _ in range(n)]
        else:  # infinite weights
            x = [rng.random() for _ in range(n)]
            w = [math.inf if rng.random() < 0.3 else rng.random() for _ in range(n)]
        out.append((x, w))
    return out


def naive_stable_sort(x, w):
    order = sorted(range(len(x)), key=lambda i: x[i])
    return reference.interpolated([x[i] for i in order], [w[i] for i in order])['value']


def main():
    fixture = json.loads((HERE / 'synthetic_fixture.json').read_text(encoding='utf-8'))
    results = {name: as_record(run_case(c)) for name, c in fixture['cases'].items()}
    tf = fixture['tf_case']
    x1 = wm.tf_row_medians([input_value(v) for v in map(str, tf['x1range'])],
                           [[input_value(str(v)) for v in r] for r in tf['tf_x1']])
    x2 = wm.tf_row_medians([input_value(v) for v in map(str, tf['x2range'])],
                           [[input_value(str(v)) for v in r] for r in tf['tf_x2']])
    tf_result = {'x1': [as_record(m) for m in x1], 'x2': [as_record(m) for m in x2],
                 'which_tf_missing': list(wm.which_tf_missing(x1))}
    undefined = {}
    for name, c in fixture['undefined_cases'].items():
        try:
            run_case(c)
            undefined[name] = 'accepted'
        except wm.UnsupportedSourceDomain as error:
            undefined[name] = f'rejected: {error}'
    stress = stress_cases(fixture['stress'])
    stress_out = [as_record(wm.weighted_median(x, w)) for x, w in stress]
    paths = {}
    for r in stress_out:
        paths[r['path']] = paths.get(r['path'], 0) + 1
    (HERE / 'synthetic_result.json').write_text(json.dumps({
        'kind': 'module_output_on_synthetic_fixture', 'historical_reproduction_ready': False,
        'cases': results, 'tf_case': tf_result, 'undefined_cases': undefined,
        'stress': {'values': [r['value'] for r in stress_out], 'paths': dict(sorted(paths.items()))},
    }, indent=1) + '\n', encoding='utf-8', newline='\n')

    # counterexamples: readings that a casual reimplementation would use instead of the source
    counter, hand_ulps = {}, {}
    for name, c in fixture['cases'].items():
        got = run_case(c)
        hand = c['hand']
        if hand is not None and hand['kind'] == 'number' and math.isfinite(reference.parse(hand['value'])):
            hand_ulps[name] = abs(int(Fraction(got.value) / Fraction(math.ulp(got.value)))
                                  - int(Fraction(reference.parse(hand['value'])) / Fraction(math.ulp(got.value))))
    ties = fixture['cases']['ties_trace']
    x, w = [float(v) for v in ties['x']], [float(v) for v in ties['w']]
    counter['ties_trace_stable_sort'] = {'module': run_case(ties).value, 'stable_sort': naive_stable_sort(x, w)}
    odd = fixture['cases']['inf_odd_is_not_the_median']
    counter['inf_odd_true_median'] = {'module': run_case(odd).value, 'true_median': 2.0}
    masks = {}
    rows = [[input_value(str(v)) for v in r] for r in tf['tf_x1']]
    masks['sum_is_zero_only'] = [i + 1 for i, r in enumerate(rows)
                                 if sum(v for v in r if v is not None and not math.isnan(v)) == 0
                                 and all(v is not None and not math.isnan(v) for v in r)]
    masks['treat_na_as_zero_weight'] = [i + 1 for i, r in enumerate(rows)
                                        if not any(v is not None and not math.isnan(v) and v > 0 for v in r)]
    masks['x1_or_x2'] = sorted({i + 1 for i, m in enumerate(x1) if m.is_na} | {i + 1 for i, m in enumerate(x2) if m.is_na})
    masks['source_1664'] = list(wm.which_tf_missing(x1))
    x_grid = [float(v) for v in tf['x1range']]
    first_row = [float(v) for v in tf['tf_x1'][0]]
    total = sum(first_row)
    counter['expectation_line_1623_is_overwritten'] = {
        'expectation': sum(a * b / total for a, b in zip(x_grid, first_row)), 'weighted_median': x1[0].value}
    check = {
        'kind': 'synthetic_fragment_check_not_historical_reproduction', 'historical_reproduction_ready': False,
        'final_variant_selected': False, 'R_execution': False,
        'hand_ulps': hand_ulps, 'maximum_hand_ulps': max(hand_ulps.values()),
        'naive_counterexamples': counter, 'masks_on_tf_case': masks,
        'masks_differ_from_source': {k: v != masks['source_1664'] for k, v in masks.items() if k != 'source_1664'},
        'undefined_cases': undefined,
    }
    (HERE / 'arithmetic_check.json').write_text(json.dumps(check, indent=1) + '\n', encoding='utf-8', newline='\n')
    print('cases', len(results), 'stress', len(stress_out), paths)
    print('masks', masks)
    print('counter', counter, 'hand max ulps', check['maximum_hand_ulps'])
    print('undefined', undefined)


if __name__ == '__main__':
    main()
