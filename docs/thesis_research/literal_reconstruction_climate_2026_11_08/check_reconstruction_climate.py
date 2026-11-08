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
import historical_reconstruction_climate as rc  # noqa: E402

_spec = importlib.util.spec_from_file_location('climate_reference', HERE / 'precise_reference.py')
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)


def value(text):
    if text == 'NA':
        return None
    if text == 'bad':
        return 'bad'
    return reference.parse(text)


def matrix(rows):
    return [[value(v) for v in r] for r in rows]


def show(v, kind):
    return kind if kind != 'number' else v.hex()


def run(case, fixture):
    return rc.reconstruction_climate(matrix(case['core']), matrix(case['tf_x1']), matrix(case['tf_x2']), fixture['x1range'],
                                     fixture['x2range'], recent=case['recent'], prior_mean_tmp=reference.parse(fixture['prior']['mean_tmp']),
                                     prior_mean_pann=reference.parse(fixture['prior']['mean_pann']))


def record(got):
    weights = lambda w: None if w is None else ['NaN' if math.isnan(v) else v.hex() for v in w]
    return {'tann': show(got.tann, got.tann_kind), 'pann': show(got.pann, got.pann_kind), 'tann_kind': got.tann_kind,
            'pann_kind': got.pann_kind, 'source': got.source, 'tann_weights': weights(got.tann_weights),
            'pann_weights': weights(got.pann_weights),
            'tann_decimal': None if got.tann_kind != 'number' else got.tann, 'pann_decimal': None if got.pann_kind != 'number' else got.pann}


def stress_cases(cfg):
    """Deterministic artificial inputs; only rng.random() is used. Grids are strictly increasing, so no ties in x."""
    rng = random.Random(cfg['seed'])
    out = []
    for number in range(cfg['cases']):
        taxa, c1, c2 = 2 + int(rng.random() * 11), 3 + int(rng.random() * 23), 3 + int(rng.random() * 23)
        core = [[0.0 if rng.random() < 0.2 else rng.random() for _ in range(taxa)] for _ in range(1 + int(rng.random() * 4))]
        tf1 = [[0.0 if rng.random() < 0.3 else rng.random() for _ in range(c1)] for _ in range(taxa)]
        tf2 = [[0.0 if rng.random() < 0.3 else rng.random() * 9 for _ in range(c2)] for _ in range(taxa)]
        if number % 7 == 3:
            tf1[int(rng.random() * taxa)][int(rng.random() * c1)] = None
        if number % 11 == 5:
            core[0][int(rng.random() * taxa)] = None
        if number % 13 == 6:
            core[0] = [0.0] * taxa
        grid = lambda n, base: [base + i * 0.75 + rng.random() * 0.5 for i in range(n)]
        out.append({'core': core, 'tf_x1': tf1, 'tf_x2': tf2, 'x1range': grid(c1, -15.0), 'x2range': grid(c2, 50.0),
                    'recent': number % 10 != 9, 'prior_tmp': 5.0 + rng.random() * 10, 'prior_pann': 200.0 + rng.random() * 500})
    return out


def run_stress(case):
    return rc.reconstruction_climate(case['core'], case['tf_x1'], case['tf_x2'], case['x1range'], case['x2range'], recent=case['recent'],
                                     prior_mean_tmp=case['prior_tmp'], prior_mean_pann=case['prior_pann'])


def main():
    fixture = json.loads((HERE / 'synthetic_fixture.json').read_text(encoding='utf-8'))
    results = {name: record(run(c, fixture)) for name, c in fixture['cases'].items()}
    stress = [record(run_stress(c)) for c in stress_cases(fixture['stress'])]
    (HERE / 'synthetic_result.json').write_text(json.dumps({
        'kind': 'module_output_on_synthetic_fixture', 'historical_reproduction_ready': False, 'cases': results,
        'stress': {'tann': [r['tann'] for r in stress], 'pann': [r['pann'] for r in stress],
                   'kinds': sorted({r['tann_kind'] for r in stress} | {r['pann_kind'] for r in stress}),
                   'prior_cases': sum(r['source'] == 'prior_recent' for r in stress)},
    }, indent=1) + '\n', encoding='utf-8', newline='\n')

    base = fixture['cases']['recent_basic']
    got = run(base, fixture)
    first, second = [1.0, 3.0], [9.0, 9.0]
    mean_row = [(a + b) / 2 for a, b in zip(first, second)]
    row_mean = rc.weighted_median(fixture['x1range'], rc.row_times_matrix(mean_row, matrix(base['tf_x1']))).value
    weights = list(got.tann_weights)
    expectation = sum(x * w for x, w in zip(fixture['x1range'], weights)) / sum(weights)
    skip_zero = run(dict(fixture['cases']['na_tf_at_zero_share'], tf_x1=[['0', '0', '0', '0'], base['tf_x1'][1]]), fixture)
    check = {
        'kind': 'synthetic_fragment_check_not_historical_reproduction', 'historical_reproduction_ready': False, 'final_variant_selected': False,
        'R_execution': False,
        'naive_counterexamples': {
            'row_mean_instead_of_first_row': {'module': got.tann, 'row_mean': row_mean, 'differs': got.tann != row_mean},
            'expectation_instead_of_median': {'module': got.tann, 'expectation': expectation, 'differs': got.tann != expectation},
            'skip_na_at_zero_share': {'module_kind': results['na_tf_at_zero_share']['tann_kind'], 'skipping_kind': skip_zero.tann_kind,
                                      'differs': results['na_tf_at_zero_share']['tann_kind'] != skip_zero.tann_kind},
        },
    }
    (HERE / 'arithmetic_check.json').write_text(json.dumps(check, indent=1) + '\n', encoding='utf-8', newline='\n')
    print({k: (v['tann'], v['pann']) for k, v in results.items() if k != 'inexact_shares'})
    print(check['naive_counterexamples'])
    print('stress', json.dumps({k: v for k, v in {'kinds': sorted({r['tann_kind'] for r in stress}), 'prior': sum(r['source'] == 'prior_recent' for r in stress)}.items()}))


if __name__ == '__main__':
    main()
