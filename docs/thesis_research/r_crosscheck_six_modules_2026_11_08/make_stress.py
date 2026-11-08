"""Write stress_fixture.json: deterministic artificial inputs with inexact doubles for the six modules.

The earlier fixtures use round numbers, which hide accumulation differences. Only rng.random() is used, values are
stored as C99 hex. Nothing here is a pollen count, an age or a transfer function of any site.
"""
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent
SEED, PER_MODULE = 20261108, 25


def h(v):
    return float(v).hex()


def age_depth(r):
    n_age, n_grid, n_taxa = 3 + int(r.random() * 8), 8 + int(r.random() * 12), 2 + int(r.random() * 6)
    offsets = [0] + sorted({1 + int(r.random() * (n_grid - 1)) for _ in range(5)})
    rho = [[0.0 if r.random() < 0.1 else r.random() * 50 for _ in range(n_grid)] for _ in range(n_age)]
    for row in rho:
        row[0] += 0.125 + r.random()  # every selected row keeps positive mass
    age = [-50.0 + 0.5 * int(r.random() * 40) for _ in range(n_age)]
    pollen = [[0.0 if r.random() < 0.3 else r.random() * 100 for _ in range(n_taxa)] for _ in offsets]
    return {'rho': [[h(v) for v in row] for row in rho], 'age': [h(v) for v in age],
            'depth': [h(100.0 + k) for k in offsets], 'pollen': [[h(v) for v in row] for row in pollen]}


def selection(r):
    n, m, cells = 3 + int(r.random() * 10), 2 + int(r.random() * 5), 3 + int(r.random() * 7)
    step = (1 + int(r.random() * 8)) * 0.5
    start = -50.0 + step * int(r.random() * 20)
    age = [start + step * i for i in range(n)]
    coefficients = [[r.random() for _ in range(m)] for _ in range(n)]
    tf = [[r.random() for _ in range(cells)] for _ in range(m)]
    return {'age': [h(v) for v in age], 'taxa_merge': r.random() < 0.5,
            'coefficients': [[h(v) for v in row] for row in coefficients], 'tf': [[h(v) for v in row] for row in tf]}


def final_pdf(r):
    n, t, cells, draws = 3 + int(r.random() * 9), 4 + int(r.random() * 11), 3 + int(r.random() * 8), 2 + int(r.random() * 4)
    available = [[0.0 if r.random() < 0.25 else r.random() * 30 for _ in range(t)] for _ in range(n)]
    for j in range(t):
        available[int(r.random() * n)][j] += 0.5 + r.random()  # no zero-maximum taxon
    for row in available:
        row[0] += 0.0625 + r.random()  # no zero row
    weights = [[0.01 + r.random() for _ in range(t)] for _ in range(draws)]
    tf = [[r.random() for _ in range(cells)] for _ in range(t)]
    pool = [r.random() for _ in range(max(1, draws - 1))]
    energy = [pool[int(r.random() * len(pool))] for _ in range(draws)]
    ids = [1 + int(r.random() * t) for _ in range(t // 2)]
    return {'available': [[h(v) for v in row] for row in available], 'weights': [[h(v) for v in row] for row in weights],
            'tf': [[h(v) for v in row] for row in tf], 'energy': [h(v) for v in energy], 'gibbs_ids': ids}


def cdf(r):
    n, cells = 2 + int(r.random() * 8), 3 + int(r.random() * 10)
    density = [[0.0 if r.random() < 0.2 else r.random() for _ in range(cells)] for _ in range(n)]
    reference = [[r.random() * 3 - 1 for _ in range(cells)] for _ in range(n)]
    return {'density': [[h(v) for v in row] for row in density], 'reference': [[h(v) for v in row] for row in reference],
            'placeholder': r.random() < 0.3}


def tf(r):
    n, cells = 2 + int(r.random() * 8), 3 + int(r.random() * 12)
    raw = [[0.0 if r.random() < 0.2 else r.random() * 7 for _ in range(cells)] for _ in range(n)]
    for row in raw:
        row[int(r.random() * cells)] += 0.25 + r.random()
    return {'raw': [[h(v) for v in row] for row in raw], 'keep': [r.random() < 0.7 for _ in range(cells)]}


def pruning(r):
    rows, taxa = 20, 6 + int(r.random() * 22)
    zero_share = [r.random() for _ in range(taxa)]
    pollen = [[0.0 if r.random() < zero_share[j] else r.random() * 40 for j in range(taxa)] for _ in range(rows)]
    return {'pollen': [[h(v) for v in row] for row in pollen], 'check': True, 'metro': r.random() < 0.3,
            'relict': taxa >= 25 and r.random() < 0.5}


def main():
    rng = random.Random(SEED)
    data = {'kind': 'synthetic_stress_fixture_not_historical_data', 'seed': SEED, 'per_module': PER_MODULE,
            'historical_reproduction_ready': False}
    for name, make in (('age_depth', age_depth), ('selection', selection), ('final_pdf', final_pdf), ('cdf', cdf),
                       ('tf', tf), ('pruning', pruning)):
        data[name] = [make(rng) for _ in range(PER_MODULE)]
    (OUT / 'stress_fixture.json').write_text(json.dumps(data, indent=0) + '\n', encoding='utf-8', newline='\n')
    print('stress fixture written', (OUT / 'stress_fixture.json').stat().st_size)


if __name__ == '__main__':
    main()
