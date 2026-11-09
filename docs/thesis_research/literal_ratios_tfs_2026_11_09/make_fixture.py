"""Write synthetic_fixture.json: invented grids, densities and proposals, in hex. No original data; the stress generator is only a source of artificial numbers."""
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent
SEED = 20261109


def tok(v):
    return '0' if v == 0 else float(v).hex()


def seq(a, b, n):
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def case(tann_range, pann_range, new_tann, old_tann, tann_dists, new_pann, old_pann, pann_dists, lower_tann, upper_tann, sds_tann, lower_pann, upper_pann, sds_pann):
    return {'num_taxa': len(new_tann), 'tann_range': [tok(v) for v in tann_range], 'pann_range': [tok(v) for v in pann_range], 'new_tann': [tok(v) for v in new_tann], 'old_tann': [tok(v) for v in old_tann],
            'tann_dists': [[tok(v) for v in row] for row in tann_dists], 'new_pann': [tok(v) for v in new_pann], 'old_pann': [tok(v) for v in old_pann], 'pann_dists': [[tok(v) for v in row] for row in pann_dists],
            'lower_tann': [tok(v) for v in lower_tann], 'upper_tann': [tok(v) for v in upper_tann], 'sds_tann_trunc': [tok(v) for v in sds_tann], 'lower_pann': [tok(v) for v in lower_pann],
            'upper_pann': [tok(v) for v in upper_pann], 'sds_pann_trunc': [tok(v) for v in sds_pann]}


T4, P4 = [0.0, 1.0, 2.0, 3.0], [0.0, 100.0, 200.0, 300.0]
TD, PD = [0.1, 0.4, 0.3, 0.2], [0.5, 0.25, 0.125, 0.125]


def smooth(rng, n):
    weights = [rng.uniform(0.05, 1.0) for _ in range(n)]
    total = sum(weights)
    return [w / total for w in weights]


def live_shape():
    rng = random.Random(SEED + 1)
    grid_t, grid_p = seq(-30.0, 40.0, 1000), seq(0.0, 4000.0, 1000)
    n = 6
    sd_t, sd_p = [rng.uniform(0.05, 0.4) for _ in range(n)], [rng.uniform(5.0, 40.0) for _ in range(n)]
    old_t, old_p = [rng.uniform(5.0, 20.0) for _ in range(n)], [rng.uniform(200.0, 900.0) for _ in range(n)]
    new_t, new_p = [v + rng.uniform(-2.5, 2.5) * s for v, s in zip(old_t, sd_t)], [v + rng.uniform(-2.5, 2.5) * s for v, s in zip(old_p, sd_p)]
    return case(grid_t, grid_p, new_t, old_t, [smooth(rng, 1000) for _ in range(n)], new_p, old_p, [smooth(rng, 1000) for _ in range(n)],
                [0.0] * n, [25.0] * n, sd_t, [0.0] * n, [1500.0] * n, sd_p)


def stress(i):
    rng = random.Random(SEED * 3 + i)
    n, gt, gp = rng.randint(3, 25), rng.randint(32, 64), rng.randint(32, 64)
    grid_t, grid_p = seq(-30.0, 40.0, gt), seq(0.0, 4000.0, gp)
    mode = i % 4  # ordinary, tight proposal, wide proposal, narrow window
    low_t, up_t, sd_t, new_t, old_t, low_p, up_p, sd_p, new_p, old_p = [], [], [], [], [], [], [], [], [], []
    for _ in range(n):
        a, b = sorted((rng.uniform(-25.0, 15.0), rng.uniform(16.0, 38.0)))
        c, d = 0.0, rng.uniform(500.0, 3500.0)
        if mode == 3:
            b, d = a + rng.uniform(0.1, 2.0), rng.uniform(20.0, 80.0)
        s_t = {0: rng.uniform(0.05, 2.0), 1: rng.uniform(1e-3, 1e-2), 2: rng.uniform(10.0, 60.0), 3: rng.uniform(0.05, 3.0)}[mode]
        s_p = {0: rng.uniform(2.0, 80.0), 1: rng.uniform(1e-2, 0.2), 2: rng.uniform(500.0, 3000.0), 3: rng.uniform(1.0, 60.0)}[mode]
        low_t.append(a); up_t.append(b); sd_t.append(s_t); low_p.append(c); up_p.append(d); sd_p.append(s_p)
        o_t, o_p = rng.uniform(a, b), rng.uniform(c, d)  # a proposal is a short step from the previous state
        old_t.append(o_t); old_p.append(o_p)
        new_t.append(min(max(o_t + rng.uniform(-2.5, 2.5) * s_t, a), b)); new_p.append(min(max(o_p + rng.uniform(-2.5, 2.5) * s_p, c), d))
    return case(grid_t, grid_p, new_t, old_t, [smooth(rng, gt) for _ in range(n)], new_p, old_p, [smooth(rng, gp) for _ in range(n)], low_t, up_t, sd_t, low_p, up_p, sd_p)


def main():
    cases = {
        'two_taxa_plain': case(T4, P4, [1.2, 0.9], [2.1, 1.9], [TD, PD[::-1]], [110.0, 90.0], [190.0, 310.0], [PD, TD], [0.0, 0.0], [3.0, 3.0], [0.5, 0.25], [0.0, 0.0], [300.0, 400.0], [40.0, 20.0]),
        'tie_takes_first_minimum': case(T4, P4, [0.5], [2.5], [TD], [150.0], [250.0], [PD], [0.0], [3.0], [1.0], [0.0], [300.0], [50.0]),
        'off_grid_takes_end': case(T4, P4, [-7.0], [9.0], [TD], [-40.0], [950.0], [PD], [-8.0], [10.0], [2.0], [-50.0], [1000.0], [100.0]),
        'start_density_zero': case(T4, P4, [1.0], [2.0], [[0.1, 0.4, 0.0, 0.5]], [100.0], [200.0], [PD], [0.0], [3.0], [0.5], [0.0], [300.0], [40.0]),
        'both_densities_zero': case(T4, P4, [1.0], [2.0], [[0.1, 0.0, 0.0, 0.9]], [100.0], [200.0], [PD], [0.0], [3.0], [0.5], [0.0], [300.0], [40.0]),
        'old_outside_truncation': case(T4, P4, [1.0], [3.0], [TD], [100.0], [200.0], [PD], [0.0], [2.0], [0.5], [0.0], [300.0], [40.0]),
        'live_shape_grid_1000': live_shape(),
        'wide_proposal_small_scale': case(T4, P4, [0.25], [0.75], [TD], [100.0], [150.0], [PD], [0.0], [1.0], [1e5], [0.0], [300.0], [1e7]),
        'both_outside_truncation': case(T4, P4, [1.0], [1.5], [TD], [100.0], [150.0], [PD], [12.0], [12.5], [0.75], [900.0], [905.0], [60.0]),
        'new_far_below_window': case(T4, P4, [1.0], [12.25], [TD], [100.0], [902.0], [PD], [12.0], [12.5], [0.75], [900.0], [905.0], [60.0]),
        'equal_bounds': case(T4, P4, [1.0], [1.0], [TD], [100.0], [150.0], [PD], [1.0], [1.0], [0.5], [0.0], [300.0], [40.0]),
        'new_outside_truncation': case(T4, P4, [3.0], [1.0], [TD], [100.0], [150.0], [PD], [0.0], [2.0], [0.5], [0.0], [300.0], [40.0]),
        'tiny_sigma_saturates': case(T4, P4, [1.0], [1.5], [TD], [100.0], [150.0], [PD], [0.0], [3.0], [1e-3], [0.0], [300.0], [1e-2]),
        'three_taxa_quarters': case(T4, P4, [0.25, 1.75, 2.75], [0.75, 1.25, 2.25], [TD, PD, [0.25] * 4], [50.0, 125.0, 275.0], [75.0, 225.0, 25.0], [PD, TD, [0.25] * 4],
                                    [0.0] * 3, [3.0] * 3, [0.5, 0.25, 0.75], [0.0] * 3, [300.0] * 3, [25.0, 50.0, 75.0]),
    }
    report = {'kind': 'synthetic_fixture_not_historical_data', 'app_cases': ['two_taxa_plain', 'tie_takes_first_minimum', 'off_grid_takes_end', 'start_density_zero', 'both_densities_zero', 'old_outside_truncation', 'live_shape_grid_1000', 'wide_proposal_small_scale'],
              'cases': cases, 'stress_cases': 120, 'stress_seed': SEED, 'stress': [stress(i) for i in range(120)], 'historical_reproduction_ready': False}
    (OUT / 'synthetic_fixture.json').write_text(json.dumps(report) + '\n', encoding='utf-8', newline='\n')
    print(len(cases), 'named', len(report['stress']), 'stress')


if __name__ == '__main__':
    main()
