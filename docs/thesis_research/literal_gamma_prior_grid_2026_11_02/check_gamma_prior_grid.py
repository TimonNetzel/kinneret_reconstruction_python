"""Compare the gamma-grid module with the independent reference on the artificial fixture; write the result files.

Naive readings serve as counterexamples: the closed form (mean/sd)^2, the grid point nearest to it (ties up,
inside the grid range) and rate = shape/mean everywhere. seq variants without the +1e-10 fuzz, without
pmin(x, to) and with one fused rounding show which grid details depend on the R implementation.
"""
import importlib.util
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import historical_gamma_prior_grid as grid_module  # noqa: E402

spec = importlib.util.spec_from_file_location('rational_reference', OUT / 'rational_reference.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def nearest_to_closed_form(mean, sd, grid):
    target = (Fraction(mean) / Fraction(sd)) ** 2
    target = min(max(target, Fraction(grid[0])), Fraction(grid[-1]))
    k = math.floor((target - Fraction(1, 10)) * 10 + Fraction(1, 2))  # nearest multiple of 0.1, ties up
    return min(max(k, 0), len(grid) - 1)


def _write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=1) + '\n', encoding='utf-8', newline='\n')


def main():
    fixture = json.loads((OUT / 'synthetic_fixture.json').read_text(encoding='utf-8'))
    grids = {upper: reference.r_seq(upper) for upper in (50000.0, 5000.0)}
    exact, results, naive, hand = {}, {}, {}, {}
    for name, case in fixture['grid_cases'].items():
        mean, sd, upper = case['mean_pann'], case['sd_pann'], case['upper']
        ref = reference.grid_reference(mean, sd, upper, grids[upper])
        got = grid_module.gamma_grid_intersection(mean, sd, upper=upper)
        exact[name] = ref
        results[name] = {'grid_length': got.grid_length, 'index': got.index, 'r_index': got.index + 1,
                         'shape': got.shape, 'shape_hex': got.shape.hex(), 'rate': got.rate,
                         'rate_expectation': got.rate_expectation, 'at_lower_boundary': got.at_lower_boundary,
                         'at_upper_boundary': got.at_upper_boundary, 'closed_form_shape': got.closed_form_shape}
        for key in ('grid_length', 'index', 'shape', 'rate', 'rate_expectation'):
            if results[name][key] != ref[key]:  # bitwise, both are IEEE binary64
                raise AssertionError(f'{name}: {key} differs from the independent reference')
        hand[name] = {'hand_r_index': case['hand_r_index'], 'r_index': ref['r_index'],
                      'agrees': case['hand_r_index'] == ref['r_index']
                      and abs(ref['shape'] - case['hand_shape_approx']) < 1e-9}
        k = nearest_to_closed_form(mean, sd, grids[upper])
        naive[name] = {'source_r_index': ref['r_index'], 'source_shape': ref['shape'],
                       'closed_form_shape': float(Fraction(ref['closed_form_shape_exact'])),
                       'nearest_grid_r_index': k + 1, 'nearest_grid_shape': grids[upper][k],
                       'nearest_grid_differs': k != ref['index'],
                       'source_rate_sqrt_branch': ref['rate'], 'naive_rate_shape_over_mean': ref['rate_expectation'],
                       'rate_differs': ref['rate'] != ref['rate_expectation']}
    if not all(h['agrees'] for h in hand.values()):
        raise AssertionError(f'hand expectation differs: {hand}')
    chains = {}
    for name, case in fixture['chain_cases'].items():
        args = {k: case[k] for k in ('mean_pann', 'sd_pann', 'sd_tmp')}
        flags = {k: case[k] for k in ('recent', 'min_age_at_least_2000')}
        ref = reference.chain_reference(*args.values(), **flags, grid=grids[50000.0])
        got = grid_module.prior_gamma_chain(*args.values(), **flags)
        stored = {'default_index': got.default.index, 'default_shape': got.default.shape,
                  'default_rate': got.default.rate, 'diff_rate': got.diff_rate,
                  'variance_doubled': got.variance_doubled, 'sd_pann': got.sd_pann, 'sd_tmp': got.sd_tmp,
                  'final_index': got.final.index, 'final_shape': got.final.shape, 'final_rate': got.final.rate,
                  'rate_after_recent_start': got.rate_after_recent_start}
        if stored != ref or got.pann_period_shape != got.default.shape or got.rgamma_1302_rate != got.final.rate:
            raise AssertionError(f'{name}: chain differs from the independent reference')
        exact[name], results[name] = ref, {**stored, 'rgamma_1302_rate': got.rgamma_1302_rate}
    # seq sensitivity: lengths and last values for every variant switch
    variants = {}
    for upper in (50000.0, 5000.0):
        base = grids[upper]
        rows = {}
        for label, kw in {'no_fuzz': dict(fuzz=False), 'no_pmin': dict(clamp=False),
                          'no_fuzz_no_pmin': dict(fuzz=False, clamp=False)}.items():
            other = reference.r_seq(upper, **kw)
            rows[label] = {'length': len(other), 'last': other[-1], 'identical_to_seq_default': other == base}
        fused = reference.r_seq(upper, fused=True)
        fused_cases = {n: reference.grid_reference(c['mean_pann'], c['sd_pann'], upper, fused, two_step_grid=False)['index']
                       for n, c in fixture['grid_cases'].items() if c['upper'] == upper}
        rows['fused_rounding'] = {'values_differing': sum(a != b for a, b in zip(fused, base)),
                                  'case_indices_changed': [n for n, i in fused_cases.items() if i != exact[n]['index']]}
        variants[str(int(upper))] = {'length': len(base), 'last': base[-1], 'variants': rows}
    _write('rational_expectations.json', {'kind': 'independent_float_scan_with_exact_rounding_checks', 'cases': exact})
    _write('synthetic_result.json', {'kind': 'module_result_on_synthetic_fixture_not_historical',
                                     'historical_reproduction_ready': False, 'cases': results})
    _write('arithmetic_check.json', {
        'kind': 'synthetic_fragment_check_not_historical_reproduction',
        'cases': len(results), 'agreement': 'bitwise equal doubles for every index, shape and rate',
        'hand_expectations': hand, 'naive_counterexamples': naive,
        'naive_nearest_grid_differs': [n for n, v in naive.items() if v['nearest_grid_differs']],
        'seq_sensitivity': variants,
        'historical_reproduction_ready': False, 'final_variant_selected': False, 'R_execution': False})
    print(f'{len(results)} cases agree bitwise with the independent reference and all hand expectations')
    print('nearest-grid reading differs in:', ', '.join(n for n, v in naive.items() if v['nearest_grid_differs']))
    for upper, v in variants.items():
        print(f'  upper {upper}: length {v["length"]}, last {v["last"]!r},',
              {k: (r.get('identical_to_seq_default'), r.get('values_differing')) for k, r in v['variants'].items()})


if __name__ == '__main__':
    main()
