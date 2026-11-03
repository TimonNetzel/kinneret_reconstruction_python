"""Compare the Dirichlet-ratio module with the independent Decimal/Spouge reference and the hand values; write results.

Counterexamples (no candidate variants): the commented alternative of line 1265 with dirichlet_spread in the proposal,
renormalised weights (as if the LaplacesDemon line assigned its result), CPython's math.lgamma instead of correctly
rounded lgamma, and a symmetric proposal (proposal_ratio = 1).
"""
import importlib.util
import json
import math
import struct
import sys
from fractions import Fraction
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import historical_dirichlet_ratio as dr  # noqa: E402

spec = importlib.util.spec_from_file_location('dirichlet_reference', OUT / 'precise_reference.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def value(text):
    return float(Fraction(text))


def special(v):
    if math.isnan(v):
        return 'NaN'
    if math.isinf(v):
        return 'Inf' if v > 0 else '-Inf'
    return v


def ulps(a, b):
    """Distance in units in the last place between two finite positive doubles."""
    ia, ib = (struct.unpack('<q', struct.pack('<d', v))[0] for v in (a, b))
    return abs(ia - ib)


def as_record(r):
    return {'densities': [special(d.value) for d in (r.weights_numerator, r.weights_denominator,
                                                     r.proposal_numerator, r.proposal_denominator)],
            'weights_ratio': special(r.weights_ratio), 'proposal_ratio': special(r.proposal_ratio),
            'parameter_ratio_before_na': special(r.parameter_ratio_before_na),
            'parameter_ratio': special(r.parameter_ratio), 'na_replaced': r.na_replaced,
            'normalisation_skipped': [d.normalisation_skipped for d in (r.weights_numerator, r.weights_denominator,
                                                                         r.proposal_numerator, r.proposal_denominator)],
            'accumulator_independent': [d.accumulator_independent for d in (r.weights_numerator,
                                        r.weights_denominator, r.proposal_numerator, r.proposal_denominator)],
            'log_densities': [special(d.log_density) for d in (r.weights_numerator, r.weights_denominator,
                                                               r.proposal_numerator, r.proposal_denominator)]}


def _cpython_lgamma_density(x, alpha):
    s = math.fsum(alpha)
    dens = (math.lgamma(s) - math.fsum(math.lgamma(a) for a in alpha)) + \
        math.fsum((a - 1.0) * math.log(v) for a, v in zip(alpha, x))
    return math.exp(dens) if dens < 709 else math.inf


def main():
    fixture = json.loads((OUT / 'synthetic_fixture.json').read_text(encoding='utf-8'))
    results, expectations, hand, naive = {}, {}, {}, {}
    for name, case in fixture['ratio_cases'].items():
        old, new, n = [value(v) for v in case['old_weights']], [value(v) for v in case['new_weights']], case['num_taxa']
        got = dr.dirichlet_ratio(old, new, n)
        rec, ref = as_record(got), reference.ratio(old, new, n)
        pipeline = {k: special(ref[k]) for k in ('weights_ratio', 'proposal_ratio', 'parameter_ratio_before_na',
                                                 'parameter_ratio')}
        pipeline['densities'] = [special(v) for v in ref['densities']]
        if any(rec[k] != pipeline[k] for k in pipeline):
            raise AssertionError(f'{name}: module differs from the independent pipeline reference')
        results[name], expectations[name] = rec, {**pipeline, 'exact': ref['exact']}
        h, checks = case['hand'], {}
        for k, t in h.get('exact', {}).items():
            checks[k] = rec[k] == value(t)
        for k, t in h.get('closed', {}).items():
            closed = reference.closed_form(t)
            checks[k + '_ulps_to_closed_form'] = ulps(rec[k], float(closed))
            checks[k + '_reference_exact_matches_closed_form'] = \
                abs(reference.Decimal(ref['exact'][k]) - closed) <= abs(closed) * reference.Decimal('1e-45')
        for k, t in h.get('specials', {}).items():
            checks[k] = rec[k] == (t if t in ('Inf', 'NaN') else value(t))
        hand[name] = checks
        spread = n * n
        with_spread = None
        try:
            with_spread = dr._div(dr.ddirichlet(old, [w * spread for w in new]).value,
                                  dr.ddirichlet(new, [w * spread for w in old]).value)
        except dr.UnsupportedSourceDomain:
            pass
        renorm = None
        if any(d for d in rec['normalisation_skipped']):
            sn = math.fsum(new)
            renorm = dr.dirichlet_ratio(old, [w / sn for w in new], n).parameter_ratio
        cpy = [_cpython_lgamma_density(x, a) for x, a in ((new, [0.5] * n), (old, [0.5] * n), (old, new), (new, old))]
        naive[name] = {
            'proposal_with_dirichlet_spread_1265': special(with_spread) if with_spread is not None else None,
            'proposal_with_spread_differs': with_spread is not None and special(with_spread) != rec['proposal_ratio'],
            'renormalised_parameter_ratio': special(renorm) if renorm is not None else None,
            'renormalisation_differs': renorm is not None and special(renorm) != rec['parameter_ratio'],
            'cpython_lgamma_densities_differing': sum(special(a) != b for a, b in zip(cpy, rec['densities'])),
            'symmetric_proposal_differs': rec['proposal_ratio'] != 1.0,
        }
    densities = {}
    for name, case in fixture['density_cases'].items():
        x, a = [value(v) for v in case['x']], [value(v) for v in case['alpha']]
        got = dr.ddirichlet(x, a)
        closed = reference.closed_form(case['hand_closed_form'])
        pipe = reference.density_pipeline(x, a)
        if got.value != pipe:
            raise AssertionError(f'{name}: density differs from the independent pipeline reference')
        densities[name] = {'value': got.value, 'log_density': got.log_density,
                           'ulps_to_closed_form': ulps(got.value, float(closed)),
                           'reference_exact_matches_closed_form':
                               abs(reference.density_exact(x, a) - closed) <= closed * reference.Decimal('1e-45')}
    bad = {n: k for n, c in hand.items() for k, v in c.items()
           if (isinstance(v, bool) and not v) or (k.endswith('_ulps_to_closed_form') and v > 4)}
    bad.update({n: 'density' for n, d in densities.items()
                if d['ulps_to_closed_form'] > 4 or not d['reference_exact_matches_closed_form']})
    if bad:
        raise AssertionError(f'hand expectation differs: {bad}')

    def write(name, data):
        (OUT / name).write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    write('reference_expectations.json', {'kind': 'independent_decimal_spouge_reference', 'cases': expectations})
    write('synthetic_result.json', {'kind': 'module_result_on_synthetic_fixture_not_historical',
                                    'historical_reproduction_ready': False, 'ratio_cases': results,
                                    'density_cases': densities})
    write('arithmetic_check.json', {
        'kind': 'synthetic_fragment_check_not_historical_reproduction',
        'ratio_cases': len(results), 'density_cases': len(densities),
        'agreement': 'module doubles equal the independent pipeline (Spouge lgamma, Decimal log/exp, each source '
                     'operation rounded once) exactly, including Inf, NaN and the 1273 replacement',
        'hand_expectations': hand, 'naive_counterexamples': naive,
        'precision_note': 'closed forms are met within the listed ulps; the source rounds after every operation, '
                          'so a few ulps from the closed form are expected and are not errors',
        'provider': fixture['provider'], 'historical_reproduction_ready': False, 'final_variant_selected': False,
        'R_execution': False})
    print(len(results), 'ratio and', len(densities), 'density cases agree exactly with the independent reference')
    print('hand:', {n: {k: v for k, v in c.items()} for n, c in hand.items() if c})
    print('densities:', {n: d['ulps_to_closed_form'] for n, d in densities.items()})
    print('counterexamples:', {n: [k for k, v in d.items() if k.endswith('differs') and v] +
                               [f"cpython {d['cpython_lgamma_densities_differing']}"] for n, d in naive.items()})


if __name__ == '__main__':
    main()
