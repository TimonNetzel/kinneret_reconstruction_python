"""Run tools/historical_wt_power.py on the fixture and store the tokens in synthetic_result.json."""
import json
import math
import random
import sys
from fractions import Fraction
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import historical_wt_power as wp  # noqa: E402


def value(token):
    if token == 'NA':
        return wp.NA
    if token == 'NaN':
        return math.nan
    if token in ('Inf', '-Inf'):
        return math.inf if token == 'Inf' else -math.inf
    if '/' in token or not token.lower().startswith(('0x', '-0x')):
        return float(Fraction(token))
    return float.fromhex(token)


def vector(tokens):
    return [value(t) for t in tokens]


def token(v):
    if v is wp.NA:
        return 'NA'
    if isinstance(v, float) and math.isnan(v):
        return 'NaN'
    if math.isinf(v):
        return 'Inf' if v > 0 else '-Inf'
    if v == 0:
        return '-0' if math.copysign(1.0, v) < 0 else '0'
    return float(v).hex()


def wb_rows(columns):
    cols = [vector(col) for col in columns]
    return [[col[i] for col in cols] for i in range(len(cols[0]))]


def run_case(case):
    r = wp.wt_power(vector(case['d']), case['l_d'], n_obs=case['n_obs'], zeros=[0.0] * case['zeros'], invflen=value(case['invflen']), wb=wb_rows(case['wb']))
    return {'x': [token(v) for v in r.x], 'dim': list(r.dim) if r.dim else None, 'power': [token(v) for v in r.values], 'kind': r.kind, 'backend': r.backend}, r


def run_chain(ch):
    r = wp.spectral_weight(predictant_current=vector(ch['predictant_current']), ages_spec=ch['ages_spec'], hi_ages_spec=ch['hi_ages_spec'], n_obs=ch['n_obs'], zeros=[0.0] * ch['zeros'],
                           invflen=value(ch['invflen']), wb=wb_rows(ch['wb']), coi_array=ch['coi'], iso_spec=vector(ch['iso_spec']))
    return {'kind': r.power.kind, 'predictant_spec': [token(v) for v in r.predictant_spec], 'correlation_kind': r.correlation.kind, 'replaced': r.weight.replaced,
            'post_spec': token(r.post_spec), 'l_d': wp.length_ages_spec2(ch['ages_spec'], ch['hi_ages_spec'])}, r


def stress_cases(spec):
    sys.path.insert(0, str(OUT))
    import importlib.util
    loader = importlib.util.spec_from_file_location('wt_fixture_bands', OUT / 'make_fixture_bands.py')
    bands = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(bands)
    rng = random.Random(spec['seed'])
    out = []
    for i in range(spec['count']):
        k = rng.choice(spec['exponents'])
        n = 2 ** k
        n_obs = rng.randint(2 ** (k - 2) + 1, 2 ** (k - 1))
        m = rng.randint(1, spec['max_columns'])
        style = i % 3
        d = [round(rng.uniform(-5, 25), 2) if style == 0 else round(rng.uniform(100, 1500), 1) if style == 1 else rng.uniform(-1, 1) for _ in range(n_obs)]
        out.append({'d': d, 'l_d': n_obs, 'n_obs': n_obs, 'zeros': n - n_obs, 'wb': bands.band_rows(n, spec['dt'], m), 'invflen': 1 / n})
    return out


def run_stress(case):
    return wp.wt_power(case['d'], case['l_d'], n_obs=case['n_obs'], zeros=[0.0] * case['zeros'], invflen=case['invflen'], wb=case['wb'])


def main():
    fixture = json.loads((OUT / 'synthetic_fixture.json').read_text(encoding='utf-8'))
    cases = {n: run_case(c)[0] for n, c in fixture['cases'].items()}
    chains = {n: run_chain(c)[0] for n, c in fixture['chains'].items()}
    (OUT / 'synthetic_result.json').write_text(json.dumps({'kind': 'module_result_on_synthetic_fixture', 'backend': wp.BACKEND, 'numpy': __import__('numpy').__version__, 'cases': cases, 'chains': chains,
                                                           'historical_reproduction_ready': False}, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(len(cases), len(chains), 'written')


if __name__ == '__main__':
    main()
