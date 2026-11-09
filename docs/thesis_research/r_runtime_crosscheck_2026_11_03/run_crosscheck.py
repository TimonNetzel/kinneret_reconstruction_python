"""Bit-exact cross check: literal original R lines (R 3.5.0) against the Python modules on the artificial fixtures.

The originals are read only and must match the SHA-256 of the earlier frozen binding before R sees them. R evaluates
mcmc_conduction.R 53-79 (sampler preparation), mcmc_functions_v1.6.R 1276-1280 (weights-stage core),
mcmc_prepare.R 1270-1281 (gamma grid, upper 50000) and 1681-1685 (num_taxa priors) verbatim in a fresh environment with base R only; no package is attached, no sampler, no random state. Inputs and outputs travel
as C99 hex doubles. Values the modules flag as accumulator dependent are compared and reported, not asserted.
Usage: run_crosscheck.py <Rscript.exe>
"""
import hashlib
import json
import math
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import historical_sampler_preparation as sp  # noqa: E402
import historical_weights_stage as ws  # noqa: E402
import historical_gamma_prior_grid as gg  # noqa: E402
import historical_tf_missing_priors as tm  # noqa: E402

SOURCES = {'prepare': ('src/mcmc_prepare.R', 'b897e84f8f2d72399de729592066341585974d521bf8164b185f20377ed5cc7e'),
           'conduction': ('src/mcmc_conduction.R', '47c59c508e15150f7d6fda775a4f6bf6c14f0148e10cefd0014fa1ce6cfe5a9d'),
           'functions': ('src/mcmc_functions_v1.6.R',
                         'b887144370e328fa016d1834ceebcd55622816d4b36838d33a26197110ab453c')}
SAMPLER = ROOT / 'docs/thesis_research/literal_sampler_preparation_2026_11_02/synthetic_fixture.json'
WEIGHTS = ROOT / 'docs/thesis_research/literal_weights_stage_2026_11_03/synthetic_fixture.json'
GAMMA = ROOT / 'docs/thesis_research/literal_gamma_prior_grid_2026_11_02/synthetic_fixture.json'
PRIOR_COUNTS = range(2, 61)
LOCAL = ROOT / '.local/r_runtime_crosscheck_2026_11_03'


def r_double(v):
    if v is None:
        return 'NA_real_'
    if isinstance(v, float) and math.isnan(v):
        return 'NaN'
    if v == -math.inf:
        return '-Inf'
    v = float(v)
    return '0' if v == 0 else v.hex()


def r_vector(values):
    return 'c(' + ', '.join(r_double(v) for v in values) + ')'


def from_hex(token):
    return {'NA': 'NA', 'NaN': 'NaN', 'Inf': 'Inf', '-Inf': '-Inf', '-0': '-0', '0': 0.0}.get(token) \
        if token in ('NA', 'NaN', 'Inf', '-Inf', '-0', '0') else float.fromhex(token)


def special(v):
    if v is None:
        return 'NA'
    if math.isnan(v):
        return 'NaN'
    if math.isinf(v):
        return 'Inf' if v > 0 else '-Inf'
    if v == 0:
        return '-0' if math.copysign(1.0, v) < 0 else 0.0
    return v


def fixture_value(text):
    return None if text == 'NA' else math.nan if text == 'NaN' else -math.inf if text == '-Inf' else \
        float(Fraction(text))


def main():
    rscript = sys.argv[1]
    cfg = json.loads((ROOT / '.local/my_reconst_paths.json').read_text(encoding='utf-8-sig'))
    base = Path(cfg['original_data_root'])
    paths = {}
    for key, (rel, sha) in SOURCES.items():
        if hashlib.sha256((base / rel).read_bytes()).hexdigest() != sha:
            raise ValueError(f'{rel} differs from the frozen binding')
        paths[key] = (base / rel).as_posix()
    sampler = json.loads(SAMPLER.read_text(encoding='utf-8'))['cases']
    weights = json.loads(WEIGHTS.read_text(encoding='utf-8'))['cases']
    lines = ['sources <- list(' + ', '.join(f'{k} = "{v}"' for k, v in paths.items()) + ')',
             'cases <- list(sampler = list(), weights = list(), gamma = list(), priors = c(' +
             ', '.join(f'{n}L' for n in PRIOR_COUNTS) + '))']
    # the literal lines 1270-1281 use the upper bound 50000
    gamma = {k: v for k, v in json.loads(GAMMA.read_text(encoding='utf-8'))['grid_cases'].items()
             if v['upper'] == 50000.0}
    for name, c in gamma.items():
        lines.append(f'cases$gamma[["{name}"]] <- list(mean_pann = {r_double(c["mean_pann"])}, '
                     f'sd_pann = {r_double(c["sd_pann"])})')
    for name, c in sampler.items():
        m = c['taxa_appear_core']
        lines.append(f'cases$sampler[["{name}"]] <- list(values = {r_vector([v for row in m for v in row])}, '
                     f'nrow = {len(m)}L, ncol = {len(m[0])}L, num_taxa = {c["num_taxa"]}L, '
                     f'gibbs = {"TRUE" if c["gibbs"] else "FALSE"})')
    for name, c in weights.items():
        lines.append(f'cases$weights[["{name}"]] <- list(values = {r_vector([fixture_value(v) for v in c["values"]])}, '
                     f'nrow = {c["nrow_taxa_array"]}L, ncol = {c["ncol_taxa_array"]}L, '
                     f'new_weights = {r_vector([float(Fraction(w)) for w in c["new_weights"]])})')
    LOCAL.mkdir(parents=True, exist_ok=True)
    (LOCAL / 'input.R').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    subprocess.run([rscript, '--vanilla', str(OUT / 'crosscheck.R'), str(LOCAL / 'input.R'), str(LOCAL / 'output.tsv')],
                   check=True)
    r_out, r_info = {}, None
    for line in (LOCAL / 'output.tsv').read_text(encoding='utf-8').splitlines():
        parts = line.split('\t')
        if parts[0] == 'R':
            r_info = {'version': parts[1], 'sizeof_longdouble': int(parts[2])}
            continue
        case, key, text = parts
        r_out.setdefault(case, {})[key] = text if key == 'parameter_stage' else [from_hex(t) for t in text.split()]

    sampler_rows = {}
    for name, c in sampler.items():
        m = [[math.nan if v is None else v for v in row] for row in c['taxa_appear_core']]
        got = sp.prepare_sampler(m, c['num_taxa'], gibbs=c['gibbs'])
        r = r_out[name]
        means = [special(x.value) for x in got.taxa_means]
        flags = [x.accumulator_independent for x in got.taxa_means]
        sampler_rows[name] = {
            'how_many_taxa_equal': r['how_many_taxa'] == [float(got.how_many_taxa)],
            'parameter_stage_equal': r['parameter_stage'] == got.parameter_stage.rsplit('_', 1)[1],
            'values_equal': r['values'] == [special(v) for v in got.values],
            'jeffreys_equal': r['jeffreys_prior'] == list(got.jeffreys_prior),
            'means_flagged_independent_equal': all(a == b for a, b, f in zip(r['taxa_means'], means, flags) if f),
            'means_flagged_dependent': [{'column': j + 1, 'module': means[j], 'R': r['taxa_means'][j],
                                         'equal': r['taxa_means'][j] == means[j]}
                                        for j, f in enumerate(flags) if not f],
        }
    weights_rows = {}
    for name, c in weights.items():
        got = ws.weights_stage([fixture_value(v) for v in c['values']], c['ncol_taxa_array'], c['nrow_taxa_array'],
                               [float(Fraction(w)) for w in c['new_weights']])
        r, ncol = r_out[name], c['ncol_taxa_array']
        flags = [s.accumulator_independent for s in got.row_sums]
        result = [special(v) for row in got.taxa_appear_core_temp for v in row]
        rows_equal = [r['result'][i * ncol:(i + 1) * ncol] == result[i * ncol:(i + 1) * ncol]
                      for i in range(len(flags))]
        weights_rows[name] = {
            'unnormalised_equal': r['unnormalised'] == [special(v) for row in got.unnormalised for v in row],
            'z_equal': r['z'] == [float(k) for k in got.z],
            'flagged_independent_rows_equal': all(e for e, f in zip(rows_equal, flags) if f),
            'row_sums_flagged_independent_equal': all(a == special(s.value) for a, s, f in
                                                      zip(r['row_sums'], got.row_sums, flags) if f),
            'rows_flagged_dependent': [{'row': i + 1, 'row_sum_module': special(got.row_sums[i].value),
                                        'row_sum_R': r['row_sums'][i], 'row_equal': rows_equal[i]}
                                       for i, f in enumerate(flags) if not f],
        }
    gamma_rows = {}
    for name, c in gamma.items():
        got, r = gg.gamma_grid_intersection(c['mean_pann'], c['sd_pann'], upper=50000.0), r_out[name]
        gamma_rows[name] = {'intersect_id_equal': r['intersect_id'] == [float(got.index + 1)],
                            'shape_equal': r['shape'] == [got.shape], 'rate_equal': r['rate'] == [got.rate]}
    prior_rows = {}
    for n in PRIOR_COUNTS:
        got, r = tm.num_taxa_priors(n), r_out[f'priors_{n}']
        prior_rows[n] = {'which_taxa_equal': r['which_taxa'] == [float(k) for k in got.which_taxa],
                         'taxa_weights_equal': r['taxa_weights'] == list(got.taxa_weights),
                         'phi_equal': r['phi'] == [got.phi], 'nr_taxa_mean_equal': r['nr_taxa_mean'] == [float(n)],
                         'nr_taxa_sd_equal': r['nr_taxa_sd'] == [float(got.nr_taxa_sd)]}
    checks = [v for rows in (sampler_rows, weights_rows, gamma_rows, prior_rows) for row in rows.values()
              for k, v in row.items() if k.endswith('equal')]
    report = {
        'kind': 'bit_exact_cross_check_literal_R_lines_on_synthetic_fixtures_not_case_reproduction',
        'R': r_info, 'platform': 'Windows x86_64, R 3.5.0 from CRAN archive (md5 and Authenticode checked)',
        'original_lines': {'sampler_preparation': 'src/mcmc_conduction.R:53-79',
                           'weights_stage_core': 'src/mcmc_functions_v1.6.R:1276-1280',
                           'gamma_grid': 'src/mcmc_prepare.R:1270-1281',
                           'num_taxa_priors': 'src/mcmc_prepare.R:1681-1685'},
        'sources_sha256': {rel: sha for rel, sha in SOURCES.values()},
        'packages_attached': [], 'random_state_used': False,
        'all_unflagged_values_bit_equal': all(checks),
        'checks': len(checks),
        'sampler_preparation': sampler_rows, 'weights_stage': weights_rows, 'gamma_grid': gamma_rows,
        'num_taxa_priors': {str(n): row for n, row in prior_rows.items()},
        'historical_reproduction_ready': False,
    }
    (OUT / 'crosscheck_result.json').write_text(json.dumps(report, indent=1, ensure_ascii=False) + '\n',
                                                encoding='utf-8', newline='\n')
    if not all(checks):
        raise AssertionError('an unflagged value differs between R and the module; see crosscheck_result.json')
    dep = [(n, x) for n, row in sampler_rows.items() for x in row['means_flagged_dependent']] + \
          [(n, x) for n, row in weights_rows.items() for x in row['rows_flagged_dependent']]
    print(r_info, len(checks), 'checks; all unflagged values bit-equal;', len(dep), 'flagged items:',
          sum(x.get('equal', x.get('row_equal')) for _, x in dep), 'equal in R')


if __name__ == '__main__':
    main()
