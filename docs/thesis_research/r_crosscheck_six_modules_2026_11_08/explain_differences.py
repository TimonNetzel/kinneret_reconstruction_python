"""Attribute the ulp differences between the modules and R: recompute the flagged values the way R 3.5.0 computes them.

Hypothesis under test: rowSums, sum and colMeans accumulate in 80-bit long double (64-bit significand), and %*% is
reference BLAS, a plain double loop in column order. NumPy instead sums in double and multiplies through its own BLAS.
The emulation below is exact-rational arithmetic with one rounding per long-double or double operation; it is only a
diagnostic and no module uses it. Writes difference_causes.json with a SHA-256 of R's values per key, so the record can
be rechecked without R. Usage: explain_differences.py [path to R's output.tsv]
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
_spec = importlib.util.spec_from_file_location('six_run', OUT / 'run_six_crosscheck.py')
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)

FLAGGED = {'age_depth': ('weights', 'pollen_at_ages'),
           'selection': ('product_initial', 'product_effective', 'product_effective_pann', 'product_effective_mean'),
           'final_pdf': ('coefficients', 'density', 'pdf_sum_depth', 'pdf_sum_age', 'gibbs_coefficients', 'gibbs_density'),
           'cdf': ('cumulative', 'score'), 'tf': ('normalized', 'masked'), 'pruning': ('spectrum',)}


def round_bits(x: Fraction, bits: int) -> Fraction:
    """x rounded to a binary significand of the given width, ties to even."""
    if x == 0:
        return x
    n, d = abs(x.numerator), x.denominator
    exponent = n.bit_length() - d.bit_length()
    if (n << max(-exponent, 0)) < (d << max(exponent, 0)):
        exponent -= 1
    shift = bits - 1 - exponent
    scaled_n = n << shift if shift >= 0 else n
    scaled_d = d if shift >= 0 else d << -shift
    q, r = divmod(scaled_n, scaled_d)
    twice = 2 * r
    if twice > scaled_d or (twice == scaled_d and q & 1):
        q += 1
    value = Fraction(q << -shift) if shift < 0 else Fraction(q, 1 << shift)
    return value if x > 0 else -value


def ld_sum(values) -> float:
    total = Fraction(0)
    for v in values:
        total = round_bits(total + Fraction(v), 64)
    return float(total)


def ld_mean(values) -> float:
    total = Fraction(0)
    for v in values:
        total = round_bits(total + Fraction(v), 64)
    return float(round_bits(total / len(values), 64))


def dot_plain(row, column) -> float:
    """Reference BLAS: y(i) = y(i) + temp * a(i, l), plain doubles in column order."""
    total = 0.0
    for a, b in zip(row, column):
        total += a * b
    return total


def product(rows, columns_matrix):
    cols = list(zip(*columns_matrix))
    return [[dot_plain(row, col) for col in cols] for row in rows]


def flat(rows):
    return [v for row in rows for v in row]


def normalise_rows(rows):
    out = []
    for row in rows:
        mass = ld_sum(row)
        out.append([v / mass for v in row])
    return out


def age_depth(c):
    offsets = [int(d - min(c['depth'])) for d in c['depth']]
    weights = normalise_rows([[row[j] for j in offsets] for row in c['rho']])
    weights = [[10e-10 if v == 0 else v for v in row] for row in weights]
    pollen = [[1e-6 if v == 0 else v for v in row] for row in c['pollen']]
    return {'weights': flat(weights), 'pollen_at_ages': flat(product(weights, [list(r) for r in pollen]))}


def selection(c):
    from_mask = lambda mask: [row for row, m in zip(c['coefficients'], mask) if m]
    masks = run.sel.preserved_selection_trace(c['age'], taxa_merge=c['taxa_merge'])
    out = {}
    if masks.initial_reconst_ages.sum() >= 2:
        out['product_initial'] = flat(product(from_mask(masks.initial_reconst_ages), c['tf']))
    effective = flat(product(from_mask(masks.effective_reconst_ages), c['tf']))
    out.update(product_effective=effective, product_effective_pann=effective, product_effective_mean=effective)
    return out


def final_pdf(c):
    columns = list(zip(*c['available']))
    maxima = [max(col) for col in columns]
    scaled = [[v / m for v, m in zip(row, maxima)] for row in c['available']]
    out, seen, total = {}, set(), None
    for i, weights in enumerate(c['weights'], 1):
        coefficients = normalise_rows([[v * w for v, w in zip(row, weights)] for row in scaled])
        density = product(coefficients, c['tf'])
        out[f'coefficients_{i}'], out[f'density_{i}'] = flat(coefficients), flat(density)
        if c['energy'][i - 1] not in seen:
            seen.add(c['energy'][i - 1])
            total = density if total is None else [[a + b for a, b in zip(r, s)] for r, s in zip(total, density)]
    summed = flat(normalise_rows(total))
    out['pdf_sum_depth'] = out['pdf_sum_age'] = summed
    if len(c['gibbs_ids']) >= 2:
        ids = [i - 1 for i in c['gibbs_ids']]
        coefficients = normalise_rows([[row[j] for j in ids] for row in scaled])
        out['gibbs_coefficients'] = flat(coefficients)
        out['gibbs_density'] = flat(product(coefficients, [c['tf'][j] for j in ids]))
    return out


def cdf(c):
    cumulative = []
    for row in c['density']:
        current, done = None, []
        for v in row:
            current = v if current is None else float(round_bits(Fraction(current) + Fraction(v), 64))
            done.append(current)
        cumulative.append(done)
    width = len(c['density'][0])
    reference = c['reference'] if not c['placeholder'] else [[0.0] * width for _ in c['density']]
    score = [(1 / width) * ld_sum([abs(a - b) for a, b in zip(row, ref)]) for row, ref in zip(cumulative, reference)]
    return {'cumulative': flat(cumulative), 'score': score}


def tf(c):
    normalized = normalise_rows(c['raw'])
    return {'normalized': flat(normalized), 'masked': flat([[v if k else 0.0 for v, k in zip(row, c['keep'])] for row in normalized])}


def pruning(c):
    got = run.tp.prune_taxa(c['pollen'], null_ratio_check=c['check'], metro_hastings=c['metro'], relict_override=c['relict'])
    if not got.executed or got.skip_site:
        return {}
    return {'spectrum': [ld_mean(col) for col in zip(*got.pollen.tolist())]}


EMULATE = {'age_depth': age_depth, 'selection': selection, 'final_pdf': final_pdf, 'cdf': cdf, 'tf': tf, 'pruning': pruning}


def short(key):
    return key.split('_')[0] if key.startswith(('coefficients_', 'density_')) and key[-1].isdigit() else key


def main(tsv=None):
    all_cases = {module: {f'{module}.{name}': case for name, case in group.items()} for module, group in run.cases().items()}
    stored_r = {}
    if tsv is not None:
        for line in Path(tsv).read_text(encoding='utf-8').splitlines():
            parts = line.split('\t')
            if parts[0] != 'info':
                stored_r.setdefault(parts[0], {})[parts[1]] = [run.from_r(t) for t in parts[2].split()] if len(parts) > 2 and parts[2] else []
    report = {}
    for module, group in all_cases.items():
        rows = {}
        for name, case in group.items():
            emulated = EMULATE[module](case)
            mine = run.MODULES[module](case)
            for key, values in emulated.items():
                tokens = [run.token(v) for v in values]
                slot = rows.setdefault(short(key), {'values': 0, 'module_bit_equal': 0, 'emulation_bit_equal': 0, 'max_emulation_ulps': 0,
                                                    'tokens': []})
                slot['tokens'] += tokens
                if tsv is not None:
                    r = stored_r[name][key]
                    for a, b, m in zip(tokens, r, mine[key]):
                        slot['values'] += 1
                        slot['module_bit_equal'] += run.ulps(m, b) == 0
                        u = run.ulps(a, b)
                        slot['emulation_bit_equal'] += u == 0
                        slot['max_emulation_ulps'] = max(slot['max_emulation_ulps'], u if u is not None else 10**9)
        report[module] = rows
    if tsv is None:
        return report
    for module, rows in report.items():
        for key, slot in rows.items():
            slot['emulated_digest'] = hashlib.sha256(' '.join(t if isinstance(t, str) else t.hex() for t in slot.pop('tokens')).encode()).hexdigest()
    result = {'kind': 'attribution_of_ulp_differences_diagnostic_emulation_only',
              'hypothesis': 'R accumulates rowSums, sum and colMeans in 80-bit long double and multiplies with reference BLAS '
                            'as a plain double loop; the modules use NumPy double sums and its BLAS',
              'R_runtime': 'R 3.5.0, Windows x86_64, sizeof(long double) = 16',
              'emulation_bit_equal_everywhere': all(s['values'] == s['emulation_bit_equal'] for r in report.values() for s in r.values()),
              'values': sum(s['values'] for r in report.values() for s in r.values()),
              'module_bit_equal': sum(s['module_bit_equal'] for r in report.values() for s in r.values()),
              'emulation_bit_equal': sum(s['emulation_bit_equal'] for r in report.values() for s in r.values()),
              'modules': report, 'historical_reproduction_ready': False}
    (OUT / 'difference_causes.json').write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8', newline='\n')
    print({k: result[k] for k in ('emulation_bit_equal_everywhere', 'values', 'module_bit_equal', 'emulation_bit_equal')})
    for module, rows in report.items():
        print(module, {k: (s['values'], s['module_bit_equal'], s['emulation_bit_equal'], s['max_emulation_ulps']) for k, s in rows.items()})


if __name__ == '__main__':
    main(sys.argv[1])
