"""Compare the pruning module with the exact reference on the artificial fixture; write the result files.

Two naive readings serve as counterexamples: removing every ratio >= 0.95 regardless of the branch, and
keeping exactly the 15 lowest ranks. Their differences show which source details the module preserves.
"""
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import historical_taxon_pruning as pruning  # noqa: E402

spec = importlib.util.spec_from_file_location('rational_reference', OUT / 'rational_reference.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def naive_ge095(case):
    rows = case['pollen']
    return [j for j in range(len(rows[0])) if Fraction(sum(r[j] == 0 for r in rows), len(rows)) >= Fraction(19, 20)]


def naive_rank15(case):
    rows = case['pollen']
    ratios = [Fraction(sum(r[j] == 0 for r in rows), len(rows)) for j in range(len(rows[0]))]
    order = sorted(range(len(ratios)), key=lambda j: (ratios[j], j))
    return sorted(order[15:])


def _write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=1) + '\n', encoding='utf-8', newline='\n')


def main():
    fixture = json.loads((OUT / 'synthetic_fixture.json').read_text(encoding='utf-8'))
    tol = fixture['tolerance']
    exact, results, comparisons, max_err = {}, {}, {}, 0.0
    for name, case in fixture['cases'].items():
        flags = {k: case[k] for k in ('null_ratio_check', 'metro_hastings', 'relict_override')}
        ref = reference.exact_pruning(case['pollen'], **flags)
        got = pruning.prune_taxa(case['pollen'], **flags)
        exact[name] = ref
        results[name] = {
            'executed': got.executed, 'branch': got.branch, 'metro_hastings': got.metro_hastings,
            'skip_site': got.skip_site, 'null_ratio': None if got.null_ratio is None else list(got.null_ratio),
            'removed_taxa': list(got.removed_taxa), 'kept_taxa': list(got.kept_taxa),
            'pruned_shape': None if got.pollen is None else list(got.pollen.shape),
            'sorting_available_spectrum': None if got.sorting_available_spectrum is None
            else got.sorting_available_spectrum.tolist()}
        for key in ('executed', 'branch', 'metro_hastings', 'skip_site', 'removed_taxa', 'kept_taxa'):
            if results[name][key] != ref[key]:
                raise AssertionError(f'{name}: {key} differs from the exact reference')
        for field in ('null_ratio', 'sorting_available_spectrum'):
            if (results[name][field] is None) != (ref[field] is None):
                raise AssertionError(f'{name}: {field} presence differs')
            if ref[field] is not None:
                err = max(abs(a - float(Fraction(b))) for a, b in zip(results[name][field], ref[field]))
                max_err = max(max_err, err)
        if got.pollen is not None and got.pollen.tolist() != [[row[j] for j in ref['kept_taxa']] for row in case['pollen']]:
            raise AssertionError(f'{name}: pruned matrix is not the kept columns in original order')
        if case['null_ratio_check'] and not case['relict_override']:
            comparisons[name] = {'source_removed': ref['removed_taxa'], 'naive_ge095_removed': naive_ge095(case),
                                 'naive_rank15_removed': naive_rank15(case)}
    if max_err > tol:
        raise AssertionError(f'largest rational deviation {max_err} exceeds {tol}')
    _write('rational_expectations.json', {'kind': 'exact_fraction_reference', 'cases': exact})
    _write('synthetic_result.json', {'kind': 'module_result_on_synthetic_fixture_not_historical',
                                     'historical_reproduction_ready': False, 'cases': results})
    differs = {n: [k for k in ('naive_ge095_removed', 'naive_rank15_removed') if c[k] != c['source_removed']]
               for n, c in comparisons.items()}
    _write('arithmetic_check.json', {
        'kind': 'synthetic_fragment_check_not_historical_reproduction',
        'cases': len(results), 'index_and_flag_agreement': 'exact', 'tolerance': tol,
        'maximum_absolute_rational_error': max_err, 'naive_counterexamples': comparisons,
        'naive_readings_differing_from_source': differs,
        'historical_reproduction_ready': False, 'final_variant_selected': False, 'R_execution': False})
    print(f'{len(results)} cases agree with the exact reference; max deviation {max_err:.3g}')
    for name, keys in differs.items():
        print(f'  {name}: differs from {", ".join(keys) or "no naive reading"}')


if __name__ == '__main__':
    main()
