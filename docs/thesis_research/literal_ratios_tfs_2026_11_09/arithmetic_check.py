"""Counterexamples, no variants: what would change on the artificial cases if a detail of the original were read differently."""
import importlib.util
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('ratios_check', OUT / 'check_ratios.py')
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)
rt = check.rt


def main():
    fixture = json.loads((OUT / 'synthetic_fixture.json').read_text(encoding='utf-8'))
    all_cases = [*fixture['cases'].values(), *fixture['stress']]
    doubles = products = 0
    for c in all_cases:
        result = check.run(c)
        exact = rt.transfer_ratio(result)
        if not math.isfinite(exact):
            continue
        plain = 1.0
        for v in rt.unlisted(result):
            plain *= v
        products += 1
        doubles += plain != exact
    no_scale = compared = 0
    for c in fixture['stress']:
        result = check.run(c)
        for row, up, down in ((p, u, d) for p, u, d in zip(result.parts, result.tann_dists_tf_ratio, result.pann_dists_tf_ratio)):
            for ratio, top, bottom in ((up, row[0], row[1]), (down, row[2], row[3])):
                compared += 1
                no_scale += ratio != rt._divide(top.density, bottom.density)
    tie = check.arguments(fixture['cases']['tie_takes_first_minimum'])
    grid = tie['tann_range']
    value = tie['new_tann'][0]
    distances = [abs(g - value) for g in grid]
    last = len(distances) - 1 - distances[::-1].index(min(distances))
    report = {'kind': 'counterexamples_not_variants', 'checks': {
        'double_accumulator_instead_of_long_double': {'changed': doubles, 'compared': products},
        'density_ratio_without_the_truncation_scale': {'changed': no_scale, 'compared': compared},
        'last_minimum_instead_of_first': {'index_first': rt.grid_index(grid, value), 'index_last': last}}, 'historical_reproduction_ready': False}
    (OUT / 'arithmetic_check.json').write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(report['checks'])


if __name__ == '__main__':
    main()
