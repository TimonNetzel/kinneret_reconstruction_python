"""Write the artificial pruning fixture from explicit zero counts; no RNG, no measured data.

Column j with z zeros out of n rows gets zeros in its first z rows and the counts j+1+(i mod 3) elsewhere.
Taxa, rows and counts are artificial; nothing here is a historical site, taxon or depth.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent

# name -> (rows, zero counts per artificial taxon, null_ratio_check, metro_hastings, relict_override, purpose)
CASES = {
    'flag_false_bound_loop': (3, [0, 1, 3, 2], False, False, False,
                              'main_v1.2.R:659 sets null.ratio.check <- FALSE: matrix stays unchanged'),
    'ratio_095_branch': (20, [19, 20, 18] + [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], False, False, False,
                         '18 - 2 > 15 sets metro_hastings; exactly 19/20 zeros is removed, 18/20 kept'),
    'rarest_15_tie_keeps_rare': (20, [20, 19, 20, 20] + [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], False, False, False,
                                 'sort()[16:17] holds 1.0 only: %in% removes all three 1.0 taxa, keeps the 0.95 taxon'),
    'short_core_descending_colon': (20, [0, 2, 5, 10, 10, 3], False, False, False,
                                    'six taxa: 16:6 counts down, NA positions ignored, both 0.5 taxa removed'),
    'all_equal_skip_site': (5, [0, 0, 0, 0], False, False, False,
                            'equal ratios: every taxon matches the maximum, R calls next()'),
    'metro_preset_true': (20, [0, 2, 5, 10, 10, 3], False, True, False,
                          'metro_hastings already TRUE: only ratios >= 0.95 are removed, here none'),
    'relict_override_26': (20, [0, 20, 20] + [1] * 4 + [20] + [2] * 14 + [20, 3, 4, 5], False, False, True,
                           'relict ids 3,8,23,24,25 forced to 0, taxon 1 forced to 1; 26 - 2 > 15'),
}


def matrix(rows, zeros):
    return [[0 if i < z else j + 1 + i % 3 for j, z in enumerate(zeros)] for i in range(rows)]


def main():
    cases = {}
    for name, (rows, zeros, check, metro, relict, purpose) in CASES.items():
        if name != 'flag_false_bound_loop':
            check = True
        cases[name] = {'purpose': purpose, 'null_ratio_check': check, 'metro_hastings': metro,
                       'relict_override': relict, 'zero_counts': zeros, 'pollen': matrix(rows, zeros)}
    fixture = {'kind': 'synthetic_fixture_not_historical_data', 'orientation': 'artificial depth rows x artificial taxa',
               'tolerance': 2e-14, 'historical_reproduction_ready': False, 'selected_final_variant': None,
               'cases': cases}
    (OUT / 'synthetic_fixture.json').write_text(json.dumps(fixture, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(f'{len(cases)} artificial cases written')


if __name__ == '__main__':
    main()
