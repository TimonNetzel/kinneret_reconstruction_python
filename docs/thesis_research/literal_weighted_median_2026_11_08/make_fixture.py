"""Write synthetic_fixture.json: artificial weightedMedian cases, TF-matrix cases and the seed of the stress set.

All numbers are exact strings: integers, fractions, 'NA', 'NaN', 'Inf', '-Inf'. Nothing here comes from a real transfer
function. ``hand`` is written by hand from the C text and never computed by the module.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
NA, NAN, INF, MINF = 'NA', 'NaN', 'Inf', '-Inf'


def c(x, w, hand=None, **extra):
    return {'x': [str(v) for v in x], 'w': [str(v) for v in w], 'hand': hand, **extra}


def num(value, path, half=None):
    return {'kind': 'number', 'value': str(value), 'path': path, 'half': half}


def na(path):
    return {'kind': 'NA', 'value': None, 'path': path, 'half': None}


CASES = {
    # values, interpolation
    'uniform_even': c([1, 2, 3, 4], [1, 1, 1, 1], num('5/2', 'interpolated', 2)),
    'uniform_odd_rounding_tie': c([1, 2, 3], [1, 1, 1], num(2, 'interpolated', 1),
                                  note='1.5*d(1/3) = 1/2 - 2^-55 is a tie and rounds to exactly 1/2, so half = 1, dy = 0'),
    'skewed': c([1, 2, 3, 4], [1, 2, 3, 2], num('14/5', 'interpolated', 2)),
    'unsorted_pairs': c([3, 1, 4, 2], [3, 1, 2, 2], num('14/5', 'interpolated', 2),
                        note='the same four (x, w) pairs as skewed in another order'),
    'dropped_nonpositive': c([1, 2, 3, 4, 5], [0, -1, 2, 0, 2], num(4, 'interpolated', 1)),
    'only_one_positive': c([7, 8, 9], [0, 3, 0], num(8, 'single_value')),
    'na_x_at_zero_weight_is_dropped': c([NA, 2, 3], [0, 1, 1], num('5/2', 'interpolated', 1)),
    'first_sorted_value': c([2, 1], ['1/1000000000000000000000000000000', 1], num(1, 'first_sorted_value', 0),
                            note='w[2]/wtotal rounds to 1, so wcum[1] = 1 - 1/2 reaches 1/2 at the first sorted value'),
    'inexact_grid': c(['1/10', '2/10', '3/10', '4/10', '5/10'], ['1/3', '1/7', '1/9', '1/11', '1/13']),
    'grid_41': c(list(range(-10, 31)), [f'1/{1 + (k - 15) ** 2}' for k in range(-10, 31)]),
    # NA paths
    'all_zero': c([1, 2, 3], [0, 0, 0], na('no_positive_weight')),
    'all_negative': c([1, 2, 3], [-1, -2, -3], na('no_positive_weight')),
    'minus_inf_without_plus_inf': c([1, 2, 3], [MINF, -1, 0], na('no_positive_weight')),
    'empty': c([], [], na('no_positive_weight')),
    'na_weight_last': c([1, 2, 3], [1, 1, NA], na('na_weight')),
    'nan_weight_first': c([1, 2, 3], [NAN, 1, 1], na('na_weight')),
    'na_x_at_positive_weight': c([1, NA, 3], [1, 1, 1], na('na_x_at_positive_weight')),
    'nan_x_at_positive_weight': c([1, NAN, 3], [1, 1, 1], na('na_x_at_positive_weight')),
    # +Inf weights
    'one_inf': c([1, 2, 3], [1, INF, 1], num(2, 'single_value')),
    'two_inf': c([1, 2, 3, 4], [INF, 1, INF, 1], num(2, 'equal_weights_even', 1)),
    'inf_odd_is_not_the_median': c([3, 1, 2], [INF, INF, INF], num(1, 'equal_weights_odd', 2),
                                   note='partitioned around index 2 but read at index 1; the true median is 2'),
    'inf_order_na_before': c([NA, 2], [1, INF], na('na_x_at_positive_weight')),
    'inf_order_na_after': c([2, NA], [INF, 1], num(2, 'single_value')),
    'minus_inf_counts_as_infinite': c([1, 2, 3], [MINF, INF, 1], num('3/2', 'equal_weights_even', 1)),
    'na_x_at_minus_inf': c([NA, 2, 3], [MINF, INF, 1], na('na_x_at_infinite_weight')),
    'inf_then_na_weight': c([1, 2], [INF, NA], na('na_weight_after_inf')),
    # infinite x (not in the source's grid, shows NaN versus NA)
    'infinite_x_gives_nan': c([MINF, INF], [1, 1], {'kind': 'NaN', 'value': None, 'path': 'interpolated', 'half': 1}),
    'infinite_x_gives_minus_inf': c([MINF, 1], [3, 1], {'kind': 'number', 'value': MINF, 'path': 'interpolated', 'half': 1}),
    # ties in x: the order comes from R_qsort_I; hand_order is traced by hand (0-based positions after compaction)
    'ties_trace': c([2, 1, 2, 1], [1, 2, 3, 4], None, hand_order=[1, 3, 2, 0],
                    note='Singleton partition traced by hand on the R text: the sorted weights are 2, 4, 3, 1'),
}

TF_CASE = {
    'x1range': [-5, 0, 5, 10],
    'x2range': [100, 200, 300, 400],
    'tf_x1': [[1, 2, 3, 2], [0, 0, 0, 0], [1, NA, 1, 1], [0, 0, 5, 0], [NAN, NAN, NAN, NAN]],
    'tf_x2': [[1, 1, 1, 1], [1, 2, 1, 1], [1, 1, 2, 1], [NA, 1, 1, 1], [2, 1, 1, 1]],
    'taxa': ['Kunsttaxon A', 'Kunsttaxon B', 'Kunsttaxon C', 'Kunsttaxon D', 'Kunsttaxon E'],
    'hand_which_tf_missing': [2, 3, 5],
    'hand_x1_kinds': ['number', 'NA', 'NA', 'number', 'NA'],
    'hand_x2_kinds': ['number', 'number', 'number', 'NA', 'number'],
    'note': 'row 4 of tf_x2 is NA but stays in the mask: line 1664 tests only the x1 medians',
}

UNDEFINED = {
    'overflow_total_weight': {'x': ['1', '2'], 'w': [f'{2 ** 1023}', f'{2 ** 1023}'],
                              'note': 'wtotal overflows to Inf, every w/wtotal is 0, half stays n + 1'},
}

STRESS = {'seed': 20261108, 'cases': 400, 'max_n': 120}


def main():
    fixture = {
        'kind': 'synthetic_weighted_median_fixture_not_a_transfer_function',
        'historical_reproduction_ready': False, 'selected_final_variant': None,
        'cases': CASES, 'tf_case': TF_CASE, 'undefined_cases': UNDEFINED, 'stress': STRESS,
    }
    (OUT / 'synthetic_fixture.json').write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + '\n', encoding='utf-8',
                                                newline='\n')
    print(len(CASES), 'cases')


if __name__ == '__main__':
    main()
