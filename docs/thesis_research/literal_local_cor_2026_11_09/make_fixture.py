"""Write synthetic_fixture.json: artificial cases for the local cor and its time-space readers. No real pollen, TF or climate value."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
NA, NAN = 'NA', 'NaN'
A, B = [1, 2, 4, 8], [3, 1, 7, 2]


def c(x, y, na_rm=False, hand=None, note=None):
    return {'x': [str(v) for v in x], 'y': [str(v) for v in y], 'na_rm': na_rm, 'hand': hand, 'note': note}


CASES = {
    'perfect_positive': c([1, 2, 3], [2, 4, 6], hand={'value': '1'}),
    'perfect_negative': c([1, 2, 3], [3, 2, 1], hand={'value': '-1'}),
    'hand_inexact': c(A, B, hand={'r_squared': '1/9545'}, note='cov 1/12, var_x 115/12, var_y 83/12: r = 1/sqrt(9545)'),
    'zero_covariance': c([1, 2, 3, 4], [1, -1, -1, 1], hand={'value': '0'}, note='exactly zero, is not replaced by 1e-10'),
    'x_na_kept': c([1, NA, 4, 8], B, hand={'kind': NA}),
    'x_na_removed': c([1, NA, 4, 8], B, na_rm=True, hand={'dropped': 1}),
    'x_nan_kept': c([1, NAN, 4, 8], B, hand={'kind': NAN}),
    'x_nan_removed': c([1, NAN, 4, 8], B, na_rm=True, hand={'dropped': 1}),
    'y_na_stays_with_na_rm': c(A, [3, NA, 7, 2], na_rm=True, hand={'kind': NA}, note='only x is screened'),
    'y_nan_stays_with_na_rm': c(A, [3, NAN, 7, 2], na_rm=True, hand={'kind': NAN}),
    'same_position_removed': c([1, NA, 4, 8], [3, NA, 7, 2], na_rm=True, hand={'dropped': 1}),
    'x_na_before_y_nan': c([1, NA, 4, 8], [3, 1, NAN, 2], hand={'kind': NA}, note='the first missing value of x decides'),
    'x_nan_before_y_na': c([1, NAN, 4, 8], [3, 1, NA, 2], hand={'kind': NAN}),
    'x_na_then_nan': c([NA, NAN, 4, 8], B, hand={'kind': NA}),
    'x_nan_then_na': c([NAN, NA, 4, 8], B, hand={'kind': NAN}),
    'x_all_missing_removed': c([NA, NA], [1, 2], na_rm=True, hand={'kind': NAN}, note='N = 0 after filtering'),
    'empty': c([], [], hand={'kind': NAN}, note='-0 / +0'),
    'single_value': c([2], [3], hand={'kind': NAN}, note='N - 1 is 0'),
    'constant_x_exact': c([5, 5, 5], [1, 2, 4], hand={'kind': NAN}, note='zero variance'),
    'constant_x_inexact': c(['1/10', '1/10', '1/10'], [1, 2, 4], note='the long double mean leaves a rounding residue; the result is a tiny number, not NaN'),
    'long_double_mean': c(['1', '0x1p-53', '0x1p-53'], [1, 2, 4], note='a double sum loses both small terms, the long double sum keeps them'),
    'x87_rounding_step': c(['1', '0x1p-53', '0x1p-65'], [1, 2, 4], note='an exact sum would round up to 1 + 2^-52; 64-bit steps drop 2^-65, then the cast meets a tie and gives 1'),
    'inexact_values': c(['1/3', '2/7', '3/11', '5/13', '7/17'], ['1/19', '2/23', '3/29', '5/31', '7/37']),
}

READERS = {
    'multiple_pann': {'multiple': True, 'tann': False, 'predictant_tann': [5, 6, 7, 9, 8], 'predictant_pann': [100, 300, 200, 400, 500],
                      'hi_ages_time': [False, True, True, True, True], 'ages_pollen': [True, True, True, True, True],
                      'predictors1': [1, 2, 3, 5, 4], 'ages_iso': [True, True, False, True, True], 'predictors2': [-1, 2, 3, 4],
                      'note': 'Kinneret shape: line 1964 reads pann, line 1966 still reads tann'},
    'multiple_tann': {'multiple': True, 'tann': True, 'predictant_tann': [5, 6, 7, 9, 8], 'predictant_pann': [100, 300, 200, 400, 500],
                      'hi_ages_time': [True, True, True, True, True], 'ages_pollen': [True, True, True, True, True],
                      'predictors1': [1, 2, 3, 5, 4], 'ages_iso': [True, True, True, True, True], 'predictors2': [1, 2, 3, 4, 6]},
    'mask_longer_than_selection': {'multiple': True, 'tann': False, 'predictant_tann': [5, 6, 7], 'predictant_pann': [1, 3, 2],
                                   'hi_ages_time': [True, True, True, True, True], 'ages_pollen': [True, True, True],
                                   'predictors1': [1, 2, 4], 'ages_iso': [True, True, True], 'predictors2': [1, 2, 4],
                                   'note': 'reading past the end gives NA in both vectors; na.rm then drops them'},
    'mask_shorter_recycled': {'multiple': True, 'tann': False, 'predictant_tann': [5, 6, 7, 9], 'predictant_pann': [1, 3, 2, 5],
                              'hi_ages_time': [True, False], 'ages_pollen': [True, True, True, True], 'predictors1': [1, 2, 4, 5],
                              'ages_iso': [True, True, True, True], 'predictors2': [1, 2, 4, 3]},
    'negative_replaced': {'multiple': True, 'tann': False, 'predictant_tann': [1, 2, 3], 'predictant_pann': [3, 2, 1],
                          'hi_ages_time': [True, True, True], 'ages_pollen': [True, True, True], 'predictors1': [1, 2, 3],
                          'ages_iso': [True, True, True], 'predictors2': [3, 2, 1]},
    'missing_replaced': {'multiple': True, 'tann': False, 'predictant_tann': [1, 2, 3], 'predictant_pann': [3, 1, 2],
                         'hi_ages_time': [True, True, True], 'ages_pollen': [True, True, True], 'predictors1': [1, NA, 3],
                         'ages_iso': [True, True, True], 'predictors2': [1, 1, 1]},
    'single_iso_1993': {'multiple': False, 'tann': False, 'sources_marine': 1, 'predictant_tann': [5, 6, 7, 9], 'predictant_pann': [1, 3, 2, 5],
                        'hi_ages_time': [True, True, True, True], 'ages_iso1': [True, True, False, True], 'ages_iso2': [False, True, True, True],
                        'iso_time': [0, 1, 2, 4], 'note': 'reached for Kinneret only if sources$core is 0'},
    'single_pollen_1999': {'multiple': False, 'tann': False, 'sources_marine': 0, 'predictant_tann': [5, 6, 7, 9], 'predictant_pann': [1, 3, 2, 5],
                           'hi_ages_time': [False, True, True, True], 'ages_pollen1': [True, True, True, True], 'ages_pollen2': [True, True, True, True],
                           'pollen': [4, 1, 3, 2], 'note': 'not reached for Kinneret: sources[,4] stays 1'},
}
READERS = {name: {k: ([str(i) if not isinstance(i, bool) else i for i in v] if isinstance(v, list) else v) for k, v in r.items()} for name, r in READERS.items()}
MISMATCH = {'x_longer_no_filter': {'x': [1, 2, 4, 8], 'y': [3, 1], 'na_rm': False}, 'y_longer_no_filter': {'x': [1, 2], 'y': [3, 1, 7, 2], 'na_rm': False},
            'y_longer_filter': {'x': [1, 2], 'y': [3, 1, 7, 2], 'na_rm': True}}


def main():
    fixture = {'kind': 'synthetic_local_cor_fixture_not_historical', 'historical_reproduction_ready': False, 'selected_final_variant': None,
               'cases': CASES, 'readers': READERS, 'length_mismatch': MISMATCH, 'stress': {'seed': 20261110, 'cases': 400}}
    (OUT / 'synthetic_fixture.json').write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    print(len(CASES), 'cases', len(READERS), 'readers')


if __name__ == '__main__':
    main()
