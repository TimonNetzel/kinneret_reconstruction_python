"""Write synthetic_fixture.json: artificial matrices for reconst and the predictants. No real pollen, TF or climate value."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
NA, NAN, INF, MINF = 'NA', 'NaN', 'Inf', '-Inf'
E, TIE = '0x1p-53', ['1', '0x1p-53', '0x1p-53']
A = [['1/2', '1/4', '1/4'], ['0', '1/2', '1/2'], ['1/4', '1/4', '1/2'], ['1/8', '3/8', '1/2']]
TF = [['10', '100'], ['20', '200'], ['30', '300']]
ONES = [['1', '1'], ['1', '1'], ['1', '1']]
T, F = True, False
ALL = [T, T, T, T]


def c(core, mask, tf, hand=None, note=None, recent=True, tann=False, prior=('11/2', '600')):
    return {'core': [[str(v) for v in r] for r in core], 'mask': mask, 'tf': [[str(v) for v in r] for r in tf], 'hand': hand, 'note': note,
            'recent': recent, 'tann': tann, 'prior': list(prior)}


CASES = {
    'hand_all_rows': c(A, ALL, TF, hand={'tann': ['35/2', '25', '45/2', '95/4'], 'pann': ['175', '250', '225', '475/2'], 'youngest': ['35/2', '175']},
                       note='Kinneret shape: every age row, tf_sample num_taxa x 2'),
    'tann_is_read': c(A, ALL, TF, tann=True, note='line 1940 instead of 1942'),
    'recent_false_uses_prior': c(A, ALL, TF, recent=False, note='lines 1853/1854'),
    'preliminary_holocene_mask': c(A, [T, T, F, F], TF, hand={'tann': ['35/2', '25'], 'pann': ['175', '250']}, note='shape of line 1466 only; line 1522 replaces it in the live path'),
    'interior_mask_youngest_is_first_selected': c(A, [F, T, T, F], TF, hand={'tann': ['25', '45/2'], 'pann': ['250', '225'], 'youngest': ['25', '250']}),
    'one_row_selected_drops_to_vector': c(A, [F, T, F, F], TF, hand={'tann': ['25'], 'pann': ['250']}, note='matrix[mask,] drops to a vector; %*% takes it as a row'),
    'one_row_matrix': c([A[0]], [T], TF, hand={'tann': ['35/2'], 'pann': ['175']}),
    'one_taxon_many_rows': c([['1/2'], ['1/4'], ['1']], [T, T, T], [['8', '16']], hand={'tann': ['4', '2', '8'], 'pann': ['8', '4', '16']}, note='one column drops; %*% takes the vector as a column'),
    'one_by_one': c([['1/2']], [T], [['8', '16']], hand={'tann': ['4'], 'pann': ['8']}),
    'no_row_selected': c(A, [F, F, F, F], TF, note='0 x 2 result, then reconst[1,] stops R at line 1825'),
    'no_row_selected_one_taxon': c([['1'], ['1']], [F, F], [['8', '16']], note='0 x 1 drops to numeric(0); %*% gives 0 x 2'),
    'empty_mask': c(A, [], TF),
    'short_mask_recycled': c(A, [T, F], TF, hand={'tann': ['35/2', '45/2'], 'pann': ['175', '225']}),
    'long_mask_stops': c(A, [T, T, T, T, T], TF),
    'na_in_mask': c(A, [T, None, T, F], TF, note='NA selects a row of NA'),
    'non_conformable_matrix': c(A, ALL, [['1', '2'], ['3', '4']]),
    'non_conformable_vector': c(A, [T, F, F, F], [['1', '2'], ['3', '4']], note='one row of length 3 against nrow(tf) 2'),
    'tie_finite': c([TIE, ['1', '0', '0']], [T, T], ONES, note='double accumulation stays at 1; long double reaches 1 + 2^-52'),
    'tie_one_row_dgemv': c([TIE], [T], ONES, note='vector-matrix path, dgemv with TRANS T'),
    'tie_with_nan_in_other_row': c([TIE, [NAN, '0', '0']], [T, T], ONES, note='simple_matprod: still a double accumulation, the finite row is unchanged'),
    'tie_with_na_in_other_row': c([TIE, [NA, '0', '0']], [T, T], ONES),
    'tie_with_inf_in_other_row': c([TIE, [INF, '0', '0']], [T, T], ONES),
    'tie_with_nan_in_tf_column_1': c([TIE, ['1', '0', '0']], [T, T], [[NAN, '1'], ['1', '1'], ['1', '1']], note='NaN in one TF column sends both columns through simple_matprod'),
    'false_positive_nan_test': c([['0x1.8p+1023', '1'], ['0x1.8p+1023', '1']], [T, T], [['0x1p-1000', '1'], ['1', '1']], note='two huge neighbours overflow in the pairwise test; values stay finite'),
    'product_overflow_finite_inputs': c([['0x1p+1023', '0x1p+1023'], ['0x1p+1023', '-0x1p+1023']], [T, T], [['4', '1'], ['4', '1']], note='BLAS path; Inf and Inf - Inf arise inside'),
    'product_overflow_blas_path': c([['0x1p+600', '1'], ['1', '1']], [T, T], [['0x1p+600', '1'], ['1', '1']], note='no pair overflows in the test, so BLAS runs; the product itself is Inf'),
    'zero_times_na': c([['0', '1'], ['1', '1']], [T, T], [[NA, '1'], ['2', '3']], note='0 * NA stays NA'),
    'zero_times_inf': c([['0', '1'], ['1', '1']], [T, T], [[INF, '1'], ['2', '3']], note='0 * Inf is NaN'),
    'inf_minus_inf': c([[INF, MINF], ['1', '1']], [T, T], [['1', '1'], ['1', '1']]),
    'token_x_na_y_nan': c([[NA, '1'], ['1', '1']], [T, T], [[NAN, '1'], ['1', '1']], note='the first operand of x * y wins'),
    'token_x_nan_y_na': c([[NAN, '1'], ['1', '1']], [T, T], [[NA, '1'], ['1', '1']]),
    'token_sum_na_then_nan': c([[NA, NAN], ['1', '1']], [T, T], [['1', '1'], ['1', '1']], note='a missing accumulator stays'),
    'token_sum_nan_then_na': c([[NAN, NA], ['1', '1']], [T, T], [['1', '1'], ['1', '1']]),
    'token_sum_inf_cancel_then_na': c([[INF, MINF, NA], ['1', '1', '1']], [T, T], [['1', '1'], ['1', '1'], ['1', '1']], note='NaN from Inf - Inf comes first and stays'),
    'negative_zero_terms': c([['-0', '1'], ['0', '0']], [T, T], [['1', '1'], ['0', '-1']], note='the accumulator starts at +0'),
    'subnormal_terms': c([['0x1p-1074', '0x1p-1074'], ['0x1.8p-1073', '0x1p-1074']], [T, T], [['1', '3'], ['1', '1/2']]),
    'inexact_values': c([['1/3', '2/7', '3/11'], ['5/13', '7/17', '11/19']], [T, T], [['1/23', '2/29'], ['3/31', '5/37'], ['7/41', '11/43']]),
    'many_taxa_order_matters': c([['0x1p+60', '1', '-0x1p+60', '1']], [T], [['1', '1'], ['1', '1'], ['1', '1'], ['1', '1']], note='plain double accumulation in taxon order gives 1, an exact sum 2'),
}

# Lines 1939-2018 after the product; Kinneret shape and the conditional branch. Line 1999 stays marked as unreached.
CHAINS = {
    'kinneret_shape_multiple': {'core': A + [['1/4', '1/2', '1/4']], 'mask': [T] * 5, 'tf': TF, 'tann': False, 'multiple': True, 'hi_ages_time': [F, T, T, T, T],
                                'ages_pollen': [T] * 5, 'predictors1': ['1', '2', '3', '5', '4'], 'ages_iso': [T, T, F, T, T], 'predictors2': ['-1', '2', '3', '4'],
                                'note': 'artificial; line 1964 reads pann, 1966 reads tann'},
    'tann_true_multiple': {'core': A + [['1/4', '1/2', '1/4']], 'mask': [T] * 5, 'tf': TF, 'tann': True, 'multiple': True, 'hi_ages_time': [T] * 5,
                           'ages_pollen': [T] * 5, 'predictors1': ['1', '2', '3', '5', '4'], 'ages_iso': [T] * 5, 'predictors2': ['1', '2', '4', '3', '6']},
    'na_row_removed_by_cor': {'core': A + [[NA, '1/2', '1/4']], 'mask': [T] * 5, 'tf': TF, 'tann': False, 'multiple': True, 'hi_ages_time': [T] * 5,
                              'ages_pollen': [T] * 5, 'predictors1': ['1', '2', '3', '5', '4'], 'ages_iso': [T] * 5, 'predictors2': ['1', '2', '4', '3', '6'],
                              'note': 'the NA predictant is screened out in x by na.rm'},
    'nan_row_in_negative_correlation': {'core': [['1', '0', '0'], ['0', '1', '0'], ['0', '0', '1'], [NAN, '1/2', '1/2']], 'mask': [T] * 4, 'tf': TF, 'tann': False, 'multiple': True,
                                        'hi_ages_time': [T] * 4, 'ages_pollen': [T] * 4, 'predictors1': ['3', '2', '1', '5'], 'ages_iso': [T] * 4, 'predictors2': ['1', '2', '3', '9']},
    'single_iso_1993': {'core': A, 'mask': ALL, 'tf': TF, 'tann': False, 'multiple': False, 'sources_marine': 1, 'hi_ages_time': ALL,
                        'ages_iso1': [T, T, F, T], 'ages_iso2': [F, T, T, T], 'iso_time': ['0', '1', '2', '4'], 'note': 'reached for Kinneret only if sources$core is 0'},
    'single_pollen_1999': {'core': A, 'mask': ALL, 'tf': TF, 'tann': False, 'multiple': False, 'sources_marine': 0, 'hi_ages_time': [F, T, T, T],
                           'ages_pollen1': ALL, 'ages_pollen2': ALL, 'pollen': ['4', '1', '3', '2'], 'note': 'not reached for Kinneret: sources[,4] stays 1'},
}
CHAINS = {name: {k: ([[str(i) for i in r] for r in v] if k in ('core', 'tf') else v) for k, v in row.items()} for name, row in CHAINS.items()}


def main():
    fixture = {'kind': 'synthetic_reconst_fixture_not_historical', 'historical_reproduction_ready': False, 'selected_final_variant': None,
               'cases': CASES, 'chains': CHAINS, 'stress': {'seed': 20261111, 'cases': 400}}
    (OUT / 'synthetic_fixture.json').write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    print(len(CASES), 'cases', len(CHAINS), 'chains')


if __name__ == '__main__':
    main()
