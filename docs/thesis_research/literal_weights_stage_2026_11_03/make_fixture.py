"""Write the artificial weights-stage fixture. All inputs are invented small numbers, never historical data.

Each case gives taxa_information$values row by row (as prepared in mcmc_conduction.R 75), ncol, nrow and EXPLICIT
weights; rdirichlet is never called. Values and weights are written as fractions that stand for the correctly rounded
double ("1/3" is the double nearest to 1/3), or as the R specials "NA", "NaN", "-Inf". Four cases reuse the values of
a case of the sampler-preparation packet (state5_case). The hand expectations follow from the source text alone:
weight k mod ncol per position, rows refilled with byrow = T, rowSums skipping NA and NaN, every cell divided by the
sum of its own row. null in a hand list means "not written by hand" (a rounded product or sum).
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent

# name -> (values, nrow, ncol, weights, state5_case, hand)
SPEC = {
    'rowwise_recycling_two_by_three': (
        ['1', '1/2', '1/4', '1/2', '1', '1'], 2, 3, ['1/2', '1/4', '1/4'], None,
        dict(products=['1/2', '1/8', '1/16', '1/4', '1/4', '1/4'], row_sums=['11/16', '3/4'], counts=[3, 3],
             result=['8/11', '2/11', '1/11', '1/3', '1/3', '1/3'], independent=[True, True], zero_rows=[],
             note='weight k mod ncol meets its own taxon; exact row sums, each quotient one IEEE division')),
    'chain_minimal_two_by_two': (
        ['1/2', '1/2', '1', '1'], 2, 2, ['1/4', '3/4'], 'minimal_two_by_two',
        dict(products=['1/8', '3/8', '1/4', '3/4'], row_sums=['1/2', '1'], counts=[2, 2],
             result=['1/4', '3/4', '1/4', '3/4'], independent=[True, True], zero_rows=[],
             note='values of the smallest prepared matrix; both rows end in the weights')),
    'chain_three_by_four_inexact': (
        ['1/2', '1/6', '1/2', '1/3', '1', '1/2', '1/4', '2/3', '1/4', '1', '1', '1'], 3, 4,
        ['1/8', '3/8', '1/4', '1/4'], 'rowwise_flatten_three_by_four',
        dict(products=['1/16', None, '1/8', None, '1/8', '3/16', '1/16', None, '1/32', '3/8', '1/4', '1/4'],
             row_sums=[None, None, '29/32'], counts=[4, 4, 4],
             result=[None, None, None, None, None, None, None, None, '1/29', '12/29', '8/29', '8/29'],
             independent=[False, False, True], zero_rows=[],
             note='1/6, 1/3 and 2/3 are not exact doubles: the first two row sums are flagged')),
    'chain_zero_column_nan': (
        ['1/4', 'NaN', '1/2', '1/2', 'NaN', '1', '1', 'NaN', '1/4'], 3, 3, ['1/2', '1/4', '1/4'],
        'zero_column_gives_nan',
        dict(products=['1/8', 'NaN', '1/8', '1/4', 'NaN', '1/4', '1/2', 'NaN', '1/16'],
             row_sums=['1/4', '1/2', '9/16'], counts=[2, 2, 2],
             result=['1/2', 'NaN', '1/2', '1/2', 'NaN', '1/2', '8/9', 'NaN', '1/9'], independent=[True, True, True],
             zero_rows=[], note='NaN cells of a zero column are skipped by rowSums and stay NaN')),
    'na_and_nan_skipped_in_row_sum': (
        ['1/4', 'NA', '1/2', 'NaN', '1', '1/2', '1', '1/2', 'NA'], 3, 3, ['1/2', '1/4', '1/4'], None,
        dict(products=['1/8', 'NA', '1/8', 'NaN', '1/4', '1/8', '1/2', '1/8', 'NA'],
             row_sums=['1/4', '3/8', '5/8'], counts=[2, 2, 2],
             result=['1/2', 'NA', '1/2', 'NaN', '2/3', '1/3', '4/5', '1/5', 'NA'], independent=[True, True, True],
             zero_rows=[], note='na.rm skips NA and NaN; without na.rm every row would be NA or NaN')),
    'zero_row_sums': (
        ['0', '0', '0', 'NA', 'NaN', 'NA', '1/2', '-1/2', '0', '1', '1', '1'], 4, 3, ['1/4', '1/4', '1/2'], None,
        dict(products=['0', '0', '0', 'NA', 'NaN', 'NA', '1/8', '-1/8', '0', '1/4', '1/4', '1/2'],
             row_sums=['0', '0', '0', '1'], counts=[3, 0, 3, 3],
             result=['NaN', 'NaN', 'NaN', 'NA', 'NaN', 'NA', 'Inf', '-Inf', 'NaN', '1/4', '1/4', '1/2'],
             independent=[True, True, True, True], zero_rows=[1, 2, 3],
             note='artificial edge: a zero row, a row without values and a cancelling row sum to 0; 0/0 is NaN, '
                  '1/8 / 0 is Inf, -1/8 / 0 is -Inf, NA stays NA')),
    'chain_minus_inf': (
        ['NaN', '2', '1', '-Inf', '4', '1/3', 'NA', '1', '2/3'], 3, 3, ['1/4', '1/4', '1/2'],
        'zero_max_negative_and_na',
        dict(products=['NaN', '1/2', '1/2', '-Inf', '1', None, 'NA', '1/4', None],
             row_sums=['1', '-Inf', None], counts=[2, 3, 2],
             result=['NaN', '1/2', '1/2', 'NaN', '-0', '-0', 'NA', None, None], independent=[True, True, False],
             zero_rows=[], note='a row with -Inf sums to -Inf: -Inf/-Inf is NaN, positive values give -0')),
    'reset_weights_uniform_third': (
        ['1', '1/2', '1/4', '1', '1', '1'], 2, 3, ['1/3', '1/3', '1/3'], None,
        dict(products=['1/3', None, None, '1/3', '1/3', '1/3'], row_sums=[None, '1'], counts=[3, 3],
             result=[None, None, None, '1/3', '1/3', '1/3'], independent=[False, False], zero_rows=[],
             note='explicit weights equal to the reset rep(1/3, 3) of line 1254 (the reset itself is not emulated); '
                  'row 2: three times the double d nearest 1/3 is exactly 1 - 2^-54, halfway between two doubles, '
                  'rounded to the even 1, so every quotient is d again; the row is still flagged')),
    'scaled_weights_power_of_two': (
        ['1', '1/2', '1/4', '1/2', '1', '1'], 2, 3, ['1', '1/2', '1/2'], None,
        dict(products=['1', '1/4', '1/8', '1/2', '1/2', '1/2'], row_sums=['11/8', '3/2'], counts=[3, 3],
             result=['8/11', '2/11', '1/11', '1/3', '1/3', '1/3'], independent=[True, True], zero_rows=[],
             note='weights summing to 2 instead of 1: the stage does not renormalise the weights; a power-of-two '
                  'scale cancels exactly, so the result equals the first case')),
}


def main():
    cases = {}
    for name, (values, nrow, ncol, weights, state5, hand) in SPEC.items():
        assert len(values) == nrow * ncol and len(weights) == ncol, name
        assert all(len(hand[k]) == nrow * ncol for k in ('products', 'result')), name
        assert all(len(hand[k]) == nrow for k in ('row_sums', 'counts', 'independent')), name
        cases[name] = {'values': values, 'nrow_taxa_array': nrow, 'ncol_taxa_array': ncol, 'new_weights': weights,
                       'state5_case': state5,
                       'hand': {**hand, 'taxa_weights': weights, 'z': list(range(1, ncol + 1)),
                                'recycled_weight': [k % ncol + 1 for k in range(nrow * ncol)]}}
    fixture = {
        'kind': 'synthetic_weights_stage_fixture_not_historical',
        'historical_reproduction_ready': False, 'selected_final_variant': None, 'dirichlet_package': None,
        'note': 'Invented values and explicit weights; rdirichlet is never called. A fraction stands for the '
                'correctly rounded double; "NA", "NaN", "Inf", "-Inf" and "-0" name the R results. null in a hand '
                'list is not written by hand.',
        'cases': cases,
    }
    (OUT / 'synthetic_fixture.json').write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + '\n',
                                                encoding='utf-8', newline='\n')
    print(f'{len(cases)} cases written')


if __name__ == '__main__':
    main()
