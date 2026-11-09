"""Write the artificial sampler-preparation fixture. All inputs are invented small numbers, never historical data.

Each case gives taxa.appear.core (rows = ages, columns = taxa; null stands for NA), num_taxa and the explicit gibbs
switch. The hand expectations follow from the source text alone and are written as fractions: values row by row
(c(t(.))), column maxima ignoring NA, NA kept, 0/0 = NaN and x/0 = -Inf for x < 0 at a zero maximum, colMeans over
the non-NA values, jeffreys_prior 0.5 per taxon and how_many_taxa = num_taxa (gibbs FALSE) or floor(num_taxa/2).
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
NA = None

# name -> (matrix, gibbs, hand: how_many_taxa, column max, values row by row, column means, accumulator independent)
SPEC = {
    'rowwise_flatten_three_by_four': (
        [[2, 1, 4, 3], [4, 3, 2, 6], [1, 6, 8, 9]], False,
        dict(how_many=4, max=['4', '6', '8', '9'],
             values=['1/2', '1/6', '1/2', '1/3', '1', '1/2', '1/4', '2/3', '1/4', '1', '1', '1'],
             means=['7/3', '10/3', '14/3', '6'], independent=[False, False, False, True],
             note='c(t(.)) runs along the rows; 1/6, 1/3, 2/3 and the means 7/3, 10/3, 14/3 are not exact doubles')),
    'na_skipped_in_mean_and_max': (
        [[1, NA, 2], [3, 4, NA], [NA, 2, 8], [4, 2, 2]], False,
        dict(how_many=3, max=['4', '4', '8'],
             values=['1/4', 'NA', '1/4', '3/4', '1', 'NA', 'NA', '1/2', '1', '1', '1/2', '1/4'],
             means=['8/3', '8/3', '4'], independent=[False, False, True],
             note='max and colMeans skip NA; NA stays NA after the division; na.rm = FALSE would give NA means')),
    'zero_column_gives_nan': (
        [[1, 0, 2], [2, 0, 4], [4, 0, 1]], False,
        dict(how_many=3, max=['4', '0', '4'],
             values=['1/4', 'NaN', '1/2', '1/2', 'NaN', '1', '1', 'NaN', '1/4'],
             means=['7/3', '0', '7/3'], independent=[False, True, False],
             note='a zero column has maximum 0: 0/0 gives NaN in every cell, its mean is 0')),
    'zero_max_negative_and_na': (
        [[0, -2, 3], [-2, -4, 1], [NA, -1, 2]], False,
        dict(how_many=3, max=['0', '-1', '3'],
             values=['NaN', '2', '1', '-Inf', '4', '1/3', 'NA', '1', '2/3'],
             means=['-1', '-7/3', '2'], independent=[True, False, True],
             note='artificial edge: at maximum 0, 0/0 is NaN, -2/0 is -Inf, NA stays NA; a negative maximum flips '
                  'the sign')),
    'gibbs_true_branch_odd_num_taxa': (
        [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], True,
        dict(how_many=2, max=['5', '4', '3', '4', '5'],
             values=['1/5', '1/2', '1', '1', '1', '1', '1', '1', '1/2', '1/5'],
             means=['3', '3', '3', '3', '3'], independent=[True, True, True, True, True],
             note='gibbs TRUE gives floor(5/2) = 2 and parameter_stage_gibbs; dead in the bound run (gibbs <- F)')),
    'weights_branch_odd_num_taxa': (
        [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], False,
        dict(how_many=5, max=['5', '4', '3', '4', '5'],
             values=['1/5', '1/2', '1', '1', '1', '1', '1', '1', '1/2', '1/5'],
             means=['3', '3', '3', '3', '3'], independent=[True, True, True, True, True],
             note='the bound branch: how_many_taxa = num_taxa = 5 and parameter_stage_weights')),
    'minimal_two_by_two': (
        [[1, 3], [2, 6]], False,
        dict(how_many=2, max=['2', '6'], values=['1/2', '1/2', '1', '1'], means=['3/2', '9/2'],
             independent=[True, True], note='smallest matrix without dimension drop')),
    'inexact_partial_sums_flagged': (
        [[0.1, 1], [0.2, 2], [0.3, 3]], False,
        dict(how_many=2, max=['0.3', '3'], values=None, means=None, independent=[False, True],
             note='0.1 + 0.2 is not an exact double: the mean of the first column depends on the accumulator; '
                  'values are the correctly rounded quotients of the stored doubles')),
}


def main():
    cases = {}
    for name, (matrix, gibbs, hand) in SPEC.items():
        ncol = len(matrix[0])
        assert all(len(row) == ncol for row in matrix), name
        cases[name] = {'taxa_appear_core': matrix, 'num_taxa': ncol, 'gibbs': gibbs,
                       'hand': {'how_many_taxa': hand['how_many'], 'how_many_taxa_line': 54 if gibbs else 57,
                                'parameter_stage': 'parameter_stage_gibbs' if gibbs else 'parameter_stage_weights',
                                'nrow_taxa_array': len(matrix), 'ncol_taxa_array': ncol,
                                'column_max': hand['max'], 'values': hand['values'], 'taxa_means': hand['means'],
                                'taxa_means_accumulator_independent': hand['independent'],
                                'jeffreys_prior': ['1/2'] * ncol, 'note': hand['note']}}
    fixture = {
        'kind': 'synthetic_sampler_preparation_fixture_not_historical',
        'historical_reproduction_ready': False, 'selected_final_variant': None,
        'note': 'Invented matrices. null stands for R NA. Hand values are fractions of the stored doubles; '
                '"NA", "NaN" and "-Inf" name the R results. A hand list of null is not written by hand.',
        'cases': cases,
    }
    (OUT / 'synthetic_fixture.json').write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + '\n',
                                                encoding='utf-8', newline='\n')
    print(f'{len(cases)} cases written')


if __name__ == '__main__':
    main()
