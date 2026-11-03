"""Write the artificial Dirichlet-ratio fixture. All weights are invented, never historical data or rdirichlet draws.

Ratio cases give old_weights, new_weights and num_taxa; density cases give x and alpha for ddirichlet alone. Values are
fractions standing for the correctly rounded double, or decimal literals. Hand expectations follow from the Dirichlet
density Gamma(sum a)/prod Gamma(a_i) * prod x_i^(a_i - 1) and the source lines: identical weights give exactly 1,
closed forms are written as expressions in sqrt and pi, and NaN products become 0 in line 1273.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent

RATIO = {
    'identical_weights': (['1/4', '1/4', '1/2'], ['1/4', '1/4', '1/2'],
                          dict(exact={'weights_ratio': '1', 'proposal_ratio': '1', 'parameter_ratio': '1'},
                               note='same numerator and denominator computation: every ratio is exactly 1')),
    'two_taxa_closed_form': (['1/2', '1/2'], ['1/4', '3/4'],
                             dict(closed={'weights_ratio': '2/sqrt(3)', 'proposal_ratio': 'sqrt(6)/4',
                                          'parameter_ratio': '1/sqrt(2)'},
                                  note='weights: (1/4*3/4)^(-1/2) / (1/2*1/2)^(-1/2) = 2/sqrt(3); proposal: '
                                       '[sqrt(2)/pi] / [4/(sqrt(3)*pi)] = sqrt(6)/4; product 1/sqrt(2)')),
    'start_weights_to_proposal': (['1/5'] * 5, ['1/10', '3/20', '1/5', '1/4', '3/10'],
                                  dict(note='start weights rep(1/5, 5) of mcmc_conduction.R:113 as old weights')),
    'reset_weights_as_new': (['1/10', '1/5', '3/10', '2/5'], ['1/4'] * 4,
                             dict(note='new weights equal to the reset rep(1/4, 4) of line 1254 (reset not '
                                       'emulated)')),
    'unnormalised_weights_kept': (['1/4', '1/4', '1/2'], ['1/2', '1/2', '1'],
                                  dict(note='new weights sum to 2: the line x / rowSums(x) assigns nothing, so the '
                                            'densities use the unnormalised weights')),
    'overflow_nan_becomes_zero': (['1/3', '1/3', '1/3'], ['1e-310', '1e-310', '1'],
                                  dict(specials={'weights_ratio': 'Inf', 'proposal_ratio': '0',
                                                 'parameter_ratio_before_na': 'NaN', 'parameter_ratio': '0'},
                                       note='artificial edge with subnormal weights: ddirichlet(new, 0.5) overflows '
                                            'to Inf, ddirichlet(old, new) underflows to 0, so 0 * Inf = NaN and '
                                            'line 1273 sets 0')),
    'thirty_three_taxa': (['1/33'] * 33, [f'{k}/561' for k in range(1, 34)],
                          dict(note='33 artificial taxa (the size of the Kinneret paper case), new weights k/561')),
    'inexact_weights': (['0.1', '0.2', '0.7'], ['0.3', '0.3', '0.4'],
                        dict(note='decimal weights: row sums not exact, accumulator flags expected False')),
}
DENSITY = {
    'jeffreys_half_half': (['1/2', '1/2'], ['1/2', '1/2'], '2/pi'),
    'beta_two_two': (['1/4', '3/4'], ['2', '2'], '9/8'),
    'uniform_four': (['1/10', '1/5', '3/10', '2/5'], ['1', '1', '1', '1'], '6'),
}


def main():
    fixture = {
        'kind': 'synthetic_dirichlet_ratio_fixture_not_historical',
        'historical_reproduction_ready': False, 'selected_final_variant': None,
        'provider': 'LaplacesDemon::ddirichlet (bound within 2018-04-23..2023-07-07 only)',
        'note': 'Invented weights; rdirichlet is never called. Fractions stand for the correctly rounded double, '
                'decimal literals for the double nearest to them.',
        'ratio_cases': {k: {'old_weights': o, 'new_weights': n, 'num_taxa': len(o), 'hand': h}
                        for k, (o, n, h) in RATIO.items()},
        'density_cases': {k: {'x': x, 'alpha': a, 'hand_closed_form': c} for k, (x, a, c) in DENSITY.items()},
    }
    for k, c in fixture['ratio_cases'].items():
        assert len(c['old_weights']) == len(c['new_weights']), k
    (OUT / 'synthetic_fixture.json').write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + '\n',
                                                encoding='utf-8', newline='\n')
    print(len(RATIO), 'ratio cases,', len(DENSITY), 'density cases')


if __name__ == '__main__':
    main()
