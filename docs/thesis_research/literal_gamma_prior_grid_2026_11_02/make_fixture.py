"""Write the artificial gamma-grid fixture. All inputs are invented round numbers, never historical priors.

Hand expectations come from the shape of f(s) = |sqrt(s)/sd - s/mean| alone: its zero is s* = (mean/sd)^2,
near s* it rises with slope 1/(2 mean) on both sides, and the concave sqrt makes points below s* slightly
cheaper than points equally far above. Between 0 and s* f has a hump with maximum at (mean/(2 sd))^2,
so for s* beyond the grid the first grid point can beat the last one.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent

GRID_CASES = {
    'centre_25': dict(mean_pann=500, sd_pann=100, upper=50000.0, hand_r_index=250, hand_shape_approx=25.0,
                      hand='s* = 25 lies on the grid (0.1 * 250); rate = sqrt(25)/100 = 0.05'),
    'midpoint_6_25': dict(mean_pann=500, sd_pann=200, upper=50000.0, hand_r_index=62, hand_shape_approx=6.2,
                          hand='s* = 6.25 halfway between 6.2 and 6.3; concave sqrt favours the lower point'),
    'asymmetric_nearest_fails': dict(mean_pann=315, sd_pann=220, upper=50000.0, hand_r_index=20,
                                     hand_shape_approx=2.0,
                                     hand='s* = 2.0501: 2.1 is nearer, but 2.0 below s* has the smaller |difference|'),
    'beyond_upper_jumps_to_first': dict(mean_pann=2500, sd_pann=10, upper=50000.0, hand_r_index=1,
                                        hand_shape_approx=0.1,
                                        hand='s* = 62500 > 50000: f(0.1) = 0.0316 < f(50000) = 2.36, first point wins'),
    'below_first_point': dict(mean_pann=30, sd_pann=150, upper=50000.0, hand_r_index=1, hand_shape_approx=0.1,
                              hand='s* = 0.04 < 0.1: f increases from the first grid point'),
    'centre_25_upper_5000': dict(mean_pann=500, sd_pann=100, upper=5000.0, hand_r_index=250,
                                 hand_shape_approx=25.0, hand='same interior result with the mcmc_prep_v1.2.R bound'),
    'just_beyond_5000_last_point': dict(mean_pann=708, sd_pann=10, upper=5000.0, hand_r_index=50000,
                                        hand_shape_approx=5000.0,
                                        hand='s* = 5012.64 > 5000: f(5000) = 0.0089 < f(0.1) = 0.0315; last point, '
                                             'clamped by pmin to exactly 5000'),
    'same_input_upper_50000': dict(mean_pann=708, sd_pann=10, upper=50000.0, hand_r_index=50126,
                                   hand_shape_approx=5012.6, hand='interior again: 5012.6 is nearer and below s*'),
}

CHAIN_CASES = {
    'chain_recent_after_2000': dict(mean_pann=500, sd_pann=100, sd_tmp=1.5, recent=True, min_age_at_least_2000=True,
                                    hand='default shape 25; doubled sd 200 gives 6.2; rate 1733 = 6.2/500 = 0.0124'),
    'chain_recent_older': dict(mean_pann=500, sd_pann=100, sd_tmp=1.5, recent=True, min_age_at_least_2000=False,
                               hand='no doubling; final equals default; rate 1733 = shape/500'),
    'chain_not_recent': dict(mean_pann=500, sd_pann=100, sd_tmp=1.5, recent=False, min_age_at_least_2000=True,
                             hand='recent & ... is FALSE: no doubling, rate stays sqrt(shape)/sd from line 1281'),
}


def main():
    fixture = {
        'kind': 'synthetic_gamma_grid_fixture_not_historical',
        'historical_reproduction_ready': False, 'selected_final_variant': None,
        'note': 'Invented prior.recent values; historical inputs come from prior_climate.R fits that are not bound.',
        'grid_cases': GRID_CASES, 'chain_cases': CHAIN_CASES,
    }
    (OUT / 'synthetic_fixture.json').write_text(json.dumps(fixture, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(f'{len(GRID_CASES)} grid and {len(CHAIN_CASES)} chain cases written')


if __name__ == '__main__':
    main()
