"""Write synthetic_fixture.json for wt_power. Every series, wb column and coi mask is invented; no pollen, speleothem or climate value."""
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent
NA, NAN, INF, MINF = 'NA', 'NaN', 'Inf', '-Inf'
D = ['1', '2', '3', '5']
PAD = 4


def hexes(values):
    return [float(v).hex() for v in values]


def wavenumbers(n, dt):
    """The vector of mcmc_prepare.R:926-928, written out for the artificial bands."""
    half = [(j * 2 * math.pi) / (n * dt) for j in range(1, n // 2 + 1)]
    return [0.0] + half + [-half[j - 1] for j in range((n - 1) // 2, 0, -1)]


def band(n, dt, scale):
    """Artificial real band with a Morlet-like shape. Written here on purpose: this is not biwavelet's rcpp_wt_bases_morlet."""
    out = []
    for k in wavenumbers(n, dt):
        out.append(math.exp(-(scale * k - 6.0) ** 2 / 2.0) * math.sqrt(2 * math.pi * scale / dt) * math.sqrt(n) if k > 0 else 0.0)
    return out


def bands(n, dt, m):
    return [hexes(band(n, dt, 2.0 * dt * 2 ** (j / 8))) for j in range(m)]


def c(d, n_obs, zeros, wb, invflen=None, l_d=None, hand=None, note=None):
    n = len(d) + zeros
    return {'d': [str(v) for v in d], 'l_d': len(d) if l_d is None else l_d, 'n_obs': n_obs, 'zeros': zeros, 'wb': [[str(v) for v in col] for col in wb],
            'invflen': str(invflen) if invflen is not None else f'1/{n}', 'hand': hand, 'note': note}


ONES, TWOS = ['1'] * 8, ['2'] * 8
X_SQ = ['49/16', '9/16', '1/16', '81/16']
SERIES_7 = ['0x1.4p+3', '0x1.8p+3', '0x1.2p+3', '0x1.cp+3', '0x1.6p+3', '0x1p+3', '0x1.ap+3']
SERIES_13 = [f'{(j * 37 % 13) + 0.25}' for j in range(13)]

CASES = {
    'identity_wb_two_columns': c(D, 4, PAD, [ONES, TWOS], hand={'power': X_SQ + ['49/4', '9/4', '1/4', '81/4'], 'dim': [4, 2]}, note='wb = 1 gives back x; a second column of 2 gives 4 x^2'),
    'one_column_drops': c(D, 4, PAD, [ONES], hand={'power': X_SQ, 'dim': None}, note='wave[seq_len(n.obs), ] drops to a vector'),
    'one_row_drops': c(D, 1, PAD, [ONES, TWOS], hand={'power': ['49/16', '49/4'], 'dim': None}, note='one row selected drops as well'),
    'one_by_one': c(D, 1, PAD, [ONES], hand={'power': ['49/16'], 'dim': None}),
    'zero_wb': c(D, 4, PAD, [['0'] * 8, ['0'] * 8], hand={'power': ['0'] * 8, 'dim': [4, 2]}),
    'constant_series': c(['7', '7', '7', '7'], 4, PAD, [ONES], hand={'power': ['0'] * 4, 'dim': None}, note='centring gives zeros'),
    'no_padding': c(D, 4, 0, [['1'] * 4], hand={'power': X_SQ, 'dim': None}, note='zeros is numeric(0)'),
    'l_d_differs_from_length': c(D, 4, PAD, [ONES], l_d=8, hand={'power': ['9/64', '25/64', '169/64', '841/64'], 'dim': None}, note='R never compares l_d with length(d)'),
    'invflen_not_one_over_n': c(D, 4, PAD, [ONES], invflen='1/4', hand={'power': ['49/4', '9/4', '1/4', '81/4'], 'dim': None}),
    'single_bin': c(D, 4, PAD, [['0', '1', '0', '0', '0', '0', '0', '0']], note='one wavenumber: the power is constant over the rows'),
    'long_double_in_the_mean': c(['1', '0x1p-53', '0x1p-53', '0x1p-53'], 4, PAD, [ONES], note='the 80-bit sum changes the mean'),
    'morlet_like_n16': c(SERIES_7, 7, 9, bands(16, 500, 5), note='shape of the live path: n = 2^ceiling(log2(n.obs) + 1); artificial band'),
    'morlet_like_n32': c(SERIES_13, 13, 19, bands(32, 500, 6), note='shape of the live path; artificial band'),
    'small_scale': c(['0x1p-500', '0x1.8p-500', '0x1.4p-500', '0x1.cp-500'], 4, PAD, [ONES]),
    'large_scale': c(['0x1p+500', '0x1.8p+500', '0x1.4p+500', '0x1.cp+500'], 4, PAD, [ONES]),
    'length_12_not_power_of_two': c(SERIES_7[:6], 6, 6, bands(12, 500, 2), note='outside the live path, which always has 2^k'),
    'prime_length_7': c(SERIES_7, 7, 0, bands(7, 500, 2), note='outside the live path'),
    'na_in_d': c(['1', NA, '3', '5'], 4, PAD, [ONES, TWOS], note='sum is NA: every centred value is NA'),
    'nan_in_d': c(['1', NAN, '3', '5'], 4, PAD, [ONES, TWOS]),
    'na_then_nan': c([NA, NAN, '3', '5'], 4, PAD, [ONES, TWOS], note='centred values NA and NaN: only "missing" is claimed'),
    'nan_then_na': c([NAN, NA, '3', '5'], 4, PAD, [ONES, TWOS]),
    'plus_inf': c(['1', INF, '3', '5'], 4, PAD, [ONES, TWOS], note='Inf - Inf is NaN at that position, -Inf elsewhere'),
    'minus_inf': c(['1', MINF, '3', '5'], 4, PAD, [ONES, TWOS]),
    'both_infinities': c([INF, MINF, '3', '5'], 4, PAD, [ONES, TWOS], note='the sum is NaN'),
    'inf_and_na': c([INF, NA, '3', '5'], 4, PAD, [ONES, TWOS]),
    'both_infinities_then_na': c([INF, MINF, NA, '5'], 4, PAD, [ONES, TWOS], note='the accumulator is already NaN when NA arrives'),
}

COI = [[1, 1], [1, 0], [0, 1], [1, 1]]


def chain(current, ages, hi, wb, coi, iso, zeros, n_obs, note=None):
    return {'predictant_current': [str(v) for v in current], 'ages_spec': ages, 'hi_ages_spec': hi, 'wb': [[str(v) for v in col] for col in wb], 'coi': coi,
            'iso_spec': [str(v) for v in iso], 'zeros': zeros, 'n_obs': n_obs, 'invflen': f'1/{len(wb[0])}', 'note': note}


T, F = True, False
SHAPE = {'zeros': PAD, 'n_obs': 4}
CHAINS = {
    'kinneret_shape_all_true': chain(D, [T, T, T, T], [T, T, T, T], [ONES, TWOS], COI, ['1', '2', '3', '4', '5', '7'], note='hi_ages_spec all TRUE as in line 885', **SHAPE),
    'ages_mask_picks_rows': chain(['9', '1', '2', '9', '3', '5'], [F, T, T, F, T, T], [T, T, T, T], [ONES, TWOS], COI, ['1', '2', '3', '4', '5', '7'], **SHAPE),
    'negative_correlation_is_replaced': chain(D, [T, T, T, T], [T, T, T, T], [ONES, TWOS], COI, ['7', '5', '4', '3', '2', '1'], note='line 2025 turns r < 0 into 1e-10', **SHAPE),
    'missing_series_is_replaced': chain(['1', NA, '3', '5'], [T, T, T, T], [T, T, T, T], [ONES, TWOS], COI, ['1', '2', '3', '4', '5', '7'], note='every power value is missing, na.rm drops all of them', **SHAPE),
    'one_column_selection': chain(D, [T, T, T, T], [T, T, T, T], [ONES], [[1], [1], [0], [1]], ['1', '3', '2'], **SHAPE),
    'morlet_like_chain': chain(SERIES_7, [T] * 7, [T] * 7, bands(16, 500, 5), [[1 if (i * j + i + 1) % 3 else 0 for j in range(5)] for i in range(7)],
                               [f'{(j * 7 % 11) + 0.5}' for j in range(sum(1 for i in range(7) for j in range(5) if (i * j + i + 1) % 3))], 9, 7, note='shape of the live path; artificial band'),
}

STRESS = {'seed': 20261109, 'count': 120, 'dt': 500, 'exponents': [3, 4, 5, 6, 7, 8, 9], 'max_columns': 10}

(OUT / 'synthetic_fixture.json').write_text(json.dumps({
    'kind': 'synthetic_wt_power_fixture', 'historical_reproduction_ready': False, 'selected_final_variant': None, 'cases': CASES, 'chains': CHAINS, 'stress': STRESS,
    'note': 'Invented numbers. The bands imitate the shape of a Morlet daughter but are not biwavelet output; no value comes from a run.',
}, indent=1) + '\n', encoding='utf-8', newline='\n')
print(len(CASES), 'cases', len(CHAINS), 'chains')
