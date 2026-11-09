"""``ratios_tfs_no_ages`` of mcmc_functions_v1.6.R 1485-1525 and its reader mcmc_conduction.R:228-230, from explicit vectors and matrices.

Explicit inputs only: no draw, no sampler, no original data. The definition at line 1485 is the one that counts: only its argument names fit
the call in mcmc_conduction.R:228 (line 1437 defines the same name with other arguments).

NOT BIT-IDENTICAL WITH R. By the user's decision of 2026-11-09 ``d_truncnorm`` (RcppDist 0.1.1, truncnorm.h 142-152) uses its own normal
density and distribution function, written from the textbook formulas on ``math.exp`` and ``math.erfc``, instead of a translation of R's GPL
``dnorm.c`` and ``pnorm.c``. Numbers differ from R 3.5.0 in the last bits; the size of that difference is measured in the evidence packet, not
assumed away. Every result names its ``backend``. The grid part of lines 1507-1513 involves no normal function and carries no such deviation.

- A grid index is the first minimum of ``|range - value|``: ``which_min`` compares with a strict ``<`` (Rcpp, 19 versions of the window, same text).
- Each ratio is an IEEE double division, so 0/0 is NaN and x/0 is infinite.
- Density of the proposal: ``d_truncnorm(old, new, ...)`` over ``d_truncnorm(new, old, ...)``. A value outside ``[a, b]`` has density 0.
- ``prod(unlist(ratios_tfs))`` (mcmc_conduction.R:230) multiplies in 80-bit long double, in the order tann_tf, pann_tf, tann_dists, pann_dists,
  and casts to double once. Above DBL_MAX it gives Inf. This arithmetic is exact here; only its factors carry the deviation.

Rejected instead of emulated: non-finite or missing inputs (``which_min`` returns NA on a missing value and the later subscript stops in Rcpp,
which cannot be run here without a compiler), ``sigma <= 0``, ``a > b``, vectors or matrices whose size differs from ``num_taxa`` and the grid,
and a long double product leaving the range of the 80-bit format.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from numbers import Real

import numpy as np

from historical_local_cor import UnsupportedSourceDomain, _divide

__all__ = ['UnsupportedSourceDomain', 'BACKEND', 'TfRatios', 'DensityParts', 'dnorm_own', 'pnorm_own', 'density_parts', 'd_truncnorm', 'grid_index', 'ratios_tfs_no_ages',
           'unlisted', 'long_double_prod', 'transfer_ratio']

BACKEND = 'math.exp and math.erfc (own textbook formulas), not bit-identical with R dnorm.c and pnorm.c'
_ONE_OVER_SQRT_2PI = 0.39894228040143267794  # 1 / sqrt(2 * pi)
_SQRT_2 = 1.41421356237309504880
_DBL_MAX = Fraction(1.7976931348623157e308)
_PRECISION = 64
_MAX_EXPONENT, _MIN_EXPONENT = 16383, -16382  # normal range of the x87 extended format


@dataclass(frozen=True)
class DensityParts:
    density: float  # the normal density at x
    p_a: float
    p_b: float
    scale: float  # p_b - p_a
    value: float  # d_truncnorm


@dataclass(frozen=True)
class TfRatios:
    num_taxa: int
    tann_tf_ratio: tuple
    pann_tf_ratio: tuple
    tann_dists_tf_ratio: tuple
    pann_dists_tf_ratio: tuple
    indices: tuple  # zero-based grid indices of (new tann, old tann, new pann, old pann), per taxon
    parts: tuple  # DensityParts of (propose tann, start tann, propose pann, start pann), per taxon
    backend: str


def _finite(name: str, v) -> float:
    if v is None or isinstance(v, (bool, np.bool_)) or not isinstance(v, (Real, np.floating, np.integer)):
        raise UnsupportedSourceDomain(f'{name} must hold real numbers')
    v = float(v)
    if not math.isfinite(v):
        raise UnsupportedSourceDomain(f'{name} holds a non-finite value, outside the checked domain')
    return v


def _vector(name: str, values, length: int | None = None) -> list:
    if isinstance(values, (str, bytes)) or (isinstance(values, np.ndarray) and values.ndim != 1):
        raise UnsupportedSourceDomain(f'{name} must be a flat numeric vector')
    out = [_finite(name, v) for v in values]
    if not out:
        raise UnsupportedSourceDomain(f'{name} is empty, outside the checked domain')
    if length is not None and len(out) != length:
        raise UnsupportedSourceDomain(f'{name} must have length {length}')
    return out


def _rows(name: str, rows, nrow: int, ncol: int) -> list:
    rows = list(rows)
    if len(rows) != nrow:
        raise UnsupportedSourceDomain(f'{name} must have {nrow} rows')
    return [_vector(name, row, ncol) for row in rows]


def _check_density(x, mu, sigma, a, b) -> tuple:
    x, mu, sigma, a, b = (_finite(n, v) for n, v in (('x', x), ('mu', mu), ('sigma', sigma), ('a', a), ('b', b)))
    if sigma <= 0:
        raise UnsupportedSourceDomain('sigma must be positive')
    if a > b:
        raise UnsupportedSourceDomain('a exceeds b, outside the checked domain')
    return x, mu, sigma, a, b


def dnorm_own(x: float, mu: float, sigma: float) -> float:
    """Normal density from the textbook formula. Not R's dnorm4."""
    z = (x - mu) / sigma
    return _ONE_OVER_SQRT_2PI * math.exp(-0.5 * z * z) / sigma


def pnorm_own(x: float, mu: float, sigma: float) -> float:
    """Normal distribution function through the complementary error function. Not R's pnorm5."""
    return 0.5 * math.erfc(-(x - mu) / sigma / _SQRT_2)


def density_parts(x, mu, sigma, a, b) -> DensityParts:
    """``d_truncnorm`` with its intermediate values."""
    x, mu, sigma, a, b = _check_density(x, mu, sigma, a, b)
    density, p_a, p_b = dnorm_own(x, mu, sigma), pnorm_own(a, mu, sigma), pnorm_own(b, mu, sigma)
    scale = p_b - p_a
    return DensityParts(density, p_a, p_b, scale, 0.0 if x < a or x > b else _divide(density, scale))


def d_truncnorm(x, mu, sigma, a, b) -> float:
    return density_parts(x, mu, sigma, a, b).value


def grid_index(grid, value) -> int:
    """Zero-based first minimum of ``|grid - value|``, as ``which_min(abs(grid - value))``."""
    grid, value = _vector('range', grid), _finite('value', value)
    best, index = abs(grid[0] - value), 0
    for i, g in enumerate(grid):
        distance = abs(g - value)
        if distance < best:
            best, index = distance, i
    return index


def ratios_tfs_no_ages(num_taxa, *, tann_range, pann_range, new_tann, old_tann, tann_dists, new_pann, old_pann, pann_dists,
                       lower_tann, upper_tann, sds_tann_trunc, lower_pann, upper_pann, sds_pann_trunc) -> TfRatios:
    """Lines 1485-1525."""
    if isinstance(num_taxa, (bool, np.bool_)) or not isinstance(num_taxa, (int, np.integer)) or num_taxa < 1:
        raise UnsupportedSourceDomain('num_taxa must be a positive integer')
    n = int(num_taxa)
    t_grid, p_grid = _vector('tann_range', tann_range), _vector('pann_range', pann_range)
    t_dists, p_dists = _rows('tann_dists', tann_dists, n, len(t_grid)), _rows('pann_dists', pann_dists, n, len(p_grid))
    new_t, old_t, new_p, old_p = (_vector(k, v, n) for k, v in (('new_tann', new_tann), ('old_tann', old_tann), ('new_pann', new_pann), ('old_pann', old_pann)))
    low_t, up_t, sd_t = (_vector(k, v, n) for k, v in (('lower_tann', lower_tann), ('upper_tann', upper_tann), ('sds_tann_trunc', sds_tann_trunc)))
    low_p, up_p, sd_p = (_vector(k, v, n) for k, v in (('lower_pann', lower_pann), ('upper_pann', upper_pann), ('sds_pann_trunc', sds_pann_trunc)))
    tf_t, tf_p, dists_t, dists_p, indices, parts = [], [], [], [], [], []
    for i in range(n):
        a, b, c, d = grid_index(t_grid, new_t[i]), grid_index(t_grid, old_t[i]), grid_index(p_grid, new_p[i]), grid_index(p_grid, old_p[i])
        indices.append((a, b, c, d))
        tf_t.append(_divide(t_dists[i][a], t_dists[i][b]))
        tf_p.append(_divide(p_dists[i][c], p_dists[i][d]))
        propose_t, start_t = density_parts(old_t[i], new_t[i], sd_t[i], low_t[i], up_t[i]), density_parts(new_t[i], old_t[i], sd_t[i], low_t[i], up_t[i])
        propose_p, start_p = density_parts(old_p[i], new_p[i], sd_p[i], low_p[i], up_p[i]), density_parts(new_p[i], old_p[i], sd_p[i], low_p[i], up_p[i])
        parts.append((propose_t, start_t, propose_p, start_p))
        dists_t.append(_divide(propose_t.value, start_t.value))
        dists_p.append(_divide(propose_p.value, start_p.value))
    return TfRatios(n, tuple(tf_t), tuple(tf_p), tuple(dists_t), tuple(dists_p), tuple(indices), tuple(parts), BACKEND)


def unlisted(result: TfRatios) -> list:
    """``unlist`` of the returned list: the four vectors one after another."""
    return [*result.tann_tf_ratio, *result.pann_tf_ratio, *result.tann_dists_tf_ratio, *result.pann_dists_tf_ratio]


def long_double_prod(values) -> float:
    """R's rprod for doubles: each multiplication kept to 64 significant bits, nearest even, one cast to double at the end."""
    mant, exponent, sign, kind = 1, 0, 1, 'finite'
    for v in values:
        v = float(v)
        if math.isnan(v):
            return math.nan
        sign *= -1 if math.copysign(1.0, v) < 0 else 1
        if math.isinf(v):
            if kind == 'zero':
                return math.nan
            kind = 'inf'
        elif v == 0:
            if kind == 'inf':
                return math.nan
            kind = 'zero'
        elif kind == 'finite':
            numerator, denominator = abs(v).as_integer_ratio()
            mant *= numerator
            exponent -= denominator.bit_length() - 1
            excess = mant.bit_length() - _PRECISION
            if excess > 0:
                low, half = mant & ((1 << excess) - 1), 1 << (excess - 1)
                mant >>= excess
                exponent += excess
                if low > half or (low == half and mant & 1):
                    mant += 1
            top = exponent + mant.bit_length() - 1
            if not _MIN_EXPONENT <= top <= _MAX_EXPONENT:
                raise UnsupportedSourceDomain('the long double product leaves the 80-bit range, outside the checked domain')
    if kind == 'zero':
        return math.copysign(0.0, sign)
    if kind == 'inf':
        return math.copysign(math.inf, sign)
    s = Fraction(mant) * (Fraction(2) ** exponent)
    return math.copysign(math.inf, sign) if s > _DBL_MAX else sign * float(s)


def transfer_ratio(result: TfRatios) -> float:
    """mcmc_conduction.R:230."""
    return long_double_prod(unlisted(result))
