"""The local ``cor`` of mcmc_functions_v1.6.R 2355-2366 and its time-space readers in the active reconstruction (1939-2018).

Explicit vectors only: no draw, no sampler, no original data. ``cor`` overrides stats::cor because main_v1.2.R:644 sources the
file into the global environment. It uses base R only.

Arithmetic as R 3.5.0 x64 on Windows performs it:

- ``sum`` accumulates in x87 ``long double`` (64 significant bits, round to nearest even) and casts to double once at the
  end, after comparing against DBL_MAX. Using a double sum here would change results, see the fixture.
- ``(x-mean_x) %*% (y-mean_y)`` on finite doubles is reference BLAS dgemv: a plain double accumulation in index order.
- ``v^2`` is ``v*v``; ``sqrt`` is correctly rounded.

Missing values. ``na.rm=TRUE`` looks only at ``x``: positions where ``x`` is NA or NaN go from both vectors, an NA in ``y``
stays and makes the result missing. Whether R then prints NA or NaN is a property of the x87/SSE operand order and is not in
the R source: measured in R 3.5.0, the first missing value of ``x`` decides, otherwise the first missing value of ``y``.
Empty and single-value input and zero variance give NaN. The reader cannot tell the two apart: ``NA < 0`` is NA and
``is.na`` is TRUE, so ``NA | TRUE`` is TRUE and the value becomes 1e-10 before squaring.

Rejected instead of emulated: infinite inputs, overflow, lengths that still differ after filtering, ``hi_time`` TRUE.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np

from historical_weighted_median import UnsupportedSourceDomain

__all__ = ['UnsupportedSourceDomain', 'NA', 'LocalCor', 'CorrelationWeight', 'TimeCorrelations', 'long_double_sum',
           'logical_index', 'local_cor', 'post_treatment', 'time_space_correlations']


class _NaType:
    def __repr__(self):
        return 'NA'


NA = _NaType()  # R's NA_real_; float('nan') stands for NaN
_DBL_MAX = int(1.7976931348623157e308)
_PRECISION = 64


@dataclass(frozen=True)
class LocalCor:
    value: float  # NaN also when kind is 'NA'
    kind: str  # 'number', 'NA' or 'NaN'
    n: int
    mean_x: float | None
    mean_y: float | None
    numerator: float | None
    sd_x: float | None
    sd_y: float | None
    x_used: tuple
    y_used: tuple


@dataclass(frozen=True)
class CorrelationWeight:
    correlation: float
    kind: str
    replaced: bool  # True when `< 0 | is.na` fired
    before_square: float
    value: float


@dataclass(frozen=True)
class TimeCorrelations:
    branch: str  # line of the active correlation call
    tann: bool
    predictant_read: str  # 'predictant_tann' or 'predictant_pann'
    pollen: CorrelationWeight | None
    iso: CorrelationWeight | None
    post_pollen: float
    post_iso: float
    post_iso_hi: float


def _is_missing(v) -> bool:
    return v is NA or (isinstance(v, float) and math.isnan(v))


def _value(name: str, v):
    if v is None or v is NA:
        return NA
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (Real, np.floating, np.integer)):
        raise UnsupportedSourceDomain(f'{name} must hold real numbers, NaN, or None/NA')
    v = float(v)
    if math.isinf(v):
        raise UnsupportedSourceDomain(f'{name}: infinite values are outside the checked domain')
    return v


def _vector(name: str, values) -> list:
    if isinstance(values, (str, bytes)) or (isinstance(values, np.ndarray) and values.ndim != 1):
        raise UnsupportedSourceDomain(f'{name} must be a flat numeric vector')
    return [_value(name, v) for v in values]


def logical_index(values, mask) -> list:
    """``values[mask]`` for a logical mask: a short mask is recycled, a long one reads NA beyond the end, NA in the mask gives NA."""
    values, mask = list(values), list(mask)
    if not mask:
        return []
    length = max(len(values), len(mask))
    out = []
    for i in range(length):
        flag = mask[i % len(mask)]
        if flag is None or flag is NA:
            out.append(NA)
        elif not isinstance(flag, (bool, np.bool_)):
            raise UnsupportedSourceDomain('a mask must hold booleans or None/NA')
        elif flag:
            out.append(values[i] if i < len(values) else NA)
    return out


def _round_64(mant: int, exp: int) -> tuple[int, int]:
    """Round the integer mantissa to 64 significant bits, nearest even."""
    bits = abs(mant).bit_length()
    if bits <= _PRECISION:
        return mant, exp
    drop = bits - _PRECISION
    magnitude = abs(mant)
    kept, rest, half = magnitude >> drop, magnitude & ((1 << drop) - 1), 1 << (drop - 1)
    if rest > half or (rest == half and kept & 1):
        kept += 1
    return (-kept if mant < 0 else kept), exp + drop


def long_double_sum(values) -> float:
    """R's rsum for finite doubles: each addition kept to 64 significant bits, one cast to double at the end."""
    mant, exp = 0, 0
    for v in values:
        numerator, denominator = float(v).as_integer_ratio()
        term_exp = 1 - denominator.bit_length()
        if mant == 0:
            mant, exp = numerator, term_exp
        else:
            shared = min(exp, term_exp)
            mant = (mant << (exp - shared)) + (numerator << (term_exp - shared))
            exp = shared
        if mant == 0:
            exp = 0
        else:
            mant, exp = _round_64(mant, exp)
    if mant == 0:
        return 0.0
    if exp >= 0:
        whole = mant << exp
        if abs(whole) > _DBL_MAX:
            raise UnsupportedSourceDomain('The sum exceeds DBL_MAX; R would give Inf, outside the checked domain')
        return float(whole)
    # Correctly rounded integer division; Python's int/int rounds half to even, also into the subnormal range
    if abs(mant) >> -exp > _DBL_MAX:
        raise UnsupportedSourceDomain('The sum exceeds DBL_MAX; R would give Inf, outside the checked domain')
    return mant / (1 << -exp)


def _divide(a: float, b: float) -> float:
    """IEEE division, also by zero."""
    if b != 0 or math.isnan(b):
        return a / b
    if a == 0 or math.isnan(a):
        return math.nan
    return math.copysign(math.inf, a) * math.copysign(1.0, b)


def _may_have_nan_or_inf(values: list[float]) -> bool:
    """mayHaveNaNOrInf of R 3.5.0 array.c, including its false positive when two neighbours overflow together."""
    n = len(values)
    if n & 1 and not math.isfinite(values[0]):
        return True
    return any(not math.isfinite(values[i] + values[i + 1]) for i in range(n & 1, n - 1, 2))


def local_cor(x, y, *, na_rm: bool = False) -> LocalCor:
    """Lines 2355-2366."""
    if not isinstance(na_rm, (bool, np.bool_)):
        raise UnsupportedSourceDomain('na_rm must be an explicit boolean; if(NA == TRUE) stops in R')
    xs, ys = _vector('x', x), _vector('y', y)
    if na_rm:
        keep = [not _is_missing(v) for v in xs]
        xs, ys = logical_index(xs, keep), logical_index(ys, keep)
    if len(xs) != len(ys):
        raise UnsupportedSourceDomain('x and y differ in length after filtering; R stops or returns a 1 x n matrix')
    n = len(ys)
    first = next((v for v in xs + ys if _is_missing(v)), None)
    if first is not None:
        kind = 'NA' if first is NA else 'NaN'
        return LocalCor(math.nan, kind, n, None, None, None, None, None, tuple(xs), tuple(ys))
    mean_x, mean_y = _divide(long_double_sum(xs), float(n)), _divide(long_double_sum(ys), float(n))
    dx, dy = [v - mean_x for v in xs], [v - mean_y for v in ys]
    if _may_have_nan_or_inf(dx) or _may_have_nan_or_inf(dy):
        raise UnsupportedSourceDomain('Overflow in the centred values; R leaves the BLAS path, outside the checked domain')
    total = 0.0
    for a, b in zip(dx, dy):
        total += b * a  # dgemv: y := y + temp * A(1,j) with temp = alpha * x(j)
    squares_x, squares_y = [v * v for v in dx], [v * v for v in dy]
    if not all(math.isfinite(v) for v in (total, *squares_x, *squares_y)):
        raise UnsupportedSourceDomain('Overflow in a product; outside the checked domain')
    divisor = float(n - 1)
    numerator = _divide(total, divisor)
    variance_x, variance_y = _divide(long_double_sum(squares_x), divisor), _divide(long_double_sum(squares_y), divisor)
    sd_x, sd_y = math.sqrt(variance_x), math.sqrt(variance_y)
    value = _divide(numerator, sd_x * sd_y)
    return LocalCor(value, 'NaN' if math.isnan(value) else 'number', n, mean_x, mean_y, numerator, sd_x, sd_y, tuple(xs), tuple(ys))


def post_treatment(result: LocalCor) -> CorrelationWeight:
    """``if(r < 0 | is.na(r)) r <- 1e-10`` and then ``r^2``, as in lines 1965/1967/1969/1970 and 1994/2000 and their squares."""
    replaced = result.kind != 'number' or result.value < 0
    before = 1e-10 if replaced else result.value
    return CorrelationWeight(result.value, result.kind, replaced, before, before * before)


def _sources_marine_is_one(value) -> bool:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Real, np.floating, np.integer)) or math.isnan(float(value)):
        raise UnsupportedSourceDomain('sources[,4] must be a number; if(NA) stops in R')
    return float(value) == 1


def time_space_correlations(*, predictant_tann, predictant_pann, tann: bool, multiple: bool, hi_ages_time, hi_time: bool = False,
                            ages_pollen=None, predictors1=None, ages_iso=None, predictors2=None, sources_marine=None,
                            ages_iso1=None, ages_iso2=None, iso_time=None, ages_pollen1=None, ages_pollen2=None,
                            pollen=None) -> TimeCorrelations:
    """Lines 1939-2018 for ``hi_time`` FALSE. The spectral block from 2021 is not part of this module."""
    for name, flag in (('tann', tann), ('multiple', multiple), ('hi_time', hi_time)):
        if not isinstance(flag, (bool, np.bool_)):
            raise UnsupportedSourceDomain(f'{name} must be an explicit boolean; if(NA) stops in R')
    if hi_time:
        raise UnsupportedSourceDomain('hi_time TRUE reads df_speleo (lines 1982, 2009); add.sites <- T sets it FALSE in mcmc_prepare.R:718')
    tann_values, pann_values = _vector('predictant_tann', predictant_tann), _vector('predictant_pann', predictant_pann)
    current = tann_values if tann else pann_values
    read = 'predictant_tann' if tann else 'predictant_pann'
    mask = list(hi_ages_time)
    if multiple:
        x_pollen = logical_index(logical_index(current, ages_pollen), mask)
        y_pollen = logical_index(_vector('predictors1', predictors1), mask)
        pollen_weight = post_treatment(local_cor(x_pollen, y_pollen, na_rm=True))
        # Line 1966 reads predictant_tann even where tann is FALSE
        x_iso = logical_index(tann_values, ages_iso)
        iso_weight = post_treatment(local_cor(x_iso, _vector('predictors2', predictors2), na_rm=True))
        return TimeCorrelations('1964+1966', bool(tann), read, pollen_weight, iso_weight, pollen_weight.value, iso_weight.value, 0.95)
    if _sources_marine_is_one(sources_marine):
        x_iso = logical_index(tann_values, ages_iso1)
        y_iso = logical_index(_vector('iso_time', iso_time), ages_iso2)
        iso_weight = post_treatment(local_cor(x_iso, y_iso, na_rm=True))
        return TimeCorrelations('1993', bool(tann), read, None, iso_weight, 0.5, iso_weight.value, 0.95)
    x_pollen = logical_index(logical_index(current, ages_pollen1), mask)
    y_pollen = logical_index(logical_index(_vector('pollen', pollen), ages_pollen2), mask)
    pollen_weight = post_treatment(local_cor(x_pollen, y_pollen, na_rm=True))
    return TimeCorrelations('1999', bool(tann), read, pollen_weight, None, pollen_weight.value, 0.5, 0.95)
