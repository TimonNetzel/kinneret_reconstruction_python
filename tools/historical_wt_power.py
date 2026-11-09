"""``wt_power`` of mcmc_functions_v1.6.R 1003-1012 and its reader 2023-2026, from an explicit ``wb`` matrix.

Explicit inputs only: no draw, no sampler, no original data. ``wb`` (a free global of the function, built by mcmc_prepare.R:932-935 through
biwavelet's ``rcpp_wt_bases_morlet``) is given as a real matrix, because that function is outside this step.

NOT BIT-IDENTICAL WITH R. By the user's decision of 2026-11-09 the transforms use NumPy's pocketfft instead of a translation of R's GPL
``fft.c``. Numbers differ from R 3.5.0 in the last bits, and the size of that difference is measured in the evidence packet, not assumed
away. Every result names its ``backend``. Lines 1004-1005 are exact: only the steps from ``fft`` on carry the deviation.

Arithmetic of R 3.5.0 x64 on Windows as far as it is reproduced:

- ``sum(d)`` accumulates in 80-bit long double and is cast to double once; ``/ l_d`` is a double division. ``l_d`` is not checked against
  ``length(d)`` in R, and mcmc_prepare.R:1817 recycles a shorter mask, so the two can differ.
- ``f * wb`` is a complex times a real matrix: ``(a + bi)(c + 0i)``. The inverse ``mvfft`` is unscaled, so ``invflen`` does the scaling.
- ``abs(wave)^2`` is ``hypot`` followed by ``x * x``, not ``re^2 + im^2`` (that is the unused ``wt_cpp`` of lines 1043-1090).

Missing values. A missing or infinite ``d`` makes every centred value non-finite, and one non-finite input reaches every output of both
transforms. Tokens are a measurement in R 3.5.0, no derivation: ``sum`` keeps the first missing token in order; a centred value keeps its
own missing token and otherwise takes the token of the mean. When all centred tokens agree the power carries that token; a mixture has
``kind`` 'mixed', and then only "missing" is claimed.

Rejected instead of emulated: ``wb`` that is complex, non-finite or not as long as ``c(x, zeros)``; ``n_obs`` outside ``1..nrow(wb)``; a
``coi_array`` of another shape or with values other than 0 and 1; an overflow on the way. The self-referencing defaults of the formals
(``n.obs=n.obs``) mean that R stops without them, so they are required here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np

from historical_local_cor import (NA, CorrelationWeight, LocalCor, UnsupportedSourceDomain, _divide, _is_missing, local_cor, logical_index,
                                  long_double_sum, post_treatment)

__all__ = ['UnsupportedSourceDomain', 'NA', 'BACKEND', 'WtPower', 'SpectralWeight', 'r_sum', 'centred', 'wt_power', 'select_coi',
           'length_ages_spec2', 'spectral_weight']

BACKEND = 'numpy.fft (pocketfft), not bit-identical with R fft.c'


@dataclass(frozen=True)
class WtPower:
    x: tuple  # line 1005, c(x, zeros): exact
    n: int
    n_obs: int
    columns: int
    dim: tuple[int, int] | None  # None when `wave[seq_len(n.obs), ]` dropped to a plain vector
    values: tuple  # column-major, as R stores them
    kind: str  # 'number', 'NA', 'NaN' or 'mixed'
    backend: str
    f: object  # complex ndarray, None unless kind is 'number'
    wave: object  # complex ndarray after line 1009, None unless kind is 'number'


@dataclass(frozen=True)
class SpectralWeight:
    power: WtPower
    predictant_spec: tuple
    correlation: LocalCor
    weight: CorrelationWeight
    post_spec: float


def _cell(name: str, v):
    if v is None or v is NA:
        return NA
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (Real, np.floating, np.integer)):
        raise UnsupportedSourceDomain(f'{name} must hold real numbers, NaN, Inf, or None/NA')
    return float(v)


def _cells(name: str, values) -> list:
    if isinstance(values, (str, bytes)) or (isinstance(values, np.ndarray) and values.ndim != 1):
        raise UnsupportedSourceDomain(f'{name} must be a flat numeric vector')
    return [_cell(name, v) for v in values]


def _count(name: str, v, minimum: int) -> int:
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < minimum:
        raise UnsupportedSourceDomain(f'{name} must be an integer of at least {minimum}')
    return int(v)


def r_sum(values) -> object:
    """``sum`` of R 3.5.0 in order: a missing accumulator stays, a missing term replaces a number or an infinity, opposite infinities give NaN."""
    values = list(values)
    state = None
    for v in values:
        if _is_missing(v):
            return v
        if math.isinf(v):
            if state is None:
                state = v
            elif state == -v:
                return math.nan
    return long_double_sum(values) if state is None else state


def centred(d, l_d, zeros) -> list:
    """Lines 1004 and 1005, exactly."""
    cells = _cells('d', d)
    _count('l_d', l_d, 1)
    total = r_sum(cells)
    mean = total if total is NA else _divide(total, float(l_d))
    out = []
    for v in cells:
        if v is NA or (isinstance(v, float) and math.isnan(v)):
            out.append(v)  # a missing left operand survives
        elif mean is NA:
            out.append(NA)
        else:
            out.append(v - mean)
    return out + [_cell('zeros', v) for v in zeros]


def _wb_matrix(wb, n: int) -> np.ndarray:
    matrix = np.asarray(wb)
    if matrix.dtype.kind == 'c':
        raise UnsupportedSourceDomain('wb must be real: rcpp_wt_bases_morlet returns a NumericVector')
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise UnsupportedSourceDomain('wb must be a matrix with at least one column, given as nrow x ncol')
    if matrix.shape[0] != n:
        raise UnsupportedSourceDomain('nrow(wb) differs from length(c(x, zeros)); R recycles or stops')
    if not np.isfinite(matrix).all():
        raise UnsupportedSourceDomain('wb holds a non-finite value, outside the checked domain')
    return matrix


def wt_power(d, l_d, *, n_obs, zeros, invflen, wb) -> WtPower:
    """Lines 1003-1012. Line 1007 (`test <- ...`) has no reader and is left out."""
    x = centred(d, l_d, zeros)
    n = len(x)
    if n == 0:
        raise UnsupportedSourceDomain('an empty series is outside the checked domain')
    matrix = _wb_matrix(wb, n)
    rows = _count('n_obs', n_obs, 1)
    if rows > n:
        raise UnsupportedSourceDomain('n.obs exceeds nrow(wb); R stops with a subscript error')
    scale = _cell('invflen', invflen)
    if scale is NA or not math.isfinite(scale):
        raise UnsupportedSourceDomain('invflen must be finite')
    columns = matrix.shape[1]
    dim = (rows, columns) if rows > 1 and columns > 1 else None
    size = rows * columns
    if not all(v is not NA and math.isfinite(v) for v in x):
        tokens = {'NA' if v is NA else 'NaN' for v in x if _is_missing(v)}
        kind = tokens.pop() if len(tokens) == 1 else 'mixed'
        filler = NA if kind == 'NA' else math.nan
        return WtPower(tuple(x), n, rows, columns, dim, tuple([filler] * size), kind, BACKEND, None, None)
    with np.errstate(over='ignore', invalid='ignore'):  # an overflow is rejected just below
        f = np.fft.fft(np.asarray(x, dtype=np.complex128))
        wave = np.fft.ifft(f[:, np.newaxis] * matrix, axis=0, norm='forward') * scale
        wave = wave[:rows, :]
        modulus = np.abs(wave)
        power = modulus * modulus
    if not (np.isfinite(f).all() and np.isfinite(wave).all() and np.isfinite(power).all()):
        raise UnsupportedSourceDomain('overflow in a transform or in the power, outside the checked domain')
    values = tuple(float(v) for v in power.flatten(order='F'))
    return WtPower(tuple(x), n, rows, columns, dim, values, 'number', BACKEND, f, wave)


def select_coi(power: WtPower, coi_array) -> tuple:
    """``power[coi_array == 1]`` in column-major order, for the n_obs x ncol matrix of 0 and 1 built in mcmc_prepare.R:903-910."""
    mask = np.asarray(coi_array, dtype=np.float64)
    if mask.shape != (power.n_obs, power.columns):
        raise UnsupportedSourceDomain('coi_array must be n.obs x ncol(wb)')
    if not np.isin(mask, (0.0, 1.0)).all():
        raise UnsupportedSourceDomain('coi_array must hold only 0 and 1')
    flags = mask.flatten(order='F') == 1.0
    return tuple(v for v, keep in zip(power.values, flags) if keep)


def length_ages_spec2(ages_spec, hi_ages_spec) -> int:
    """mcmc_prepare.R:1817, ``length(which(ages_spec[hi_ages_spec]))``. The shorter ``hi_ages_spec`` is recycled over the full ``ages_spec``."""
    return sum(1 for v in logical_index(ages_spec, hi_ages_spec) if v is True or isinstance(v, np.bool_) and bool(v))


def spectral_weight(*, predictant_current, ages_spec, hi_ages_spec, n_obs, zeros, invflen, wb, coi_array, iso_spec) -> SpectralWeight:
    """Lines 2023-2026 with the local ``cor`` of 2355-2366."""
    d = logical_index(logical_index(_cells('predictant_current', predictant_current), ages_spec), hi_ages_spec)
    power = wt_power(d, length_ages_spec2(ages_spec, hi_ages_spec), n_obs=n_obs, zeros=zeros, invflen=invflen, wb=wb)
    predictant_spec = select_coi(power, coi_array)
    correlation = local_cor(predictant_spec, iso_spec, na_rm=True)
    weight = post_treatment(correlation)
    return SpectralWeight(power, predictant_spec, correlation, weight, weight.value)
