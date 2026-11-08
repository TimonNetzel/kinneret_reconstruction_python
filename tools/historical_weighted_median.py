"""matrixStats::weightedMedian for the call of mcmc_prepare.R 1631/1651 and the NA mask of which.tf.missing (1664).

The provider is bound only within the CRAN window 2018-04-23..2023-07-07: matrixStats, first on the search path. The
13 versions there differ in the index plumbing, documentation and logical x; with comments removed and the index
expression written once, the C kernel is one single text (literal_weighted_median_2026_11_08/helper_comparison.json).
Translated from ``weightedMedian_lowlevel_template.h`` (matrixStats, Artistic-2.0, Henrik Bengtsson) for double x, w
with ``idxs = NULL``, ``na.rm = FALSE``, ``interpolate = TRUE``, ``ties = NULL``. Every double operation of the C code
is one IEEE operation here; nothing is accumulated in long double. The sort routines of R live in ``r_sort_gpl.py``.

When the result is NA, as read from the kernel and checked against R 3.5.0 + matrixStats 0.56.0:

* any NA/NaN weight anywhere gives NA, whatever x holds;
* without a +Inf weight: NA iff some x is NA/NaN at a finite weight > 0, or no weight is > 0;
* with a +Inf weight, first at position p: NA iff some x is NA/NaN at a finite weight > 0 before p, or at any
  weight +Inf or -Inf; an NA in x behind p at a finite weight is ignored, so the outcome depends on the order;
* x at a weight <= 0 is never looked at, unless a +Inf weight exists and the weight is -Inf;
* otherwise a number, or NaN (not NA) from Inf - Inf or 0 * Inf when x itself holds infinities.

Further source facts kept literally: -Inf weights count as infinite in the +Inf branch; the equal-weight branch reads
``x[half-1]`` for odd n after partitioning around ``half``, so it is not the median in general; ties in x are ordered by
the unstable ``R_qsort_I``. If the total weight overflows to Inf (or rounding keeps the cumulative weight below 1/2),
``half`` stays n + 1 and C reads outside its arrays; that undefined path is rejected, not emulated.

No R execution, no random state. Inputs: NA_real_ is None, NaN is float nan.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np

from r_sort_gpl import r_psort, r_qsort_i


class UnsupportedSourceDomain(ValueError):
    """Input needs R/runtime semantics outside this finite fragment."""


@dataclass(frozen=True)
class WeightedMedian:
    value: float  # NaN for both NA_real_ and NaN
    kind: str  # 'number', 'NA' (early return of NA_REAL) or 'NaN' (arithmetic on infinite x)
    is_na: bool  # R's is.na: TRUE for NA and NaN
    path: str
    n_used: int  # data points left after dropping
    total_weight: float | None  # wtotal; None where the kernel returned before computing it
    half: int | None  # 0-based index where the cumulative weight reached 1/2; None on other paths


def _double(name: str, v) -> float:
    if v is None:
        return math.nan
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (Real, np.floating, np.integer)):
        raise UnsupportedSourceDomain(f'{name} must hold real numbers or None for NA')
    return float(v)


def _vector(name: str, values) -> list[float]:
    if isinstance(values, (str, bytes)) or (isinstance(values, np.ndarray) and values.ndim != 1):
        raise UnsupportedSourceDomain(f'{name} must be a flat numeric vector')
    return [_double(name, v) for v in values]


def _na(path: str, n_used: int = 0) -> WeightedMedian:
    return WeightedMedian(math.nan, 'NA', True, path, n_used, None, None)


def _done(res: float, path: str, n_used: int, total: float | None = None, half: int | None = None) -> WeightedMedian:
    return WeightedMedian(res, 'NaN' if math.isnan(res) else 'number', math.isnan(res), path, n_used, total, half)


def weighted_median(x, w) -> WeightedMedian:
    """``weightedMedian(x, w)`` with all defaults, for one pair of double vectors."""
    x, w = _vector('x', x), _vector('w', w)
    if len(x) != len(w):
        raise UnsupportedSourceDomain("'x' and 'w' are of different lengths (the C entry point stops)")
    n = len(x)
    wtmp = [0.0] * n
    nxt, equal_weights = 0, False
    for ii in range(n):
        wtmp[ii] = 0.0
        weight = w[ii]
        if math.isnan(weight):
            return _na('na_weight')
        if weight <= 0:
            continue  # drop non-positive weights; x[ii] is not read
        if math.isinf(weight):
            nxt = 0
            for jj in range(n):
                wtmp[jj] = 0.0
                weight = w[jj]
                if math.isinf(weight):  # true for -Inf as well, literally
                    if math.isnan(x[jj]):
                        return _na('na_x_at_infinite_weight')
                    wtmp[jj] = 1.0
                    nxt += 1
                elif math.isnan(weight):
                    return _na('na_weight_after_inf')
            equal_weights = True
            break
        if math.isnan(x[ii]):
            return _na('na_x_at_positive_weight')
        wtmp[ii] = weight
        nxt += 1
    if nxt == 0:
        return _na('no_positive_weight')

    xtmp, wnew, wtotal = [], [], 0.0
    for ii in range(n):
        if wtmp[ii] > 0:
            xtmp.append(x[ii])
            wnew.append(wtmp[ii])
            wtotal += wtmp[ii]
    nx = nxt
    if nx == 1:
        return _done(xtmp[0], 'single_value', 1, wtotal)
    if equal_weights:
        half = (nx + 1) // 2
        r_psort(xtmp, nx, half)
        if nx % 2 == 1:
            return _done(xtmp[half - 1], 'equal_weights_odd', nx, wtotal, half)
        r_psort(xtmp, half, half - 1)
        return _done((xtmp[half - 1] + xtmp[half]) / 2, 'equal_weights_even', nx, wtotal, half)

    idxs = list(range(nx))
    r_qsort_i(xtmp, idxs)
    wcum = [0.0] * nx
    tmp_d2, half = 0.0, nx + 1  # "default is last" in the source
    for ii in range(nx):
        tmp_d = wnew[idxs[ii]] / wtotal
        tmp_d2 += tmp_d
        wcum[ii] = tmp_d2 - (tmp_d / 2)
        if wcum[ii] >= 0.5:
            half = ii
            break
    if half == 0:  # half == nx is unreachable: the loop only leaves half < nx or nx + 1
        return _done(xtmp[0], 'first_sorted_value', nx, wtotal, 0)
    if half == nx + 1:
        raise UnsupportedSourceDomain('the cumulative weight never reached 1/2 (total weight overflow or rounding): '
                                      'the C code reads outside its arrays, undefined and not emulated')
    dx = xtmp[half] - xtmp[half - 1]
    big_d = wcum[half] - wcum[half - 1]  # > 0: wcum[half] >= 1/2 > wcum[half - 1]
    dy = 0.5 - wcum[half]
    dx = (dy / big_d) * dx
    return _done(dx + xtmp[half], 'interpolated', nx, wtotal, half)


def tf_row_medians(x_range, tf_rows) -> tuple[WeightedMedian, ...]:
    """The loops 1629-1640 / 1649-1660 with ``taxa_change`` FALSE: one weightedMedian per row of the TF matrix."""
    grid = _vector('x_range', x_range)
    if isinstance(tf_rows, (str, bytes)) or len(tf_rows) == 0:
        raise UnsupportedSourceDomain('The TF matrix must have at least one row (1:nrow would count down otherwise)')
    out = []
    for row in tf_rows:
        values = _vector('tf row', row)
        if len(values) != len(grid):
            raise UnsupportedSourceDomain('Every TF row needs one value per grid point')
        out.append(weighted_median(grid, values))
    return tuple(out)


def which_tf_missing(x1_medians) -> tuple[int, ...]:
    """``which(is.na(tf.x1.current.mean))`` (1664), 1-based; only the x1 medians are tested in the source."""
    return tuple(i + 1 for i, m in enumerate(x1_medians) if m.is_na)
