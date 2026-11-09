"""Deterministic core of parameter_stage_weights, mcmc_functions_v1.6.R 1259 and 1276-1280; no sampler is run.

``new_weights`` is an EXPLICIT input. In the source it comes from ``rdirichlet`` (1248, random state, provider
package unbound) or from the reset ``rep(1/num_taxa, num_taxa)`` after 1000 draws (1253-1257); neither is emulated,
and ``ddirichlet``/``parameter_ratio`` (1269-1274) are not computed.

Kept literally: ``posterior_temp$taxa_weights <- new_weights`` (1259); ``taxa_information$values * new_weights``
(1276) recycles the ncol weights over the row-wise flattened values, so position k gets weight k mod ncol, i.e. the
weight of its own taxon; ``matrix(..., ncol, nrow, byrow = T)`` refills the rows; ``rowSums(na.rm = T)`` (1277)
skips NA and NaN, a row without values sums to 0 and a row with -Inf sums to -Inf; the matrix divided by the row-sum
vector recycles that vector down the columns, so every cell is divided by the sum of its own row (one IEEE division:
NA stays NA, NaN stays NaN, 0/0 is NaN, x/0 is +-Inf, x/-Inf is a signed zero, -Inf/-Inf is NaN);
``z <- 1:ncol(...)`` (1280).

Values use the representation of tools/historical_sampler_preparation.py: None is R NA, float NaN is R NaN, -inf is
-Inf. R may accumulate a row sum in extended precision (platform and R version unbound), so every row sum is the
correctly rounded exact sum and carries a flag: it is accumulator and order independent when all kept values are
multiples of a common power of two 2^e and their absolute sum stays below 2^(e+53), so that every partial sum in any
order is an exact double. Rejected instead of emulated: +Inf and negative zero in the values (not produced by the
prepared taxa_information), a length other than nrow*ncol (``matrix`` would recycle or truncate), fewer than two rows
or columns, weights that are not one finite positive double per column (the loop 1249 only leaves with
min(new_weights) > 0), and products or quotients that overflow or underflow to zero. No R execution, no random state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from numbers import Real

import numpy as np

MAX_COUNT = 2**31 - 1  # R integer dimensions
NA = None


class UnsupportedSourceDomain(ValueError):
    """Input needs R/runtime semantics outside this finite fragment."""


@dataclass(frozen=True)
class RowSum:
    value: float  # rowSums(, na.rm = T): 0.0 for a row without values, -inf for a row with -Inf
    exact: Fraction | None  # None for -Inf
    count: int  # cells kept after na.rm (neither NA nor NaN)
    accumulator_independent: bool  # every partial sum in any order is an exact double


@dataclass(frozen=True)
class WeightsStage:
    taxa_weights: tuple[float, ...]  # 1259: posterior_temp$taxa_weights <- new_weights
    recycled_weight: tuple[int, ...]  # 1276: one-based weight index per position of values (k mod ncol + 1)
    products: tuple[float | None, ...]  # taxa_information$values * new_weights; None is NA
    unnormalised: tuple[tuple[float | None, ...], ...]  # 1276: matrix(..., byrow = T)
    row_sums: tuple[RowSum, ...]  # 1277: rowSums(taxa.appear.core.temp, na.rm = T)
    taxa_appear_core_temp: tuple[tuple[float | None, ...], ...]  # 1277: posterior_temp$taxa.appear.core.temp
    zero_sum_rows: tuple[int, ...]  # one-based rows with row sum 0
    z: tuple[int, ...]  # 1280: 1:ncol


def _count(name: str, value) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise UnsupportedSourceDomain(f"{name} must be an integer count")
    value = int(value)
    if not 2 <= value <= MAX_COUNT:
        raise UnsupportedSourceDomain(f"{name} outside 2..2^31-1: matrix() would drop or misshape")
    return value


def _real(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Real, np.floating, np.integer)):
        raise UnsupportedSourceDomain(f"{name} must be real numbers (no bool or string)")
    return float(value)


def _values(values, size: int) -> list[float | None]:
    if isinstance(values, (str, bytes)) or (isinstance(values, np.ndarray) and values.ndim != 1):
        raise UnsupportedSourceDomain("taxa_information$values must be a flat sequence")
    out = []
    for v in values:
        if v is None:
            out.append(None)
            continue
        if isinstance(v, (list, tuple, np.ndarray)):
            raise UnsupportedSourceDomain("taxa_information$values must be a flat sequence")
        x = _real(v, "taxa_information$values")
        if x == math.inf:
            raise UnsupportedSourceDomain("+Inf is not produced by the prepared taxa_information")
        if x == 0 and math.copysign(1.0, x) < 0:
            raise UnsupportedSourceDomain("negative zero is not produced by the prepared taxa_information")
        out.append(x)
    if len(out) != size:
        raise UnsupportedSourceDomain("length(values) must equal nrow*ncol; matrix() would recycle or truncate")
    return out


def _weights(new_weights, ncol: int) -> tuple[float, ...]:
    if isinstance(new_weights, (str, bytes)):
        raise UnsupportedSourceDomain("new_weights must be a sequence")
    weights = tuple(_real(w, "new_weights") for w in new_weights)
    if len(weights) != ncol:
        raise UnsupportedSourceDomain("exactly one weight per taxon (ncol); no recycling of other lengths")
    if not all(math.isfinite(w) and w > 0 for w in weights):
        raise UnsupportedSourceDomain("weights must be finite and positive: the loop 1249 needs min(new_weights) > 0")
    return weights


def _two_adic(x: float) -> int:
    """Exponent of the lowest set bit of a nonzero finite double."""
    denominator = Fraction(x).denominator
    if denominator > 1:
        return -(denominator.bit_length() - 1)
    n = abs(Fraction(x).numerator)
    return (n & -n).bit_length() - 1


def row_sum_na_rm(row) -> RowSum:
    """rowSums(x, na.rm = T) for one row: NA (None) and NaN skipped."""
    kept = [v for v in row if v is not None and not math.isnan(v)]
    if any(v == -math.inf for v in kept):
        return RowSum(value=-math.inf, exact=None, count=len(kept), accumulator_independent=True)
    exact = sum((Fraction(v) for v in kept), Fraction(0))
    try:
        value = float(exact)  # correctly rounded
    except OverflowError as error:
        raise UnsupportedSourceDomain("row sum outside the double range") from error
    if math.isinf(value):
        raise UnsupportedSourceDomain("row sum outside the double range")
    nonzero = [v for v in kept if v != 0]
    independent = (not nonzero or
                   sum(abs(Fraction(v)) for v in nonzero) < Fraction(2) ** (min(map(_two_adic, nonzero)) + 53))
    return RowSum(value=value, exact=exact, count=len(kept), accumulator_independent=independent)


def _divide(x: float | None, s: float) -> float | None:
    """One cell of matrix / rowSums vector (IEEE double division)."""
    if x is None:
        return None
    if math.isnan(x):
        return math.nan
    if s == 0:
        return math.nan if x == 0 else math.copysign(math.inf, x)
    q = x / s  # Python float division is the IEEE double division for s != 0
    if math.isfinite(x) and math.isfinite(s) and (math.isinf(q) or (q == 0 and x != 0)):
        raise UnsupportedSourceDomain("quotient overflows or underflows to zero")
    return q


def weights_stage(values, ncol_taxa_array: int, nrow_taxa_array: int, new_weights) -> WeightsStage:
    """Lines 1259 and 1276-1280 with explicit weights; the inputs stay unchanged."""
    ncol = _count("ncol_taxa_array", ncol_taxa_array)
    nrow = _count("nrow_taxa_array", nrow_taxa_array)
    flat = _values(values, nrow * ncol)
    weights = _weights(new_weights, ncol)
    recycled = tuple(k % ncol for k in range(nrow * ncol))  # R recycling: length is a multiple of ncol
    products = []
    for v, j in zip(flat, recycled):
        if v is None:
            products.append(None)
            continue
        p = v * weights[j]
        if math.isfinite(v) and (math.isinf(p) or (p == 0 and v != 0)):
            raise UnsupportedSourceDomain("product overflows or underflows to zero")
        products.append(p)
    rows = tuple(tuple(products[i * ncol:(i + 1) * ncol]) for i in range(nrow))  # matrix(..., byrow = T)
    sums = tuple(row_sum_na_rm(row) for row in rows)
    # matrix / vector recycles the vector down the columns: cell (i, j) at column-major position i + j*nrow gets
    # sums[(i + j*nrow) %% nrow] = sums[i]
    normalised = tuple(tuple(_divide(x, sums[(i + j * nrow) % nrow].value) for j, x in enumerate(row))
                       for i, row in enumerate(rows))
    return WeightsStage(
        taxa_weights=weights, recycled_weight=tuple(j + 1 for j in recycled), products=tuple(products),
        unnormalised=rows, row_sums=sums, taxa_appear_core_temp=normalised,
        zero_sum_rows=tuple(i + 1 for i, s in enumerate(sums) if s.value == 0), z=tuple(range(1, ncol + 1)))
