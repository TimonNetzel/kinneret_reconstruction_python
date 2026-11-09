"""Sampler preparation from num_taxa and taxa.appear.core, mcmc_conduction.R 53-79; no sampler is run.

``gibbs`` is an explicit boolean: the bound main loop sets ``gibbs <- F`` once (main_v1.2.R:624, depth 0), so
``posterior$how_many_taxa <- num_taxa`` (57) and ``parameter_stage_weights`` (58) are active; the Gibbs branch
``floor(num_taxa/2)`` (54) is reported but dead in the bound run.

Kept literally: ``c(t(taxa.appear.core))`` flattens row by row (line 64), ``ncol``/``nrow`` of the raw matrix
(65-66), ``colMeans(na.rm = T)`` of the raw matrix (67; NA skipped, a column of NA only gives NaN), the column
loop ``x[, j] / max(x[, j], na.rm = T)`` (70-73; NA stays NA, a zero column max gives 0/0 = NaN and x/0 = -Inf
for x < 0), the overwrite of ``taxa_information$values`` by the normalised matrix in 75 and
``posterior$jeffreys_prior <- rep(0.5, num_taxa)`` (79).

Input NaN stands for R NA. ``taxa_means`` has no reader in the bound loop files; R may accumulate the column sum in
extended precision (platform and R version unbound), so the module reports the correctly rounded exact mean and
flags every mean that is not independent of the accumulator. Rejected instead of emulated: a column of NA only
(``max`` gives -Inf with a warning), one row or one column (``[, which_taxa]`` in mcmc_prepare.R:1454/1666 drops to
a vector, ``ncol`` would be NULL), ``num_taxa`` different from ``ncol`` (the loop 70 would misalign or fail),
infinite values and negative zero (the sign of a zero maximum is not bound). No R execution, no random state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

MAX_TAXA = 2**31 - 1  # R integer counts


class UnsupportedSourceDomain(ValueError):
    """Input needs R/runtime semantics outside this finite fragment."""


@dataclass(frozen=True)
class ColumnMean:
    value: float  # NaN for a column of NA only (0/0)
    exact: Fraction | None  # None when no value is left after na.rm
    count: int  # non-NA values
    accumulator_independent: bool  # every partial sum and the mean are exact doubles


@dataclass(frozen=True)
class SamplerPreparation:
    gibbs: bool
    how_many_taxa: int  # 57: num_taxa; 54 (Gibbs, dead in the bound run): floor(num_taxa/2)
    how_many_taxa_line: int
    parameter_stage: str  # parameter_stage_weights (58) or parameter_stage_gibbs (55)
    num_taxa: int
    nrow_taxa_array: int  # 66
    ncol_taxa_array: int  # 65
    values_line64: tuple[float | None, ...]  # c(t(taxa.appear.core)), overwritten in 75; None is NA
    values: tuple[float | None, ...]  # 75: c(t(taxa.appear.core.temp)); None is NA, nan is NaN
    taxa_appear_core_temp: tuple[tuple[float | None, ...], ...]  # rows of the normalised matrix
    column_max: tuple[float, ...]  # max(taxa.appear.core[, j], na.rm = T)
    taxa_means: tuple[ColumnMean, ...]  # 67, raw matrix; no reader in the bound loop files
    jeffreys_prior: tuple[float, ...]  # 79; read only by parameter_stage_gibbs


def _rows(taxa_appear_core) -> np.ndarray:
    try:
        raw = np.asarray(taxa_appear_core)
    except (TypeError, ValueError) as error:
        raise UnsupportedSourceDomain("taxa.appear.core must be a real numeric matrix") from error
    if raw.dtype.kind not in "iuf" or raw.ndim != 2:
        raise UnsupportedSourceDomain("taxa.appear.core must be a real numeric matrix (no bool, string or ragged)")
    m = raw.astype(np.float64)
    if m.shape[0] < 2 or m.shape[1] < 2:
        raise UnsupportedSourceDomain("one row or one column: [, which_taxa] drops to a vector and ncol() is NULL;"
                                      " not emulated")
    if np.isinf(m).any():
        raise UnsupportedSourceDomain("infinite values are outside the checked domain")
    if (np.signbit(m) & (m == 0)).any():
        raise UnsupportedSourceDomain("negative zero: the sign of a zero column maximum is not bound")
    return m


def _count(name: str, value) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise UnsupportedSourceDomain(f"{name} must be an integer count")
    value = int(value)
    if not 2 <= value <= MAX_TAXA:
        raise UnsupportedSourceDomain(f"{name} outside 2..2^31-1")
    return value


def how_many_taxa(num_taxa: int, gibbs: bool) -> tuple[int, int, str]:
    """Lines 53-59: (how_many_taxa, source line, parameter stage)."""
    if not isinstance(gibbs, bool):
        raise UnsupportedSourceDomain("gibbs must be an explicit boolean")
    n = _count("num_taxa", num_taxa)
    return (n // 2, 54, "parameter_stage_gibbs") if gibbs else (n, 57, "parameter_stage_weights")


def col_means_na_rm(taxa_appear_core) -> tuple[ColumnMean, ...]:
    """colMeans(x, na.rm = T) per column; a column of NA only gives NaN (0/0)."""
    m = np.asarray(taxa_appear_core, dtype=np.float64)
    out = []
    for column in m.T:
        kept = [float(v) for v in column if not math.isnan(v)]
        if not kept:
            out.append(ColumnMean(value=math.nan, exact=None, count=0, accumulator_independent=True))
            continue
        total, exact_sums = Fraction(0), True
        for v in kept:
            total += Fraction(v)
            exact_sums &= Fraction(float(total)) == total
        mean = total / len(kept)
        value = float(mean)  # correctly rounded
        out.append(ColumnMean(value=value, exact=mean, count=len(kept),
                              accumulator_independent=exact_sums and Fraction(value) == mean))
    return tuple(out)


def _flatten_rowwise(m: np.ndarray) -> tuple[float | None, ...]:
    """c(t(m)): row after row; NaN marks NA in the input."""
    return tuple(None if math.isnan(v) else float(v) for v in m.reshape(-1, order="C"))


def prepare_sampler(taxa_appear_core, num_taxa: int, *, gibbs: bool) -> SamplerPreparation:
    """Lines 53-79 for one site; the input stays unchanged."""
    m = _rows(taxa_appear_core)
    n = _count("num_taxa", num_taxa)
    if n != m.shape[1]:
        raise UnsupportedSourceDomain("num_taxa must equal ncol(taxa.appear.core); the loop 70 would misalign")
    how_many, line, stage = how_many_taxa(n, gibbs)
    na = np.isnan(m)
    if na.all(axis=0).any():
        raise UnsupportedSourceDomain("a column of NA only: max(, na.rm = T) is -Inf with a warning; not emulated")
    maxima = np.where(na, -np.inf, m).max(axis=0)
    temp = m.copy()
    for j in range(n):  # for(j in 1:num_taxa)
        with np.errstate(divide="ignore", invalid="ignore"):
            temp[:, j] = m[:, j] / maxima[j]  # NaN (NA) stays NaN; 0/0 is NaN, x/0 is -Inf for x < 0
    values = tuple(None if na_ij else float(v)
                   for v, na_ij in zip(temp.reshape(-1, order="C"), na.reshape(-1, order="C")))
    rows = tuple(tuple(None if na[i, j] else float(temp[i, j]) for j in range(n)) for i in range(m.shape[0]))
    return SamplerPreparation(
        gibbs=gibbs, how_many_taxa=how_many, how_many_taxa_line=line, parameter_stage=stage, num_taxa=n,
        nrow_taxa_array=m.shape[0], ncol_taxa_array=m.shape[1], values_line64=_flatten_rowwise(m),
        values=values, taxa_appear_core_temp=rows, column_max=tuple(float(v) for v in maxima),
        taxa_means=col_means_na_rm(m), jeffreys_prior=(0.5,) * n)
