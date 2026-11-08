"""reconstruction_climate of mcmc_functions_v1.6.R 1188-1199, with every global as an explicit input.

The bound run never calls this function: its only caller is mcmc_sampling_v1.4.R:79, a file that nothing sources. The
bound reconstruction (1781ff.) takes post_tann from predictant_tann[1] (line 1850), i.e. from sampled TF values. This
module therefore documents a dormant source path and activates nothing.

Kept literally. R is lazy, so with ``recent`` FALSE none of the taxon or TF arguments is read and any invalid value there
is harmless; only the globals prior.recent$mean.tmp and $mean.pann are returned. With ``recent`` TRUE only the FIRST
row of taxa.appear.core is used. ``which.taxa`` is never read. x1range.bck, x2range.bck and prior.recent are globals in
the source and are explicit arguments here.

``row %*% matrix`` on finite doubles is R's BLAS path: dgemv with TRANS = "T", per column one plain double sum in taxon
order. With an NA/NaN input R 3.5.0 uses simple_matprod instead (double as well); the touched columns are then NaN and
the weighted median is NA, so the distinction cannot be observed here. 0 * NA is NA: a taxon with share 0 and an NA TF
still makes the median NA. Infinite inputs are rejected, not emulated. No R execution, no random state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np

from historical_weighted_median import UnsupportedSourceDomain, WeightedMedian, weighted_median

__all__ = ['UnsupportedSourceDomain', 'RecentClimate', 'row_times_matrix', 'reconstruction_climate']


@dataclass(frozen=True)
class RecentClimate:
    tann: float  # NaN stands for NA as well
    pann: float
    tann_kind: str  # 'number', 'NA' or 'NaN'
    pann_kind: str
    source: str  # 'weighted_median_first_row' or 'prior_recent'
    tann_weights: tuple[float, ...] | None  # the row %*% matrix result; None when not computed
    pann_weights: tuple[float, ...] | None
    tann_median: WeightedMedian | None
    pann_median: WeightedMedian | None


def _double(name: str, v) -> float:
    if v is None:
        return math.nan
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (Real, np.floating, np.integer)):
        raise UnsupportedSourceDomain(f'{name} must hold real numbers or None for NA')
    v = float(v)
    if math.isinf(v):
        raise UnsupportedSourceDomain(f'{name}: infinite values are outside the checked domain')
    return v


def _vector(name: str, values) -> list[float]:
    if isinstance(values, (str, bytes)) or (isinstance(values, np.ndarray) and values.ndim != 1):
        raise UnsupportedSourceDomain(f'{name} must be a flat numeric vector')
    return [_double(name, v) for v in values]


def _matrix(name: str, values) -> list[list[float]]:
    if isinstance(values, (str, bytes)) or len(values) == 0:
        raise UnsupportedSourceDomain(f'{name} must be a nonempty matrix')
    rows = [_vector(name, row) for row in values]
    if len({len(r) for r in rows}) != 1 or len(rows[0]) == 0:
        raise UnsupportedSourceDomain(f'{name} must be rectangular and nonempty')
    return rows


def row_times_matrix(row: list[float], matrix: list[list[float]]) -> list[float]:
    """``row %*% matrix`` as dgemv("T"): for each column one plain double sum, in row order of the matrix."""
    if len(row) != len(matrix):
        raise UnsupportedSourceDomain('The taxon row must have one value per TF row; R would not conform or recycle')
    out = []
    for column in zip(*matrix):
        total = 0.0
        for share, cell in zip(row, column):
            total += cell * share
        out.append(total)
    return out


def reconstruction_climate(taxa_appear_core, tf_x1, tf_x2, x1range, x2range, *, recent: bool, prior_mean_tmp: float,
                           prior_mean_pann: float) -> RecentClimate:
    """Lines 1188-1199. Arguments that R would not force are not looked at when ``recent`` is False."""
    if not isinstance(recent, (bool, np.bool_)):
        raise UnsupportedSourceDomain('recent must be an explicit boolean; if(NA) stops in R')
    if not recent:
        tann, pann = _double('prior_mean_tmp', prior_mean_tmp), _double('prior_mean_pann', prior_mean_pann)
        kind = lambda original, value: 'NA' if original is None else 'NaN' if math.isnan(value) else 'number'
        return RecentClimate(tann, pann, kind(prior_mean_tmp, tann), kind(prior_mean_pann, pann), 'prior_recent',
                             None, None, None, None)
    first = _matrix('taxa_appear_core', taxa_appear_core)[0]
    weights_x1 = row_times_matrix(first, _matrix('tf_x1', tf_x1))
    weights_x2 = row_times_matrix(first, _matrix('tf_x2', tf_x2))
    grid_x1, grid_x2 = _vector('x1range', x1range), _vector('x2range', x2range)
    if len(grid_x1) != len(weights_x1) or len(grid_x2) != len(weights_x2):
        raise UnsupportedSourceDomain("Grid and TF width differ: the C entry point stops on 'x' and 'w' lengths")
    tann, pann = weighted_median(grid_x1, weights_x1), weighted_median(grid_x2, weights_x2)
    return RecentClimate(tann.value, pann.value, tann.kind, pann.kind, 'weighted_median_first_row', tuple(weights_x1),
                         tuple(weights_x2), tann, pann)
