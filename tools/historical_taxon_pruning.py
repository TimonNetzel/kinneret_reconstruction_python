"""Finite deterministic taxon pruning from preserved mcmc_prepare.R lines 51-80.

Rows are ordered depth samples, columns ordered taxa (``all.avail``). The step only runs when the caller
passes ``null_ratio_check=True``; the bound reconstruction loop in main_v1.2.R:659 sets it to FALSE, so this
module documents a conditional branch and claims no run activation. ``metro_hastings`` and the relict
override (``sitename == "Lake Ohrid"`` here, a free variable ``ohrid`` in mcmc_prep.R) are explicit inputs.

R semantics kept literally: ``>= 0.95`` on double ratios, ``sort(null.ratio)[16:ncol]`` including the
descending colon and NA positions for fewer than 16 taxa, tie extension through ``%in%``, and ``next()``
reported as ``skip_site``. Vector extension by out-of-range assignment and dimension drop to a single
column are rejected instead of emulated. No R execution and no choice between source copies.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RELICT_IDS_R_1_BASED = (3, 8, 23, 24, 25)
MAX_ROWS = 10**6  # double k/n >= 0.95 equals exact k/n >= 19/20 here (see tests)


class UnsupportedSourceDomain(ValueError):
    """Input needs R/runtime semantics outside this finite real-valued fragment."""


@dataclass(frozen=True)
class PruningResult:
    executed: bool
    branch: str  # "not_executed" | "rarest_15" | "ratio_095"
    metro_hastings: bool
    skip_site: bool
    null_ratio: tuple[float, ...] | None
    removed_taxa: tuple[int, ...]  # 0-based original columns
    kept_taxa: tuple[int, ...]
    pollen: np.ndarray | None  # None when R would call next()
    sorting_available_spectrum: np.ndarray | None  # colMeans; R leaves it untouched when not executed


def _matrix(values) -> np.ndarray:
    try:
        supplied = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as error:
        raise UnsupportedSourceDomain("Pollen must be a nonempty real numeric matrix") from error
    if supplied.ndim != 2 or any(size == 0 for size in supplied.shape):
        raise UnsupportedSourceDomain("Pollen must be a nonempty depth-by-taxon matrix")
    if supplied.dtype.kind not in "iuf":
        raise UnsupportedSourceDomain("Pollen requires real numeric values; no coercion from strings, bools or complex")
    if supplied.shape[0] > MAX_ROWS:
        raise UnsupportedSourceDomain("More rows than the checked double/rational threshold range")
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.array(supplied, dtype=np.float64, copy=True)
    if not np.isfinite(values).all() or (values < 0).any():
        raise UnsupportedSourceDomain("Pollen values must be finite and nonnegative; R's NA counting is not emulated")
    return values


def null_ratios(pollen) -> np.ndarray:
    """R 53-55: share of exact zeros per taxon column, as double division."""
    values = _matrix(pollen)
    return (values == 0).sum(axis=0) / values.shape[0]


def _r_index(values, index_from: int, index_to: int) -> list:
    """``values[index_from:index_to]`` in R: colon may descend, positions beyond the end give NA (None)."""
    step = 1 if index_to >= index_from else -1
    return [values[i - 1] if 1 <= i <= len(values) else None for i in range(index_from, index_to + step, step)]


def prune_taxa(pollen, *, null_ratio_check: bool, metro_hastings: bool, relict_override: bool) -> PruningResult:
    """R 51-80 for one site; the input matrix and column order stay unchanged."""
    if not all(isinstance(flag, bool) for flag in (null_ratio_check, metro_hastings, relict_override)):
        raise UnsupportedSourceDomain("Control flags must be explicit booleans")
    values = _matrix(pollen)
    n_taxa = values.shape[1]
    if not null_ratio_check:
        return PruningResult(False, "not_executed", metro_hastings, False, None, (), tuple(range(n_taxa)),
                             values, None)
    ratio = null_ratios(values)
    if relict_override:
        if n_taxa < max(RELICT_IDS_R_1_BASED):
            raise UnsupportedSourceDomain("Relict override needs at least 25 taxa; R would extend null.ratio with NA")
        ratio[[i - 1 for i in RELICT_IDS_R_1_BASED]] = 0.0
        ratio[0] = 1.0
    rare = np.flatnonzero(ratio >= 0.95)
    if n_taxa - len(rare) > 15:
        metro_hastings = True
    if not metro_hastings:
        table = [v for v in _r_index(sorted(ratio.tolist()), 16, n_taxa) if v is not None]
        removed = np.flatnonzero(np.isin(ratio, table))
        branch = "rarest_15"
    else:
        removed = rare
        branch = "ratio_095"
    kept = np.setdiff1d(np.arange(n_taxa), removed)
    summary = dict(executed=True, branch=branch, metro_hastings=metro_hastings,
                   null_ratio=tuple(ratio.tolist()), removed_taxa=tuple(removed.tolist()), kept_taxa=tuple(kept.tolist()))
    if len(removed) == n_taxa:
        return PruningResult(skip_site=True, pollen=None, sorting_available_spectrum=None, **summary)
    if len(kept) == 1:
        raise UnsupportedSourceDomain("One remaining taxon: R drops to a vector and colMeans fails; not emulated")
    pruned = values[:, kept]
    return PruningResult(skip_site=False, pollen=pruned, sorting_available_spectrum=pruned.mean(axis=0), **summary)
