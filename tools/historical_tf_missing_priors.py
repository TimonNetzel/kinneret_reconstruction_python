"""Removal of taxa without transfer function and the num_taxa priors from mcmc_prepare.R 1664-1685.

The NA mask ``is.na(tf.x1.current.mean)`` is an explicit input: in the source it comes from ``weightedMedian``
(1631, matrixStats without bound version), which stays open, so this module never derives it. Only the x1 mean is
tested in 1664; a taxon whose x2 mean alone is NA is kept. Taxa share one order across all objects: columns of
``all.avail``, ``all.avail.age`` and ``taxa.appear.core``, rows of ``tf.x1/x2.current`` and the mean vectors.

Kept literally: ``which`` (ascending positions), the guard ``length(which.tf.missing) != 0`` (without it,
``x[-integer(0)]`` would select nothing), negative indexing that keeps the remaining order, ``taxa_change`` as an
explicit boolean (in the bound source it is only set TRUE behind ``& FALSE``, 1500-1501), ``num_taxa`` from
``ncol(all.avail)`` in 1435 when nothing is removed and ``nrow(tf.x1.current)`` in 1678 otherwise, then 1681-1685:
``which_taxa <- c(1:num_taxa)``, ``rep(1/num_taxa, num_taxa)``, ``num_taxa/2``, ``num_taxa`` and
``round(num_taxa/4)`` with R's documented digits = 0 rounding (IEC 60559, ties to the even digit).
``prior$phi1``/``phi2`` from 1439-1440 are not updated and are reported as stale; ``posterior$nr.taxa.mean[1]``
is discarded by ``posterior <- list()`` in 1706. Dimension drop (one remaining taxon, or a matrix whose other
extent is 1 when columns or rows are removed) and an empty result (``c(1:0)``) are rejected instead of emulated.
No R execution, no sampling.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MAX_TAXA = 2**31 - 1  # R integer positions; num_taxa/4 is then an exact double


class UnsupportedSourceDomain(ValueError):
    """Input needs R/runtime semantics outside this finite fragment."""


@dataclass(frozen=True)
class NumTaxaPriors:
    num_taxa: int
    which_taxa: tuple[int, ...]  # c(1:num_taxa), 1-based
    taxa_weights: tuple[float, ...]  # rep(1/num_taxa, times = num_taxa)
    phi: float  # num_taxa/2
    nr_taxa_mean: int  # num_taxa
    nr_taxa_sd: int  # round(num_taxa/4)
    nr_taxa_sd_is_tie: bool  # num_taxa/4 ends in exactly .5, i.e. num_taxa %% 4 == 2


@dataclass(frozen=True)
class MissingTfResult:
    which_tf_missing: tuple[int, ...]  # which(is.na(tf.x1.current.mean)), 1-based
    removal_applied: bool  # length(which.tf.missing) != 0
    kept: tuple[int, ...]  # 0-based positions of the remaining taxa, in source order
    taxa: tuple[str, ...]  # colnames(all.avail) after 1676
    all_avail: np.ndarray
    all_avail_age: np.ndarray
    taxa_appear_core: np.ndarray
    tf_x1: np.ndarray
    tf_x2: np.ndarray
    x1_mean: tuple[float, ...]
    x2_mean: tuple[float, ...]
    x1_mean_change: tuple[float, ...] | None  # reduced only if taxa_change; the _change TF matrices never are
    x2_mean_change: tuple[float, ...] | None
    num_taxa_line: int  # 1435 (ncol(all.avail), nothing removed) or 1678 (nrow(tf.x1.current))
    priors: NumTaxaPriors  # 1681-1685
    stale_phi1: float  # 1440: num_taxa/2 with the 1435 count, not reassigned after removal
    stale_phi2: float  # 1439: (num_taxa-2)/2 with the 1435 count
    posterior_nr_taxa_mean_survives: bool  # always False: posterior <- list() in 1706


def r_round_digits0(x: float) -> int:
    """``round(x)`` for a finite nonnegative double: nearest integer, exact ties to the even digit."""
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 or x >= 2.0**52:
        raise UnsupportedSourceDomain("round needs one finite nonnegative double below 2^52")
    whole = math.floor(x)
    rest = x - whole  # exact below 2^52
    if rest > 0.5 or (rest == 0.5 and whole % 2 == 1):
        return whole + 1
    return whole


def num_taxa_priors(num_taxa: int) -> NumTaxaPriors:
    """Lines 1681-1685 for a positive integer count; the same formulas as 1437-1442 without phi1/phi2."""
    if isinstance(num_taxa, (bool, np.bool_)) or not isinstance(num_taxa, (int, np.integer)):
        raise UnsupportedSourceDomain("num_taxa must be an integer count")
    num_taxa = int(num_taxa)
    if num_taxa < 1:
        raise UnsupportedSourceDomain("num_taxa < 1: R would build c(1:0) = c(1, 0) and rep(Inf, 0); not emulated")
    if num_taxa > MAX_TAXA:
        raise UnsupportedSourceDomain("num_taxa beyond R integer positions")
    return NumTaxaPriors(num_taxa=num_taxa, which_taxa=tuple(range(1, num_taxa + 1)),
                         taxa_weights=(1 / num_taxa,) * num_taxa, phi=num_taxa / 2, nr_taxa_mean=num_taxa,
                         nr_taxa_sd=r_round_digits0(num_taxa / 4), nr_taxa_sd_is_tie=num_taxa % 4 == 2)


def _matrix(name: str, values) -> np.ndarray:
    try:
        array = np.array(values, dtype=np.float64) if np.asarray(values).dtype.kind in "iuf" else None
    except (TypeError, ValueError, OverflowError) as error:
        raise UnsupportedSourceDomain(f"{name} must be a real numeric matrix") from error
    if array is None or array.ndim != 2 or 0 in array.shape:
        raise UnsupportedSourceDomain(f"{name} must be a nonempty real numeric matrix (no bool, string or ragged)")
    return array


def _vector(name: str, values) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not hasattr(values, "__len__"):
        raise UnsupportedSourceDomain(f"{name} must be a numeric vector")
    out = []
    for v in values:
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
            raise UnsupportedSourceDomain(f"{name} must hold real numbers; NA is passed as NaN")
        out.append(float(v))
    return tuple(out)


def remove_missing_tfs(x1_mean_is_na, *, taxa, all_avail, all_avail_age, taxa_appear_core, tf_x1, tf_x2,
                       x1_mean, x2_mean, taxa_change: bool, x1_mean_change=None,
                       x2_mean_change=None) -> MissingTfResult:
    """Lines 1664-1685 for one site; inputs stay unchanged, values are carried and never recomputed."""
    if not isinstance(taxa_change, bool):
        raise UnsupportedSourceDomain("taxa_change must be an explicit boolean")
    if isinstance(x1_mean_is_na, (str, bytes)) or not hasattr(x1_mean_is_na, "__len__") \
            or not all(isinstance(v, (bool, np.bool_)) for v in x1_mean_is_na):
        raise UnsupportedSourceDomain("The NA mask must be explicit booleans, one per taxon")
    mask = [bool(v) for v in x1_mean_is_na]
    names = tuple(taxa)
    if not all(isinstance(n, str) for n in names):
        raise UnsupportedSourceDomain("Taxon names must be strings")
    by_column = {n: _matrix(n, v) for n, v in (("all.avail", all_avail), ("all.avail.age", all_avail_age),
                                               ("taxa.appear.core", taxa_appear_core))}
    by_row = {n: _matrix(n, v) for n, v in (("tf.x1.current", tf_x1), ("tf.x2.current", tf_x2))}
    vectors = {"tf.x1.current.mean": _vector("tf.x1.current.mean", x1_mean),
               "tf.x2.current.mean": _vector("tf.x2.current.mean", x2_mean)}
    change = {"tf.x1.current.mean_change": x1_mean_change, "tf.x2.current.mean_change": x2_mean_change}
    if taxa_change:
        if any(v is None for v in change.values()):
            raise UnsupportedSourceDomain("taxa_change = TRUE needs both _change mean vectors")
        vectors.update({n: _vector(n, v) for n, v in change.items()})
    elif any(v is not None for v in change.values()):
        raise UnsupportedSourceDomain("taxa_change = FALSE: the _change means stay NULL (1628/1648)")
    n_taxa = by_column["all.avail"].shape[1]  # num_taxa in 1435; all.avail is not resized before 1676
    sizes = ([len(mask), len(names)] + [m.shape[1] for m in by_column.values()]
             + [m.shape[0] for m in by_row.values()] + [len(v) for v in vectors.values()])
    if any(size != n_taxa for size in sizes):
        raise UnsupportedSourceDomain("All objects must share one taxon order and count; R would misalign or fail")

    which = tuple(i + 1 for i, missing in enumerate(mask) if missing)
    kept = tuple(i for i, missing in enumerate(mask) if not missing)
    stale = dict(stale_phi1=n_taxa / 2, stale_phi2=(n_taxa - 2) / 2, posterior_nr_taxa_mean_survives=False)
    if not which:  # guard 1665: nothing is indexed, num_taxa stays ncol(all.avail) from 1435
        return MissingTfResult(
            which_tf_missing=(), removal_applied=False, kept=kept, taxa=names,
            all_avail=by_column["all.avail"], all_avail_age=by_column["all.avail.age"],
            taxa_appear_core=by_column["taxa.appear.core"], tf_x1=by_row["tf.x1.current"],
            tf_x2=by_row["tf.x2.current"], x1_mean=vectors["tf.x1.current.mean"],
            x2_mean=vectors["tf.x2.current.mean"], x1_mean_change=vectors.get("tf.x1.current.mean_change"),
            x2_mean_change=vectors.get("tf.x2.current.mean_change"), num_taxa_line=1435,
            priors=num_taxa_priors(n_taxa), **stale)
    if len(kept) == 0:
        raise UnsupportedSourceDomain("Every TF missing: nrow 0 gives num_taxa 0 and c(1:0); not emulated")
    if len(kept) == 1:
        raise UnsupportedSourceDomain("One remaining taxon: [-idx] drops to a vector and nrow() is NULL; not emulated")
    for name, m in by_column.items():
        if m.shape[0] == 1:
            raise UnsupportedSourceDomain(f"{name} has one row: [, -idx] would drop to a vector; not emulated")
    for name, m in by_row.items():
        if m.shape[1] == 1:
            raise UnsupportedSourceDomain(f"{name} has one column: [-idx, ] would drop to a vector; not emulated")
    pick = list(kept)
    reduced = {n: tuple(v[i] for i in pick) for n, v in vectors.items()}
    num_taxa = by_row["tf.x1.current"][pick, :].shape[0]  # 1678: nrow(tf.x1.current)
    return MissingTfResult(
        which_tf_missing=which, removal_applied=True, kept=kept, taxa=tuple(names[i] for i in pick),
        all_avail=by_column["all.avail"][:, pick], all_avail_age=by_column["all.avail.age"][:, pick],
        taxa_appear_core=by_column["taxa.appear.core"][:, pick], tf_x1=by_row["tf.x1.current"][pick, :],
        tf_x2=by_row["tf.x2.current"][pick, :], x1_mean=reduced["tf.x1.current.mean"],
        x2_mean=reduced["tf.x2.current.mean"], x1_mean_change=reduced.get("tf.x1.current.mean_change"),
        x2_mean_change=reduced.get("tf.x2.current.mean_change"), num_taxa_line=1678,
        priors=num_taxa_priors(num_taxa), **stale)
