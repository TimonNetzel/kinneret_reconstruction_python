"""Finite deterministic TF preparation from preserved mcmc_prepare.R.

Rows are ordered taxa, columns ordered climate cells. R lines 1581-1586 divide
by rowSums, and 1613-1618 later zero excluded columns. No second normalization
is added. Call the two functions separately for TANN and PANN as appropriate.
This fragment neither estimates TFs nor infers an inside.range convention,
climate units, final source variant, taxon order or historical run readiness.
"""
from __future__ import annotations

import numpy as np


class UnsupportedSourceDomain(ValueError):
    """Input needs R/runtime semantics outside this finite real-valued fragment."""


def _matrix(values) -> np.ndarray:
    try:
        supplied = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as error:
        raise UnsupportedSourceDomain("TFs must be a nonempty real numeric matrix") from error
    if supplied.ndim != 2 or any(size == 0 for size in supplied.shape):
        raise UnsupportedSourceDomain("TFs must be a nonempty taxon-by-climate-cell matrix")
    if supplied.dtype.kind not in "iuf":
        raise UnsupportedSourceDomain("TFs require real numeric values; no coercion from strings, bools or complex")
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.array(supplied, dtype=np.float64, copy=True)
    if not np.isfinite(values).all() or (values < 0).any():
        raise UnsupportedSourceDomain("TF values must be finite and nonnegative; no imputation or clipping")
    return values


def normalize_tf_rows(transfer_functions) -> np.ndarray:
    """R 1581-1586 on finite nonnegative rows with finite positive mass.

    This is a discrete row sum, without climate-cell widths or integration.
    R's NA/Inf, zero-row and recycling paths are outside the supported domain.
    The result owns its data; the input and taxon/cell order stay unchanged.
    """
    values = _matrix(transfer_functions)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mass = values.sum(axis=1, keepdims=True)
        if not np.isfinite(mass).all() or (mass <= 0).any():
            raise UnsupportedSourceDomain("Each original TF row needs finite positive mass")
        return values / mass


def mask_tf_cells(transfer_functions, keep_cells) -> np.ndarray:
    """R 1613-1618 with an EXPLICIT boolean keep mask, no inferred bounds.

    Accepts finite nonnegative rows, including zero rows. Sets excluded climate
    columns to zero, without dividing again or adding replacement mass. A fully
    discarded row remains zero; this does not make later PDF normalization valid.
    The caller determines which preparation stage is supplied; this function
    never silently normalizes it. Exactly one boolean per climate cell is needed.
    """
    values = _matrix(transfer_functions)
    try:
        keep = np.asarray(keep_cells)
    except (TypeError, ValueError) as error:
        raise UnsupportedSourceDomain("A one-dimensional boolean climate-cell mask is required") from error
    if keep.ndim != 1 or keep.dtype.kind != "b" or len(keep) != values.shape[1]:
        raise UnsupportedSourceDomain("Mask must contain exactly one boolean per climate cell; no recycling")
    values[:, ~keep] = 0.0
    return values
