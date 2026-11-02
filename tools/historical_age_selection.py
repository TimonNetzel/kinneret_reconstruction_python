"""Preserved finite age-grid membership and row-selection fragments, not a runner.

Sources: literal_age_selection_2026_11_02/source_binding.json. Resolution/run
selection, age models, taxon matching and stochastic code remain excluded.
"""
from dataclasses import dataclass

import numpy as np


class UnsupportedSourceDomain(ValueError):
    """Input needs source semantics outside the documented finite fragment."""


def _finite(values, ndim, name, *, nonnegative=False):
    data = np.array(values, dtype=np.float64, copy=True)
    if data.ndim != ndim or any(n == 0 for n in data.shape):
        raise UnsupportedSourceDomain(f"{name} must be a nonempty {ndim}D array")
    if not np.isfinite(data).all() or (nonnegative and (data < 0).any()):
        raise UnsupportedSourceDomain(f"{name} contains unsupported values")
    return data


def period_grids(age):
    """mcmc_prepare.R:435-437: seq endpoints with diff(age)[1].

    Only exact half-unit ages within +/-1e9 and a positive first difference
    are supported. Integer half-unit arithmetic avoids inventing R seq's
    floating endpoint tolerance. No regular-spacing check or sorting is added.
    """
    values = _finite(age, 1, "age")
    if (values.size < 2 or (np.abs(values) > 1e9).any()
            or (values * 2 != np.floor(values * 2)).any()):
        raise UnsupportedSourceDomain("Need at least two finite exact half-unit ages within +/-1e9")
    half_units = (values * 2).astype(np.int64)
    step = int(half_units[1]) - int(half_units[0])
    if step <= 0:
        raise UnsupportedSourceDomain("The first age difference must be positive")
    grids = {}
    for name, lo, hi in (("hol", -50, 11450), ("lg", 11500, 115000), ("eem", 115500, 125000)):
        start, stop = 2 * lo, 2 * hi
        count = (stop - start) // step + 1
        grids[name] = (start + np.arange(count, dtype=np.int64) * step) / 2
    return grids


def period_membership(age):
    """mcmc_prepare.R:440-442: exact %in%, preserving input row order.

    Uses age, not site.age. Does not interpolate or classify by interval.
    """
    grids = period_grids(age)
    values = _finite(age, 1, "age")
    return {name: np.isin(values, grid) for name, grid in grids.items()}


@dataclass(frozen=True)
class PreservedAgeSelection:
    initial_reconst_ages: np.ndarray
    effective_reconst_ages: np.ndarray
    taxa_change: bool


def preserved_selection_trace(age, *, taxa_merge):
    """mcmc_prepare.R:1462-1467,1500-1522, assuming these lines are reached.

    Caller supplies taxa.merge; no taxon identification or successful full
    execution is inferred. The literal & FALSE disables the change branch.
    In both merge cases the preserved control flow ends with all-TRUE rows.
    """
    if not isinstance(taxa_merge, (bool, np.bool_)):
        raise UnsupportedSourceDomain("taxa_merge must be an explicit boolean")
    masks = period_membership(age)
    all_rows = np.ones(masks["hol"].size, dtype=bool)
    initial = masks["hol"].copy() if taxa_merge else all_rows.copy()
    return PreservedAgeSelection(initial, all_rows, False)


def rw_selected_age_product(coefficients, transfer_functions, reconst_ages):
    """mcmc_functions_v1.6.R:461-462 / 470: strict logical row subset then @.

    Inputs are already prepared coefficients/TFs. No weighting, scaling,
    normalization, taxa_change append or age-label synchronization is added.
    R dimension-drop paths, mask recycling and integer subscripts are excluded.
    """
    values = _finite(coefficients, 2, "coefficients", nonnegative=True)
    functions = _finite(transfer_functions, 2, "transfer_functions", nonnegative=True)
    mask = np.asarray(reconst_ages)
    if mask.ndim != 1 or mask.dtype.kind != "b" or mask.size != values.shape[0]:
        raise UnsupportedSourceDomain("Need one boolean per coefficient row; no recycling")
    if values.shape[1] < 2 or np.count_nonzero(mask) < 2:
        raise UnsupportedSourceDomain("R empty/dimension-drop paths are outside this fragment")
    if values.shape[1] != functions.shape[0]:
        raise UnsupportedSourceDomain("Coefficient columns must match ordered TF rows")
    with np.errstate(over="ignore", invalid="ignore"):
        result = values[mask, :] @ functions
    if not np.isfinite(result).all():
        raise UnsupportedSourceDomain("Product exceeds the finite supported domain")
    return result
