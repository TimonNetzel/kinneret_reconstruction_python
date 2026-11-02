"""Finite deterministic chronology fragments; no historical runner or age model.

Bound sources and exclusions: literal_age_depth_2026_11_02/source_binding.json.
Kernel rows are ages, columns are a one-unit depth grid; sampled columns match
pollen ROWS. Resolution IDs are one-based R list positions, never a default.
"""
from numbers import Integral

import numpy as np


class UnsupportedSourceDomain(ValueError):
    """Input requires R semantics or source branches outside this fragment."""


def _finite(values, ndim, name, *, nonnegative=False):
    data = np.array(values, dtype=np.float64, copy=True)
    if data.ndim != ndim or any(n == 0 for n in data.shape):
        raise UnsupportedSourceDomain(f"{name} must be a nonempty {ndim}D array")
    if not np.isfinite(data).all() or (nonnegative and (data < 0).any()):
        raise UnsupportedSourceDomain(f"{name} contains unsupported values")
    return data


def select_list_age_grid(rho_grids, age_grids, resolution_id):
    """check_ages.R:177,179: explicit paired R-list selection, returned as copies.

    No site dispatch, Dead Sea override, empty-grid fallback or final selection.
    A matrix plus flat ages is rejected, rather than wrapped into a list.
    """
    if not isinstance(rho_grids, (list, tuple)) or not isinstance(age_grids, (list, tuple)):
        raise UnsupportedSourceDomain("Paired lists of matrices and age vectors required")
    if len(rho_grids) != len(age_grids):
        raise UnsupportedSourceDomain("Resolution lists must have equal lengths")
    if (isinstance(resolution_id, (bool, np.bool_))
            or not isinstance(resolution_id, Integral)
            or not 1 <= resolution_id <= len(rho_grids)):
        raise UnsupportedSourceDomain("Resolution ID must be an explicit one-based integer")
    rho = _finite(rho_grids[resolution_id - 1], 2, "selected rho", nonnegative=True)
    ages = _finite(age_grids[resolution_id - 1], 1, "selected ages")
    if rho.shape[0] != ages.size:
        raise UnsupportedSourceDomain("Kernel rows must match the selected age vector")
    return rho, ages


def shift_duplicate_ages_once(ages):
    """check_ages.R:230-232: add 0.5 to subsequent exact duplicates ONCE.

    No sorting, iteration, interpolation or uniqueness repair. The caller's
    site.age remains untouched; source selection occurred before this shift.
    """
    result = _finite(ages, 1, "ages")
    seen = set()
    for i, value in enumerate(result.copy()):
        if float(value) in seen:
            result[i] += 0.5
        seen.add(float(value))
    if not np.isfinite(result).all():
        raise UnsupportedSourceDomain("Shift exceeds the finite domain")
    return result


def depth_kernel_at_samples(rho_depth_given_age, depths):
    """mcmc_prepare.R:11-20, non-Birkat-Ram path.

    depths must already be ordered, unique and aligned with pollen rows, with
    exact integer offsets from min(depth). No truncation of fractional R indices.
    rho must already be the explicitly selected matrix, not a resolution list.
    The assumed one-unit grid origin is min(depth), exactly as in the source.
    """
    rho = _finite(rho_depth_given_age, 2, "rho_depth_given_age", nonnegative=True)
    depth = _finite(depths, 1, "depths")
    if (np.diff(depth) <= 0).any():
        raise UnsupportedSourceDomain("Unsorted/duplicate depths need the upstream source audit")
    with np.errstate(over="ignore", invalid="ignore"):
        offsets = depth - depth.min()
    if (not np.isfinite(offsets).all() or (offsets != np.floor(offsets)).any()
            or (offsets >= rho.shape[1]).any()):
        raise UnsupportedSourceDomain("Depth offsets must be exact in-bounds grid indices")
    selected = rho[:, offsets.astype(np.intp)]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mass = selected.sum(axis=1, keepdims=True)
        if not np.isfinite(mass).all() or (mass <= 0).any():
            raise UnsupportedSourceDomain("Selected rows need finite positive mass")
        weights = selected / mass
    # Source literal 10e-10 is 1e-9. No second normalization here.
    weights[weights == 0] = 10e-10
    return weights


def pollen_at_ages(depth_weights, all_available):
    """mcmc_prepare.R:83,91-94: replace pollen zeros, then W @ P.

    all_available is the already selected taxon table. Taxon filtering, NA to
    zero in check_ages.R, and the later NA-output fill are outside this function.
    Weights are used as supplied; no extra row/column normalization.
    """
    weights = _finite(depth_weights, 2, "depth_weights", nonnegative=True)
    pollen = _finite(all_available, 2, "all_available", nonnegative=True)
    if weights.shape[1] != pollen.shape[0]:
        raise UnsupportedSourceDomain("Weight columns must match aligned pollen rows")
    pollen[pollen == 0] = 1e-6
    with np.errstate(over="ignore", invalid="ignore"):
        result = weights @ pollen
    if not np.isfinite(result).all():
        raise UnsupportedSourceDomain("Product exceeds the finite supported domain")
    return result
