"""Deterministic fragments of preserved R final_reconstruction, not a runner.

Source: original:src/mcmc_functions_v1.6.R (SHA-256 recorded in the evidence
packet literal_final_pdf_2026_11_02). These functions reproduce only the finite,
nonnegative, nondegenerate domain of the preserved PDF-sum branch. They do not
select the final historical Kinneret variant, apply chronology, select posterior
samples by IQR/IAT/Pareto, or estimate/repair transfer functions.

Rows are samples, columns are taxa (or climate cells after applying TFs).
Draw and taxon IDs returned/accepted here are ONE-BASED, as in the R source.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import numpy as np


class UnsupportedSourceDomain(ValueError):
    """The fragment cannot reproduce this input without extending R semantics."""


def _finite(values, ndim: int, name: str, *, nonnegative: bool = True) -> np.ndarray:
    data = np.array(values, dtype=np.float64, copy=True)
    if data.ndim != ndim or any(size == 0 for size in data.shape):
        raise UnsupportedSourceDomain(f"{name} must be a nonempty {ndim}-dimensional array")
    if not np.isfinite(data).all() or (nonnegative and (data < 0).any()):
        raise UnsupportedSourceDomain(f"{name} must contain finite supported values")
    return data


def _row_normalize(values: np.ndarray) -> np.ndarray:
    # No smoothing, clipping, replacement of zeros, or earlier normalization.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mass = values.sum(axis=1, keepdims=True)
        if not np.isfinite(values).all() or not np.isfinite(mass).all() or (mass <= 0).any():
            raise UnsupportedSourceDomain("A row has nonpositive or nonfinite mass")
        return values / mass


def column_max_taxa(all_available) -> np.ndarray:
    """R lines 411-416: each taxon divided by its own column maximum."""
    values = _finite(all_available, 2, "all_available")
    maxima = values.max(axis=0)
    if (maxima <= 0).any():
        raise UnsupportedSourceDomain("Zero-maximum taxon: R's 0/0 path is outside this fragment")
    return values / maxima


def rw_coefficients(all_available, taxa_weights) -> np.ndarray:
    """R lines 417-419, 450-455: row-major recycling then row normalization."""
    scaled = column_max_taxa(all_available)
    weights = _finite(taxa_weights, 1, "taxa_weights")
    if weights.size != scaled.shape[1]:
        raise UnsupportedSourceDomain("Exactly one weight per taxon is required; no recycling repair")
    with np.errstate(over="ignore", invalid="ignore"):
        return _row_normalize(scaled * weights)


def gibbs_coefficients(all_available, taxon_ids: Sequence[int]) -> np.ndarray:
    """R lines 434-436: select one-based columns AFTER column-max scaling.

    Reordering and repeated columns are preserved; no deduplication is added.
    """
    scaled = column_max_taxa(all_available)
    ids = tuple(taxon_ids)
    if (not ids or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral)
                       or i < 1 or i > scaled.shape[1] for i in ids)):
        raise UnsupportedSourceDomain("Taxon IDs must be valid one-based integers")
    # R drops dimensions for a single selected column; rowSums then fails.
    if len(ids) == 1 or scaled.shape[0] == 1:
        raise UnsupportedSourceDomain("R subsetting dimension-drop path is outside this fragment")
    return _row_normalize(scaled[:, [i - 1 for i in ids]])


def apply_taxon_densities(coefficients, transfer_functions) -> np.ndarray:
    """R lines 439-443 / 458-462: matrix product, with TF rows left untouched."""
    spectrum = _finite(coefficients, 2, "coefficients")
    functions = _finite(transfer_functions, 2, "transfer_functions")
    if spectrum.shape[1] != functions.shape[0]:
        raise UnsupportedSourceDomain("Coefficient columns must match ordered TF rows")
    with np.errstate(over="ignore", invalid="ignore"):
        result = spectrum @ functions
    if not np.isfinite(result).all():
        raise UnsupportedSourceDomain("Matrix product is outside the finite supported domain")
    return result


def unique_energy_ids(energy) -> tuple[int, ...]:
    """R line 375: first occurrence of each exact energy, without rounding."""
    scores = _finite(energy, 1, "energy", nonnegative=False)
    seen = set()
    ids = []
    for index, score in enumerate(scores, start=1):
        if float(score) not in seen:
            seen.add(float(score))
            ids.append(index)
    return tuple(ids)


@dataclass(frozen=True)
class LiteralPdfSum:
    """Output of a source fragment, not an historical case result."""
    draw_ids: tuple[int, ...]
    density: np.ndarray


def sum_pdf_branch(draw_densities, energy) -> LiteralPdfSum:
    """R 375, 428, 543-547, 711-715, 755-760: sum raw rows, then normalize.

    Shape: draws x samples x climate cells. Only duplicate-energy removal is
    included. Energy values are NOT mixture weights. Preceding draw-selection
    steps and the stochastic averaging_data branch are deliberately absent.
    """
    densities = _finite(draw_densities, 3, "draw_densities")
    scores = _finite(energy, 1, "energy", nonnegative=False)
    if len(scores) != densities.shape[0]:
        raise UnsupportedSourceDomain("Exactly one energy per supplied draw is required")
    ids = unique_energy_ids(scores)
    total = np.zeros(densities.shape[1:], dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        for draw_id in ids:
            total += densities[draw_id - 1]
    return LiteralPdfSum(ids, _row_normalize(total))
