"""Literal finite cumulative-density comparison; not a historical runner.

Source spans: literal_cdf_discrepancy_2026_11_02/source_binding.json.
The source calls its output EMD. This fragment adds neither probability
normalization nor physical cell widths and does not select a final run.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np


class UnsupportedSourceDomain(ValueError):
    """Input requires source/runtime semantics outside this finite fragment."""


def _finite(values, name, *, ndim=None, nonnegative=False):
    data = np.array(values, dtype=np.float64, copy=True)
    if data.size == 0 or (ndim is not None and data.ndim != ndim):
        raise UnsupportedSourceDomain(f"{name} has an unsupported empty shape or rank")
    if not np.isfinite(data).all() or (nonnegative and (data < 0).any()):
        raise UnsupportedSourceDomain(f"{name} contains unsupported values")
    return data


def _positive_integer(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            or value <= 0):
        raise UnsupportedSourceDomain(f"{name} must be an explicit positive integer")
    return int(value)


@dataclass(frozen=True)
class LiteralReferencePair:
    first: np.ndarray
    second: np.ndarray
    zero_placeholder_replaced: bool


def prepare_reference_pair(ecdf_ref1, ecdf_ref2, *, rho_rows, inherited_dims):
    """R 369-370: first reference length==1 replaces BOTH references with zeros.

    inherited_dims is the environment value used BEFORE the local assignment
    dims <- length(x1range.bck) at R 389. It is never inferred or repaired.
    The second reference is not inspected or converted on the replacement path.
    Non-placeholder matrices are copied without renormalizing or cumulating.
    """
    first = _finite(ecdf_ref1, "ecdf_ref1")
    if first.size == 1:
        rows = _positive_integer(rho_rows, "rho_rows")
        width = _positive_integer(inherited_dims, "inherited_dims")
        zeros = np.zeros((rows, width), dtype=np.float64)
        return LiteralReferencePair(zeros, zeros.copy(), True)
    if first.ndim != 2:
        raise UnsupportedSourceDomain("Non-placeholder ecdf_ref1 must be a matrix")
    second = _finite(ecdf_ref2, "ecdf_ref2", ndim=2)
    return LiteralReferencePair(first, second, False)


def raw_row_cumsums(density):
    """R 485-486: raw rowCumsums in supplied climate-cell order, no normalization.

    Only finite nonnegative nonempty matrices are supported. Zero rows are
    preserved; there is no mass correction, NA removal, sorting or grid fitting.
    """
    values = _finite(density, "density", ndim=2, nonnegative=True)
    with np.errstate(over="ignore", invalid="ignore"):
        cumulative = np.cumsum(values, axis=1)
    if not np.isfinite(cumulative).all():
        raise UnsupportedSourceDomain("Cumulative sum exceeds the finite domain")
    return cumulative


@dataclass(frozen=True)
class LiteralCdfDiscrepancy:
    cumulative: np.ndarray
    score: np.ndarray


def cdf_discrepancy(density, reference_cdf, *, dims):
    """R 485-489: (1/dims) * sum(abs(raw row cumulative - supplied reference)).

    dims is the explicitly supplied CURRENT width, distinct from inherited_dims.
    Shapes must agree exactly; no R recycling or placeholder-axis repair.
    References may be finite nonmonotone/signed arrays: the source does not
    enforce CDF validity. This function does not turn them into valid CDFs.
    """
    width = _positive_integer(dims, "dims")
    cumulative = raw_row_cumsums(density)
    reference = _finite(reference_cdf, "reference_cdf", ndim=2)
    if cumulative.shape[1] != width or reference.shape != cumulative.shape:
        raise UnsupportedSourceDomain("Density, reference and current dims must match exactly")
    with np.errstate(over="ignore", invalid="ignore"):
        score = (1.0 / width) * np.abs(cumulative - reference).sum(axis=1)
    if not np.isfinite(score).all():
        raise UnsupportedSourceDomain("Discrepancy exceeds the finite domain")
    return LiteralCdfDiscrepancy(cumulative, score)
