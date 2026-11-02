"""Gamma prior parameters from the preserved shape grid in mcmc_prepare.R (114-121, 1264-1281, 526-536, 1733).

The precipitation prior ``prior.recent`` (mean.pann, sd.pann; prior_climate.R:128-130 fits) is converted to a gamma
shape/rate pair by scanning ``shape = seq(0.1, upper, by = 0.1)`` and taking the first ``which.min`` of
``|sqrt(shape)/sd - shape/mean|``. Only base R: ``seq``, ``/``, ``sqrt``, ``-``, ``abs``, ``which.min``, ``[``.

Kept literally: ``seq.default`` with ``n <- as.integer((to - from)/by + 1e-10)``, ``from + (0:n) * by`` as two
separately rounded double operations and ``pmin(x, to)``; first index on ties; ``rate`` from the sqrt branch
(``rate2``). The closed form ``shape = (mean/sd)^2`` is reported only as a control, never as a substitute.
The site conditions (``recent``, ``min(age) >= 2000``) are explicit booleans; ``age`` is not evaluated here.
Inputs outside finite positive scalars are rejected instead of emulated. No R execution, no sampling: the seeded
``rgamma`` in 1301-1302 and the sampler readers of ``shape``/``rate`` stay outside this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SEQ_FROM = 0.1
SEQ_BY = 0.1
# grid upper bound per preserved source copy; mcmc_prepare.R is the one sourced by main_v1.2.R:675
UPPER_BY_SOURCE = {"mcmc_prepare.R": 50000.0, "mcmc_prep_v1.2.R": 5000.0}
SEQ_FUZZ = 1e-10


class UnsupportedSourceDomain(ValueError):
    """Input needs R/runtime semantics outside this finite real-valued fragment."""


@dataclass(frozen=True)
class GridResult:
    upper: float
    grid_length: int
    index: int  # 0-based; R's intersect_id is index + 1
    shape: float  # shape_temp[intersect_id]
    rate: float  # rate_temp2[intersect_id] = sqrt(shape)/sd
    rate_expectation: float  # rate_temp1[intersect_id] = shape/mean, not assigned to ``rate`` by these lines
    at_lower_boundary: bool
    at_upper_boundary: bool
    closed_form_shape: float  # (mean/sd)^2, control only


@dataclass(frozen=True)
class PriorChainResult:
    default: GridResult  # 114-121, before the variance doubling
    pann_period_shape: float  # 127: pann_{hist,mh,lgm,past}_shape
    pann_period_rate: float  # 128: pann_{hist,mh,lgm,past}_rate
    diff_rate: float  # 531/536 (else of stad_interstad <- FALSE): shape/10 with the default grid shape
    variance_doubled: bool  # 1264: recent & min(age) >= 2000
    sd_pann: float
    sd_tmp: float
    final: GridResult  # 1270-1281
    rate_after_recent_start: float  # 1733 inside if(recent): rate <- shape/mean.pann, else line 1281
    rgamma_1302_rate: float  # the rate the seeded rgamma in 1302 would receive (not sampled here)


def _scalar(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise UnsupportedSourceDomain(f"{name} must be one real number; R would recycle vectors")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise UnsupportedSourceDomain(f"{name} must be finite and positive")
    return value


def r_seq_by(from_: float, to: float, by: float) -> np.ndarray:
    """``seq(from, to, by = by)`` via seq.default for double arguments and positive ``by``."""
    from_, to, by = (_scalar(n, v) for n, v in (("from", from_), ("to", to), ("by", by)))
    delta = to - from_
    if delta <= 0:
        raise UnsupportedSourceDomain("Only increasing sequences of the preserved grid are supported")
    n = delta / by
    if n > np.iinfo(np.int32).max:
        raise UnsupportedSourceDomain("R stops: 'by' argument is much too small")
    count = int(n + SEQ_FUZZ)
    steps = np.arange(count + 1, dtype=np.float64) * by  # (0L:n) * by, rounded once
    return np.minimum(from_ + steps, to)  # + from, rounded again; pmin(x, to)


def r_which_min(values) -> int:
    """0-based ``which.min``: first index of the minimum; NA/NaN would be skipped in R and is rejected here."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or np.isnan(array).any():
        raise UnsupportedSourceDomain("which.min needs a nonempty numeric vector without NA")
    return int(np.argmin(array))


def gamma_grid_intersection(mean_pann, sd_pann, *, upper: float) -> GridResult:
    """One literal grid block (115-121 = 402-408 = 526-531 = 1270-1281); ``upper`` must name a source copy."""
    if upper not in UPPER_BY_SOURCE.values():
        raise UnsupportedSourceDomain("Grid upper bound must be one of the preserved source values 50000 or 5000")
    mean, sd = _scalar("mean.pann", mean_pann), _scalar("sd.pann", sd_pann)
    shape = r_seq_by(SEQ_FROM, upper, SEQ_BY)
    rate1 = shape / mean
    rate2 = np.sqrt(shape) / sd
    index = r_which_min(np.abs((rate2 - rate1) - 0))
    return GridResult(upper=upper, grid_length=len(shape), index=index, shape=float(shape[index]),
                      rate=float(rate2[index]), rate_expectation=float(rate1[index]),
                      at_lower_boundary=index == 0, at_upper_boundary=index == len(shape) - 1,
                      closed_form_shape=(mean / sd) ** 2)


def prior_gamma_chain(mean_pann, sd_pann, sd_tmp, *, recent: bool, min_age_at_least_2000: bool) -> PriorChainResult:
    """mcmc_prepare.R order for one site: default grid, doubling 1264-1267, grid 1270-1281, start value 1733."""
    if not all(isinstance(flag, bool) for flag in (recent, min_age_at_least_2000)):
        raise UnsupportedSourceDomain("Site conditions must be explicit booleans")
    upper = UPPER_BY_SOURCE["mcmc_prepare.R"]
    mean, sd_pann, sd_tmp = _scalar("mean.pann", mean_pann), _scalar("sd.pann", sd_pann), _scalar("sd.tmp", sd_tmp)
    default = gamma_grid_intersection(mean, sd_pann, upper=upper)
    doubled = recent and min_age_at_least_2000
    if doubled:
        sd_pann, sd_tmp = sd_pann * 2, sd_tmp * 2
    final = gamma_grid_intersection(mean, sd_pann, upper=upper)
    return PriorChainResult(
        default=default, pann_period_shape=default.shape, pann_period_rate=default.rate,
        diff_rate=default.shape / 10, variance_doubled=doubled, sd_pann=sd_pann, sd_tmp=sd_tmp, final=final,
        rate_after_recent_start=final.shape / mean if recent else final.rate, rgamma_1302_rate=final.rate)
