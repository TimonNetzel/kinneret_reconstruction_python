"""Dirichlet ratio of parameter_stage_weights, mcmc_functions_v1.6.R 1269-1274, with EXPLICIT old and new weights.

The provider of the unqualified ``ddirichlet`` is bound only within the CRAN window 2018-04-23..2023-07-07:
LaplacesDemon, first on the search path, with the same text in 16.1.0-16.1.6 (r_runtime_crosscheck_2026_11_03).
Kept literally from that text: ``rbind(x)`` and ``matrix(alpha, nrow(x), length(alpha), byrow=TRUE)`` for one row;
``if(any(rowSums(x) != 1)) x / rowSums(x)`` evaluates the row sum but assigns nothing, so x is NOT renormalised;
``x < 0`` and ``alpha <= 0`` stop; ``dens = (lgamma(rowSums(alpha)) - rowSums(lgamma(alpha))) +
rowSums((alpha-1)*log(x))`` and ``exp(dens)`` unless ``log``. From the source: weights_ratio with alpha
``rep(0.5, num_taxa)`` (global num_taxa), proposal_ratio = ddirichlet(old, new)/ddirichlet(new, old) without
dirichlet_spread, their product, and ``if(is.na(parameter_ratio)) parameter_ratio <- 0`` (NA and NaN become 0, Inf
stays).

Precision: every double operation of the source is one IEEE operation here; lgamma, log and exp are evaluated with an
80-digit Decimal intermediate and rounded once to the nearest double. R uses its own nmath lgamma and the C library's
log/exp, which need not be correctly rounded, and may sum rowSums in long double; row sums here are the correctly
rounded exact sums with the accumulator flag of historical_weights_stage. Distances to R 3.5.0 are measured in the
packet, not assumed to be zero. Rejected: lengths that differ or differ from num_taxa (non-conformable matrices in
R), fewer than two weights, non-finite, zero, negative or non-numeric weights (the loop 1249 needs min > 0; ddirichlet
would stop for alpha <= 0). No random state, no R execution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from numbers import Real

import numpy as np

from historical_weights_stage import row_sum_na_rm

PREC = 80


class UnsupportedSourceDomain(ValueError):
    """Input needs R/runtime semantics outside this finite fragment."""


def _pi() -> Decimal:
    """pi by Machin's formula at PREC + 10 digits."""
    with localcontext() as ctx:
        ctx.prec = PREC + 10

        def arctan_inv(n):
            x, n2, total, k, sign = Decimal(1) / n, n * n, Decimal(0), 1, 1
            term = x
            while term:
                total += sign * term / k
                term /= n2
                k += 2
                sign = -sign
            return total
        return 16 * arctan_inv(5) - 4 * arctan_inv(239)


def _bernoulli(m: int) -> list[Fraction]:
    """B_0..B_m (Akiyama-Tanigawa), B_1 = +1/2 convention unused here."""
    a, out = [Fraction(0)] * (m + 1), []
    for n in range(m + 1):
        a[n] = Fraction(1, n + 1)
        for j in range(n, 0, -1):
            a[j - 1] = j * (a[j - 1] - a[j])
        out.append(a[0])
    return out


_B = _bernoulli(60)
_LN_2PI_HALF = None


def lgamma_cr(x: float) -> float:
    """log|Gamma(x)| for x > 0, rounded once to the nearest double (shift to >= 40, Stirling series)."""
    global _LN_2PI_HALF
    if not (math.isfinite(x) and x > 0):
        raise UnsupportedSourceDomain('lgamma needs a finite positive argument here')
    if x in (1.0, 2.0):
        return 0.0
    with localcontext() as ctx:
        ctx.prec = PREC + 10
        if _LN_2PI_HALF is None:
            _LN_2PI_HALF = (2 * _pi()).ln() / 2
        z, shift = Decimal(x), Decimal(0)
        while z < 40:
            shift += z.ln()
            z += 1
        series = sum(Decimal(_B[2 * k].numerator) / Decimal(_B[2 * k].denominator)
                     / (2 * k * (2 * k - 1) * z ** (2 * k - 1)) for k in range(1, 26))
        value = (z - Decimal('0.5')) * z.ln() - z + _LN_2PI_HALF + series - shift
        return float(+value)


def log_cr(x: float) -> float:
    """log(x) for 0 < x < Inf rounded once; log(0) = -Inf."""
    if x == 0:
        return -math.inf
    with localcontext() as ctx:
        ctx.prec = PREC
        return float(Decimal(x).ln())


def exp_cr(d: float) -> float:
    """exp(d) rounded once; overflows to Inf and underflows to (sub)normal or zero like IEEE."""
    if d == math.inf:
        return math.inf
    if d == -math.inf:
        return 0.0
    if math.isnan(d):
        return math.nan
    if d > 1000:  # beyond log(DBL_MAX) = 709.78
        return math.inf
    if d < -1000:  # below log of the smallest subnormal, -744.44
        return 0.0
    with localcontext() as ctx:
        ctx.prec = PREC
        ctx.Emax, ctx.Emin = 10**6, -10**6
        return float(Decimal(d).exp())


def _div(a: float, b: float) -> float:
    """One IEEE double division (Python raises at b == 0)."""
    if b == 0:
        return math.nan if (a == 0 or math.isnan(a)) else math.copysign(math.inf, a) * math.copysign(1.0, b)
    return a / b


@dataclass(frozen=True)
class Density:
    value: float  # exp(dens), or dens with log=TRUE
    log_density: float  # dens
    lgamma_of_alpha_sum: float  # lgamma(rowSums(alpha))
    sum_of_lgamma_alpha: float  # rowSums(lgamma(alpha))
    sum_of_weighted_logs: float  # rowSums((alpha-1)*log(x))
    row_sum_x: float  # rowSums(x), only tested against 1
    normalisation_skipped: bool  # rowSums(x) != 1: the source line divides but assigns nothing
    accumulator_independent: bool  # all four row sums are exact doubles in any order


@dataclass(frozen=True)
class DirichletRatio:
    num_taxa: int
    weights_numerator: Density  # ddirichlet(new_weights, rep(0.5, num_taxa))
    weights_denominator: Density  # ddirichlet(old_weights, rep(0.5, num_taxa))
    weights_ratio: float
    proposal_numerator: Density  # ddirichlet(old_weights, new_weights)
    proposal_denominator: Density  # ddirichlet(new_weights, old_weights)
    proposal_ratio: float
    parameter_ratio_before_na: float  # proposal_ratio * weights_ratio
    parameter_ratio: float  # 1273: NA/NaN -> 0
    na_replaced: bool


def _vector(name: str, values) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or (isinstance(values, np.ndarray) and values.ndim != 1):
        raise UnsupportedSourceDomain(f'{name} must be a flat numeric vector')
    out = []
    for v in values:
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (Real, np.floating, np.integer)):
            raise UnsupportedSourceDomain(f'{name} must be real numbers (no bool, string or NA)')
        out.append(float(v))
    return tuple(out)


def ddirichlet(x, alpha, *, log: bool = False) -> Density:
    """LaplacesDemon::ddirichlet for one vector x (16.1.0-16.1.6 text)."""
    x, alpha = _vector('x', x), _vector('alpha', alpha)
    if len(x) != len(alpha) or len(x) < 2:
        raise UnsupportedSourceDomain('x and alpha need equal length >= 2; other lengths are non-conformable in R')
    if any(math.isnan(v) for v in x + alpha):
        raise UnsupportedSourceDomain('NA/NaN: if(any(rowSums(x) != 1)) would stop with a missing value')
    row_x = row_sum_na_rm(x)
    if any(v < 0 for v in x):
        raise UnsupportedSourceDomain('x must be non-negative (ddirichlet stops)')
    if any(a <= 0 for a in alpha):
        raise UnsupportedSourceDomain('alpha must be positive (ddirichlet stops)')
    if not all(math.isfinite(v) for v in x + alpha):
        raise UnsupportedSourceDomain('infinite weights are outside the checked domain')
    if any(v == 0 for v in x):
        # log(0) = -Inf gives +Inf or NaN terms; rowSums here has no na.rm, so that path is not emulated
        raise UnsupportedSourceDomain('x = 0 (log(0) path) is outside the checked domain')
    s_alpha = row_sum_na_rm(alpha)
    s_lgamma = row_sum_na_rm([lgamma_cr(a) for a in alpha])
    s_logs = row_sum_na_rm([(a - 1.0) * log_cr(v) for a, v in zip(alpha, x)])
    dens = (lgamma_cr(s_alpha.value) - s_lgamma.value) + s_logs.value
    return Density(value=dens if log else exp_cr(dens), log_density=dens,
                   lgamma_of_alpha_sum=lgamma_cr(s_alpha.value), sum_of_lgamma_alpha=s_lgamma.value,
                   sum_of_weighted_logs=s_logs.value, row_sum_x=row_x.value,
                   normalisation_skipped=row_x.value != 1.0,
                   accumulator_independent=all(s.accumulator_independent for s in (row_x, s_alpha, s_lgamma, s_logs)))


def dirichlet_ratio(old_weights, new_weights, num_taxa: int) -> DirichletRatio:
    """Lines 1269-1274 for explicit weights; num_taxa is the global taxon count."""
    if isinstance(num_taxa, (bool, np.bool_)) or not isinstance(num_taxa, (int, np.integer)) or num_taxa < 2:
        raise UnsupportedSourceDomain('num_taxa must be an integer count >= 2')
    old, new = _vector('old_weights', old_weights), _vector('new_weights', new_weights)
    if not (len(old) == len(new) == int(num_taxa)):
        raise UnsupportedSourceDomain('old and new weights need length num_taxa (rep(0.5, num_taxa) otherwise '
                                      'mismatches)')
    if not all(math.isfinite(v) and v > 0 for v in old + new):
        raise UnsupportedSourceDomain('weights must be finite and > 0 (loop 1249, start 1/num_taxa)')
    half = (0.5,) * int(num_taxa)
    wn, wd = ddirichlet(new, half), ddirichlet(old, half)
    pn, pd = ddirichlet(old, new), ddirichlet(new, old)
    weights_ratio, proposal_ratio = _div(wn.value, wd.value), _div(pn.value, pd.value)
    raw = proposal_ratio * weights_ratio
    return DirichletRatio(num_taxa=int(num_taxa), weights_numerator=wn, weights_denominator=wd,
                          weights_ratio=weights_ratio, proposal_numerator=pn, proposal_denominator=pd,
                          proposal_ratio=proposal_ratio, parameter_ratio_before_na=raw,
                          parameter_ratio=0.0 if math.isnan(raw) else raw, na_replaced=math.isnan(raw))
