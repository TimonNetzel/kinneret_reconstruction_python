"""``reconst`` and the two predictants of mcmc_functions_v1.6.R 1817-1820 and their direct readers, from a given ``tf_sample``.

Explicit matrices only: no draw (lines 1812-1813 stay outside), no sampler, no original data. Base R only, so no package binding.

Selection, line 1818. ``taxa.appear.core[reconst_ages,]`` is a matrix index with a logical vector: a shorter mask is recycled, a longer
one makes R stop, NA gives a row of NA. R drops a dimension of extent 1, so one selected row (or one taxon) becomes a plain vector, and
``%*%`` then treats it as a row vector when its length equals ``nrow(tf_sample)``, else as a column when ``tf_sample`` has one row.
In the live path ``reconst_ages`` is all TRUE for every site: mcmc_prepare.R:1466 is overwritten at 1522 because line 1500 ends in ``& FALSE``.

Arithmetic of ``%*%`` as R 3.5.0 x64 on Windows performs it with options(matprod = "default"):

- If mayHaveNaNOrInf fires on either operand (a pairwise-sum test, so two huge finite neighbours can trigger it) the product runs in
  simple_matprod, three loops with a plain double accumulator. Only options(matprod = "internal") uses internal_matprod with long double.
- Otherwise reference BLAS: dgemm, or dgemv with TRANS = "T" when one row is selected. Each is also a plain double accumulation in taxon
  order, so every finite row is bit-identical on both paths: a NaN elsewhere in the matrix does not change the finite rows.
- Accumulation therefore must not use Python's ``sum``, which is compensated for floats from 3.12 on, and is not R's long-double ``sum``.

NA and NaN. Which token survives depends on the operand order of the compiled loop and is not in the R source. Measured in R 3.5.0
on all 1296 combinations: a product takes the first missing operand of ``x * y`` and a sum keeps a missing accumulator before a missing
product. ``0 * NA`` stays NA and ``0 * Inf`` is NaN.

Rejected instead of emulated: rows of unequal length, a taxon matrix without rows or columns, ``tf_sample`` that is not n x 2, ``taxa_change`` TRUE.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np

from historical_local_cor import NA, UnsupportedSourceDomain, _may_have_nan_or_inf

__all__ = ['UnsupportedSourceDomain', 'NA', 'Selected', 'Reconst', 'ReconstructionBlock', 'select_rows', 'matrix_product', 'reconst_predictant',
           'youngest_values', 'recent_climate', 'predictant_current', 'reconstruction_block']


@dataclass(frozen=True)
class Selected:
    """The value of ``taxa.appear.core[reconst_ages,]``: ``dim`` is None when R dropped it to a plain vector."""
    dim: tuple[int, int] | None
    values: tuple  # row-wise for a matrix, flat for a vector


@dataclass(frozen=True)
class Reconst:
    selected: Selected
    path: str  # 'dgemm', 'dgemv_t', 'simple_matprod' or 'zero_extent'
    nrow: int
    tann: tuple  # reconst[,1]
    pann: tuple  # reconst[,2]


@dataclass(frozen=True)
class ReconstructionBlock:
    reconst: Reconst
    youngest_values: tuple
    post_tann: object
    post_pann: object
    predictant_current: tuple
    current_name: str


def _missing(v) -> bool:
    return v is NA or (isinstance(v, float) and math.isnan(v))


def _number(name: str, v):
    if v is None or v is NA:
        return NA
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (Real, np.floating, np.integer)):
        raise UnsupportedSourceDomain(f'{name} must hold real numbers, NaN, Inf or None/NA')
    return float(v)


def _matrix(name: str, rows) -> list[list]:
    if isinstance(rows, (str, bytes)):
        raise UnsupportedSourceDomain(f'{name} must be a list of rows')
    out = [[_number(name, v) for v in row] for row in rows]
    if not out or not out[0]:
        raise UnsupportedSourceDomain(f'{name} needs at least one row and one column')
    if any(len(row) != len(out[0]) for row in out):
        raise UnsupportedSourceDomain(f'{name}: rows differ in length')
    return out


def _flag(name: str, value) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise UnsupportedSourceDomain(f'{name} must be an explicit boolean; if(NA) stops in R')
    return bool(value)


def select_rows(taxa_appear_core, reconst_ages) -> Selected:
    """Line 1818, left of ``%*%``."""
    matrix, mask = _matrix('taxa_appear_core', taxa_appear_core), list(reconst_ages)
    nrow, ncol = len(matrix), len(matrix[0])
    if len(mask) > nrow:
        raise UnsupportedSourceDomain('(subscript) logical subscript too long: R stops at line 1818')
    chosen = []
    if mask:
        for i in range(nrow):
            flag = mask[i % len(mask)]
            if flag is None or flag is NA:
                chosen.append([NA] * ncol)
            elif not isinstance(flag, (bool, np.bool_)):
                raise UnsupportedSourceDomain('reconst_ages must hold booleans or None/NA')
            elif flag:
                chosen.append(matrix[i])
    k = len(chosen)
    if k == 1:
        return Selected(None, tuple(chosen[0]))
    if ncol == 1:
        return Selected(None, tuple(row[0] for row in chosen))
    return Selected((k, ncol), tuple(tuple(row) for row in chosen))


def _times(a, b):
    if _missing(a):
        return a
    return b if _missing(b) else a * b


def _plus(total, term):
    if _missing(total):
        return total
    return term if _missing(term) else total + term


def _nan_or_inf_in(values: list) -> bool:
    return _may_have_nan_or_inf([math.nan if v is NA else v for v in values])


def matrix_product(selected: Selected, tf_sample) -> Reconst:
    """Line 1818, ``%*%`` with the shape rules of do_matprod and the arithmetic of matprod."""
    tf = _matrix('tf_sample', tf_sample)
    nry = len(tf)
    if len(tf[0]) != 2:
        raise UnsupportedSourceDomain('tf_sample must have two columns (mcmc_conduction.R:115)')
    if selected.dim is None:
        length = len(selected.values)
        if length == nry:
            rows = [list(selected.values)]
        elif nry == 1:
            rows = [[v] for v in selected.values]
        else:
            raise UnsupportedSourceDomain('non-conformable arguments: R stops at line 1818')
    else:
        rows = [list(r) for r in selected.values]
        if selected.dim[1] != nry:
            raise UnsupportedSourceDomain('non-conformable arguments: R stops at line 1818')
    nrx = len(rows)
    if nrx == 0:
        return Reconst(selected, 'zero_extent', 0, (), ())
    ncx = len(rows[0])
    column_major_x = [rows[i][j] for j in range(ncx) for i in range(nrx)]
    column_major_y = [tf[j][k] for k in range(2) for j in range(nry)]
    if _nan_or_inf_in(column_major_x) or _nan_or_inf_in(column_major_y):
        path = 'simple_matprod'
    else:
        path = 'dgemv_t' if nrx == 1 else 'dgemm'
    columns = ([], [])
    for row in rows:
        for k in range(2):
            total = 0.0
            for j in range(ncx):  # explicit double accumulation; Python's sum() is compensated from 3.12 on
                total = _plus(total, _times(row[j], tf[j][k]))
            columns[k].append(total)
    return Reconst(selected, path, nrx, tuple(columns[0]), tuple(columns[1]))


def reconst_predictant(taxa_appear_core, reconst_ages, tf_sample) -> Reconst:
    """Lines 1817-1820."""
    return matrix_product(select_rows(taxa_appear_core, reconst_ages), tf_sample)


def youngest_values(reconst: Reconst) -> tuple:
    """Line 1825, ``reconst[1,]``. The first selected row, not the first row of the core; R stops if nothing was selected."""
    if reconst.nrow == 0:
        raise UnsupportedSourceDomain('subscript out of bounds: reconst[1,] on zero rows stops R at line 1825')
    return reconst.tann[0], reconst.pann[0]


def _first(values):
    return values[0] if values else NA


def recent_climate(reconst: Reconst, recent: bool, prior_mean_tmp, prior_mean_pann) -> tuple:
    """Lines 1844-1857. The prior means are globals in the source and explicit here."""
    if _flag('recent', recent):
        return _first(reconst.tann), _first(reconst.pann)
    return prior_mean_tmp, prior_mean_pann


def predictant_current(reconst: Reconst, tann: bool) -> tuple[str, tuple]:
    """Lines 1939-1943."""
    return ('predictant_tann', reconst.tann) if _flag('tann', tann) else ('predictant_pann', reconst.pann)


def reconstruction_block(*, taxa_appear_core, reconst_ages, tf_sample, tann: bool, recent: bool, prior_mean_tmp=None, prior_mean_pann=None,
                         taxa_change: bool = False) -> ReconstructionBlock:
    """Lines 1817-1857 and 1939-1943, in source order, for ``taxa_change`` FALSE."""
    if _flag('taxa_change', taxa_change):
        raise UnsupportedSourceDomain('taxa_change TRUE draws from rnorm (line 1837); mcmc_prepare.R:1500 keeps it FALSE')
    reconst = reconst_predictant(taxa_appear_core, reconst_ages, tf_sample)
    youngest = youngest_values(reconst)
    post_tann, post_pann = recent_climate(reconst, recent, prior_mean_tmp, prior_mean_pann)
    name, current = predictant_current(reconst, tann)
    return ReconstructionBlock(reconst, youngest, post_tann, post_pann, current, name)
