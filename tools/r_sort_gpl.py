"""Python translation of two sort routines of R, kept in their own file because of their licence.

Translated from R 3.5.0, ``src/main/qsort-body.c`` (``R_qsort_I``, CACM algorithm 347 by R. C. Singleton with the
remark by Peto) and the ``psort_body`` macro of ``src/main/sort.c`` (``rPsort``).
Copyright (C) 2002-2017 The R Core Team. This file, unlike the MIT-licensed rest of the repository, stays under the
GNU General Public License, version 2 or later, as the R sources it was translated from. See
https://www.R-project.org/Licenses/.

Both routines are not stable and their permutation of equal values is part of the observable result of
matrixStats::weightedMedian, which is the only reason for translating them literally. Only finite-or-infinite doubles
without NaN reach them; R's ``rcmp`` then reduces to ``<``.
"""
from __future__ import annotations


def r_qsort_i(values: list, index: list) -> None:
    """``R_qsort_I(v, I, 1, n)``: sort ``values`` ascending in place and permute ``index`` alongside."""
    n = len(values)
    if n != len(index):
        raise ValueError('values and index differ in length')
    v, ind = [None, *values], [None, *index]  # the C body switches to 1-indexing with --v and --I
    il, iu = [0] * 40, [0] * 40
    i, j = 1, n
    r = 0.375
    ii, m = i, 1
    label = 'L10'
    while True:
        if label == 'L10':
            if i < j:
                if r < 0.5898437:
                    r += 0.0390625
                else:
                    r -= 0.21875
                label = 'L20'
            else:
                label = 'L80'
            continue
        if label == 'L20':
            k = i
            ij = i + int((j - i) * r)
            it, vt = ind[ij], v[ij]
            if v[i] > vt:
                ind[ij], ind[i] = ind[i], it
                it = ind[ij]
                v[ij], v[i] = v[i], vt
                vt = v[ij]
            l = j
            if v[j] < vt:
                ind[ij], ind[j] = ind[j], it
                it = ind[ij]
                v[ij], v[j] = v[j], vt
                vt = v[ij]
                if v[i] > vt:
                    ind[ij], ind[i] = ind[i], it
                    it = ind[ij]
                    v[ij], v[i] = v[i], vt
                    vt = v[ij]
            while True:
                l -= 1
                while v[l] > vt:
                    l -= 1
                tt, vtt = ind[l], v[l]
                k += 1
                while v[k] < vt:
                    k += 1
                if k > l:
                    break
                ind[l], ind[k] = ind[k], tt
                v[l], v[k] = v[k], vtt
            m += 1
            if l - i <= j - k:
                il[m], iu[m] = k, j
                j = l
            else:
                il[m], iu[m] = i, l
                i = k
            label = 'after'
            continue
        if label == 'L80':
            if m == 1:
                values[:] = v[1:]
                index[:] = ind[1:]
                return
            i, j = il[m], iu[m]
            m -= 1
            label = 'after'
            continue
        # common continuation after the if/else of L10
        if j - i > 10:
            label = 'L20'
            continue
        if i == ii:
            label = 'L10'
            continue
        i -= 1
        while True:  # L100
            while True:
                i += 1
                if i == j:
                    label = 'L80'
                    break
                it, vt = ind[i + 1], v[i + 1]
                if not v[i] <= vt:
                    break
            if label == 'L80':
                break
            k = i
            while True:  # L110
                ind[k + 1] = ind[k]
                v[k + 1] = v[k]
                k -= 1
                if not vt < v[k]:
                    break
            ind[k + 1], v[k + 1] = it, vt


def r_psort(x: list, n: int, k: int) -> None:
    """``rPsort(x, n, k)``, k 0-based and k < n: x[k] lands on its place, smaller to the left, larger to the right."""
    low, high = 0, n - 1
    while low < high:
        v = x[k]
        i, j = low, high
        while i <= j:
            while x[i] < v:
                i += 1
            while v < x[j]:
                j -= 1
            if i <= j:
                x[i], x[j] = x[j], x[i]
                i += 1
                j -= 1
        if j < k:
            low = i
        if k < i:
            high = j
