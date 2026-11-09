"""Artificial Morlet-like real bands for the stress cases; deterministic and free of any biwavelet code."""
import math


def wavenumbers(n, dt):
    half = [(j * 2 * math.pi) / (n * dt) for j in range(1, n // 2 + 1)]
    return [0.0] + half + [-half[j - 1] for j in range((n - 1) // 2, 0, -1)]


def band(n, dt, scale):
    return [math.exp(-(scale * k - 6.0) ** 2 / 2.0) * math.sqrt(2 * math.pi * scale / dt) * math.sqrt(n) if k > 0 else 0.0 for k in wavenumbers(n, dt)]


def band_rows(n, dt, m):
    cols = [band(n, dt, 2.0 * dt * 2 ** (j / 8)) for j in range(m)]
    return [[col[i] for col in cols] for i in range(n)]
