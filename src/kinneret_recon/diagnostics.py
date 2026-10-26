"""Konvergenzdiagnostik: rangnormiertes Split-R-hat und effektive Stichprobengroesse (ESS)
nach Vehtari, Gelman, Simpson, Carpenter & Buerkner (2021), Bayesian Analysis 16(2)."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata

from .sampler import Posterior


def _split(x: np.ndarray) -> np.ndarray:
    """(M, N) -> (2M, N//2): jede Kette in zwei Haelften teilen."""
    n = x.shape[1] // 2
    return np.concatenate([x[:, :n], x[:, -n:]], axis=0)


def _rank_normalize(x: np.ndarray) -> np.ndarray:
    r = rankdata(x, axis=None).reshape(x.shape)
    return norm.ppf((r - 0.375) / (x.size + 0.25))


def _rhat_basic(x: np.ndarray) -> float:
    m, n = x.shape
    if np.all(x == x.flat[0]):
        return np.nan
    w = x.var(axis=1, ddof=1).mean()
    b = n * x.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((n - 1) / n * w + b / n) / w))


def rhat(x: np.ndarray) -> float:
    """Rangnormiertes, gefaltetes Split-R-hat einer Groesse x mit Form (Ketten, Iterationen)."""
    xs = _split(np.asarray(x, dtype=float))
    folded = np.abs(xs - np.median(xs))
    return max(_rhat_basic(_rank_normalize(xs)), _rhat_basic(_rank_normalize(folded)))


def _autocov(x: np.ndarray) -> np.ndarray:
    """Autokovarianz je Kette (biased, Normierung 1/N) per FFT."""
    m, n = x.shape
    xc = x - x.mean(axis=1, keepdims=True)
    size = 2 ** int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(xc, size, axis=1)
    return np.fft.irfft(f * np.conj(f), size, axis=1)[:, :n] / n


def ess(x: np.ndarray) -> float:
    """ESS mit Geyers initialer monotoner Folge (wie Stan/ArviZ), x mit Form (Ketten, Iterationen)."""
    x = np.asarray(x, dtype=float)
    m, n = x.shape
    if n < 4 or np.all(x == x.flat[0]):
        return np.nan
    acov = _autocov(x)
    mean_var = acov[:, 0].mean() * n / (n - 1)
    var_plus = mean_var * (n - 1) / n + (x.mean(axis=1).var(ddof=1) if m > 1 else 0.0)
    rho = np.zeros(n)
    rho[0] = 1.0
    rho_even, rho_odd = 1.0, 1.0 - (mean_var - acov[:, 1].mean()) / var_plus
    rho[1] = rho_odd
    t = 1
    while t < n - 3 and rho_even + rho_odd > 0:
        rho_even = 1.0 - (mean_var - acov[:, t + 1].mean()) / var_plus
        rho_odd = 1.0 - (mean_var - acov[:, t + 2].mean()) / var_plus
        if rho_even + rho_odd >= 0:
            rho[t + 1], rho[t + 2] = rho_even, rho_odd
        t += 2
    max_t = t - 2
    t = 1
    while t <= max_t - 2:  # monotone Folge erzwingen
        if rho[t + 1] + rho[t + 2] > rho[t - 1] + rho[t]:
            rho[t + 1] = rho[t + 2] = (rho[t - 1] + rho[t]) / 2
        t += 2
    total = m * n
    tau = -1.0 + 2.0 * rho[: max_t + 1].sum() + rho[max_t + 1]
    tau = max(tau, 1.0 / np.log10(total))
    return float(total / tau)


def ess_bulk(x: np.ndarray) -> float:
    return ess(_rank_normalize(_split(np.asarray(x, dtype=float))))


def ess_tail(x: np.ndarray) -> float:
    """Minimum der ESS fuer die Indikatoren x <= q05 und x <= q95."""
    xs = _split(np.asarray(x, dtype=float))
    q05, q95 = np.quantile(xs, [0.05, 0.95])
    return min(ess((xs <= q05).astype(float)), ess((xs <= q95).astype(float)))


def parameter_table(post: Posterior, taxa_names, biome_labels) -> dict[str, np.ndarray]:
    """Alle Groessen nach dem Burn-in mit Form (Ketten, Iterationen), benannt."""
    m = post.post_mask
    B = len(biome_labels)
    out = {"log_target": post.log_target[:, m], "R2": post.expl_variance[:, m],
           "T0": post.recent_temp[:, m], "P0": post.recent_pann[:, m]}
    if np.ptp(post.alpha) > 0:
        out["alpha"] = post.alpha[:, m]
    for b, lab in enumerate(biome_labels):
        out[f"T_{lab}"] = post.tf_sample[:, m, b]
        out[f"P_{lab}"] = post.tf_sample[:, m, B + b]
    for k, name in enumerate(taxa_names):
        out[f"w[{name}]"] = post.taxa_weights[:, m, k]
    return {k: v for k, v in out.items() if np.all(np.isfinite(v))}


def summarize(post: Posterior, taxa_names, biome_labels) -> list[dict]:
    """Tabelle mit Mittelwert, Quantilen, R-hat und ESS je Groesse."""
    rows = []
    for name, x in parameter_table(post, taxa_names, biome_labels).items():
        q = np.quantile(x, [0.025, 0.5, 0.975])
        rows.append(dict(
            parameter=name, mean=float(x.mean()), sd=float(x.std()), q2_5=float(q[0]), median=float(q[1]),
            q97_5=float(q[2]), rhat=rhat(x),
            ess_bulk=ess_bulk(x), ess_tail=ess_tail(x),
        ))
    return rows
