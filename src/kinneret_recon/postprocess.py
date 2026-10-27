"""Auswertung der Posterior-Stichproben (entspricht post_processing() des Originals)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from . import _core
from .config import PostprocessConfig
from .model import Setup, weighted_quantiles
from .sampler import Posterior

QUARTILES = (0.25, 0.5, 0.75)
LINEAR_PANN_MAX = 6000.0  # obere Grenze des linearen Niederschlagsrasters [mm]


@dataclass
class Results:
    age: np.ndarray
    depth: np.ndarray
    temp_grid: np.ndarray  # Raster der Dichten fuer T
    pann_grid: np.ndarray  # lineares Raster der Dichten fuer P (0 bis 6000 mm)
    biome_grid: np.ndarray  # Raster der Dichten fuer Biomanteile (0 bis 1)
    # Stichproben der Rekonstruktion: (S, n_age) bzw. (S, n_depth); Biome (S, n, B)
    temp_age: np.ndarray
    pann_age: np.ndarray
    biomes_age: np.ndarray
    temp_depth: np.ndarray
    pann_depth: np.ndarray
    biomes_depth: np.ndarray
    # normierte Dichten je Zeitschritt/Tiefe (Summe 1)
    temp_dens_age: np.ndarray  # (n_age, dims)
    pann_dens_age: np.ndarray
    biomes_dens_age: np.ndarray  # (B, n_age, dims)
    temp_dens_depth: np.ndarray
    pann_dens_depth: np.ndarray
    biomes_dens_depth: np.ndarray
    # Quartile (3, n) bzw. (3, B, n)
    temp_age_q: np.ndarray
    pann_age_q: np.ndarray
    biomes_age_q: np.ndarray
    temp_depth_q: np.ndarray
    pann_depth_q: np.ndarray
    biomes_depth_q: np.ndarray
    # Transferfunktionen und Taxa-Gewichte
    post_weights: np.ndarray  # (S, K)
    tf_post_temp: np.ndarray  # (S, B)
    tf_post_pann: np.ndarray
    tf_prior_temp: np.ndarray  # (S, B) Stichproben der Prior-Randverteilungen
    tf_prior_pann: np.ndarray
    prior_temp_ci_size: np.ndarray  # (B,) Breite des 95-%-Intervalls
    post_temp_ci_size: np.ndarray
    prior_pann_ci_size: np.ndarray
    post_pann_ci_size: np.ndarray
    post_tf_probs: np.ndarray  # (B, dims_P, dims_T), auf Maximum 1 normiert
    quantile_method: str
    reconstruction: str = "mean"  # "mean" oder "mixture"

    # Stichproben der Rekonstruktion werden nicht gespeichert (aus posterior.npz reproduzierbar)
    _not_saved = ("temp_age", "pann_age", "biomes_age", "temp_depth", "pann_depth", "biomes_depth")

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, **{f.name: getattr(self, f.name) for f in fields(self) if f.name not in self._not_saved})

    @classmethod
    def load(cls, path: str | Path) -> "Results":
        d = np.load(path)
        kw = {f.name: (str(d[f.name]) if f.name in ("quantile_method", "reconstruction") else d[f.name])
              for f in fields(cls) if f.name not in cls._not_saved and f.name in d.files}
        return cls(**kw, **{k: None for k in cls._not_saved})


def biome_ratios(spectrum: np.ndarray, weights: np.ndarray, assign: np.ndarray) -> np.ndarray:
    """Biomanteile (S, n, B) fuer Stichproben der Taxa-Gewichte (S, K) und Spektren (n, K)."""
    b = np.einsum("nk,sk,kb->snb", spectrum, weights, assign, optimize=True)
    return b / b.sum(axis=2, keepdims=True)


def reconstruct(ratios: np.ndarray, tf: np.ndarray, num_biomes: int):
    """Mittlere Rekonstruktion aus Biomanteilen (S, n, B) und TF-Stichproben (S, 2B)."""
    t = np.einsum("snb,sb->sn", ratios, tf[:, :num_biomes])
    p = np.einsum("snb,sb->sn", ratios, tf[:, num_biomes:])
    return t, p


def reconstruct_mixture(ratios: np.ndarray, tf: np.ndarray, num_biomes: int, rng: np.random.Generator):
    """Stichproben der Mischverteilung nach Gl. (3): je Stichprobe s und Alter a ein Biom b mit
    Wahrscheinlichkeit ratios[s, a, b]; Klima = TF-Klimawert dieses Bioms in Stichprobe s."""
    u = rng.random(ratios.shape[:2])[..., None]
    b = np.minimum((u > np.cumsum(ratios, axis=2)).sum(axis=2), num_biomes - 1)  # (S, n)
    t = np.take_along_axis(tf[:, :num_biomes], b, axis=1)
    p = np.take_along_axis(tf[:, num_biomes:], b, axis=1)
    return t, p


def densities(samples: np.ndarray, grid: np.ndarray, n_threads: int) -> np.ndarray:
    """Normierte Kerndichte (Summe 1) je Spalte von samples (S, n) auf grid -> (n, dims)."""
    dens = _core.kde1d_rows(np.ascontiguousarray(samples.T), grid[0], grid[1] - grid[0], len(grid), n_threads)
    # entartete Spalten (alle Stichproben gleich, z. B. haengengebliebene Kette): Punktmasse am naechsten Gitterpunkt
    for i in np.flatnonzero(~np.isfinite(dens).all(axis=1) | (dens.sum(axis=1) <= 0)):
        dens[i] = 0.0
        dens[i, np.abs(grid - samples[0, i]).argmin()] = 1.0
    return dens / dens.sum(axis=1, keepdims=True)


def kde_quartiles(dens: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Quartile aus Dichten auf grid wie im Original: lineare Interpolation auf 10000 Punkte."""
    fine = np.linspace(grid.min(), grid.max(), 10000)
    q = np.empty((3, dens.shape[0]))
    for i in range(dens.shape[0]):
        p = interp1d(grid, dens[i], kind="linear", fill_value="extrapolate")(fine)
        p[p < 0] = 1e-6
        p = p / sum(p)
        q[:, i] = weighted_quantiles(fine, p, np.array(QUARTILES))
    return q


def _quartiles(samples, dens, grid, method):
    if method == "samples":
        return np.quantile(samples, QUARTILES, axis=0)
    return kde_quartiles(dens, grid)


def postprocess(setup: Setup, post: Posterior, cfg: PostprocessConfig, prior: Posterior | None = None) -> Results:
    """prior: optionaler Lauf ohne Kalibrierungsterme (Modell-Prior der TF-Klimawerte). Ohne ihn werden die
    Prior-Stichproben wie im Original aus den Randverteilungen auf dem (nichtlinearen) Klimagitter gezogen."""
    inp = setup.inputs
    B, dims = setup.num_biomes, len(inp.temp_range)
    nt = cfg.n_threads
    w = post.pooled("taxa_weights")
    tf = post.pooled("tf_sample")

    temp_grid = np.linspace(inp.temp_range.min(), inp.temp_range.max(), dims)
    pann_grid = np.linspace(0, LINEAR_PANN_MAX, dims)
    biome_grid = np.linspace(0, 1, dims)
    # Das Original interpoliert die T-Dichten mit inp.temp_range als Stuetzstellen
    temp_axis_kde = inp.temp_range

    out = {}
    mix_rng = np.random.default_rng(cfg.seed + 1)
    for name, spectrum in (("age", setup.taxa_spectrum_age), ("depth", setup.taxa_spectrum_depth)):
        ratios = biome_ratios(spectrum, w, inp.biomes_assign)
        if cfg.reconstruction == "mixture":
            t, p = reconstruct_mixture(ratios, tf, B, mix_rng)
        else:
            t, p = reconstruct(ratios, tf, B)
        td = densities(t, temp_grid, nt)
        pd = densities(p, pann_grid, nt)
        bd = np.stack([densities(ratios[:, :, b], biome_grid, nt) for b in range(B)])
        out[f"temp_{name}"], out[f"pann_{name}"], out[f"biomes_{name}"] = t, p, ratios
        out[f"temp_dens_{name}"], out[f"pann_dens_{name}"], out[f"biomes_dens_{name}"] = td, pd, bd
        out[f"temp_{name}_q"] = _quartiles(t, td, temp_axis_kde, cfg.quantile_method)
        out[f"pann_{name}_q"] = _quartiles(p, pd, pann_grid, cfg.quantile_method)
        out[f"biomes_{name}_q"] = np.stack(
            [_quartiles(ratios[:, :, b], bd[b], biome_grid, cfg.quantile_method) for b in range(B)], axis=1)

    # Prior- und Posterior-Transferfunktionen
    S = w.shape[0]
    if prior is not None:
        ptf = prior.pooled("tf_sample")
        prior_t, prior_p = ptf[:, :B], ptf[:, B:]
    else:
        rng = np.random.default_rng(cfg.seed)
        prior_t = np.empty((S, B))
        prior_p = np.empty((S, B))
        for b in range(B):
            prior_t[:, b] = rng.choice(inp.temp_range, size=S, p=setup.biome_tf_temp[b], replace=True)
            prior_p[:, b] = rng.choice(inp.pann_range, size=S, p=setup.biome_tf_pann[b], replace=True)
    post_t, post_p = tf[:, :B], tf[:, B:]
    ci = lambda a: np.diff(np.quantile(a, [0.025, 0.975], axis=0), axis=0)[0]

    # 2D-Posterior der TF-Stichproben je Biom auf dem (P x T)-Gitter
    pr, tr = inp.pann_range, inp.temp_range
    da, db = (pr.max() - pr.min()) / (dims - 1), (tr.max() - tr.min()) / (dims - 1)
    probs = np.empty((B, dims, dims))
    for b in range(B):
        z = _core.kde2d_grid(post_p[:, b].copy(), post_t[:, b].copy(), pr.min(), da, dims, tr.min(), db, dims)
        probs[b] = z / z.max()

    return Results(
        age=inp.age, depth=inp.depth, temp_grid=temp_grid, pann_grid=pann_grid, biome_grid=biome_grid,
        post_weights=w, tf_post_temp=post_t, tf_post_pann=post_p, tf_prior_temp=prior_t, tf_prior_pann=prior_p,
        prior_temp_ci_size=ci(prior_t), post_temp_ci_size=ci(post_t),
        prior_pann_ci_size=ci(prior_p), post_pann_ci_size=ci(post_p),
        post_tf_probs=probs, quantile_method=cfg.quantile_method, reconstruction=cfg.reconstruction, **out,
    )
