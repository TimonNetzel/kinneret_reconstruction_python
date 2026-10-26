"""Aufbereitung der Daten und Parameter fuer den Sampler (entspricht load_data.py und
mcmc_preparation.py des Originals)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import gamma, norm

from . import _core
from .config import SamplerConfig
from .data import Inputs


@dataclass
class Setup:
    inputs: Inputs
    # Transferfunktion auf dem Klimagitter: Pr(B = b | T, P), Form (n_out, dims_P, dims_T)
    biome_given_climate: np.ndarray
    biome_tf_temp: np.ndarray  # (B, dims) Randverteilung ueber T je Biom
    biome_tf_pann: np.ndarray  # (B, dims) Randverteilung ueber P je Biom
    # Alters-Tiefen-Transformation und Kerndaten im Altersraum
    depth_age_trans: np.ndarray  # (n_age, n_depth)
    taxa_spectrum_depth: np.ndarray  # (n_depth, K), zeilenweise auf 1 normiert
    taxa_spectrum_age: np.ndarray  # (n_age, K)
    ap_age: np.ndarray  # (n_age,)
    # Vorschlags- und Priorparameter
    pann_shape: float
    pann_rate: float
    dirichlet_spread: float
    jeffreys: np.ndarray
    tf_quantiles_temp: np.ndarray  # (B, 3): 5 %, 50 %, 95 %
    tf_quantiles_pann: np.ndarray
    tf_sd: np.ndarray  # (2B,)
    tf_lower: np.ndarray  # (2B,)
    tf_upper: np.ndarray  # (2B,)
    beta_shape1: float
    beta_shape2: float

    @property
    def num_taxa(self) -> int:
        return self.inputs.num_taxa

    @property
    def num_biomes(self) -> int:
        return self.inputs.num_biomes

    def core_model(self) -> dict:
        """Modellbeschreibung fuer den C++-Kern."""
        inp = self.inputs
        return dict(
            spectrum=self.taxa_spectrum_age,
            assign=inp.biomes_assign,
            ap=self.ap_age,
            wts_in_hidden=inp.wts_in_hidden,
            wts_hidden_out=inp.wts_hidden_out,
            wts_bias_hidden=inp.wts_bias_hidden,
            wts_bias_out=inp.wts_bias_out,
            normal_params=inp.normal_params,
            dirichlet_spread=float(self.dirichlet_spread),
            jeffreys=self.jeffreys,
            tf_sd=self.tf_sd,
            tf_lower=self.tf_lower,
            tf_upper=self.tf_upper,
            beta_shape1=self.beta_shape1,
            beta_shape2=self.beta_shape2,
            temp_mean=float(inp.recent_climate[0]),
            temp_sd=float(inp.recent_climate[1]),
            pann_shape=float(self.pann_shape),
            pann_rate=float(self.pann_rate),
        )


def weighted_quantiles(x: np.ndarray, weights: np.ndarray, probs) -> np.ndarray:
    """Quantile einer diskreten Verteilung auf dem Raster x (wie my_quantiles im Original)."""
    n = len(x)
    rw = np.cumsum(weights) / np.sum(weights)
    q = np.zeros(len(probs))
    for i, p in enumerate(probs):
        if p == 0:
            q[i] = x[0]
        elif p == 1:
            q[i] = x[n - 1]
        else:
            select = np.min(np.where(rw >= p))
            q[i] = np.mean(x[select:(select + 2)]) if rw[select] == p else x[select]
    return q


def normalize_temp(t, normal_params):
    return (t - normal_params[0]) / normal_params[1]


def normalize_pann(p, normal_params):
    return norm.ppf(gamma.cdf(np.sqrt(p), normal_params[2], scale=1 / normal_params[3]))


def transfer_function_grid(inp: Inputs, legacy: bool = False) -> np.ndarray:
    """Pr(B | T, P) des neuronalen Netzes auf dem (P x T)-Gitter, Form (n_out, dims, dims).

    legacy=True wertet das Netz mit der Originalfunktion aus (bitidentisch zum Original)."""
    dims = len(inp.temp_range)
    t_norm = normalize_temp(inp.temp_range, inp.normal_params)
    p_norm = normalize_pann(inp.pann_range, inp.normal_params)
    # Reihenfolge wie im Original: T laeuft schnell, P langsam -> reshape ergibt [P, T]
    grid_t = np.tile(t_norm, dims)
    grid_p = np.repeat(p_norm, dims)
    if legacy:
        probs = _core.legacy_nnet_probs(grid_t, grid_p, inp.wts_in_hidden, inp.wts_hidden_out,
                                        inp.wts_bias_hidden, inp.wts_bias_out)
    else:
        probs = _core.nnet_probs(_nnet_model(inp), grid_t, grid_p)
    return np.ascontiguousarray(probs.T).reshape(-1, dims, dims)


def _nnet_model(inp: Inputs) -> dict:
    # minimale Modellbeschreibung, mit der der C++-Kern nur das Netz auswertet
    k, b = inp.num_taxa, inp.num_biomes
    return dict(
        spectrum=np.ones((1, k)), assign=inp.biomes_assign, ap=np.zeros(1),
        wts_in_hidden=inp.wts_in_hidden, wts_hidden_out=inp.wts_hidden_out,
        wts_bias_hidden=inp.wts_bias_hidden, wts_bias_out=inp.wts_bias_out,
        normal_params=inp.normal_params, dirichlet_spread=1.0, jeffreys=np.ones(k),
        tf_sd=np.ones(2 * b), tf_lower=np.zeros(2 * b), tf_upper=np.ones(2 * b),
        beta_shape1=1.0, beta_shape2=1.0, temp_mean=0.0, temp_sd=1.0, pann_shape=1.0, pann_rate=1.0,
    )


def gamma_moments_grid(mean: float, sd: float) -> tuple[float, float]:
    """Gamma-Parameter wie im Original: Momentenanpassung per Suche auf einem Raster."""
    r = np.linspace(0.001, 0.5, num=3000)
    s1 = r * mean
    s2 = (r * sd) ** 2
    i = np.argmin(np.abs(s1 - s2))
    return float(s1[i]), float(r[i])


def gamma_moments_exact(mean: float, sd: float) -> tuple[float, float]:
    """Gamma-Parameter (Form, Rate) mit Mittelwert mean und Standardabweichung sd."""
    return mean**2 / sd**2, mean / sd**2


def build_setup(inp: Inputs, cfg: SamplerConfig) -> Setup:
    B, K = inp.num_biomes, inp.num_taxa

    # Transferfunktion und Randverteilungen je Biom
    # (Summation je Biom auf 2D-Arrays wie im Original, damit die Ergebnisse bitidentisch sind)
    bgc = transfer_function_grid(inp, legacy=cfg.kernel == "legacy")
    biome_tf_temp = np.stack([np.sum(bgc[b], axis=0) for b in range(B)])
    biome_tf_temp = biome_tf_temp / np.sum(biome_tf_temp, axis=1, keepdims=True)
    biome_tf_pann = np.stack([np.sum(bgc[b], axis=1) for b in range(B)])
    biome_tf_pann = biome_tf_pann / np.sum(biome_tf_pann, axis=1, keepdims=True)

    # Alters-Tiefen-Transformation: Pr(Tiefe | Alter) an den Probentiefen, je Alter normiert
    cols = (inp.depth - np.min(inp.depth)).astype(int)
    dat = np.ascontiguousarray(inp.depth_given_age[:, cols], dtype=float)
    for i in range(dat.shape[0]):
        dat[i, :] = dat[i, :] / np.sum(dat[i, :])

    spectrum_depth = inp.taxa_counts_depth / np.sum(inp.taxa_counts_depth, axis=1, keepdims=True)
    spectrum_age = np.empty((len(inp.age), K))
    for k in range(K):
        spectrum_age[:, k] = np.matmul(dat, spectrum_depth[:, k])
    ap_age = np.matmul(dat, inp.ap_depth)

    # Prior fuer den rezenten Niederschlag
    rc = inp.recent_climate
    if cfg.exact_gamma_moments:
        pann_shape, pann_rate = gamma_moments_exact(rc[2], rc[3])
    else:
        pann_shape, pann_rate = gamma_moments_grid(rc[2], rc[3])

    # Vorschlaege fuer die Transferfunktions-Stichproben: trunkierte Normalverteilungen
    t_range, p_range = inp.temp_range, inp.pann_range
    scale = cfg.tf_proposal_scale if cfg.tf_proposal_scale is not None else 2.38**2 / K
    q_t = np.empty((B, 3))
    q_p = np.empty((B, 3))
    sd_t = np.empty(B)
    sd_p = np.empty(B)
    for b in range(B):
        q_t[b] = weighted_quantiles(t_range, biome_tf_temp[b], np.array([0.05, 0.5, 0.95]))
        q_p[b] = weighted_quantiles(p_range, biome_tf_pann[b], np.array([0.05, 0.5, 0.95]))
        e_t = np.matmul(t_range, biome_tf_temp[b])
        e_p = np.matmul(p_range, biome_tf_pann[b])
        sd_t[b] = np.sqrt(np.matmul((t_range - e_t) ** 2, biome_tf_temp[b])) * scale
        sd_p[b] = np.sqrt(np.matmul((p_range - e_p) ** 2, biome_tf_pann[b])) * scale

    return Setup(
        inputs=inp,
        biome_given_climate=bgc,
        biome_tf_temp=biome_tf_temp,
        biome_tf_pann=biome_tf_pann,
        depth_age_trans=dat,
        taxa_spectrum_depth=spectrum_depth,
        taxa_spectrum_age=spectrum_age,
        ap_age=ap_age,
        pann_shape=pann_shape,
        pann_rate=pann_rate,
        dirichlet_spread=float(cfg.dirichlet_spread if cfg.dirichlet_spread is not None else K**2),
        jeffreys=np.repeat(0.5, K),
        tf_quantiles_temp=q_t,
        tf_quantiles_pann=q_p,
        tf_sd=np.concatenate([sd_t, sd_p]),
        tf_lower=np.concatenate([q_t[:, 0], q_p[:, 0]]),
        tf_upper=np.concatenate([q_t[:, 2], q_p[:, 2]]),
        beta_shape1=cfg.beta_shape1,
        beta_shape2=cfg.beta_shape2,
    )
