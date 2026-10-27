"""Einlesen der Eingangsdaten (npz-Dateien in ``data/in``)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Inputs:
    # Altersmodell
    depth_given_age: np.ndarray  # (n_age, n_depth_cm): Pr(Tiefe | Alter), 1-cm-Raster
    age: np.ndarray  # (n_age,) regelmaessiges 50-Jahre-Raster [cal a BP]
    age_mean: np.ndarray  # (n_depth,) mittleres Alter je Probentiefe (altes Altersmodell)
    depth: np.ndarray  # (n_depth,) Probentiefen [cm]
    # Kerndaten
    taxa_counts_depth: np.ndarray  # (n_depth, n_taxa) Pollenanteile je Tiefe
    biomes_assign: np.ndarray  # (n_taxa, n_biomes) Zuordnung Taxon -> Biom
    taxa_names: np.ndarray  # (n_taxa,)
    ap_depth: np.ndarray  # (n_depth,) Baumpollenanteil [%]
    # Transferfunktion (neuronales Netz) und rezentes Klima
    wts_in_hidden: np.ndarray
    wts_hidden_out: np.ndarray
    wts_bias_hidden: np.ndarray
    wts_bias_out: np.ndarray
    temp_range: np.ndarray  # (dims,) T_DJF-Raster [°C]
    pann_range: np.ndarray  # (dims,) P_ANN-Raster [mm], nichtlinear
    normal_params: np.ndarray  # (4,) Normierung: mu_T, sd_T, Gamma-Form, Gamma-Rate (fuer sqrt(P))
    recent_climate: np.ndarray  # (4,) CRU: mean_T, sd_T, mean_P, sd_P
    # Darstellung
    terrain_colors_rgb: np.ndarray  # (4, 500) RGBA wie terrain.colors() in R

    @property
    def num_taxa(self) -> int:
        return self.taxa_counts_depth.shape[1]

    @property
    def num_biomes(self) -> int:
        return self.biomes_assign.shape[1]


def load_inputs(data_dir: str | Path) -> Inputs:
    d = Path(data_dir)
    adr = np.load(d / "age_depth_relationship.npz")
    core = np.load(d / "core_data.npz")
    nnet = np.load(d / "nnet_fit_params.npz")
    colors = np.load(d / "terrain_colors.npz")
    inp = Inputs(
        depth_given_age=adr["depth_given_age"],
        age=adr["age_regular"],
        age_mean=adr["age_mean"],
        depth=adr["depth"],
        taxa_counts_depth=core["taxa_spectrum_depth"],
        biomes_assign=core["biomes_assign"],
        taxa_names=core["taxa_names"],
        ap_depth=core["ap_depth"],
        wts_in_hidden=nnet["wts_in_hidden"],
        wts_hidden_out=nnet["wts_hidden_out"],
        wts_bias_hidden=nnet["wts_bias_hidden"],
        wts_bias_out=nnet["wts_bias_out"],
        temp_range=nnet["temp_range"],
        pann_range=nnet["pann_range"],
        normal_params=nnet["normal_params"],
        recent_climate=nnet["recent_climate"],
        terrain_colors_rgb=colors["terrain_colors_rgb"],
    )
    _validate(inp)
    return inp


def _validate(inp: Inputs) -> None:
    n_depth, n_taxa = inp.taxa_counts_depth.shape
    checks = {
        "depth": inp.depth.shape == (n_depth,),
        "ap_depth": inp.ap_depth.shape == (n_depth,),
        "biomes_assign": inp.biomes_assign.shape[0] == n_taxa,
        "taxa_names": inp.taxa_names.shape == (n_taxa,),
        "depth_given_age": inp.depth_given_age.shape[0] == inp.age.shape[0],
        "Tiefenraster": int(inp.depth.max() - inp.depth.min()) < inp.depth_given_age.shape[1],
        "temp/pann_range": inp.temp_range.shape == inp.pann_range.shape,
        "Biomzuordnung eindeutig": np.allclose(inp.biomes_assign.sum(axis=1), 1.0),
        "Netz-Ausgaenge": inp.wts_bias_out.shape[0] >= inp.biomes_assign.shape[1],
    }
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise ValueError(f"Inkonsistente Eingangsdaten: {', '.join(bad)}")
