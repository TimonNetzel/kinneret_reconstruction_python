"""Konfiguration eines Rekonstruktionslaufs (TOML-Datei oder Voreinstellungen)."""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SamplerConfig:
    # "corrected": ueberarbeiteter Sampler; "legacy": bitidentischer Original-Sampler (nur 1 Kette)
    kernel: str = "corrected"
    sample_length: int = 1_000_000
    burn_in: int = 250_000
    thin: int = 75
    n_chains: int = 4
    seed: int = 2023
    # Korrekturen (nur fuer kernel = "corrected"); jede einzeln abschaltbar
    dirichlet_hastings_fix: bool = True  # kappa in der Hastings-Korrektur der Taxa-Gewichte
    exact_gamma_moments: bool = True  # Gamma-Parameter des P0-Priors analytisch statt per Gitter
    dispersed_init: bool = True  # zufaellige, gestreute Startwerte je Kette (fuer R-hat)
    # Modell- und Abstimmungsparameter
    dirichlet_spread: float | None = None  # None: num_taxa**2 wie im Paper
    tf_proposal_scale: float | None = None  # Faktor auf die TF-Standardabw.; None: 2.38**2/num_taxa wie im Original
    beta_shape1: float = 3.0
    beta_shape2: float = 3.0
    # Terme der Zielverteilung (nur fuer Validierungslaeufe abschalten)
    use_calibration: bool = True
    use_tf_likelihood: bool = True
    use_taxa_prior: bool = True


@dataclass
class PostprocessConfig:
    # "samples": Quartile direkt aus den Stichproben; "kde": ueber geglaettete Dichten wie im Original
    quantile_method: str = "samples"
    n_threads: int = 0  # 0: alle Kerne
    seed: int = 2023  # fuer die Stichproben der Prior-Transferfunktionen


@dataclass
class Config:
    data_dir: str = "data/in"
    output_dir: str = "results/run"
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    plots: bool = True
    report: bool = True

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        d = dict(d)
        sampler = SamplerConfig(**d.pop("sampler", {}))
        post = PostprocessConfig(**d.pop("postprocess", {}))
        cfg = cls(sampler=sampler, postprocess=post, **d)
        cfg.validate()
        return cfg

    @classmethod
    def from_toml(cls, path: str | Path) -> "Config":
        with open(path, "rb") as f:
            return cls.from_dict(tomllib.load(f))

    def validate(self) -> None:
        s = self.sampler
        if s.kernel not in ("corrected", "legacy"):
            raise ValueError("sampler.kernel muss 'corrected' oder 'legacy' sein")
        if s.kernel == "legacy" and s.n_chains != 1:
            raise ValueError("der Original-Sampler laeuft nur mit einer Kette (n_chains = 1)")
        if not 0 <= s.burn_in < s.sample_length or s.thin < 1:
            raise ValueError("ungueltige Werte fuer sample_length, burn_in oder thin")
        if self.postprocess.quantile_method not in ("samples", "kde"):
            raise ValueError("postprocess.quantile_method muss 'samples' oder 'kde' sein")


def legacy_config(**overrides: Any) -> Config:
    """Einstellungen, die exakt dem Originalskript (main.py, Tag v1.0-paper) entsprechen."""
    cfg = Config(
        sampler=SamplerConfig(
            kernel="legacy",
            n_chains=1,
            dirichlet_hastings_fix=False,
            exact_gamma_moments=False,
            dispersed_init=False,
        ),
        postprocess=PostprocessConfig(quantile_method="kde"),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    cfg.validate()
    return cfg
