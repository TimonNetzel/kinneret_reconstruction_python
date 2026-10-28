"""Konfiguration eines Rekonstruktionslaufs (TOML-Datei oder Voreinstellungen)."""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Prior des rezenten Klimas der Voreinstellung (seit v3): CRU TS 4.07, 1961-1990, Einzugsgebiet 50 km um den See,
# Gauss-Gewichtung; Sd = Unsicherheit des Gebietsmittels (kinneret-recon cru-prior -r 50, Pruefbericht 3.9)
CRU_PRIOR_50KM = {"recent_temp_mean": 11.09, "recent_temp_sd": 0.87,
                  "recent_pann_mean": 521.8, "recent_pann_sd": 41.4}


@dataclass
class SamplerConfig:
    # "corrected": ueberarbeiteter Sampler; "legacy": bitidentischer Original-Sampler (nur 1 Kette)
    kernel: str = "corrected"
    # Voreinstellung (hierarchischer Prior der Taxa-Gewichte): 4 x 10 000 Stichproben; alpha mischt langsam,
    # daher 8 Mio. Iterationen je Kette (ESS(alpha) ~ 2000). Mit festem Dir(alpha) genuegen 2 Mio. / 500 000 / 150.
    sample_length: int = 8_000_000
    burn_in: int = 2_000_000
    thin: int = 600
    n_chains: int = 4
    seed: int = 2023
    # Korrekturen (nur fuer kernel = "corrected"); jede einzeln abschaltbar
    dirichlet_hastings_fix: bool = True  # kappa in der Hastings-Korrektur der Taxa-Gewichte
    exact_gamma_moments: bool = True  # Gamma-Parameter des P0-Priors analytisch statt per Gitter
    dispersed_init: bool = True  # zufaellige, gestreute Startwerte je Kette (fuer R-hat)
    # Vorschlag fuer die Taxa-Gewichte: "dirichlet" (Random-Walk Dir(kappa w) wie im Original) oder
    # "log_gamma" (Random-Walk auf log y mit w = y / sum(y), y ~ Gamma(0,5); gleiche Zielverteilung,
    # Schrittweiten werden im Burn-in angepasst). Nur fuer kernel = "corrected".
    weights_proposal: str = "log_gamma"
    target_accept: float = 0.234
    # Prior der Taxa-Gewichte Dir(alpha, ..., alpha):
    #   "fixed": alpha = taxa_prior_alpha (Paper: 0,5 = Jeffreys)
    #   "hierarchical": alpha wird mitgeschaetzt, log(alpha) ~ N(log(alpha_prior_median), alpha_prior_log_sd^2)
    #                   (nur mit weights_proposal = "log_gamma")
    taxa_prior: str = "hierarchical"
    taxa_prior_alpha: float = 0.5
    alpha_prior_median: float = 1.0
    alpha_prior_log_sd: float = 1.0
    # Modell- und Abstimmungsparameter
    # Streuungsparameter des Dirichlet-Vorschlags: kappa = dirichlet_spread_factor * num_taxa**2
    # (Paper: Faktor 1). Mit korrekter Hastings-Korrektur aendert er nur die Effizienz, nicht die
    # Zielverteilung; Faktor 4 ergibt hier ~14 % Akzeptanz und die beste Durchmischung.
    dirichlet_spread_factor: float = 4.0
    tf_proposal_scale: float | None = None  # Faktor auf die TF-Standardabw.; None: 2.38**2/num_taxa wie im Original
    # Trunkierung der TF-Stichproben: Quantile der Prior-Randverteilungen je Biom (Paper: 5 % und 95 %)
    tf_bound_probs: tuple[float, float] = (0.05, 0.95)
    beta_shape1: float = 3.0
    beta_shape2: float = 3.0
    # Gl. (14) des Papers als Prior der TF-Klimawerte: N_t(mu(psi), sigma(psi)) auf [a, b] (Standard: aus,
    # d. h. N_t nur als Vorschlag und gleichverteilter Prior auf [a, b] wie im Originalcode)
    tf_normal_prior: bool = False
    # Prior des rezenten Klimas (juengste Altersstufe): N(T) und Gamma(P) mit Mittelwert und Sd.
    # Voreinstellung: CRU_PRIOR_50KM. None: Werte aus den Eingangsdaten wie im Paper (Mittel und raeumliche Sd
    # mehrerer Levante-Gitterpunkte, 10,74 +- 2,48 degC und 556 +- 135 mm)
    recent_temp_mean: float | None = CRU_PRIOR_50KM["recent_temp_mean"]
    recent_temp_sd: float | None = CRU_PRIOR_50KM["recent_temp_sd"]
    recent_pann_mean: float | None = CRU_PRIOR_50KM["recent_pann_mean"]
    recent_pann_sd: float | None = CRU_PRIOR_50KM["recent_pann_sd"]
    # Proxy-Pool-Modul (R2 der P-Rekonstruktion mit der Baumpollenkurve AP):
    #   ap_exclude_taxa: Gehoelz-Taxa, die aus AP herausgerechnet werden (z. B. Kulturbaeume, data.CULTIVATED_TAXA)
    #   calibration_age_range: nur Altersstufen in [von, bis] cal BP gehen in R2 ein (None: alle)
    ap_exclude_taxa: tuple[str, ...] = ()
    calibration_age_range: tuple[float, float] | None = None
    # Terme der Zielverteilung (nur fuer Validierungslaeufe abschalten)
    use_calibration: bool = True
    use_tf_likelihood: bool = True
    use_taxa_prior: bool = True


@dataclass
class PostprocessConfig:
    # "samples": Quartile direkt aus den Stichproben; "kde": ueber geglaettete Dichten wie im Original
    quantile_method: str = "samples"
    # Klimarekonstruktion je Stichprobe:
    #   "mean": Mittelwert der Biom-Klimawerte, gewichtet mit den Biomanteilen (wie im Originalcode)
    #   "mixture": Mischverteilung nach Gl. (3) des Papers: je Stichprobe und Alter ein Biom b mit
    #              Wahrscheinlichkeit p_b, Klima = Klimawert dieses Bioms
    reconstruction: str = "mean"
    # Referenz fuer "Prior"-Groessen (CI-Verhaeltnis der Transferfunktionen):
    #   "model": eigener Lauf ohne Kalibrierungsterme, d. h. exakt der Prior des Samplers inkl. Trunkierung
    #   "marginal": Randverteilungen auf dem Klimagitter wie im Original
    prior_reference: str = "model"
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
        sd = dict(d.pop("sampler", {}))
        for k in ("tf_bound_probs", "ap_exclude_taxa", "calibration_age_range"):
            if sd.get(k) is not None:
                sd[k] = tuple(sd[k])
        sampler = SamplerConfig(**sd)
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
        if s.weights_proposal not in ("dirichlet", "log_gamma"):
            raise ValueError("sampler.weights_proposal muss 'dirichlet' oder 'log_gamma' sein")
        if s.weights_proposal == "log_gamma" and (s.kernel == "legacy" or not s.dirichlet_hastings_fix):
            raise ValueError("weights_proposal = 'log_gamma' nur mit kernel = 'corrected' und dirichlet_hastings_fix")
        if s.taxa_prior not in ("fixed", "hierarchical"):
            raise ValueError("sampler.taxa_prior muss 'fixed' oder 'hierarchical' sein")
        if s.taxa_prior == "hierarchical" and s.weights_proposal != "log_gamma":
            raise ValueError("taxa_prior = 'hierarchical' nur mit weights_proposal = 'log_gamma'")
        if s.taxa_prior_alpha <= 0 or s.alpha_prior_median <= 0 or s.alpha_prior_log_sd <= 0:
            raise ValueError("taxa_prior_alpha, alpha_prior_median und alpha_prior_log_sd muessen > 0 sein")
        if s.kernel == "legacy" and s.taxa_prior_alpha != 0.5:
            raise ValueError("der Original-Sampler hat den festen Prior Dir(0,5)")
        lo, hi = s.tf_bound_probs
        if not 0.0 <= lo < 0.5 < hi <= 1.0:
            raise ValueError("sampler.tf_bound_probs muss (unteres, oberes) Quantil mit 0 <= u < 0,5 < o <= 1 sein")
        if self.postprocess.prior_reference not in ("model", "marginal"):
            raise ValueError("postprocess.prior_reference muss 'model' oder 'marginal' sein")
        for k in ("recent_temp_sd", "recent_pann_sd", "recent_pann_mean"):
            v = getattr(s, k)
            if v is not None and v <= 0:
                raise ValueError(f"sampler.{k} muss > 0 sein")
        if s.kernel == "legacy" and (s.ap_exclude_taxa or s.calibration_age_range is not None):
            raise ValueError("ap_exclude_taxa und calibration_age_range gibt es nur fuer kernel = 'corrected'")
        if s.calibration_age_range is not None and (len(s.calibration_age_range) != 2
                                                     or s.calibration_age_range[0] >= s.calibration_age_range[1]):
            raise ValueError("sampler.calibration_age_range muss [von, bis] mit von < bis sein")
        if s.kernel == "legacy" and s.tf_normal_prior:
            raise ValueError("tf_normal_prior gibt es nur fuer kernel = 'corrected'")
        if self.postprocess.reconstruction not in ("mean", "mixture"):
            raise ValueError("postprocess.reconstruction muss 'mean' oder 'mixture' sein")
        if self.postprocess.quantile_method not in ("samples", "kde"):
            raise ValueError("postprocess.quantile_method muss 'samples' oder 'kde' sein")


def legacy_config(**overrides: Any) -> Config:
    """Einstellungen, die exakt dem Originalskript (main.py, Tag v0.1.0) entsprechen."""
    cfg = Config(
        sampler=SamplerConfig(
            kernel="legacy",
            n_chains=1,
            dirichlet_hastings_fix=False,
            exact_gamma_moments=False,
            dispersed_init=False,
            weights_proposal="dirichlet",
            dirichlet_spread_factor=1.0,
            taxa_prior="fixed",
            recent_temp_mean=None,
            recent_temp_sd=None,
            recent_pann_mean=None,
            recent_pann_sd=None,
            sample_length=1_000_000,
            burn_in=250_000,
            thin=75,
        ),
        postprocess=PostprocessConfig(quantile_method="kde", prior_reference="marginal"),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    cfg.validate()
    return cfg
