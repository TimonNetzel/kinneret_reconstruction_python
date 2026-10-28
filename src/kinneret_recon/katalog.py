"""Lauf-Katalog, Faktor-Register und Standort-Register (Grundlage der App, docs/app_konzept.md).

Ein Lauf des Katalogs ist ein Ordner ``reference/laeufe/<id>/`` mit einer Beschreibung ``lauf.toml``:
Titel und Beschreibung (de/en), Standort, Rolle, Konfiguration, Basislauf sowie Faktor und Stufe, die der Lauf
gegenueber seinem Basislauf aendert. Die Faktoren und ihre Stufen stehen im Register ``FAKTOREN``. Ein neuer
Sensitivitaetsfaktor ist ein neuer Eintrag dort plus Laeufe; ein neuer Lauf ist eine Konfiguration in ``configs/``
plus ein Ordner mit ``lauf.toml``.
"""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import CRU_PRIOR_50KM, Config

# Texte (Titel, Beschreibungen, Bezeichnungen) duerfen Formeln in LaTeX-Schreibweise enthalten, z. B.
# "$T_\\mathrm{DJF}$"; die App setzt sie um (formel.py, app/kern.js).

KATALOG_DIR = Path("reference/laeufe")
SPRACHEN = ("de", "en")

# Felder, die nur Laufzeit und Genauigkeit der Ketten bestimmen, nicht das Modell
LAUFZEIT_FELDER = ("sampler.sample_length", "sampler.burn_in", "sampler.thin", "sampler.seed")
# Felder ohne Einfluss auf die Ergebnisse
NEUTRALE_FELDER = ("output_dir", "plots", "report", "postprocess.n_threads")
# Prior des rezenten Klimas: None bedeutet "aus den Eingangsdaten" (inputs.recent_climate in dieser Reihenfolge)
RECENT_FELDER = ("sampler.recent_temp_mean", "sampler.recent_temp_sd", "sampler.recent_pann_mean",
                 "sampler.recent_pann_sd")


def _t(de: str, en: str) -> dict[str, str]:
    return {"de": de, "en": en}


# ---------------------------------------------------------------------------------------------------------------------
# Standort-Register (vorerst nur Kinneret; Ausbau in Schritt 5 des App-Konzepts)

@dataclass(frozen=True)
class Standort:
    key: str
    name: dict[str, str]
    data_dir: str
    lat: float
    lon: float
    elevation_m: float
    # Raster fuer Dichten in den App-Daten: (von, bis, Schritt)
    temp_raster: tuple[float, float, float] = (-10.0, 30.0, 0.5)
    pann_raster: tuple[float, float, float] = (0.0, 1500.0, 20.0)


def _kinneret() -> Standort:
    from .cru import LAKE_ELEVATION_M, LAKE_LAT, LAKE_LON

    return Standort("kinneret", _t("See Genezareth", "Lake Kinneret"), "data/in", LAKE_LAT, LAKE_LON,
                    LAKE_ELEVATION_M)


STANDORTE: dict[str, Standort] = {"kinneret": _kinneret()}


# ---------------------------------------------------------------------------------------------------------------------
# Faktor-Register

GRUPPEN: dict[str, dict[str, str]] = {
    "referenz": _t("Referenzläufe", "Reference runs"),
    "korrekturen": _t("Sampler-Korrekturen", "Sampler corrections"),
    "gewichte": _t("Prior der Taxa-Gewichte", "Prior of the taxa weights"),
    "klima_prior": _t("Prior des rezenten Klimas", "Prior of the recent climate"),
    "r2": _t("$R^2$-Prior", "$R^2$ prior"),
    "baumpollen": _t("Baumpollenkurve", "Arboreal pollen curve"),
    "modellform": _t("Modellform", "Model form"),
}

ROLLEN: dict[str, dict[str, str]] = {
    "referenz": _t("Referenz", "Reference"),
    "variante": _t("Variante des Standards", "Variant of the standard"),
    "sensitivitaet": _t("Sensitivität", "Sensitivity"),
    "validierung": _t("Validierung", "Validation"),
    "experiment": _t("Experiment", "Experiment"),
}


@dataclass(frozen=True)
class Stufe:
    key: str
    label: dict[str, str]
    werte: dict[str, Any]  # Konfigurationsfeld ("abschnitt.feld") -> Wert


@dataclass(frozen=True)
class Faktor:
    key: str
    gruppe: str
    label: dict[str, str]
    beschreibung: dict[str, str]
    felder: tuple[str, ...]  # Konfigurationsfelder, die der Faktor aendern darf
    stufen: tuple[Stufe, ...]

    def stufe(self, key: str) -> Stufe:
        for s in self.stufen:
            if s.key == key:
                return s
        raise KeyError(f"Faktor {self.key!r} hat keine Stufe {key!r}")


_ORIGINAL = {"sampler.kernel": "legacy", "sampler.n_chains": 1, "sampler.dispersed_init": False,
             "sampler.dirichlet_hastings_fix": False, "sampler.weights_proposal": "dirichlet",
             "sampler.dirichlet_spread_factor": 1.0, "sampler.exact_gamma_moments": False,
             "postprocess.quantile_method": "kde", "postprocess.prior_reference": "marginal"}
_A = {**_ORIGINAL, "sampler.kernel": "corrected", "sampler.n_chains": 4, "sampler.dispersed_init": True,
      "postprocess.prior_reference": "model"}
_B = {**_A, "sampler.dirichlet_hastings_fix": True, "sampler.weights_proposal": "log_gamma",
      "sampler.dirichlet_spread_factor": 4.0}
_C = {**_B, "sampler.exact_gamma_moments": True}


def _recent(t: float | None, ts: float | None, p: float | None, ps: float | None) -> dict[str, Any]:
    return dict(zip(RECENT_FELDER, (t, ts, p, ps)))


_V2 = {"sampler.taxa_prior": "fixed", "sampler.taxa_prior_alpha": 0.5, **_recent(None, None, None, None)}
_V3 = {"sampler.taxa_prior": "hierarchical", "sampler.alpha_prior_median": 1.0, "sampler.alpha_prior_log_sd": 1.0,
       **dict(zip(RECENT_FELDER, CRU_PRIOR_50KM.values()))}

FAKTOREN: dict[str, Faktor] = {f.key: f for f in (
    Faktor("standardversion", "referenz", _t("Standard v2 / v3", "Standard v2 / v3"),
           _t("v3 (seit 27.10.2026): hierarchischer Prior der Taxa-Gewichte und Prior des rezenten Klimas aus "
              "CRU TS 4.07 für ein Einzugsgebiet von 50 km; v2: Dir(0,5) und Prior wie im Paper "
              "(Prüfbericht 3.10).",
              "v3 (since 27 Oct 2026): hierarchical prior of the taxa weights and prior of the recent climate from "
              "CRU TS 4.07 for a 50 km catchment; v2: Dir(0.5) and prior as in the paper (review report 3.10)."),
           ("sampler.taxa_prior", "sampler.taxa_prior_alpha", "sampler.alpha_prior_median",
            "sampler.alpha_prior_log_sd", *RECENT_FELDER),
           (Stufe("v2", _t("Standard v2", "Standard v2"), _V2), Stufe("v3", _t("Standard v3", "Standard v3"), _V3))),
    Faktor("korrekturen", "korrekturen", _t("Korrekturen des Samplers", "Sampler corrections"),
           _t("Schrittweise vom Original zum korrigierten Sampler: A neuer Sampler mit Korrekturen aus, B mit "
              "korrekter Hastings-Korrektur ($\\kappa$), C mit exakter Momentenanpassung des Gamma-Priors für $P_0$, Standard "
              "mit Quartilen aus den Stichproben (Prüfbericht 3.1, Abschnitt 4).",
              "Step by step from the original to the corrected sampler: A new sampler with corrections off, B with "
              "the correct Hastings correction ($\\kappa$), C with exact moment matching of the gamma prior for $P_0$, "
              "standard with quartiles from the samples (review report 3.1, section 4)."),
           tuple(_ORIGINAL),
           (Stufe("original", _t("Original", "Original"), _ORIGINAL),
            Stufe("a", _t("A: neuer Sampler, Korrekturen aus", "A: new sampler, corrections off"), _A),
            Stufe("b", _t("B: + Hastings-Korrektur", "B: + Hastings correction"), _B),
            Stufe("b_dirichlet", _t("B′: wie B, Dirichlet-Vorschlag", "B′: as B, Dirichlet proposal"),
                  {**_B, "sampler.weights_proposal": "dirichlet"}),
            Stufe("c", _t("C: + exakter Gamma-Prior", "C: + exact gamma prior"), _C),
            Stufe("standard", _t("Standard: + Quartile aus Stichproben", "Standard: + quartiles from samples"),
                  {**_C, "postprocess.quantile_method": "samples"}))),
    Faktor("gewichte_prior", "gewichte", _t("Prior der Taxa-Gewichte", "Prior of the taxa weights"),
           _t("Dirichlet-Prior Dir($\\alpha, \\ldots, \\alpha$) der 33 Taxa-Gewichte mit festem $\\alpha$ oder mitgeschätztem "
              "$\\alpha$ (hierarchisch, $\\log\\alpha \\sim N(\\log 1, 1)$). Die Daten bestimmen die Gewichte nur schwach (Prüfbericht 3.6).",
              "Dirichlet prior Dir($\\alpha, \\ldots, \\alpha$) of the 33 taxa weights with fixed $\\alpha$ or with $\\alpha$ "
              "estimated jointly (hierarchical, $\\log\\alpha \\sim N(\\log 1, 1)$). The data constrain the weights only weakly (review report 3.6)."),
           ("sampler.taxa_prior", "sampler.taxa_prior_alpha", "sampler.alpha_prior_median",
            "sampler.alpha_prior_log_sd"),
           (Stufe("dir_05", _t("Dir(0,5)", "Dir(0.5)"), {"sampler.taxa_prior": "fixed", "sampler.taxa_prior_alpha": 0.5}),
            Stufe("dir_1", _t("Dir(1)", "Dir(1)"), {"sampler.taxa_prior": "fixed", "sampler.taxa_prior_alpha": 1.0}),
            Stufe("dir_2", _t("Dir(2)", "Dir(2)"), {"sampler.taxa_prior": "fixed", "sampler.taxa_prior_alpha": 2.0}),
            Stufe("hierarchisch", _t("Dir($\\alpha$), $\\alpha$ hierarchisch", "Dir($\\alpha$), hierarchical $\\alpha$"),
                  {"sampler.taxa_prior": "hierarchical", "sampler.alpha_prior_median": 1.0,
                   "sampler.alpha_prior_log_sd": 1.0}))),
    Faktor("klima_prior", "klima_prior", _t("Prior des rezenten Klimas", "Prior of the recent climate"),
           _t("Mittel und Sd von $T_\\mathrm{DJF}$ und $P_\\mathrm{ANN}$ der jüngsten Altersstufe. Der Prior legt das absolute Niveau der "
              "Rekonstruktion fest; der zeitliche Verlauf von P ist robust (Prüfbericht 3.8, 3.9).",
              "Mean and sd of $T_\\mathrm{DJF}$ and $P_\\mathrm{ANN}$ of the youngest age step. The prior sets the absolute level of the "
              "reconstruction; the temporal course of P is robust (review report 3.8, 3.9)."),
           RECENT_FELDER,
           (Stufe("paper", _t("Paper: Levante-Gitterpunkte (10,74 ± 2,48 °C, 556 ± 135 mm)",
                              "Paper: Levant grid points (10.74 ± 2.48 °C, 556 ± 135 mm)"), _recent(None, None, None, None)),
            Stufe("gitterpunkt", _t("CRU-Gitterpunkt am See (11,37 °C, 440 mm)", "CRU grid point at the lake (11.37 °C, 440 mm)"),
                  _recent(11.37, 2.48, 440.0, 135.0)),
            Stufe("eng", _t("Gebietsmittel, enge Sd (1 °C, 20 %)", "Area mean, narrow sd (1 °C, 20 %)"),
                  _recent(10.74, 1.0, 556.0, 111.0)),
            Stufe("gitterpunkt_eng", _t("Gitterpunkt, enge Sd", "Grid point, narrow sd"), _recent(11.37, 1.0, 440.0, 88.0)),
            Stufe("cru_25", _t("CRU 25 km", "CRU 25 km"), _recent(11.50, 1.23, 480.0, 44.9)),
            Stufe("cru_50", _t("CRU 50 km", "CRU 50 km"), dict(zip(RECENT_FELDER, CRU_PRIOR_50KM.values()))),
            Stufe("cru_100", _t("CRU 100 km", "CRU 100 km"), _recent(10.45, 0.99, 512.0, 39.1)))),
    Faktor("r2_prior", "r2", _t("$R^2$-Prior", "$R^2$ prior"),
           _t("Beta-Prior von $R^2$ der Niederschlagsrekonstruktion mit der Baumpollenkurve (Proxy-Pool-Modul); "
              "Mittel 0,25 / 0,5 / 0,75 wie in Tab. 2 des Papers (Prüfbericht 3.7, 3.8).",
              "Beta prior of $R^2$ of the precipitation reconstruction with the arboreal pollen curve (proxy pool "
              "module); mean 0.25 / 0.5 / 0.75 as in Table 2 of the paper (review report 3.7, 3.8)."),
           ("sampler.beta_shape1", "sampler.beta_shape2"),
           (Stufe("r2_025", _t("Beta(0,92; 2,77), Mittel 0,25", "Beta(0.92, 2.77), mean 0.25"),
                  {"sampler.beta_shape1": 0.92, "sampler.beta_shape2": 2.77}),
            Stufe("r2_05", _t("Beta(3; 3), Mittel 0,5", "Beta(3, 3), mean 0.5"),
                  {"sampler.beta_shape1": 3.0, "sampler.beta_shape2": 3.0}),
            Stufe("r2_05_262", _t("Beta(2,62; 2,62), Mittel 0,5", "Beta(2.62, 2.62), mean 0.5"),
                  {"sampler.beta_shape1": 2.62, "sampler.beta_shape2": 2.62}),
            Stufe("r2_075", _t("Beta(2,77; 0,92), Mittel 0,75", "Beta(2.77, 0.92), mean 0.75"),
                  {"sampler.beta_shape1": 2.77, "sampler.beta_shape2": 0.92}))),
    Faktor("ap_taxa", "baumpollen", _t("Taxa der Baumpollenkurve", "Taxa of the arboreal pollen curve"),
           _t("Baumpollenkurve für $R^2$ mit allen Gehölz-Taxa oder ohne Kulturbäume (Olea, Vitis, Juglans). "
              "Ohne Olea sind die feuchten Phasen schwächer, das P-Muster hält (Prüfbericht 3.11).",
              "Arboreal pollen curve for $R^2$ with all woody taxa or without cultivated trees (Olea, Vitis, Juglans). "
              "Without Olea the wet phases are weaker, the P pattern holds (review report 3.11)."),
           ("sampler.ap_exclude_taxa",),
           (Stufe("alle", _t("alle Gehölz-Taxa", "all woody taxa"), {"sampler.ap_exclude_taxa": ()}),
            Stufe("ohne_kultur", _t("ohne Kulturbäume", "without cultivated trees"),
                  {"sampler.ap_exclude_taxa": ("Olea europaea", "Vitis", "Juglans regia")}))),
    Faktor("kalibrierung_alter", "baumpollen", _t("Altersbereich der Kalibrierung", "Age range of the calibration"),
           _t("Altersstufen, die in $R^2$ eingehen: alle oder nur 4000–9000 cal BP, vor dem starken menschlichen "
              "Einfluss auf die Vegetation (Prüfbericht 3.11).",
              "Age steps entering $R^2$: all or only 4000–9000 cal BP, before the strong human impact on the vegetation "
              "(review report 3.11)."),
           ("sampler.calibration_age_range",),
           (Stufe("alle", _t("alle Alter", "all ages"), {"sampler.calibration_age_range": None}),
            Stufe("4_9ka", _t("4000–9000 cal BP", "4000–9000 cal BP"), {"sampler.calibration_age_range": (4000.0, 9000.0)}))),
    Faktor("rekonstruktion", "modellform", _t("Mittelwert oder Mischverteilung", "Mean or mixture"),
           _t("Klima je Stichprobe als mit den Biomanteilen gewichteter Mittelwert (Standard) oder als "
              "Mischverteilung nach Gl. (3) des Papers. Die Mischverteilung hat beim Niederschlag mehrere Gipfel und "
              "wird als Dichte gezeigt (Prüfbericht 3.8, 3.10).",
              "Climate per sample as the mean weighted with the biome ratios (standard) or as the mixture of "
              "Eq. (3) of the paper. The mixture is multimodal for precipitation and is shown as a density "
              "(review report 3.8, 3.10)."),
           ("postprocess.reconstruction",),
           (Stufe("mean", _t("Mittelwert", "Mean"), {"postprocess.reconstruction": "mean"}),
            Stufe("mixture", _t("Mischverteilung (Gl. 3)", "Mixture (Eq. 3)"), {"postprocess.reconstruction": "mixture"}))),
    Faktor("tf_prior", "modellform", _t("Prior der Klimawerte der Biome", "Prior of the biome climate values"),
           _t("Gl. (14) des Papers nur als Vorschlag (Standard, gleichverteilter Prior auf dem Trunkierungsbereich) "
              "oder zusätzlich als Prior $N_\\mathrm{t}(\\mu, \\sigma)$; als Prior zählt er die Transferfunktion doppelt (Prüfbericht 3.8).",
              "Eq. (14) of the paper as proposal only (standard, uniform prior on the truncation range) or also as "
              "prior $N_\\mathrm{t}(\\mu, \\sigma)$; as a prior it counts the transfer function twice (review report 3.8)."),
           ("sampler.tf_normal_prior",),
           (Stufe("vorschlag", _t("nur Vorschlag", "proposal only"), {"sampler.tf_normal_prior": False}),
            Stufe("prior", _t("$N_\\mathrm{t}$ als Prior", "$N_\\mathrm{t}$ as prior"), {"sampler.tf_normal_prior": True}))),
    Faktor("tf_grenzen", "modellform", _t("Trunkierung der Transferfunktionen", "Truncation of the transfer functions"),
           _t("Quantile der Prior-Randverteilungen je Biom, auf die die Klimawerte trunkiert werden "
              "(Paper: 5 % und 95 %; Prüfbericht 3.3).",
              "Quantiles of the prior marginals per biome to which the climate values are truncated "
              "(paper: 5 % and 95 %; review report 3.3)."),
           ("sampler.tf_bound_probs",),
           (Stufe("5_95", _t("5 % / 95 %", "5 % / 95 %"), {"sampler.tf_bound_probs": (0.05, 0.95)}),
            Stufe("1_99", _t("1 % / 99 %", "1 % / 99 %"), {"sampler.tf_bound_probs": (0.01, 0.99)}))),
)}


# ---------------------------------------------------------------------------------------------------------------------
# Einstellungen und Fingerabdruck

def flache_einstellungen(cfg: Config, recent_climate=None) -> dict[str, Any]:
    """Konfiguration als flaches Woerterbuch {"abschnitt.feld": Wert} ohne neutrale Felder. Mit recent_climate
    (inputs.recent_climate) werden None-Werte des Priors des rezenten Klimas durch die Werte der Eingangsdaten
    ersetzt, die der Sampler dann verwendet."""
    out = {}
    for k, v in cfg.to_dict().items():
        if isinstance(v, dict):
            out.update({f"{k}.{kk}": vv for kk, vv in v.items()})
        else:
            out[k] = v
    for k in NEUTRALE_FELDER:
        out.pop(k, None)
    if recent_climate is not None:
        for k, rc in zip(RECENT_FELDER, recent_climate):
            if out[k] is None:
                out[k] = float(rc)
    return {k: list(v) if isinstance(v, tuple) else v for k, v in out.items()}


def _gleich(a, b) -> bool:
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_gleich(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return (isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool)
                and not isinstance(b, bool) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12))
    return a == b


def unterschiede(a: dict[str, Any], b: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    return {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if not _gleich(a.get(k), b.get(k))}


def daten_hash(data_dir: str | Path) -> str:
    """Pruefsumme der Eingangsdaten (die von data.load_inputs gelesenen Dateien)."""
    h = hashlib.sha256()
    for name in ("age_depth_relationship.npz", "core_data.npz", "nnet_fit_params.npz"):
        h.update((Path(data_dir) / name).read_bytes())
    return h.hexdigest()[:12]


def fingerabdruck(einst: dict[str, Any], daten: str, nur_sampler: bool = False) -> str:
    """Fingerabdruck einer Konfiguration (flache Einstellungen mit aufgeloestem rezentem Prior) und der Daten.
    nur_sampler: nur die Felder, die die Ketten bestimmen (gleiche Ketten koennen neu ausgewertet werden)."""
    d = {k: v for k, v in einst.items() if not nur_sampler or k.startswith("sampler.")}
    d.pop("data_dir", None)
    s = json.dumps({"daten": daten, **d}, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def passt_zu_stufe(einst: dict[str, Any], stufe: Stufe, recent_climate=None) -> bool:
    werte = dict(stufe.werte)
    if recent_climate is not None:
        for k, rc in zip(RECENT_FELDER, recent_climate):
            if k in werte and werte[k] is None:
                werte[k] = float(rc)
    return all(_gleich(einst.get(k), list(v) if isinstance(v, tuple) else v) for k, v in werte.items())


def stufe_von(einst: dict[str, Any], faktor: Faktor, recent_climate=None) -> str | None:
    for s in faktor.stufen:
        if passt_zu_stufe(einst, s, recent_climate):
            return s.key
    return None


# ---------------------------------------------------------------------------------------------------------------------
# Katalog

@dataclass
class Lauf:
    id: str
    ordner: Path
    titel: dict[str, str]
    beschreibung: dict[str, str]
    config: str  # Pfad der Konfiguration relativ zum Hauptverzeichnis
    standort: str = "kinneret"
    rolle: str = "sensitivitaet"
    basis: str | None = None
    faktor: str | None = None
    stufe: str | None = None
    quelle: str | None = None  # Abschnitt im Pruefbericht o. Ae.
    hinweis: dict[str, str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "Lauf":
        with open(path, "rb") as f:
            d = tomllib.load(f)
        known = {"titel", "beschreibung", "config", "standort", "rolle", "basis", "faktor", "stufe", "quelle", "hinweis"}
        return cls(id=path.parent.name, ordner=path.parent, **{k: v for k, v in d.items() if k in known},
                   extra={k: v for k, v in d.items() if k not in known})

    def lade_config(self, root: Path) -> Config:
        return Config.from_toml(root / self.config)

    def as_dict(self) -> dict[str, Any]:
        return dict(id=self.id, titel=self.titel, beschreibung=self.beschreibung, config=self.config,
                    standort=self.standort, rolle=self.rolle, basis=self.basis, faktor=self.faktor, stufe=self.stufe,
                    quelle=self.quelle, hinweis=self.hinweis)


def lade_katalog(root: Path | None = None) -> dict[str, Lauf]:
    root = Path(root or Path.cwd())
    return {p.parent.name: Lauf.load(p) for p in sorted((root / KATALOG_DIR).glob("*/lauf.toml"))}


def reihenfolge(katalog: dict[str, Lauf], ids=None) -> list[str]:
    """Laeufe so geordnet, dass jeder Basislauf vor seinen Varianten steht."""
    ids = list(katalog) if ids is None else list(ids)
    out: list[str] = []

    def add(i, stack=()):
        if i in out or i not in ids:
            return
        if i in stack:
            raise ValueError(f"Zyklus in den Basislaeufen: {' -> '.join(stack + (i,))}")
        b = katalog[i].basis
        if b:
            add(b, stack + (i,))
        out.append(i)

    for i in ids:
        add(i)
    return out


@dataclass
class Pruefung:
    lauf: str
    fehler: list[str] = field(default_factory=list)
    einstellungen: dict[str, Any] = field(default_factory=dict)
    fingerabdruck: str = ""
    fingerabdruck_ketten: str = ""
    recent_climate: list[float] | None = None  # rezentes Klima der Eingangsdaten des Standorts


def pruefe_katalog(katalog: dict[str, Lauf], root: Path | None = None) -> dict[str, Pruefung]:
    """Prueft jeden Lauf: Pflichtangaben, Konfiguration, Faktor und Stufe, Abweichung vom Basislauf nur in den
    Feldern des Faktors (und Laufzeit-Feldern). Liefert auch die Fingerabdruecke."""
    from .data import load_inputs

    root = Path(root or Path.cwd())
    inputs, hashes = {}, {}
    out: dict[str, Pruefung] = {}
    for lid, lauf in katalog.items():
        p = out[lid] = Pruefung(lid)
        for k in SPRACHEN:
            if not lauf.titel.get(k) or not lauf.beschreibung.get(k):
                p.fehler.append(f"Titel oder Beschreibung fehlt ({k})")
        if lauf.rolle not in ROLLEN:
            p.fehler.append(f"unbekannte Rolle {lauf.rolle!r}")
        st = STANDORTE.get(lauf.standort)
        if st is None:
            p.fehler.append(f"unbekannter Standort {lauf.standort!r}")
            continue
        try:
            cfg = lauf.lade_config(root)
        except (OSError, ValueError, TypeError) as e:
            p.fehler.append(f"Konfiguration {lauf.config}: {e}")
            continue
        if Path(cfg.data_dir) != Path(st.data_dir):
            p.fehler.append(f"data_dir {cfg.data_dir!r} passt nicht zum Standort ({st.data_dir!r})")
        if st.key not in inputs:
            inputs[st.key] = load_inputs(root / st.data_dir).recent_climate
            hashes[st.key] = daten_hash(root / st.data_dir)
        p.recent_climate = [float(v) for v in inputs[st.key]]
        p.einstellungen = flache_einstellungen(cfg, inputs[st.key])
        p.fingerabdruck = fingerabdruck(p.einstellungen, hashes[st.key])
        p.fingerabdruck_ketten = fingerabdruck(p.einstellungen, hashes[st.key], nur_sampler=True)
    for lid, lauf in katalog.items():
        p = out[lid]
        if not p.einstellungen:
            continue
        rc = inputs[lauf.standort]
        if (lauf.faktor is None) != (lauf.stufe is None):
            p.fehler.append("faktor und stufe nur gemeinsam angeben")
        f = FAKTOREN.get(lauf.faktor) if lauf.faktor else None
        if lauf.faktor and f is None:
            p.fehler.append(f"unbekannter Faktor {lauf.faktor!r}")
        if f is not None:
            try:
                if not passt_zu_stufe(p.einstellungen, f.stufe(lauf.stufe), rc):
                    p.fehler.append(f"Konfiguration passt nicht zur Stufe {lauf.stufe!r} von {f.key!r}")
            except KeyError as e:
                p.fehler.append(str(e.args[0]))
        if lauf.basis is None:
            continue
        if lauf.basis not in katalog:
            p.fehler.append(f"Basislauf {lauf.basis!r} fehlt im Katalog")
            continue
        if katalog[lauf.basis].standort != lauf.standort:
            p.fehler.append("Basislauf hat einen anderen Standort")
        if not out[lauf.basis].einstellungen:
            continue
        erlaubt = set(LAUFZEIT_FELDER) | (set(f.felder) if f is not None else set())
        fremd = {k: v for k, v in unterschiede(out[lauf.basis].einstellungen, p.einstellungen).items()
                 if k not in erlaubt}
        if fremd:
            p.fehler.append("weicht vom Basislauf ausserhalb des Faktors ab: " +
                            ", ".join(f"{k} {a!r} -> {b!r}" for k, (a, b) in fremd.items()))
    try:
        reihenfolge(katalog)
    except ValueError as e:
        for p in out.values():
            p.fehler.append(str(e))
    return out
