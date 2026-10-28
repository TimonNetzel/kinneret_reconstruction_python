"""Kompakte Zusammenfassungen der Laeufe fuer die App (``app.json`` je Lauf, Standortdaten, Katalog-Index).

Die vollstaendigen Ergebnisse eines Laufs (Ketten, ~60 MB) bleiben ausserhalb des Repos in ``results/<lauf>/``.
Die Zusammenfassung enthaelt alles, was die Ansichten der App brauchen (docs/app_konzept.md, Abschnitt 2):
Quantile von T und P je Alter und Tiefe (Mittelwert und Mischverteilung), Dichten der Mischverteilung auf einem
groben Raster, Biomanteile, Baumpollenkurve, Taxa-Gewichte, R2/T0/P0, Transferfunktionen und Diagnostik.

Format: JSON (direkt im Browser lesbar), Ziel < 1 MB je Lauf. Dichten sind je Zeile auf das Maximum 100 normiert
und als ganze Zahlen gespeichert.
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.stats import beta, gamma, gaussian_kde, norm

from . import __version__
from .config import Config
from .data import ARBOREAL_TAXA, CULTIVATED_TAXA, Inputs, load_inputs
from .katalog import (FAKTOREN, GRUPPEN, ROLLEN, STANDORTE, Lauf, Pruefung, Standort, daten_hash,
                      flache_einstellungen, fingerabdruck, stufe_von)
from .model import Setup, ap_without, build_setup
from .postprocess import Results, biome_ratios, densities, reconstruct, reconstruct_mixture
from .sampler import Posterior

FORMAT = 1
APP_DIR = Path("reference/app")
PROBS = (0.025, 0.25, 0.5, 0.75, 0.975)
PROB_KEYS = ("q2_5", "q25", "q50", "q75", "q97_5")
N_SPUR = 300  # Punkte je Kette in den Spuren
REFERENZ_DATEIEN = ("config.toml", "meta.json", "diagnostik.csv", "rekonstruktion_alter.csv", "rekonstruktion_tiefe.csv")


def _r(a, nd: int):
    """Gerundete (verschachtelte) Liste; NaN und inf werden zu None."""
    a = np.asarray(a, dtype=float)
    out = np.round(a, nd).tolist()
    if not np.all(np.isfinite(a)):
        def fix(x):
            return [fix(v) for v in x] if isinstance(x, list) else (x if np.isfinite(x) else None)
        out = fix(out)
    return out


def _sig(x: float, n: int = 4):
    """Auf n signifikante Stellen gerundet (fuer Kennzahlen unterschiedlicher Groessenordnung)."""
    if x is None or not np.isfinite(x):
        return None
    return float(f"{x:.{n}g}")


def _dichte_int(d: np.ndarray) -> list:
    """Dichten (n, dims) je Zeile auf das Maximum 100 normiert, als ganze Zahlen."""
    m = d.max(axis=1, keepdims=True)
    m[m <= 0] = 1.0
    return np.rint(100 * d / m).astype(int).tolist()


def raster(spec: tuple[float, float, float]) -> np.ndarray:
    lo, hi, step = spec
    return np.round(np.linspace(lo, hi, int(round((hi - lo) / step)) + 1), 6)


def _quantile(x: np.ndarray, nd: int) -> dict[str, list]:
    """Quantile PROBS und Mittelwert je Spalte von x (S, n)."""
    q = np.quantile(x, PROBS, axis=0)
    out = {k: _r(q[i], nd) for i, k in enumerate(PROB_KEYS)}
    out["mittel"] = _r(x.mean(axis=0), nd)
    return out


def _kurve(samples: np.ndarray, xs: np.ndarray, prior_pdf) -> dict[str, list]:
    """Posterior (Kerndichte) und Prior auf xs, je auf das Maximum 1 normiert."""
    if np.ptp(samples) > 0:
        d = gaussian_kde(samples).evaluate(xs)
    else:  # entartete Stichprobe (haengengebliebene Kette)
        d = np.zeros_like(xs)
        d[np.abs(xs - samples[0]).argmin()] = 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(np.isfinite(prior_pdf(xs)), prior_pdf(xs), np.nan)
    return {"x": _r(xs, 4), "posterior": _r(d / d.max(), 3), "prior": _r(p / np.nanmax(p), 3)}


def _skalar(samples: np.ndarray, nd: int) -> dict[str, Any]:
    q = np.quantile(samples, PROBS)
    return {**{k: round(float(q[i]), nd) for i, k in enumerate(PROB_KEYS)}, "mittel": round(float(samples.mean()), nd)}


def gewichte_prior_quantile(s, num_taxa: int, n: int = 200_000, seed: int = 2023) -> list[float]:
    """Quantile PROBS der Randverteilung eines Taxa-Gewichts unter dem Prior Dir(alpha, ..., alpha):
    w_k ~ Beta(alpha, (K-1) alpha); hierarchisch mit log alpha ~ N(log median, sd^2) (Monte Carlo)."""
    if s.taxa_prior == "hierarchical":
        rng = np.random.default_rng(seed)
        a = np.exp(rng.normal(np.log(s.alpha_prior_median), s.alpha_prior_log_sd, n))
        w = rng.beta(a, (num_taxa - 1) * a)
        return [float(v) for v in np.quantile(w, PROBS)]
    a = s.taxa_prior_alpha
    return [float(v) for v in beta.ppf(PROBS, a, (num_taxa - 1) * a)]


def _spuren(post: Posterior, biome_labels) -> dict[str, Any]:
    B = len(biome_labels)
    reihen = {"log_target": post.log_target, "R2": post.expl_variance, "T0": post.recent_temp, "P0": post.recent_pann}
    if np.ptp(post.alpha) > 0:
        reihen["alpha"] = post.alpha
    for b, lab in enumerate(biome_labels):
        reihen[f"T_{lab}"] = post.tf_sample[:, :, b]
    for b, lab in enumerate(biome_labels):
        reihen[f"P_{lab}"] = post.tf_sample[:, :, B + b]
    idx = np.unique(np.linspace(0, len(post.iterations) - 1, min(N_SPUR, len(post.iterations))).astype(int))
    return {"iteration": post.iterations[idx].astype(int).tolist(), "burn_in": int(post.burn_in),
            "werte": {k: [_sig_list(v[c, idx]) for c in range(v.shape[0])]
                      for k, v in reihen.items() if np.all(np.isfinite(v))}}


def _sig_list(x: np.ndarray, n: int = 5) -> list[float]:
    return [float(f"{v:.{n}g}") for v in x]


def _lies_csv(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        out.append({k: (v if k == "parameter" else _sig(float(v), 5)) for k, v in r.items()})
    return out


@dataclass
class LaufDaten:
    """Alles, was aus results/<lauf>/ fuer die Zusammenfassung gelesen wird."""
    config: Config
    inputs: Inputs
    setup: Setup
    posterior: Posterior
    results: Results
    diagnostik: list[dict[str, Any]]
    meta: dict[str, Any]

    @classmethod
    def load(cls, outdir: Path, root: Path) -> "LaufDaten":
        cfg = Config.from_toml(outdir / "config.toml")
        inp = load_inputs(root / cfg.data_dir)
        return cls(cfg, inp, build_setup(inp, cfg.sampler), Posterior.load(outdir / "posterior.npz"),
                   Results.load(outdir / "results.npz"), _lies_csv(outdir / "diagnostik.csv"),
                   json.loads((outdir / "meta.json").read_text(encoding="utf-8")))


def _rekonstruktion(setup: Setup, post: Posterior, cfg: Config, st: Standort) -> dict[str, Any]:
    """Mittelwert und Mischverteilung je Alter und Tiefe aus den Ketten. Die Mischverteilung zieht ihre
    Zufallszahlen wie postprocess() (Seed postprocess.seed + 1, erst Alter, dann Tiefe)."""
    inp, B, nt = setup.inputs, setup.num_biomes, cfg.postprocess.n_threads
    w, tf = post.pooled("taxa_weights"), post.pooled("tf_sample")
    t_grid, p_grid = raster(st.temp_raster), raster(st.pann_raster)
    rng = np.random.default_rng(cfg.postprocess.seed + 1)
    out: dict[str, Any] = {"standard": cfg.postprocess.reconstruction}
    biome: dict[str, Any] = {}
    for name, achse, x, spectrum in (("alter", "age", inp.age, setup.taxa_spectrum_age),
                                     ("tiefe", "depth", inp.depth, setup.taxa_spectrum_depth)):
        ratios = biome_ratios(spectrum, w, inp.biomes_assign)
        tm, pm = reconstruct(ratios, tf, B)
        tx, px = reconstruct_mixture(ratios, tf, B, rng)
        out[name] = {
            "x": _r(x, 1),
            "mean": {"T": _quantile(tm, 3), "P": _quantile(pm, 1)},
            "mixture": {"T": _quantile(tx, 3), "P": _quantile(px, 1),
                        "dichte": {"T": _dichte_int(densities(tx, t_grid, nt)),
                                   "P": _dichte_int(densities(px, p_grid, nt))}},
        }
        q = np.quantile(ratios, PROBS, axis=0)  # (5, n, B)
        biome[name] = {k: _r(q[i].T, 4) for i, k in enumerate(PROB_KEYS)}
        del ratios, tm, pm, tx, px
    out["raster"] = {"T": _r(t_grid, 3), "P": _r(p_grid, 3)}
    return {"rekonstruktion": out, "biome": biome}


def zusammenfassung(lauf: Lauf, daten: LaufDaten, root: Path, pruefung: Pruefung | None = None) -> dict[str, Any]:
    """Kompakte Zusammenfassung eines gerechneten Laufs (Inhalt von app.json)."""
    from .plots import BIOME_LABELS
    from .report import methods_text

    cfg, setup, post, res, inp = daten.config, daten.setup, daten.posterior, daten.results, daten.inputs
    st = STANDORTE[lauf.standort]
    B, K = setup.num_biomes, setup.num_taxa
    labels = BIOME_LABELS[:B]
    dh = daten_hash(root / cfg.data_dir)
    einst = flache_einstellungen(cfg, inp.recent_climate)
    d: dict[str, Any] = {
        "format": FORMAT,
        "id": lauf.id,
        "lauf": lauf.as_dict(),
        "einstellungen": einst,
        "fingerabdruck": fingerabdruck(einst, dh),
        "fingerabdruck_ketten": fingerabdruck(einst, dh, nur_sampler=True),
        "daten_hash": dh,
        "meta": daten.meta,
        "methoden": {k: methods_text(SimpleNamespace(config=cfg, setup=setup, posterior=post), k) for k in ("de", "en")},
    }
    if pruefung is not None and pruefung.fingerabdruck and pruefung.fingerabdruck != d["fingerabdruck"]:
        raise ValueError(f"{lauf.id}: Ergebnisse passen nicht zur Konfiguration {lauf.config}")
    d.update(_rekonstruktion(setup, post, cfg, st))
    # Quartile, wie der Lauf sie berichtet (Methode postprocess.quantile_method), nur ueber das Alter
    d["rekonstruktion"]["quartile_lauf"] = {"methode": res.quantile_method, "rekonstruktion": res.reconstruction,
                                            "T": _r(res.temp_age_q, 3), "P": _r(res.pann_age_q, 1)}

    ap_tiefe = ap_without(inp, cfg.sampler.ap_exclude_taxa) if cfg.sampler.ap_exclude_taxa else inp.ap_depth
    d["ap"] = {"ausgeschlossen": list(cfg.sampler.ap_exclude_taxa), "tiefe": _r(ap_tiefe, 3),
               "alter": _r(setup.ap_age, 3),
               "kalibrierung_alter": None if setup.ap_mask is None else _r(setup.ap_mask, 0)}

    r2, t0, p0 = post.pooled("expl_variance"), post.pooled("recent_temp"), post.pooled("recent_pann")
    d["kalibrierung"] = {
        "R2": {**_skalar(r2, 4), "prior": {"verteilung": "beta", "a": setup.beta_shape1, "b": setup.beta_shape2},
               "kurve": _kurve(r2, np.linspace(0, 1, 101), lambda x: beta.pdf(x, setup.beta_shape1, setup.beta_shape2))},
        "T0": {**_skalar(t0, 3), "prior": {"verteilung": "normal", "mittel": setup.temp_mean, "sd": setup.temp_sd},
               "kurve": _kurve(t0, np.linspace(0, 20, 101), lambda x: norm.pdf(x, setup.temp_mean, setup.temp_sd))},
        "P0": {**_skalar(p0, 1), "prior": {"verteilung": "gamma", "form": setup.pann_shape, "rate": setup.pann_rate,
                                           "mittel": setup.pann_mean, "sd": setup.pann_sd},
               "kurve": _kurve(p0, np.linspace(0, 1000, 101),
                               lambda x: gamma.pdf(x, setup.pann_shape, scale=1 / setup.pann_rate))},
    }

    w = res.post_weights
    qw = np.quantile(w, PROBS, axis=0)
    a = post.pooled("alpha")
    d["gewichte"] = {
        **{k: _r(qw[i], 5) for i, k in enumerate(PROB_KEYS)},
        "mittel": _r(w.mean(axis=0), 5),
        "prior": dict(zip(PROB_KEYS, _r(gewichte_prior_quantile(cfg.sampler, K), 5))),
        "alpha": _skalar(a, 3) if cfg.sampler.taxa_prior == "hierarchical" else {"fest": cfg.sampler.taxa_prior_alpha},
    }

    t_grid, p_grid = raster(st.temp_raster), raster(st.pann_raster)
    from . import _core

    dens2d = []
    for b in range(B):
        z = _core.kde2d_grid(res.tf_post_pann[:, b].copy(), res.tf_post_temp[:, b].copy(), p_grid[0],
                             p_grid[1] - p_grid[0], len(p_grid), t_grid[0], t_grid[1] - t_grid[0], len(t_grid))
        z = np.where(np.isfinite(z), z, 0.0)
        dens2d.append(np.rint(100 * z / z.max()).astype(int).tolist() if z.max() > 0 else
                      np.zeros(z.shape, dtype=int).tolist())  # (P, T)
    qt, qp = np.quantile(res.tf_post_temp, PROBS, axis=0), np.quantile(res.tf_post_pann, PROBS, axis=0)
    pqt, pqp = np.quantile(res.tf_prior_temp, PROBS, axis=0), np.quantile(res.tf_prior_pann, PROBS, axis=0)
    d["transferfunktionen"] = {
        "biome": labels,
        "prior_referenz": cfg.postprocess.prior_reference,
        "T": {"posterior": {k: _r(qt[i], 3) for i, k in enumerate(PROB_KEYS)},
              "prior": {k: _r(pqt[i], 3) for i, k in enumerate(PROB_KEYS)},
              "grenzen": _r(np.stack([setup.tf_lower[:B], setup.tf_upper[:B]], axis=1), 3),
              "ci_posterior": _r(res.post_temp_ci_size, 3), "ci_prior": _r(res.prior_temp_ci_size, 3),
              "ci_verhaeltnis": _r(res.post_temp_ci_size / res.prior_temp_ci_size, 4)},
        "P": {"posterior": {k: _r(qp[i], 1) for i, k in enumerate(PROB_KEYS)},
              "prior": {k: _r(pqp[i], 1) for i, k in enumerate(PROB_KEYS)},
              "grenzen": _r(np.stack([setup.tf_lower[B:], setup.tf_upper[B:]], axis=1), 1),
              "ci_posterior": _r(res.post_pann_ci_size, 1), "ci_prior": _r(res.prior_pann_ci_size, 1),
              "ci_verhaeltnis": _r(res.post_pann_ci_size / res.prior_pann_ci_size, 4)},
        "dichte_posterior": dens2d,  # je Biom (P-Raster, T-Raster), Maximum 100
    }

    probleme = [r["parameter"] for r in daten.diagnostik
                if not ((r["rhat"] or np.inf) <= 1.01 and (r["ess_bulk"] or 0) >= 400)]
    d["diagnostik"] = {"akzeptanz": _r(post.acceptance_rate, 4), "n_stichproben": int(r2.shape[0]),
                       "n_ketten": int(post.n_chains), "probleme": probleme, "tabelle": daten.diagnostik,
                       "spuren": _spuren(post, labels)}
    d["kennzahlen"] = kennzahlen(d)
    return d


def kennzahlen(d: dict[str, Any]) -> dict[str, Any]:
    """Kennzahlen eines Laufs fuer Uebersichten (Mittel ueber alle Alter, wie Pruefbericht 3.8-3.11)."""
    rk = d["rekonstruktion"]["alter"][d["rekonstruktion"]["standard"]]
    t50, p50 = np.array(rk["T"]["q50"]), np.array(rk["P"]["q50"])
    iqr = lambda v: float(np.mean(np.array(rk[v]["q75"]) - np.array(rk[v]["q25"])))
    tab = d["diagnostik"]["tabelle"]
    ess = [r["ess_bulk"] for r in tab if r["ess_bulk"] is not None]
    rh = [r["rhat"] for r in tab if r["rhat"] is not None]
    k = d["kalibrierung"]
    tf = d["transferfunktionen"]
    return {
        "T_niveau": round(float(t50.mean()), 3), "P_niveau": round(float(p50.mean()), 1),
        "T_iqr": round(iqr("T"), 3), "P_iqr": round(iqr("P"), 1),
        "P_spanne": round(float(np.ptp(p50)), 1),
        "T_jung": round(float(t50[0]), 3), "P_jung": round(float(p50[0]), 1),
        "R2_median": k["R2"]["q50"], "T0_median": k["T0"]["q50"], "P0_median": k["P0"]["q50"],
        "ci_verhaeltnis_T": tf["T"]["ci_verhaeltnis"], "ci_verhaeltnis_P": tf["P"]["ci_verhaeltnis"],
        "ess_min": _sig(min(ess), 4) if ess else None, "rhat_max": _sig(max(rh), 4) if rh else None,
        "laufzeit_s": round(sum(d["meta"].get("timings_s", {}).values()), 1),
    }


def vergleich(d: dict[str, Any], basis: dict[str, Any]) -> dict[str, Any]:
    """Verschiebung des Niveaus, Aenderung des IQR und Korrelation der Anomalien (Median minus Mittel ueber alle
    Alter) gegenueber einem Basislauf, je mit der Standard-Rekonstruktion des Laufs (Pruefbericht 3.8)."""
    out = {}
    for v in ("T", "P"):
        a = d["rekonstruktion"]["alter"][d["rekonstruktion"]["standard"]][v]
        b = basis["rekonstruktion"]["alter"][basis["rekonstruktion"]["standard"]][v]
        ma, mb = np.array(a["q50"]), np.array(b["q50"])
        ia = np.mean(np.array(a["q75"]) - np.array(a["q25"]))
        ib = np.mean(np.array(b["q75"]) - np.array(b["q25"]))
        r = np.corrcoef(ma - ma.mean(), mb - mb.mean())[0, 1] if np.ptp(ma) > 0 and np.ptp(mb) > 0 else np.nan
        nd = 3 if v == "T" else 1
        out[v] = {"niveau_diff": round(float((ma - mb).mean()), nd), "max_abs_diff": round(float(np.abs(ma - mb).max()), nd),
                  "iqr_aenderung_rel": round(float(ia / ib - 1), 4), "r_anomalie": _sig(float(r), 4)}
    return out


def standort_daten(st: Standort, root: Path) -> dict[str, Any]:
    """Daten des Standorts, die fuer alle Laeufe gleich sind (Taxa, Spektren, Baumpollenkurve, Prior der TF)."""
    from .config import SamplerConfig
    from .plots import BIOME_COLS, BIOME_LABELS, BIOME_NAMES

    inp = load_inputs(root / st.data_dir)
    setup = build_setup(inp, SamplerConfig())
    B = inp.num_biomes
    t_grid, p_grid = raster(st.temp_raster), raster(st.pann_raster)
    # Prior der TF: Pr(Biom | T, P) auf dem groben Raster (lineare Interpolation auf dem nichtlinearen P-Gitter)
    from scipy.interpolate import RegularGridInterpolator

    pts = np.stack(np.meshgrid(p_grid, t_grid, indexing="ij"), axis=-1)
    prior2d = []
    for b in range(B):
        f = RegularGridInterpolator((inp.pann_range, inp.temp_range), setup.biome_given_climate[b],
                                    bounds_error=False, fill_value=0.0)
        z = f(pts)
        prior2d.append(np.rint(100 * z / z.max()).astype(int).tolist())
    names = [str(n) for n in inp.taxa_names]
    return {
        "format": FORMAT, "key": st.key, "name": st.name, "lat": st.lat, "lon": st.lon, "hoehe_m": st.elevation_m,
        "daten_hash": daten_hash(root / st.data_dir),
        "biome": {"labels": BIOME_LABELS[:B], "namen": BIOME_NAMES[:B], "farben": BIOME_COLS[:B]},
        "taxa": {"namen": names, "biom": np.argmax(inp.biomes_assign, axis=1).astype(int).tolist(),
                 "gehoelz": [n in ARBOREAL_TAXA for n in names], "kultur": [n in CULTIVATED_TAXA for n in names]},
        "alter": _r(inp.age, 1), "tiefe": _r(inp.depth, 1), "alter_mittel_tiefe": _r(inp.age_mean, 1),
        "spektrum_alter": _r(setup.taxa_spectrum_age, 5), "spektrum_tiefe": _r(setup.taxa_spectrum_depth, 5),
        "ap": {"tiefe": _r(inp.ap_depth, 3), "tiefe_ohne_kultur": _r(ap_without(inp, CULTIVATED_TAXA), 3),
               "alter": _r(setup.ap_age, 3),
               "alter_ohne_kultur": _r(build_setup(inp, SamplerConfig(ap_exclude_taxa=CULTIVATED_TAXA)).ap_age, 3)},
        "rezentes_klima_daten": _r(inp.recent_climate, 4),
        "raster": {"T": _r(t_grid, 3), "P": _r(p_grid, 3)},
        "tf_prior_dichte": prior2d,  # je Biom (P-Raster, T-Raster), Maximum 100
    }


def schreibe_json(d: dict[str, Any], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    s = json.dumps(d, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    path.write_text(s, encoding="utf-8")
    return len(s.encode())


def katalog_index(katalog: dict[str, Lauf], pruefungen: dict[str, Pruefung], root: Path) -> dict[str, Any]:
    """Index aller Laeufe fuer die App: Register, Beschreibungen, Stufen je Faktor, Kennzahlen und Vergleich mit dem
    Basislauf (aus den vorhandenen app.json)."""
    from .pipeline import git_revision

    apps = {}
    for lid, lauf in katalog.items():
        p = root / lauf.ordner / "app.json"
        if p.exists():
            apps[lid] = json.loads(p.read_text(encoding="utf-8"))
    laeufe = []
    for lid, lauf in katalog.items():
        pr = pruefungen[lid]
        e = {**lauf.as_dict(), "fingerabdruck": pr.fingerabdruck, "fingerabdruck_ketten": pr.fingerabdruck_ketten,
             "stufen": {f.key: stufe_von(pr.einstellungen, f, pr.recent_climate) for f in FAKTOREN.values()} if pr.einstellungen else {},
             "app_json": (lauf.ordner / "app.json").relative_to(root).as_posix() if lid in apps else None}
        if lid in apps:
            e["kennzahlen"] = apps[lid]["kennzahlen"]
            e["git"] = apps[lid]["meta"].get("git")
            if apps[lid]["fingerabdruck"] != pr.fingerabdruck:
                e["veraltet"] = True  # app.json gehoert zu einer anderen Konfiguration
            if lauf.basis in apps:
                e["vergleich_basis"] = vergleich(apps[lid], apps[lauf.basis])
        laeufe.append(e)
    # aktueller Standard: der Lauf mit den Voreinstellungen des Pakets
    standard = None
    for lid, lauf in katalog.items():
        pr = pruefungen[lid]
        if pr.einstellungen and pr.fingerabdruck == fingerabdruck(
                flache_einstellungen(Config(data_dir=STANDORTE[lauf.standort].data_dir), pr.recent_climate),
                daten_hash(root / STANDORTE[lauf.standort].data_dir)):
            standard = lid
            break
    return {
        "format": FORMAT, "version": __version__, "git": git_revision(root),
        "standard_lauf": standard, "gruppen": GRUPPEN, "rollen": ROLLEN,
        "faktoren": [{"key": f.key, "gruppe": f.gruppe, "label": f.label, "beschreibung": f.beschreibung,
                      "felder": list(f.felder),
                      "stufen": [{"key": s.key, "label": s.label,
                                  "werte": {k: list(v) if isinstance(v, tuple) else v for k, v in s.werte.items()}}
                                 for s in f.stufen]} for f in FAKTOREN.values()],
        "standorte": [{"key": s.key, "name": s.name, "daten": f"standort_{s.key}.json"}
                      for s in STANDORTE.values() if any(l.standort == s.key for l in katalog.values())],
        "laeufe": laeufe,
    }


def abgleich_referenz(outdir: Path, refdir: Path) -> dict[str, Any] | None:
    """Vergleicht rekonstruktion_alter.csv eines neu gerechneten Laufs mit der gesicherten Fassung."""
    a, b = outdir / "rekonstruktion_alter.csv", refdir / "rekonstruktion_alter.csv"
    if not (a.exists() and b.exists()):
        return None
    if a.read_bytes() == b.read_bytes():
        return {"bitgleich": True}
    ra, rb = (list(csv.DictReader(open(p, encoding="utf-8"))) for p in (a, b))
    out = {"bitgleich": False}
    for v in ("T_DJF_q50", "P_ANN_q50"):
        x = np.array([float(r[v]) for r in ra]) - np.array([float(r[v]) for r in rb])
        out[v] = {"mittlere_diff": round(float(x.mean()), 4), "max_abs_diff": round(float(np.abs(x).max()), 4)}
    return out


def kopiere_referenz(outdir: Path, refdir: Path, ueberschreiben: bool = False) -> list[str]:
    """Kopiert Einstellungen, Metadaten und Tabellen eines Laufs in seinen Katalog-Ordner (fehlende Dateien, oder
    alle mit ueberschreiben)."""
    done = []
    for name in REFERENZ_DATEIEN:
        if (outdir / name).exists() and (ueberschreiben or not (refdir / name).exists()):
            shutil.copy2(outdir / name, refdir / name)
            done.append(name)
    return done
