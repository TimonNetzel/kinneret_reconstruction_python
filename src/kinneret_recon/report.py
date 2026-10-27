"""HTML-Bericht eines Laufs (eigenstaendige Datei, Abbildungen eingebettet)."""

from __future__ import annotations

import base64
import datetime as dt
import html
from pathlib import Path

import numpy as np

from . import __version__
from .config import CRU_PRIOR_50KM
from .plots import BIOME_NAMES

CSS = """
:root { --fg:#1d1d1f; --muted:#5f6368; --bg:#ffffff; --line:#d9d9de; --accent:#2b6cb0; --warn:#b42318; --ok:#1e7e34; }
@media (prefers-color-scheme: dark) { :root { --fg:#e8e8ea; --muted:#a0a3a8; --bg:#17181a; --line:#3a3b3f;
  --accent:#7fb2f0; --warn:#ff8a80; --ok:#7bd88f; } img { background:#fff; } }
body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; color:var(--fg); background:var(--bg);
  max-width: 1050px; margin: 0 auto; padding: 24px 16px 64px; line-height: 1.5; }
h1 { font-size: 1.6rem; margin-bottom: .2rem; } h2 { margin-top: 2.2rem; border-bottom: 1px solid var(--line); }
.muted { color: var(--muted); } .warn { color: var(--warn); font-weight: 600; } .ok { color: var(--ok); font-weight: 600; }
table { border-collapse: collapse; font-size: .88rem; margin: .6rem 0 1rem; }
th, td { border-bottom: 1px solid var(--line); padding: 3px 10px; text-align: right; }
th:first-child, td:first-child { text-align: left; }
img { max-width: 100%; height: auto; border: 1px solid var(--line); }
.scroll { overflow-x: auto; } code { font-size: .9em; }
"""


def _img(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{html.escape(path.stem)}">'


def _table(rows: list[list], header: list[str]) -> str:
    h = "".join(f"<th>{html.escape(str(c))}</th>" for c in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>'


def _fmt(x, nd=2):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:,.{nd}f}".replace(",", " ")


def _n(x: int) -> str:
    return f"{x:,}".replace(",", "\u202f")


def methods_text(run) -> str:
    """Automatisch erzeugte Methodenbeschreibung der Einstellungen (zur Uebernahme in Manuskripte)."""
    s, setup, post = run.config.sampler, run.setup, run.posterior
    n_post = post.pooled("expl_variance").shape[0]
    kern = ("des Original-Samplers (bitidentisch zu Netzel et al., 2025)" if s.kernel == "legacy"
            else "des überarbeiteten Samplers (kinneret_recon %s)" % __version__)
    if s.kernel == "corrected" and s.weights_proposal == "log_gamma":
        proposal = ("Die Taxa-Gewichte wurden als w = y/Σy mit y_k ~ Gamma(0,5) parametrisiert und mit einem "
                    "Normal-Random-Walk auf log y aktualisiert, im Wechsel mit einem eigenen Schritt für die "
                    "Klimawerte der Transferfunktionen; die Schrittweiten wurden während des Burn-in an eine "
                    f"Akzeptanzrate von {s.target_accept:g} angepasst. ")
    else:
        proposal = (f"Die Taxa-Gewichte wurden gemeinsam mit den Klimawerten der Transferfunktionen über einen "
                    f"Dirichlet-Vorschlag mit Streuungsparameter κ = {setup.dirichlet_spread:g} aktualisiert. ")
    if s.taxa_prior == "hierarchical":
        a = post.pooled("alpha")
        taxa_prior = (f"Der Prior der {setup.num_taxa} Taxa-Gewichte ist Dir(α, …, α) mit mitgeschätzter Konzentration, "
                      f"log α ~ N(log {s.alpha_prior_median:g}, {s.alpha_prior_log_sd:g}²); Posterior-Median "
                      f"α = {np.median(a):.2f} (95 %: {np.quantile(a, .025):.2f}–{np.quantile(a, .975):.2f}). ")
    else:
        taxa_prior = f"Der Prior der {setup.num_taxa} Taxa-Gewichte ist Dir({s.taxa_prior_alpha:g}). "
    recent = {k: getattr(s, k) for k in CRU_PRIOR_50KM}
    if recent == CRU_PRIOR_50KM:
        recent_src = ("Der Prior des rezenten Klimas stammt aus CRU TS 4.07 (1961–1990) für ein Einzugsgebiet von 50 km "
                      "um den See (Gauß-Gewichtung); seine Sd ist die Unsicherheit des Gebietsmittels. ")
    elif all(v is None for v in recent.values()):
        recent_src = ("Der Prior des rezenten Klimas ist Mittel und räumliche Sd mehrerer CRU-Gitterpunkte der "
                      "Levante wie in Netzel et al. (2025). ")
    else:
        recent_src = ""
    pp = run.config.postprocess
    recon = ("Die Klimarekonstruktion je Altersstufe ist die Mischverteilung nach Gl. (3): je Stichprobe ein Biom "
             "mit Wahrscheinlichkeit seines Anteils und dessen Klimawert. " if pp.reconstruction == "mixture" else
             "Die Klimarekonstruktion je Stichprobe ist der mit den Biomanteilen gewichtete Mittelwert der "
             "Biom-Klimawerte. ")
    if s.tf_normal_prior:
        recon += "Die Klimawerte der Biome haben den Prior N_t(μ(ψ), σ(ψ)) auf dem Trunkierungsbereich (Gl. 14). "
    fixes = []
    if s.kernel == "corrected" and s.weights_proposal == "dirichlet":
        fixes.append("Hastings-Korrektur des Dirichlet-Vorschlags " +
                     ("mit" if s.dirichlet_hastings_fix else "ohne") + " Streuungsparameter κ")
    if s.kernel == "corrected":
        fixes.append("Gamma-Prior des rezenten Niederschlags " +
                     ("über exakte Momentenanpassung" if s.exact_gamma_moments else "über Gittersuche"))
    return (
        f"Die Posterior-Verteilung wurde mit {s.n_chains} Markov-Kette(n) {kern} mit je {_n(s.sample_length)} "
        f"Iterationen geschätzt. Die ersten {_n(s.burn_in)} Iterationen wurden als Burn-in verworfen, danach wurde "
        f"jede {s.thin}. Iteration verwendet ({_n(n_post)} Stichproben insgesamt). " + proposal +
        taxa_prior + f"R² der Niederschlagsrekonstruktion mit dem Baumpollenanteil hat den Prior "
        f"Beta({setup.beta_shape1:g}, {setup.beta_shape2:g}), die rezente Temperatur N({setup.temp_mean:.2f}, "
        f"{setup.temp_sd:.2f}²) und der rezente Niederschlag Gamma(Form {setup.pann_shape:.3f}, "
        f"Rate {setup.pann_rate:.5f}). " + recent_src + ("Einstellungen: " + "; ".join(fixes) + ". " if fixes else "") +
        recon +
        f"Konvergenz wurde mit dem rangnormierten Split-R̂ und der effektiven Stichprobengröße nach Vehtari et al. "
        f"(2021) geprüft."
    )


def write_report(run, path: Path) -> None:
    cfg, setup, post, res, summary = run.config, run.setup, run.posterior, run.results, run.summary
    figs = {k: [p for p in v if p.suffix == ".png"][0] for k, v in run.figures.items()}
    bad = [r for r in summary if not (r["rhat"] <= 1.01 and r["ess_bulk"] >= 400)]
    rh_max = np.nanmax([r["rhat"] for r in summary])
    ess_min = np.nanmin([r["ess_bulk"] for r in summary])
    status = (f'<span class="ok">Konvergenzkriterien erfüllt</span> (max. R̂ = {rh_max:.3f}, min. ESS = {ess_min:.0f})'
              if not bad else
              f'<span class="warn">{len(bad)} von {len(summary)} Größen verfehlen R̂ ≤ 1,01 oder ESS ≥ 400</span> '
              f'(max. R̂ = {rh_max:.3f}, min. ESS = {ess_min:.0f})')

    parts = [f"<h1>Klimarekonstruktion See Genezareth</h1>",
             f'<p class="muted">Erstellt {dt.datetime.now():%d.%m.%Y %H:%M} · kinneret_recon {__version__} · '
             f'Ausgabe <code>{html.escape(str(run.outdir))}</code></p>',
             "<h2>Überblick</h2>",
             _table([
                 ["Sampler", cfg.sampler.kernel],
                 ["Ketten × Iterationen", f"{cfg.sampler.n_chains} × {cfg.sampler.sample_length:,}".replace(",", " ")],
                 ["Burn-in / Ausdünnung", f"{cfg.sampler.burn_in:,} / {cfg.sampler.thin}".replace(",", " ")],
                 ["Akzeptanzrate je Kette", ", ".join(f"{a:.3f}" for a in post.acceptance_rate)],
                 ["Laufzeit (s)", ", ".join(f"{k} {v:.1f}" for k, v in run.timings.items())],
                 ["Quartile berechnet aus", "Stichproben" if res.quantile_method == "samples" else "Kerndichten (wie Original)"],
             ], ["", ""]),
             f"<p>{status}</p>",
             "<h2>Methodenbeschreibung</h2>", f"<p>{methods_text(run)}</p>"]

    if "reconstruction" in figs:
        parts += ["<h2>Rekonstruktion</h2>", _img(figs["reconstruction"]),
                  '<p class="muted">Farbe: Posterior-Dichte je Altersstufe; Linien: Median und Quartile.</p>']
    sel = [i for i, a in enumerate(res.age) if a % 1000 == 0 or i == 0]
    rows = [[f"{res.age[i]:.0f}", _fmt(res.temp_age_q[1, i]), f"{_fmt(res.temp_age_q[0, i])} – {_fmt(res.temp_age_q[2, i])}",
             _fmt(res.pann_age_q[1, i], 0), f"{_fmt(res.pann_age_q[0, i], 0)} – {_fmt(res.pann_age_q[2, i], 0)}"]
            for i in sel]
    parts.append(_table(rows, ["Alter [cal a BP]", "T_DJF Median [°C]", "T_DJF 25–75 %", "P_ANN Median [mm]",
                               "P_ANN 25–75 %"]))
    parts.append('<p class="muted">Vollständige Tabellen: <code>rekonstruktion_alter.csv</code>, '
                 '<code>rekonstruktion_tiefe.csv</code>.</p>')

    if "biomes" in figs:
        parts += ["<h2>Biomanteile</h2>", _img(figs["biomes"])]

    parts.append("<h2>Taxa-Gewichte</h2>")
    if "taxa_weights" in figs:
        parts.append(_img(figs["taxa_weights"]))
    w = res.post_weights
    order = np.argsort(-np.median(w, axis=0))
    names = setup.inputs.taxa_names
    biome_of = np.argmax(setup.inputs.biomes_assign, axis=1)
    parts.append(_table([[html.escape(str(names[k])), BIOME_NAMES[biome_of[k]], _fmt(np.median(w[:, k]), 4),
                          f"{_fmt(np.quantile(w[:, k], .025), 4)} – {_fmt(np.quantile(w[:, k], .975), 4)}"]
                         for k in order[:10]], ["Taxon (Top 10)", "Biom", "Median", "95-%-Intervall"]))

    parts.append("<h2>Transferfunktionen</h2>")
    for k in ("transfer_functions_2D_dists", "transfer_functions_ratio"):
        if k in figs:
            parts.append(_img(figs[k]))
    B = setup.num_biomes
    parts.append(_table([[BIOME_NAMES[b], _fmt(np.median(res.tf_post_temp[:, b]), 1),
                          _fmt(res.post_temp_ci_size[b] / res.prior_temp_ci_size[b], 2),
                          _fmt(np.median(res.tf_post_pann[:, b]), 0),
                          _fmt(res.post_pann_ci_size[b] / res.prior_pann_ci_size[b], 2)] for b in range(B)],
                        ["Biom", "T Median [°C]", "T: CI post/prior", "P Median [mm]", "P: CI post/prior"]))
    ref = cfg.postprocess.prior_reference
    parts.append('<p class="muted">CI: Breite des 95-%-Intervalls. Prior-Referenz: ' + (
        "Lauf desselben Modells ohne Kalibrierungsterme (enthält die Trunkierung der TF-Werte; Verhältnisse < 1 "
        "zeigen damit nur Information aus den Daten)." if ref == "model" else
        "Randverteilungen auf dem Klimagitter wie im Original. Achtung: Die Trunkierung der TF-Werte allein ergibt "
        "bereits Verhältnisse < 1.") + "</p>")

    for k, title in (("independent_proposal_dists", "Kalibrierung"), ("age_depth_transform", "Alters-Tiefen-Transformation")):
        if k in figs:
            parts += [f"<h2>{title}</h2>", _img(figs[k])]

    parts.append("<h2>Konvergenzdiagnostik</h2>")
    for k in ("diagnostics_traces", "diagnostics_rhat_ess"):
        if k in figs:
            parts.append(_img(figs[k]))
    key = [r for r in summary if not r["parameter"].startswith("w[")]
    parts.append(_table([[r["parameter"], _fmt(r["mean"], 3), _fmt(r["sd"], 3), _fmt(r["rhat"], 3),
                          _fmt(r["ess_bulk"], 0), _fmt(r["ess_tail"], 0)] for r in key],
                        ["Größe", "Mittel", "Sd", "R̂", "ESS bulk", "ESS tail"]))
    parts.append('<p class="muted">Alle Größen inkl. Taxa-Gewichte: <code>diagnostik.csv</code>.</p>')

    doc = (f'<!doctype html><html lang="de"><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width, initial-scale=1">'
           f"<title>Kinneret-Rekonstruktion</title><style>{CSS}</style></head><body>{''.join(parts)}</body></html>")
    path.write_text(doc, encoding="utf-8")
