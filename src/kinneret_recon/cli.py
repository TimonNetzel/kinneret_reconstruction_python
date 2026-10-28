"""Kommandozeile: ``kinneret-recon run|compare|config|cru-extract|cru-prior|katalog|batch|app-daten``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Config, legacy_config


def _apply_overrides(cfg: Config, a: argparse.Namespace) -> Config:
    s = cfg.sampler
    if a.output:
        cfg.output_dir = a.output
    if a.data:
        cfg.data_dir = a.data
    for name in ("chains", "samples", "burn_in", "thin", "seed"):
        v = getattr(a, name)
        if v is not None:
            setattr(s, {"chains": "n_chains", "samples": "sample_length"}.get(name, name), v)
    if a.no_hastings_fix:
        s.dirichlet_hastings_fix = False
    if a.quantiles:
        cfg.postprocess.quantile_method = a.quantiles
    if a.reconstruction:
        cfg.postprocess.reconstruction = a.reconstruction
    if a.no_plots:
        cfg.plots = cfg.report = False
    if a.no_report:
        cfg.report = False
    cfg.validate()
    return cfg


def cmd_run(a: argparse.Namespace) -> int:
    from .pipeline import execute

    if a.legacy:
        cfg = legacy_config(output_dir="results/legacy")
    elif a.config:
        cfg = Config.from_toml(a.config)
    else:
        cfg = Config()
    execute(_apply_overrides(cfg, a), Path.cwd(), reuse=a.from_run)
    return 0


def cmd_compare(a: argparse.Namespace) -> int:
    import numpy as np

    from .data import load_inputs
    from .pipeline import write_csv
    from .plots import plot_comparison, plot_weights_comparison
    from .postprocess import Results

    runs = {}
    for spec in a.runs:
        label, _, path = spec.partition("=") if "=" in spec else (Path(spec).name, "", spec)
        runs[label] = Results.load(Path(path) / "results.npz")
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    plot_comparison(runs, out)
    names = load_inputs(a.data).taxa_names
    plot_weights_comparison(runs, list(names), out)
    labels = list(runs)
    base = runs[labels[0]]
    rows = []
    for lab in labels[1:]:
        r = runs[lab]
        for v, unit in (("temp", "°C"), ("pann", "mm")):
            d = getattr(r, f"{v}_age_q")[1] - getattr(base, f"{v}_age_q")[1]
            rows.append(dict(vergleich=f"{lab} - {labels[0]}", groesse=f"{v} Median [{unit}]",
                             mittlere_diff=round(float(d.mean()), 4), max_abs_diff=round(float(np.abs(d).max()), 4)))
    if rows:
        write_csv(rows, out / "vergleich.csv")
        for r in rows:
            print(f"{r['vergleich']:40s} {r['groesse']:16s} mittl. Diff {r['mittlere_diff']:9.3f}  "
                  f"max |Diff| {r['max_abs_diff']:9.3f}")
    print(f"Vergleich in {out}")
    return 0


def cmd_config(a: argparse.Namespace) -> int:
    from .pipeline import write_toml

    cfg = legacy_config() if a.legacy else Config()
    path = Path(a.path)
    write_toml(cfg, path)
    print(f"Konfiguration geschrieben: {path}")
    return 0


def cmd_cru_extract(a: argparse.Namespace) -> int:
    from . import cru

    if a.download:
        cru.download(a.raw)
    out = cru.extract(a.raw, a.output)
    print(f"Ausschnitt geschrieben: {out}")
    return 0


def cmd_cru_prior(a: argparse.Namespace) -> int:
    import csv
    import json

    from . import cru

    data = cru.CruData.load(a.cru)
    if a.toml and len(a.radius) > 1:
        raise SystemExit("--toml nur mit einem Radius")
    rows = []
    for radius in a.radius:
        s = cru.PriorSettings(radius_km=radius, kernel=a.kernel, period=tuple(a.period),
                              target_period=None if a.no_target_period else tuple(a.target_period),
                              lapse_rate=a.lapse_rate, ref_elevation=a.ref_elevation,
                              center=(a.lat, a.lon), alt_radii=tuple(a.alt_radii),
                              extra_temp_sd=a.extra_temp_sd, extra_pann_rel_sd=a.extra_pann_rel_sd)
        res = cru.recent_prior(data, s)
        print(res.table())
        rows.append(res)
    if a.toml:
        Path(a.toml).write_text(rows[0].toml(), encoding="utf-8")
        print(f"Prior-Einstellungen geschrieben: {a.toml}")
    if a.json:
        Path(a.json).write_text(json.dumps([r.as_dict() for r in rows], indent=2, ensure_ascii=False), encoding="utf-8")
    if a.cells:
        with open(a.cells, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["radius_km", *rows[0].cells[0]])
            w.writeheader()
            for r in rows:
                for c in r.cells:
                    w.writerow({"radius_km": r.settings.radius_km, **{k: round(v, 4) for k, v in c.items()}})
    return 0


def _katalog_pruefen(ids):
    from .katalog import lade_katalog, pruefe_katalog, reihenfolge

    kat = lade_katalog()
    unbekannt = [i for i in ids or () if i not in kat]
    if unbekannt:
        raise SystemExit(f"nicht im Katalog: {', '.join(unbekannt)}")
    pr = pruefe_katalog(kat)
    return kat, pr, reihenfolge(kat, ids or None)


def _ergebnis_status(lauf, pruefung, outdir: Path) -> str:
    """'vorhanden', wenn results/<lauf>/ vollstaendig ist und zur Konfiguration des Katalogs passt."""
    from .katalog import daten_hash, fingerabdruck, flache_einstellungen

    if not all((outdir / f).exists() for f in ("config.toml", "posterior.npz", "results.npz", "meta.json")):
        return "fehlt"
    try:
        cfg = Config.from_toml(outdir / "config.toml")
    except (OSError, ValueError, TypeError):
        return "veraltet"
    fp = fingerabdruck(flache_einstellungen(cfg, pruefung.recent_climate), daten_hash(Path(cfg.data_dir)))
    return "vorhanden" if fp == pruefung.fingerabdruck else "veraltet"


def cmd_katalog(a: argparse.Namespace) -> int:
    from .katalog import FAKTOREN

    kat, pr, ids = _katalog_pruefen(a.laeufe)
    if a.faktoren:
        for f in FAKTOREN.values():
            print(f"{f.key:20s} [{f.gruppe}] {f.label['de']}: " + ", ".join(s.key for s in f.stufen))
        return 0
    res = Path(a.results)
    n_fehler = 0
    print(f"{'Lauf':24s} {'Rolle':13s} {'Basis':18s} {'Faktor = Stufe':34s} {'Ergebnisse':10s} {'app.json':8s} Ketten")
    for i in ids:
        l, p = kat[i], pr[i]
        st = _ergebnis_status(l, p, res / i) if not p.fehler else "–"
        fs = f"{l.faktor} = {l.stufe}" if l.faktor else ""
        app = "ja" if (l.ordner / "app.json").exists() else "nein"
        print(f"{i:24s} {l.rolle:13s} {l.basis or '':18s} {fs:34s} {st:10s} {app:8s} {p.fingerabdruck_ketten}")
        for e in p.fehler:
            print(f"    FEHLER: {e}")
        n_fehler += len(p.fehler)
    print(f"{len(ids)} Laeufe, {n_fehler} Fehler")
    return 1 if n_fehler else 0


def _schreibe_app_daten(ids, kat, pr, res: Path, ueberschreiben: bool, log=print) -> None:
    from .app_daten import (APP_DIR, LaufDaten, abgleich_referenz, katalog_index, kopiere_referenz, schreibe_json,
                            standort_daten, zusammenfassung)
    from .katalog import STANDORTE

    root = Path.cwd()
    for i in ids:
        if _ergebnis_status(kat[i], pr[i], res / i) != "vorhanden":
            log(f"{i}: keine passenden Ergebnisse in {res / i} (kinneret-recon batch {i})")
            continue
        ab = abgleich_referenz(res / i, kat[i].ordner)
        kopiert = kopiere_referenz(res / i, kat[i].ordner, ueberschreiben)
        n = schreibe_json(zusammenfassung(kat[i], LaufDaten.load(res / i, root), root, pr[i]), kat[i].ordner / "app.json")
        info = "" if ab is None else (" (Tabellen bitgleich mit der Referenz)" if ab["bitgleich"] else
                                      f" (Abweichung zur Referenz: T {ab['T_DJF_q50']}, P {ab['P_ANN_q50']})")
        log(f"{i}: app.json {n / 1024:.0f} KB{info}" + (f"; kopiert: {', '.join(kopiert)}" if kopiert else ""))
    for key in sorted({kat[i].standort for i in kat}):
        n = schreibe_json(standort_daten(STANDORTE[key], root), APP_DIR / f"standort_{key}.json")
        log(f"Standort {key}: {n / 1024:.0f} KB")
    n = schreibe_json(katalog_index(kat, pr, root), APP_DIR / "katalog.json")
    log(f"Katalog-Index {APP_DIR / 'katalog.json'}: {n / 1024:.0f} KB")


def cmd_app_daten(a: argparse.Namespace) -> int:
    kat, pr, ids = _katalog_pruefen(a.laeufe)
    bad = [i for i in ids if pr[i].fehler]
    if bad:
        raise SystemExit(f"Katalog fehlerhaft ({', '.join(bad)}); siehe kinneret-recon katalog")
    _schreibe_app_daten(ids, kat, pr, Path(a.results), a.ueberschreiben)
    return 0


def cmd_batch(a: argparse.Namespace) -> int:
    """Rechnet alle (oder die genannten) Laeufe des Katalogs, deren Ergebnisse fehlen oder veraltet sind, und
    schreibt danach die App-Daten. Laeufe mit gleichen Ketten (gleicher Fingerabdruck der Sampler-Einstellungen)
    werden nur einmal gesampelt und sonst neu ausgewertet."""
    from .pipeline import execute

    kat, pr, ids = _katalog_pruefen(a.laeufe)
    bad = [i for i in ids if pr[i].fehler]
    if bad:
        raise SystemExit(f"Katalog fehlerhaft ({', '.join(bad)}); siehe kinneret-recon katalog")
    res = Path(a.results)
    status = {i: ("neu" if a.neu else _ergebnis_status(kat[i], pr[i], res / i)) for i in ids}
    plan = []
    ketten: dict[str, str] = {}  # Fingerabdruck der Ketten -> Lauf mit vorhandenen oder geplanten Ketten
    for i in kat:
        if i not in ids and _ergebnis_status(kat[i], pr[i], res / i) == "vorhanden":
            ketten.setdefault(pr[i].fingerabdruck_ketten, i)
    for i in ids:
        if status[i] == "vorhanden":
            ketten.setdefault(pr[i].fingerabdruck_ketten, i)
    for i in ids:
        if status[i] == "vorhanden":
            continue
        spender = ketten.get(pr[i].fingerabdruck_ketten)
        plan.append((i, spender))
        ketten.setdefault(pr[i].fingerabdruck_ketten, i)
    for i, spender in plan:
        print(f"{i:24s} " + (f"Ketten von {spender} neu auswerten" if spender else "rechnen"))
    print(f"{len(plan)} von {len(ids)} Laeufen zu rechnen")
    if a.dry_run:
        return 0
    for i, spender in plan:
        cfg = kat[i].lade_config(Path.cwd())
        cfg.output_dir = str(res / i)
        cfg.plots = cfg.report = a.plots
        print(f"\n=== {i} ({kat[i].config})", flush=True)
        execute(cfg, Path.cwd(), reuse=res / spender if spender else None)
    _schreibe_app_daten(ids, kat, pr, res, a.ueberschreiben)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="kinneret-recon", description="Bayessche Klimarekonstruktion See Genezareth")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="kompletten Lauf ausfuehren")
    r.add_argument("-c", "--config", help="TOML-Konfiguration")
    r.add_argument("--legacy", action="store_true", help="exakt wie das Originalskript (Tag v0.1.0)")
    r.add_argument("-o", "--output", help="Ausgabeverzeichnis")
    r.add_argument("--data", help="Verzeichnis der Eingangsdaten")
    r.add_argument("--chains", type=int)
    r.add_argument("--samples", type=int, help="Iterationen je Kette")
    r.add_argument("--burn-in", dest="burn_in", type=int)
    r.add_argument("--thin", type=int)
    r.add_argument("--seed", type=int)
    r.add_argument("--no-hastings-fix", action="store_true", help="Hastings-Korrektur wie im Original (ohne kappa)")
    r.add_argument("--quantiles", choices=["samples", "kde"])
    r.add_argument("--no-plots", action="store_true", help="keine Abbildungen und kein Bericht")
    r.add_argument("--no-report", action="store_true")
    r.add_argument("--from-run", dest="from_run", help="Ketten eines frueheren Laufs neu auswerten (kein Sampling)")
    r.add_argument("--reconstruction", choices=["mean", "mixture"],
                   help="Rekonstruktion als Mittelwert der Biom-Klimawerte oder als Mischverteilung (Gl. 3)")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("compare", help="Rekonstruktionen mehrerer Laeufe vergleichen")
    c.add_argument("runs", nargs="+", help="Laufverzeichnisse, optional als Name=Pfad")
    c.add_argument("-o", "--output", default="results/vergleich")
    c.add_argument("--data", default="data/in")
    c.set_defaults(func=cmd_compare)

    g = sub.add_parser("config", help="Vorlage einer Konfigurationsdatei schreiben")
    g.add_argument("path", nargs="?", default="config.toml")
    g.add_argument("--legacy", action="store_true")
    g.set_defaults(func=cmd_config)

    from . import cru

    x = sub.add_parser("cru-extract", help="Levante-Ausschnitt aus den CRU-Rohdateien erzeugen")
    x.add_argument("raw", help="Verzeichnis der Rohdateien (CRU TS 4.07 *.dat.nc.gz, CRU CL 2.0 grid_10min_*.dat.gz)")
    x.add_argument("-o", "--output", default="data/in/cru_levante.npz")
    x.add_argument("--download", action="store_true", help="fehlende Rohdateien von crudata.uea.ac.uk laden (~320 MB)")
    x.set_defaults(func=cmd_cru_extract)

    q = sub.add_parser("cru-prior", help="Prior des rezenten Klimas fuer ein Einzugsgebiet aus CRU-Daten berechnen")
    q.add_argument("--cru", default="data/in/cru_levante.npz", help="Ausschnitt (kinneret-recon cru-extract)")
    q.add_argument("-r", "--radius", type=float, nargs="+", default=[50.0], help="Radius(e) des Einzugsgebiets [km]")
    q.add_argument("--kernel", choices=cru.KERNELS, default="gauss",
                   help="Entfernungsgewichtung: gauss (sigma = R/2), uniform, linear (1 - d/R)")
    q.add_argument("--period", type=int, nargs=2, default=[1961, 1990], metavar=("VON", "BIS"))
    q.add_argument("--target-period", type=int, nargs=2, default=[1991, 2020], metavar=("VON", "BIS"),
                   help="Zeitraum der juengsten Altersstufe; Abstand geht als Unsicherheit ein")
    q.add_argument("--no-target-period", action="store_true", help="ohne Zeitraum-Komponente")
    q.add_argument("--lapse-rate", type=float, help="Hoehenkorrektur von T [degC/km] auf --ref-elevation")
    q.add_argument("--ref-elevation", type=float, default=cru.LAKE_ELEVATION_M, help="Bezugshoehe [m]")
    q.add_argument("--lat", type=float, default=cru.LAKE_LAT)
    q.add_argument("--lon", type=float, default=cru.LAKE_LON)
    q.add_argument("--alt-radii", type=float, nargs="+", default=[25.0, 50.0, 100.0],
                   help="alternative Radien fuer die Repraesentativitaet [km]")
    q.add_argument("--extra-temp-sd", type=float, default=0.0, help="zusaetzlicher Fehler fuer T [degC]")
    q.add_argument("--extra-pann-rel-sd", type=float, default=0.0, help="zusaetzlicher relativer Fehler fuer P")
    q.add_argument("--toml", help="Prior als [sampler]-Abschnitt in diese Datei schreiben")
    q.add_argument("--json", help="alle Kennzahlen als JSON")
    q.add_argument("--cells", help="beteiligte Gitterzellen mit Gewichten als CSV")
    q.set_defaults(func=cmd_cru_prior)

    k = sub.add_parser("katalog", help="Lauf-Katalog anzeigen und pruefen (reference/laeufe/*/lauf.toml)")
    k.add_argument("laeufe", nargs="*", help="nur diese Laeufe")
    k.add_argument("--results", default="results", help="Verzeichnis der Ergebnisse")
    k.add_argument("--faktoren", action="store_true", help="Faktor-Register anzeigen")
    k.set_defaults(func=cmd_katalog)

    b = sub.add_parser("batch", help="fehlende oder veraltete Laeufe des Katalogs rechnen, danach app-daten")
    b.add_argument("laeufe", nargs="*", help="nur diese Laeufe (Standard: alle)")
    b.add_argument("--results", default="results", help="Verzeichnis der Ergebnisse (je Lauf ein Unterordner)")
    b.add_argument("-n", "--dry-run", action="store_true", help="nur anzeigen, was gerechnet wuerde")
    b.add_argument("--neu", action="store_true", help="auch vorhandene Ergebnisse neu rechnen")
    b.add_argument("--plots", action="store_true", help="Abbildungen und Bericht je Lauf erzeugen")
    b.add_argument("--ueberschreiben", action="store_true",
                   help="Tabellen und Metadaten im Katalog-Ordner durch die neuen ersetzen (sonst nur fehlende)")
    b.set_defaults(func=cmd_batch)

    d = sub.add_parser("app-daten", help="kompakte Zusammenfassungen (app.json) aus vorhandenen Ergebnissen schreiben")
    d.add_argument("laeufe", nargs="*", help="nur diese Laeufe (Standard: alle)")
    d.add_argument("--results", default="results", help="Verzeichnis der Ergebnisse (je Lauf ein Unterordner)")
    d.add_argument("--ueberschreiben", action="store_true",
                   help="Tabellen und Metadaten im Katalog-Ordner durch die neuen ersetzen (sonst nur fehlende)")
    d.set_defaults(func=cmd_app_daten)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
