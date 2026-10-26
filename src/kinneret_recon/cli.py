"""Kommandozeile: ``kinneret-recon run|compare|config``."""

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
    execute(_apply_overrides(cfg, a), Path.cwd())
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="kinneret-recon", description="Bayessche Klimarekonstruktion See Genezareth")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="kompletten Lauf ausfuehren")
    r.add_argument("-c", "--config", help="TOML-Konfiguration")
    r.add_argument("--legacy", action="store_true", help="exakt wie das Originalskript (Tag v1.0-paper)")
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

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
