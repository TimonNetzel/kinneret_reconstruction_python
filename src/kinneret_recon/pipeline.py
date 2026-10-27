"""Kompletter Lauf: Daten -> Sampling -> Auswertung -> Diagnostik -> Abbildungen -> Bericht."""

from __future__ import annotations

import csv
import dataclasses
import json
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import __version__, diagnostics, sampler
from .config import Config
from .data import load_inputs
from .model import Setup, build_setup
from .postprocess import Results, postprocess
from .sampler import Posterior


@dataclass
class Run:
    config: Config
    setup: Setup
    posterior: Posterior
    results: Results
    summary: list[dict]
    outdir: Path
    timings: dict[str, float] = field(default_factory=dict)
    figures: dict = field(default_factory=dict)


def git_revision(path: Path) -> str:
    try:
        rev = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unbekannt"


def _toml_value(v) -> str:
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    return json.dumps(str(v))


def write_toml(cfg: Config, path: Path) -> None:
    """Schreibt die (flache, zweistufige) Konfiguration als TOML; None-Werte entfallen."""
    d = cfg.to_dict()
    lines = [f"{k} = {_toml_value(v)}" for k, v in d.items() if not isinstance(v, dict) and v is not None]
    for sec, vals in d.items():
        if isinstance(vals, dict):
            lines.append(f"\n[{sec}]")
            lines += [f"{k} = {_toml_value(v)}" for k, v in vals.items() if v is not None]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def reconstruction_table(res: Results, kind: str = "age") -> list[dict]:
    """Quantile der Rekonstruktion je Alter bzw. Tiefe (aus den Stichproben)."""
    x = res.age if kind == "age" else res.depth
    t, p, b = getattr(res, f"temp_{kind}"), getattr(res, f"pann_{kind}"), getattr(res, f"biomes_{kind}")
    probs = [0.025, 0.25, 0.5, 0.75, 0.975]
    tq, pq = np.quantile(t, probs, axis=0), np.quantile(p, probs, axis=0)
    bm = np.median(b, axis=0)
    rows = []
    for i in range(len(x)):
        row = {("age_cal_BP" if kind == "age" else "depth_cm"): float(x[i])}
        for j, pr in enumerate(probs):
            row[f"T_DJF_q{pr * 100:g}"] = round(float(tq[j, i]), 4)
        row["T_DJF_mean"] = round(float(t[:, i].mean()), 4)
        for j, pr in enumerate(probs):
            row[f"P_ANN_q{pr * 100:g}"] = round(float(pq[j, i]), 2)
        row["P_ANN_mean"] = round(float(p[:, i].mean()), 2)
        for k in range(bm.shape[1]):
            row[f"biome{k + 1}_median"] = round(float(bm[i, k]), 5)
        rows.append(row)
    return rows


def execute(cfg: Config, root: Path | None = None, log=print, reuse: Path | None = None) -> Run:
    """reuse: Verzeichnis eines frueheren Laufs; dessen Ketten (posterior.npz, prior.npz) werden neu ausgewertet,
    statt neu zu sampeln. Die Sampler-Einstellungen muessen dann zu diesem Lauf passen."""
    root = Path(root or Path.cwd())
    outdir = (root / cfg.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    timings = {}

    t = time.perf_counter()
    inputs = load_inputs(root / cfg.data_dir)
    setup = build_setup(inputs, cfg.sampler)
    timings["vorbereitung"] = time.perf_counter() - t

    s = cfg.sampler
    if reuse is not None:
        reuse = Path(reuse).resolve()
        log(f"Ketten aus {reuse} werden neu ausgewertet (kein Sampling)")
        post = Posterior.load(reuse / "posterior.npz")
        if post.taxa_weights.shape[0] != s.n_chains or post.burn_in != s.burn_in:
            raise ValueError("Die Sampler-Einstellungen passen nicht zum wiederverwendeten Lauf")
        timings["sampling"] = 0.0
    else:
        log(f"Sampling: {s.kernel}, {s.n_chains} Kette(n) x {s.sample_length:,} Iterationen ...")
        t = time.perf_counter()
        post = sampler.run(setup, s, cfg.postprocess.n_threads)
        timings["sampling"] = time.perf_counter() - t
    log(f"  Akzeptanzrate {np.round(post.acceptance_rate, 3)} ({timings['sampling']:.1f} s)")
    if np.any(post.acceptance_rate < 0.01):
        log("  WARNUNG: Akzeptanzrate < 1 % - die Kette ist vermutlich haengengeblieben (siehe Diagnostik)")
    post.save(outdir / "posterior.npz")

    prior = None
    if cfg.postprocess.prior_reference == "model" and reuse is not None and (reuse / "prior.npz").exists():
        prior = Posterior.load(reuse / "prior.npz")
        if reuse != outdir:
            prior.save(outdir / "prior.npz")
    elif cfg.postprocess.prior_reference == "model":
        # Prior-Referenz: dasselbe Modell ohne Kalibrierungsterme (also ohne Information aus den Daten)
        t = time.perf_counter()
        ps = dataclasses.replace(s, kernel="corrected", use_calibration=False, seed=s.seed + 1,
                                 weights_proposal="log_gamma", dirichlet_hastings_fix=True)
        prior = sampler.run(build_setup(inputs, ps), ps, cfg.postprocess.n_threads)
        prior.save(outdir / "prior.npz")
        timings["prior_lauf"] = time.perf_counter() - t

    t = time.perf_counter()
    res = postprocess(setup, post, cfg.postprocess, prior)
    timings["auswertung"] = time.perf_counter() - t
    res.save(outdir / "results.npz")
    write_csv(reconstruction_table(res, "age"), outdir / "rekonstruktion_alter.csv")
    write_csv(reconstruction_table(res, "depth"), outdir / "rekonstruktion_tiefe.csv")

    from .plots import BIOME_LABELS

    t = time.perf_counter()
    summary = diagnostics.summarize(post, list(inputs.taxa_names), BIOME_LABELS[: setup.num_biomes])
    timings["diagnostik"] = time.perf_counter() - t
    write_csv(summary, outdir / "diagnostik.csv")
    bad = [r["parameter"] for r in summary if not (r["rhat"] <= 1.01 and r["ess_bulk"] >= 400)]
    log(f"  Auswertung {timings['auswertung']:.1f} s; Diagnostik: {len(bad)} von {len(summary)} Groessen "
        f"mit R-hat > 1,01 oder ESS < 400")

    run = Run(cfg, setup, post, res, summary, outdir, timings)
    if cfg.plots:
        from .plots import all_figures

        t = time.perf_counter()
        run.figures = all_figures(setup, post, res, summary, outdir / "plots")
        timings["abbildungen"] = time.perf_counter() - t

    write_toml(cfg, outdir / "config.toml")
    meta = dict(
        version=__version__, git=git_revision(root), python=platform.python_version(),
        numpy=np.__version__, platform=platform.platform(), timings_s={k: round(v, 2) for k, v in timings.items()},
        acceptance_rate=post.acceptance_rate.tolist(), n_samples_post=int(post.pooled("expl_variance").shape[0]),
        diagnostics_problems=bad,
        posterior_aus=str(reuse) if reuse is not None else None,
    )
    for k in ("min_log_weight", "acceptance_rate_c", "final_step_z", "final_tf_scale"):
        if k in post.meta:
            meta[k] = post.meta[k]
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    if cfg.report:
        from .report import write_report

        write_report(run, outdir / "bericht.html")
    log(f"Ergebnisse in {outdir}")
    return run
