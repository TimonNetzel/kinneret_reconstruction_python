"""Independently recalculate local climate curves, verify provenance, optionally replay.

Uses direct matrix products from input spectra and saved posterior draws;
does not call pipeline/postprocess reconstruction or CSV-writing functions.
Reports archived runtime differences without changing references or tolerances.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".local/matplotlib"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def climate_draws(inputs, weights, tf, kind):
    """Direct, per-position weighted sum; independent of package postprocessing."""
    spectrum = inputs.taxa_counts_depth.copy()
    spectrum /= spectrum.sum(axis=1)[:, None]
    if kind == "age":
        indices = (inputs.depth - inputs.depth.min()).astype(int)
        transform = inputs.depth_given_age[:, indices].copy()
        transform /= transform.sum(axis=1)[:, None]
        spectrum = transform @ spectrum
    count, positions = len(weights), len(spectrum)
    temp = np.empty((count, positions))
    pann = np.empty_like(temp)
    b = inputs.num_biomes
    for j, row in enumerate(spectrum):
        masses = (weights * row) @ inputs.biomes_assign
        masses /= masses.sum(axis=1)[:, None]
        temp[:, j] = (masses * tf[:, :b]).sum(axis=1)
        pann[:, j] = (masses * tf[:, b:]).sum(axis=1)
    return temp, pann


def csv_check(path, temp, pann):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    checks = []
    for prefix, draws, digits in (("T_DJF", temp, 4), ("P_ANN", pann, 2)):
        for probability in (0.025, 0.25, 0.5, 0.75, 0.975):
            column = f"{prefix}_q{100 * probability:g}"
            actual = np.array([float(row[column]) for row in rows])
            expected = np.round(np.quantile(draws, probability, axis=0), digits)
            checks.append(dict(column=column, passed=bool(np.array_equal(actual, expected)),
                               max_difference=float(np.abs(actual - expected).max())))
        actual = np.array([float(row[f"{prefix}_mean"]) for row in rows])
        expected = np.round(draws.mean(axis=0), digits)
        checks.append(dict(column=f"{prefix}_mean", passed=bool(np.array_equal(actual, expected)),
                           max_difference=float(np.abs(actual - expected).max())))
    return checks


def diagnostics(rows):
    return dict(parameters=len(rows), max_rhat=max(float(r["rhat"]) for r in rows),
                min_ess_bulk=min(float(r["ess_bulk"]) for r in rows),
                min_ess_tail=min(float(r["ess_tail"]) for r in rows),
                failures=[r["parameter"] for r in rows if not (float(r["rhat"]) <= 1.01 and float(r["ess_bulk"]) >= 400)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--corrected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--baseline-replay-record", type=Path)
    parser.add_argument("--corrected-replay-record", type=Path)
    args = parser.parse_args()
    if not args.replay and not (args.baseline_replay_record and args.corrected_replay_record):
        parser.error("Use --replay or provide both verified replay records")
    out = (ROOT / args.output).resolve()
    if ROOT not in out.parents or out.relative_to(ROOT).parts[0] != "results" or out.exists():
        raise SystemExit("Use a new output directory under results/")
    out.mkdir(parents=True)
    from kinneret_recon import sampler
    from kinneret_recon.config import Config
    from kinneret_recon.data import load_inputs
    from kinneret_recon.diagnostics import rhat, ess_bulk
    from kinneret_recon.model import build_setup
    from kinneret_recon.sampler import Posterior
    inputs = load_inputs(ROOT / "data/in")
    report = dict(scientific_status="provisional_not_approved", case="Kinneret_paper_biome_TDJF_PANN",
                  verification_script_sha256=sha(__file__),
                  dissertation_taxon_multiproxy_reproduced=False, runs={},
                  criteria=dict(rhat_max=1.01, ess_bulk_min=400, csv_comparison="exact_at_existing_export_rounding"))
    curve_summaries = {}
    chain_fields = ("taxa_weights", "tf_sample", "expl_variance", "recent_temp", "recent_pann", "log_target", "alpha", "iterations")
    for label, directory in (("Original lokal", args.baseline), ("Korrigiert lokal", args.corrected)):
        directory = (ROOT / directory).resolve()
        print(f"Checking {label}: {directory.name}", flush=True)
        provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
        hash_failures = []
        for key, parent in (("input_sha256", ROOT), ("scientific_source_sha256", ROOT), ("outputs_sha256", directory)):
            for name, expected in provenance[key].items():
                if sha(parent / name) != expected:
                    hash_failures.append(f"{key}:{name}")
        cfg = Config.from_toml(directory / "config.toml")
        post = Posterior.load(directory / "posterior.npz")
        weights, tf = post.pooled("taxa_weights"), post.pooled("tf_sample")
        setup = build_setup(inputs, cfg.sampler)
        invariants = dict(weights_finite=bool(np.isfinite(weights).all()), weights_nonnegative=bool((weights >= 0).all()),
                          weights_normalized=bool(np.allclose(weights.sum(axis=1), 1, rtol=0, atol=1e-12)),
                          climate_finite=bool(np.isfinite(tf).all()),
                          transfer_values_in_bounds=bool(((tf >= setup.tf_lower) & (tf <= setup.tf_upper)).all()))
        all_csv = {}
        climate_diagnostics = []
        for kind in ("age", "depth"):
            temp, pann = climate_draws(inputs, weights, tf, kind)
            all_csv[kind] = csv_check(directory / f"rekonstruktion_{'alter' if kind == 'age' else 'tiefe'}.csv", temp, pann)
            invariants[f"{kind}_pann_nonnegative"] = bool((pann >= 0).all())
            if kind == "age":
                curve_summaries[label] = [np.quantile(x, [0.025, 0.25, 0.5, 0.75, 0.975], axis=0) for x in (temp, pann)]
                if cfg.sampler.n_chains > 1:
                    for variable, array in (("T_DJF", temp), ("P_ANN", pann)):
                        chained = array.reshape(cfg.sampler.n_chains, -1, len(inputs.age))
                        for j, age in enumerate(inputs.age):
                            rr, ee = rhat(chained[:, :, j]), ess_bulk(chained[:, :, j])
                            climate_diagnostics.append(dict(variable=variable, age_cal_BP=float(age), rhat=rr, ess_bulk=ee,
                                                            passed=bool(rr <= 1.01 and ee >= 400)))
        with (directory / "diagnostik.csv").open(encoding="utf-8", newline="") as stream:
            diagnostic_rows = list(csv.DictReader(stream))
        record = dict(run=directory.relative_to(ROOT).as_posix(), source_and_output_hash_failures=hash_failures,
                      invariants=invariants, csv_checks=all_csv, parameter_diagnostics=diagnostics(diagnostic_rows),
                      climate_diagnostics=climate_diagnostics, chains=cfg.sampler.n_chains,
                      iterations_per_chain=cfg.sampler.sample_length, retained_postburn_draws=len(weights),
                      acceptance_rate=post.acceptance_rate.tolist(), elapsed_s=provenance["elapsed_s"],
                      peak_working_set_bytes=provenance["peak_working_set_bytes"])
        replay_record = args.baseline_replay_record if label == "Original lokal" else args.corrected_replay_record
        if replay_record is not None:
            replay = json.loads((ROOT / replay_record).read_text(encoding="utf-8"))
            assert replay["status"] == "completed" and replay["posterior_unchanged"]
            assert replay["posterior_sha256"] == sha(directory / "posterior.npz")
            assert replay["sampler_config"] == json.loads(json.dumps(cfg.to_dict()["sampler"]))
            assert replay["extension_sha256"] == provenance["extension_sha256"]
            for key in ("source_sha256", "input_sha256"):
                assert all(sha(ROOT / name) == expected for name, expected in replay[key].items())
            record["deterministic_replay"] = dict(elapsed_s=replay["elapsed_s"], fields=replay["fields"],
                                                   record_sha256=sha(ROOT / replay_record), threads=replay["threads"])
        elif args.replay:
            print(f"Replaying {label}: {cfg.sampler.n_chains} x {cfg.sampler.sample_length}", flush=True)
            started = time.perf_counter()
            repeated = sampler.run(setup, cfg.sampler, n_threads=2)
            record["deterministic_replay"] = dict(elapsed_s=time.perf_counter() - started,
                fields={key: bool(np.array_equal(getattr(post, key), getattr(repeated, key), equal_nan=True)) for key in chain_fields})
            del repeated
        report["runs"][label] = record
        # Persist intermediate evidence before the next potentially long replay.
        (out / "verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        del setup, post, temp, pann
    with np.load(ROOT / "reference/v1.0-paper/posterior_thinned.npz") as archived:
        archive_weights = archived["taxa_weights"].reshape(-1, inputs.num_taxa)
        archive_tf = archived["tf_sample"].reshape(-1, 2 * inputs.num_biomes)
        temp, pann = climate_draws(inputs, archive_weights, archive_tf, "age")
        curve_summaries["Archivierte Referenz"] = [np.quantile(x, [0.025, 0.25, 0.5, 0.75, 0.975], axis=0) for x in (temp, pann)]
    comparisons = []
    for first, second in (("Original lokal", "Archivierte Referenz"), ("Korrigiert lokal", "Original lokal")):
        for j, variable in enumerate(("T_DJF", "P_ANN")):
            difference = curve_summaries[first][j][2] - curve_summaries[second][j][2]
            comparisons.append(dict(first=first, second=second, variable=variable,
                                    signed_mean_median_difference=float(difference.mean()),
                                    max_absolute_median_difference=float(np.abs(difference).max())))
    report["comparisons"] = comparisons
    report["legacy_runtime_equivalence"] = json.loads((ROOT / ".local/legacy_same_runtime_check.json").read_text())
    report["existing_test_summaries"] = {
        name: dict(summary=(ROOT / ".local" / name).read_text(encoding="utf-8").strip().splitlines()[-1],
                   sha256=sha(ROOT / ".local" / name))
        for name in ("python-tests-lf.log", "cpp-tests.log", "sampler-tests.log") if (ROOT / ".local" / name).exists()}
    length_record = ROOT / ".local/dirichlet_prior_length_diagnostic.json"
    report["dirichlet_prior_length_diagnostic"] = json.loads(length_record.read_text()) if length_record.exists() else []
    runtime_record = ROOT / ".local/random_runtime_probe.json"
    report["random_runtime_probe"] = json.loads(runtime_record.read_text(encoding="utf-8-sig")) if runtime_record.exists() else None
    report["archived_bitwise_test"] = "failed_and_preserved_runtime_dependency" if report["legacy_runtime_equivalence"]["archive_acceptance_mismatches"] else "short_chain_acceptance_matched"
    report["technical_checks_passed"] = report["legacy_runtime_equivalence"]["passed"] and all(not run["source_and_output_hash_failures"] and all(run["invariants"].values())
        and all(check["passed"] for rows in run["csv_checks"].values() for check in rows)
        and all(run.get("deterministic_replay", {}).get("fields", {}).values()) for run in report["runs"].values())
    corrected = report["runs"]["Korrigiert lokal"]
    report["corrected_convergence_passed"] = not corrected["parameter_diagnostics"]["failures"] and all(row["passed"] for row in corrected["climate_diagnostics"])
    (out / "verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    colors = {"Korrigiert lokal": "#a64216", "Original lokal": "#444c56", "Archivierte Referenz": "#286b9e"}
    for j, ax in enumerate(axes):
        q = curve_summaries["Korrigiert lokal"][j]
        ax.fill_between(inputs.age / 1000, q[0], q[4], color=colors["Korrigiert lokal"], alpha=.13, label="95-%-Intervall, korrigiert")
        ax.fill_between(inputs.age / 1000, q[1], q[3], color=colors["Korrigiert lokal"], alpha=.23, label="50-%-Intervall, korrigiert")
        for label in ("Korrigiert lokal", "Original lokal", "Archivierte Referenz"):
            ax.plot(inputs.age / 1000, curve_summaries[label][j][2], color=colors[label],
                    linestyle=":" if label == "Archivierte Referenz" else "-", linewidth=1.4, label=label)
        ax.set_ylabel((r"Wintertemperatur $T_\mathrm{DJF}$ [°C]", r"Jahresniederschlag $P_\mathrm{ANN}$ [mm/Jahr]")[j])
        ax.grid(alpha=.2)
        ax.set_xlim(-.05, 8.95)  # Nutzerwunsch: jung links, alt rechts.
    axes[0].legend(fontsize=8, loc="upper left", ncols=2)
    axes[1].set_xlabel("Alter [ka cal BP]")
    fig.suptitle("Kinneret: vorläufiges Referenzbeispiel\nVorhandenes Paper-Biommodell; taxonbasierte Dissertation nicht reproduziert", fontsize=12)
    report["plot_rendering"] = dict(age_axis="young_to_old_left_to_right", x_limits_ka_cal_BP=[-.05, 8.95])
    (out / "verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fig.savefig(out / "comparison.png", dpi=180)
    plt.close(fig)
    write_review(out, report)
    print(json.dumps(dict(technical_checks_passed=report["technical_checks_passed"],
                          corrected_convergence_passed=report["corrected_convergence_passed"], comparisons=comparisons), indent=2), flush=True)
    return 0 if report["technical_checks_passed"] and report["corrected_convergence_passed"] else 1


def write_review(out, report):
    rows = []
    for name, run in report["runs"].items():
        d = run["parameter_diagnostics"]
        rows.append(f"<tr><td>{html.escape(name)}</td><td>{run['chains']} × {run['iterations_per_chain']:,}</td>"
                    f"<td>{d['max_rhat']:.5f}</td><td>{d['min_ess_bulk']:.0f}</td><td>{run['elapsed_s']:.1f} s</td></tr>")
    differences = []
    for row in report["comparisons"]:
        unit = "°C" if row["variable"] == "T_DJF" else "mm/Jahr"
        differences.append(f"<li>{html.escape(row['first'])} gegenüber {html.escape(row['second'])}, {row['variable']}: "
                           f"mittlere Mediandifferenz {row['signed_mean_median_difference']:.4f} {unit}, "
                           f"maximal absolut {row['max_absolute_median_difference']:.4f} {unit}.</li>")
    image = base64.b64encode((out / "comparison.png").read_bytes()).decode()
    technical = "bestanden" if report["technical_checks_passed"] else "NICHT bestanden"
    convergence = "bestanden" if report["corrected_convergence_passed"] else "NICHT bestanden"
    test_summaries = "".join(f"<li>{html.escape(name)}: {html.escape(row['summary'])}</li>" for name, row in report["existing_test_summaries"].items())
    longer = report["dirichlet_prior_length_diagnostic"][-1] if report["dirichlet_prior_length_diagnostic"] else None
    longer_note = (f"<p>Der weitere fehlgeschlagene Kurztest betrifft die optionale Dirichlet-Vorschlagsvariante. Mit vierfacher Länge und unveränderter Mittelwerttoleranz besteht die Diagnose: R̂ maximal {max(longer['rhat']):.5f}, Bulk-ESS mindestens {min(longer['ess_bulk']):.0f}. Dies dokumentiert die Längenabhängigkeit der Durchmischung; der ursprüngliche Test bleibt fehlgeschlagen. Das Referenzbeispiel verwendet die schon vorhandene Log-Gamma-Vorschlagsvariante.</p>"
                   if longer else "<p>Keine ergänzende lokale Dirichlet-Längendiagnose eingelesen.</p>")
    archive_note = ("Der archivierte Bitgleichheitstest schlägt mit dieser C++-Laufzeit fehl."
                    if report["legacy_runtime_equivalence"]["archive_acceptance_mismatches"]
                    else "Die Akzeptanzentscheidungen stimmen im kurzen Archivvergleich überein; die vollständigen Testprotokolle sind separat zu beurteilen.")
    page = f"""<!doctype html><html lang="de"><meta charset="utf-8"><title>Kinneret: vorläufiges Referenzbeispiel</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;max-width:1080px;margin:32px auto;padding:0 20px;color:#17212b}}h1,h2{{line-height:1.2}}.notice{{background:#fff2d6;border-left:5px solid #b86d00;padding:16px}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;border-bottom:1px solid #cdd3d8;padding:10px}}img{{width:100%;height:auto}}code{{background:#edf1f4;padding:2px 4px}}</style>
<h1>Kinneret: vorläufiges Referenzbeispiel</h1>
<p class="notice"><strong>Fachliche Freigabe steht aus.</strong> Reproduziert und geprüft wird der vorhandene Paperfall mit drei Biomen, 33 Taxa und Wintertemperatur. Dies ist keine vollständige Reproduktion der taxonbasierten Multiproxy-Dissertation und keine Freigabe weiterer Fossilfälle.</p>
<p>Technische Integritäts- und Nachrechnungsprüfungen: <strong>{technical}</strong>. Konvergenzkriterien der korrigierten Variante: <strong>{convergence}</strong>. Die Einzelprüfungen stehen in <a href="verification.json">verification.json</a>.</p>
<img alt="Kinneret-Vergleich mit Unsicherheitsintervallen" src="data:image/png;base64,{image}">
<h2>Ausgeführte Läufe</h2><table><tr><th>Lauf</th><th>Ketten × Iterationen</th><th>max. R̂</th><th>min. Bulk-ESS</th><th>Gesamtlaufzeit</th></tr>{''.join(rows)}</table>
<p>Vorhandene Kriterien: R̂ ≤ 1,01 und Bulk-ESS ≥ 400. Beim Original stammen die Split-Diagnosen aus einer einzigen Kette; sie ersetzen keine unabhängigen Startketten. Korrigiert wurden zusätzlich sämtliche 181 Altersstufen beider Klimagrößen geprüft. Ausdünnung 600 speichert insgesamt 10.000 Nach-Burn-in-Ziehungen aus vier Ketten.</p>
<h2>Quantitativer Vergleich</h2><ul>{''.join(differences)}</ul>
<p>Alle Kurvenmediane oben wurden direkt aus gespeicherten Stichproben mit derselben Quantilmethode nachgerechnet. Die exportierten CSV-Quantile und Mittel wurden unabhängig bis zur vorhandenen Rundungsgenauigkeit kontrolliert. Unterschiede sind deskriptive Befunde; hieraus wird keine neue fachliche Toleranz abgeleitet.</p>
<p>Die Intervalle beschreiben die Parameterunsicherheit des gewichteten Biom-Klimamittelwerts. Die zusätzliche Streuung einer vollständigen Biom-Mischverteilung gehört zu einer anderen Auswerteoption und ist hier nicht enthalten.</p>
<h2>Erhaltene Grenzen</h2><p>{archive_note} Der lokale Originalvergleich über 20.000 Iterationen ist in verification.json je Ausgabefeld dokumentiert. Referenzdateien und vorhandene Tests werden nicht verändert. Der Legacy-Sampler enthält außerdem die bereits dokumentierte fehlerhafte Hastings-Korrektur; die korrigierte Variante verwendet den vorhandenen reparierten Sampler mit denselben Paperankern und festem Dir(0,5)-Prior.</p>
<p>Ergebnisse der ausgewählten bestehenden Tests (keine vollständige Testsuite):</p><ul>{test_summaries}</ul>
{longer_note}
<p>Offen bleiben die Zuordnung der vollständigen Dissertationsfassung, historische Taxon-Transferfunktionen und vollständige Eingaben der Multiproxy-Läufe. Quellen-, Chronologie- und fachliche Freigaben werden nicht ersetzt.</p>
<h2>Nachvollziehen</h2><p>Jeder Lauf enthält <code>config.toml</code>, <code>posterior.npz</code>, <code>provenance.json</code>, <code>run.log</code>, Tabellen und einen ausführlichen Bericht. Provenienz protokolliert Quell-/Eingabe-/Binärhashes, Bibliotheken, Build, Seeds, Status und Speicherbedarf. In <code>verification.json</code> stehen die Ergebnisse der deterministischen Wiederholungen.</p>
</html>"""
    (out / "review.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
