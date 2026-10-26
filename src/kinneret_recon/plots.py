"""Abbildungen (Abbildungen des Originals plus Konvergenzdiagnostik und Laufvergleich)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.interpolate import splev, splrep  # noqa: E402
from scipy.stats import beta, gamma, gaussian_kde, norm  # noqa: E402

from .model import Setup  # noqa: E402
from .postprocess import Results  # noqa: E402
from .sampler import Posterior  # noqa: E402

BIOME_NAMES = ["Mediterranean biome", "Irano-Turanian biome", "Saharo-Arabian biome"]
BIOME_LABELS = ["Med", "IT", "SA"]
BIOME_COLS = ["#A2CD5A", "#EE8262", "#FFB90F"]
T_LABEL = r"$\mathregular{T_{DJF}}$ [°C]"
P_LABEL = r"$\mathregular{P_{ANN}}$ [mm]"


def _save(fig, outdir: Path, name: str, formats=("pdf", "png")) -> list[Path]:
    paths = []
    for fmt in formats:
        p = outdir / f"{name}.{fmt}"
        fig.savefig(p, dpi=150)
        paths.append(p)
    plt.close(fig)
    return paths


def terrain_cmap(setup: Setup) -> ListedColormap:
    return ListedColormap(setup.inputs.terrain_colors_rgb.T)


def _nearest(template, value):
    return template[np.abs(template - value).argmin()]


def plot_reconstruction(setup: Setup, res: Results, outdir: Path):
    inp = setup.inputs
    age, dims, tr = res.age, len(inp.temp_range), inp.temp_range
    cmap = terrain_cmap(setup)
    t_q, p_q = res.temp_age_q, res.pann_age_q

    lo = _nearest(np.arange(-30, 31, 5), t_q[1].min() - 5)
    hi = _nearest(np.arange(-30, 31, 5), t_q[1].max() + 5)
    lo_id, hi_id = np.abs(tr - lo).argmin(), np.abs(tr - hi).argmin()
    t_img = res.temp_dens_age.T[::-1][dims - hi_id:dims - lo_id]

    p_hi = _nearest(np.arange(0, 4001, 200), round(p_q[1].max() + 300))
    plo_id, phi_id = np.abs(res.pann_grid - 0).argmin(), np.abs(res.pann_grid - p_hi).argmin()
    p_img = res.pann_dens_age.T[::-1][dims - phi_id:dims - plo_id]

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    for ax, img, ext, q, title, ylab in (
        (axs[0], t_img, [tr[lo_id], tr[hi_id]], t_q, "(a) Temperature", T_LABEL),
        (axs[1], p_img, [res.pann_grid[plo_id], res.pann_grid[phi_id]], p_q, "(b) Precipitation", P_LABEL),
    ):
        im = ax.imshow(img, cmap=cmap, extent=[age.min(), age.max(), *ext], interpolation="none", aspect="auto")
        fig.colorbar(im, ax=ax)
        ax.plot(age, q[0], color="black", lw=1, ls="dashed")
        ax.plot(age, q[1], color="black", lw=2)
        ax.plot(age, q[2], color="black", lw=1, ls="dashed")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylab)
    axs[1].set_xlabel("Age [cal a BP]")
    fig.tight_layout()
    return _save(fig, outdir, "reconstruction")


def plot_biomes(setup: Setup, res: Results, outdir: Path):
    age, cmap = res.age, terrain_cmap(setup)
    B = setup.num_biomes
    fig, axs = plt.subplots(B, 1, figsize=(10, 8))
    for b in range(B):
        ax = axs[b]
        im = ax.imshow(res.biomes_dens_age[b].T[::-1], cmap=cmap, extent=[age.min(), age.max(), 0, 1],
                       interpolation="none", aspect="auto")
        fig.colorbar(im, ax=ax)
        ax.plot(age, res.biomes_age_q[0, b], color="black", lw=1, ls="dashed")
        ax.plot(age, res.biomes_age_q[1, b], color="black", lw=2)
        ax.plot(age, res.biomes_age_q[2, b], color="black", lw=1, ls="dashed")
        ax.set_title(f"({'abc'[b]}) {BIOME_NAMES[b]}", loc="left", fontweight="bold")
    axs[-1].set_xlabel("Age [cal a BP]")
    fig.tight_layout()
    return _save(fig, outdir, "biomes")


def plot_taxa_weights(setup: Setup, res: Results, outdir: Path):
    inp = setup.inputs
    biome_id = np.argmax(inp.biomes_assign, axis=1)
    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(left=0.3, right=0.95, top=0.9, bottom=0.05)
    bp = plt.boxplot(res.post_weights, orientation="horizontal", tick_labels=inp.taxa_names, notch=True,
                     patch_artist=True, medianprops=dict(color="black"), whis=None)
    xmax = max(0.25, float(np.quantile(res.post_weights, 0.99)))
    plt.xlim([0, xmax])
    plt.title("Posterior and prior taxa weights", loc="left", fontweight="bold")
    for patch, b in zip(bp["boxes"], biome_id):
        patch.set_facecolor(BIOME_COLS[b])
    plt.axvline(x=1 / setup.num_taxa, color="black", zorder=3)
    return _save(fig, outdir, "taxa_weights")


def plot_transfer_functions(setup: Setup, res: Results, outdir: Path):
    inp = setup.inputs
    dims, tr, pr = len(inp.temp_range), inp.temp_range, inp.pann_range
    # nichtlineare P-Achse auf eine lineare Achse abbilden (wie im Original)
    spl = splrep(pr, np.linspace(0, dims, dims))
    x2 = np.linspace(pr.min(), pr.max(), dims)
    ids = np.maximum(np.round(splev(x2, spl)).astype(int) - 1, 0)
    legend = [Line2D([], [], color="k", ls="dashed", label="Prior"), Line2D([], [], color="k", label="Posterior")]
    fig, axs = plt.subplots(1, setup.num_biomes, figsize=(15, 5))
    for b, ax in enumerate(axs):
        prior = setup.biome_given_climate[b][ids, :]
        c1 = ax.contour(tr, x2, prior, levels=[0.5], colors="k", linestyles="dashed")
        c2 = ax.contour(tr, x2, res.post_tf_probs[b], levels=[0.5], colors="k")
        ax.clabel(c1, inline=0, fontsize=11, fmt="%1.1f")
        ax.clabel(c2, inline=0, fontsize=11, fmt="%1.1f")
        ax.set_xlim(-10, 30)
        ax.set_ylim(0, 1500)
        ax.set_xlabel(T_LABEL)
        ax.set_title(f"({'abc'[b]}) {BIOME_NAMES[b]} probabilites", loc="left", fontweight="bold")
        ax.legend(handles=legend)
    axs[0].set_ylabel(P_LABEL)
    fig.tight_layout()
    return _save(fig, outdir, "transfer_functions_2D_dists")


def plot_tf_ratio(setup: Setup, res: Results, outdir: Path):
    rt = res.post_temp_ci_size / res.prior_temp_ci_size
    rp = res.post_pann_ci_size / res.prior_pann_ci_size
    y = np.arange(setup.num_biomes)
    h = 0.35
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.subplots_adjust(left=0.25, right=0.95, top=0.9, bottom=0.05)
    ax.barh(y, rt, height=h, color="orange", label=r"$\mathregular{T_{DJF}}$")
    ax.barh(y + h, rp, height=h, color="lightblue", label=r"$\mathregular{P_{ANN}}$")
    ax.set_yticks(y + h / 2)
    ax.set_yticklabels(BIOME_NAMES[: setup.num_biomes])
    ax.set_xlim([0, max(rt.max(), rp.max()) + 0.2])
    ax.legend(loc="lower right")
    ax.set_title("Ratio of posterior to prior CI sizes", loc="left", fontweight="bold")
    return _save(fig, outdir, "transfer_functions_ratio")


def plot_calibration(setup: Setup, post: Posterior, outdir: Path):
    """Posterior und Prior der Kalibrierungsgroessen (independent_proposal_dists im Original)."""
    inp = setup.inputs
    dims, tr, pr, rc = len(inp.temp_range), inp.temp_range, inp.pann_range, inp.recent_climate
    panels = [
        (post.pooled("recent_temp"), np.linspace(tr.min(), tr.max(), dims),
         lambda x: norm.pdf(x, rc[0], rc[1]), np.linspace(tr.min(), tr.max(), 1000), (0, 20), T_LABEL,
         "(a) Temperature"),
        (post.pooled("recent_pann"), np.linspace(pr.min(), pr.max(), dims),
         lambda x: gamma.pdf(x, setup.pann_shape, scale=1 / setup.pann_rate), np.linspace(pr.min(), pr.max(), 1000),
         (0, 1000), P_LABEL, "(b) Precipitation"),
        (post.pooled("expl_variance"), np.linspace(0, 1, dims),
         lambda x: beta.pdf(x, setup.beta_shape1, setup.beta_shape2), np.linspace(0, 1, 1000), (0, 1),
         r"$\mathregular{R^2}$", "(c) Arboreal pollen"),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(12, 5))
    fig.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.10)
    for ax, (s, xs, prior, px, xlim, xlab, title) in zip(axs, panels):
        d = gaussian_kde(s).evaluate(xs)
        d /= d.max()
        pd = prior(px)
        ax.plot(xs, d, color="black")
        ax.fill_between(np.hstack([xs.min(), xs]), np.hstack([0, d]), color="lightgrey")
        ax.plot(px, pd / pd.max(), color="orange")
        ax.set_xlabel(xlab)
        ax.set_xlim(*xlim)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_title(title, loc="left", fontweight="bold")
    axs[0].set_ylabel("Normalized density")
    axs[0].legend(("Posterior", "Prior"))
    return _save(fig, outdir, "independent_proposal_dists")


def plot_age_depth(setup: Setup, outdir: Path):
    inp = setup.inputs
    fig, axs = plt.subplots(2, 1, figsize=(11, 12))
    fig.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.10)
    axs[0].plot(inp.depth, inp.ap_depth, color="black")
    axs[0].fill_between(np.hstack([inp.depth.min(), inp.depth]), np.hstack([0, inp.ap_depth]), color="lightgrey")
    axs[0].set_ylabel("%")
    axs[0].set_xlabel("Depth [cm]")
    axs[0].set_xlim(inp.depth.min(), inp.depth.max())
    axs[0].set_ylim(0, 100)
    axs[0].set_title("(a) Arboreal pollen from Lake Kinneret with respect to depth", loc="left", fontweight="bold")
    axs[1].plot(inp.age, setup.ap_age, color="black")
    axs[1].fill_between(np.hstack([inp.age.min(), inp.age]), np.hstack([0, setup.ap_age]), color="lightgrey")
    axs[1].plot(inp.age_mean, inp.ap_depth, color="orange", lw=2)
    axs[1].set_ylabel("%")
    axs[1].set_xlabel("Age [cal a BP]")
    axs[1].set_xlim(inp.age.min(), inp.age.max())
    axs[1].set_ylim(0, 100)
    axs[1].legend(("New age-depth-transformation", "Old age-depth-transformation"))
    axs[1].set_title("(b) As (a), but with respect to age", loc="left", fontweight="bold")
    return _save(fig, outdir, "age_depth_transform")


def plot_traces(post: Posterior, outdir: Path, biome_labels=BIOME_LABELS):
    """Spuren aller Ketten (inkl. Burn-in) fuer die wichtigsten Groessen."""
    B = len(biome_labels)
    series = [("log target", post.log_target), (r"$R^2$", post.expl_variance), ("T0 [°C]", post.recent_temp),
              ("P0 [mm]", post.recent_pann)]
    if np.ptp(post.alpha) > 0:
        series.append((r"$\alpha$ (Dir-Prior)", post.alpha))
    series += [(f"T {lab} [°C]", post.tf_sample[:, :, b]) for b, lab in enumerate(biome_labels)]
    series += [(f"P {lab} [mm]", post.tf_sample[:, :, B + b]) for b, lab in enumerate(biome_labels)]
    series = [s for s in series if np.all(np.isfinite(s[1]))]
    n = len(series)
    fig, axs = plt.subplots(n, 1, figsize=(11, 1.6 * n), sharex=True)
    it = post.iterations
    for ax, (lab, x) in zip(np.atleast_1d(axs), series):
        for c in range(x.shape[0]):
            ax.plot(it, x[c], lw=0.4, alpha=0.8)
        ax.axvline(post.burn_in, color="k", ls=":", lw=1)
        ax.set_ylabel(lab, fontsize=8)
        ax.tick_params(labelsize=7)
    np.atleast_1d(axs)[-1].set_xlabel("Iteration")
    fig.suptitle("Spuren der Ketten (gepunktet: Ende des Burn-in)", fontsize=10)
    fig.tight_layout()
    return _save(fig, outdir, "diagnostics_traces")


def plot_rhat(summary: list[dict], outdir: Path):
    names = [r["parameter"] for r in summary]
    rh = np.array([r["rhat"] for r in summary])
    es = np.array([r["ess_bulk"] for r in summary])
    fig, axs = plt.subplots(1, 2, figsize=(11, max(4, 0.18 * len(names))), sharey=True)
    y = np.arange(len(names))
    axs[0].barh(y, rh, color=np.where(rh > 1.01, "#d62728", "#7f7f7f"))
    axs[0].axvline(1.01, color="k", ls="--", lw=1)
    axs[0].set_xlim(min(0.99, np.nanmin(rh) - 0.005), max(1.05, np.nanmax(rh) + 0.01))
    axs[0].set_xlabel("R-hat (Grenze 1,01)")
    axs[1].barh(y, es, color=np.where(es < 400, "#d62728", "#7f7f7f"))
    axs[1].axvline(400, color="k", ls="--", lw=1)
    axs[1].set_xscale("log")
    axs[1].set_xlabel("ESS bulk (Grenze 400)")
    axs[0].set_yticks(y)
    axs[0].set_yticklabels(names, fontsize=6)
    axs[0].invert_yaxis()
    fig.tight_layout()
    return _save(fig, outdir, "diagnostics_rhat_ess")


def plot_comparison(runs: dict[str, Results], outdir: Path, name: str = "comparison"):
    """Median und 50-%-Intervall der Rekonstruktion mehrerer Laeufe uebereinander."""
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # erster Lauf (Referenz): graue Flaeche (25-75 %) und schwarze Medianlinie; uebrige Laeufe: Median
    # durchgezogen, Quartile gestrichelt
    for i, (lab, r) in enumerate(runs.items()):
        for ax, q in ((axs[0], r.temp_age_q), (axs[1], r.pann_age_q)):
            if i == 0:
                ax.fill_between(r.age, q[0], q[2], color="0.85", lw=0, label=f"{lab} (25–75 %)")
                ax.plot(r.age, q[1], color="black", lw=1.6, label=f"{lab} (Median)")
            else:
                col = cols[(i - 1) % len(cols)]
                ax.plot(r.age, q[1], color=col, lw=1.3, label=lab)
                ax.plot(r.age, q[0], color=col, lw=0.7, ls="--")
                ax.plot(r.age, q[2], color=col, lw=0.7, ls="--")
    axs[0].set_ylabel(T_LABEL)
    axs[1].set_ylabel(P_LABEL)
    axs[1].set_xlabel("Age [cal a BP]")
    axs[0].set_title("(a) Temperature", loc="left", fontweight="bold")
    axs[1].set_title("(b) Precipitation", loc="left", fontweight="bold")
    axs[0].legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    return _save(fig, outdir, name)


def plot_weights_comparison(runs: dict[str, Results], taxa_names, outdir: Path, name: str = "comparison_weights"):
    labels = list(runs)
    K = len(taxa_names)
    fig, ax = plt.subplots(figsize=(8, 9))
    h = 0.8 / len(labels)
    for i, lab in enumerate(labels):
        w = runs[lab].post_weights
        med = np.median(w, axis=0)
        lo, hi = np.quantile(w, [0.25, 0.75], axis=0)
        y = np.arange(K) + i * h
        ax.barh(y, med, height=h, label=lab, xerr=[med - lo, hi - med], error_kw=dict(lw=0.6))
    ax.axvline(1 / K, color="k", lw=1, label="1/K")
    ax.axvline(beta(0.5, 0.5 * (K - 1)).median(), color="k", lw=1, ls="--", label="Prior-Median Dir(0,5)")
    ax.set_yticks(np.arange(K) + 0.4 - h / 2)
    ax.set_yticklabels(taxa_names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Taxon-Gewicht (Median, 50-%-Intervall)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, outdir, name)


def all_figures(setup: Setup, post: Posterior, res: Results, summary, outdir: Path) -> dict[str, list[Path]]:
    outdir.mkdir(parents=True, exist_ok=True)
    return {
        "reconstruction": plot_reconstruction(setup, res, outdir),
        "biomes": plot_biomes(setup, res, outdir),
        "taxa_weights": plot_taxa_weights(setup, res, outdir),
        "transfer_functions_2D_dists": plot_transfer_functions(setup, res, outdir),
        "transfer_functions_ratio": plot_tf_ratio(setup, res, outdir),
        "independent_proposal_dists": plot_calibration(setup, post, outdir),
        "age_depth_transform": plot_age_depth(setup, outdir),
        "diagnostics_traces": plot_traces(post, outdir),
        "diagnostics_rhat_ess": plot_rhat(summary, outdir),
    }
