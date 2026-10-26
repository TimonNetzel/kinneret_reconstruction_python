"""MCMC-Sampling: Startwerte, Seeds, parallele Ketten (entspricht mcmc_execution.py)."""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import _core
from .config import SamplerConfig
from .model import Setup


@dataclass
class Posterior:
    """Gespeicherte Iterationen aller Ketten. Arrays haben die Form (Kette, Iteration, ...)."""

    iterations: np.ndarray  # (n_keep,) Iterationsindizes der gespeicherten Zustaende
    burn_in: int
    taxa_weights: np.ndarray  # (C, n_keep, K)
    tf_sample: np.ndarray  # (C, n_keep, 2B): T je Biom, dann P je Biom
    expl_variance: np.ndarray  # (C, n_keep) R2 der P-Rekonstruktion mit AP
    recent_temp: np.ndarray  # (C, n_keep)
    recent_pann: np.ndarray  # (C, n_keep)
    log_target: np.ndarray  # (C, n_keep); NaN fuer den Original-Sampler
    alpha: np.ndarray  # (C, n_keep) Konzentration des Dirichlet-Priors der Taxa-Gewichte (konstant, falls fest)
    acceptance_rate: np.ndarray  # (C,) ueber alle Iterationen
    meta: dict = field(default_factory=dict)

    @property
    def n_chains(self) -> int:
        return self.taxa_weights.shape[0]

    @property
    def post_mask(self) -> np.ndarray:
        return self.iterations >= self.burn_in

    def pooled(self, name: str) -> np.ndarray:
        """Alle Ketten nach dem Burn-in aneinandergehaengt: (C * n_post, ...)."""
        a = getattr(self, name)[:, self.post_mask]
        return a.reshape((-1,) + a.shape[2:])

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path, iterations=self.iterations, burn_in=self.burn_in, taxa_weights=self.taxa_weights,
            tf_sample=self.tf_sample, expl_variance=self.expl_variance, recent_temp=self.recent_temp,
            recent_pann=self.recent_pann, log_target=self.log_target, alpha=self.alpha,
            acceptance_rate=self.acceptance_rate,
        )

    @classmethod
    def load(cls, path: str | Path) -> "Posterior":
        d = np.load(path)
        kw = {k: d[k] for k in d.files if k != "burn_in"}
        kw.setdefault("alpha", np.full(kw["log_target"].shape, 0.5))  # aeltere Dateien ohne alpha
        return cls(**kw, burn_in=int(d["burn_in"]))


def legacy_random_stream(seed: int, sample_length: int):
    """Zufallszahlen exakt wie mcmc_preparation.py/mcmc_execution.py des Originals."""
    rnd = random.Random(seed)
    init_expl_variance = rnd.gauss(0.2, 0.0005)
    init_recent_temp = rnd.gauss(4, 0.005)
    init_recent_pann = rnd.gauss(200, 5)
    # Obergrenze durch die trunkierte Normalverteilung im C++-Code vorgegeben
    seed_samples = np.array([round(rnd.uniform(1, 127773)) for _ in range(sample_length)], dtype=np.int32)
    return init_expl_variance, init_recent_temp, init_recent_pann, seed_samples


def legacy_keep_indices(cfg: SamplerConfig) -> np.ndarray:
    """Ausduennung wie post_processing() des Originals."""
    n_post = int((cfg.sample_length - cfg.burn_in) / cfg.thin)
    return np.linspace(cfg.burn_in, cfg.sample_length - 1, n_post, dtype=np.int32).astype(np.int64)


def run_legacy(setup: Setup, cfg: SamplerConfig) -> Posterior:
    ev0, t0, p0, seeds = legacy_random_stream(cfg.seed, cfg.sample_length)
    K = setup.num_taxa
    init_w = np.full(K, 1 / K)
    init_c = np.concatenate([setup.tf_quantiles_temp[:, 2], setup.tf_quantiles_pann[:, 2]])
    keep = legacy_keep_indices(cfg)
    r = _core.run_chain_legacy(setup.core_model(), init_w, init_c, ev0, t0, p0, seeds, keep)
    acc = r["accepted"]
    return Posterior(
        iterations=keep, burn_in=cfg.burn_in,
        taxa_weights=r["taxa_weights"][None], tf_sample=r["tf_sample"][None],
        expl_variance=r["expl_variance"][None], recent_temp=r["recent_temp"][None],
        recent_pann=r["recent_pann"][None], log_target=np.full((1, len(keep)), np.nan),
        alpha=np.full((1, len(keep)), 0.5),
        acceptance_rate=np.array([acc[1:].mean()]),
        meta=dict(kernel="legacy", accepted=acc),
    )


def initial_states(setup: Setup, cfg: SamplerConfig, options: dict, rng: np.random.Generator, n: int):
    """Startwerte je Kette. Gestreut: w ~ Dir(5), c ~ U(untere, obere Grenze), nur Zustaende mit
    endlicher Zielverteilung."""
    K = setup.num_taxa
    model = setup.core_model()
    states = []
    for _ in range(n):
        if not cfg.dispersed_init:
            w = np.full(K, 1 / K)
            c = setup.tf_upper.copy()
        else:
            for _attempt in range(1000):
                w = rng.dirichlet(np.full(K, 5.0))
                c = rng.uniform(setup.tf_lower, setup.tf_upper)
                if np.isfinite(_core.evaluate_state(model, options, w, c)["log_target"]):
                    break
            else:
                raise RuntimeError("keine gueltigen Startwerte gefunden")
        states.append((w, c))
    return states


def run_corrected(setup: Setup, cfg: SamplerConfig, n_threads: int = 0) -> Posterior:
    options = dict(
        dirichlet_hastings_fix=cfg.dirichlet_hastings_fix,
        use_calibration=cfg.use_calibration,
        use_tf_likelihood=cfg.use_tf_likelihood,
        use_taxa_prior=cfg.use_taxa_prior,
        log_gamma_rw=cfg.weights_proposal == "log_gamma",
        adapt_until=cfg.burn_in,
        target_accept=cfg.target_accept,
        sample_alpha=cfg.taxa_prior == "hierarchical",
        alpha_prior_mu=float(np.log(cfg.alpha_prior_median)),
        alpha_prior_sd=cfg.alpha_prior_log_sd,
    )
    ss = np.random.SeedSequence(cfg.seed)
    init_rng = np.random.default_rng(ss.spawn(1)[0])
    chain_seeds = [int(s.generate_state(1, dtype=np.uint64)[0]) for s in ss.spawn(cfg.n_chains)]
    inits = initial_states(setup, cfg, options, init_rng, cfg.n_chains)
    keep = np.arange(0, cfg.sample_length, cfg.thin, dtype=np.int64)
    model = setup.core_model()

    def one(i):
        w, c = inits[i]
        return _core.run_chain(model, options, w, c, cfg.sample_length, keep, chain_seeds[i])

    workers = n_threads or min(cfg.n_chains, _cpu_count())
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(one, range(cfg.n_chains)))

    stack = lambda k: np.stack([r[k] for r in res])
    n_acc = stack("n_accepted")
    n_acc_c = stack("n_accepted_c")
    return Posterior(
        iterations=keep, burn_in=cfg.burn_in,
        taxa_weights=stack("taxa_weights"), tf_sample=stack("tf_sample"),
        expl_variance=stack("expl_variance"), recent_temp=stack("recent_temp"),
        recent_pann=stack("recent_pann"), log_target=stack("log_target"), alpha=stack("alpha"),
        acceptance_rate=n_acc[:, -1] / max(keep[-1], 1),
        meta=dict(kernel="corrected", options=options, chain_seeds=chain_seeds,
                  min_log_weight=[r["min_log_weight"] for r in res], n_accepted=n_acc,
                  **({"acceptance_rate_c": (n_acc_c[:, -1] / max(keep[-1], 1)).tolist(),
                      "final_step_z": [r["final_step_z"] for r in res],
                      "final_tf_scale": [r["final_tf_scale"] for r in res]}
                     if cfg.weights_proposal == "log_gamma" else {})),
    )


def run(setup: Setup, cfg: SamplerConfig, n_threads: int = 0) -> Posterior:
    if cfg.kernel == "legacy":
        return run_legacy(setup, cfg)
    return run_corrected(setup, cfg, n_threads)


def _cpu_count() -> int:
    import os

    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
