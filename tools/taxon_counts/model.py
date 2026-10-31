"""Anonymous finite-state count model. No real taxon models or fossil inputs."""
from pathlib import Path
import json
import math

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / "docs/thesis_research/counts_observation_design_2026_10_31"


class ModelConflict(ValueError):
    """No supported state assigns positive probability to the observation."""


def config():
    return json.loads((DESIGN / "synthetic_plan.json").read_text())


def grid(cfg):
    g = cfg["grid"]
    axis = np.linspace(g["lower"], g["upper"], g["points_per_axis"])
    return np.array([(u, v) for u in axis for v in axis])


def vegetation(points, cfg, shift=0.0):
    c = np.asarray(points, dtype=float)
    if c.ndim != 2 or c.shape[1] != 2 or len(c) == 0:
        raise ValueError("expected nonempty N x 2 climate points")
    if (not np.isfinite(c).all() or (c < cfg["grid"]["lower"]).any()
            or (c > cfg["grid"]["upper"]).any()):
        raise ValueError("outside synthetic climate support")
    if not np.isfinite(shift):
        raise ValueError("invalid shared shift")
    a = c @ np.asarray(cfg["response"]["coefficients_u_v"]).T
    a += shift * np.asarray(cfg["response"]["shared_loadings"])
    a -= a.max(axis=1, keepdims=True)
    ex = np.exp(a)
    return ex / ex.sum(axis=1, keepdims=True)


def flux_probabilities(flux, allocation):
    x, a = np.asarray(flux, dtype=float), np.asarray(allocation, dtype=float)
    if x.ndim != 2 or len(x) == 0 or a.ndim != 2 or a.shape[1] != x.shape[1] or a.shape[0] == 0:
        raise ValueError("invalid flux/allocation dimensions")
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("invalid flux")
    if not np.isfinite(a).all() or (a < 0).any() or not np.allclose(a.sum(axis=0), 1, atol=1e-12, rtol=0):
        raise ValueError("allocation must be column stochastic")
    lam = x @ a.T
    total = lam.sum(axis=1, keepdims=True)
    if not np.isfinite(lam).all() or (total <= 0).any():
        raise ValueError("zero or invalid total flux")
    return lam / total


def probabilities(points, cfg, shift=0.0, allocation=None, production=None, detection=None, background=None):
    v = vegetation(points, cfg, shift)
    r = np.asarray(cfg["production"] if production is None else production, dtype=float)
    d = np.asarray(cfg["detection"] if detection is None else detection, dtype=float)
    b = np.asarray(cfg["background"] if background is None else background, dtype=float)
    for values in (r, d, b):
        if values.shape != (v.shape[1],) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("invalid source rates")
    if (d > 1).any():
        raise ValueError("invalid detection rates")
    a = cfg["allocation_base"] if allocation is None else allocation
    return flux_probabilities(d * (r * v + b), a)


def log_multinomial(q, counts):
    q, y = np.asarray(q, dtype=float), np.asarray(counts, dtype=float)
    if q.ndim != 2 or y.ndim != 1 or len(y) == 0 or q.shape[1] != len(y):
        raise ValueError("invalid probability/count dimensions")
    if not np.isfinite(q).all() or (q < 0).any() or not np.allclose(q.sum(1), 1, atol=1e-12, rtol=0):
        raise ValueError("invalid probability simplex")
    if not np.isfinite(y).all() or (y < 0).any() or (y != np.floor(y)).any():
        raise ValueError("counts must be finite nonnegative integers")
    coefficient = math.lgamma(float(y.sum()) + 1) - sum(math.lgamma(float(k) + 1) for k in y)
    positive = y > 0
    with np.errstate(divide="ignore"):
        terms = np.log(q[:, positive]) * y[positive]
    return coefficient + terms.sum(axis=1)


def observation_logpmf(q, counts, total, scheme="complete", selection_documented=False):
    q, y = np.asarray(q, dtype=float), np.asarray(counts, dtype=float)
    if q.ndim != 2 or y.ndim != 1 or q.shape[1] != len(y):
        raise ValueError("invalid observation dimension")
    if total is None or not np.isscalar(total) or not np.isfinite(total) or total < 0 or total != np.floor(total):
        raise ValueError("missing or invalid counting denominator")
    if not (np.isnan(y) | (np.isfinite(y) & (y >= 0) & (y == np.floor(y)))).all():
        raise ValueError("invalid counts")
    # Validate the entire original simplex, even when every observation is missing.
    log_multinomial(q, np.zeros(len(y)))
    observed = ~np.isnan(y)
    if scheme == "complete":
        if not observed.all() or y.sum() != total:
            raise ValueError("incomplete counts or inconsistent total")
        return log_multinomial(q, y)
    if scheme == "marginal":
        remainder = total - y[observed].sum()
        if remainder < 0 or (observed.all() and remainder != 0):
            raise ValueError("observed counts exceed or disagree with denominator")
        if not observed.any():
            return np.zeros(len(q))
        if observed.all():
            return log_multinomial(q, y)
        collapsed = np.column_stack([q[:, observed], q[:, ~observed].sum(axis=1)])
        return log_multinomial(collapsed, np.r_[y[observed], remainder])
    if scheme == "conditional":
        if not selection_documented or not observed.any() or y[observed].sum() != total:
            raise ValueError("conditional subset needs documented selection and selected total")
        selected = q[:, observed]
        denom = selected.sum(axis=1, keepdims=True)
        if (denom <= 0).any():
            raise ValueError("selected group outside conditional support")
        return log_multinomial(selected / denom, y[observed])
    raise ValueError("unknown counting scheme")


def log_likelihood(points, counts, total, cfg, model="base", scheme="complete", selection_documented=False, allocation=None):
    if model == "base":
        q = probabilities(points, cfg, allocation=allocation)
        return observation_logpmf(q, counts, total, scheme, selection_documented)
    if model not in {"sample_shared_mixture", "grain_redrawn_matching_marginals"}:
        raise ValueError("unknown count model")
    settings = cfg["shared"]
    components = [probabilities(points, cfg, h, allocation) for h in settings["shifts"]]
    weights = np.asarray(settings["weights"])
    if model == "grain_redrawn_matching_marginals":
        q = sum(w * p for w, p in zip(weights, components))
        return observation_logpmf(q, counts, total, scheme, selection_documented)
    terms = [np.log(w) + observation_logpmf(p, counts, total, scheme, selection_documented)
             for w, p in zip(weights, components)]
    return np.logaddexp.reduce(terms)


def prior_mass(points, cfg, prior):
    if prior == "uniform":
        mass = np.ones(len(points))
    elif prior == "informative":
        p = cfg["informative_prior"]
        mass = np.exp(-((points - p["mean"]) ** 2).sum(axis=1) / (2 * p["sd"] ** 2))
    else:
        raise ValueError("unknown prior")
    return mass / mass.sum()


def invert(points, counts, total, cfg, prior="uniform", **kwargs):
    ll = log_likelihood(points, counts, total, cfg, **kwargs)
    logmass = np.log(prior_mass(points, cfg, prior)) + ll
    peak = logmass.max()
    if not np.isfinite(peak):
        raise ModelConflict("all supported states impossible; no posterior")
    w = np.exp(logmass - peak)
    return w / w.sum(), float(peak + np.log(w.sum())), ll


def summarize(points, w, cfg, truth=None):
    axis = np.unique(points[:, 0])
    a = w.reshape(len(axis), len(axis))
    tail = (1 - cfg["posterior_mass"]) / 2
    quantiles = [[float(axis[min(np.searchsorted(np.cumsum(m), p), len(axis) - 1)])
                  for p in (tail, 1 - tail)] for m in (a.sum(1), a.sum(0))]
    desc = np.sort(w)[::-1]
    cutoff = desc[min(np.searchsorted(np.cumsum(desc), cfg["posterior_mass"]), len(desc) - 1)]
    hpd = w >= cutoff
    mean = w @ points
    result = {"mean": mean.tolist(), "marginal_95": quantiles, "hpd_cutoff": float(cutoff),
              "hpd_cells": int(hpd.sum()), "hpd_mass": float(w[hpd].sum()),
              "edge_mass": float(w[(np.abs(points) == cfg["grid"]["upper"]).any(axis=1)].sum())}
    if truth is not None:
        i = int(np.argmin(((points - truth) ** 2).sum(axis=1)))
        if not np.allclose(points[i], truth, atol=1e-14, rtol=0):
            raise ValueError("truth must be a supported grid state")
        result.update({"mean_error": float(np.linalg.norm(mean - truth)),
                       "truth_mass": float(w[i]), "truth_in_hpd": bool(hpd[i])})
    return result


def generate(truth, total, seed, cfg, shared=False, allocation=None):
    rng = np.random.default_rng(seed)
    h = float(rng.choice(cfg["shared"]["shifts"], p=cfg["shared"]["weights"])) if shared else 0.0
    p = probabilities([truth], cfg, h, allocation)[0]
    return rng.multinomial(total, p).tolist(), h
