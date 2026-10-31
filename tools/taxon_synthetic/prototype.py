"""Finite-state synthetic research model; no real taxon or fossil inputs."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import platform

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "docs/thesis_research/t1_synthetic_t2_2026_10_31"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_config():
    return json.loads((PACKAGE / "config.json").read_text())


def grid(config):
    g = config["grid"]
    axis = np.linspace(g["lower"], g["upper"], g["points_per_axis"])
    return np.array([(u, v) for u in axis for v in axis])


def probabilities(points, config, shift=0.0, family="signed", detection=None):
    c = np.asarray(points, dtype=float)
    if c.ndim != 2 or c.shape[1] != 2 or len(c) == 0:
        raise ValueError("expected nonempty N x 2 climate points")
    if (not np.isfinite(c).all() or (c < config["grid"]["lower"]).any()
            or (c > config["grid"]["upper"]).any()):
        raise ValueError("climate outside fixed synthetic support")
    if family not in {"signed", "folded"}:
        raise ValueError("unknown response family")
    if not np.isfinite(shift):
        raise ValueError("invalid shift")
    u = c[:, 0] if family == "signed" else c[:, 0] ** 2 - config["folded_offset"]
    a = config["logit_slope"] * np.column_stack((u, -u, c[:, 1], -c[:, 1]))
    s, f = detection if detection is not None else (config["sensitivity"], config["false_positive"])
    if not (np.isfinite([s, f]).all() and 0 <= f <= s <= 1):
        raise ValueError("expected 0 <= false positive <= sensitivity <= 1")
    return f + (s - f) / (1.0 + np.exp(-(a + shift)))


def observations(y):
    y = np.asarray(y, dtype=float)
    if y.ndim != 2 or y.shape[1] != 4:
        raise ValueError("expected R x 4 observations")
    if not (np.isnan(y) | (y == 0) | (y == 1)).all():
        raise ValueError("observations must be 0, 1 or NaN (missing)")
    return y


def log_likelihood(points, y, config, model="independent", family="signed", detection=None):
    y = observations(y)
    if model not in {"independent", "shared"}:
        raise ValueError("unknown observation model")
    shifts = [0.0] if model == "independent" else [-config["shared_shift"], config["shared_shift"]]
    components = []
    for shift in shifts:
        q = probabilities(points, config, shift, family, detection)
        # Missing observations contribute exactly zero to the log likelihood.
        with np.errstate(divide="ignore"):
            terms = np.where(y[None, :, :] == 1, np.log(q)[:, None, :],
                             np.where(y[None, :, :] == 0, np.log1p(-q)[:, None, :], 0.0))
        components.append(terms.sum(axis=2))
    per_row = components[0] if model == "independent" else np.logaddexp(*components) - np.log(2.0)
    return per_row.sum(axis=1)


def prior_mass(points, config, prior):
    if prior == "uniform":
        mass = np.ones(len(points))
    elif prior == "informative":
        p = config["informative_prior"]
        mass = np.exp(-((points - p["mean"]) ** 2).sum(axis=1) / (2 * p["sd"] ** 2))
    else:
        raise ValueError("unknown prior")
    return mass / mass.sum()


def invert(points, y, config, prior="uniform", **kwargs):
    base = prior_mass(points, config, prior)
    ll = log_likelihood(points, y, config, **kwargs)
    log_weights = np.log(base) + ll
    peak = log_weights.max()
    if not np.isfinite(peak):
        raise ValueError("observations impossible under every supported state")
    shifted = np.exp(log_weights - peak)
    return shifted / shifted.sum(), float(peak + np.log(shifted.sum())), ll


def summarize(points, weights, config, truth=None):
    mean = weights @ points
    axis = np.unique(points[:, 0])
    m = weights.reshape(len(axis), len(axis))
    tails = (1.0 - config["posterior_mass"]) / 2
    intervals = []
    for marginal in (m.sum(axis=1), m.sum(axis=0)):
        cumulative = np.cumsum(marginal)
        intervals.append([float(axis[min(np.searchsorted(cumulative, q), len(axis) - 1)])
                          for q in (tails, 1 - tails)])
    descending = np.sort(weights)[::-1]
    cutoff = descending[min(np.searchsorted(np.cumsum(descending), config["posterior_mass"]), len(weights) - 1)]
    hpd = weights >= cutoff
    edge = (np.abs(points) == config["grid"]["upper"]).any(axis=1)
    result = {"mean": mean.tolist(), "marginal_95": intervals,
              "hpd_cells": int(hpd.sum()), "hpd_mass": float(weights[hpd].sum()),
              "hpd_cutoff": float(cutoff), "edge_mass": float(weights[edge].sum())}
    if truth is not None:
        index = int(np.argmin(((points - truth) ** 2).sum(axis=1)))
        if not np.allclose(points[index], truth, atol=1e-14, rtol=0):
            raise ValueError("truth must be a supported grid state")
        result.update({"mean_error": float(np.linalg.norm(mean - truth)),
                       "truth_in_hpd": bool(hpd[index]), "truth_mass": float(weights[index])})
    return result


def generate(truth, seed, config, model="independent", family="signed"):
    if model not in {"independent", "shared"}:
        raise ValueError("unknown generator")
    rng = np.random.default_rng(seed)
    n = config["replicates"]
    shifts = np.zeros(n) if model == "independent" else rng.choice([-config["shared_shift"], config["shared_shift"]], n)
    # Explicit Z then Y draws; inversion analytically marginalizes both Z and H.
    p = np.stack([probabilities([truth], config, h, family, (1.0, 0.0))[0] for h in shifts])
    z = rng.random((n, 4)) < p
    chance = np.where(z, config["sensitivity"], config["false_positive"])
    return (rng.random((n, 4)) < chance).astype(float)


def run(output):
    config = load_config()
    output.mkdir(parents=True, exist_ok=True)
    points = grid(config)
    datasets, runs, masses = {}, [], []

    def add(name, y, truth, generator, models, family="signed", detection=None):
        datasets[name] = {"truth": truth, "generator": generator, "family": family,
                          "y": [[None if np.isnan(v) else int(v) for v in row] for row in y]}
        for model in models:
            for prior in config["priors"]:
                weights, log_evidence, ll = invert(points, y, config, prior, model=model,
                                                   family=family, detection=detection)
                result = {"id": f"{name}:{model}:{prior}", "dataset": name, "model": model,
                          "prior": prior, "family": family, "detection_override": detection,
                          "observed": int(np.isfinite(y).sum()), "log_evidence": log_evidence,
                          **summarize(points, weights, config, truth)}
                if name == "conflict":
                    q = probabilities(points, config)
                    tail = float(np.max(q[:, 0] * q[:, 1]) ** len(y))
                    result.update({"conflict_tail_upper_bound": tail,
                                   "conflict_flag": tail < config["conflict_tail_threshold"]})
                if name in {"all_missing", "no_detection_information"}:
                    result["prior_max_error"] = float(np.max(np.abs(weights - prior_mass(points, config, prior))))
                if name == "only_u":
                    shape = (config["grid"]["points_per_axis"],) * 2
                    p0 = prior_mass(points, config, prior).reshape(shape).sum(axis=0)
                    result["v_prior_max_error"] = float(np.max(np.abs(weights.reshape(shape).sum(axis=0) - p0)))
                if name == "folded":
                    shape = (config["grid"]["points_per_axis"],) * 2
                    result["likelihood_reflection_max_error"] = float(np.max(np.abs(ll.reshape(shape) - ll.reshape(shape)[::-1])))
                    result["posterior_reflection_max_error"] = float(np.max(np.abs(weights.reshape(shape) - weights.reshape(shape)[::-1])))
                    result["negative_u_mass"] = float(weights[points[:, 0] < 0].sum())
                    result["positive_u_mass"] = float(weights[points[:, 0] > 0].sum())
                runs.append(result)
                masses.append(weights)

    for generator, models in [("independent", ["independent"]), ("shared", ["shared", "independent"])]:
        for i, truth in enumerate(config["states"]):
            for seed in config["seeds"]:
                add(f"known_{generator}_{i}_{seed}", generate(truth, seed, config, generator), truth, generator, models)
    for i, truth in enumerate(config["boundary_states"]):
        add(f"boundary_{i}", generate(truth, config["boundary_seed"], config), truth, "independent", ["independent"])
    basic = generate(config["states"][0], config["seeds"][0], config)
    y = np.full_like(basic, np.nan)
    y[:, :2] = 1
    add("conflict", y, None, "constructed_inconsistent", ["independent"])
    y = basic.copy()
    y[:, 3] = np.nan
    add("missing_one", y, config["states"][0], "independent_fixed_mask", ["independent"])
    y[:, 2] = np.nan
    add("only_u", y, config["states"][0], "independent_fixed_mask", ["independent"])
    add("all_missing", np.full_like(basic, np.nan), None, "fixed_missing", ["independent", "shared"])
    f = config["uninformative_detection"]
    add("no_detection_information", basic, None, "fixed_data", ["independent"], detection=[f, f])
    add("folded", generate(config["folded_state"], config["folded_seed"], config, family="folded"),
        config["folded_state"], "independent_folded", ["independent"], family="folded")
    write_json(output / "observations.json", datasets)
    write_json(output / "results.json", runs)
    # Fixed gzip timestamp/name and fixed decimal round trips permit byte comparison.
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["u", "v"] + [r["id"] for r in runs])
    for i, point in enumerate(points):
        writer.writerow([format(v, ".17g") for v in point] + [format(w[i], ".17g") for w in masses])
    (output / "posterior.csv.gz").write_bytes(gzip.compress(buffer.getvalue().encode(), mtime=0))
    write_json(output / "environment.json", {"python": platform.python_version(), "numpy": np.__version__,
               "design_commit": "2b41c377af45035ec8447bd2b2b435bf55121aac",
               "sbc": "not performed", "real_inputs_loaded": False})
    sources = [PACKAGE / "config.json", PACKAGE / "DESIGN.md", Path(__file__)]
    files = {str(p.relative_to(ROOT)): p for p in sources}
    files.update({f"outputs/{name}": output / name for name in
                  ["observations.json", "results.json", "posterior.csv.gz", "environment.json"]})
    write_json(output / "manifest.json", {name:
               {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
               for name, p in files.items()})
    print(f"Saved {len(runs)} posteriors from {len(datasets)} datasets in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output.resolve())
