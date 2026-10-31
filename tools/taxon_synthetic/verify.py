"""Recompute saved outputs by direct probability products; no producer import."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "docs/thesis_research/t1_synthetic_t2_2026_10_31"


def verify(output, report):
    cfg = json.loads((PACKAGE / "config.json").read_text())
    datasets = json.loads((output / "observations.json").read_text())
    results = json.loads((output / "results.json").read_text())
    with gzip.open(output / "posterior.csv.gz", "rt") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        matrix = np.array([[float(v) for v in row] for row in reader])
    points, masses = matrix[:, :2], matrix[:, 2:]
    axis = np.linspace(-1, 1, 41)
    np.testing.assert_array_equal(points, [(u, v) for u in axis for v in axis])
    assert len(results) == 108 and len(datasets) == 38
    assert sum(r["dataset"].startswith("known_") for r in results) == 90
    assert sum(r["dataset"].startswith("boundary_") for r in results) == 4
    assert header[2:] == [row["id"] for row in results]
    assert np.isfinite(masses).all() and (masses >= 0).all()
    # Recreate every generated dataset and every fixed mask from the locked plan.
    def regenerate(truth, seed, shared=False, folded=False):
        rng = np.random.default_rng(seed)
        shifts = rng.choice([-cfg["shared_shift"], cfg["shared_shift"]], cfg["replicates"]) if shared else np.zeros(cfg["replicates"])
        u, v = truth
        if folded:
            u = u ** 2 - cfg["folded_offset"]
        a = cfg["logit_slope"] * np.array([u, -u, v, -v])
        occupancy_probability = 1 / (1 + np.exp(-(a[None, :] + shifts[:, None])))
        z = rng.random((cfg["replicates"], 4)) < occupancy_probability
        q = np.where(z, cfg["sensitivity"], cfg["false_positive"])
        return (rng.random((cfg["replicates"], 4)) < q).astype(float)

    planned = {}
    for generator in ("independent", "shared"):
        for i, truth in enumerate(cfg["states"]):
            for seed in cfg["seeds"]:
                planned[f"known_{generator}_{i}_{seed}"] = regenerate(truth, seed, shared=generator == "shared")
    for i, truth in enumerate(cfg["boundary_states"]):
        planned[f"boundary_{i}"] = regenerate(truth, cfg["boundary_seed"])
    basic = planned["known_independent_0_101"]
    planned["conflict"] = np.full_like(basic, np.nan)
    planned["conflict"][:, :2] = 1
    planned["missing_one"] = basic.copy()
    planned["missing_one"][:, 3] = np.nan
    planned["only_u"] = planned["missing_one"].copy()
    planned["only_u"][:, 2] = np.nan
    planned["all_missing"] = np.full_like(basic, np.nan)
    planned["no_detection_information"] = basic
    planned["folded"] = regenerate(cfg["folded_state"], cfg["folded_seed"], folded=True)
    assert set(planned) == set(datasets)
    for name, expected in planned.items():
        saved = np.array([[np.nan if v is None else v for v in row] for row in datasets[name]["y"]])
        np.testing.assert_array_equal(saved, expected)
    max_mass_error = 0.0
    max_summary_error = 0.0
    prior_only = 0
    for k, row in enumerate(results):
        data = datasets[row["dataset"]]
        obs = np.array([[np.nan if v is None else v for v in r] for r in data["y"]])
        assert obs.shape == (40, 4)
        assert (np.isnan(obs) | (obs == 0) | (obs == 1)).all()
        assert row["observed"] == int(np.isfinite(obs).sum())
        u = points[:, 0] if row["family"] == "signed" else points[:, 0] ** 2 - cfg["folded_offset"]
        logits = cfg["logit_slope"] * np.array([u, -u, points[:, 1], -points[:, 1]]).T
        s, f = row["detection_override"] or [cfg["sensitivity"], cfg["false_positive"]]
        shifts = [0] if row["model"] == "independent" else [-cfg["shared_shift"], cfg["shared_shift"]]
        conditional = []
        for h in shifts:
            q = f + (s - f) / (1 + np.exp(-(logits + h)))
            terms = np.ones((len(points), len(obs), 4))
            for j in range(4):
                terms[:, obs[:, j] == 1, j] = q[:, j, None]
                terms[:, obs[:, j] == 0, j] = 1 - q[:, j, None]
            conditional.append(np.prod(terms, axis=2))
        # Average joint row probabilities before multiplying independent rows.
        likelihood = np.prod(np.mean(conditional, axis=0), axis=1)
        prior = np.ones(len(points))
        if row["prior"] == "informative":
            p = cfg["informative_prior"]
            prior = np.exp(-np.sum((points - p["mean"]) ** 2, axis=1) / (2 * p["sd"] ** 2))
        prior /= sum(prior)
        raw = likelihood * prior
        expected = raw / sum(raw)
        actual = masses[:, k]
        error = float(np.max(abs(actual - expected)))
        max_mass_error = max(max_mass_error, error)
        assert error <= cfg["numerical_tolerance"], row["id"]
        assert abs(sum(actual) - 1) < 1e-12
        expected_mean = np.sum(points * expected[:, None], axis=0)
        mean_error = np.max(abs(expected_mean - row["mean"]))
        max_summary_error = max(max_summary_error, float(mean_error))
        assert mean_error < 1e-12
        assert abs(np.log(sum(raw)) - row["log_evidence"]) < 1e-12
        edge = (np.abs(points) == 1).any(axis=1)
        assert abs(sum(expected[edge]) - row["edge_mass"]) < 1e-12
        cutoff = sorted(expected, reverse=True)[np.searchsorted(np.cumsum(sorted(expected, reverse=True)), .95)]
        # Ties can differ by floating rounding in a second implementation.
        assert abs(cutoff - row["hpd_cutoff"]) < 1e-12
        selected = actual >= row["hpd_cutoff"]
        assert int(selected.sum()) == row["hpd_cells"]
        assert abs(sum(actual[selected]) - row["hpd_mass"]) < 1e-12
        assert row["hpd_mass"] >= .95 - 1e-12
        for j in range(2):
            marginal = expected.reshape(41, 41).sum(axis=1 if j == 0 else 0)
            for bound, target in zip(row["marginal_95"][j], [.025, .975]):
                below = marginal[axis < bound].sum()
                through = marginal[axis <= bound].sum()
                assert below <= target + 1e-12 and through >= target - 1e-12
        if data["truth"] is not None:
            idx = np.argmin(np.sum((points - data["truth"]) ** 2, axis=1))
            assert np.max(abs(points[idx] - data["truth"])) < 1e-14
            assert abs(np.linalg.norm(expected_mean - data["truth"]) - row["mean_error"]) < 1e-12
            assert abs(expected[idx] - row["truth_mass"]) < 1e-12
            assert bool(selected[idx]) == row["truth_in_hpd"]
        if row["dataset"] in {"all_missing", "no_detection_information"}:
            assert np.max(abs(actual - prior)) < 1e-12
            assert row["prior_max_error"] < 1e-12
            if row["dataset"] == "all_missing":
                assert abs(row["log_evidence"]) < 1e-12
            prior_only += 1
        if row["dataset"] == "only_u":
            assert np.max(abs(actual.reshape(41, 41).sum(0) - prior.reshape(41, 41).sum(0))) < 1e-12
            assert row["v_prior_max_error"] < 1e-12
        if row["dataset"] == "folded":
            assert np.max(abs(likelihood.reshape(41, 41) - likelihood.reshape(41, 41)[::-1])) < 1e-12
            assert row["likelihood_reflection_max_error"] < 1e-12
            if row["prior"] == "uniform":
                assert row["posterior_reflection_max_error"] < 1e-12
            assert abs(sum(actual[points[:, 0] < 0]) - row["negative_u_mass"]) < 1e-12
            assert abs(sum(actual[points[:, 0] > 0]) - row["positive_u_mass"]) < 1e-12
        if row["dataset"] == "conflict":
            assert abs(row["conflict_tail_upper_bound"] - .25 ** 40) < 1e-35
            assert row["conflict_flag"]
    manifest = json.loads((output / "manifest.json").read_text())
    for name, value in manifest.items():
        path = output / name.split("/", 1)[1] if name.startswith("outputs/") else ROOT / name
        payload = path.read_bytes()
        assert hashlib.sha256(payload).hexdigest() == value["sha256"]
        assert len(payload) == value["bytes"]
    summary = {"status": "passed", "posteriors_checked": len(results), "datasets": len(datasets),
               "grid_states_per_posterior": len(points), "prior_only_cases": prior_only,
               "datasets_recreated_from_locked_plan": len(planned),
               "maximum_mass_absolute_error": max_mass_error,
               "maximum_mean_absolute_error": max_summary_error,
               "manifest_files_checked": len(manifest),
               "method": "direct products and explicit per-row mixture; no producer import",
               "shared_dependencies": ["Python", "NumPy"], "independent_human_review": False,
               "sbc": "not performed"}
    report.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    verify(args.output.resolve(), args.report.resolve())
