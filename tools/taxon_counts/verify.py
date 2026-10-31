"""Direct factorial/product audit; does not import either production module."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / "docs/thesis_research/counts_observation_design_2026_10_31"


def compositions(n, k):
    if k == 1:
        yield (n,)
    else:
        for j in range(n + 1):
            for suffix in compositions(n - j, k - 1):
                yield (j,) + suffix


def verify(output, report):
    cfg = json.loads((DESIGN / "synthetic_plan.json").read_text())
    datasets = json.loads((output / "observations.json").read_text())
    result = json.loads((output / "results.json").read_text())
    runs = result["runs"]
    with gzip.open(output / "posterior.csv.gz", "rt") as f:
        reader = csv.reader(f)
        header = next(reader)
        a = np.array([[float(x) for x in row] for row in reader])
    points, saved = a[:, :2], a[:, 2:]
    axis = np.linspace(-1, 1, 21)
    np.testing.assert_array_equal(points, [(u, v) for u in axis for v in axis])
    assert len(runs) == 52 and len(datasets) == 25 and saved.shape == (441, 52)
    assert header[2:] == [r["id"] for r in runs]
    assert np.isfinite(saved).all() and (saved >= 0).all()

    def q(c, h=0.0, allocation=None):
        c = np.asarray(c)
        logits = np.column_stack([2*c[:, 0], -2*c[:, 0], 2*c[:, 1], -2*c[:, 1], np.zeros(len(c))])
        logits += h * np.array([1, 1, -1, -1, 0])
        v = np.exp(logits)
        v /= v.sum(axis=1, keepdims=True)
        allocation = cfg["allocation_base"] if allocation is None else allocation
        flux = (v * cfg["production"] + cfg["background"]) * cfg["detection"]
        lam = flux @ np.asarray(allocation).T
        return lam / lam.sum(axis=1, keepdims=True)

    def direct(p, y, n, scheme):
        observed = np.array([v is not None for v in y])
        vals = [int(v) for v in y if v is not None]
        if scheme == "marginal":
            p = np.column_stack([p[:, observed], p[:, ~observed].sum(1)])
            vals += [n - sum(vals)]
        elif scheme == "conditional":
            p = p[:, observed]
            p /= p.sum(1, keepdims=True)
        coefficient = math.factorial(n) // math.prod(math.factorial(v) for v in vals)
        return coefficient * np.prod(p ** np.array(vals), axis=1)

    def likelihood(data, kind, c=points):
        if kind == "base":
            return direct(q(c, allocation=data["allocation"]), data["counts"], data["total"], data["scheme"])
        components = [q(c, h, data["allocation"]) for h in cfg["shared"]["shifts"]]
        if kind == "sample_shared_mixture":
            return sum(w * direct(p, data["counts"], data["total"], data["scheme"])
                       for w, p in zip(cfg["shared"]["weights"], components))
        assert kind == "grain_redrawn_matching_marginals"
        p = sum(w * p for w, p in zip(cfg["shared"]["weights"], components))
        return direct(p, data["counts"], data["total"], data["scheme"])

    expected = {}
    def add(name, counts, n, truth=None, allocation=None, scheme="complete", documented=False, seed=None, h=None):
        expected[name] = {"counts": counts, "total": n, "truth": truth, "scheme": scheme,
            "allocation": cfg["allocation_base"] if allocation is None else allocation,
            "selection_documented": documented, "seed": seed, "generated_shift": h,
            "denominator_kind": "selected_subset" if scheme == "conditional" else "full_counted_pool"}

    for i, truth in enumerate(cfg["base"]["states"]):
        for n in cfg["base"]["totals"]:
            for seed in cfg["base"]["seeds"]:
                y = np.random.default_rng(seed).multinomial(n, q([truth])[0]).tolist()
                add(f"base_s{i}_N{n}_seed{seed}", y, n, truth, seed=seed, h=0.0)
    s = cfg["shared"]
    rng = np.random.default_rng(s["seed"])
    h = float(rng.choice(s["shifts"], p=s["weights"]))
    y = rng.multinomial(s["total"], q([s["state"]], h)[0]).tolist()
    add("shared", y, s["total"], s["state"], seed=s["seed"], h=h)
    s = cfg["merged"]
    y = np.random.default_rng(s["seed"]).multinomial(s["total"], q([s["state"]], allocation=cfg["allocation_merged"])[0]).tolist()
    add("merged", y, s["total"], s["state"], allocation=cfg["allocation_merged"], seed=s["seed"], h=0.0)
    s = cfg["missing_group"]
    add("missing_group", [v if i in s["observed_indices_zero_based"] else None for i,v in enumerate(s["full_counts"])], s["total"], scheme="marginal")
    s = cfg["conditional_subset"]
    add("conditional_subset", [v if i in s["selected_indices_zero_based"] else None for i,v in enumerate(s["full_counts"])], s["selected_total"], scheme="conditional", documented=True)
    s = cfg["all_missing"]
    add("all_missing", s["counts"], s["known_total"], scheme="marginal")
    for name in ("zero_total", "observed_zero"):
        s = cfg[name]
        add(name, s["counts"], s["total"])
    assert expected == datasets, "saved observations differ from locked plan"
    expected_ids = {f"{name}:{kind}:{prior}" for name in expected
                    for kind in (cfg["shared"]["inference"] if name == "shared" else ["base"])
                    for prior in cfg["priors"]}
    assert {r["id"] for r in runs} == expected_ids
    mass_error, mean_error = 0.0, 0.0
    for i, r in enumerate(runs):
        data = datasets[r["dataset"]]
        p = np.ones(441)
        if r["prior"] == "informative":
            par = cfg["informative_prior"]
            p = np.exp(-np.sum((points - par["mean"])**2, axis=1) / (2 * par["sd"]**2))
        p /= p.sum()
        lik = likelihood(data, r["model"])
        assert np.isfinite(lik).all() and (lik > 0).all(), "unexpected underflow or zero in planned valid cases"
        raw = p * lik
        w = raw / raw.sum()
        mass_error = max(mass_error, float(np.max(abs(w - saved[:, i]))))
        assert np.max(abs(w - saved[:, i])) < 1e-12, r["id"]
        assert abs(saved[:, i].sum() - 1) < 1e-12
        mu = np.sum(w[:, None] * points, axis=0)
        mean_error = max(mean_error, float(np.max(abs(mu - r["mean"]))))
        assert np.max(abs(mu - r["mean"])) < 1e-12
        assert abs(np.log(raw.sum()) - r["log_evidence"]) < 1e-12
        assert r["scheme"] == data["scheme"] and r["total"] == data["total"]
        assert r["denominator_kind"] == data["denominator_kind"]
        assert r["observed_categories"] == sum(v is not None for v in data["counts"])
        assert r["status"] == "computed"
        assert abs(w[(abs(points) == 1).any(axis=1)].sum() - r["edge_mass"]) < 1e-12
        descending = np.sort(w)[::-1]
        cutoff = descending[np.searchsorted(np.cumsum(descending), .95)]
        assert abs(cutoff - r["hpd_cutoff"]) < 1e-12
        chosen = saved[:, i] >= r["hpd_cutoff"]
        assert int(chosen.sum()) == r["hpd_cells"] and r["hpd_mass"] >= .95 - 1e-12
        assert abs(saved[chosen, i].sum() - r["hpd_mass"]) < 1e-12
        for j in range(2):
            marginal = w.reshape(21, 21).sum(axis=1 if j == 0 else 0)
            for bound, target in zip(r["marginal_95"][j], [.025, .975]):
                assert marginal[axis < bound].sum() <= target + 1e-12
                assert marginal[axis <= bound].sum() >= target - 1e-12
        if data["truth"] is not None:
            idx = np.argmin(np.sum((points - data["truth"])**2, axis=1))
            assert np.max(abs(points[idx] - data["truth"])) < 1e-14
            assert abs(np.linalg.norm(mu - data["truth"]) - r["mean_error"]) < 1e-12
            assert abs(w[idx] - r["truth_mass"]) < 1e-12
            assert bool(chosen[idx]) == r["truth_in_hpd"]
        if r["dataset"] in {"all_missing", "zero_total"}:
            assert r["prior_max_error"] < 1e-12 and np.max(abs(saved[:, i] - p)) < 1e-12
            assert abs(r["log_evidence"]) < 1e-12
        if r["dataset"] == "merged":
            assert np.max(abs(lik.reshape(21, 21) - lik.reshape(21, 21)[::-1])) < 1e-12
            assert r["likelihood_reflection_error"] < 1e-12
            if r["prior"] == "uniform":
                assert r["posterior_reflection_error"] < 1e-12
            for side in ("negative", "positive"):
                selected = points[:, 0] < 0 if side == "negative" else points[:, 0] > 0
                assert abs(w[selected].sum() - r[f"{side}_u_mass"]) < 1e-12
    pmfs = []
    for truth in cfg["pmf_enumeration"]["states"]:
        for n in cfg["pmf_enumeration"]["totals"]:
            ys = list(compositions(n, 5))
            for kind in cfg["pmf_enumeration"]["models"]:
                law = "base" if kind == "base" else "sample_shared_mixture"
                value = sum(likelihood({"counts": list(y), "total": n, "allocation": cfg["allocation_base"], "scheme": "complete"}, law, np.array([truth]))[0] for y in ys)
                assert abs(value - 1) < 1e-12
                pmfs.append({"truth": truth, "N": n, "model": law, "vectors": len(ys), "sum": float(value)})
    negative = result["negative_cases"]
    assert len(negative) == 2 and {r["prior"] for r in negative} == set(cfg["priors"])
    for r in negative:
        assert r["case"] == "all_supported_states_impossible"
        assert r["status"] == "not_estimable" and r["posterior_saved"] is False
    impossible = np.asarray(cfg["allocation_base"])
    impossible[0, 0], impossible[1, 0] = 0, 1
    bad = direct(q(points, allocation=impossible), [1, 0, 0, 0, 0], 1, "complete")
    assert (bad == 0).all()
    manifest = json.loads((output / "manifest.json").read_text())
    for name, item in manifest.items():
        path = output / name.split("/", 1)[1] if name.startswith("outputs/") else ROOT / name
        payload = path.read_bytes()
        assert len(payload) == item["bytes"] and hashlib.sha256(payload).hexdigest() == item["sha256"]
    summary = {"status": "passed", "posteriors_checked": 52, "states_per_posterior": 441,
               "datasets_recreated": 25, "explicit_non_estimable_cases": 2,
               "maximum_mass_absolute_error": mass_error, "maximum_mean_absolute_error": mean_error,
               "pmf_enumerations": pmfs, "manifest_files_checked": len(manifest),
               "method": "factorial direct products and exact finite mixtures; no producer import",
               "shared_dependencies": ["Python", "NumPy"], "independent_human_review": False, "sbc": "not performed"}
    report.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k:v for k,v in summary.items() if k != "pmf_enumerations"}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    verify(args.output.resolve(), args.report.resolve())
