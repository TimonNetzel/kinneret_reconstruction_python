"""Execute precisely the saved anonymous 52-case counts plan."""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import platform

import numpy as np

import model

PACKAGE = model.ROOT / "docs/thesis_research/counts_synthetic_t2_2026_10_31"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(output):
    cfg = model.config()
    output.mkdir(parents=True, exist_ok=True)
    points = model.grid(cfg)
    datasets, rows, masses = {}, [], []

    def add(name, counts, total, truth=None, models=("base",), scheme="complete", allocation=None,
            selection_documented=False, seed=None, generated_shift=None):
        data = {"counts": counts, "total": total, "truth": truth, "scheme": scheme,
                "allocation": cfg["allocation_base"] if allocation is None else allocation,
                "selection_documented": selection_documented, "seed": seed,
                "generated_shift": generated_shift,
                "denominator_kind": "selected_subset" if scheme == "conditional" else "full_counted_pool"}
        datasets[name] = data
        for kind in models:
            for prior in cfg["priors"]:
                w, evidence, ll = model.invert(points, counts, total, cfg, prior, model=kind, scheme=scheme,
                        allocation=allocation, selection_documented=selection_documented)
                r = {"id": f"{name}:{kind}:{prior}", "dataset": name, "model": kind, "prior": prior,
                     "status": "computed", "scheme": scheme, "total": total,
                     "denominator_kind": data["denominator_kind"],
                     "observed_categories": sum(x is not None for x in counts),
                     "log_evidence": evidence, **model.summarize(points, w, cfg, truth)}
                if name in {"all_missing", "zero_total"}:
                    r["prior_max_error"] = float(np.max(abs(w - model.prior_mass(points, cfg, prior))))
                if name == "merged":
                    shape = (cfg["grid"]["points_per_axis"],) * 2
                    r["likelihood_reflection_error"] = float(np.max(abs(ll.reshape(shape) - ll.reshape(shape)[::-1])))
                    r["posterior_reflection_error"] = float(np.max(abs(w.reshape(shape) - w.reshape(shape)[::-1])))
                    r["negative_u_mass"] = float(w[points[:, 0] < 0].sum())
                    r["positive_u_mass"] = float(w[points[:, 0] > 0].sum())
                rows.append(r)
                masses.append(w)

    for i, truth in enumerate(cfg["base"]["states"]):
        for n in cfg["base"]["totals"]:
            for seed in cfg["base"]["seeds"]:
                y, h = model.generate(truth, n, seed, cfg)
                add(f"base_s{i}_N{n}_seed{seed}", y, n, truth, seed=seed, generated_shift=h)
    s = cfg["shared"]
    y, h = model.generate(s["state"], s["total"], s["seed"], cfg, shared=True)
    add("shared", y, s["total"], s["state"], models=s["inference"], seed=s["seed"], generated_shift=h)
    s = cfg["merged"]
    y, h = model.generate(s["state"], s["total"], s["seed"], cfg, allocation=cfg["allocation_merged"])
    add("merged", y, s["total"], s["state"], allocation=cfg["allocation_merged"], seed=s["seed"], generated_shift=h)
    s = cfg["missing_group"]
    y = [v if i in s["observed_indices_zero_based"] else None for i, v in enumerate(s["full_counts"])]
    add("missing_group", y, s["total"], scheme="marginal")
    s = cfg["conditional_subset"]
    y = [v if i in s["selected_indices_zero_based"] else None for i, v in enumerate(s["full_counts"])]
    add("conditional_subset", y, s["selected_total"], scheme="conditional", selection_documented=True)
    s = cfg["all_missing"]
    add("all_missing", s["counts"], s["known_total"], scheme="marginal")
    for name in ("zero_total", "observed_zero"):
        s = cfg[name]
        add(name, s["counts"], s["total"])
    negative = []
    allocation = np.asarray(cfg["allocation_base"], dtype=float)
    allocation[0, 0], allocation[1, 0] = 0, 1
    for prior in cfg["priors"]:
        try:
            model.invert(points, [1, 0, 0, 0, 0], 1, cfg, prior, allocation=allocation)
        except model.ModelConflict as e:
            negative.append({"case": "all_supported_states_impossible", "prior": prior,
                             "status": "not_estimable", "reason": str(e), "posterior_saved": False})
        else:
            raise AssertionError("impossible case produced a posterior")
    assert len(rows) == cfg["valid_posteriors_planned"]
    write_json(output / "observations.json", datasets)
    write_json(output / "results.json", {"runs": rows, "negative_cases": negative})
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["u", "v"] + [r["id"] for r in rows])
    for i, point in enumerate(points):
        writer.writerow([format(x, ".17g") for x in point] + [format(w[i], ".17g") for w in masses])
    (output / "posterior.csv.gz").write_bytes(gzip.compress(buf.getvalue().encode(), mtime=0))
    write_json(output / "environment.json", {"python": platform.python_version(), "numpy": np.__version__,
               "base_commit": "19a75426320d0143c09cd00c756bc31f97cef0d0",
               "design_content_commit": "62e0ee6272323246f6d29eff5f390a12136f116b",
               "real_inputs_loaded": False, "sbc": "not performed"})
    sources = [model.DESIGN / n for n in ("DESIGN_COUNTS.md", "SYNTHETIC_PLAN.md", "synthetic_plan.json")]
    sources += [Path(__file__), Path(model.__file__)]
    files = {str(p.relative_to(model.ROOT)): p for p in sources}
    files.update({f"outputs/{n}": output / n for n in ("observations.json", "results.json", "posterior.csv.gz", "environment.json")})
    write_json(output / "manifest.json", {n: {"bytes": p.stat().st_size,
               "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for n, p in files.items()})
    print(f"Saved {len(rows)} posteriors, {len(datasets)} datasets and {len(negative)} explicit non-estimable cases")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output.resolve())
