"""Compare the separately compiled original and packaged legacy sampler locally.

Requires tools/build_windows.py --original. The archived cross-platform test is
reported separately and is never overwritten or relaxed by this check.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import sysconfig
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".local/legacy_same_runtime_check.json")
    args = parser.parse_args()
    from kinneret_recon import _core, sampler
    from kinneret_recon.config import legacy_config
    from kinneret_recon.data import load_inputs
    from kinneret_recon.model import build_setup

    binary = ROOT / "build/windows" / ("pycpp" + sysconfig.get_config_var("EXT_SUFFIX"))
    spec = importlib.util.spec_from_file_location("pycpp", binary)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    with np.load(ROOT / "tests/data/legacy_short_chain.npz") as ref:
        setup = build_setup(load_inputs(ROOT / "data/in"), legacy_config().sampler)
        model = setup.core_model()
        model.update(tf_sd=ref["sds_tfs"], tf_lower=ref["tfs_lower"], tf_upper=ref["tfs_upper"],
                     spectrum=ref["taxa_spectrum_age"], ap=ref["ap_age"],
                     pann_shape=float(ref["shape"]), pann_rate=float(ref["rate"]))
        n, k, b = int(ref["sample_length"]), setup.num_taxa, setup.num_biomes
        ev, temp, pann, seeds = sampler.legacy_random_stream(int(ref["seed"]), n)
        weights, climate = np.full(k, 1 / k), ref["tfs_upper"]
        tf = [model[key].tolist() for key in ("wts_in_hidden", "wts_hidden_out", "wts_bias_hidden",
                                             "wts_bias_out", "normal_params", "tf_sd", "tf_lower", "tf_upper")]
        proposal = [[model["dirichlet_spread"]], model["jeffreys"].tolist()]
        proposal += [[model[key]] for key in ("beta_shape1", "beta_shape2", "temp_mean", "temp_sd", "pann_shape", "pann_rate")]
        core = [[b], [k], [len(setup.inputs.age)], model["ap"].tolist(),
                model["spectrum"].ravel(order="F").tolist(), model["assign"].ravel(order="F").tolist()]
        prior = [[v] + [0.] * (n - 1) for v in (ev, temp, pann, 0.)]
        started = time.perf_counter()
        raw = original.mcmc_execution([weights.tolist()] + [[] for _ in range(n - 1)],
                                     [climate.tolist()] + [[] for _ in range(n - 1)],
                                     prior, core, proposal, tf, [[n], seeds.tolist()])
        old_seconds = time.perf_counter() - started
        migrated = _core.run_chain_legacy(model, weights, climate, ev, temp, pann, seeds, ref["keep"])
        checks = []
        for index, key in enumerate(("accepted", "tf_sample", "taxa_weights", "expl_variance", "recent_temp", "recent_pann")):
            old = np.asarray(raw[index])
            if key == "taxa_weights":
                old = old.reshape(n, k)
            elif key == "tf_sample":
                old = old.reshape(n, 2 * b)
            if key != "accepted":
                old = old[ref["keep"]]
            new = migrated[key]
            checks.append(dict(field=key, equal=bool(np.array_equal(old, new)),
                               max_abs_difference=float(np.max(np.abs(old - new)))))
        archive_mismatches = np.flatnonzero(migrated["accepted"] != ref["acceptance"])
    report = dict(n_iterations=n, original_runtime_s=old_seconds, fields=checks,
                  comparison="separately_built_unmodified_original_vs_packaged_legacy_same_local_runtime",
                  passed=all(row["equal"] for row in checks),
                  archive_acceptance_mismatches=int(len(archive_mismatches)),
                  archive_first_acceptance_mismatch=int(archive_mismatches[0]) if len(archive_mismatches) else None,
                  original_source_sha256=hashlib.sha256((ROOT / "legacy/src/Pycpp_functions.cpp").read_bytes()).hexdigest(),
                  original_binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
