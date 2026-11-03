"""Replay one completed posterior while other independent postprocessing runs.

No run files are changed. The record binds results to posterior bytes, inputs,
sampler configuration, scientific sources and the compiled extension.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--posterior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if ROOT not in output.parents or output.relative_to(ROOT).parts[0] not in {".local", "results"} or output.exists():
        raise SystemExit("Use a new local record under .local/ or results/")
    if args.threads < 1:
        raise SystemExit("--threads must be positive")
    from kinneret_recon import _core, sampler
    from kinneret_recon.config import Config
    from kinneret_recon.data import load_inputs
    from kinneret_recon.model import build_setup
    from kinneret_recon.sampler import Posterior
    cfg = Config.from_toml(ROOT / args.config)
    post_path = ROOT / args.posterior
    fingerprint = sha(post_path)
    saved = Posterior.load(post_path)
    source_files = list((ROOT / "cpp").glob("*")) + list((ROOT / "src/kinneret_recon").glob("*.py"))
    report = dict(status="started", tool_sha256=sha(__file__), posterior_sha256=fingerprint,
                  extension_sha256=sha(_core.__file__), sampler_config=cfg.to_dict()["sampler"],
                  source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in source_files if p.is_file()},
                  input_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in sorted((ROOT / cfg.data_dir).glob("*.npz"))},
                  threads=args.threads)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Replaying {cfg.sampler.kernel}: {cfg.sampler.n_chains} x {cfg.sampler.sample_length} with {args.threads} worker(s)", flush=True)
    started = time.perf_counter()
    repeated = sampler.run(build_setup(load_inputs(ROOT / cfg.data_dir), cfg.sampler), cfg.sampler, n_threads=args.threads)
    fields = ("taxa_weights", "tf_sample", "expl_variance", "recent_temp", "recent_pann", "log_target", "alpha", "iterations")
    report.update(status="completed", elapsed_s=time.perf_counter() - started,
                  fields={key: bool(np.array_equal(getattr(saved, key), getattr(repeated, key), equal_nan=True)) for key in fields},
                  posterior_unchanged=sha(post_path) == fingerprint)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "elapsed_s", "fields", "posterior_unchanged")}), flush=True)
    return 0 if report["posterior_unchanged"] and all(report["fields"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
