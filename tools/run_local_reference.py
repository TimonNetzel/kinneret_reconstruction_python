"""Run an existing reconstruction config, with local provenance and no overwrites.

Example: .venv/Scripts/python tools/run_local_reference.py --config configs/original.toml
         --output results/local_original --threads 1
The scientific pipeline is imported unchanged. This helper only records input,
source, binary and environment fingerprints, exit status and peak process memory.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def peak_memory():
    if sys.platform != "win32":
        return None
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (key, ctypes.c_size_t) for key in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
            "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    fn = ctypes.WinDLL("psapi").GetProcessMemoryInfo
    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    if not fn(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return None
    return int(counters.PeakWorkingSetSize)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    out = (ROOT / args.output).resolve()
    if ROOT not in out.parents or out.relative_to(ROOT).parts[0] != "results":
        raise SystemExit("Local run output must be a new directory under results/")
    if out.exists():
        raise SystemExit(f"Refusing to overwrite an existing run: {out}")
    if args.threads < 1:
        raise SystemExit("--threads must be positive")
    os.chdir(ROOT)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".local/matplotlib"))
    from kinneret_recon import _core
    from kinneret_recon.config import Config
    from kinneret_recon.pipeline import execute
    cfg = Config.from_toml(ROOT / args.config)
    cfg.output_dir = out.relative_to(ROOT).as_posix()
    cfg.postprocess.n_threads = args.threads
    cfg.validate()
    code_files = [ROOT / "tools" / name for name in ("run_local_reference.py", "verify_local_legacy.py", "build_windows.py")]
    code_files += list((ROOT / "cpp").glob("*")) + list((ROOT / "src/kinneret_recon").glob("*.py"))
    tracked_diff = subprocess.run(["git", "diff", "--binary", "HEAD"], check=True, capture_output=True).stdout
    provenance = dict(started_utc=datetime.now(timezone.utc).isoformat(), status="started",
                      command=[sys.executable, *sys.argv], python=platform.python_version(),
                      packages={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "matplotlib", "pybind11")},
                      git_head=subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(),
                      git_branch=subprocess.run(["git", "branch", "--show-current"], check=True, capture_output=True, text=True).stdout.strip(),
                      tracked_diff_sha256=hashlib.sha256(tracked_diff).hexdigest(),
                      scientific_source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(code_files) if p.is_file()},
                      extension_sha256=sha(_core.__file__), config_source=args.config.as_posix(), config_source_sha256=sha(ROOT / args.config),
                      input_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in sorted((ROOT / cfg.data_dir).glob("*.npz"))},
                      effective_config=cfg.to_dict(), scientific_status="provisional_not_approved")
    build_record = ROOT / ".local/build-windows.json"
    if build_record.exists():
        provenance["compiler_build"] = json.loads(build_record.read_text(encoding="utf-8"))
    out.mkdir(parents=True)
    (out / "local_changes.patch").write_bytes(tracked_diff)
    started = time.perf_counter()
    try:
        with (out / "run.log").open("w", encoding="utf-8", buffering=1) as stream:
            def log(message):
                stream.write(str(message) + "\n")
                print(message, flush=True)
            run = execute(cfg, ROOT, log=log)
        provenance.update(status="completed", exit_code=0, timings_s=run.timings)
    except BaseException:
        provenance.update(status="failed", exit_code=1, error=traceback.format_exc())
        raise
    finally:
        provenance.update(elapsed_s=time.perf_counter() - started, peak_working_set_bytes=peak_memory(),
                          completed_utc=datetime.now(timezone.utc).isoformat())
        provenance["outputs_sha256"] = {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
        (out / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
