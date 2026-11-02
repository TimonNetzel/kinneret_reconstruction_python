"""Build the existing pybind11 kernel with a portable LLVM-MinGW compiler.

No system installation, registry or global PATH changes. Use the project venv:
  python tools/build_windows.py --compiler PATH/TO/clang++.exe
The scientific C++ sources and floating-point flags are the same as setup.py.
Compiler installation/download is intentionally separate from this build tool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig
import time

import pybind11


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", required=True, type=Path)
    parser.add_argument("--original", action="store_true", help="Build the unmodified legacy/src/Pycpp_functions.cpp for an independent equivalence check")
    args = parser.parse_args()
    if sys.platform != "win32" or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise SystemExit("This helper targets 64-bit Windows CPython only; use setup.py elsewhere")
    compiler = args.compiler.resolve(strict=True)
    root = Path(__file__).resolve().parents[1]
    build = root / "build/windows"
    build.mkdir(parents=True, exist_ok=True)
    local = root / ".local"
    local.mkdir(exist_ok=True)
    import_library = Path(sys.base_prefix) / "libs" / f"python{sys.version_info.major}{sys.version_info.minor}.lib"
    if not import_library.is_file():
        raise FileNotFoundError(import_library)
    env = os.environ.copy()
    env["PATH"] = str(compiler.parent) + os.pathsep + env.get("PATH", "")
    version = subprocess.run([str(compiler), "--version"], check=True, capture_output=True,
                             text=True, env=env).stdout
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    module_name = "pycpp" if args.original else "_core"
    source = root / ("legacy/src/Pycpp_functions.cpp" if args.original else "cpp/bindings.cpp")
    output = build / f"{module_name}{suffix}"
    # UCRT exposes the existing M_PI/M_E/M_SQRT* macros only with this switch.
    command = [str(compiler), "-std=c++17", "-O3", "-ffp-contract=off", "-DNDEBUG", "-D_USE_MATH_DEFINES",
               "-shared", "-static", "-I" + pybind11.get_include(),
               "-I" + sysconfig.get_path("include"), "-I" + str(root / "cpp"),
               str(source), str(import_library), "-o", str(output)]
    started = time.perf_counter()
    result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
    manifest = dict(compiler_version=version, command=command, python=sys.version,
                    pybind11=pybind11.__version__, exit_code=result.returncode,
                    elapsed_s=time.perf_counter() - started,
                    source_sha256={p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in ([source] if args.original else sorted((root / "cpp").glob("*"))) if p.is_file()})
    log_name = "build-windows-original" if args.original else "build-windows"
    (local / f"{log_name}.log").write_text(version + result.stdout + result.stderr, encoding="utf-8")
    if result.returncode == 0:
        destination = output if args.original else root / "src/kinneret_recon" / output.name
        if destination != output:
            os.replace(output, destination)
        manifest["binary_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
        manifest["extension"] = destination.relative_to(root).as_posix()
    (local / f"{log_name}.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(version, end="", flush=True)
    print(result.stdout + result.stderr, end="", flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    print(f"Built {manifest['extension']} in {manifest['elapsed_s']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
