# Baut den C++-Kern (pybind11). Alle uebrigen Metadaten stehen in pyproject.toml.
from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

setup(
    ext_modules=[
        Pybind11Extension(
            "kinneret_recon._core",
            ["cpp/bindings.cpp"],
            include_dirs=["cpp"],
            cxx_std=17,
            # wie im Original: -O3, keine FMA-Kontraktion (bitidentische Ergebnisse des Original-Samplers)
            extra_compile_args=["-O3", "-ffp-contract=off"],
        )
    ],
)
