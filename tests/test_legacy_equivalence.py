"""Der Legacy-Modus des Pakets muss das Original (Tag v0.1.0) exakt reproduzieren.

Die Testdaten in tests/data/legacy_short_chain.npz wurden mit dem Originalcode (legacy/src) erzeugt.
"""

from pathlib import Path

import numpy as np
import pytest

from kinneret_recon import sampler
from kinneret_recon.config import legacy_config
from kinneret_recon.data import load_inputs
from kinneret_recon.model import build_setup
from kinneret_recon.postprocess import postprocess

ROOT = Path(__file__).resolve().parents[1]
REF = np.load(ROOT / "tests/data/legacy_short_chain.npz")


@pytest.fixture(scope="module")
def setup():
    return build_setup(load_inputs(ROOT / "data/in"), legacy_config().sampler)


def test_setup_matches_original(setup):
    """Vorbereitung (NumPy) wie im Original. Auf derselben Maschine ist sie bitidentisch; zwischen
    verschiedenen CPUs koennen NumPys SIMD-Pfade die letzte Stelle aendern, daher hier mit Toleranz."""
    tol = dict(rtol=1e-13, atol=0)
    np.testing.assert_allclose(setup.tf_sd, REF["sds_tfs"], **tol)
    np.testing.assert_array_equal(setup.tf_lower, REF["tfs_lower"])  # Gitterwerte: exakt
    np.testing.assert_array_equal(setup.tf_upper, REF["tfs_upper"])
    assert setup.pann_shape == REF["shape"] and setup.pann_rate == REF["rate"]
    np.testing.assert_allclose(setup.taxa_spectrum_age, REF["taxa_spectrum_age"], rtol=1e-13, atol=1e-16)
    np.testing.assert_allclose(setup.ap_age, REF["ap_age"], **tol)
    np.testing.assert_allclose(setup.biome_tf_temp, REF["biome_tf_temp"], rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(setup.biome_tf_pann, REF["biome_tf_pann"], rtol=1e-12, atol=1e-18)


def test_random_stream_matches_original():
    ev, t, p, _ = sampler.legacy_random_stream(int(REF["seed"]), 10)
    assert (ev, t, p) == (REF["init_ev"], REF["init_t"], REF["init_p"])


def test_legacy_chain_bitidentical(setup):
    from kinneret_recon import _core

    n = int(REF["sample_length"])
    ev, t, p, seeds = sampler.legacy_random_stream(int(REF["seed"]), n)
    K = setup.num_taxa
    # Eingaben exakt wie im Originallauf (unabhaengig von NumPys CPU-abhaengigen Rundungen), damit hier
    # nur der C++-Sampler geprueft wird
    model = setup.core_model()
    model.update(tf_sd=REF["sds_tfs"], tf_lower=REF["tfs_lower"], tf_upper=REF["tfs_upper"],
                 spectrum=REF["taxa_spectrum_age"], ap=REF["ap_age"],
                 pann_shape=float(REF["shape"]), pann_rate=float(REF["rate"]))
    r = _core.run_chain_legacy(model, np.full(K, 1 / K), REF["tfs_upper"], ev, t, p, seeds, REF["keep"])
    np.testing.assert_array_equal(r["accepted"], REF["acceptance"])
    for k in ("taxa_weights", "tf_sample", "expl_variance", "recent_temp", "recent_pann"):
        np.testing.assert_array_equal(r[k], REF[k], err_msg=k)


@pytest.mark.slow
def test_full_run_matches_reference(setup):
    """Vollstaendiger Lauf (1 Mio. Iterationen, ~40 s) gegen reference/v1.0-paper. Bitgleichheit gilt nur auf
    Rechnern, deren NumPy dieselben SIMD-Pfade nutzt wie der Referenzrechner (hier: lokal, nicht CI)."""
    cfg = legacy_config()
    post = sampler.run(setup, cfg.sampler)
    ref_post = np.load(ROOT / "reference/v1.0-paper/posterior_thinned.npz")
    np.testing.assert_array_equal(post.taxa_weights, ref_post["taxa_weights"])
    np.testing.assert_array_equal(post.tf_sample, ref_post["tf_sample"])

    res = postprocess(setup, post, cfg.postprocess)
    ref = np.load(ROOT / "reference/v1.0-paper/post_process.npz")
    for v in ("temp", "pann"):
        for d in ("age", "depth"):
            for i, q in enumerate((25, 50, 75)):
                np.testing.assert_array_equal(getattr(res, f"{v}_{d}_q")[i], ref[f"{v}_{d}_{q}"])
            np.testing.assert_allclose(getattr(res, f"{v}_dens_{d}"), ref[f"post_{v}_array_{d}"], rtol=0, atol=1e-12)
    for d in ("age", "depth"):
        for i, q in enumerate((25, 50, 75)):
            np.testing.assert_array_equal(getattr(res, f"biomes_{d}_q")[i], ref[f"biome_{d}_{q}"])
    np.testing.assert_array_equal(res.post_temp_ci_size, ref["post_temp_ci_size"])
    np.testing.assert_allclose(res.post_tf_probs, ref["post_tf_probs"], rtol=0, atol=1e-12)
