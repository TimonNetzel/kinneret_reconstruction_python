"""Lauf-Katalog, Faktor-Register und App-Daten."""

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from kinneret_recon.app_daten import LaufDaten, kennzahlen, schreibe_json, standort_daten, vergleich, zusammenfassung
from kinneret_recon.config import Config, legacy_config
from kinneret_recon.katalog import (FAKTOREN, STANDORTE, Lauf, flache_einstellungen, lade_katalog, pruefe_katalog,
                                    reihenfolge, stufe_von)
from kinneret_recon.pipeline import execute, write_toml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def katalog():
    return lade_katalog(ROOT)


def test_katalog_fehlerfrei(katalog):
    pr = pruefe_katalog(katalog, ROOT)
    assert len(katalog) >= 27
    assert {i: p.fehler for i, p in pr.items() if p.fehler} == {}
    # gleiche Ketten, nur andere Auswertung
    assert pr["standard_aktuell"].fingerabdruck_ketten == pr["n_mixture_standard"].fingerabdruck_ketten
    assert pr["standard_v3"].fingerabdruck_ketten == pr["v3_mixture"].fingerabdruck_ketten
    assert pr["standard_aktuell"].fingerabdruck != pr["n_mixture_standard"].fingerabdruck
    order = reihenfolge(katalog)
    for i, lauf in katalog.items():
        if lauf.basis:
            assert order.index(lauf.basis) < order.index(i)


def test_stufen_der_standards(katalog):
    pr = pruefe_katalog(katalog, ROOT)
    for lid, erwartet in (("standard_aktuell", {"gewichte_prior": "dir_05", "klima_prior": "paper", "r2_prior": "r2_05",
                                                "standardversion": "v2", "korrekturen": "standard"}),
                          ("standard_v3", {"gewichte_prior": "hierarchisch", "klima_prior": "cru_50",
                                           "standardversion": "v3", "rekonstruktion": "mean"}),
                          ("legacy", {"korrekturen": "original", "klima_prior": "paper"})):
        p = pr[lid]
        for f, s in erwartet.items():
            assert stufe_von(p.einstellungen, FAKTOREN[f], p.recent_climate) == s, (lid, f)


def test_original_toml_entspricht_legacy():
    assert Config.from_toml(ROOT / "configs/original.toml").to_dict() == legacy_config(output_dir="results/legacy").to_dict()


def test_toml_behaelt_prior_aus_daten(tmp_path):
    """None beim Prior des rezenten Klimas (aus den Eingangsdaten) darf beim Schreiben nicht verloren gehen;
    ein fehlender Eintrag bedeutet seit v3 den CRU-Prior."""
    cfg = legacy_config()
    write_toml(cfg, tmp_path / "c.toml")
    back = Config.from_toml(tmp_path / "c.toml")
    assert back.sampler.recent_temp_mean is None and back.sampler.recent_pann_sd is None
    assert back.to_dict() == cfg.to_dict()


def _kleiner_katalog(tmp_path, extra_variante: str = "") -> Path:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/basis.toml").write_text('[sampler]\ntaxa_prior = "fixed"\n')
    (tmp_path / "configs/var.toml").write_text('[sampler]\ntaxa_prior = "fixed"\ntaxa_prior_alpha = 2.0\n'
                                               + extra_variante)
    for lid, cfg, extra in (("basis", "basis", ""), ("var", "var", 'basis = "basis"\nfaktor = "gewichte_prior"\n'
                                                                      'stufe = "dir_2"\n')):
        d = tmp_path / "reference/laeufe" / lid
        d.mkdir(parents=True)
        (d / "lauf.toml").write_text(f'config = "configs/{cfg}.toml"\n{extra}\n[titel]\nde = "x"\nen = "x"\n\n'
                                     '[beschreibung]\nde = "x"\nen = "x"\n')
    (tmp_path / "data").symlink_to(ROOT / "data")
    return tmp_path


def test_pruefung_erkennt_abweichung_ausserhalb_des_faktors(tmp_path):
    root = _kleiner_katalog(tmp_path)
    pr = pruefe_katalog(lade_katalog(root), root)
    assert pr["var"].fehler == []
    shutil.rmtree(root / "configs")
    shutil.rmtree(root / "reference")
    (root / "data").unlink()
    root = _kleiner_katalog(tmp_path, "beta_shape1 = 2.0\n")
    pr = pruefe_katalog(lade_katalog(root), root)
    assert any("beta_shape1" in e for e in pr["var"].fehler)


def test_pruefung_erkennt_falsche_stufe(tmp_path):
    root = _kleiner_katalog(tmp_path)
    p = root / "reference/laeufe/var/lauf.toml"
    p.write_text(p.read_text().replace('"dir_2"', '"dir_1"'))
    pr = pruefe_katalog(lade_katalog(root), root)
    assert any("Stufe" in e for e in pr["var"].fehler)


@pytest.mark.parametrize("reconstruction", ["mean", "mixture"])
def test_zusammenfassung_kurzer_lauf(tmp_path, reconstruction):
    cfg = Config()
    cfg.sampler.sample_length, cfg.sampler.burn_in, cfg.sampler.thin, cfg.sampler.n_chains = 6000, 2000, 20, 2
    cfg.postprocess.reconstruction = reconstruction
    cfg.data_dir = str(ROOT / "data/in")
    cfg.output_dir = str(tmp_path / "lauf")
    cfg.plots = cfg.report = False
    execute(cfg, ROOT, log=lambda *a: None)
    lauf = Lauf(id="test", ordner=tmp_path, titel={"de": "t", "en": "t"}, beschreibung={"de": "b", "en": "b"},
                config="configs/standard.toml")
    d = zusammenfassung(lauf, LaufDaten.load(tmp_path / "lauf", ROOT), ROOT)
    n = schreibe_json(d, tmp_path / "app.json")
    assert n < 1_000_000
    d = json.loads((tmp_path / "app.json").read_text())
    rk = d["rekonstruktion"]
    assert rk["standard"] == reconstruction
    n_age, n_depth = len(rk["alter"]["x"]), len(rk["tiefe"]["x"])
    # Quantile der Standard-Rekonstruktion stimmen mit den Tabellen des Laufs ueberein
    rows = list(csv.DictReader(open(tmp_path / "lauf/rekonstruktion_alter.csv")))
    for v, col, tol in (("T", "T_DJF_q50", 1e-3), ("P", "P_ANN_q50", 0.06)):
        np.testing.assert_allclose(rk["alter"][reconstruction][v]["q50"], [float(r[col]) for r in rows], atol=tol)
    np.testing.assert_allclose(np.array(d["biome"]["alter"]["q50"])[0], [float(r["biome1_median"]) for r in rows],
                               atol=1e-4)
    dichte = np.array(rk["tiefe"]["mixture"]["dichte"]["P"])
    assert dichte.shape == (n_depth, len(rk["raster"]["P"])) and dichte.max() == 100
    assert np.array(d["transferfunktionen"]["dichte_posterior"]).shape == (3, len(rk["raster"]["P"]),
                                                                          len(rk["raster"]["T"]))
    assert len(d["gewichte"]["q50"]) == 33 and len(d["ap"]["alter"]) == n_age
    assert d["gewichte"]["alpha"]["q50"] > 0
    assert set(d["diagnostik"]["spuren"]["werte"]) >= {"R2", "T0", "P0", "alpha"}
    # Prior der Gewichte: Median der Randverteilung liegt unter 1/K (schiefe Beta-Verteilung)
    assert 0 < d["gewichte"]["prior"]["q50"] < 1 / 33
    assert d["methoden"]["de"].startswith("Die Posterior-Verteilung wurde mit 2 Markov-Kette(n)")
    assert d["methoden"]["en"].startswith("The posterior distribution was estimated with 2 Markov chain(s)")
    assert ("mixture of Eq. (3)" in d["methoden"]["en"]) == (reconstruction == "mixture")
    assert d["kennzahlen"] == kennzahlen(d)
    v = vergleich(d, d)
    assert v["T"]["niveau_diff"] == 0 and v["P"]["r_anomalie"] == pytest.approx(1.0)


def test_standort_daten():
    d = standort_daten(STANDORTE["kinneret"], ROOT)
    assert len(d["taxa"]["namen"]) == 33 and sum(d["taxa"]["kultur"]) == 3
    assert np.array(d["spektrum_alter"]).shape == (len(d["alter"]), 33)
    np.testing.assert_allclose(np.array(d["spektrum_tiefe"]).sum(axis=1), 1, atol=1e-3)
    prior = np.array(d["tf_prior_dichte"])
    assert prior.shape == (3, len(d["raster"]["P"]), len(d["raster"]["T"])) and prior.max() == 100
    json.dumps(d, allow_nan=False)


def test_flache_einstellungen_loest_prior_auf():
    e = flache_einstellungen(legacy_config(), [1.0, 2.0, 3.0, 4.0])
    assert [e[f"sampler.recent_{k}"] for k in ("temp_mean", "temp_sd", "pann_mean", "pann_sd")] == [1, 2, 3, 4]
    assert "output_dir" not in e and "postprocess.n_threads" not in e
