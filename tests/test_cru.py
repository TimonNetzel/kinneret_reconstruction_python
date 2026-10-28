"""Prior des rezenten Klimas aus CRU-Daten (Modul cru, Befehle cru-extract und cru-prior)."""

import gzip
from pathlib import Path

import numpy as np
import pytest
from scipy.io import netcdf_file

from kinneret_recon import cru
from kinneret_recon.cli import main
from kinneret_recon.config import CRU_PRIOR_50KM, Config, SamplerConfig
from kinneret_recon.katalog import standort

ROOT = Path(__file__).resolve().parents[1]
ST = standort("kinneret", ROOT)
SEE = (ST.lat, ST.lon)


@pytest.fixture(scope="module")
def data():
    return cru.CruData.load(ST.cru_pfad(ROOT))


def test_weights_sum_and_mask():
    lat, lon = np.arange(31.25, 34.5, 0.5), np.arange(34.25, 37.5, 0.5)
    valid = np.ones((len(lat), len(lon)), bool)
    valid[0, :] = False
    for kernel in cru.KERNELS:
        w = cru.catchment_weights(lat, lon, valid, 80.0, kernel, SEE)
        assert w.sum() == pytest.approx(1.0)
        assert np.all(w[0] == 0)
        # kleiner Radius: die Zelle mit dem See traegt am meisten
        w = cru.catchment_weights(lat, lon, valid, 20.0, kernel, SEE)
        i, j = np.unravel_index(np.argmax(w), w.shape)
        assert (lat[i], lon[j]) == (32.75, 35.75)
    # weiter entfernte Zellen erhalten bei groesserem Radius mehr Gewicht
    w1 = cru.catchment_weights(lat, lon, valid, 50.0, "gauss", SEE)
    w2 = cru.catchment_weights(lat, lon, valid, 150.0, "gauss", SEE)
    assert w2.max() < w1.max()


def test_distance():
    # 1 Grad Breite ~ 111,2 km
    assert cru.distance_km(32.0, 35.0, 33.0, 35.0) == pytest.approx(111.19, rel=1e-3)


def test_sd_of_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 2.0, 30)
    sd, n_eff = cru.sd_of_mean(x)
    assert n_eff <= 30
    assert sd == pytest.approx(x.std(ddof=1) / np.sqrt(n_eff))
    # positiv autokorrelierte Reihe: weniger effektive Werte
    y = np.cumsum(rng.normal(size=30)) * 0.3 + rng.normal(size=30)
    assert cru.sd_of_mean(y)[1] < 30


def test_lake_cell_matches_paper(data):
    # kleiner Radius: nur die Zelle am See, Abb. 2c des Papers: T_DJF ~ 11,4 degC, P_ANN ~ 440 mm
    r = cru.recent_prior(data, cru.PriorSettings.fuer_standort(ST, radius_km=10.0))
    assert r.n_cells == 1
    assert r.temp_mean == pytest.approx(11.39, abs=0.01)
    assert r.pann_mean == pytest.approx(440.1, abs=0.5)
    assert r.temp_spatial_sd == pytest.approx(0.0, abs=1e-9)


def test_prior_components(data):
    r = cru.recent_prior(data, cru.PriorSettings.fuer_standort(ST, radius_km=50.0, extra_pann_rel_sd=0.1))
    assert set(r.temp_components) == {"zeitmittel", "gebiet", "produkt", "zeitraum", "zusatz"}
    assert r.temp_sd == pytest.approx(np.sqrt(sum(v ** 2 for v in r.temp_components.values())))
    assert r.pann_components["zusatz"] == pytest.approx(0.1 * r.pann_mean)
    # alle Monatswerte im Gebiet sind stationsgestuetzt
    assert r.stn0_temp == 0 and r.stn0_pann == 0
    # die Unsicherheit ist kleiner als die raeumliche Streuung der bisherigen Methode
    assert r.temp_sd < r.temp_spatial_sd and r.pann_sd < r.pann_spatial_sd
    # Hoehenkorrektur auf den Seespiegel erwaermt
    rh = cru.recent_prior(data, cru.PriorSettings.fuer_standort(ST, radius_km=50.0, lapse_rate=5.5))
    assert rh.temp_mean > r.temp_mean + 2
    assert rh.pann_mean == pytest.approx(r.pann_mean)


def test_cli_writes_config(tmp_path):
    out = tmp_path / "prior.toml"
    cells = tmp_path / "zellen.csv"
    assert main(["cru-prior", "--cru", str(ST.cru_pfad(ROOT)), "-r", "50",
                 "--toml", str(out), "--cells", str(cells)]) == 0
    cfg = Config.from_toml(out)
    assert cfg.sampler.recent_temp_mean == pytest.approx(11.09, abs=0.05)
    assert 0 < cfg.sampler.recent_pann_sd < cfg.sampler.recent_pann_mean
    assert cells.read_text().startswith("radius_km,lat,lon,gewicht")


def _write_ts(path: Path, var: str, y0: int, lat, lon, value):
    n = 120
    tmp = path.with_suffix("")
    with netcdf_file(tmp, "w") as nc:
        nc.createDimension("lon", len(lon))
        nc.createDimension("lat", len(lat))
        nc.createDimension("time", n)
        nc.createVariable("lon", "f4", ("lon",))[:] = lon
        nc.createVariable("lat", "f4", ("lat",))[:] = lat
        v = nc.createVariable(var, "f4", ("time", "lat", "lon"))
        v[:] = value(np.arange(n))[:, None, None] * np.ones((n, len(lat), len(lon)))
        nc.createVariable("stn", "i4", ("time", "lat", "lon"))[:] = 3
    with open(tmp, "rb") as f, gzip.open(path, "wb") as g:
        g.write(f.read())
    tmp.unlink()


def test_extract_synthetic(tmp_path):
    raw = tmp_path / "roh"
    raw.mkdir()
    lat, lon = np.arange(31.25, 34.5, 0.5), np.arange(34.25, 37.5, 0.5)
    for y in range(cru.YEARS[0], cru.YEARS[1], 10):
        # tmp: Monat m hat m Grad; pre: 10 mm je Monat
        _write_ts(raw / f"cru_ts4.07.{y}.{y + 9}.tmp.dat.nc.gz", "tmp", y, lat, lon, lambda t: (t % 12).astype(float))
        _write_ts(raw / f"cru_ts4.07.{y}.{y + 9}.pre.dat.nc.gz", "pre", y, lat, lon, lambda t: np.full(len(t), 10.0))
    lat10 = np.arange(31.083, 34.6, 1 / 6)
    lon10 = np.arange(34.083, 37.6, 1 / 6)
    ll = [(a, b) for a in lat10 for b in lon10]
    with gzip.open(raw / "grid_10min_elv.dat.gz", "wt") as f:
        f.writelines(f"{a:9.3f}{b:9.3f}{0.1:9.3f}\n" for a, b in ll)
    with gzip.open(raw / "grid_10min_tmp.dat.gz", "wt") as f:
        f.writelines(f"{a:9.3f}{b:9.3f}" + "".join(f"{m:7.1f}" for m in range(12)) + "\n" for a, b in ll)
    with gzip.open(raw / "grid_10min_pre.dat.gz", "wt") as f:
        f.writelines(f"{a:9.3f}{b:9.3f}" + "".join(f"{10.0:7.1f}" for _ in range(12))
                     + "".join(f"{50.0:7.1f}" for _ in range(12)) + "\n" for a, b in ll)
    out = cru.extract(raw, tmp_path / "ausschnitt.npz", ST.cru.ausschnitt)
    d = cru.CruData.load(out)
    assert d.tmp.shape == (70, 12, len(lat), len(lon))
    np.testing.assert_allclose(d.elv10[np.isfinite(d.elv10)], 100.0)
    r = cru.recent_prior(d, cru.PriorSettings.fuer_standort(ST, radius_km=50.0))
    # T_DJF = (11 + 0 + 1) / 3 = 4, P_ANN = 120 mm; konstante Felder: alle Komponenten 0
    assert r.temp_mean == pytest.approx(4.0, abs=1e-5)
    assert r.pann_mean == pytest.approx(120.0, abs=1e-4)
    assert r.temp_cl2 == pytest.approx(4.0, abs=1e-5) and r.pann_cl2 == pytest.approx(120.0, abs=1e-4)
    assert r.temp_sd == pytest.approx(0.0, abs=1e-5) and r.pann_sd == pytest.approx(0.0, abs=1e-3)


def test_default_prior_is_cru_50km(data):
    """Die Voreinstellung des Samplers ist der CRU-Prior fuer 50 km (gerundet wie in der TOML-Ausgabe), berechnet mit
    den Einstellungen des Standort-Registers."""
    assert ST.cru.radius_km == 50.0
    r = cru.recent_prior(data, cru.PriorSettings.fuer_standort(ST))
    assert CRU_PRIOR_50KM == {"recent_temp_mean": round(r.temp_mean, 2), "recent_temp_sd": round(r.temp_sd, 2),
                              "recent_pann_mean": round(r.pann_mean, 1), "recent_pann_sd": round(r.pann_sd, 1)}
    s = SamplerConfig()
    assert {k: getattr(s, k) for k in CRU_PRIOR_50KM} == CRU_PRIOR_50KM
