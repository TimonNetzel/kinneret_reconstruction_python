"""Prior des rezenten Klimas aus CRU-Daten fuer ein festgelegtes Einzugsgebiet.

Grundlage sind die Monatswerte von CRU TS 4.07 (Harris et al., 2020; 0,5 Grad, ``tmp`` und ``pre``) und die
10-Minuten-Klimatologie CRU CL 2.0 (New et al., 2002) mit ihrem Hoehenmodell. ``extract`` schneidet die
Levante aus den Rohdateien aus und speichert den Ausschnitt (``data/in/cru_levante.npz``, < 1 MB);
``recent_prior`` berechnet daraus Mittel und Unsicherheit von T_DJF und P_ANN fuer ein Einzugsgebiet
(Radius um den See, entfernungsgewichtet).

Die Sd des Priors ist als Unsicherheit des Klimas im Einzugsgebiet gemeint, nicht als raeumliche Streuung
der Gitterzellen. Sie setzt sich aus unabhaengigen Komponenten zusammen (quadratisch addiert):

* ``zeitmittel``: Stichprobenfehler des Periodenmittels aus der zwischenjaehrlichen Variabilitaet
  (effektiver Stichprobenumfang mit AR(1)-Korrektur),
* ``gebiet``: Repraesentativitaet, mittlere quadratische Abweichung der Mittel anderer plausibler
  Einzugsgebiete (Radien und Gewichtungen) vom gewaehlten,
* ``produkt``: Datenfehler, Differenz CRU TS 4.07 gegen CRU CL 2.0 fuer dasselbe Gebiet (1961-1990),
* ``zeitraum``: Differenz zum Zeitraum, den die juengste Altersstufe darstellen soll (Standard 1991-2020;
  die Altersstufe -50 cal BP entspricht 2000 CE, der Prior bleibt aber wie die Transferfunktion bei 1961-1990),
* ``zusatz`` (optional): frei waehlbarer weiterer Fehler, z. B. Messfehler des Niederschlags.
"""

from __future__ import annotations

import gzip
import io
import math
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# See Genezareth: Seemitte und mittlerer Seespiegel
LAKE_LAT = 32.82
LAKE_LON = 35.59
LAKE_ELEVATION_M = -212.0

# Ausschnitt (Mittelpunkte der 0,5-Grad-Zellen): deckt Radien bis ~250 km um den See ab
BOX = (29.0, 36.5, 32.5, 39.5)  # lat_min, lat_max, lon_min, lon_max
YEARS = (1951, 2020)  # 1951-1960 nur fuer den Dezember 1960 (DJF 1961)

CRU_TS_URL = "https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.07/cruts.2304141047.v4.07"
CRU_CL_URL = "https://crudata.uea.ac.uk/cru/data/hrg/tmc"
EARTH_RADIUS_KM = 6371.0
KERNELS = ("gauss", "uniform", "linear")


# --------------------------------------------------------------------------- Rohdaten und Ausschnitt

def _ts_names(var: str, years: tuple[int, int] = YEARS) -> list[str]:
    return [f"cru_ts4.07.{y}.{y + 9}.{var}.dat.nc.gz" for y in range(years[0], years[1], 10)]


def _cl_names() -> list[str]:
    return [f"grid_10min_{v}.dat.gz" for v in ("elv", "tmp", "pre")]


def download(raw_dir: str | Path) -> None:
    """Rohdateien (CRU TS 4.07 als Jahrzehntdateien, CRU CL 2.0) nach ``raw_dir`` laden (~320 MB)."""
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    urls = [f"{CRU_TS_URL}/{v}/{n}" for v in ("tmp", "pre") for n in _ts_names(v)]
    urls += [f"{CRU_CL_URL}/{n}" for n in _cl_names()]
    for url in urls:
        target = raw / url.rsplit("/", 1)[1]
        if target.exists():
            continue
        print(f"lade {url}")
        tmp = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.rename(target)


def _read_ts(raw: Path, var: str, box=BOX):
    from scipy.io import netcdf_file

    fields, stns, lat, lon = [], [], None, None
    for name in _ts_names(var):
        with gzip.open(raw / name) as gz:
            nc = netcdf_file(io.BytesIO(gz.read()), "r", mmap=False)
        la, lo = nc.variables["lat"][:].astype(float), nc.variables["lon"][:].astype(float)
        il = (la > box[0]) & (la < box[1])
        jl = (lo > box[2]) & (lo < box[3])
        x = nc.variables[var][:][:, il][:, :, jl].astype(np.float32)
        x[x > 1e30] = np.nan
        fields.append(x)
        stns.append(nc.variables["stn"][:][:, il][:, :, jl].astype(np.int16))
        lat, lon = la[il], lo[jl]
        nc.close()
    x = np.concatenate(fields)
    n_year = x.shape[0] // 12
    shape = (n_year, 12, len(lat), len(lon))
    return lat, lon, x.reshape(shape), np.concatenate(stns).reshape(shape)


def _read_cl(path: Path, n_values: int, box) -> tuple[np.ndarray, np.ndarray]:
    """CRU CL 2.0 (ASCII: lat, lon, Werte) auf den Ausschnitt einlesen: (lat/lon je Zeile, Werte)."""
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            la, lo = float(line[:9]), float(line[9:18])
            if box[0] < la < box[1] and box[2] < lo < box[3]:
                rows.append([la, lo] + [float(line[18 + 7 * k: 25 + 7 * k]) for k in range(n_values)]
                            if n_values > 1 else [la, lo, float(line[18:27])])
    a = np.asarray(rows)
    return a[:, :2], a[:, 2:]


def extract(raw_dir: str | Path, out_path: str | Path) -> Path:
    """Levante-Ausschnitt aus den Rohdateien in ``raw_dir`` als npz speichern."""
    raw = Path(raw_dir)
    lat, lon, tmp, stn_tmp = _read_ts(raw, "tmp")
    lat_p, lon_p, pre, stn_pre = _read_ts(raw, "pre")
    if not (np.array_equal(lat, lat_p) and np.array_equal(lon, lon_p)):
        raise ValueError("tmp- und pre-Raster passen nicht zusammen")
    # CRU CL 2.0 auf ein regelmaessiges 10'-Raster bringen (nur Landzellen sind in den Dateien)
    box10 = (BOX[0] - 0.25, BOX[1] + 0.25, BOX[2] - 0.25, BOX[3] + 0.25)
    ll_e, elv = _read_cl(raw / "grid_10min_elv.dat.gz", 1, box10)
    ll_t, t10 = _read_cl(raw / "grid_10min_tmp.dat.gz", 12, box10)
    ll_p, p10 = _read_cl(raw / "grid_10min_pre.dat.gz", 24, box10)  # 12 Mittel, 12 Variationskoeffizienten
    lat10 = np.unique(np.round(ll_e[:, 0], 3))
    lon10 = np.unique(np.round(ll_e[:, 1], 3))

    def grid(ll, vals):
        g = np.full((vals.shape[1], len(lat10), len(lon10)), np.nan, np.float32)
        i = np.searchsorted(lat10, np.round(ll[:, 0], 3))
        j = np.searchsorted(lon10, np.round(ll[:, 1], 3))
        g[:, i, j] = vals.T
        return g

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, lat=lat, lon=lon, years=np.arange(YEARS[0], YEARS[0] + tmp.shape[0]),
        tmp=tmp, pre=pre, stn_tmp=stn_tmp, stn_pre=stn_pre,
        lat10=lat10, lon10=lon10, elv10=grid(ll_e, elv)[0] * 1000.0,
        tmp10=grid(ll_t, t10), pre10=grid(ll_p, p10[:, :12]),
        source=np.array("CRU TS 4.07 (cruts.2304141047.v4.07), tmp/pre, 0,5 Grad; "
                        "CRU CL 2.0 10' (New et al., 2002), tmp/pre/elv"),
    )
    return out


@dataclass(frozen=True)
class CruData:
    lat: np.ndarray  # (n_lat,) Zellmitten [Grad N]
    lon: np.ndarray  # (n_lon,)
    years: np.ndarray  # (n_year,)
    tmp: np.ndarray  # (n_year, 12, n_lat, n_lon) [Grad C]
    pre: np.ndarray  # (n_year, 12, n_lat, n_lon) [mm/Monat]
    stn_tmp: np.ndarray  # Anzahl beitragender Stationen je Monatswert (0: Klimatologie eingesetzt)
    stn_pre: np.ndarray
    lat10: np.ndarray  # CRU CL 2.0, 10'-Raster
    lon10: np.ndarray
    elv10: np.ndarray  # (n_lat10, n_lon10) [m]
    tmp10: np.ndarray  # (12, n_lat10, n_lon10)
    pre10: np.ndarray  # (12, n_lat10, n_lon10)

    @classmethod
    def load(cls, path: str | Path) -> "CruData":
        d = np.load(path)
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})

    def elevation(self) -> np.ndarray:
        """Mittlere Hoehe [m] der 0,5-Grad-Zellen aus dem 10'-Hoehenmodell."""
        out = np.full((len(self.lat), len(self.lon)), np.nan)
        for i, la in enumerate(self.lat):
            mi = np.abs(self.lat10 - la) < 0.25
            for j, lo in enumerate(self.lon):
                mj = np.abs(self.lon10 - lo) < 0.25
                v = self.elv10[np.ix_(mi, mj)]
                if np.isfinite(v).any():
                    out[i, j] = np.nanmean(v)
        return out


# --------------------------------------------------------------------------- Einzugsgebiet

def distance_km(lat1, lon1, lat2, lon2):
    """Grosskreisentfernung (Haversine) [km]."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def kernel_weight(d: np.ndarray, radius_km: float, kernel: str) -> np.ndarray:
    """Gewicht nach Entfernung d: ``gauss`` (sigma = R/2), ``uniform`` oder ``linear`` (1 - d/R); 0 fuer d > R."""
    inside = d <= radius_km
    if kernel == "gauss":
        w = np.exp(-0.5 * (d / (radius_km / 2)) ** 2)
    elif kernel == "uniform":
        w = np.ones_like(d)
    elif kernel == "linear":
        w = 1 - d / radius_km
    else:
        raise ValueError(f"unbekannte Gewichtung {kernel!r}, erlaubt: {', '.join(KERNELS)}")
    return np.where(inside, w, 0.0)


def catchment_weights(lat: np.ndarray, lon: np.ndarray, valid: np.ndarray, radius_km: float,
                      kernel: str = "gauss", center: tuple[float, float] = (LAKE_LAT, LAKE_LON),
                      n_sub: int = 10) -> np.ndarray:
    """Gewichte der Gitterzellen fuer das Einzugsgebiet (Summe 1).

    Jede Zelle wird in n_sub x n_sub Teilpunkte zerlegt; das Zellgewicht ist die Summe der Kernelgewichte
    der Teilpunkte mal ihrer Flaeche (cos lat). So hat auch ein Radius unter der Zellgroesse einen
    stetigen Einfluss. Zellen ohne Daten (Meer) erhalten Gewicht 0.
    """
    dlat = float(np.median(np.diff(lat)))
    dlon = float(np.median(np.diff(lon)))
    off = (np.arange(n_sub) + 0.5) / n_sub - 0.5
    sub_lat = lat[:, None] + off[None, :] * dlat  # (n_lat, n_sub)
    sub_lon = lon[:, None] + off[None, :] * dlon  # (n_lon, n_sub)
    d = distance_km(center[0], center[1], sub_lat[:, None, :, None], sub_lon[None, :, None, :])
    w = kernel_weight(d, radius_km, kernel) * np.cos(np.radians(sub_lat))[:, None, :, None]
    w = w.sum(axis=(2, 3))
    w[~valid] = 0.0
    if w.sum() <= 0:
        raise ValueError(f"keine Landzelle mit Daten innerhalb von {radius_km} km")
    return w / w.sum()


# --------------------------------------------------------------------------- Prior

def seasonal_series(data: CruData, y0: int, y1: int):
    """T_DJF (Dezember des Vorjahrs, Januar, Februar) und P_ANN je Jahr y0..y1: (n, n_lat, n_lon)."""
    if y0 - 1 < data.years[0] or y1 > data.years[-1]:
        raise ValueError(f"Zeitraum {y0}-{y1} nicht im Ausschnitt ({data.years[0] + 1}-{data.years[-1]})")
    i = np.arange(y0, y1 + 1) - data.years[0]
    t = (data.tmp[i - 1, 11] + data.tmp[i, 0] + data.tmp[i, 1]) / 3.0
    p = data.pre[i].sum(axis=1)
    stn = np.stack([data.stn_tmp[i - 1, 11], data.stn_tmp[i, 0], data.stn_tmp[i, 1]], axis=1)
    return t.astype(float), p.astype(float), stn, data.stn_pre[i]


def sd_of_mean(x: np.ndarray) -> tuple[float, float]:
    """Sd des Mittels einer Jahresreihe mit AR(1)-Korrektur: (Sd, effektiver Umfang)."""
    n = len(x)
    a = x - x.mean()
    r1 = float(np.sum(a[1:] * a[:-1]) / np.sum(a * a)) if np.any(a) else 0.0
    r1 = min(max(r1, 0.0), 0.9)
    n_eff = n * (1 - r1) / (1 + r1)
    return float(x.std(ddof=1) / math.sqrt(n_eff)), n_eff


def _lapse(t: np.ndarray, elev: np.ndarray, lapse_rate: float | None, ref_elevation: float) -> np.ndarray:
    if lapse_rate is None:
        return t
    return t + lapse_rate / 1000.0 * (elev - ref_elevation)


@dataclass
class PriorSettings:
    radius_km: float = 50.0
    kernel: str = "gauss"
    period: tuple[int, int] = (1961, 1990)
    # Zeitraum, den die juengste Altersstufe darstellen soll (None: keine Zeitraum-Komponente)
    target_period: tuple[int, int] | None = (1991, 2020)
    # Hoehenkorrektur von T auf ref_elevation [Grad C/km]; None: keine (Klima am Standort der Vegetation)
    lapse_rate: float | None = None
    ref_elevation: float = LAKE_ELEVATION_M
    center: tuple[float, float] = (LAKE_LAT, LAKE_LON)
    # alternative Einzugsgebiete fuer die Repraesentativitaet
    alt_radii: tuple[float, ...] = (25.0, 50.0, 100.0)
    alt_kernels: tuple[str, ...] = KERNELS
    # zusaetzliche Fehlerkomponente: T absolut [Grad C], P relativ zum Mittel
    extra_temp_sd: float = 0.0
    extra_pann_rel_sd: float = 0.0


@dataclass
class PriorResult:
    settings: PriorSettings
    temp_mean: float
    pann_mean: float
    temp_sd: float
    pann_sd: float
    # Komponenten der Sd (quadratisch addiert)
    temp_components: dict[str, float] = field(default_factory=dict)
    pann_components: dict[str, float] = field(default_factory=dict)
    # zur Information
    temp_spatial_sd: float = float("nan")  # gewichtete raeumliche Streuung (bisherige Methode)
    pann_spatial_sd: float = float("nan")
    temp_interannual_sd: float = float("nan")
    pann_interannual_sd: float = float("nan")
    temp_cl2: float = float("nan")  # dasselbe Gebiet in CRU CL 2.0 (1961-1990)
    pann_cl2: float = float("nan")
    n_cells: int = 0  # Zellen mit Gewicht > 1 %
    stn0_temp: float = float("nan")  # Anteil der Monatswerte ohne Station (gewichtet)
    stn0_pann: float = float("nan")
    cells: list[dict] = field(default_factory=list)

    def table(self) -> str:
        """Kurzbeschreibung mit allen Komponenten."""
        s = self.settings
        lines = [f"Einzugsgebiet: Radius {s.radius_km:g} km um ({s.center[0]}, {s.center[1]}), Gewichtung {s.kernel}, "
                 f"{self.n_cells} Zellen mit Gewicht > 1 %; CRU TS 4.07 {s.period[0]}-{s.period[1]}"
                 + (f"; Hoehenkorrektur {s.lapse_rate:g} degC/km auf {s.ref_elevation:g} m" if s.lapse_rate else "")]
        for name, m, sd, comp, spat, cl2, unit, fmt in (
                ("T_DJF", self.temp_mean, self.temp_sd, self.temp_components, self.temp_spatial_sd, self.temp_cl2, "degC", ".2f"),
                ("P_ANN", self.pann_mean, self.pann_sd, self.pann_components, self.pann_spatial_sd, self.pann_cl2, "mm", ".1f")):
            parts = ", ".join(f"{k} {v:{fmt}}" for k, v in comp.items())
            lines.append(f"  {name}: {m:{fmt}} +- {sd:{fmt}} {unit}  (Komponenten: {parts}; "
                         f"raeumliche Sd {spat:{fmt}}; CRU CL 2.0 {cl2:{fmt}})")
        return "\n".join(lines)

    def toml(self) -> str:
        s = self.settings
        return (f"# Prior des rezenten Klimas: CRU TS 4.07 {s.period[0]}-{s.period[1]}, Radius {s.radius_km:g} km "
                f"um ({s.center[0]}, {s.center[1]}),\n# Gewichtung {s.kernel}"
                + (f", Hoehenkorrektur {s.lapse_rate:g} degC/km auf {s.ref_elevation:g} m" if s.lapse_rate else "")
                + f"; berechnet mit kinneret-recon cru-prior\n[sampler]\n"
                f"recent_temp_mean = {self.temp_mean:.2f}\nrecent_temp_sd = {self.temp_sd:.2f}\n"
                f"recent_pann_mean = {self.pann_mean:.1f}\nrecent_pann_sd = {self.pann_sd:.1f}\n")

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("cells")
        return d


def _area_mean(field_: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.nansum(field_ * w, axis=(-2, -1))


def _cl2_means(data: CruData, s: PriorSettings) -> tuple[float, float]:
    t = data.tmp10[[11, 0, 1]].mean(axis=0)
    p = data.pre10.sum(axis=0)
    t = _lapse(t, data.elv10, s.lapse_rate, s.ref_elevation)
    valid = np.isfinite(t) & np.isfinite(p)
    w = catchment_weights(data.lat10, data.lon10, valid, s.radius_km, s.kernel, s.center, n_sub=4)
    return float(_area_mean(np.where(valid, t, 0), w)), float(_area_mean(np.where(valid, p, 0), w))


def recent_prior(data: CruData, s: PriorSettings | None = None) -> PriorResult:
    s = s or PriorSettings()
    elev = data.elevation()
    t_y, p_y, stn_t, stn_p = seasonal_series(data, *s.period)
    t_y = _lapse(t_y, elev, s.lapse_rate, s.ref_elevation)
    valid = np.isfinite(t_y).all(axis=0) & np.isfinite(p_y).all(axis=0)

    def means(radius, kernel, period=s.period):
        ty, py, _, _ = seasonal_series(data, *period)
        ty = _lapse(ty, elev, s.lapse_rate, s.ref_elevation)
        w = catchment_weights(data.lat, data.lon, valid, radius, kernel, s.center)
        return float(_area_mean(np.where(valid, ty, 0), w).mean()), float(_area_mean(np.where(valid, py, 0), w).mean())

    w = catchment_weights(data.lat, data.lon, valid, s.radius_km, s.kernel, s.center)
    st = _area_mean(np.where(valid, t_y, 0), w)
    sp = _area_mean(np.where(valid, p_y, 0), w)
    tm, pm = float(st.mean()), float(sp.mean())
    tc, pc = {}, {}
    tc["zeitmittel"], _ = sd_of_mean(st)
    pc["zeitmittel"], _ = sd_of_mean(sp)

    alt = [means(r, k) for r in s.alt_radii for k in s.alt_kernels]
    tc["gebiet"] = float(np.sqrt(np.mean([(a[0] - tm) ** 2 for a in alt])))
    pc["gebiet"] = float(np.sqrt(np.mean([(a[1] - pm) ** 2 for a in alt])))

    ref = (1961, 1990)
    t_ref, p_ref = (tm, pm) if tuple(s.period) == ref else means(s.radius_km, s.kernel, ref)
    t_cl, p_cl = _cl2_means(data, s)
    tc["produkt"], pc["produkt"] = abs(t_ref - t_cl), abs(p_ref - p_cl)

    if s.target_period is not None:
        t_tg, p_tg = means(s.radius_km, s.kernel, tuple(s.target_period))
        tc["zeitraum"], pc["zeitraum"] = abs(t_tg - tm), abs(p_tg - pm)

    if s.extra_temp_sd > 0 or s.extra_pann_rel_sd > 0:
        tc["zusatz"], pc["zusatz"] = s.extra_temp_sd, s.extra_pann_rel_sd * pm

    clim_t, clim_p = t_y.mean(axis=0), p_y.mean(axis=0)
    wv = w[valid]
    spatial = lambda c: float(np.sqrt(np.sum(wv * (c[valid] - np.sum(wv * c[valid])) ** 2)))
    cells = [dict(lat=float(data.lat[i]), lon=float(data.lon[j]), gewicht=float(w[i, j]),
                  entfernung_km=float(distance_km(s.center[0], s.center[1], data.lat[i], data.lon[j])),
                  hoehe_m=float(elev[i, j]), T_DJF=float(clim_t[i, j]), P_ANN=float(clim_p[i, j]))
             for i, j in zip(*np.nonzero(w > 0))]
    cells.sort(key=lambda c: -c["gewicht"])
    return PriorResult(
        settings=s, temp_mean=tm, pann_mean=pm,
        temp_sd=float(np.sqrt(sum(v ** 2 for v in tc.values()))),
        pann_sd=float(np.sqrt(sum(v ** 2 for v in pc.values()))),
        temp_components=tc, pann_components=pc,
        temp_spatial_sd=spatial(clim_t), pann_spatial_sd=spatial(clim_p),
        temp_interannual_sd=float(st.std(ddof=1)), pann_interannual_sd=float(sp.std(ddof=1)),
        temp_cl2=t_cl, pann_cl2=p_cl, n_cells=int(np.sum(w > 0.01)),
        stn0_temp=float(np.sum(w * (stn_t == 0).mean(axis=(0, 1)))),
        stn0_pann=float(np.sum(w * (stn_p == 0).mean(axis=(0, 1)))),
        cells=cells,
    )
