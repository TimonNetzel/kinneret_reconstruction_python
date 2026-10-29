"""Karte des CRU-Ausschnitts mit den Einzugsgebieten (Radien) um den See.

Aufruf: python tools/karte_cru.py [docs/abbildungen/cru_einzugsgebiet.png]
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kinneret_recon import cru
from kinneret_recon.katalog import standort

out = sys.argv[1] if len(sys.argv) > 1 else "docs/abbildungen/cru_einzugsgebiet.png"
st = standort("kinneret")
d = cru.CruData.load(st.cru_pfad(Path.cwd()))
t, p, _, _ = cru.seasonal_series(d, 1961, 1990)
fields = [(t.mean(0), "T_DJF 1961–1990 [°C]", "Oranges"), (p.mean(0), "P_ANN 1961–1990 [mm]", "Blues")]
lat0, lon0 = st.lat, st.lon
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), constrained_layout=True)
for ax, (f, title, cmap) in zip(axes, fields):
    m = ax.pcolormesh(d.lon, d.lat, f, cmap=cmap, shading="nearest")
    fig.colorbar(m, ax=ax, shrink=0.85)
    for r in (25, 50, 100):
        a = np.linspace(0, 2 * np.pi, 200)
        dl = r / 111.19
        ax.plot(lon0 + dl * np.cos(a) / np.cos(np.radians(lat0)), lat0 + dl * np.sin(a), color="0.2", lw=1.2)
        ax.text(lon0 + 0.02, lat0 + dl + 0.03, f"{r} km", fontsize=8, color="0.15")
    ax.plot(lon0, lat0, marker="o", ms=8, mfc="white", mec="black", mew=1.5)
    ax.set_xlim(33.8, 37.4)
    ax.set_ylim(31.0, 34.6)
    ax.set_aspect(1 / np.cos(np.radians(lat0)))
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Länge [°O]")
    ax.set_ylabel("Breite [°N]")
fig.suptitle("CRU TS 4.07, 0,5°: Einzugsgebiete um den See Genezareth (Kreis = Radius R, Gauß-Gewichtung mit σ = R/2)",
             fontsize=10)
fig.savefig(out, dpi=130)
print(out)
