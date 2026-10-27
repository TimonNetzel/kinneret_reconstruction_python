"""Niveau, Unsicherheit und Anomalien mehrerer Laeufe relativ zu einem Basislauf (erster Name).

Liest ``rekonstruktion_alter.csv`` aus ``results/<lauf>/`` oder, falls nicht vorhanden, aus
``reference/laeufe/<lauf>/``. Anomalien sind die Median-Kurven abzueglich ihres Mittels ueber alle Alter;
r ist die Korrelation der Anomalien mit dem Basislauf (wie Pruefbericht, Abschnitt 3.8).

Aufruf: python tools/anomalien.py standard o_cru_r25 o_cru_r50 o_cru_r100
"""
import csv
import sys
from pathlib import Path

import numpy as np


def load(run):
    for base in ("results", "reference/laeufe"):
        p = Path(base) / run / "rekonstruktion_alter.csv"
        if p.exists():
            rows = list(csv.DictReader(open(p)))
            return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    raise FileNotFoundError(f"rekonstruktion_alter.csv fuer {run!r} nicht gefunden")


runs = sys.argv[1:]
base = load(runs[0])
anom = lambda x: x - x.mean()
print(f"{'Lauf':22s} {'T':>6s} {'P':>5s} {'IQR_T':>6s} {'IQR_P':>6s} {'r_T':>6s} {'r_P':>6s} {'Spanne_P':>8s} "
      f"{'T(-50)':>7s} {'P(-50)':>7s}")
for r in runs:
    x = load(r)
    t, p = x["T_DJF_q50"], x["P_ANN_q50"]
    rt = np.corrcoef(anom(t), anom(base["T_DJF_q50"]))[0, 1]
    rp = np.corrcoef(anom(p), anom(base["P_ANN_q50"]))[0, 1]
    print(f"{r:22s} {t.mean():6.2f} {p.mean():5.0f} {(x['T_DJF_q75'] - x['T_DJF_q25']).mean():6.2f} "
          f"{(x['P_ANN_q75'] - x['P_ANN_q25']).mean():6.0f} {rt:6.3f} {rp:6.3f} {np.ptp(p):8.0f} "
          f"{t[0]:7.2f} {p[0]:7.0f}")
