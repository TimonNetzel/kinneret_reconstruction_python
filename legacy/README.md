# Originalcode (Paper-Version)

Dieser Ordner enthält die Skripte zur Publikation

> Netzel, T., Miebach, A., Litt, T., Hense, A. (2025): *New probabilistic methods for quantitative
> climate reconstructions applied to palynological data from Lake Kinneret.* Clim. Past 21, 357–380.
> https://doi.org/10.5194/cp-21-357-2025

Der unveränderte Stand ist im Git-Tag **`v0.1.0`** gesichert (identisch mit `main` bis auf `LICENSE.md`; Zenodo:
doi:10.5281/zenodo.8214290). Die Dateien hier unterscheiden sich
davon nur durch minimale Kompatibilitäts-Patches. Ohne sie läuft der Code mit aktuellen Versionen
von Python, NumPy und Matplotlib nicht mehr (getestet: Python 3.13, NumPy 2.5, Matplotlib 3.11).
**Am Algorithmus und an den Ergebnissen ändert keiner dieser Patches etwas.**

| Datei | Patch | Grund |
|---|---|---|
| `main.py` | Endung des C++-Moduls über `sysconfig` statt `python3-config` | `python3-config` kann zu einer anderen Python-Version gehören; dann ist `pycpp` nicht importierbar |
| `src/Py_functions.py` | `np.linspace(..., int(...))` | NumPy ≥ 1.18 verlangt eine ganze Zahl |
| `src/Py_functions.py` | `np.diff(...)[0]` bei den CI-Breiten | NumPy ≥ 2 erlaubt keine Zuweisung eines Arrays an ein Skalar-Element |
| `src/mcmc_postprocessing.py` | `ListedColormap(...)` statt `cm.get_cmap(lut=500)` | in Matplotlib 3.9 entfernt |
| `src/mcmc_postprocessing.py` | Legende der Konturplots über `Line2D`-Platzhalter | `ContourSet.collections` in Matplotlib 3.10 entfernt |
| `src/mcmc_postprocessing.py` | `boxplot(tick_labels=...)` | `labels=` in Matplotlib 3.9 umbenannt |

## Ausführen

```bash
cd legacy
python3 main.py
```

`data/in` ist ein symbolischer Link auf die Eingangsdaten im Hauptverzeichnis.

## Referenzergebnisse

Die Ergebnisse eines vollständigen Laufs (Seed 2023) liegen in `reference/v1.0-paper/`:

- `post_process.npz`: Ausgabe der Auswertung
- `posterior_thinned.npz`: die 10 000 ausgewerteten Iterationen der Kette. Bitidentisch zu
  `posterior.npz` des Originals, das mit 352 MB zu groß fürs Repository ist.
- `plots/`: die sieben Abbildungen

Gemessene Laufzeit auf 4 CPU-Kernen: MCMC etwa 60 s (Spitzenspeicher etwa 3 GB), Auswertung und
Abbildungen 576 s.

Hinweis: `prior_*_ci_size` sind nicht reproduzierbar. `np.random.choice` wird in der Auswertung
nicht geseedet, die Werte schwanken deshalb von Lauf zu Lauf um etwa 1 %.
