# Kinneret-Rekonstruktion

Bayessche Rekonstruktion von Wintertemperatur (T_DJF) und Jahresniederschlag (P_ANN) aus
Pollendaten des See Genezareth (Lake Kinneret). Das Paket setzt das hierarchische Modell aus

> Netzel, T., Miebach, A., Litt, T., Hense, A. (2025): *New probabilistic methods for quantitative
> climate reconstructions applied to palynological data from Lake Kinneret.* Clim. Past 21, 357–380.
> https://doi.org/10.5194/cp-21-357-2025

als installierbares Python-Paket mit C++-Kern um.

| Ordner | Inhalt |
|---|---|
| `src/kinneret_recon/` | Python-Paket (Konfiguration, Daten, Modell, Sampler, Auswertung, Diagnostik, Abbildungen, Bericht, Kommandozeile) |
| `cpp/` | C++-Kern: Original-Sampler (`legacy_kernel.hpp`), überarbeiteter Sampler (`kernel.hpp`), Kerndichteschätzer (`kde.hpp`) |
| `configs/` | Konfigurationen der Standard-, Vergleichs- und Sensitivitätsläufe |
| `legacy/` | Originalskripte der Publikation (lauffähig gemacht; unverändert im Tag `v1.0-paper`) |
| `reference/v1.0-paper/` | Referenzergebnisse des Originals |
| `docs/` | Prüfbericht (gefundene Fehler, Auswirkungen) und Projektstand (Plan, Entscheidungen, offene Punkte) |
| `reference/laeufe/`, `reference/vergleiche/` | Tabellen, Diagnostik und Einstellungen aller Vergleichs- und Sensitivitätsläufe |
| `tools/` | Hilfsskripte für Auswertung und Abstimmung |
| `tests/` | Tests (`pytest`; lange Läufe mit `pytest -m slow`) |

## Installation

Voraussetzungen: Python ≥ 3.11 und ein C++17-Compiler (z. B. g++).

```bash
pip install -e ".[test]"     # kompiliert den C++-Kern einmalig
pytest                       # Tests (~30 s)
```

## Benutzung

```bash
kinneret-recon run                         # Standardlauf (überarbeiteter Sampler, 4 Ketten)
kinneret-recon run -c configs/standard.toml
kinneret-recon run --legacy                # exakt wie das Originalskript (bitidentische Ergebnisse)
kinneret-recon run --chains 8 --samples 4000000 -o results/lang
kinneret-recon config meine_config.toml    # Vorlage mit allen Einstellungen schreiben
kinneret-recon compare Original=results/legacy Neu=results/standard -o results/vergleich
```

Ein Lauf schreibt in sein Ausgabeverzeichnis:

| Datei | Inhalt |
|---|---|
| `bericht.html` | Bericht mit allen Abbildungen, Tabellen, Konvergenzdiagnostik und automatischer Methodenbeschreibung |
| `rekonstruktion_alter.csv`, `rekonstruktion_tiefe.csv` | Quantile (2,5/25/50/75/97,5 %) und Mittel von T_DJF und P_ANN, Biomanteile |
| `diagnostik.csv` | Mittel, Sd, Quantile, R̂, ESS (bulk/tail) für alle Parameter |
| `plots/` | Abbildungen als PDF und PNG |
| `posterior.npz`, `results.npz` | Ketten (ausgedünnt) und Auswertung für eigene Analysen |
| `config.toml`, `meta.json` | verwendete Einstellungen, Git-Stand, Versionen, Laufzeiten, Seeds |

Mit Seed und Konfiguration ist jeder Lauf bitgenau reproduzierbar. Der Zufallszahlengenerator ist
portabel und liefert mit jedem standardkonformen Compiler dieselben Zahlen.

## Sampler

* **`kernel = "legacy"`**: der Original-Sampler. Ketten, Quartile und Abbildungen sind bitidentisch
  zu `legacy/main.py`, Laufzeit etwa 1 min statt etwa 11 min.
* **`kernel = "corrected"`** (Standard): überarbeiteter Sampler. Jede Korrektur lässt sich in der
  Konfiguration einzeln abschalten:
  * `dirichlet_hastings_fix`: Die Hastings-Korrektur des Dirichlet-Vorschlags enthält den
    Streuungsparameter κ. Im Original fehlt er, die Kette trifft dort nicht die beabsichtigte
    Posterior (siehe `docs/pruefbericht.md`).
  * `exact_gamma_moments`: Die Parameter des Gamma-Priors für den rezenten Niederschlag werden
    analytisch bestimmt statt per Gittersuche.
  * `weights_proposal = "log_gamma"`: Random Walk auf log y mit w = y / Σy. Die Zielverteilung
    ist dieselbe, die Kette mischt aber etwa 50-mal effizienter als mit dem Dirichlet-Random-Walk.
  * mehrere Ketten mit gestreuten Startwerten und Konvergenzdiagnostik nach Vehtari et al. (2021).
* `postprocess.quantile_method = "samples"`: Quartile direkt aus den Stichproben. Das Original
  berechnet sie über geglättete Kerndichten (`"kde"`).

## Leistung (4 CPU-Kerne)

| | Original | Paket, `--legacy` | Paket, Standard |
|---|---|---|---|
| Sampling | ~60 s, 1 Kette, ~3 GB RAM | 36 s, 1 Kette | ~90 s, 4 Ketten × 2 Mio. |
| Prior-Referenzlauf | – | – | ~90 s |
| Auswertung + Abbildungen | 576 s | ~20 s | ~45 s |
