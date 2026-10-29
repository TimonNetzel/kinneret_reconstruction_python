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
| `legacy/` | Originalskripte der Publikation (lauffähig gemacht; unverändert im Tag `v0.1.0` = Zenodo-Version) |
| `reference/v1.0-paper/` | Referenzergebnisse des Originals |
| `docs/` | Prüfbericht (gefundene Fehler, Auswirkungen), Abgleich Paper ↔ Code, Projektstand (Plan, Entscheidungen, offene Punkte), App-Konzept, Paper (`docs/paper/`) |
| `reference/laeufe/`, `reference/vergleiche/` | Lauf-Katalog: je Lauf Beschreibung (`lauf.toml`), Tabellen, Diagnostik, Einstellungen und kompakte Zusammenfassung für die App (`app.json`); Vergleiche |
| `reference/app/` | Katalog-Index und Standortdaten für die App |
| `tools/` | Hilfsskripte für Auswertung und Abstimmung |
| `tests/` | Tests (`pytest`; lange Läufe mit `pytest -m slow`) |

## Installation

Voraussetzungen: Python ≥ 3.11 und ein C++17-Compiler (z. B. g++).

```bash
pip install -e ".[test]"     # kompiliert den C++-Kern einmalig
pytest                       # Tests (~1 min; lange Tests: pytest -m slow)
```

## Benutzung

```bash
kinneret-recon run                         # Standardlauf (Voreinstellungen, siehe unten), ~13 min
kinneret-recon run -c configs/standard.toml # dasselbe mit ausgeschriebenen Einstellungen
kinneret-recon run -c configs/standard_v2.toml  # bisheriger Standard (Dir(0,5), Prior wie im Paper), ~4 min
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

Mit Seed und Konfiguration ist jeder Lauf bitgenau reproduzierbar (mit demselben Code-Stand; `meta.json`
enthält den Git-Stand). Der Zufallszahlengenerator ist portabel und liefert mit jedem standardkonformen
Compiler dieselben Zahlen.

## Voreinstellungen (Standard ab v3)

Seit dem 27.10.2026 (vom Autor festgelegt) gilt als Standard:

| Einstellung | Standard v3 | bis v2 |
|---|---|---|
| Prior der Taxa-Gewichte | hierarchisch: Dir(α), log α ~ N(log 1, 1) | Dir(0,5) wie im Paper |
| Prior des rezenten Klimas | CRU TS 4.07, Einzugsgebiet 50 km: 11,09 ± 0,87 °C, 521,8 ± 41,4 mm | Mittel und räumliche Sd mehrerer Gitterpunkte: 10,74 ± 2,48 °C, 556 ± 135 mm |
| Länge je Kette (4 Ketten) | 8 Mio., Burn-in 2 Mio., Ausdünnung 600 | 2 Mio., 500 000, 150 |

Alle anderen Einstellungen sind unverändert. Die Konfigurationen der Vergleichsreihe (`configs/a_*` bis
`configs/o_*`, `configs/standard_v2.toml`) schreiben die Werte bis v2 ausdrücklich fest und liefern dieselben
Läufe wie zuvor. `sampler.recent_* = None` (in TOML-Dateien `"daten"`) verwendet die Werte der Eingangsdaten
wie im Paper; ein fehlender Eintrag bedeutet den CRU-Prior der Voreinstellung.

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
* `postprocess.reconstruction = "mean"` (Standard) oder `"mixture"`: Rekonstruktion als Mittelwert der
  Biom-Klimawerte oder als Mischverteilung nach Gl. (3) des Papers.
* Weitere Modelloptionen: `sampler.tf_normal_prior` (Gl. 14 als Prior),
  `sampler.recent_temp_mean/_sd` und `sampler.recent_pann_mean/_sd` (Prior des heutigen Klimas),
  `sampler.taxa_prior` (Dir(α) fest oder hierarchisch), `sampler.beta_shape1/2` (R²-Prior),
  `sampler.tf_bound_probs` (Trunkierung), `sampler.ap_exclude_taxa` (Baumpollenkurve ohne z. B.
  Kulturbäume) und `sampler.calibration_age_range` (R² nur über einen Altersbereich).
* `kinneret-recon run ... --from-run results/<lauf>`: vorhandene Ketten neu auswerten, ohne neu zu
  sampeln. Nützlich für andere Auswerteoptionen.

## Lauf-Katalog und App-Daten

Alle Läufe der Vergleichsreihe bilden einen Katalog: je Lauf ein Ordner `reference/laeufe/<lauf>/` mit einer
Beschreibung `lauf.toml` (Titel und Beschreibung de/en, Konfiguration, Rolle, Basislauf, Faktor und Stufe).
Die Sensitivitätsfaktoren und ihre Stufen stehen im Faktor-Register (`src/kinneret_recon/katalog.py`).
Aus den vollständigen Ergebnissen entsteht je Lauf eine kompakte Zusammenfassung `app.json` (< 1 MB) für die
App (`docs/app_konzept.md`). Jeder Lauf gehört zu einem Standort des Standort-Registers: ein Ordner unter `data/`
mit den Eingangsdaten und einer Beschreibung `standort.toml` (Lage, Seehöhe, Biome mit Farben, CRU-Ausschnitt und
Einzugsgebiet; Kinneret: `data/in/standort.toml`).

```bash
kinneret-recon katalog                 # Katalog anzeigen und prüfen (Stufen, Abweichung vom Basislauf)
kinneret-recon katalog --faktoren      # Faktor-Register
kinneret-recon batch -n                # zeigen, welche Läufe fehlen oder veraltet sind
kinneret-recon batch                   # fehlende Läufe rechnen (results/<lauf>/), dann app-daten (~2 h für alle)
kinneret-recon batch o_cru_r25 o_cru_r50   # nur diese Läufe
kinneret-recon app-daten               # app.json, reference/app/katalog.json und Standortdaten neu schreiben
```

**Ketten sichern und wiederverwenden:** Die Ketten (`posterior.npz`, `prior.npz`, ~720 MB für alle Läufe) liegen
nicht in diesem Repo, sondern im privaten Daten-Repo `kinneret_reconstruction_laeufe`, abgelegt nach ihrem
Fingerabdruck (`ketten/<fingerabdruck>/`).

```bash
git clone https://github.com/TimonNetzel/kinneret_reconstruction_laeufe ../kinneret_reconstruction_laeufe
kinneret-recon batch --ketten ../kinneret_reconstruction_laeufe     # Ketten neu auswerten statt sampeln (~15 min statt ~2 h)
kinneret-recon ketten sichern ../kinneret_reconstruction_laeufe     # neue Ketten aus results/ ablegen, dann dort committen
```

`batch` erkennt am Fingerabdruck der Einstellungen, ob vorhandene Ergebnisse passen, und sampelt Läufe mit
gleichen Ketten (z. B. Mittelwert und Mischverteilung) nur einmal. Ein neuer Lauf ist eine Konfiguration in
`configs/` plus ein Ordner mit `lauf.toml`; ein neuer Faktor ein Eintrag in `FAKTOREN`.

## App (offline)

Die App zeigt die gerechneten Läufe im Browser, auf PC und Handy, auf Deutsch und Englisch, hell oder dunkel.
Sie braucht kein Netz.

```bash
kinneret-recon app                     # Ordner results/app/: index.html per Doppelklick öffnen
kinneret-recon app --eine-datei        # results/kinneret-app.html (~5 MB): aufs Handy kopieren, in Chrome öffnen
kinneret-recon app --serve             # Ordner bauen und im WLAN bereitstellen (http://<PC>:8000/)
node tools/app_pruefen.js results/app/index.html results/kinneret-app.html   # Browser-Test (Playwright)
```

Ansichten: **Rekonstruktion** (T, P, Biomanteile, Baumpollenkurve über Alter oder Tiefe; Mittelwert oder
Mischverteilung; Auswahl „Detail“ zeigt Verteilung, Biomanteile und Spektrum zu einem Alter) und **Sensitivität**
(Faktor und Basislauf wählen, Varianten überlagern, Differenzen, Tabelle mit Niveau, IQR und Korrelation der
Anomalien), **Taxa-Gewichte** (Posterior gegen Prior, Vergleich mehrerer Läufe), **Transferfunktionen**
(Klimaräume der Biome, CI-Verhältnis) und **Lauf** (Steckbrief, Methodentext de/en zum Kopieren, Kalibrierung,
Einstellungen, Konvergenzdiagnostik). Formelzeichen stehen in den Texten in LaTeX-Schreibweise
(`$T_\mathrm{DJF}$`) und werden mit echten Indizes gesetzt (`formel.py`). Die App liest `reference/app/` und `reference/laeufe/*/app.json`; nach neuen Läufen
`kinneret-recon app` erneut aufrufen.

## Prior des rezenten Klimas aus CRU-Daten

Der Prior der jüngsten Altersstufe (`sampler.recent_*`) lässt sich reproduzierbar für ein festgelegtes
Einzugsgebiet berechnen; die Voreinstellung ist das Ergebnis für 50 km. Grundlage ist ein Ausschnitt der Levante aus CRU TS 4.07 (Monatswerte 1951–2020,
0,5°) und CRU CL 2.0 (10′, mit Höhenmodell) in `data/in/cru_levante.npz` (0,5 MB). Lage, Höhe, Ausschnitt und
Einzugsgebiet kommen aus dem Standort-Register (`[cru]` in `standort.toml`, Auswahl mit `--standort`); die Optionen
überschreiben sie.

```bash
kinneret-recon cru-prior -r 25 50 100                  # Mittel und Unsicherheit für drei Radien
kinneret-recon cru-prior -r 50 --toml prior.toml       # als [sampler]-Abschnitt für eine Konfiguration
kinneret-recon cru-prior -r 50 --kernel uniform --period 1991 2020 --cells zellen.csv
kinneret-recon cru-extract roh/ --download             # Ausschnitt neu erzeugen (lädt ~320 MB)
```

Das Einzugsgebiet ist ein Radius um den See mit Entfernungsgewichtung (`gauss` mit σ = R/2, `uniform`
oder `linear`). Die Sd ist die Unsicherheit des Gebietsmittels, nicht die räumliche Streuung der Zellen:
Stichprobenfehler des 30-Jahres-Mittels, Repräsentativität (andere Radien und Gewichtungen), Datenfehler
(CRU TS gegen CRU CL 2.0) und der Abstand zum Zeitraum der jüngsten Altersstufe (1991–2020). Optional sind
eine Höhenkorrektur von T (`--lapse-rate`) und ein Zusatzfehler. Details: `docs/pruefbericht.md`,
Abschnitt 3.9.

## Leistung (4 CPU-Kerne)

| | Original | Paket, `--legacy` | Paket, Standard v2 | Paket, Standard v3 |
|---|---|---|---|---|
| Sampling | ~60 s, 1 Kette, ~3 GB RAM | 36 s, 1 Kette | ~90 s, 4 Ketten × 2 Mio. | ~5,4 min, 4 Ketten × 8 Mio. |
| Prior-Referenzlauf | – | – | ~90 s | ~5,1 min |
| Auswertung + Abbildungen | 576 s | ~20 s | ~45 s | ~40 s |
