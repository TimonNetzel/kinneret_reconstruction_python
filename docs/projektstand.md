# Projektstand

Stand: 28.10.2026. Dieses Dokument fasst Ziel, Plan, Ergebnisse, Entscheidungen und offene Punkte
zusammen, damit die Arbeit jederzeit wieder aufgenommen werden kann.

## Ziel

Den Code zur Publikation Netzel et al. (2025), *Clim. Past* 21, 357–380
(https://doi.org/10.5194/cp-21-357-2025) prüfen und verbessern. Daraus eine App bauen, mit der sich
Rekonstruktionen für weitere Arbeiten und Publikationen automatisiert rechnen lassen.

Rahmen (vom Autor festgelegt):

- Nutzung nur lokal durch den Autor, Code in einem privaten GitHub-Repo.
- Vorerst nur Kinneret. Die Transferfunktion (neuronales Netz) gilt als gegeben, ihr Training kommt
  in einem späteren Schritt.
- Korrekturen direkt einbauen, das Original bleibt erhalten.
- Dokumentation auf Deutsch.
- Entwicklung bis 27.10.2026 im Branch `claude/netzel-python-package-analysis-adcv39`, mit Pull
  Request #1 nach `main` gemergt (Merge-Commit, die Commit-Hashes in der Doku bleiben gültig).
  Weitere Arbeit in neuen Branches ab `main`.

## Plan und Stand

| Schritt | Inhalt | Stand |
|---|---|---|
| 1 | Original lauffähig machen, Referenzlauf sichern | ✅ `legacy/`, `reference/v1.0-paper/` |
| 2 | Paper Gleichung für Gleichung gegen den Code prüfen | ✅ `docs/abgleich_paper.md`, Befunde 12–16 |
| 3 | Umbau zum Paket ohne Verhaltensänderung | ✅ Legacy-Modus bitidentisch (Tests) |
| 4 | Korrekturen einzeln, mit Vorher/Nachher-Vergleich | ✅ `docs/pruefbericht.md`, `reference/laeufe/` |
| 5 | Performance | ✅ Auswertung 576 s → 10 s; MCMC 60 s → 37 s (Legacy) |
| 6 | Vollständiger Lauf, Bericht | ✅ `reference/v2-standard/`, `reference/laeufe/g_dir_hierarchisch/` |
| – | Flexibler Prior der Taxa-Gewichte | ✅ Dir(α) fest oder hierarchisch |
| – | CRU-Prior aus festgelegtem Einzugsgebiet | ✅ `kinneret-recon cru-prior`, Prüfbericht 3.9 |
| – | Neuer Standard v3 (hierarchischer Prior, CRU 50 km) | ✅ `reference/v3-standard/`, Prüfbericht 3.10 |
| – | Tests Klima gegen Landnutzung (AP ohne Kulturbäume, Kalibrierung 4–9 ka) | ✅ Prüfbericht 3.11 |
| App 1 | Lauf-Katalog, Faktor-Register, kompakte Zusammenfassungen, Batch-Befehl; alle Läufe neu gerechnet | ✅ `docs/app_konzept.md` Abschnitt 7 |
| App 2 | App-Grundgerüst (statisch, offline, responsiv, de/en) mit Rekonstruktion und Sensitivität | ✅ `kinneret-recon app`, App-Konzept Abschnitt 9 |
| App 3 | Taxa-Gewichte, Transferfunktionen, Diagnostik, Steckbrief | ✅ App-Konzept Abschnitt 10 |
| App 4 | Abbildungsexport im Copernicus-Format (PDF, SVG, PNG), CSV | ✅ App-Konzept Abschnitt 12 |
| App 5 | Standort-Register (Kinneret als erster Standort) | offen |

## Wichtigste Ergebnisse (Details: `docs/pruefbericht.md`)

1. **Fehler in der Hastings-Korrektur der Taxa-Gewichte:** κ fehlt. Belegt mit einem Prior-only-Test.
2. **Klimarekonstruktion robust:** Unter allen Korrekturen und Priors ändern sich die Mediane um
   ≤ 0,17 °C und ≤ 12 mm. Die Unsicherheit von P war um etwa 10–15 % unterschätzt.
3. **CI-Verhältnis der Transferfunktionen:** Für das IT- und das SA-Biom ist die Verengung ein
   Trunkierungseffekt. Echte Information aus den Daten gibt es nur für das mediterrane Biom (~20 %).
4. **Taxa-Gewichte:** Die Daten bestimmen sie nur schwach. Die Rangfolge ist ab α ≥ 1 bzw. im
   hierarchischen Modell stabil (*Amaranthaceae*, *Q. calliprinos* ↑; *Olea* ↓), die Beträge hängen
   vom Prior ab.
5. **Abgleich mit dem Paper:** Der Originalcode reproduziert die Zahlen des Papers. κ fehlt nur im
   Code (das Paper beschreibt den Vorschlag nicht). Der Sensitivitätslauf mit R² = 0,25 bleibt im
   Python-Code hängen. Die Unsicherheitsbänder zeigen den Mittelwert statt der Mischverteilung aus
   Gl. (3) (IQR bei P etwa ein Drittel).
6. **Neuer Sampler (log-Gamma-Random-Walk):** gleiche Zielverteilung, etwa 50-mal effizienter als
   der Dirichlet-Random-Walk.
7. **Prior des rezenten Klimas:** Die bisherige Sd (2,48 °C, 135 mm) war die räumliche Streuung der
   CRU-Zellen. Als echte Unsicherheit eines 50-km-Einzugsgebiets ergeben sich 11,09 ± 0,87 °C und
   522 ± 41 mm. Damit sinkt der IQR der Rekonstruktion um ~30 %, das T-Niveau steigt um 0,8 °C, das
   P-Niveau bleibt (−2 mm), der P-Verlauf ist unverändert (*r* = 0,999).
8. **Standard v3** (hierarchischer Prior der Gewichte, CRU-Prior 50 km): gegenüber v2 T +0,74 °C,
   P +2 mm, IQR von P −36 %, P-Verlauf unverändert (*r* = 0,998). Gewichte wie im hierarchischen Lauf.
9. **Klima oder Landnutzung:** Die Baumpollenkurve ist vor allem eine *Olea*-Kurve (ohne Kulturbäume
   *r* = −0,06 mit der vollen Kurve). Das P-Muster hält trotzdem (*r* = 0,90 ohne Kulturbäume); die
   feuchten Phasen sind ohne *Olea* um 21–27 mm schwächer.

## Entscheidungen

| Datum | Entscheidung |
|---|---|
| 26.10.2026 | Originalcode bleibt erhalten (`legacy/`, Commit `782c6a0`), Korrekturen kommen ins neue Paket |
| 26.10.2026 | C++ bleibt für den Sampler; die Auswertung läuft in NumPy und C++ (Dichteschätzung) |
| 26.10.2026 | Der Prior der Taxa-Gewichte muss nicht starr sein. Der Autor bevorzugt den **hierarchischen Prior** (sauberer) |
| 27.10.2026 | Rekonstruktion: Mittelwert und Mischverteilung (Gl. 3) als Option. **Mittelwert ist Standard** (vom Autor bestätigt) |
| 27.10.2026 | Gl. (14): N_t als Prior wählbar, Standard bleibt „nur Vorschlag“ (sonst doppelte Zählung der TF) |
| 27.10.2026 | R²-Prior: Beta(3; 3) bleibt, Beta(2,62; 2,62) liefert dasselbe Ergebnis |
| 27.10.2026 | Branch mit PR #1 nach `main` gemergt (vom Autor freigegeben) |
| 27.10.2026 | **Standard v3:** hierarchischer Prior der Taxa-Gewichte und CRU-Prior für 50 km (11,09 ± 0,87 °C, 521,8 ± 41,4 mm), 8 Mio. Iterationen je Kette (vom Autor festgelegt). Vergleichskonfigurationen schreiben die Werte bis v2 fest |
| 28.10.2026 | App im Browser (PC und Handy, de/en), vorerst nur fertige Läufe, erweiterbar für Läufe und Standorte (`docs/app_konzept.md`). „AP ohne Kulturbäume“ als Variante, Standard bleibt v3 |
| 28.10.2026 | Ketten aller Läufe im privaten Daten-Repo `kinneret_reconstruction_laeufe` (nach Fingerabdruck; `kinneret-recon ketten sichern`, `batch --ketten`), nicht im Haupt-Repo (Größe) |
| 28.10.2026 | Formelzeichen in LaTeX-Schreibweise mit echten Indizes in App, Methodentext und Bericht ($T_\mathrm{DJF}$ statt T_DJF; vom Autor gewünscht, App-Konzept Abschnitt 11); Methodentext auch als LaTeX kopierbar |
| 28.10.2026 | App offline-fähig: Ordner per Doppelklick (PC), eine HTML-Datei fürs Android-Handy, PWA über localhost/HTTPS; Technik statische Web-App mit Plotly.js (vom Autor festgelegt, App-Konzept Abschnitt 8) |
| 28.10.2026 | App-Daten als JSON (`app.json` je Lauf, < 1 MB), Dichten auf dem Raster T −10…30 °C (0,5 °C) und P 0…1500 mm (20 mm); Lauf-Beschreibungen in `reference/laeufe/<lauf>/lauf.toml` |
| 28.10.2026 | Abbildungsexport: Breiten 8,3 / 12 / 17,5 cm, Schrift 8 pt serifenlos, PNG 300/600 dpi, PDF über den Druckdialog des Browsers (Vektor, ohne Zusatzbibliothek), SVG; CSV je Diagramm. Werte aus dem Paper nachgemessen, da die Copernicus-Seiten aus der Sitzung nicht erreichbar waren (App-Konzept Abschnitt 12) |
| 27.10.2026 | Prior des rezenten Klimas einstellbar. Er bestimmt das absolute Niveau (bis 1,1 °C / 89 mm), der P-Verlauf ist robust |

## Offene Punkte

Erledigte Punkte (Antworten des Autors zu Befund 12, 14 und 15, CRU-Prior, hierarchischer Prior, R-Fassung) stehen in
Prüfbericht 3.8–3.10, `docs/abgleich_paper.md` Abschnitt 5 und unter „Entscheidungen“.

1. **Zusatzfehler für den Niederschlag im CRU-Prior?** (`kinneret-recon cru-prior -r 50 --extra-pann-rel-sd 0.1`:
   522 ± 67 mm statt 522 ± 41 mm; Entscheidung des Autors)
2. **Prior-Referenzlauf kürzen:** Er dient nur dem CI-Verhältnis der Transferfunktionen und macht die Hälfte der
   11 min des Standardlaufs aus; z. B. 2 Mio. Iterationen.
3. **Biomfarben bei Farbsehschwäche** (App-Konzept Abschnitt 12, „Offen“; Entscheidung des Autors).
4. **Export-Maße** mit den aktuellen Copernicus-Richtlinien abgleichen (Seiten aus der Cloud-Umgebung bisher nicht
   erreichbar).
5. **Befund 13:** Wie entstanden die $R^2$ = 0,25-Kurven im Paper? Beide Code-Fassungen hängen mit Beta(0,92; 2,77);
   der Autor weiß es nicht mehr sicher.
6. **Tag `v1.0-paper`** ist entbehrlich: `v0.1.0` enthält denselben Code (nur `LICENSE.md` anders) und entspricht
   der Zenodo-Version des Papers.
7. **Später:** Training der Transferfunktion (dieselbe CRU-Version TS 4.07, 1961–1990, 0,5° wie für den Prior),
   weitere Standorte (nach App Schritt 5).

## Reproduktion

```bash
pip install -e ".[test]"
pytest                                    # schnelle Tests (~1 min)
pytest -m slow                            # lange Tests (Bitgleichheit mit dem Original, hierarchischer Prior)
kinneret-recon run --legacy               # Original, ~1 min
kinneret-recon run -c configs/standard.toml     # Standard v3, ~11 min
kinneret-recon run -c configs/standard_v2.toml  # Standard bis v2, ~4 min
```

Alle Läufe der Vergleichsreihe: `configs/*.toml`. Gesicherte Ergebnisse (Tabellen, Diagnostik,
Einstellungen, Metadaten): `reference/laeufe/<lauf>/`, Vergleiche in `reference/vergleiche/`.
Die großen Dateien (`posterior.npz`, `results.npz`, Abbildungen) sind nicht im Repo. Jeder Lauf ist
mit seiner `config.toml` (inkl. Seed) und demselben Code-Stand (`meta.json`) bitgenau wiederholbar.
Referenzläufe der Standards: `reference/v1.0-paper/` (Original), `reference/v2-standard/`,
`reference/v3-standard/`.

## Arbeitsumgebung

- In der Sitzung vom 27.10.2026 (Nachmittag) waren `crudata.uea.ac.uk` und `dap.ceda.ac.uk`
  erreichbar, `data.ceda.ac.uk` nicht (403). Ob `cp.copernicus.org` und `zenodo.org` erreichbar sind,
  hängt von der Umgebung ab. Das Paper liegt in `docs/paper/`, der CRU-Ausschnitt in
  `data/in/cru_levante.npz`; für die Arbeit sind keine weiteren Downloads nötig.
- Gemessen auf 4 CPU-Kernen mit Python 3.13, NumPy 2.5, SciPy 1.18, Matplotlib 3.11 und g++ 13.

## Übergabe an eine neue Sitzung

Stand 28.10.2026, App Schritt 4 abgeschlossen und per Pull Request #7 nach `main` gemergt. Weiterarbeit in einem
neuen Branch ab `main`. Alles Nötige steht im Repo. **Zum Einstieg nur diesen Abschnitt lesen**, weitere Dokumente gezielt nach Bedarf (Verweise unten).

**Nächster Schritt: App Schritt 5, Standort-Register** (Kinneret als einziger Standort). Lesen: App-Konzept
Abschnitt 2 Punkt 4 und Abschnitt 7. Heute ist der Standort fest verdrahtet in `cru.py`, `katalog.py`
(Standort-Register), `app_daten.py` und `data/in/`; die App-Daten des Standorts stehen in
`reference/app/standort_kinneret.json`.

- **Fingerabdrücke müssen unverändert bleiben**, sonst findet `batch --ketten` die gesicherten Ketten nicht mehr
  (~2 h Neurechnen). Ebenso sollen `reference/**/app.json` und `standort_kinneret.json` bitgleich bleiben.
- Prüfen ohne Neurechnen: `kinneret-recon katalog` (Fingerabdrücke), `kinneret-recon app-daten` + `git diff`,
  `python3 -m pytest`. Kein `batch`, keine Läufe nötig.
- Den Standort-Wähler der App nur vorbereiten (er erscheint erst ab zwei Standorten).
- Biomfarben gehören ins Standort-Register; ob sie sich ändern, ist offen (Offene Punkte, Punkt 3).

**Einrichten (ca. 5 min):** `pip install -e ".[test]"` (baut den C++-Kern), dann `python3 -m pytest`
(49 Tests, ~4 min; Browser-Test nur mit Node und Playwright). App bauen und prüfen: `kinneret-recon app`,
`kinneret-recon app --eine-datei`, `node tools/app_pruefen.js results/app/index.html results/kinneret-app.html`.

**Ketten im Daten-Repo:** Die Ketten aller 27 Läufe liegen im privaten Repo `TimonNetzel/kinneret_reconstruction_laeufe`
(`ketten/<fingerabdruck>/`, Zuordnung in `index.json`, 719 MB). Nur für Neuauswertungen nötig: Repo anhängen und
klonen, dann `kinneret-recon batch --ketten ../kinneret_reconstruction_laeufe` (~15 min statt ~2 h). Neue Ketten mit
`kinneret-recon ketten sichern ../kinneret_reconstruction_laeufe` ablegen, ein Commit je Kettensatz.

**Konventionen:** Doku auf Deutsch. Formelzeichen in Texten in LaTeX-Schreibweise zwischen `$...$` (App-Konzept
Abschnitt 11). Eine neue App-Ansicht: App-Konzept Abschnitt 10; ein exportierbares Diagramm: `KR.plot`, Abschnitt 12.

**Wo steht was:** Ergebnisse und Befunde im Prüfbericht (`docs/pruefbericht.md`); App: Schritt 1 Abschnitt 7,
Offline Abschnitt 8, Schritt 2 Abschnitt 9, Schritt 3 Abschnitt 10, Formeln Abschnitt 11, Schritt 4 Abschnitt 12;
Abgleich mit dem Paper und der R-Fassung in `docs/abgleich_paper.md`; Entscheidungen und offene Punkte oben.

**Danach:** siehe „Offene Punkte“.

**Cloud-Umgebung:**

- `python3 -m pytest` statt `pytest` (anderes Python ohne NumPy).
- Lange Befehle im Hintergrund; `kinneret-recon` puffert Ausgaben in Dateien bis zum Ende. `pgrep -f "<muster>"`
  findet die eigene Warteschleife mit, daher Muster wie `"configs/[x]yz"`.
- Tags lassen sich über den Git-Proxy nicht pushen, Branches schon.
- Copernicus-Seiten waren nicht erreichbar (für Offene Punkte, Punkt 4, Domains in den Netzwerkeinstellungen
  freigeben); erreichbar waren `crudata.uea.ac.uk` und `dap.ceda.ac.uk`.
