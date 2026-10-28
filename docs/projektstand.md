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
| App 2 | App-Grundgerüst (statisch, responsiv, de/en) mit Rekonstruktion und Sensitivität | offen |

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
| 28.10.2026 | App-Daten als JSON (`app.json` je Lauf, < 1 MB), Dichten auf dem Raster T −10…30 °C (0,5 °C) und P 0…1500 mm (20 mm); Lauf-Beschreibungen in `reference/laeufe/<lauf>/lauf.toml` |
| 27.10.2026 | Prior des rezenten Klimas einstellbar. Er bestimmt das absolute Niveau (bis 1,1 °C / 89 mm), der P-Verlauf ist robust |

## Offene Punkte (nächste Sitzung)

1. **Antworten des Autors zum Abgleich (27.10.2026) eingearbeitet**, siehe Prüfbericht 3.8:
   - Befund 12: beide Varianten umgesetzt, Empfehlung Mittelwert.
   - Befund 14: Autor weiß es nicht mehr. N_t als Prior getestet; Empfehlung: nur Vorschlag.
   - Befund 15: getestet, kein Unterschied; Beta(3; 3) bleibt.
   - CRU-Prior: Mittel mehrerer Levante-Gitterpunkte. Bewertet; bestimmt das absolute Niveau.
   - Befund 13: Die Paper-Ergebnisse stammen aus der R-Fassung, Befund 13 betrifft nur Python.

   **R-Fassung geprüft** (`TimonNetzel/kinneret_reconstruction_R`, `docs/abgleich_paper.md`
   Abschnitt 5): gleicher C++-Kern und gleiche Daten. Befund 1 betrifft damit die Ergebnisse des
   Papers direkt, und Befund 13 gilt auch für R.

   **CRU-Prior aus einem festgelegten Einzugsgebiet: erledigt** (27.10.2026, Prüfbericht 3.9), seit
   v3 Standard. **Noch offen:** ob ein Zusatzfehler für den Niederschlag eingehen soll
   (`kinneret-recon cru-prior -r 50 --extra-pann-rel-sd 0.1`: 522 ± 67 mm).
2. ~~Standard auf den hierarchischen Prior umstellen~~: erledigt (v3, Prüfbericht 3.10).
3. **Tag `v1.0-paper`:** Er ist nicht mehr nötig. `v0.1.0` enthält denselben Code (Unterschied nur
   `LICENSE.md`) und entspricht der Zenodo-Version des Papers.
4. **Laufzeit des Standardlaufs (v3):** 11 min. Der Prior-Referenzlauf (nur für das CI-Verhältnis der
   Transferfunktionen) nimmt die Hälfte davon ein und ließe sich kürzen, z. B. auf 2 Mio. Iterationen.
5. **Später:** Training der Transferfunktion, weitere Standorte.

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

Stand 28.10.2026 (dritte Sitzung, App Schritt 1), letzter Commit siehe `git log`. Alles Nötige steht im Repo;
aus der alten Sitzung muss nichts übernommen werden.

**Nächster Schritt: App Schritt 2** (`docs/app_konzept.md`, Abschnitt 5): App-Grundgerüst mit Rekonstruktion
und Sensitivität. Die Daten dafür liegen fertig im Repo: `reference/app/katalog.json` (Register und alle
Läufe mit Kennzahlen), `reference/app/standort_kinneret.json` und `reference/laeufe/<lauf>/app.json`.
Vorher die Technik entscheiden (Abschnitt 6).

**Erledigt in dieser Sitzung: App Schritt 1** (`docs/app_konzept.md`, Abschnitt 7)

- Module `katalog.py` (Lauf-Katalog, Faktor- und Standort-Register, Prüfung, Fingerabdruck) und
  `app_daten.py` (Zusammenfassungen); Befehle `kinneret-recon katalog`, `batch`, `app-daten`; Tests
  `tests/test_katalog.py`.
- 27 Lauf-Beschreibungen `reference/laeufe/*/lauf.toml`, neue Konfigurationen `configs/original.toml`,
  `configs/n_mixture_v2.toml`, `configs/v3_mixture.toml`, neuer Ordner `reference/laeufe/standard_v3/`.
- Alle Läufe neu gerechnet (`kinneret-recon batch`), Zusammenfassungen im Repo. 21 von 27 Läufen sind bitgleich
  mit den gesicherten Tabellen; die übrigen stammen aus älteren Code-Ständen und weichen nur durch die
  Zufallszahlenfolge ab (≤ 0,06 °C, ≤ 3,1 mm; Mischverteilung v2 ≤ 7,6 mm), `docs/app_konzept.md` Abschnitt 7.
- Fehler behoben: `write_toml` ließ `recent_* = None` (Prior aus den Eingangsdaten) weg. Seit v3 las
  `Config.from_toml` das als CRU-Prior; betroffen waren geschriebene `config.toml` von Läufen mit dem
  Original-Prior, z. B. bei `--from-run` mit `results/legacy`. Jetzt steht dort `"daten"`. Die gesicherten
  `reference/laeufe/*/config.toml` aus der Zeit vor v3 ohne `recent_*` bedeuten den Prior aus den
  Eingangsdaten.

**Einrichten (ca. 5 min):**

```bash
pip install -e ".[test]"   # baut den C++-Kern
python3 -m pytest          # 41 Tests, ~1 min
```

Die Ergebnisordner `results/` sind nicht im Repo. Wer Ketten für Neuauswertungen braucht
(`--from-run`), rechnet sie neu: `kinneret-recon run` (Standard v3, ~11 min),
`kinneret-recon run -c configs/standard_v2.toml` (~4 min), `kinneret-recon run --legacy` (~1 min).

**Erledigt (zweite Sitzung): CRU-Prior aus einem festgelegten Einzugsgebiet** (Prüfbericht 3.9)

- Modul `src/kinneret_recon/cru.py`, Befehle `kinneret-recon cru-extract` und `cru-prior`, Tests
  `tests/test_cru.py`, Ausschnitt `data/in/cru_levante.npz` (CRU TS 4.07 1951–2020 und CRU CL 2.0).
- Läufe `configs/o_cru_r25.toml`, `o_cru_r50.toml`, `o_cru_r100.toml`; Ergebnisse in
  `reference/laeufe/o_cru_r*/`, Vergleich in `reference/vergleiche/cru_radius/`
  (`python tools/anomalien.py standard_aktuell o_cru_r25 o_cru_r50 o_cru_r100`).
- Die bisherigen Werte 10,74 ± 2,48 °C / 556 ± 135 mm sind Mittel und räumliche Sd von etwa 3–8
  Zellen im Umkreis von 50–100 km; die genaue Auswahl ist nicht eindeutig bestimmbar.

**Erledigt (zweite Sitzung): Standard v3** (Prüfbericht 3.10, vom Autor festgelegt)

- `SamplerConfig`: `taxa_prior = "hierarchical"`, 8 Mio. / 2 Mio. / 600, Prior des rezenten Klimas
  `CRU_PRIOR_50KM` (`src/kinneret_recon/config.py`); `configs/standard.toml` = Voreinstellung.
- Alle Vergleichskonfigurationen schreiben die Werte bis v2 fest (bitgleich geprüft), der alte
  Standard heißt jetzt `configs/standard_v2.toml`.
- Referenzlauf `reference/v3-standard/`, Vergleich in `reference/vergleiche/v3/`, Mischverteilung
  nach Gl. (3) in `reference/laeufe/v3_mixture/` (drei Gipfel beim Niederschlag, Prüfbericht 3.10).
- Abbildung `independent_proposal_dists` korrigiert (Prior-Kurve von T0, Legende).

**Erledigt (zweite Sitzung): Tests Klima gegen Landnutzung** (Prüfbericht 3.11)

- Optionen `sampler.ap_exclude_taxa` und `sampler.calibration_age_range` (Maske im C++-Kern),
  `data.ARBOREAL_TAXA`, `data.CULTIVATED_TAXA`; Läufe `configs/p_*.toml`.
- Offen: Entscheidung des Autors, ob AP ohne Kulturbäume Standard wird.

**Mögliche nächste Schritte:**

0. App nach `docs/app_konzept.md`, weiter mit Schritt 2.
1. Prior-Referenzlauf kürzen (halbiert die Laufzeit des Standardlaufs, siehe „Offene Punkte“, Punkt 4).
2. Zusatzfehler für den Niederschlag im CRU-Prior entscheiden (Punkt 1).
3. Training der Transferfunktion; dabei dieselbe CRU-Version (TS 4.07, 1961–1990, 0,5°) verwenden
   wie für den Prior.

**Weitere offene Punkte:** siehe „Offene Punkte“ oben, außerdem:

- Befund 13: Wie entstanden die R² = 0,25-Kurven im Paper? Beide Code-Fassungen hängen mit
  Beta(0,92; 2,77). Der Autor weiß es nicht mehr sicher.
- Pull Request #1 (Branch → `main`) ist mit Zustimmung des Autors gemergt (27.10.2026).

**Technische Hinweise für die Arbeit in der Cloud-Umgebung:**

- `pgrep -f "<muster>"` findet auch die eigene Warteschleife, deren Befehlszeile das Muster
  enthält. Deshalb Muster wie `"configs/[x]yz"` verwenden oder auf eine Datei warten.
- Ausgaben von `kinneret-recon` in Dateien sind gepuffert und erscheinen erst am Ende.
- Lange Läufe im Hintergrund starten; 4 Ketten × 2 Mio. dauern ~1,5 min Sampling plus ~1,5 min
  Prior-Referenzlauf plus ~0,5 min Auswertung.
- Ein Push von Tags ist über den Git-Proxy nicht möglich, Branches schon.
- In der Cloud-Umgebung zeigt `pytest` auf eine andere Python-Installation ohne NumPy; `python3 -m pytest`
  verwenden.
- `kinneret-recon batch` (alle Läufe) dauert ~2 h und sollte im Hintergrund laufen; abgebrochene Läufe
  rechnet ein erneuter Aufruf nach, fertige erkennt er am Fingerabdruck.
