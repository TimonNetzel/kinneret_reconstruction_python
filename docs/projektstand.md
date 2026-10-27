# Projektstand

Stand: 27.10.2026. Dieses Dokument fasst Ziel, Plan, Ergebnisse, Entscheidungen und offene Punkte
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
- Entwicklung im Branch `claude/netzel-python-package-analysis-adcv39`.

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

## Entscheidungen

| Datum | Entscheidung |
|---|---|
| 26.10.2026 | Originalcode bleibt erhalten (`legacy/`, Commit `782c6a0`), Korrekturen kommen ins neue Paket |
| 26.10.2026 | C++ bleibt für den Sampler; die Auswertung läuft in NumPy und C++ (Dichteschätzung) |
| 26.10.2026 | Der Prior der Taxa-Gewichte muss nicht starr sein. Der Autor bevorzugt den **hierarchischen Prior** (sauberer) |
| 27.10.2026 | Rekonstruktion: Mittelwert und Mischverteilung (Gl. 3) als Option. **Mittelwert ist Standard** (vom Autor bestätigt) |
| 27.10.2026 | Gl. (14): N_t als Prior wählbar, Standard bleibt „nur Vorschlag“ (sonst doppelte Zählung der TF) |
| 27.10.2026 | R²-Prior: Beta(3; 3) bleibt, Beta(2,62; 2,62) liefert dasselbe Ergebnis |
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

   **CRU-Prior aus einem festgelegten Einzugsgebiet: erledigt** (27.10.2026, Prüfbericht 3.9).
   **Noch offen:** Entscheidung des Autors, ob der Prior für 50 km (`configs/o_cru_r50.toml`)
   Standard wird, und ob ein Zusatzfehler für den Niederschlag eingehen soll.
2. **Standard auf den hierarchischen Prior umstellen:** In `SamplerConfig` steht noch
   `taxa_prior = "fixed"`. Umzustellen sind `taxa_prior = "hierarchical"` und die
   Standardlänge (8 Mio. Iterationen, Burn-in 2 Mio., Ausdünnung 600), danach Tests und README.
   Bis dahin den hierarchischen Prior über `configs/g_dir_hierarchisch.toml` verwenden.
3. **Tag `v1.0-paper`:** Er ist nicht mehr nötig. `v0.1.0` enthält denselben Code (Unterschied nur
   `LICENSE.md`) und entspricht der Zenodo-Version des Papers.
4. **Laufzeit des hierarchischen Laufs:** etwa 13 min. Der Prior-Referenzlauf nimmt die Hälfte davon
   ein und ließe sich kürzen.
5. **Später:** Training der Transferfunktion, weitere Standorte, Mergen des Branches in `main`.

## Reproduktion

```bash
pip install -e ".[test]"
pytest                                    # schnelle Tests (~45 s)
pytest -m slow                            # lange Tests (Bitgleichheit mit dem Original, hierarchischer Prior)
kinneret-recon run --legacy               # Original, ~1 min
kinneret-recon run -c configs/standard.toml
kinneret-recon run -c configs/g_dir_hierarchisch.toml
```

Alle Läufe der Vergleichsreihe: `configs/*.toml`. Gesicherte Ergebnisse (Tabellen, Diagnostik,
Einstellungen, Metadaten): `reference/laeufe/<lauf>/`, Vergleiche in `reference/vergleiche/`.
Die großen Dateien (`posterior.npz`, `results.npz`, Abbildungen) sind nicht im Repo. Jeder Lauf ist
mit seiner `config.toml` (inkl. Seed) bitgenau wiederholbar.

## Arbeitsumgebung

- In der Sitzung vom 27.10.2026 (Nachmittag) waren `crudata.uea.ac.uk` und `dap.ceda.ac.uk`
  erreichbar, `data.ceda.ac.uk` nicht (403). Ob `cp.copernicus.org` und `zenodo.org` erreichbar sind,
  hängt von der Umgebung ab. Das Paper liegt in `docs/paper/`, der CRU-Ausschnitt in
  `data/in/cru_levante.npz`; für die Arbeit sind keine weiteren Downloads nötig.
- Gemessen auf 4 CPU-Kernen mit Python 3.13, NumPy 2.5, SciPy 1.18, Matplotlib 3.11 und g++ 13.

## Übergabe an eine neue Sitzung

Stand 27.10.2026 (zweite Sitzung), letzter Commit siehe `git log`. Alles Nötige steht im Repo; aus der
alten Sitzung muss nichts übernommen werden.

**Einrichten (ca. 5 min):**

```bash
pip install -e ".[test]"   # baut den C++-Kern
pytest                     # 26 Tests, ~1,5 min
```

Die Ergebnisordner `results/` sind nicht im Repo. Wer Ketten für Neuauswertungen braucht
(`--from-run`), rechnet sie neu: `kinneret-recon run -c configs/standard.toml` (~4 min),
`kinneret-recon run --legacy` (~1 min).

**Erledigt in dieser Sitzung: CRU-Prior aus einem festgelegten Einzugsgebiet** (Prüfbericht 3.9)

- Modul `src/kinneret_recon/cru.py`, Befehle `kinneret-recon cru-extract` und `cru-prior`, Tests
  `tests/test_cru.py`, Ausschnitt `data/in/cru_levante.npz` (CRU TS 4.07 1951–2020 und CRU CL 2.0).
- Läufe `configs/o_cru_r25.toml`, `o_cru_r50.toml`, `o_cru_r100.toml`; Ergebnisse in
  `reference/laeufe/o_cru_r*/`, Vergleich in `reference/vergleiche/cru_radius/`
  (`python tools/anomalien.py standard_aktuell o_cru_r25 o_cru_r50 o_cru_r100`).
- Die bisherigen Werte 10,74 ± 2,48 °C / 556 ± 135 mm sind Mittel und räumliche Sd von etwa 3–8
  Zellen im Umkreis von 50–100 km; die genaue Auswahl ist nicht eindeutig bestimmbar.

**Mögliche nächste Schritte:**

1. Entscheidung des Autors zum CRU-Prior (Empfehlung: 50 km, Gauß; ggf. mit
   `--extra-pann-rel-sd 0.1`). Wird er Standard, die Werte in `SamplerConfig` (oder in
   `nnet_fit_params.npz`) eintragen, Tests und README anpassen.
2. Standard auf den hierarchischen Prior der Gewichte umstellen (siehe „Offene Punkte“, Punkt 2);
   sinnvoll zusammen mit Punkt 1 in einem neuen Referenzlauf `reference/v3-standard/`.
3. Training der Transferfunktion; dabei dieselbe CRU-Version (TS 4.07, 1961–1990, 0,5°) verwenden
   wie für den Prior.

**Weitere offene Punkte:** siehe „Offene Punkte“ oben, außerdem:

- Befund 13: Wie entstanden die R² = 0,25-Kurven im Paper? Beide Code-Fassungen hängen mit
  Beta(0,92; 2,77). Der Autor weiß es nicht mehr sicher.
- Pull Request #1 (Branch → `main`) ist offen, CI grün. Mergen, wenn der Autor zustimmt. Eine neue
  Sitzung muss die PR-Beobachtung ggf. neu abonnieren.
- Optional: hierarchischen Prior der Gewichte als Standard (der Autor bevorzugt ihn, Standard ist
  noch Dir(0,5) wie im Paper).

**Technische Hinweise für die Arbeit in der Cloud-Umgebung:**

- `pgrep -f "<muster>"` findet auch die eigene Warteschleife, deren Befehlszeile das Muster
  enthält. Deshalb Muster wie `"configs/[x]yz"` verwenden oder auf eine Datei warten.
- Ausgaben von `kinneret-recon` in Dateien sind gepuffert und erscheinen erst am Ende.
- Lange Läufe im Hintergrund starten; 4 Ketten × 2 Mio. dauern ~1,5 min Sampling plus ~1,5 min
  Prior-Referenzlauf plus ~0,5 min Auswertung.
- Ein Push von Tags ist über den Git-Proxy nicht möglich, Branches schon.
