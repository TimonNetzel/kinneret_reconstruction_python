# Claude: nächster Schritt nach der synthetischen Modulverbindung

Stand 02.11.2026. Aktueller Auftrag für die Fortsetzung auf
`dissertation-reconstruction` in [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).
Vorgängerstand `674bf132691aadf51359ee83052987a97e7c8587`; dieses Paket enthält die
gemeinsame synthetische Prüfung und den neuen Startauftrag. Aktuellen Git-/PR-
Head zuerst prüfen. Den Paketcommit zeigt `git log -1 -- tests/test_historical_phase1_chain.py`.
Deutsch, kurze Fortschrittsmeldungen, keine Subagenten.

## Einstieg und gesicherter Stand

1. Rootdateien `CLAUDE.md`, `AGENTS.md`, neuesten
   [Projektstatus](PROJECT_STATUS.md) und [Arbeitsplan](LOCAL_DISS_RECONSTRUCTION_PLAN.md) lesen.
2. [Ergebnis der gemeinsamen Prüfung](literal_phase1_chain_2026_11_02/README.md)
   mit `source_binding.json`, `synthetic_fixture.json`, `rational_expectations.json`,
   `synthetic_result.json` und `verification.json` lesen.
3. [Gezielten Originalauszug für diesen Auftrag](literal_phase1_chain_2026_11_02/next_step_source_excerpt.json)
   lesen. Er enthält die unten genannten Zeilen aus `src/mcmc_prepare.R`, SHA-256
   `b897e84f8f2d72399de729592066341585974d521bf8164b185f20377ed5cc7e`,
   bereits früher gebunden und im aktuellen Paket erneut vor/nach Lesen bestätigt.

Der [vorherige Startauftrag](CLAUDE_NEXT_STEP_2026_11_02.md) ist abgeschlossen
und als früherer Nachweis eingefroren. Diese Datei ist der aktuelle nächste
Auftrag. Den alten Startauftrag nicht ändern: Sein Hash ist Teil eines früheren
Prüfpakets. Neue Ergebnisse erhalten ein eigenes Paket und aktuelle Einstiegslinks.

Die vier vorhandenen Module bleiben unverändert. Die neue Test-/Belegverbindung
prüft eine 4×5-Chronologiematrix über drei Probentiefen, zwei Taxa und vier Zellen
bis zu Roh-CDF-Scores und normierter synthetischer PDF. Nur Original-Ziehungs-IDs
1 und 3 sind aktiv; Energiegewichte werden nicht eingeführt. Exakte Bruchrechnung
ohne NumPy oder Modulimporte bestätigt alle numerischen Zwischenstufen bis auf
höchstens 2.23e-16. **33 neue Tests, mit Vorpaketen 200 grün.**

Die bewusst ungleichen TF-Massen 2 und 3 im gemeinsamen Beispiel sind ein
Schnittstellentest. Sie sind kein Nachweis einer historischen TF-Vorbereitung.
Genau die fehlende deterministische Vorstufe ist der nächste technische Auftrag.

## Konkreter Auftrag: TF-Zeilennormierung und spätere Zellmaskierung

Erstelle als eigenständigen begrenzten Ausschnitt:

- `tools/historical_tf_preparation.py` mit reinen Funktionen für die belegte
  TF-Zeilennormierung und das anschließende Nullsetzen explizit ausgewählter
  Klimazellen. Keine Historien-/Standortwahl oder produktive App-/CLI-Schnittstelle.
- `tests/test_historical_tf_preparation.py` mit unabhängigen rationalen Sollwerten
  und gezielten Grenzen der beiden Schritte.
- Ein eigenes Belegpaket `docs/thesis_research/literal_tf_preparation_2026_11_02/`
  mit Quellenbindung, Formeln, Achsen/Einheiten, synthetischer Fixture,
  Ergebnissen, Grenzen und Prüfnachweis.

Nutze den mitgelieferten Quellauszug. Eine neue pauschale Inventur oder ein
zusätzlicher Speicherort ist nicht nötig. Bei lokalem Zugriff Originale nur
lesen und nur über `.local/my_reconst_paths.json` auflösen. Ohne diesen Zugriff
mit den portablen gebundenen Auszügen beginnen und keine lokale Rohquellen-
prüfung behaupten. Keine Originalarrays, Ersatzdaten oder neuen TFs beschaffen.

Die nachzubildenden Zeilen sind:

| Zeilen in `mcmc_prepare.R` | Verhalten | Umfang des nächsten Ausschnitts |
| --- | --- | --- |
| 1581–1582 | Jede TF-Zeile durch ihre eigene Summe teilen | Endliche nichtnegative Matrix mit positiver Zeilenmasse |
| 1583–1586 | Dasselbe für Change-TFs, nur falls `taxa_change` | Als bedingte Quellvariante dokumentieren; deaktivierten Zweig nicht einschalten |
| 1613–1614 | Zellen außerhalb der jeweiligen `inside.range`-Maske auf 0 setzen | Im Nachbau ausdrücklich übergebene synthetische Boolmaske; keine unbekannte Randkonvention auswählen |
| 1615–1618 | Change-TFs ebenso maskieren | Keine automatische Aktivierung oder finale Quellwahl |

Für eine Matrix `F` der Dimension **Taxa T × Klimazellen G** gilt zuerst

`N[t,k] = F[t,k] / sum_l F[t,l]`.

Mit einer ausdrücklich übergebenen G-elementigen Behaltemaske `B` anschließend

`M[t,k] = N[t,k] falls B[k], sonst 0`.

Es ist an diesen Zeilen keine zweite Normierung nach dem Nullsetzen belegt.
Normierung und Maske müssen getrennte Schritte bleiben. Die boolesche Maske
wird nicht aus eigener Annahme über Ein-/Ausschluss der Randwerte erzeugt.
Die genaue historische `inside.range`-/Paketbindung bleibt offen; wörtlichen
Aufruf und diese Grenze dokumentieren.

## Kleine synthetische Kontrolle und Abnahmekriterien

Geeignetes Beispiel mit zwei Taxa und vier Zellen:

`F = [[2,0,0,0],[0,0,1,2]]`.

Vor Maskierung ergeben sich `[[1,0,0,0],[0,0,1/3,2/3]]`.
Mit ausdrücklich künstlicher Behaltemaske `[T,T,F,T]` bleibt
`[[1,0,0,0],[0,0,0,2/3]]`, Zeilensummen `1` und `2/3`.
Eine anschließende Renormierung würde die zweite Zeile verändern und darf nicht
still ergänzt werden. Dieses Beispiel schätzt keine historischen TFs.

Erforderlich sind:

- Rationale Sollwerte für Normierung und anschließendes Maskieren, unterschiedliche
  Zeilenmassen und mindestens eine rechteckige Matrix zur Achsenprüfung.
- All-TRUE-Maske erhält die normierten TFs; teilweise oder vollständig verworfene
  Zellen dürfen nicht durch Ersatzmasse oder erneute Normierung gefüllt werden.
  Das besagt nicht, dass nachfolgende vollständige PDF-Pfade mit Nullmassen laufen.
- Eingaben bleiben unverändert. Maske muss exakt zur Klimazellachse passen;
  Taxonzeilen nicht mit Klimazellen verwechseln. Keine Index-/Längenreparatur.
- Endlichen Eingabebereich ausdrücklich festlegen; insbesondere ursprüngliche
  Nullsummen, NA/Inf und R-`na.rm`-/Recyclingpfade nicht still emulieren oder
  reparieren. Grenzen als solche benennen, keine methodische Korrektur behaupten.
- Falls eine zusätzliche Verbindungsprobe sinnvoll ist, eine eigene Kunstfixture
  mit explizit vorbereiteten TFs verwenden. Die bisherige ungleiche-Massen-
  Fixture und ihre eingefrorenen Sollwerte bleiben erhalten.
- Alle bisherigen 200 Tests und die neuen gezielten Fälle prüfen. Belegvertrag,
  bisherige Pakete, Module und Originale erhalten; Prüfung und Grenzen berichten.
  Aktuelle Einstiegslinks/Status und einen nächsten konkreten Auftrag aktualisieren.
  Geprüftes Paket committen und auf dem bestehenden Branch pushen; kein Merge.

## Bewusst ausgeschlossene umgebende Rechenschritte

Der Auszug enthält zur Einordnung auch stochastische Ziehungen und Quantile
(1591–1602). Sie werden nicht ausgeführt. Die Quelle überschreibt die daraus
abgeleiteten Bereiche in 1607–1608 mit `range(x1range.bck)` bzw.
`range(x2range.bck)`. Dies beweist keinen historischen Lauf oder konkrete
Randmaske; die `inside.range`-Grenze bleibt bestehen.

Die Produkte 1623–1624 werden später durch `weightedMedian`-Zuweisungen
(1631/1651) überschrieben. Keine ungebundene Paket-/Tie-Konvention nachbauen;
diesen Unterschied lediglich dokumentieren. Keine Splines, Medianberechnung,
fehlenden TFs-/Taxonpruning-Pfade (1663–1699), RNG-Schritte oder neue
Sampler-/Kalibrierungsarbeit in diesen Ausschnitt hineinziehen.

## Arbeitsumgebung, Prüfkommandos und Gates

Nur den vorhandenen, vom Nutzer benannten Checkout verwenden; keine Klone oder
Worktrees. Vor Beginn `git status --short --branch`, `git rev-parse HEAD` und
PR #77 prüfen; lokale Änderungen erhalten. Keine Resets, keine automatischen
Merges und keine Annahme, dass `main` den offenen PR bereits enthält.

```powershell
.venv/Scripts/python.exe -m pytest tests/test_historical_phase1_chain.py tests/test_historical_cdf_discrepancy.py tests/test_historical_age_selection.py tests/test_historical_age_depth.py tests/test_historical_final_pdf.py tests/test_kinneret_phase1_spec.py -q -o cache_dir=.local/phase1_spec_v1/pytest_cache
.venv/Scripts/python.exe tools/check_kinneret_phase1_spec.py
.venv/Scripts/python.exe tools/check_kinneret_phase1_spec.py --with-local
```

Vor neuen Änderungen: **200 bestanden**. Die Belegprüfer erwarten **Exitcode 2**
bei intakten Belegen und offenen Voraussetzungen: 26 Belegdateien, 19 Befunde,
50 Snapshotdateien, elf Nachweisbündel und `selected_variant=null`. Lokal bisher
28 unveränderte Dateien und zwölf fehlende Pfade; ohne Originalzugriff die lokale
Prüfung als nicht ausgeführt kennzeichnen. Neuen Testumfang tatsächlich zählen.

Finale Basis `core_data_additional_sites_v1.2.rdata`, finale Listenchronologie,
numerische historische TFs und Laufzuordnung fehlen unverändert. Der Nutzer
kennt keinen weiteren Speicherort/Erzeuger; nicht wieder fragen oder pauschal
inventarisieren. Ältere Biomcontainer oder sieben moderne Artenmodelle sind
keine Ersatzbelege. Kein vollständiger taxonbasierter Fall ist reproduziert.
Keine finale Kinneret-Variante wählen, keine neuen Altersmodelle, Trainings,
Sampler oder Fossilrekonstruktionen starten. App/CLI, Baseline, Referenzen,
Originale und Monticchio-/Chronologie-Gates erhalten. Phase 1 vor Phase 2,
Plots jung links, alt rechts.
