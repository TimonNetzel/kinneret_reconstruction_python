# Historische TF-Vorbereitung: Normierung und Zellmaske

Stand 02.11.2026, Ausgangscommit `956951b`, bestehender Branch
`dissertation-reconstruction`, [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).
Ziel bleibt der historische Dissertationsnachbau mit anschließender Umsetzung
in der App. Dieses Paket ergänzt einen fehlenden Rechenschritt und prüft dessen
Anschluss an die vier vorhandenen Module. Noch kein vollständiger Originalfall.

## Implementierter Quellenumfang

[Originalbindung](source_binding.json) und [portabler Auszug](source_excerpt.json)
binden `original:src/mcmc_prepare.R`, 82.630 Bytes, SHA-256
`b897e84f8f2d72399de729592066341585974d521bf8164b185f20377ed5cc7e`.
Hash vor/nach Lesen identisch zum früheren Beleg; alle vier übernommenen
Quellspannen stimmen zeilenweise mit der lokalen Originalquelle überein.
Die Originalquelle wurde nur gelesen, keine R-Ausführung.

| R-Zeilen | Python-Funktion | Verhalten |
| --- | --- | --- |
| 1581–1586 | `normalize_tf_rows` | Jede Taxonzeile durch ihre eigene Summe teilen; TANN/PANN separat. |
| 1613–1618 | `mask_tf_cells` | Ausgeschlossene Klimazellspalten auf null setzen; explizite Boolmaske. |

Implementierung: [historical_tf_preparation.py](../../../tools/historical_tf_preparation.py).
Tests: [test_historical_tf_preparation.py](../../../tests/test_historical_tf_preparation.py).

Für **Taxa × Klimazellen** gilt `N[t,k] = F[t,k] / Σ_l F[t,l]`, danach
`M[t,k] = N[t,k]` bei `B[k]`, sonst `0`. Eine zweite Normierung wird nicht
eingeführt. Die Funktionen verändern ihre Eingaben nicht und erhalten die
übergebene Reihenfolge. TANN und PANN können verschiedene Klimazellbreiten und
Masken haben. Es wird kein gemeinsames Klimaraster erzwungen.

Domäne: nichtleere reelle numerische Matrizen, in float64 überführt, endliche
nichtnegative Werte und endliche positive Zeilensummen vor der Normierung.
Die Maske enthält genau einen Boolwert je Klimazelle. Nullzeilen nach der Maske
bleiben null; dies erteilt keine Lauffreigabe für nachfolgende PDF-Normierung.
NA/Inf, ursprüngliche Nullsummen, Summenüberlauf, komplexe Werte, Strings,
implizite Boolumwandlung und R-Recycling-/Dimensionsverlustpfade werden nicht
still emuliert oder repariert. Das ist eine begrenzte Domäne, keine Korrektur
historischer Methodik.

Es handelt sich um diskrete Summierung ohne Klimazellbreiten. Physikalische
Dichteintegration, Einheitenumrechnung und historische Achsenzuordnung werden
nicht daraus abgeleitet. Taxon-/Zellnamen und Einheiten sind im vollständigen
Eingabevertrag separat zu binden.

## Unabhängige Kunstrechnung und Anschluss

[Fixture](synthetic_fixture.json), [skalare Bruchreferenz](rational_reference.py),
[exakte Sollwerte](rational_expectations.json) und [Ergebnisse](synthetic_result.json).
Die unabhängige Referenz verwendet `Fraction` und explizite Summen, ohne NumPy
oder das zu prüfende Modul zu importieren.

Das Quellenbeispiel `[[2,0,0,0],[0,0,1,2]]` wird zu
`[[1,0,0,0],[0,0,1/3,2/3]]`. Mit der ausdrücklich künstlichen Behaltemaske
`[True,True,False,True]` verbleibt `[[1,0,0,0],[0,0,0,2/3]]`, Zeilenmasse
`1` beziehungsweise `2/3`. Ein rechteckiges 3×5-Beispiel und eine unabhängige
2×3-PANN-Testachse ergänzen die Kontrolle. Keine historischen TF-Werte.

Für eine **neue synthetische Verbindungsprobe** werden diese vorbereiteten TFs
an die unveränderte frühere Kunstchronologie angeschlossen: Alters-/Tiefenkern
→ Pollen → RW-Koeffizienten → vorbereitete TFs → Rohdichten/Roh-CDF-Scores →
PDF-Summe. Die alte Fixture, ihre ungleichen TF-Massen und ihre Ergebnisse
bleiben bytegleich erhalten. Die unabhängige alte Bruchreferenz erhält für diese
zusätzliche Probe exakte rationale vorbereitete TF-Werte statt Python-Modulergebnisse.
Nur Original-Ziehungs-IDs **1 und 3** werden aktiv; Energiegrößen bleiben ohne
Mischungsgewicht. Kein Runner für einen historischen Standort oder neue Ziehungen.

**Prüfergebnis:** 51 neue Tests und alle bisherigen 200 Teil-/Belegvertragstests,
insgesamt **251 bestanden**. Maximale absolute Abweichung aller numerisch
kontrollierten Zwischenstufen von der Bruchrechnung **2,22 × 10⁻¹⁶** bei
Toleranz `2e-14`. Eine zusätzliche Normierung nach Maskierung verändert eine
TF-Zelle maximal um `1/3` und die End-PDF der neuen Kunstkette maximal um
`0,10001949796977816`. [Arithmetik-/Gegenbeleg](arithmetic_check.json).
Diese Differenzen sind synthetische Sensitivitätskontrollen, keine historischen
Fehlerabschätzungen oder fossilen Unsicherheiten.

## Grenzen des historischen Nachbaus

Die in der Quelle bedingten Change-TF-Formeln werden dokumentiert, aber der
Taxonwechselzweig wird nicht aktiviert. Es wird keine `inside.range`-Randkonvention,
Paket-/Laufumgebung oder finale Maske ausgewählt. Die umgebenden `sample`-/Quantil-
Schritte, Splines, `weightedMedian` und Taxon-Pruning bleiben außerhalb dieses
Pakets. Die früher nachfolgenden Zuweisungen/Überschreibungen bleiben als
Quellkonflikte sichtbar; kein vollständiger historischer Vorbereitungsablauf.

Finale Basis/Listenchronologie, numerische TFs, Referenz-/Ankerdaten und Laufbindung
bleiben ungebunden. Alle elf Nachweisbündel bleiben offen. Die neue Klimabestands-
prüfung ist in [climate_modelling](../climate_modelling_audit_2026_11_02/README.md)
gesichert; keine erneute Inventur und keine bereits beantwortete Speicherortfrage.
Neue Trainings, Sampler, Altersmodelle oder Fossilläufe wurden nicht gestartet.
Baseline, App/CLI und fallbezogene Monticchio-/Chronologie-Gates bleiben erhalten.
Plots jung links, alt rechts; Phase 1 vor Phase 2.

## Prüfen und nächster App-Schritt

```powershell
.venv/Scripts/python.exe docs/thesis_research/literal_tf_preparation_2026_11_02/check_tf_preparation.py
.venv/Scripts/python.exe -m pytest tests/test_historical_tf_preparation.py tests/test_historical_phase1_chain.py tests/test_historical_cdf_discrepancy.py tests/test_historical_age_selection.py tests/test_historical_age_depth.py tests/test_historical_final_pdf.py tests/test_kinneret_phase1_spec.py -q -o cache_dir=.local/tf_preparation_2026_11_02/pytest_cache --junitxml=.local/tf_preparation_2026_11_02/pytest.xml
.venv/Scripts/python.exe tools/check_kinneret_phase1_spec.py
.venv/Scripts/python.exe tools/check_kinneret_phase1_spec.py --with-local
.venv/Scripts/python.exe docs/thesis_research/literal_tf_preparation_2026_11_02/verify_packet.py
```

164 frühere Dateien einschließlich fünf früherer Übergaben, vier Module und sechs
Testmodule sind bytegleich zum Ausgangscommit erhalten. 39 Dateihash-Einträge
der bisherigen Fragmentmanifeste und 110 relative Dokumentationslinks geprüft.
Beide Phase-1-Prüfer erwarten weiterhin Exitcode 2; ein intakter Belegvertrag ist
keine historische Lauffreigabe. [Gesamtprüfung und Manifest](verification.json).
Jedes abgeschlossene Paket erhält auf Nutzerwunsch eine eigene konkrete
Claude-Übergabe. Für dieses Paket:
[Claude-Auftrag: geprüfte Teile in der App zugänglich machen](../CLAUDE_NEXT_STEP_APP_REPRODUCTION_2026_11_02.md).
