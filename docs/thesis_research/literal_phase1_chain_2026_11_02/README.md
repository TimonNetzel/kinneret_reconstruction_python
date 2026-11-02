# Gemeinsame synthetische Prüfung der Phase-1-Teilmodule

Stand 02.11.2026, Basis `674bf132691aadf51359ee83052987a97e7c8587`,
Branch `dissertation-reconstruction`, [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).

**Abgeschlossen:** Die vier getrennten Teilmodule sind über ein gemeinsames
künstliches Beispiel auf Achsen, Rechenreihenfolge, Original-Ziehungs-IDs und
Normierungszeitpunkte geprüft. **33 neue Tests, zusammen 200 grün.** Alle
numerischen Zwischenstufen liegen höchstens `2.23e-16` von einer unabhängigen
exakten Bruchrechnung entfernt; Toleranz `2e-14`.

Das Paket enthält ausschließlich Test-/Nachweiscode. Kein historischer Runner,
keine neue App-/CLI-Funktion und keine originale Fallrekonstruktion. Die vier
Module bleiben unverändert. Der
[neue Claude-Auftrag](../CLAUDE_NEXT_STEP_TF_PREPARATION_2026_11_02.md)
beschreibt den nächsten abgegrenzten Schritt samt portablen Quellauszügen.
Der [vorherige Auftrag](../CLAUDE_NEXT_STEP_2026_11_02.md) ist hiermit bearbeitet
und bleibt als eingefrorener Nachweis unverändert.

## Quelle und genaue Testgrenze

[source_binding.json](source_binding.json) bindet die vier vorhandenen Module
und ihre Quellberichte an SHA-256. Drei gezielt gelesene Originalskripte
(`src/check_ages.R`, `src/mcmc_prepare.R`, `src/mcmc_functions_v1.6.R`) stimmen
vor/nach Lesen mit den bisherigen Hashbindungen überein. Originalpfade werden
nur über `.local/my_reconst_paths.json` aufgelöst. Es wurden ausschließlich
Skripte für Hash-/Auszugsbindung gelesen, keine numerischen Originalcontainer.

Die Quellstellen und Grenzen bleiben die der vorhandenen Teilpakete:

| Schritt | Quellstellen | Eingabe → Ausgabe im gemeinsamen Beispiel |
| --- | --- | --- |
| Listenwahl/Alterslabels | `check_ages.R:177,179,230–232` | Explizite R-Listenposition 2; vier Alter, einmalige Duplikatverschiebung |
| Probentiefenkern | `mcmc_prepare.R:11–20` | rho(D|A) **4 × 5** → Gewichtsmatrix **4 × 3** |
| Pollenübertragung | `mcmc_prepare.R:83,91–94` | Gewichtsmatrix × Pollen **3 × 2** → **4 × 2** |
| Perioden-/Auswahlzustand | `mcmc_prepare.R:435–442,1462–1467,1500–1522` | Masken aus `age`, wirksamer All-TRUE-Zustand nur bei Erreichen der betreffenden Zeilen |
| RW-Gewichtung | `mcmc_functions_v1.6.R:411–425,450–455` | Maximumskalierung und zwei vorgegebene Gewichtungen → Koeffizienten **4 × 2** |
| Zeilenauswahl/TF-Produkt | `mcmc_functions_v1.6.R:461–462` | **4 × 2** × synthetische TFs **2 × 4** → Rohdichten **4 × 4** |
| Kumulative Abweichung | `mcmc_functions_v1.6.R:369–370,485–489` | Nullreferenz mit expliziten Breiten; vier Rohscores je verbliebener Ziehung |
| Ziehungs-IDs/Summe/PDF | `mcmc_functions_v1.6.R:375,428,543–547,711–715,755–760` | Ursprüngliche IDs `[1,3]`; Rohdichten summieren, anschließend zeilenweise normieren |

**Keine vollständige Quellenkette:** Die synthetischen TF-Zeilen haben bewusst
Massen 2 und 3. Das prüft die allgemeine endliche Schnittstelle der vorhandenen
Module und macht vorgezogene Normierungen sichtbar. Es behauptet nicht, dass
solche Werte die historische TF-Vorbereitung durchlaufen hätten. Deren
Zeilennormierung (1581–1586), anschließende Maskierung und weitere Verarbeitung
sind in diesem Paket nicht nachgebaut. Genau diese Grenze ist der nächste
Claude-Auftrag. Keine Original-TFs werden durch die Kunstwerte ersetzt.

## Gemeinsame Eingaben und Zustände

[synthetic_fixture.json](synthetic_fixture.json) benennt vier Alterszeilen,
drei Probentiefen, zwei Taxa und vier Klimazellen ausdrücklich als künstlich.
Die Tiefen `[10,12,14]` wählen die R-Spalten `[1,3,5]`. Nicht gewählte Spalten
besitzen große Werte, sodass eine Normierung über das vollständige Gitter
andere Resultate ergäbe. Die Einheiten sind abstrakt; keine echten cm-,
Temperatur- oder PANN-Reihen werden ausgegeben.

`site.age=[-50,0,0,11500]` bleibt erhalten. Die einmalige Verschiebung ergibt
`age=[-50,0,0.5,11500]`. Die exakte Holozänmitgliedschaft lautet `[T,T,F,F]`,
die Glazialmitgliedschaft `[F,F,F,T]`. Bei explizit künstlichem
`taxa_merge_control=TRUE` wird die vorläufige Holozänmaske im erhaltenen
Kontrollfluss auf alle Zeilen zurückgesetzt, sofern die Quellzeilen erreicht
werden. Dies belegt weder reales Taxonmatching noch historischen Ausführungserfolg.

Die Tiefengewichte werden nach Normierung exakt an Nullpositionen auf 1e-9
gesetzt; ihre Zeilenmassen sind `[1,1+1e-9,1+2e-9,1]`. Eine zweite Normierung
wird nicht eingefügt. Pollennullen werden getrennt vor dem Produkt zu 1e-6.
Die unterschiedliche Zahl ersetzter Tiefennullen hält auch den Einfluss einer
zusätzlichen Kernnormierung in der vollständigen Kunstrechnung sichtbar.

Die Energien `[4,4,9]` bestimmen **vor aktiven Dichte-/CDF-Aufrufen** die
Originalpositionen `[1,3]`. Nur für diese IDs existieren Gewichtungen und
Scores. Ein endlicher, beliebig veränderbarer Sentinel belegt im Test die
unbenutzte zweite Position des an `sum_pdf_branch` übergebenen Hilfstensors.
Diese Testeingabe ist keine Berechnung der historisch verworfenen Ziehung.
Keine Neunummerierung, Energiegewichtung oder neue Ziehungsauswahl erfolgt.

## Unabhängige Bruchrechnung und kleine Kontrollidentitäten

[rational_reference.py](rational_reference.py) verwendet ausschließlich
`fractions.Fraction` und explizite skalare Summen, ohne NumPy oder Importe der
zu prüfenden Module. Es berechnet die festgelegte Kunstrechnung exakt.
[rational_expectations.json](rational_expectations.json) friert diese Brüche
für Alter, Tiefengewichte, Pollen, Maximumskalierung, beide aktiven Gewichtungen,
Rohdichten, kumulative Werte, Scores, Rohsumme und normierte PDF ein.

Handkontrolle: Die erste Gewichtszeile `[1/4,1/2,1/4]` mit Pollenzeilen
`[[2,0],[0,4],[6,2]]` ergibt nach Nullersetzung die Alters-Pollenzeile
`[4000001/2000000,10000001/4000000]`.
Die dritte Tiefengewichtszeile lautet `[1e-9,1e-9,1]` und besitzt Masse `1+2e-9`.

Für TF-Zeilen `[2,0,0,0]` und `[0,0,1,2]` und einen RW-Koeffizientenvektor
`[c,1-c]` gelten zusätzlich die direkt prüfbaren Identitäten:

`D = [2c,0,1−c,2−2c]`, `C = [2c,2c,1+c,3−c]`,

`Rohmasse = 3−c`, `Nullreferenzscore = 1+c`.

Mit erstem Taxonkoeffizienten `t` der dritten Originalziehung und `q=c+t` ist

`Rohsumme = [2q,0,2−q,4−2q]`, `Summenmasse = 6−q`.

Die endgültige synthetische PDF ist diese Rohsumme geteilt durch `6−q`.
Der Nullreferenzscore je aktiver Ziehung wird vorher berechnet; er ist kein
Mischungsgewicht und kein physikalischer Abstand zu einer echten Referenz-CDF.

## Was die gemeinsame Prüfung erkennt und was sie nicht erkennt

[Tests](../../../tests/test_historical_phase1_chain.py) vergleichen alle
Zwischenstufen und beide aktiven Ziehungen mit den exakten Sollwerten.
Gezielte Gegenbeispiele zeigen:

- Jede Ziehung vor der Summe zu normieren verändert die endgültige PDF maximal
  um rund `0.00282`.
- Die gelieferten synthetischen TF-Zeilen still zu normieren verändert sie
  maximal um rund `0.09716`.
- Eine zusätzliche Tiefenkernnormierung oder ausgelassene Pollen-Nullersetzung
  verändert die Kunstrechnung ebenfalls; die Tests prüfen diese Einflüsse.
- Matrix statt Listenchronologie, transponierte Kern-/Pollen-/TF-Matrizen,
  verkürzte Altersachse, fremde Maskenlänge und widersprüchliche Platzhalterbreite
  werden an den tatsächlichen Modulgrenzen abgewiesen.

Gleiche Form allein erkennt keine semantische Vertauschung zweier Taxonzeilen.
Die explizite Achsenzuordnung und der Vergleich mit den festen Sollwerten
zeigen einen solchen Fehler in diesem Beispiel; es wurde keine allgemeine
Taxon-/Provenienzprüfung als Produktfunktion eingeführt.

Eingaben bleiben unverändert. Energiegrößen mit unverändertem Duplikatmuster
und der unbenutzte Sentinel ändern weder Scores noch Ausgabe. Der Gibbs-Pfad,
Taxonwechsel, tatsächliches TF-Matching, Prior-/IAT-/Pareto-Selektion,
stochastische Schritte und R-Sondersemantik außerhalb der bereits dokumentierten
endlichen Domänen bleiben ausgeschlossen.

## Wiederholung und Nachweise

[check_chain.py](check_chain.py) ist ausschließlich Test-/Belegcode für die
explizite Kunstfixture. Es verändert keines der vier Module und liest keine
Originale. [collect_evidence.py](collect_evidence.py) prüft drei bekannte
Skripthashes, erzeugt den Bruchnachweis und die numerischen Ergebnisse sowie den
gezielten Quellauszug für den nächsten Schritt. Keine Inventur oder R-Ausführung.
[synthetic_result.json](synthetic_result.json) enthält Zwischenstufen,
Abweichungen und Gegenbeispiele. [verification.json](verification.json)
hält Tests, Belegvertrag und Erhaltung fest.

```powershell
.venv/Scripts/python.exe docs/thesis_research/literal_phase1_chain_2026_11_02/collect_evidence.py
.venv/Scripts/python.exe -m pytest tests/test_historical_phase1_chain.py tests/test_historical_cdf_discrepancy.py tests/test_historical_age_selection.py tests/test_historical_age_depth.py tests/test_historical_final_pdf.py tests/test_kinneret_phase1_spec.py -q -o cache_dir=.local/phase1_spec_v1/pytest_cache
```

Die Tests benötigen keine privaten Originalpfade. Nur die erneute Skripthash-/
Auszugsbindung im Sammler benötigt die private Konfiguration.
Kein historischer R-/Paketlaufzeitvergleich; `float64` kann anders runden.
Die Oracle-Gleichheit prüft die festgelegten Formeln, nicht einen Original-R-Lauf.

## Ergebnis und nächster Einstieg für Claude

[Neuer konkreter Startauftrag](../CLAUDE_NEXT_STEP_TF_PREPARATION_2026_11_02.md):
TF-Zeilennormierung und anschließende Zellmaskierung als nächsten begrenzten
Ausschnitt isolieren. Der bereits hashgebundene
[next_step_source_excerpt.json](next_step_source_excerpt.json) enthält die
relevanten Originalzeilen einschließlich der ausgeschlossenen stochastischen
und späteren Median-/Pruningpfade. Der Auftrag kann mit diesen portablen
Quellnachweisen begonnen werden; kein weiterer Speicherort ist nötig.

Der Phase-1-Belegvertrag bleibt unverändert: elf offene Nachweisbündel.
Finale Originalbasis/Listenchronologie, historische numerische TFs und
Laufzuordnung fehlen. Kein finaler Kinneret-Pfad gewählt und kein Fall
vollständig reproduziert. Originale, App/CLI, Baseline, Referenzen und
fallbezogene Monticchio-/Chronologie-Gates bleiben erhalten. Keine Trainings,
Sampler, Altersmodelle, Fossilrekonstruktionen oder Phase-2-Verbesserungen.
Plots jung links, alt rechts. Bestehender Branch/PR #77, kein Merge.
