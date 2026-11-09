# `reconstruction_climate` (`mcmc_functions_v1.6.R:1188–1199`)

Stand 08.11.2026, Ausgangscommit `4373353` (PR-Kopf), bestehender Branch `dissertation-reconstruction`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).
Auftrag: Nutzerentscheidung „Ja, nachbauen“ nach der
[Übergabe nach der Gegenprobe](../CLAUDE_NEXT_STEP_AFTER_SIX_MODULE_CROSSCHECK_2026_11_08.md).
Dreizehnter begrenzter Rechenschritt, aber **ohne Wirkung auf den gebundenen Lauf**. Noch kein Originalfall.

## Wichtigster Befund: Der gebundene Lauf ruft diese Funktion nicht auf

[bind_source.py](bind_source.py) → [Bindung](source_binding.json), [Auszug](source_excerpt.json),
[Determinismusprüfung](determinism_audit.json). Alle 540 `.R`-Dateien in `diss` wurden durchsucht.

- Einziger Aufruf: `mcmc_sampling_v1.4.R:79`. Diese Datei wird nirgends gesourct und auch sonst nicht erwähnt.
- `main_v1.2.R:675–677` lädt `mcmc_prepare.R`, `mcmc_conduction.R` und `mcmc_trimming_posterior_trafo.R`.
- Der gebundene Lauf nimmt in `mcmc_functions_v1.6.R:1850–1851` `predictant_tann[1]`/`predictant_pann[1]`. Diese Werte entstehen
  aus den Ziehungen `rtruncnorm_cpp` (1812–1820) und bleiben Zufall.
- `mcmc_prepare.R:1408–1413` rechnet ähnlich, steht aber hinter `if(recent & !add.sites)` (1380). Mit `add.sites <- T`
  (`main_v1.2.R:73`) wird auch das nicht erreicht; die Schleife zieht zudem mit `sample()`.

Das Modul dokumentiert also einen ruhenden Quellweg. Es verändert keinen Zwischenstand des Nachbaus.

## Implementierter Umfang

[historical_reconstruction_climate.py](../../../tools/historical_reconstruction_climate.py),
[Tests](../../../tests/test_historical_reconstruction_climate.py).

- `recent = FALSE`: Es wird nur `prior.recent$mean.tmp/mean.pann` zurückgegeben. R wertet die übrigen Argumente wegen Lazy
  Evaluation nicht aus; auch ungültige Werte schaden nicht. In der R-Gegenprobe sind sie `stop()`-Ausdrücke.
- `recent = TRUE`: nur die **erste** Zeile von `taxa.appear.core`, dann `Zeile %*% TF` und `weightedMedian` (gebunden im Fenster).
- `which.taxa` wird nie gelesen. `x1range.bck`, `x2range.bck` und `prior.recent` sind in der Quelle global, hier explizit.
- `Zeile %*% TF` ist bei endlichen Zahlen `dgemv("T")`: je Spalte eine einfache Double-Summe in Taxonreihenfolge. Damit ist
  dieser Teil, anders als die älteren NumPy-Module, gegen R bitgleich.
- `0 · NA` ist NA: Ein Taxon mit Anteil 0 und NA-TF macht den Median NA.

Abgelehnt statt emuliert: unendliche Werte, nicht konforme Dimensionen, `recent` ohne Boolwert.

## Unabhängige Referenz und Gegenbeispiele

[Fixture](synthetic_fixture.json) mit 8 Kunstfällen und 200 deterministischen Stressfällen. [Referenz](precise_reference.py): exakte
Brüche mit einmaliger Rundung je Operation, Median aus der Referenz des [Medianpakets](../literal_weighted_median_2026_11_08/README.md),
ohne Modulimport ([Sollwerte](reference_expectations.json)). [Ergebnisse](synthetic_result.json), [Gegenbelege](arithmetic_check.json).

- Alle Fälle bitgleich mit der Referenz. Handwerte: `recent_basic` ergibt 25/3 und 220.
- Gegenbeispiele (keine Kandidatenvarianten): Zeilenmittel statt erster Zeile, Erwartungswert statt Median, NA bei Anteil 0 überspringen.

## R 3.5.0 mit matrixStats 0.56.0

[run_r_crosscheck.py](run_r_crosscheck.py), [r_crosscheck.R](r_crosscheck.R) lesen **1188–1199 wörtlich aus der Originaldatei** und rufen die
Funktion auf 8 + 200 Fällen auf, die globalen Namen liegen in der Umgebung der Funktion. Ergebnis: [r_crosscheck_result.json](r_crosscheck_result.json).
**5.510 von 5.510 Werten sind bitgleich** (0 ulp), NA und NaN unterschieden. Weitere Zahlen im [Prüfnachweis](verification.json).

## Prüfen

```powershell
.venv/Scripts/python.exe docs/thesis_research/literal_reconstruction_climate_2026_11_08/bind_source.py
.venv/Scripts/python.exe docs/thesis_research/literal_reconstruction_climate_2026_11_08/make_fixture.py
.venv/Scripts/python.exe docs/thesis_research/literal_reconstruction_climate_2026_11_08/precise_reference.py
.venv/Scripts/python.exe docs/thesis_research/literal_reconstruction_climate_2026_11_08/check_reconstruction_climate.py
.venv/Scripts/python.exe docs/thesis_research/literal_reconstruction_climate_2026_11_08/run_r_crosscheck.py .local/R/R-3.5.0/bin/x64/Rscript.exe .local/R/library_ms
```

Nicht ausgeführt: die Gesamtsuite, `verify_packet.py`, `tools/verify_*`; Sampler, Ziehungen, Trainings, Altersmodelle, Fossilrekonstruktionen.

## Grenzen

- Kein Originalfall, keine echten Transferfunktionen.
- Ein nachgebauter, aber im gebundenen Lauf nicht aufgerufener Weg zeigt nichts über die Kinneret-Rekonstruktion.
- Alle elf Nachweisbündel bleiben offen; keine Variante gewählt. Frühere Pakete und Module sind bytegleich.
