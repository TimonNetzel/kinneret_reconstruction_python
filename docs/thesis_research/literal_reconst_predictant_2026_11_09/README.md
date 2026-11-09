# `reconst` und die Prädiktanten aus vorgegebenem `tf_sample` (`mcmc_functions_v1.6.R:1817–1820`)

Stand 09.11.2026, Ausgangscommit `38775e1` (PR-Kopf, `MERGEABLE`), bestehender Branch `dissertation-reconstruction`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).
Auftrag: Nutzerauftrag nach der [Übergabe nach dem lokalen cor](../CLAUDE_NEXT_STEP_AFTER_LOCAL_COR_2026_11_09.md).
Fünfzehnter begrenzter Rechenschritt. Explizite Matrizen, keine Ziehung, kein Sampler. `rtruncnorm_cpp` aus Zeile 1812–1813 bleibt draußen. Noch kein Originalfall.

## Abweichungen von der Übergabe, belegt

1. **Der 3-Schleifen-Pfad von `%*%` rechnet nicht in `long double`.** Der Auftrag nannte `simple_matprod` mit `long double`. In der CRAN-Quelle von R 3.5.0 hat
   `array.c:686 internal_matprod` einen `long double`-Akkumulator, `array.c:702 simple_matprod` aber `double`. Nur `simple_matprod` läuft bei `options(matprod = "default")` (Zeile 776–777);
   `options.Rd` sagt dasselbe. Messung in R 3.5.0: `c(1, 2^-53, 2^-53)` gibt neben einer NaN-Zeile `1`, nicht `1 + 2^-52` ([Fall](synthetic_fixture.json) `tie_with_nan_in_other_row`).
   Das frühere Modul `historical_reconstruction_climate.py` hatte dies richtig.
2. **`reconst_ages` ist in jedem Lauf „alle Zeilen“.** Die Holozänmaske aus `mcmc_prepare.R:1466` erreicht Zeile 1818 nie: Zeile 1500 endet in `& FALSE`, deshalb überschreibt Zeile 1522 die Maske wieder.
   Eine einzelne ausgewählte Zeile entsteht im lebenden Pfad also nur bei `length(age) == 1`. Die Maskenlogik ist trotzdem vollständig nachgebildet.
3. **Das Argument `tf_sample` wird nie gelesen.** `mcmc_conduction.R:146` übergibt dort `tf_x12_mean_current[...]`, der Rumpf liest aber nur `posterior_temp$tf_sample`; das Argument bleibt ein unausgewertetes Promise.

## Erreichbarkeit

[bind_source.py](bind_source.py) → [Bindung](source_binding.json), [Auszug](source_excerpt.json), [Determinismusprüfung](determinism_audit.json). Kommentare und Zeichenketten (auch eingebettetes C++) sind vor der Suche entfernt.

- **Aufruf:** nur `mcmc_conduction.R:146` in jeder Iteration, Definition 1781 (ersetzt 855), Rumpf bis 2046.
- **Eingabe:** `posterior_temp$taxa.appear.core.temp` aus Zeile 1277 (`parameter_stage_weights`, `gibbs` F), ohne `dimnames`. Sie kann NA, NaN oder Inf enthalten (Gewichtsstufe); ob das eintritt, hängt von der finalen Pollenmatrix ab.
- **Zeilenzahl:** `all.avail.age` hat `length(age)` Zeilen (Zeile 91), `check_ages.R` ist einziger Schreiber von `age` und wird in Zeile 3 vorher geladen. Die Maske passt also zu `nrow`.

| Zeile | liest | Bedingung | Kinneret (`add.sites <- T`, Slot 3) |
|---|---|---|---|
| 1819, 1820 | `reconst` | keine | **erreicht** |
| 1825 | `reconst[1,]` | keine | **erreicht**; bei null Zeilen stoppt R. Schreibt `posterior$youngest_values[i,]` (`mcmc_conduction.R:420`) |
| 1850, 1851 | `predictant_*` | `recent` | erreicht, wenn `recent` TRUE ist, also ein Alter ≤ 11000 existiert (`mcmc_prepare.R:728, 1184`): **ohne finale Alter offen** |
| 1831–1837 | `predictant_*` | `taxa_change` | nicht erreicht: bleibt FALSE, einziges TRUE in Zeile 1501 hinter `& FALSE` |
| 1867–1902 | `predictant_*` | `past_climate_module` | nicht erreicht: `main_v1.2.R:670` setzt F |
| 1923–1929 | `reconst_tann`, `reconst_pann` | `stad_interstad` | nicht erreicht: `mcmc_prepare.R:468`; die beiden Variablen sind nirgends zugewiesen |
| 1940 | `predictant_tann` | `tann` | nicht erreicht |
| **1942** | `predictant_pann` | `!tann` | **erreicht** (`mcmc_prepare.R:1249`) |
| 1964, 1966 | `predictant_current`, `predictant_tann` | `multiple` | erreicht bei `sources$core` 1 ([lokales cor](../literal_local_cor_2026_11_09/README.md)) |
| 1993 | `predictant_tann` | `!multiple` | nur bei `sources$core` 0 |
| 1982, 1999, 2009, 2031 | | | nicht erreicht |
| 2023 | `predictant_current` | `spectral` | hängt an `sources$speleothem`, braucht `wt_power` |

Der globale `reconst` und die globalen `predictant_*` aus `mcmc_conduction.R:109–110` werden überdeckt und nie gelesen. `final_reconstruction` (liest `reconst_ages` in 461–470) hat in den geladenen Dateien keinen Aufrufer.
**Kein Paket** ist beteiligt: nur `[`, `%*%` und `$`.

## Implementierter Umfang

[historical_reconst_predictant.py](../../../tools/historical_reconst_predictant.py), [Tests](../../../tests/test_historical_reconst_predictant.py).

- **Dimensionen:** Eine ausgewählte Zeile fällt zum Vektor; `%*%` nimmt ihn als Zeile, wenn die Länge `nrow(tf_sample)` ist, sonst als Spalte, wenn `tf_sample` eine Zeile hat. Null Zeilen geben eine 0×2-Matrix, leere Prädiktanten und einen Abbruch erst in Zeile 1825. Eine Maske, die länger als `nrow` ist, stoppt R in Zeile 1818; eine kürzere wird recycelt; NA wählt eine NA-Zeile.
- **Rechenweg:** Endliche Zahlen laufen über Referenz-BLAS (`dgemm`, bei einer Zeile `dgemv` mit `TRANS = "T"`), NaN/Inf über `simple_matprod`. **Beide akkumulieren einfach in Double in Taxonreihenfolge.** Eine NaN-Zeile ändert also die endlichen Zeilen nicht. Python-`sum()` ist ab 3.12 kompensiert und wird nicht benutzt.
- **NA und NaN:** Gemessen in R 3.5.0 auf 1.296 von 1.296 Kombinationen ([Messung](nan_rule_measurement.json), [Skript](measure_nan_rule.py); von vier geprüften Regeln passt nur diese): Das Produkt nimmt den ersten fehlenden Operanden von `x * y`, die Summe behält einen fehlenden Akkumulator. `0 * NA` bleibt NA, `0 * Inf` ist NaN. Diese Regel ist ein Messbefund, keine Herleitung.
- **Leser:** `youngest_values` (1825), `recent`-Zweige (1844–1857), `predictant_current` (1939–1943). `taxa_change` TRUE wird abgelehnt statt emuliert.

## Unabhängige Referenz und R-Gegenprobe

[Fixture](synthetic_fixture.json): 38 Kunstfälle, 6 Ketten, 400 Stressfälle. [Referenz](precise_reference.py): exakte Brüche mit einem Rundungsschritt je Operation, spaltenweise wie `dgemm` ([Sollwerte](reference_expectations.json)). [Ergebnisse](synthetic_result.json).

[run_r_crosscheck.py](run_r_crosscheck.py), [r_crosscheck.R](r_crosscheck.R) lesen `1817–1820`, `1825`, `1829–1857`, `1939–1943` und in den Ketten `1939–2018` mit `2355–2366` **wörtlich aus der Originaldatei** in R 3.5.0. Eingaben als C99-Hex.
Ergebnis: [r_crosscheck_result.json](r_crosscheck_result.json). **3.009 von 3.009 Werten sind bitgleich** (0 ulp), keine NA/NaN-Abweichung, keine abweichende Stoppstelle. Alle vier Pfade sind in den Stressfällen getroffen (`dgemm` 202, `dgemv_t` 42, `simple_matprod` 146, leer 10).

[Gegenbeispiele](arithmetic_check.json), keine Varianten: Ein `long double`-Akkumulator ändert 367 von 2.032 endlichen Stresswerten, Python-`sum()` und eine exakte Summe ebenfalls je 367; NumPy `@` ändert hier nichts.

Im [Prüfnachweis](verification.json): 229 neue, 1.172 frühere, zusammen 1.401 Tests grün.

## Prüfen

```powershell
.venv/Scripts/python.exe docs/thesis_research/literal_reconst_predictant_2026_11_09/bind_source.py .local/reconst_predictant_2026_11_09/r_source/R-3.5.0.tar.gz
.venv/Scripts/python.exe docs/thesis_research/literal_reconst_predictant_2026_11_09/make_fixture.py
.venv/Scripts/python.exe docs/thesis_research/literal_reconst_predictant_2026_11_09/precise_reference.py
.venv/Scripts/python.exe docs/thesis_research/literal_reconst_predictant_2026_11_09/check_reconst_predictant.py
.venv/Scripts/python.exe docs/thesis_research/literal_reconst_predictant_2026_11_09/run_r_crosscheck.py .local/R/R-3.5.0/bin/x64/Rscript.exe
.venv/Scripts/python.exe docs/thesis_research/literal_reconst_predictant_2026_11_09/measure_nan_rule.py .local/R/R-3.5.0/bin/x64/Rscript.exe
.venv/Scripts/python.exe docs/thesis_research/literal_reconst_predictant_2026_11_09/record_verification.py
```

Das R-Quellarchiv liegt unversioniert unter `.local`; es wird nur gehasht und nach Zeilennummern gelesen. **In dieses Repo ist kein GPL-Text übernommen.** Nicht ausgeführt: die Gesamtsuite, `verify_packet.py`, `tools/verify_*`; Sampler, Ziehungen, Trainings, Altersmodelle, Fossilrekonstruktionen.

## Grenzen

- Kein Originalfall, keine echten Transferfunktionen oder Pollenmatrizen. Die Matrizen sind erfunden.
- `recent` und `sources$core` brauchen die finalen v1.2-Eingaben; beide sind offen.
- Die NA/NaN-Regel und die Rechenweise gelten für R 3.5.0 x64 unter Windows. Die Originalpfade (`/automount/…`) deuten auf Linux; dessen Rechenweise wurde nicht geprüft.
- Die Zeilen 1812–1813 (Ziehung), 1829–1839 mit `taxa_change` TRUE und `wt_power` sind nicht enthalten. Alle elf Nachweisbündel bleiben offen; keine Variante gewählt. Frühere Pakete und Module sind bytegleich.
