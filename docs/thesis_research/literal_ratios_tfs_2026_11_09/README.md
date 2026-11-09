# `ratios_tfs_no_ages` und sein Leser (`mcmc_functions_v1.6.R:1485–1525`, `mcmc_conduction.R:228–230`)

Stand 09.11.2026, Ausgangscommit `3ddf98e` (PR-Kopf, `MERGEABLE`), bestehender Branch `dissertation-reconstruction`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).
Auftrag: Nutzerauftrag nach der [Übergabe nach der rtruncnorm-Bindung](../CLAUDE_NEXT_STEP_AFTER_RTRUNCNORM_BINDING_2026_11_09.md).
Siebzehnter begrenzter Rechenschritt. Explizite Vektoren und Matrizen, keine Ziehung, kein Sampler. Noch kein Originalfall.

## Nutzerentscheidung: nicht bitgleich zu R

Am 09.11.2026 hat der Nutzer entschieden: **eigene Rechnung mit gemessener und ausdrücklich markierter Abweichung** für `d_truncnorm`, keine Übersetzung von R-`dnorm.c` und `pnorm.c`.
Die Rechnung nutzt die Lehrbuchformeln auf `math.exp` und `math.erfc`. In dieses Repo ist **kein GPL-Text** übernommen.

| Vergleich mit R 3.5.0, 120 Stressfälle | Werte | bitgleich | höchstens | relativ |
|---|---:|---:|---:|---:|
| Gitterquotienten (Zeilen 1507–1513) | 3.406 | **alle** | 0 ulp | 0 |
| Gitterindizes | alle | **alle** | – | – |
| `prod` mit den Faktoren aus R (exakte Nachbildung) | 120 | **alle** | 0 ulp | 0 |
| `d_truncnorm` | 6.812 | 5.588 | 22 ulp | 3,2e-15 |
| Dichtequotienten (Zeilen 1516–1522) | 3.406 | 2.533 | 15 ulp | 3,3e-15 |
| `transfer_ratio` (Zeile 230) | 120 | 39 | 35 ulp | 4,5e-15 |

Keine Abweichung bei NaN oder Inf. Die Gegenprobe ist auf Kunstfällen gemessen und gilt für R 3.5.0 x64 unter Windows und Python 3.12.10.

Größere ulp-Zahlen bei `p_a` allein sind harmlos: R gibt bei 13 Werten exakt 0 aus, `erfc` einen winzigen Wert (höchstens 8e-309). Alle Werte über 256 ulp sind kleiner als 2e-49, die Skala dabei mindestens 0,5.
Auch R ist nicht exakt: Die 150-stellige Referenz findet auf 23 Quotienten bei Modul und R denselben größten Abstand von 14.518 ulp. Er liegt im Kunstfall mit sehr weitem Vorschlag, wo `p_b - p_a` stark auslöscht.

## Abweichungen von der Übergabe, belegt

1. **Die C++-Zeilen konnten nicht wörtlich laufen.** Hier gibt es keinen Compiler und kein Rcpp. R 3.5.0 führt stattdessen die **R-Fassung derselben Funktion wörtlich** aus (`mcmc_functions_v1.6.R:1529–1552`, im Original auskommentiert), dazu `mcmc_conduction.R:228` und `230`. `dtnorm` ist im Prüfskript eine kurze R-Formulierung der gestutzten Dichte und nutzt die echten `pnorm` und `dnorm` von R 3.5.0, also dieselben Funktionen, die `R::pnorm` und `R::dnorm` aufrufen.
2. **Doppelte Rundung.** `prod` multipliziert in 80 Bit und wandelt einmal nach `double`. Das Ergebnis kann vom korrekt gerundeten Produkt abweichen. Die Nachbildung ist bitgleich zu R, sobald R seine eigenen Faktoren liefert.

## Erreichbarkeit

[bind_source.py](bind_source.py) → [Bindung](source_binding.json).

| Stelle | Bedingung | Kinneret (`add.sites <- T`, Slot 3) |
|---|---|---|
| `mcmc_conduction.R:228` | keine, in der Schleife ab Zeile 138 | läuft in jeder Iteration |
| `mcmc_conduction.R:230` | keine | läuft in jeder Iteration, Leser ist Zeile 302 (`acc_prob`) |
| `mcmc_functions_v1.6.R:1437` | überschrieben | nicht die aktive Fassung: nur die Argumentnamen von 1485 passen zum Aufruf |

Für `d_truncnorm` gilt die [Bindung an RcppDist](../literal_rtruncnorm_binding_2026_11_09/README.md). `which_min.h` ist in allen **19** Rcpp-Fassungen des Fensters 2018-04-23…2023-07-07 bytegleich; `abs` ist `fabs`. Installierte Fassungen des historischen Laufs sind unbekannt.

## Implementierter Umfang

[historical_ratios_tfs.py](../../../tools/historical_ratios_tfs.py), [Tests](../../../tests/test_historical_ratios_tfs.py).

- **Gitter:** erstes Minimum von `|range - Wert|`, weil `which_min` strikt mit `<` vergleicht.
- **Quotienten:** IEEE-Division, also 0/0 → NaN und x/0 → ±Inf.
- **Dichte:** `d_truncnorm(old, new, …)` geteilt durch `d_truncnorm(new, old, …)`; ein Wert außerhalb von `[a, b]` hat Dichte 0.
- **Produkt:** Reihenfolge `tann_tf`, `pann_tf`, `tann_dists`, `pann_dists`.
- **Abgelehnt statt nachgebildet:** nicht endliche oder fehlende Eingaben (`which_min` gäbe NA, der folgende Index stoppt in Rcpp; das lässt sich hier ohne Compiler nicht laufen lassen), `sigma <= 0`, `a > b`, falsche Längen und Formen, ein Produkt außerhalb des 80-Bit-Bereichs.

## Unabhängige Referenz und R-Gegenprobe

[Fixture](synthetic_fixture.json): 14 Kunstfälle, 120 Stressfälle. [Referenz](precise_reference.py): 150 Stellen, ohne NumPy, ohne `erfc` und ohne das Modul ([Sollwerte](reference_expectations.json)). [Ergebnisse](synthetic_result.json). Für `equal_bounds` und den 1000er-Fall gibt es keine Referenz.

[run_r_crosscheck.py](run_r_crosscheck.py), [r_crosscheck.R](r_crosscheck.R), Ergebnis: [r_crosscheck_result.json](r_crosscheck_result.json).

[Gegenbeispiele](arithmetic_check.json), keine Varianten: Ein Double-Akkumulator ändert 68 von 127 Produkten; ohne die Skala der Stutzung ändern sich 2.186 von 3.406 Dichtequotienten; das letzte statt des ersten Minimums wählt im Bindungsfall Index 1 statt 0.

Im [Prüfnachweis](verification.json) stehen die tatsächlich gelaufenen Testzahlen: 212 neue Tests, 1.858 gezielte Tests grün in 22 Dateien.

## Prüfen

```powershell
.venv/Scripts/python.exe docs/thesis_research/literal_ratios_tfs_2026_11_09/make_fixture.py
.venv/Scripts/python.exe docs/thesis_research/literal_ratios_tfs_2026_11_09/check_ratios.py
.venv/Scripts/python.exe docs/thesis_research/literal_ratios_tfs_2026_11_09/precise_reference.py
.venv/Scripts/python.exe docs/thesis_research/literal_ratios_tfs_2026_11_09/run_r_crosscheck.py .local/R/R-3.5.0/bin/x64/Rscript.exe
.venv/Scripts/python.exe -I docs/thesis_research/literal_ratios_tfs_2026_11_09/bind_source.py .local/ratios_tfs_2026_11_09/cran/Rcpp .local/reconst_predictant_2026_11_09/r_source/R-3.5.0.tar.gz
.venv/Scripts/python.exe docs/thesis_research/literal_ratios_tfs_2026_11_09/arithmetic_check.py
.venv/Scripts/python.exe -I docs/thesis_research/literal_ratios_tfs_2026_11_09/record_verification.py
```

Nicht ausgeführt: Kompilierung der C++-Zeilen, die Gesamtsuite, `verify_packet.py`, `tools/verify_*`; Sampler, Ziehungen, Trainings, Altersmodelle, Fossilrekonstruktionen.

## Grenzen

- Kein Originalfall. Alle Gitter, Dichten und Vorschläge sind erfunden; die numerischen Transferfunktionen fehlen weiter.
- `dtnorm` im Prüfskript ist keine Ausführung des C++-Headers. Der C++-Text ist nur gelesen.
- Die Abweichung ist auf Kunstfällen gemessen. Die Originalpfade deuten auf Linux; dessen `exp` und Rechenweise sind nicht geprüft.
- Alle elf Nachweisbündel bleiben offen; keine Variante gewählt. Frühere Pakete und Module sind bytegleich.
