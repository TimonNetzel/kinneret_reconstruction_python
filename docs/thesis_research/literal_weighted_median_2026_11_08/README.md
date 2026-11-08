# `weightedMedian` für die NA-Maske von `which.tf.missing` (`mcmc_prepare.R:1628–1664`)

Stand 08.11.2026, Ausgangscommit `cfd0ad8` (PR-Kopf), bestehender Branch `dissertation-reconstruction`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).
Auftrag: [Übergabe nach dem Dirichlet-Quotienten](../CLAUDE_NEXT_STEP_AFTER_DIRICHLET_RATIO_2026_11_03.md).
Zwölfter begrenzter Rechenschritt: **expliziter Aufruf, keine echten Transferfunktionen.** Noch kein vollständiger
Originalfall.

## Wann ist das Ergebnis NA?

Das war die offene Frage für die Maske `which.tf.missing`. Für `weightedMedian(x, w)` mit Standardwerten und
Double-Vektoren gilt, aus dem C-Text gelesen und gegen R bestätigt:

| Bedingung | Ergebnis |
| --- | --- |
| irgendein Gewicht ist NA oder NaN | **NA**, unabhängig von `x` und von `+Inf`-Gewichten |
| kein Gewicht `> 0` (also auch alle Gewichte 0) | **NA** |
| kein `+Inf`, ein `x` ist NA/NaN bei endlichem Gewicht `> 0` | **NA** |
| erstes `+Inf` an Stelle p, ein `x` ist NA/NaN bei endlichem Gewicht `> 0` vor p | **NA** |
| ein `x` ist NA/NaN bei Gewicht `+Inf` oder `-Inf`, sobald ein `+Inf` existiert | **NA** |
| ein `x` ist NA/NaN bei Gewicht `<= 0` ohne `+Inf`, oder bei endlichem Gewicht hinter dem ersten `+Inf` | wird nicht gelesen, **kein NA** |
| sonst | Zahl; `NaN` (nicht NA) nur bei unendlichem `x` durch `Inf − Inf` oder `0 · Inf` |

**Folge für die Maske:** Eine TF-Zeile fällt genau dann in `which.tf.missing`, wenn sie irgendein NA/NaN enthält oder
kein positives Gewicht hat. Das gilt bei endlichem `x1range.bck`. Zeile 1664 prüft nur den Median von `x1`; ein NA
allein im `x2`-Median entfernt das Taxon nicht. Die Reihenfolgeabhängigkeit bei `+Inf` zeigen die Kunstfälle
`inf_order_na_before` und `inf_order_na_after`.

## Beleg zuerst

[bind_source.py](bind_source.py) → [Bindung](source_binding.json), [Auszüge](source_excerpt.json),
[Determinismusprüfung](determinism_audit.json), [Binärpaket](binary_package.json), [R-Sortierung](r_sort_source.json).

- `mcmc_prepare.R` per Hash; Schleifen 1629–1640 und 1649–1660 und Zeile 1664 auf oberster Ebene.
- `taxa_change <- FALSE` in 1462; 1501 steht wie bekannt hinter `& FALSE`.
- 1623/1624 (Erwartungswert) werden von `c()` in 1628/1648 überschrieben und nie gelesen. `tann_expand`/`pann_expand`
  (`spline`) haben außerhalb von Kommentaren keine Leser.
- `x1range.bck` stammt aus dem fehlenden TF-Container. Double wird aus dem Aufruf und aus `all_in_all.R` nur
  angenommen; ein integer `x` liefe über den Integer-Kern. Nicht gebunden.

**Paketbindung nur im Fenster** 2018-04-23…2023-07-07 ([Laufzeitbelege](../r_runtime_crosscheck_2026_11_03/README.md)).

## 13 Fassungen, ein Rechenweg

[compare_helpers.py](compare_helpers.py) → [Vergleich](helper_comparison.json), [Diffs](helper_diffs.txt). Tarballs
wurden nur im Speicher gelesen, die Hashes entsprechen den zuvor aufgezeichneten.

- Gemeinsame Hilfsdateien ändern sich in 0.60.1 (Indexumbau, `useNames`), 0.56.0 (Fehlertexte) und 1.0.0
  (`R_INDEX_GET` mit zusätzlichem Schalter `check_i_NA`).
- Für `idxs = NULL` wird daraus überall `x[i]`. Mit entfernten Kommentaren und einheitlich geschriebenem Indexausdruck
  ist der C-Kern in **allen 13 Fassungen ein einziger Text**.
- `X_QSORT_I` = `R_qsort_I`, `X_PSORT` = `rPsort`, `X_ISNAN` = `ISNAN`, `X_NA` = `NA_REAL` in allen Fassungen.

**R-Sortierung** ([R 3.5.0, CRAN](r_sort_source.json)): Im Fenster gab es an `qsort-body.c` nur den Commit
`c275d31` (`if(I) --I`, ohne Wirkung bei gesetztem Index), an `psort_body` und `rcmp` keine Änderung bis zum Fensterende.
Die Historie stammt aus dem Spiegel `wch/r-source`.

## Implementierter Umfang

[historical_weighted_median.py](../../../tools/historical_weighted_median.py),
[r_sort_gpl.py](../../../tools/r_sort_gpl.py), [Tests](../../../tests/test_historical_weighted_median.py).

Quellnah übernommen:

- die Schleife über die Gewichte einschließlich des Rücksprungs bei `+Inf`;
- `-Inf` zählt im `+Inf`-Zweig als unendlich (`isinf`), außerhalb wird es verworfen;
- der Gleichgewichtszweig liest bei ungeradem n `x[half-1]`, nachdem um `half` partitioniert wurde. Das ist im
  Allgemeinen **nicht** der Median: `x = (3, 1, 2)` mit `+Inf` ergibt 1;
- `R_qsort_I` ist instabil, gleiche `x` behalten eine bestimmte Reihenfolge, die das Ergebnis beeinflusst;
- Interpolation mit `wcum[i] = Σ − w/2`, jede Double-Operation einzeln.

Abgelehnt statt emuliert: eine Gesamtmasse, die auf `Inf` überläuft (oder Rundung, die `half` auf `n + 1` belässt).
C liest dann außerhalb seiner Felder; das ist undefiniert.

**Lizenz:** `r_sort_gpl.py` ist eine Übersetzung von R-Quellen und bleibt GPL-2-oder-später, abweichend vom MIT-Rest des
Repositorys. Der Kern in `historical_weighted_median.py` ist aus matrixStats (Artistic-2.0) übersetzt und nennt das.

## Unabhängige Referenz, Handwerte, Gegenbeispiele

[Fixture](synthetic_fixture.json) mit 29 Kunstfällen, einer TF-Matrix und 400 deterministischen Stressfällen
([Erzeuger](make_fixture.py)). Die [Referenz](precise_reference.py) entscheidet NA über eine geschlossene Regel statt
der Schleifen und rechnet endliche Werte in exakten Brüchen, mit einmaliger Rundung je IEEE-Operation; das Modul
importiert sie nicht ([Sollwerte](reference_expectations.json)). [Prüfskript](check_weighted_median.py) →
[Ergebnisse](synthetic_result.json), [Gegenbelege](arithmetic_check.json).

- Alle 29 Fälle stimmen bitgleich mit der Referenz überein; die Handwerte werden auf 0 ulp getroffen.
- Die NA-Regel stimmt für alle 400 Stressfälle.
- Gegenbeispiele (keine Kandidatenvarianten):
  - stabile Sortierung statt `R_qsort_I`: 1,4 statt 1,2857… bei Ties;
  - Maske nur aus „Summe 0“: `[2]`;
  - NA wie Gewicht 0: `[2, 5]`;
  - Maske aus x1 oder x2: `[2, 3, 4, 5]`;
  - Quelle 1664: `[2, 3, 5]`.

## R 3.5.0 mit matrixStats 0.56.0

Vor der Installation festgehalten ([binary_package.json](binary_package.json)):
`cran-archive.r-project.org/bin/windows/contrib/3.5/matrixStats_0.56.0.zip`, SHA-256
`71740915432553e0f88ae493b937c2c4cceab97aa203da5749d21ed5aba382cb`, gebaut unter R 3.5.3. Das ist die einzige
Fassung für R 3.5 im Archiv, sie liegt im Fenster. Installation nur in `.local/R/library_ms`.

[run_r_crosscheck.py](run_r_crosscheck.py), [r_crosscheck.R](r_crosscheck.R) werten die 29 Fälle und die 400
Stressfälle mit `weightedMedian` aus sowie **1628–1640, 1648–1660 und 1664 wörtlich aus der Originaldatei** auf der
Kunst-Matrix. Ergebnis: [r_crosscheck_result.json](r_crosscheck_result.json).

- **439 Werte, alle bitgleich** (0 ulp). NA und NaN werden unterschieden und stimmen überein.
- `which.tf.missing` stimmt: `2 3 5`.
- Gelaufen ist nur diese eine Fassung. Die anderen zwölf sind nur als Quelltext verglichen.

## Zusatzbeleg: Was `tann <- FALSE` im Kinneret-Lauf bewirkt

[bind_tann.py](bind_tann.py) → [tann_binding.json](tann_binding.json), nur gesammelt, nicht gedeutet.

- 1248 ist die Bedingung, die Zuweisung steht in **1249**. Frühere Dokumente zitieren 1248; sie bleiben eingefroren.
- Oberste Ebene; 1209 nimmt Kinneret aus; danach folgt keine Zuweisung mehr.
- `reconstruction` ist in `mcmc_functions_v1.6.R` zweimal definiert (855, 1781), die spätere bleibt beim `source` übrig.
- Darin liest genau **Zeile 1939** `tann`: Bei `FALSE` wird `predictant_pann` zu `predictant_current`.
- `predictant_current` wird in 1964, 1999, 2023 und 2031 weiterverwendet. Was das für die Korrelationen bedeutet, ist
  hier nicht bewertet.

## Prüfen

```powershell
.venv/Scripts/python.exe -I docs/thesis_research/literal_weighted_median_2026_11_08/compare_helpers.py .local/evidence_r_env_2026_11_03/cran_cache/tarballs
.venv/Scripts/python.exe docs/thesis_research/literal_weighted_median_2026_11_08/bind_source.py .local/evidence_r_env_2026_11_03/cran_cache/tarballs .local/weighted_median_2026_11_08/dl_rsrc/R-3.5.0.tar.gz .local/weighted_median_2026_11_08/dl_bin/matrixStats_0.56.0.zip .local/weighted_median_2026_11_08/dl_bin/headers.txt
.venv/Scripts/python.exe docs/thesis_research/literal_weighted_median_2026_11_08/bind_tann.py
.venv/Scripts/python.exe docs/thesis_research/literal_weighted_median_2026_11_08/make_fixture.py
.venv/Scripts/python.exe docs/thesis_research/literal_weighted_median_2026_11_08/precise_reference.py
.venv/Scripts/python.exe docs/thesis_research/literal_weighted_median_2026_11_08/check_weighted_median.py
.venv/Scripts/python.exe docs/thesis_research/literal_weighted_median_2026_11_08/run_r_crosscheck.py .local/R/R-3.5.0/bin/x64/Rscript.exe .local/R/library_ms
.venv/Scripts/python.exe -m pytest tests/test_historical_weighted_median.py <die vierzehn bisherigen Testdateien> -q
```

**Ergebnis:** siehe [Prüfnachweis](verification.json). Nicht ausgeführt: die Gesamtsuite, `verify_packet.py` und
`tools/verify_*`.

## Grenzen

- Kein Originalfall, keine echten Transferfunktionen; deren Zeilen und Raster bleiben unbekannt.
- Der Anbieter ist nur im Fenster gebunden; welche Fassung der Lauf nutzte, bleibt offen.
- Bitgleichheit gilt für R 3.5.0, x86_64, Windows. Für eine i386-/x87-Fassung ist sie nicht geprüft.
- Alle elf Nachweisbündel bleiben offen; keine Variante gewählt. Frühere Pakete, Module, Fixtures und Tests sind
  bytegleich.
