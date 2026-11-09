# `rtruncnorm_cpp`: Quellen- und Bindungsprüfung ohne Ziehung (`mcmc_functions_v1.6.R:1556–1570`, `1812–1813`)

Stand 09.11.2026, Ausgangscommit `52a6acc` (PR-Kopf, `MERGEABLE`), bestehender Branch `dissertation-reconstruction`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).
Auftrag: Vorschlag aus der [Übergabe nach wt_power](../CLAUDE_NEXT_STEP_AFTER_WT_POWER_2026_11_09.md), vom Nutzer fortgesetzt.
**Keine Ziehung, keine Kompilierung, kein R-Lauf.** Es ist kein Rechenmodul; darum gibt es keinen neuen App-Stand. Kein Originalfall.

## Ergebnis

| Frage | Befund | Beleg |
|---|---|---|
| Wo steht die Funktion? | genau eine Definition in 95 R-Dateien des Quellbaums, Zeile 1563, im `sourceCpp`-Block 1556–1570 | [source_binding.json](source_binding.json) |
| Wer ruft sie? | nur `mcmc_functions_v1.6.R:1812` und `1813` | ebenda |
| Woher kommt `r_truncnorm`? | `#include <truncnorm.h>` (1558) mit `Rcpp::depends(RcppDist)` (1560) | ebenda |
| Welche Fassung? | im Fenster 2018-04-23…2023-07-07 gab es auf CRAN **genau eine: RcppDist 0.1.1**, veröffentlicht 2018-10-28 | [rcppdist_versions.json](rcppdist_versions.json) |
| Wie zieht sie? | Inversion: ein `R::runif(0, 1)` geht durch `q_truncnorm`, keine Verwerfung | Header 0.1.1, Zeilen 225–228 und 209–223 |
| Wie viel Zufall? | ein `unif_rand` je Element, also `num_taxa` je Aufruf und `2 · num_taxa` je Iteration | aus den Quellen gelesen, **nicht gemessen** |

## Bindung und ihre Grenze

- 0.1.1 ist die einzige CRAN-Fassung, die im Fenster aktuell war. **Vom Fensterbeginn bis 2018-10-27 gab es auf CRAN keine Fassung.**
- Welche Fassung der historische Lauf installiert hatte, ist **unbekannt**. Eine Installation außerhalb von CRAN, etwa von GitHub, ist nicht ausgeschlossen und nicht geprüft.
- 0.1.1.1 (2025-06-12, außerhalb des Fensters) hat einen bytegleichen Header (`c1fdd6889a30…`). Das ist nur ein Vergleich, keine Bindung.
- RcppDist und R stehen unter GPL. Dieses Paket speichert nur Fassungen, Daten, Hashes, Zeilennummern und die Anzahl von Funktionsnamen, **keinen Quelltext**.

## Was die Quellen zeigen

Eigene Beschreibung, keine Übernahme:

- **Rechenweg:** `r_truncnorm` setzt eine Gleichverteilte in das Quantil der gestutzten Normalverteilung ein. `q_truncnorm` nutzt zweimal `R::pnorm` und einmal `R::qnorm` und begrenzt das Ergebnis auf `[a, b]`. Zufall steckt nur in `r_truncnorm`.
- **Zufallsverbrauch:** Die Schleife in Zeile 1565 läuft über `i < n`. Je Element ein `R::runif(0, 1)`. In R 3.5.0 wiederholt `runif.c:36` nur, solange der Wert nicht in (0, 1) liegt, und `RNG.c:97–101` (`fixup`, für Mersenne-Twister Zeile 134) hält die Werte strikt innen. Reihenfolge: erst alle Taxa für `tann` (1812), dann für `pann` (1813).
- **Generator:** `RNG_DEFAULT` ist in R 3.5.0 Mersenne-Twister (`RNG.c:31`, `46`). In `main_v1.2.R`, `mcmc_functions_v1.6.R`, `mcmc_prepare.R` und `mcmc_conduction.R` steht kein aktives `RNGkind` oder `RNGversion`. Andere geladene Dateien und Startprofile sind nicht geprüft.
- **Argumente:** Mittel ist der vorige Zustand. `sd` ist `sd(Stichprobe) · 2,38² / (2 · num_taxa)` (`mcmc_prepare.R:1922–1923`). Untere Grenze ist für `tann` das 5-%-Quantil, für `pann` **0**; obere Grenze das 95-%-Quantil (1913–1914). Die Startwerte `posterior$tf_sample[[1]]` sind die Mediane (1945–1946).

## Markiert, nicht korrigiert

- **Zweiter, getrennter Anbieter:** `mcmc_prepare.R:1886` lädt das R-Paket `truncnorm`, und `1957–1958` ziehen damit `posterior$tf_sample_tann/pann[[1]]`. Das ist **nicht** RcppDist und **nicht gebunden**. Es verbraucht Zufall vor dem Sampler.
- **Zustand beim Samplerstart nicht belegt.** Aktive `set.seed`-Stellen: `mcmc_prepare.R:1297, 1301, 1332, 1341, 1354, 1896, 1898` und `main_v1.2.R:1055`. Weitere Verbraucher stehen unter anderem in 1897/1899 und 1979–1980. Diese Liste ist **keine** vollständige Verfolgung des Zufallszustands.
- **`propose_cpp` (1357) hat keinen aktiven Aufrufer**, sein einziger Aufruf (1799) ist auskommentiert.
- **Aktive Dichte:** `ratios_tfs_no_ages` ist zweimal definiert (1437, 1485). Nur die Argumentnamen der Fassung 1485 passen zum Aufruf `mcmc_conduction.R:228`. Dort liest `d_truncnorm` in 1516–1522. Die Funktion nutzt keinen Zufall (Header 142–152: je zweimal `R::pnorm` und `R::dnorm`).

## Prüfen

[bind_rtruncnorm.py](bind_rtruncnorm.py), [Tests](../../../tests/test_rtruncnorm_binding.py), [Prüfnachweis](verification.json).

```powershell
.venv/Scripts/python.exe -I docs/thesis_research/literal_rtruncnorm_binding_2026_11_09/bind_rtruncnorm.py .local/rtruncnorm_2026_11_09/cran/RcppDist .local/reconst_predictant_2026_11_09/r_source/R-3.5.0.tar.gz
```

Tatsächlich gelaufen: das Skript gegen die eingefrorenen Quell-Hashes, Exit 0; 32 neue Tests, zusammen **1.646** gezielte Tests grün in 21 Dateien, Exit 0.

Nicht ausgeführt: Kompilierung, Ziehung, R-Lauf, App-Tests und Browserläufe (am App-Code hat sich nichts geändert), Gesamtsuite, `verify_packet.py`, `tools/verify_*`; Sampler, Trainings, Altersmodelle, Fossilrekonstruktionen.

## Grenzen

- Der Zufallsverbrauch ist aus dem Quelltext gelesen, nicht gemessen. Die Weiterleitung von `R::runif` auf `runif` in Rcpp ist nicht gegen eine Rcpp-Fassung gebunden.
- Kein `tf_sample`, kein Vorschlag und kein Zustand wurde berechnet. Die Eingaben (`tf.x1.current` usw.) fehlen numerisch.
- Alle elf Nachweisbündel bleiben offen; keine Variante gewählt. Frühere Pakete, Module und App-Stände 1–13 sind unverändert.

Nächste Übergabe: [Claude-Auftrag nach der rtruncnorm-Bindung](../CLAUDE_NEXT_STEP_AFTER_RTRUNCNORM_BINDING_2026_11_09.md).
