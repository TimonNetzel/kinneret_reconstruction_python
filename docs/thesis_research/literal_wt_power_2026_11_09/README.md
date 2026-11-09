# `wt_power` und sein Leser (`mcmc_functions_v1.6.R:1003–1012`, `2023–2026`)

Stand 09.11.2026, Ausgangscommit `9afab2e` (PR-Kopf, `MERGEABLE`), bestehender Branch `dissertation-reconstruction`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).
Auftrag: Nutzerauftrag nach der [Übergabe nach reconst/predictant](../CLAUDE_NEXT_STEP_AFTER_RECONST_2026_11_09.md).
Sechzehnter begrenzter Rechenschritt. Explizite Reihen und ein explizites `wb`, keine Ziehung, kein Sampler. Noch kein Originalfall.

## Nutzerentscheidung: nicht bitgleich zu R

Am 09.11.2026 hat der Nutzer entschieden: **NumPy-FFT mit gemessener und ausdrücklich markierter ulp-Abweichung**, keine Übersetzung von R-`fft.c`.
Die Lizenzfrage zu `tools/r_sort_gpl.py` bleibt damit unberührt und offen. In dieses Repo ist **kein GPL-Text** übernommen.

| Schritt | Vergleich mit R 3.5.0, 120 Stressfälle | bitgleich | höchstens | relativ zum Spaltenmaximum |
|---|---|---:|---:|---:|
| Zeile 1004/1005, zentrierte Reihe | 22.256 Werte | **alle** | 0 ulp | 0 |
| `fft(x)` | 44.512 Real- und Imaginärteile | 8.662 | 36.864 ulp | 7,9e-16 |
| `mvfft(...) * invflen` | 79.752 Teile | 10.630 | 393.216 ulp | 1,4e-15 |
| `abs(wave)^2` (Leistung) | 39.876 Werte | 5.938 | 606 ulp | 2,5e-15 |

Große ulp-Zahlen entstehen bei Werten nahe null. Darum steht die Spalte „relativ“ daneben. Die gesamte Kette bis `post_spec` (Zeile 2026) wich auf sechs Kunstketten um höchstens **12 ulp** ab, absolut 8,3e-17.
Auch R ist nicht exakt: Auf den 17 Referenzfällen liegt die NumPy-Leistung höchstens 8 ulp vom wahren Wert (90-stellige direkte DFT) entfernt, R 3.5.0 höchstens 15. Das ist eine Messung auf Kunstwerten, keine allgemeine Qualitätsaussage.
Allein das Modul aus `hypot` unterscheidet sich schon: Auf R-eigenem `wave` sind 24.795 von 39.876 Leistungen bitgleich, höchstens 5 ulp daneben.

## Abweichungen von der Übergabe, belegt

1. **`wt_power` läuft nicht nur in Zeile 2023.** Bei `sources$speleothem == 1` rufen auch `mcmc_prepare.R:947` und `966` (100-mal in `system.time`) die Funktion auf, ebenso `wt_cpp` in `950` und `971`. Die Ergebnisse `test`/`test2` haben keinen Leser.
2. **`mcmc_prepare.R:1145` setzt für Kinneret `sources$speleothem <- 0`, aber nur hinter `if(case_studies)`.** `main_v1.2.R:672` setzt `case_studies <- F`, im gebundenen Lauf greift die Zeile also nicht. Die Erreichbarkeit bleibt offen, wie die Übergabe sagte.
3. **Keine Maskierung gefunden** ([fft_masking.json](fft_masking.json)): 17 CRAN-Fassungen von `signal`, `pracma`, `WaveletComp`, `biwavelet` im Fenster 2018-04-23…2023-07-07 exportieren und definieren weder `fft` noch `mvfft`. Die übrigen angehängten Pakete sind **nicht** geprüft, und welche Fassungen der historische Lauf hatte, ist unbekannt.

## Erreichbarkeit

[bind_source.py](bind_source.py) → [Bindung](source_binding.json), [Determinismusprüfung](determinism_audit.json).

| Stelle | Bedingung | Kinneret (`add.sites <- T`, Slot 3) |
|---|---|---|
| `mcmc_functions_v1.6.R:2023–2026` | `spectral` | **offen:** `mcmc_prepare.R:1203` braucht `sources$speleothem == 1`, das hängt an `d18o.age` und den finalen Koordinaten (759–814) |
| `mcmc_prepare.R:947`, `966` | `sources$speleothem == 1` | offen, gleiche Bedingung; ohne Leser |
| `mcmc_functions_v1.6.R:2031` | `hi_spec` | nicht erreicht: `mcmc_prepare.R:881` schließt Kinneret aus |
| `mcmc_functions_v1.6.R:991` | im `reconstruction` aus Zeile 855 | nicht erreicht: Zeile 1781 ersetzt die Definition |
| Zeile 1007 (`test <- …`) | keine | läuft, ohne Leser; im Modul weggelassen |

**Markiert, nicht korrigiert:**

- `mcmc_prepare.R:1817` indiziert das längere `ages_spec` mit dem kürzeren `hi_ages_spec`, das recycelt wird. Für Orte mit `hi_spec` kann `length_ages_spec2` dadurch von der Länge der Reihe abweichen. Kinneret ist keiner davon (`hi_ages_spec` ist überall TRUE, Zeile 885).
- `site.age` wird in `mcmc_prepare.R:825–826` überschrieben. `ages_spec` muss darum nicht zur Länge von `predictant_current` passen. Das Modul wendet die logische Indizierung von R an und lehnt eine Reihe ab, die dann nicht mehr zu `wb` passt.

## Implementierter Umfang

[historical_wt_power.py](../../../tools/historical_wt_power.py), [Tests](../../../tests/test_historical_wt_power.py).

- **Exakt (Zeile 1004/1005):** `sum(d)` in 80-Bit-`long double`, einmal auf `double` gewandelt, dann `/ l_d`. `l_d` wird in R nie mit `length(d)` verglichen.
- **Mit markierter Abweichung (1006–1010):** `f * wb` ist komplex mal reell, das inverse `mvfft` ist unskaliert, `abs(wave)^2` ist `hypot` und dann `x * x`, nicht `re² + im²`. Jedes Ergebnis nennt sein `backend`.
- **Dimensionen:** `wave[seq_len(n.obs), ]` fällt bei einer Zeile oder einer Spalte zum Vektor. `[coi_array == 1]` liest spaltenweise.
- **Leser:** Zeile 2023–2026 mit dem lokalen `cor` (na.rm) und der Ersetzung durch `1e-10`.
- **`wb` ist eine explizite Eingabe.** `rcpp_wt_bases_morlet` (biwavelet) gehört nicht zu diesem Schritt. In allen vier gelesenen biwavelet-Fassungen liefert es `NumericVector`, `wb` ist also reell.
- **Abgelehnt statt nachgebildet:** komplexes oder nicht endliches `wb`, falsche Zeilenzahl, `n_obs` außerhalb `1..nrow(wb)`, ein `coi_array` anderer Form oder mit anderen Werten als 0/1, Überlauf. Die Formalargumente `n.obs=n.obs` sind selbstbezüglich, R stoppt ohne sie; hier sind sie Pflicht.
- **NA, NaN, Inf** (gemessen in R 3.5.0, keine Herleitung): `sum` behält das erste fehlende Token der Reihe. Ein fehlender linker Operand bleibt, sonst übernimmt `d - mean` das Token des Mittels. Eine fehlende oder unendliche Eingabe macht alle Leistungswerte fehlend. Bei einheitlichem Token gilt dieses; bei einer Mischung beansprucht das Modul nur „fehlend“ (`kind` `mixed`).

## Unabhängige Referenz und R-Gegenprobe

[Fixture](synthetic_fixture.json): 26 Kunstfälle, 6 Ketten, 120 Stressfälle in der Form des lebenden Pfads (`n = 2^k`). Die Bänder sind selbst geschrieben und imitieren nur die Form einer Morlet-Tochter; sie sind **keine** biwavelet-Ausgabe. [Referenz](precise_reference.py): direkte DFT mit 90 Stellen, ohne NumPy und ohne das Modul ([Sollwerte](reference_expectations.json)). [Ergebnisse](synthetic_result.json).

[run_r_crosscheck.py](run_r_crosscheck.py), [r_crosscheck.R](r_crosscheck.R) lesen `1003–1012`, `2023–2026`, `2355–2366` und `mcmc_prepare.R:1816–1817` **wörtlich aus der Originaldatei** in R 3.5.0. Eingaben als C99-Hex. Ergebnis: [r_crosscheck_result.json](r_crosscheck_result.json). Keine Token- oder Fehlend-Abweichung.

[Gegenbeispiele](arithmetic_check.json), keine Varianten: Ein Double-Akkumulator im Mittel ändert 4.052 von 22.256 zentrierten Werten, `re² + im²` wie im ungenutzten `wt_cpp` 26.350 von 39.876 Leistungen. Ein skaliertes `ifft` mal `n` und eine reelle FFT-Eingabe ändern hier nichts (`n = 2^k`).

Im [Prüfnachweis](verification.json) stehen die tatsächlich gelaufenen Testzahlen.

## Prüfen

```powershell
.venv/Scripts/python.exe -I docs/thesis_research/literal_wt_power_2026_11_09/check_fft_masking.py .local/wt_power_2026_11_09/cran
.venv/Scripts/python.exe docs/thesis_research/literal_wt_power_2026_11_09/bind_source.py .local/reconst_predictant_2026_11_09/r_source/R-3.5.0.tar.gz
.venv/Scripts/python.exe docs/thesis_research/literal_wt_power_2026_11_09/make_fixture.py
.venv/Scripts/python.exe docs/thesis_research/literal_wt_power_2026_11_09/check_wt_power.py
.venv/Scripts/python.exe docs/thesis_research/literal_wt_power_2026_11_09/precise_reference.py
.venv/Scripts/python.exe docs/thesis_research/literal_wt_power_2026_11_09/run_r_crosscheck.py .local/R/R-3.5.0/bin/x64/Rscript.exe
.venv/Scripts/python.exe docs/thesis_research/literal_wt_power_2026_11_09/arithmetic_check.py
.venv/Scripts/python.exe docs/thesis_research/literal_wt_power_2026_11_09/record_verification.py
```

Nicht ausgeführt: die Gesamtsuite, `verify_packet.py`, `tools/verify_*`; Sampler, Ziehungen, Trainings, Altersmodelle, Fossilrekonstruktionen.

## Grenzen

- Kein Originalfall, keine echte Speleothemreihe, kein echtes `wb` und kein echtes `coi_array`. Alle Zahlen sind erfunden.
- Die Abweichung ist auf Kunstfällen mit `n` bis 512 gemessen. Gültig für R 3.5.0 x64 unter Windows und NumPy 2.5.4; die Originalpfade deuten auf Linux, dessen Rechenweise wurde nicht geprüft.
- Ob `wt_power` für Kinneret überhaupt läuft, ist offen. `rcpp_wt_bases_morlet`, `biwavelet::wt` und die Glättung der Speleothemreihe sind nicht enthalten.
- Alle elf Nachweisbündel bleiben offen; keine Variante gewählt. Frühere Pakete und Module sind bytegleich.
