# Lokales `cor` und seine Zeitraum-Leser (`mcmc_functions_v1.6.R:2355–2366`, `1939–2018`)

Stand 09.11.2026, Ausgangscommit `437f6f4` (PR-Kopf), bestehender Branch `dissertation-reconstruction`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).
Auftrag: Nutzerauftrag nach der [Übergabe nach reconstruction_climate](../CLAUDE_NEXT_STEP_AFTER_RECONSTRUCTION_CLIMATE_2026_11_08.md).
Vierzehnter begrenzter Rechenschritt. Explizite Vektoren, keine Ziehung, kein Sampler. Noch kein Originalfall.

## Erreichbarkeit: vor dem Nachbau belegt

[bind_source.py](bind_source.py) → [Bindung](source_binding.json), [Auszug](source_excerpt.json), [Determinismusprüfung](determinism_audit.json).
Kommentare und Zeichenketten sind vor der Suche entfernt. Die Schleife `main_v1.2.R:651–684` ist die benutzte Bindung; die Datei als Ganzes
ist nicht parsbar (überzählige Klammern in 740, 788, 829).

- **Eine einzige Definition.** `cor <- function` steht in 1 von 540 `.R`-Dateien, hier auf oberster Ebene. `main_v1.2.R:644` sourct sie
  vor der Schleife in die globale Umgebung, daher überdeckt sie `stats::cor` für allen Code dort.
- **Keine Paketfunktionen.** Es werden nur `is.na`, `length`, `sum`, `sqrt`, `%*%`, `^`, `/`, `-` und `[` benutzt.
  Darum ist kein Anbieter zu binden.
- **Tote Aufrufe:** `correlation_decomposition` (136–138) hat keinen Aufrufer; 965–992 gehören zur ersten `reconstruction`-Definition, die 1781 ersetzt.
  `mcmc_prepare.R:1084` berechnet `cor23` nur außerhalb Israels und hat keinen lebenden Leser (1093 ist auskommentiert).

| Zeile | Bedingung | Kinneret (`add.sites <- T`, `case_studies <- F`) |
|---|---|---|
| 1964, 1966 | `multiple` | **erreicht, wenn `sources$core` 1 ist** |
| 1993 | `!multiple`, `sources[,4] == 1` | nur wenn `sources$core` 0 ist |
| **1999** | `!multiple`, `sources[,4] != 1` | **nicht erreicht:** `sources[,4]` bleibt 1 |
| 1982, 2009 | `hi_time` | nicht erreicht: `add.sites` setzt `hi_time <- FALSE` (`mcmc_prepare.R:718`) |
| 2024 | `spectral` | hängt an `sources$speleothem`, braucht `wt_power`, hier nicht nachgebaut |
| 2032 | `spectral & hi_spec` | nicht erreicht: `mcmc_prepare.R:881` schließt Kinneret aus |

**Die Übergabe nannte 1964–1970 und 1999. Das gilt nur für 1964–1970.** Zeile 1999 ist im Kinneret-Zweig nicht erreichbar, weil
`sources[,4]` nur in 1072 (unbedingt 1) gesetzt und für Kinneret nirgends zurückgesetzt wird (1123 in `if(FALSE)`, 1143–1173 hinter `case_studies`,
1212 namentlich ausgeschlossen). 1999 ist trotzdem nachgebaut und als unerreichbar markiert.

**`multiple` ist für Kinneret genau `sources$core`** ([Tabelle](source_binding.json), alle vier Kombinationen).
Speläothem ändert daran nichts; `mcmc_prepare.R:1209` und `1248–1252` setzen für Kinneret nichts an `multiple`.
Dass `sources$core` 1 wird, verlangt zwei Dinge:

1. `!inside.giesecke`. Gemessen: Die dem Standort nächste Giesecke-Breite ist 34,75° mit einer einzigen Länge 24,75°, Kinneret liegt bei etwa 35,5° Ost, also außerhalb.
   Die Koordinaten stammen aus dem **älteren** Container `core_data_all_andrea_taxa_additional_sites.rdata`, nicht aus dem fehlenden v1.2-Stand.
   Der Befund trägt aber für jeden Ort südlich von 35° und östlich von 25,75°.
2. `var(predictors$pollen…) != 0` (`mcmc_prepare.R:662`). **Ohne die finalen Eingaben nicht auswertbar.** Das bleibt die offene Restbedingung.

Trifft sie nicht zu, läuft 1993 statt 1964/1966. Auch das ruft `cor` auf.

Für Kinneret gilt außerdem `tann <- FALSE`: Zeile 1964 liest `predictant_pann`, **Zeile 1966 liest trotzdem `predictant_tann`.**
`hi_ages_time` bleibt die Maske `!inside.range(pollen.age, c(-60,6500))`, obwohl `hi_time` FALSE ist. Ob sie zu den ausgewählten Vektoren passt,
hängt von den finalen Altern ab; das Modul bildet die R-Regel nach (kurze Maske wird recycelt, zu lange liest NA).

## Implementierter Umfang

[historical_local_cor.py](../../../tools/historical_local_cor.py), [Tests](../../../tests/test_historical_local_cor.py).

- **`sum` in `long double`:** Jede Addition auf 64 Bit (nächste gerade Zahl), genau ein Abschneiden auf Double am Ende. Mit einer einfachen Double-Summe
  ändern sich 199 von 300 endlichen Stressfällen ([Gegenbeleg](arithmetic_check.json)). Eine exakte Summe wäre ebenfalls falsch (Fall `x87_rounding_step`).
- `%*%` ist bei endlichen Zahlen Referenz-BLAS `dgemv`: eine einfache Double-Summe in Indexfolge. `v^2` ist `v*v`.
- **NA und NaN:**
  - `na.rm=TRUE` prüft **nur `x`**. NA oder NaN in `x` entfernen die Position aus beiden Vektoren; ein NA in `y` bleibt und macht das Ergebnis fehlend.
  - Ob das Ergebnis NA oder NaN heißt, steht nicht in der R-Quelle. **Gemessen in R 3.5.0:** Der erste fehlende Wert in `x` entscheidet, sonst der erste in `y`.
    Leere Eingabe, ein einziger Wert und Varianz 0 geben NaN. Diese Regel ist ein Messbefund, keine Herleitung.
- **Nachbehandlung `< 0 | is.na(...)` → `1e-10`:** Bei NA oder NaN ist `NA | TRUE` TRUE, beide werden gleich behandelt. Null und `-0` werden nicht ersetzt. Quadriert wird danach, also 1e-20.
  Werte knapp über 1 bleiben stehen.
- Abgelehnt statt emuliert: unendliche Eingaben, Überlauf, nach dem Filtern ungleiche Längen, `hi_time` TRUE.

## Unabhängige Referenz und R-Gegenprobe

[Fixture](synthetic_fixture.json): 23 Kunstfälle, 8 Leserfälle, 400 deterministische Stressfälle. [Referenz](precise_reference.py): exakte Brüche mit
expliziten Rundungsregeln, ohne Modulimport ([Sollwerte](reference_expectations.json)). [Ergebnisse](synthetic_result.json).

[run_r_crosscheck.py](run_r_crosscheck.py), [r_crosscheck.R](r_crosscheck.R) lesen `2355–2366` und `1939–2018` **wörtlich aus der Originaldatei** in R 3.5.0. Eingaben als C99-Hex.
Ergebnis: [r_crosscheck_result.json](r_crosscheck_result.json). **447 von 447 Werten sind bitgleich** (0 ulp), keine einzige NA/NaN-Abweichung.
Bei ungleichen Längen brechen R und Modul gemeinsam ab. Die Konstanten 1e-10, 0,95 und 0,5 sind in R und Python dieselben Doubles.
Im [Prüfnachweis](verification.json): 135 neue, 1.037 frühere, zusammen 1.172 Tests grün.

## Prüfen

```powershell
.venv/Scripts/python.exe docs/thesis_research/literal_local_cor_2026_11_09/bind_source.py
.venv/Scripts/python.exe docs/thesis_research/literal_local_cor_2026_11_09/make_fixture.py
.venv/Scripts/python.exe docs/thesis_research/literal_local_cor_2026_11_09/precise_reference.py
.venv/Scripts/python.exe docs/thesis_research/literal_local_cor_2026_11_09/check_local_cor.py
.venv/Scripts/python.exe docs/thesis_research/literal_local_cor_2026_11_09/run_r_crosscheck.py .local/R/R-3.5.0/bin/x64/Rscript.exe
```

Nicht ausgeführt: die Gesamtsuite, `verify_packet.py`, `tools/verify_*`; Sampler, Ziehungen, Trainings, Altersmodelle, Fossilrekonstruktionen.

## Grenzen

- Kein Originalfall, keine echten Transferfunktionen oder Referenzkurven.
- `sources$core` und die Altersmaske brauchen die finalen v1.2-Eingaben; beide sind offen.
- Die NA/NaN-Regel und die Rechenweise gelten für R 3.5.0 x64 unter Windows. Die Originalpfade (`/automount/…`) deuten auf Linux; dessen Rechenweise wurde nicht geprüft.
- Zeile 2024 und `wt_power` sind nicht enthalten. Alle elf Nachweisbündel bleiben offen; keine Variante gewählt. Frühere Pakete und Module sind bytegleich.
