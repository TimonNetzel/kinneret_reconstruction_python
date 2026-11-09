# Historisches Gamma-Prior-Raster: `seq`, `which.min` und Varianzverdopplung

Stand 02.11.2026, Ausgangscommit `94f81cc` (PR-Kopf, `main` eingemergt), bestehender
Branch `dissertation-reconstruction`, [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).
Auftrag: [Claude-Übergabe nach dem Taxon-Pruning](../CLAUDE_NEXT_STEP_AFTER_TAXON_PRUNING_2026_11_02.md).
Ziel bleibt der historische Dissertationsnachbau mit Umsetzung in der App.
Dieses Paket ergänzt einen siebten begrenzten Rechenschritt. Noch kein vollständiger Originalfall.

## Welcher Schritt ist deterministisch? (Beleg zuerst)

[Determinismusprüfung](determinism_audit.json) aus dem Quelltext, ohne R-Ausführung.
Sieben Originale, alle per SHA-256 an frühere eingefrorene Bindungen geknüpft und
nur gelesen ([Bindung](source_binding.json), [17 Auszüge](source_excerpt.json)).
Klammertiefen und Blockgleichheit hat [bind_source.py](bind_source.py) programmatisch geprüft.

| Kandidat | Zeilen `mcmc_prepare.R` | Zufall | Paketsemantik | Ergebnis |
| --- | --- | --- | --- | --- |
| **Gamma-Prior-Raster** | 114–128, 526–536, 1264–1281, 1733 | nein | nein (`seq`, `/`, `sqrt`, `abs`, `which.min`) | **Nachgebildet.** |
| `which.tf.missing` | 1664–1679 | nein | nein | Mit expliziter Maske deterministisch, aber einziger Auslöser ist die offene `weightedMedian`-Ausgabe; dazu Dimensionsverlust und `taxa_change`. **Nicht gewählt**, für ein späteres Paket dokumentiert. |
| `weightedMedian` | 1631/1632, 1651/1652 | nein | ja (`matrixStats` ohne Fassung, `main_v1.2.R:27`) | Offen im Bündel Laufzeit/RNG. |
| `set.seed(2022)` + `rnorm`/`rgamma` | 1297–1303 | ja | – | Liest `shape`/`rate`; RNG-Art/R-Fassung ungebunden → nur dokumentiert. |
| Startwert | 1733–1734 | 1734 ja | – | 1733 (`rate <- shape/mean.pann`) nachgebildet; `rbeta` in 1734 nur dokumentiert. |

**Aktivierung im gebundenen Hauptlauf:** `main_v1.2.R:675` lädt `mcmc_prepare.R`.
Die Rasterblöcke 115–121 und 1270–1281 liegen auf Klammertiefe 0, also **unbedingt**.
526–531 ist der `else`-Zweig von `if(stad_interstad)` mit wörtlich `stad_interstad <- FALSE`
(Zeile 468) und damit ebenfalls aktiv. 402–408 (Birkat Ram) ist bedingt und liefert
mit denselben Eingaben denselben Wert; 213/281/369 sind auskommentiert.
Leser (nicht ausgeführt): Sampler `mcmc_conduction.R:156–170`,
Energie `mcmc_functions_v1.6.R:229–246`, Nachbearbeitung `mcmc_trimming_posterior_trafo.R:637`.

**Quellvariante:** `mcmc_prep_v1.2.R:1103` ist bis auf die Obergrenze **5000** statt 50000
gleich. Die Obergrenze ist deshalb ein **expliziter Parameter** (nur 50000 oder 5000);
die Kette über `mcmc_prepare.R` verwendet 50000, weil nur diese Datei vom Hauptlauf geladen wird.

## Implementierter Quellenumfang

Implementierung: [historical_gamma_prior_grid.py](../../../tools/historical_gamma_prior_grid.py),
Tests: [test_historical_gamma_prior_grid.py](../../../tests/test_historical_gamma_prior_grid.py).

1. `seq(0.1, upper, by = 0.1)` wie `seq.default`: `n = as.integer((to − from)/by + 1e-10)`,
   dann `from + (0:n)*by` als **zwei getrennt gerundete** Double-Operationen, dann `pmin(x, to)`.
2. `rate1 = shape/mean.pann`, `rate2 = sqrt(shape)/sd.pann`,
   Index = **erstes** `which.min(|rate2 − rate1|)`; `shape = shape[id]`, `rate = rate2[id]`.
3. Kette (`prior_gamma_chain`): Standardraster → `pann_{hist,mh,lgm,past}_{shape,rate}` (127–128)
   und `diff_rate = shape/10` (536); Verdopplung von `sd.pann` und `sd.tmp` nur bei
   `recent & min(age) >= 2000` (1264–1267, beide Bedingungen **explizite Boolwerte**, `age` wird
   nicht ausgewertet); zweites Raster (1270–1281); im Zweig `if(recent)` überschreibt 1733
   `rate` mit `shape/mean.pann`. Gemeldet wird auch die `rate`, die das geseedete `rgamma`
   in 1302 erhielte – es wird nichts gezogen.

Die geschlossene Lösung `shape = (mean/sd)²` wird nur als **Kontrolle** gemeldet, nie als Ersatz.
Domäne: endliche positive skalare Werte. **Abgelehnt statt emuliert:** Vektoren (R würde
recyceln), NA/NaN (R übersprünge sie in `which.min`), Null/negativ, andere Obergrenzen,
nicht-boolesche Bedingungen.

### Versionsabhängigkeit von `seq`

Die R-Fassung ist ungebunden. Für die Obergrenze **50000** ist `(50000 − 0,1)/0,1` exakt
`499999` und der letzte Wert exakt `50000` → Fuzz und `pmin` sind wirkungslos, alle vier
Lesarten liefern dasselbe Raster (500 000 Werte). Für die Kopie mit **5000** gilt das nicht:
ohne Fuzz 49 999 Werte (letzter `4999.900000000001`), ohne `pmin` letzter Wert
`5000.000000000001`. Eine einmal gerundete (fusionierte) Variante änderte 143 681 der
500 000 Werte, aber keine Fallentscheidung; das Modul folgt der zweistufigen R-Arithmetik.

## Unabhängige Referenz, Handrechnung und Gegenbeispiele

[Fixture-Erzeuger](make_fixture.py) mit Handerwartungen, [Fixture](synthetic_fixture.json),
[Referenz](rational_reference.py) (reine Python-Doubles für den Vollscan, ohne NumPy und ohne
Modulimport; jede Rundung um den gewählten Index und an beiden Enden zusätzlich exakt mit
`Fraction` nachgewiesen: Rasterwert, Quotient, Wurzel über Mittelpunktquadrate, Differenz),
[Referenzwerte](rational_expectations.json), [Modulergebnisse](synthetic_result.json),
[Prüfskript](check_gamma_prior_grid.py), [Arithmetik-/Gegenbeleg](arithmetic_check.json).

Handregel: `f(s) = |√s/sd − s/mean|` hat die Nullstelle `s* = (mean/sd)²`, steigt dort beidseitig
mit `1/(2·mean)`, Punkte unterhalb von `s*` sind wegen der konkaven Wurzel etwas billiger;
zwischen 0 und `s*` liegt ein Buckel mit Maximum bei `(mean/(2·sd))²`.

| Kunstfall (mean, sd, Obergrenze) | Hand | Quelle (R-Index → shape) | naive Lesart |
| --- | --- | --- | --- |
| 500, 100, 50000 | `s* = 25` | 250 → `25.000000000000004`, `rate = 0.05` | gleich; `rate1 = 0.05000000000000001` |
| 500, 200, 50000 | `s* = 6,25` genau mittig | 62 → `6.2` | nächster Punkt (auf) `6.3` |
| 315, 220, 50000 | `s* = 2,0501` | 20 → `2.0` | nächster Punkt `2.1` |
| 2500, 10, 50000 | `s* = 62500` jenseits | 1 → `0.1` (Buckel!) | Abschneiden an `50000` |
| 30, 150, 50000 | `s* = 0,04` | 1 → `0.1` | gleich |
| 708, 10, 5000 | `s* = 5012,64` knapp jenseits | 50000 → `5000` (durch `pmin`) | gleich |
| 708, 10, 50000 | wie oben | 50126 → `5012.6` | gleich |

Kette mit 500/100/`sd.tmp` 1,5: nur `recent = TRUE` **und** `min(age) ≥ 2000` verdoppeln
(→ `6.2`, `sd.tmp` 3); dann ist `rate` in 1281 `0.012449899597988732`, nach 1733 aber
`6.2/500 = 0.0124`. Ohne `recent` bleibt `rate = 0.05`, mit `recent` ohne Verdopplung
wird sie `0.05000000000000001`. `diff_rate` nutzt stets das unverdoppelte `shape`.

Alle 11 Fälle stimmen **bitgleich** mit der Referenz und mit allen acht Handerwartungen überein.
Die Gegenbeispiele sind synthetische Kontrollen, keine historische Fehlerabschätzung.

## Prüfen

```powershell
.venv/Scripts/python.exe docs/thesis_research/literal_gamma_prior_grid_2026_11_02/bind_source.py
.venv/Scripts/python.exe docs/thesis_research/literal_gamma_prior_grid_2026_11_02/make_fixture.py
.venv/Scripts/python.exe docs/thesis_research/literal_gamma_prior_grid_2026_11_02/check_gamma_prior_grid.py
.venv/Scripts/python.exe -m pytest tests/test_historical_gamma_prior_grid.py tests/test_historical_taxon_pruning.py tests/test_historical_tf_preparation.py tests/test_historical_phase1_chain.py tests/test_historical_cdf_discrepancy.py tests/test_historical_age_selection.py tests/test_historical_age_depth.py tests/test_historical_final_pdf.py tests/test_kinneret_phase1_spec.py -q --junitxml=.local/gamma_prior_grid_2026_11_02/pytest.xml
.venv/Scripts/python.exe tools/check_kinneret_phase1_spec.py
.venv/Scripts/python.exe tools/check_kinneret_phase1_spec.py --with-local
.venv/Scripts/python.exe docs/thesis_research/literal_gamma_prior_grid_2026_11_02/record_verification.py
```

`bind_source.py` braucht die privaten Quellpfade aus `.local/` und bricht ab, wenn ein Original
von seiner früheren Bindung abweicht. Ergebnis: **55 neue Tests, mit den bisherigen 297
zusammen 352 bestanden**; beide Phase-1-Prüfer weiterhin Exit 2.
[Prüfnachweis](verification.json). Nicht ausgeführt: Gesamtsuite `pytest tests`
(Sampler-Unittests), `literal_tf_preparation_2026_11_02/verify_packet.py` (kein Regressionsprüfer).

## Grenzen

Kein Originalfall, keine Originalzahlen: `prior.recent` stammt aus `fitdistr`-Anpassungen in
`prior_climate.R:128–130` (mit Untergrenzen `sd.tmp ≥ 2`, `sd.pann ≥ 150`), deren Eingaben
ungebunden sind. Keine R- oder Paketgleichheit behauptet, nur Bitgleichheit mit
IEEE-Double-Arithmetik der unabhängigen Referenz. Keine Laufbindung, keine Auswertung von
`age`. `weightedMedian`, `which.tf.missing` und alle Zufallsschritte bleiben unimplementiert.
Alle elf Nachweisbündel bleiben offen; keine Variante gewählt. Keine Trainings, Sampler,
Altersmodelle oder Fossilläufe. Frühere Pakete, Module, Fixtures und Tests sind bytegleich
zum Ausgangscommit.
