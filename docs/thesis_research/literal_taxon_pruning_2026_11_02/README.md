# Historisches Taxon-Pruning: Nullanteil, 15-Taxa-Regel und Abbruchsignal

Stand 02.11.2026, Ausgangscommit `47a926c`, bestehender Branch
`dissertation-reconstruction`, [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).
Ziel bleibt der historische Dissertationsnachbau mit anschließender Umsetzung
in der App. Dieses Paket ergänzt einen sechsten begrenzten Rechenschritt.
Noch kein vollständiger Originalfall.

## Welcher Schritt ist deterministisch? (Beleg zuerst)

[Determinismusprüfung](determinism_audit.json) aus dem Quelltext, ohne R-Ausführung.
Alle Quellen sind per SHA-256 an frühere eingefrorene Bindungen geknüpft und
wurden nur gelesen ([Bindung](source_binding.json), [Auszug](source_excerpt.json)).

| Kandidat | Zeilen `mcmc_prepare.R` | Zufall | Ergebnis |
| --- | --- | --- | --- |
| **Taxon-Pruning** | 51–80 | nein | Nur base R (`which`, `%in%`, `sort`, `:`, `colMeans`). **Nachgebildet.** |
| `weightedMedian` | 1631/1632, 1651/1652 | nein | Aktiv, aber nicht in den Quellen definiert; `main_v1.2.R:27` lädt `matrixStats` ohne Versionsangabe. Interpolations-, Bindungs- und Nullgewichtsregel hängen an der Paketfassung → nur dokumentiert (Bündel Laufzeit/RNG). |
| `spline(..., n=dims2)` | 1627, 1647 | nein | `tann_expand`/`pann_expand` werden nur in auskommentierten Zeilen 1637/1657 gelesen → ohne Wirkung, nur dokumentiert. |
| Erwartungswert | 1623–1624 | nein | Wird 1628/1648 auf `c()` gesetzt und durch `weightedMedian` neu gefüllt → überschrieben. |
| `sample`/`quantile` | 1591–1603 | ja, ohne Seed | Durch `range()` in 1607–1608 überschrieben → nur dokumentiert. |
| `sample` im Rezentzweig | 1399 | ja, ohne Seed | Nur dokumentiert. |
| `set.seed(2023)` + `sample`/`density`/`quantile` | 1896–1914 | ja, mit Seed | RNG-Art, `sample.kind` und R-Fassung ungebunden → keine Emulation, kein Ersatzseed. |

**Aktivierung:** Im gebundenen Rekonstruktionslauf setzt `main_v1.2.R:659`
`null.ratio.check <- FALSE` vor `source("src/mcmc_prepare.R")` (Zeile 675).
Das Pruning ist dort also **abgeschaltet**. `TRUE` steht nur in der
Pflanzenkartenschleife `main_v1.2.R:904–913` mit `mcmc_prep.R`, hinter `stop()`
in Zeile 895. Das Modul bildet einen **bedingten** Zweig nach; eine
Laufaktivierung wird nicht behauptet und nicht gewählt.

Kopien: `mcmc_prep_v1.2.R:147–176` ist zeilengleich zu 51–80; `mcmc_prep.R:144–173`
weicht nur in Zeile 57 ab (`if(ohrid)` statt `if(sitename == "Lake Ohrid")`).
Die Reliktbedingung ist deshalb ein **expliziter Boolwert** des Aufrufers.

## Implementierter Quellenumfang

Implementierung: [historical_taxon_pruning.py](../../../tools/historical_taxon_pruning.py),
Tests: [test_historical_taxon_pruning.py](../../../tests/test_historical_taxon_pruning.py).

Für eine Matrix **Tiefenproben × Taxa** (`all.avail`) mit `n` Zeilen und `T` Spalten:

1. Nullanteil `r[t] = #{i : P[i,t] == 0} / n` (Double-Division wie in R).
2. Reliktzweig (nur wenn übergeben): `r[3,8,23,24,25] = 0`, danach `r[1] = 1`.
3. `metro_hastings` wird **nur auf TRUE** gesetzt, wenn `T − #{r ≥ 0,95} > 15`.
4. Ist `metro_hastings` FALSE: entfernt werden alle Taxa, deren Wert in
   `sort(r)[16:T]` vorkommt. Dabei gilt R wörtlich: bei `T < 16` zählt `16:T`
   **abwärts**, Positionen jenseits des Endes sind `NA`, und `%in%` erweitert
   auf **alle gleichen Werte**.
5. Sonst: entfernt werden alle Taxa mit `r ≥ 0,95`.
6. Werden alle Taxa entfernt, ruft R `next()` → das Modul meldet `skip_site`.
7. Sonst Spalten in Originalreihenfolge behalten und
   `sorting.available.spectrum = colMeans(...)` bilden.

Domäne: nichtleere reelle numerische Matrix, endlich und nichtnegativ, höchstens
10⁶ Zeilen (dort ist `k/n ≥ 0.95` in Double gleich `k/n ≥ 19/20` exakt; Test
vollständig bis n = 2000 und an allen Grenzwerten bis 10⁶). Steuerwerte müssen
echte Boolwerte sein. **Nicht emuliert, sondern abgelehnt:** NA-Zählung,
Vektorverlängerung durch den Reliktzweig bei weniger als 25 Taxa und der
Dimensionsverlust auf eine einzelne Restspalte (R würde danach in `colMeans`
abbrechen). `next()` über `source()` hinweg wird nur als Signal gemeldet.
`colMeans` rechnet in R mit `long double`; Bitgleichheit wird nicht behauptet.
Zeile 83 (Null → `1e-6`) gehört zum bereits geprüften
[Alters-/Tiefenpaket](../literal_age_depth_2026_11_02/README.md).

## Unabhängige Kunstrechnung und Gegenbeispiele

[Fixture-Erzeuger](make_fixture.py) (ohne Zufall, nur aus Nullzahlen je Kunst-
taxon), [Fixture](synthetic_fixture.json), [Bruchreferenz](rational_reference.py)
(nur `Fraction` und Listen, ohne NumPy oder Modulimport),
[exakte Sollwerte](rational_expectations.json), [Modulergebnisse](synthetic_result.json),
[Prüfskript](check_taxon_pruning.py) und [Arithmetik-/Gegenbeleg](arithmetic_check.json).

| Kunstfall | Erwartung von Hand | entfernt (0-basiert) |
| --- | --- | --- |
| Flag FALSE wie im Hauptlauf | Matrix unverändert | – |
| 18 Taxa, Anteile 19/20, 1, 18/20 … | 18 − 2 > 15 → Regel ≥ 0,95; genau 0,95 fällt, 0,9 bleibt | 0, 1 |
| 17 Taxa, Anteile 1, 19/20, 1, 1 … | 17 − 4 ≤ 15; `sort(r)[16:17] = {1}` → alle drei Einsen fallen, das 0,95-Taxon **bleibt** | 0, 2, 3 |
| 6 Taxa, Anteile 0 … 0,5, 0,5 | `16:6` zählt abwärts, nur Position 6 = 0,5 → **beide 0,5-Taxa fallen** | 3, 4 |
| 4 Taxa ohne Nullen | alle gleich dem Maximum → alle fallen → `next()` | alle → `skip_site` |
| wie 6 Taxa, `metro_hastings` schon TRUE | nur ≥ 0,95 → nichts | – |
| 26 Taxa, Reliktzweig | Taxon 1 auf 1 gezwungen, 3/8/23 trotz 1,0 auf 0 | 0, 1 |

Alle Indizes und Steuerwerte stimmen exakt mit der Bruchreferenz überein;
größte Abweichung der Nullanteile und Spaltenmittel **0** (Toleranz `2e-14`).
Zwei naive Lesarten widersprechen der Quelle: „entferne alle ≥ 0,95“ in drei
Fällen und „behalte genau die 15 kleinsten Ränge“ in vier Fällen. Diese
Unterschiede sind synthetische Kontrollen, keine historische Fehlerabschätzung.

## Prüfen

```powershell
.venv/Scripts/python.exe docs/thesis_research/literal_taxon_pruning_2026_11_02/bind_source.py
.venv/Scripts/python.exe docs/thesis_research/literal_taxon_pruning_2026_11_02/make_fixture.py
.venv/Scripts/python.exe docs/thesis_research/literal_taxon_pruning_2026_11_02/check_taxon_pruning.py
.venv/Scripts/python.exe -m pytest tests/test_historical_taxon_pruning.py tests/test_historical_tf_preparation.py tests/test_historical_phase1_chain.py tests/test_historical_cdf_discrepancy.py tests/test_historical_age_selection.py tests/test_historical_age_depth.py tests/test_historical_final_pdf.py tests/test_kinneret_phase1_spec.py -q
.venv/Scripts/python.exe tools/check_kinneret_phase1_spec.py
.venv/Scripts/python.exe tools/check_kinneret_phase1_spec.py --with-local
```

`bind_source.py` braucht die privaten Quellpfade aus `.local/` und bricht ab,
wenn ein Original von seiner früheren Bindung abweicht. Ergebnis: **46 neue
Tests, mit den bisherigen 251 zusammen 297 bestanden**; beide Phase-1-Prüfer
erwarten weiterhin Exit 2. [Prüfnachweis](verification.json). Die Gesamtsuite
`pytest tests` wurde wegen enthaltener Sampler-Unittests nicht ausgeführt;
`literal_tf_preparation_2026_11_02/verify_packet.py` ist kein Regressionsprüfer
und wurde nicht ausgeführt.

## Grenzen

Kein Originalfall, keine Originalzahlen, keine R- oder Paketgleichheit, keine
Laufbindung. `weightedMedian`, Splines und alle Zufallsschritte bleiben
unimplementiert. Alle elf Nachweisbündel bleiben offen; keine Variante gewählt.
Keine Trainings, Sampler, Altersmodelle oder Fossilläufe. Frühere Pakete,
Module, Fixtures und Tests sind bytegleich zum Ausgangscommit.
