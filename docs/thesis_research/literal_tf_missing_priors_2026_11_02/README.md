# Historische Entfernung fehlender Transferfunktionen und `num_taxa`-Priors

Stand 02.11.2026, Ausgangscommit `7d3d0f6` (PR-Kopf, App-Stand 3), bestehender Branch
`dissertation-reconstruction`, [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).
Auftrag: [Claude-Übergabe nach dem Gamma-Raster](../CLAUDE_NEXT_STEP_AFTER_GAMMA_GRID_2026_11_02.md).
Ziel bleibt der historische Dissertationsnachbau mit Umsetzung in der App.
Dieses Paket ergänzt einen achten begrenzten Rechenschritt. Noch kein vollständiger Originalfall.

## Welcher Schritt ist deterministisch? (Beleg zuerst)

[Determinismusprüfung](determinism_audit.json) aus dem Quelltext, ohne R-Ausführung.
Fünf Originale, alle per SHA-256 an frühere eingefrorene Bindungen geknüpft und nur gelesen
([Bindung](source_binding.json), [18 Auszüge](source_excerpt.json)). [bind_source.py](bind_source.py)
prüft programmatisch (Kommentare und Zeichenketten entfernt): Klammertiefen, alle Zuweisungsstellen der
beteiligten Objekte und dass die Prior-Felder außerhalb ihrer Zuweisungen nirgends vorkommen.

| Kandidat | Zeilen `mcmc_prepare.R` | Zufall | Paketsemantik | Ergebnis |
| --- | --- | --- | --- | --- |
| **`which.tf.missing` + `num_taxa`-Priors** | 1664–1685 (+1435–1442) | nein | nein (`which`, `length`, negatives `[`, `nrow`, `c`, `:`, `rep`, `/`, `round`) | **Nachgebildet**, NA-Maske explizit. |
| Schwellen Mittel ± Streuung | 131–137 | nein | ja | Nicht gewählt: Periodenzweige (200–208, 268–276, 356–364) überschreiben sie aus `fitdistr`-Eingaben und `quantile`; `tann_thresh`/`pann_thresh` stammen aus geseedeten Ziehungen (1297–1303); gelesen nur über `inside.range` (`mcmc_functions_v1.6.R:319–328`), dessen Semantik ungebunden ist. |
| `weightedMedian` | 1631/1632, 1651/1652 | nein | ja (`matrixStats` ohne Fassung) | Offen; liefert im Original die NA-Maske, darum ist sie hier **Eingabe**. |
| `set.seed(2022)` + `rnorm`/`rgamma` | 1297–1303 | ja | – | Nur dokumentiert. |
| Startwerte mit `rbeta` | 1709–1738 | ja | – | Nur dokumentiert. |
| TF-Varianzstichprobe `sample` | 1591–1592 | ja | – | Nur dokumentiert (durch `range` in 1607–1608 überschrieben). |

**Aktivierung im gebundenen Hauptlauf:** `main_v1.2.R:675` lädt `mcmc_prepare.R`. 1664–1665 und
1681–1685 liegen auf Klammertiefe 0; die Entfernung 1666–1678 steht im datenabhängigen Schutz
`length(which.tf.missing) != 0`. `taxa_change` ist `FALSE` (1462) und wird nur in 1501 hinter
`if(any(...) & FALSE)` (1500) wahr – der Zweig 1671–1674 ist im gebundenen Lauf tot.

**Woher `num_taxa` kommt:** Ohne fehlende TF bleibt `num_taxa <- ncol(all.avail)` aus **1435**
(Tiefe 0). `all.avail` und `all.avail.age` werden zwischen 1435 und 1664 nicht verkleinert (Zuweisungen
nur 70/76/83/91/93/99 davor und 1676/1677 danach; `colnames` nur umbenannt in 1370/1530). Mit
Entfernung gilt `num_taxa <- nrow(tf.x1.current)` aus **1678**.

**Leser:** `num_taxa` selbst wirkt weiter (1703 `combs`, Sampler `mcmc_conduction.R:54–57, 113`,
Nachbearbeitung). `prior$phi`, `phi1`, `phi2`, `taxa.weights`, `nr.taxa.mean`, `nr.taxa.sd` kommen in den
fünf gebundenen Laufdateien außerhalb von 1437–1442/1682–1685 **nicht** vor (nur ein Kommentar in
`mcmc_functions_v1.6.R:1213`). `posterior <- list()` in **1706** verwirft `posterior$nr.taxa.mean[1]` aus 1684.
`prior$phi1`/`phi2` (1439–1440) werden nach der Entfernung **nicht** neu gesetzt und behalten den alten Wert.

## Implementierter Quellenumfang

Implementierung: [historical_tf_missing_priors.py](../../../tools/historical_tf_missing_priors.py),
Tests: [test_historical_tf_missing_priors.py](../../../tests/test_historical_tf_missing_priors.py).

1. **Maske:** `is.na(tf.x1.current.mean)` als explizite Boolliste, eine je Taxon; nie aus Werten abgeleitet.
   Nur der x1-Mittelwert zählt – ein Taxon mit NA allein im x2-Mittel bleibt.
2. **Leere Maske:** Der Schutz 1665 verhindert jede Indizierung; `num_taxa` bleibt aus 1435. Ohne Schutz wäre
   `-integer(0)` gleich `integer(0)`, und `x[integer(0)]` wählte **nichts** aus.
3. **Reihenfolge:** `which` liefert aufsteigende Positionen; negative Indizierung behält die übrigen Taxa in
   Quellreihenfolge – in Spalten von `all.avail`, `all.avail.age`, `taxa.appear.core` und Zeilen von
   `tf.x1/x2.current` sowie in beiden Mittelwertvektoren. Alle Objekte müssen dieselbe Taxonzahl haben.
4. **`taxa_change`** ist ein expliziter Boolwert; nur bei `TRUE` verlieren auch die `_change`-Mittel die
   Positionen. Die `_change`-TF-Matrizen selbst kürzt die Quelle nicht.
5. **Priors 1681–1685:** `which_taxa = 1:n`, `taxa.weights = rep(1/n, n)`, `phi = n/2`, `nr.taxa.mean = n`,
   `nr.taxa.sd = round(n/4)`. Gemeldet werden zusätzlich die veralteten `phi1 = n₁₄₃₅/2`, `phi2 = (n₁₄₃₅−2)/2`.

**`round` bei genau ,5:** `n/4` ist für ganze `n` exakt; Gleichstände gibt es genau für `n mod 4 = 2`.
Das Modul folgt der dokumentierten R-Regel für `digits = 0` (Hilfe `?Round`: IEC 60559, „go to the even
digit“) – `round(0.5) = 0`, `round(1.5) = 2`, `round(2.5) = 2`. Die R-Fassung ist ungebunden, R wurde nicht
ausgeführt. Kaufmännisches Aufrunden wiche für `n = 2, 10, 18, 26, 34 …` (`n mod 8 = 2`) um +1 ab.
Weil `prior$nr.taxa.sd` im gebundenen Lauf keinen aktiven Leser hat, ändert diese Lesart dort nichts.

**Abgelehnt statt emuliert:** ein Resttaxon (Dimensionsverlust zum Vektor, `nrow` wäre `NULL`), null
Resttaxa (`c(1:0) = c(1, 0)`), bei Entfernung eine Matrix mit Ausdehnung 1 auf der anderen Achse
(`drop`), ungleiche Taxonzahlen, nicht-boolesche Maske oder `taxa_change`, nichtnumerische Matrizen.

## Unabhängige Referenz, Handrechnung und Gegenbeispiele

[Fixture-Erzeuger](make_fixture.py) mit Handerwartungen, [Fixture](synthetic_fixture.json),
[Referenz](rational_reference.py) (Listen, Ganzzahl-`divmod` für die Rundung, `Fraction` für jedes Double;
ohne NumPy und ohne Modulimport), [Referenzwerte](rational_expectations.json),
[Modulergebnisse](synthetic_result.json), [Prüfskript](check_tf_missing_priors.py),
[Arithmetik-/Gegenbeleg](arithmetic_check.json).

| Kunstfall | Taxa | fehlend (R) | `num_taxa` (Zeile) | `round(n/4)` | Aufrunden | Besonderheit |
| --- | --- | --- | --- | --- | --- | --- |
| leere Maske | 5 | – | 5 (1435) | 1 | 1 | keine Indizierung |
| zwei entfernt | 6 | 2, 5 | 4 (1678) | 1 | 1 | A, C, D, F in Reihenfolge |
| Gleichstand n = 2 | 4 | 1, 4 | 2 | **0** | 1 | `phi1`/`phi2` bleiben 2 und 1 |
| Gleichstand n = 6 | 7 | 7 | 6 | 2 | 2 | gleich |
| Gleichstand n = 10 | 12 | 3, 9 | 10 | **2** | 3 | |
| Gleichstand n = 18 | 19 | 10 | 18 | **4** | 5 | |
| NA nur im x2-Mittel | 4 | 2 | 3 | 1 | 1 | Taxon C bleibt; „x1 oder x2“ entfernte 2, 3 |
| `taxa_change = TRUE` | 5 | 2 | 4 | 1 | 1 | `_change`-Mittel mitgekürzt; Zweig im Lauf tot |

Alle 8 Fälle stimmen **exakt** mit der Referenz und allen Handerwartungen überein (Positionen, Namen,
Werte, Doubles). Die Gegenbeispiele sind synthetische Kontrollen, keine historische Fehlerabschätzung.

## Prüfen

```powershell
.venv/Scripts/python.exe docs/thesis_research/literal_tf_missing_priors_2026_11_02/bind_source.py
.venv/Scripts/python.exe docs/thesis_research/literal_tf_missing_priors_2026_11_02/make_fixture.py
.venv/Scripts/python.exe docs/thesis_research/literal_tf_missing_priors_2026_11_02/check_tf_missing_priors.py
.venv/Scripts/python.exe -m pytest tests/test_historical_tf_missing_priors.py tests/test_historical_gamma_prior_grid.py tests/test_historical_taxon_pruning.py tests/test_historical_tf_preparation.py tests/test_historical_phase1_chain.py tests/test_historical_cdf_discrepancy.py tests/test_historical_age_selection.py tests/test_historical_age_depth.py tests/test_historical_final_pdf.py tests/test_kinneret_phase1_spec.py -q --junitxml=.local/tf_missing_priors_2026_11_02/pytest.xml
.venv/Scripts/python.exe tools/check_kinneret_phase1_spec.py
.venv/Scripts/python.exe tools/check_kinneret_phase1_spec.py --with-local
.venv/Scripts/python.exe docs/thesis_research/literal_tf_missing_priors_2026_11_02/record_verification.py
```

`bind_source.py` braucht die privaten Quellpfade aus `.local/` und bricht ab, wenn ein Original von seiner
früheren Bindung abweicht. Ergebnis: **85 neue Tests, mit den bisherigen 352 zusammen 437 bestanden**;
beide Phase-1-Prüfer weiterhin Exit 2. [Prüfnachweis](verification.json). Nicht ausgeführt: Gesamtsuite
`pytest tests` (Sampler-Unittests), `literal_tf_preparation_2026_11_02/verify_packet.py` und die
zeitpunktgebundenen `tools/verify_*` (keine Regressionsprüfer).

## Grenzen

Kein Originalfall, keine Originalzahlen: Die echte NA-Maske hängt an `weightedMedian` (offen) und an den
historischen TFs (ungebunden). Keine R- oder Paketgleichheit behauptet; die Rundungsregel ist die
dokumentierte, nicht eine ausgeführte. Keine Laufbindung. `weightedMedian`, Spline-Erweiterung,
`tf_x12_mean_current`, `combs` und alle Zufallsschritte bleiben unimplementiert. Alle elf Nachweisbündel
bleiben offen; keine Variante gewählt. Keine Trainings, Sampler, Altersmodelle oder Fossilläufe. Frühere
Pakete, Module, Fixtures und Tests sind bytegleich zum Ausgangscommit.
