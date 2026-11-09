# Deterministischer Kern der Gewichtsstufe `parameter_stage_weights`

Stand 03.11.2026, Ausgangscommit `235f797` (PR-Kopf, App-Stand 5), bestehender Branch
`dissertation-reconstruction`, [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).
Auftrag: [Claude-Übergabe nach der Sampler-Vorbereitung](../CLAUDE_NEXT_STEP_AFTER_SAMPLER_PREP_2026_11_02.md).
Ziel bleibt der historische Dissertationsnachbau mit Umsetzung in der App.
Dieses Paket ergänzt einen zehnten begrenzten Rechenschritt: **nur den deterministischen Kern mit expliziten
Gewichten, keine Dirichlet-Ziehung und kein Samplerlauf**. Noch kein vollständiger Originalfall.

## Welcher Schritt ist deterministisch? (Beleg zuerst)

[Determinismusprüfung](determinism_audit.json) aus dem Quelltext, ohne R-Ausführung.
Dieselben acht Originale wie im Vorpaket sind per SHA-256 an dessen eingefrorene
[Bindung](../literal_sampler_preparation_2026_11_02/source_binding.json) geknüpft und werden nur gelesen
([Bindung](source_binding.json), [15 Auszüge](source_excerpt.json)).
[bind_source.py](bind_source.py) prüft programmatisch, mit entfernten Kommentaren und Zeichenketten:

- Klammertiefen (1259, 1276, 1277 und 1280 stehen auf Tiefe 1 des Funktionsrumpfs, also unbedingt nach der
  `while`-Schleife);
- alle Zuweisungen und Vorkommen von `parameter_stage_weights`, `parameter_stage`, `rdirichlet`, `ddirichlet`,
  `new_weights`, `dirichlet_spread`, `$taxa.appear.core.temp`, `$z` und `$taxa_weights`;
- dass `matrix`, `rowSums`, `ncol`, `nrow`, `min`, `rep` und `is.na` nirgends umdefiniert werden;
- im **Rohtext einschließlich der Rcpp-Zeichenketten**, dass `rdirichlet`/`ddirichlet` nirgends definiert werden;
- alle `library()`-Aufrufe in Laufreihenfolge.

| Kandidat | Zeilen | Zufall | Paketsemantik | Ergebnis |
| --- | --- | --- | --- | --- |
| **Kern der Gewichtsstufe** | `mcmc_functions_v1.6.R:1259, 1276–1280` | nein | nein (`*`, `matrix`, `rowSums`, `/`, `ncol`, `:`) | **Nachgebildet**, Gewichte explizit |
| `rdirichlet` | 1246–1251 | ja | ja (Anbieter ungebunden) | Nur dokumentiert |
| Rücksetzung nach 1000 Versuchen | 1253–1257 | folgt aus 1248 | ja | Nur dokumentiert; ihr Wert `rep(1/num_taxa, num_taxa)` ist als explizite Eingabe prüfbar |
| `ddirichlet`, `parameter_ratio`, `is.na` | 1269–1274 | nein | ja | Nur dokumentiert |
| Sampler-Initialisierung | `mcmc_conduction.R:109–135` ohne 118/133 | nein | nein | Alternative für ein späteres Paket |
| `tf_x12_mean_current` | `mcmc_prepare.R:1692–1700` | nein | nein | Nur dokumentiert, Eingaben hängen an `weightedMedian` |
| `weightedMedian` | `mcmc_prepare.R:1631/1632/1651/1652` | nein | ja (`matrixStats` ohne Fassung) | Offen |
| Akzeptanz `runif`, `rmvnorm_rcpp`, `set.seed(2022)`/`rgamma`, `rbeta`, `sample` | `mcmc_conduction.R:304/133`, `mcmc_prepare.R:1297–1303/1709–1738/1591` | ja | – | Nur dokumentiert, kein Ersatzseed |

**Kontrollfluss:** `parameter_stage_weights` ist einmal definiert (1242). `gibbs <- F` (`main_v1.2.R:624`, einzige
Zuweisung) wählt sie in `mcmc_conduction.R:58`; aufgerufen wird sie in jeder Iteration über `parameter_stage(...)`
(142, einziger Aufruf). Startgewichte: `rep(1/num_taxa, num_taxa)` (113), Startstreuung
`dirichlet_spread <- num_taxa^2` (`mcmc_prepare.R:1867`). Die Stufe liest `num_taxa` als **globale** Variable
(1254, 1269) und ihr Argument `taxa.appear.core` nie.

**Leser der Ergebnisse** (nur dokumentiert, nichts ausgeführt):

- `posterior_temp$taxa.appear.core.temp`: `reconstruction` (letzte Definition 1781, ältere 855) liest es in 1817,
  `reconst <- taxa.appear.core[reconst_ages,] %*% posterior_temp$tf_sample` (1818).
- `posterior_temp$z`: Zeilenauswahl `tf.x1.current[z,]`, `tf.x2.current[z,]`, `tf_x12_mean_current[z,]` in
  `mcmc_conduction.R:146`; gespeichert 306/355.
- `posterior_temp$taxa_weights`: Akzeptanz 305/354; bei Ablehnung stellt 329/330 Gewichte und `z` zurück, nicht aber
  `taxa.appear.core.temp`. Das wird in der nächsten Iteration in 1277 überschrieben, bevor 1817 es liest.

**Wer liefert `rdirichlet`/`ddirichlet`? (belegt, nicht geraten)**

- Beide Aufrufe sind unqualifiziert (1248, 1269, 1270). In den acht gebundenen Dateien gibt es keine Definition,
  auch nicht in eingebettetem Rcpp-Code oder per `assign`.
- Gebunden ist nur die **Reihenfolge** der `library()`-Aufrufe: 52 in `main_v1.2.R:4–62` vor der Schleife, dazu
  `npreg` und `truncnorm` (`mcmc_prepare.R:1885–1886`) und `colorRamps` (`mcmc_trimming_posterior_trafo.R:416`,
  nach dem ersten Samplerlauf) innerhalb der gebundenen Schleife ([Liste](source_binding.json)).
- R sucht im zuletzt angehängten Paket zuerst. **Welche** dieser Pakete `rdirichlet` bzw. `ddirichlet` exportieren,
  zeigen die gebundenen Quellen nicht: Es gibt keine Paketfassung und keine Namensraumdatei, und lokal ist
  keine R-Installation vorhanden (weder im `PATH` noch unter `C:\Program Files`). Der Anbieter bleibt **ungebunden**, kein Paket wird gewählt.
- `SBMSplitMerge::rdirichlet` (1216) ist qualifiziert und steht in der toten Gibbs-Stufe. Es belegt nichts für 1248.
- `rdirichlet(...)[1,]` setzt zudem eine Matrix als Rückgabe voraus; auch das ist Paketsemantik.

**Quellbefunde ohne Korrektur:**

- Frühere Pakete sprechen von „50 angehängten Paketen“. Programmatisch gezählt sind es 52 `library()`-Aufrufe vor der
  Schleife und drei weitere darin. Die früheren Angaben bleiben unverändert; ihre Annahme (keine Überdeckung der
  Basisfunktionen) gilt hier gleichermaßen für alle 55 Pakete.
- Der Vorschlagsquotient 1270 enthält `dirichlet_spread` nicht (bereits in
  [LOCAL_SOURCE_CROSSWALK.md](../LOCAL_SOURCE_CROSSWALK.md) festgehalten).
- `main_v1.2.R` hat weiter überzählige `}` in 740, 788 und 829.

## Implementierter Quellenumfang

Implementierung: [historical_weights_stage.py](../../../tools/historical_weights_stage.py),
Tests: [test_historical_weights_stage.py](../../../tests/test_historical_weights_stage.py).
Eingaben in der Darstellung des Vorpakets: `None` ist R-`NA`, `nan` ist `NaN`, `-inf` ist `-Inf`.

1. **1259:** `taxa_weights` übernimmt die expliziten Gewichte unverändert, ohne Renormierung.
2. **Recycling 1276:** `values` hat die Länge `nrow·ncol`, die Gewichte die Länge `ncol`. Position `k` bekommt
   Gewicht `k mod ncol`. Weil Stand 5 zeilenweise abflacht, trifft jedes Gewicht sein eigenes Taxon.
3. **`matrix(..., byrow = T)`** füllt die Produkte Zeile für Zeile zurück.
4. **`rowSums(na.rm = T)` 1277** überspringt NA **und** NaN. Eine Zeile ohne Werte ergibt 0, eine Zeile mit `-Inf`
   ergibt `-Inf`.
   - Ausgegeben wird der korrekt gerundete exakte Bruch.
   - R summiert möglicherweise in erweiterter Genauigkeit (R-Fassung und Plattform ungebunden). Jede Zeile trägt
     darum ein Kennzeichen „akkumulatorunabhängig“. Es ist eine **hinreichende** Bedingung: Alle Werte sind
     Vielfache einer gemeinsamen Zweierpotenz `2^e`, und ihre Betragssumme bleibt unter `2^(e+53)`. Dann ist jede
     Teilsumme in **jeder** Reihenfolge ein exakter Double. Die Referenz prüft dasselbe Ziel unabhängig durch
     Aufzählen aller Teilsummen.
5. **Division 1277:** Matrix durch Vektor recycelt den Vektor spaltenweise. Zelle `(i, j)` wird also durch die
   Summe **ihrer eigenen Zeile** geteilt, je Zelle eine IEEE-Division.
   - NA bleibt NA, NaN bleibt NaN.
   - Zeilensumme 0: `0/0` gibt NaN, `x/0` gibt `±Inf`.
   - Zeilensumme `-Inf`: `-Inf/-Inf` gibt NaN, endliche Werte ergeben eine vorzeichenbehaftete Null.
6. **`z <- 1:ncol` 1280.**

**Abgelehnt statt emuliert:**

- `+Inf` und negative Null in `values` (von Stand 5 nicht erzeugt);
- Länge ungleich `nrow·ncol` (`matrix` würde recyceln oder kürzen) und weniger als zwei Zeilen oder Spalten;
- Gewichte, die nicht genau ein endlicher positiver Double je Spalte sind (die Schleife 1249 endet nur mit
  `min(new_weights) > 0`);
- Produkte oder Quotienten mit Über- oder Unterlauf auf null;
- nichtnumerische, boolesche oder verschachtelte Eingaben.

## Unabhängige Referenz, Handrechnung und Gegenbeispiele

- [Fixture-Erzeuger](make_fixture.py) mit Handerwartungen als Brüche und [Fixture](synthetic_fixture.json)
- [Referenz](rational_reference.py): Listen und `Fraction`, ohne NumPy und ohne Modulimport;
  [Referenzwerte](rational_expectations.json)
- [Modulergebnisse](synthetic_result.json), [Prüfskript](check_weights_stage.py),
  [Arithmetik-/Gegenbeleg](arithmetic_check.json)

| Kunstfall | Matrix | Besonderheit |
| --- | --- | --- |
| zeilenweises Recycling | 2×3 | Zeilensummen 11/16 und 3/4; Ergebnis 8/11, 2/11, 1/11 und dreimal 1/3 |
| Kette Minimalfall | 2×2 | Werte aus Stand 5; beide Zeilen 1/4 und 3/4 |
| Kette 3×4 ungenau | 3×4 | Werte 1/6, 1/3, 2/3 aus Stand 5: Zeilen 1–2 markiert, Zeile 3 exakt 1/29, 12/29, 8/29, 8/29 |
| Kette Nullspalte | 3×3 | NaN-Spalte aus Stand 5 wird übersprungen und bleibt NaN |
| NA und NaN übersprungen | 3×3 | je zwei Werte pro Zeile, Ergebnis 1/2, 2/3, 1/3, 4/5, 1/5 |
| Zeilensumme 0 | 4×3 | Nullzeile, Zeile ohne Werte, sich aufhebende Zeile: NaN, `Inf`, `-Inf`, NA |
| Kette `-Inf` | 3×3 | Zeilensumme `-Inf`: NaN und `-0`; Zeile 3 markiert |
| Rücksetzgewichte `rep(1/3, 3)` | 2×3 | `3·d = 1 − 2⁻⁵⁴` rundet zur geraden 1; markiert |
| Gewichte ×2 | 2×3 | Summe der Gewichte 2: gleiches Ergebnis wie der erste Fall |

Alle 9 Fälle stimmen **exakt** (einschließlich Vorzeichen der Null) mit der Referenz und allen Handerwartungen
überein.

- **Kette:** In den vier Kettenfällen liefert das Modul aus Stand 5 genau die Fixture-Werte.
- **Gegenprobe mit [`rw_coefficients`](../literal_final_pdf_2026_11_02/README.md):** Das ist dasselbe Muster in
  `final_reconstruction` 450–455. Es stimmt in allen exakten Zeilen der beiden endlichen positiven Kettenfälle
  überein; die markierten Zeilen werden nur berichtet.
- **Zweierpotenzen:** Ein Faktor 2, 1/4 oder 1024 auf allen Gewichten ändert das Ergebnis nicht.
- **Gleichverteilte Gewichte:** `rep(1/3, 3)` weicht in Zeile 1 des Rücksetzfalls um eine Einheit der letzten Stelle
  von der ungewichteten Zeilennormierung ab. Diese Zeile ist markiert, R kann dort zusätzlich abweichen.

Gegenbeispiele (keine Kandidatenvarianten), in **allen** 9 Fällen abweichend:

- spaltenweise abgeflachte Werte `c(.)` mit demselben Recycling (2 bis 8 fremd gewichtete Positionen je Fall);
- `matrix(byrow = F)`;
- Zeilensummenvektor entlang der Zeilen recycelt;
- Spaltensummen statt Zeilensummen;
- ohne `na.rm` gingen in vier Fällen Zeilen mit NA oder NaN ganz verloren.

Das sind synthetische Kontrollen, keine historische Fehlerabschätzung.

## Prüfen

```powershell
.venv/Scripts/python.exe docs/thesis_research/literal_weights_stage_2026_11_03/bind_source.py
.venv/Scripts/python.exe docs/thesis_research/literal_weights_stage_2026_11_03/make_fixture.py
.venv/Scripts/python.exe docs/thesis_research/literal_weights_stage_2026_11_03/check_weights_stage.py
.venv/Scripts/python.exe -m pytest tests/test_historical_weights_stage.py tests/test_historical_sampler_preparation.py tests/test_historical_tf_missing_priors.py tests/test_historical_gamma_prior_grid.py tests/test_historical_taxon_pruning.py tests/test_historical_tf_preparation.py tests/test_historical_phase1_chain.py tests/test_historical_cdf_discrepancy.py tests/test_historical_age_selection.py tests/test_historical_age_depth.py tests/test_historical_final_pdf.py tests/test_kinneret_phase1_spec.py -q -o cache_dir=.local/weights_stage_2026_11_03/pytest_cache --junitxml=.local/weights_stage_2026_11_03/pytest.xml
.venv/Scripts/python.exe tools/check_kinneret_phase1_spec.py
.venv/Scripts/python.exe tools/check_kinneret_phase1_spec.py --with-local
.venv/Scripts/python.exe docs/thesis_research/literal_weights_stage_2026_11_03/record_verification.py
```

`bind_source.py` braucht die privaten Quellpfade aus `.local/` und bricht ab, wenn ein Original von seiner früheren
Bindung abweicht.

Ergebnis: **134 neue Tests, mit den bisherigen 555 zusammen 689 bestanden**. Beide Phase-1-Prüfer geben
weiterhin Exit 2. [Prüfnachweis](verification.json).

Nicht ausgeführt:

- Gesamtsuite `pytest tests` (enthält Sampler-Unittests),
- `literal_tf_preparation_2026_11_02/verify_packet.py` und die zeitpunktgebundenen `tools/verify_*`
  (keine Regressionsprüfer),
- R, `rdirichlet`/`ddirichlet`, Trainings, Sampler, Altersmodelle, Fossilrekonstruktionen.

## Grenzen

- Kein Originalfall und keine Originalzahlen: `taxa_information` hängt an Alters-Tiefen-Modell, Pollendaten und der
  NA-Maske aus `weightedMedian` (offen); die echten Gewichte stammen aus `rdirichlet` (Anbieter ungebunden).
- Keine R- oder Paketgleichheit behauptet. `rowSums` ist nur für markierte Zeilen bitgenau festgelegt.
- Keine Laufbindung.
- Dirichlet-Ziehung, Rücksetzung, `ddirichlet`-Quotient, Akzeptanz, `reconstruction`, Sampler-Initialisierung und alle
  Zufallsschritte bleiben unimplementiert.
- Alle elf Nachweisbündel bleiben offen; keine Variante gewählt.
- Frühere Pakete, Module, Fixtures und Tests sind bytegleich zum Ausgangscommit.
