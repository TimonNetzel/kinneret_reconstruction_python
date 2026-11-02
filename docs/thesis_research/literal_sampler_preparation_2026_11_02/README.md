# Historische Sampler-Vorbereitung aus `num_taxa` und `taxa.appear.core`

Stand 02.11.2026, Ausgangscommit `0a3dcdb` (PR-Kopf, App-Stand 4), bestehender Branch
`dissertation-reconstruction`, [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).
Auftrag: [Claude-Übergabe nach den fehlenden TF](../CLAUDE_NEXT_STEP_AFTER_TF_MISSING_2026_11_02.md).
Ziel bleibt der historische Dissertationsnachbau mit Umsetzung in der App.
Dieses Paket ergänzt einen neunten begrenzten Rechenschritt: **nur die Vorbereitung, kein Samplerlauf**.
Noch kein vollständiger Originalfall.

## Welcher Schritt ist deterministisch? (Beleg zuerst)

[Determinismusprüfung](determinism_audit.json) aus dem Quelltext, ohne R-Ausführung.
Acht Originale sind per SHA-256 an frühere eingefrorene Bindungen geknüpft und werden nur gelesen:
die fünf Laufdateien sowie `check_ages.R`, `prior_climate.R` und `loess_functions.R`, die vor
`mcmc_conduction.R` gesourct werden ([Bindung](source_binding.json), [13 Auszüge](source_excerpt.json)).
[bind_source.py](bind_source.py) prüft programmatisch, mit entfernten Kommentaren und Zeichenketten
(auch mehrzeiliges Rcpp):

- Klammertiefen,
- alle Zuweisungen von `gibbs`, `F`, `T`, `num_taxa`, `taxa.appear.core` und `taxa_information`,
- dass keine verwendete Basisfunktion umdefiniert wird,
- alle Vorkommen der vorbereiteten Felder, also ihre Leser.

| Kandidat | Zeilen | Zufall | Paketsemantik | Ergebnis |
| --- | --- | --- | --- | --- |
| **Sampler-Vorbereitung** | `mcmc_conduction.R:53–79` (+`main_v1.2.R:624`) | nein | nein (`floor`, `c`, `t`, `ncol`, `nrow`, `colMeans`, `max`, `[<-`, `rep`) | **Nachgebildet** |
| `tf_x12_mean_current` | `mcmc_prepare.R:1692–1700` | nein | nein | Nur dokumentiert: Spaltenbindung, Eingaben hängen an `weightedMedian`; Leser `mcmc_conduction.R:146`, `mcmc_functions_v1.6.R:445/470/858`. |
| `combs` | `mcmc_prepare.R:1702–1703` | nein | nein | Nur dokumentiert: **kein Leser** in den acht Dateien. |
| `weightedMedian` | `mcmc_prepare.R:1631/1632/1651/1652` | nein | ja (`matrixStats` ohne Fassung) | Offen; NA-Maske bleibt Eingabe. |
| `rmvnorm_rcpp` | `mcmc_conduction.R:133` | ja | – | Nur dokumentiert. |
| `set.seed(2022)`/`rgamma`, `rbeta`, `sample` | `mcmc_prepare.R:1297–1303`, 1709–1738, 1591 | ja | – | Nur dokumentiert, kein Ersatzseed. |

**`gibbs <- F`:** Die einzige Zuweisung von `gibbs` in allen acht Dateien steht in `main_v1.2.R:624`
auf Klammertiefe 0, vor der gebundenen Schleife 651–684. `F` und `T` werden nirgends neu belegt.
Damit gilt `posterior$how_many_taxa <- num_taxa` (57) mit `parameter_stage_weights` (58).
Der Gibbs-Zweig (`floor(num_taxa/2)`, 54) wird gemeldet, ist im gebundenen Lauf aber tot.
`gibbs` in `mcmc_functions_v1.6.R:366/2051` ist jeweils ein lokaler Funktionsparameter.

**Leser** (nur dokumentiert, nichts davon ausgeführt):

- `taxa_information$values`, `ncol_taxa_array` und `nrow_taxa_array`: aktiver Leser
  `parameter_stage_weights` (`mcmc_functions_v1.6.R:1276`). Dort wird
  `matrix(values * new_weights, …, byrow = T)` gebildet.
- `new_weights` hat die Länge `ncol` und wird über den zeilenweise abgeflachten Vektor recycelt.
  Nur deshalb trifft jedes Gewicht sein Taxon.
- `taxa_information$taxa_means`: **kein Leser**.
- `posterior$jeffreys_prior`: nur `parameter_stage_gibbs` (1216), also tot.
- `posterior$how_many_taxa`: setzt den Start `z = 1:how_many_taxa` (`mcmc_conduction.R:114`).
  Die Gewichtsstufe setzt `z` in jeder Iteration auf `1:ncol` zurück (1280).
- `taxa.appear.core.temp` wird in 142 übergeben. `parameter_stage_weights` liest dieses Argument nie.

**Weitere Quellbefunde (keine Korrektur):**

- `main_v1.2.R` hat drei schließende Klammern ohne öffnende (740, 788, 829). Die zugehörigen
  Öffnungen sind auskommentiert, alles liegt nach der gebundenen Schleife.
- Ein `source()` der ganzen Datei bräche deshalb mit einem Parse-Fehler ab. Die Datei wurde also
  abschnittsweise ausgeführt; das bleibt eine Frage der Laufbindung.
- `mcmc_prepare.R:1703` bindet `c` als Schleifenvariable an eine Ganzzahl (747 nur lokal in einer
  Funktion). R überspringt bei einem Aufruf `c(...)` Bindungen, die keine Funktion sind; das ist
  dokumentierte Semantik, nicht ausgeführt.
- Wie in allen bisherigen Paketen gilt die Annahme, dass die 50 angehängten Pakete die
  Basisfunktionen nicht überdecken. Ihre Namensräume sind ungebunden.

## Implementierter Quellenumfang

Implementierung: [historical_sampler_preparation.py](../../../tools/historical_sampler_preparation.py),
Tests: [test_historical_sampler_preparation.py](../../../tests/test_historical_sampler_preparation.py).

1. **`gibbs`** ist ein expliziter Boolwert.
   - `FALSE` (gebunden): `how_many_taxa = num_taxa` (Zeile 57) und `parameter_stage_weights`.
   - `TRUE`: `floor(num_taxa/2)` (Zeile 54) und `parameter_stage_gibbs`, nur gemeldet.
2. **Zeilenweise Abflachung:** `c(t(m))` reiht Zeile an Zeile, Position `k = i·ncol + j`.
   - `ncol` und `nrow` stammen aus der Rohmatrix.
   - `values` aus Zeile 64 ist roh und wird in **Zeile 75** durch die normierte Matrix überschrieben.
3. **`colMeans(na.rm = T)`** wirkt auf die **Rohmatrix**: NA wird übersprungen, eine Spalte nur aus NA
   gibt `0/0 = NaN`.
   - Ausgegeben wird der exakte Bruch und der korrekt gerundete Double.
   - R summiert möglicherweise in erweiterter Genauigkeit (R-Fassung und Plattform ungebunden).
     Darum trägt jedes Mittel das Kennzeichen „akkumulatorunabhängig“: Alle Zwischensummen und das
     Mittel sind exakte Doubles. Ohne Leser ändert das im Lauf nichts.
4. **Spaltennormierung 70–73:** `x[, j] / max(x[, j], na.rm = T)`, je Zelle eine IEEE-Division.
   - NA bleibt NA.
   - Bei Maximum 0 gibt `0/0` NaN und `x/0` für `x < 0` `-Inf`.
   - Ein negatives Maximum kehrt das Vorzeichen um.
5. **`jeffreys_prior`** ist `rep(0.5, num_taxa)`.

**Abgelehnt statt emuliert:**

- eine Spalte nur aus NA: `max` gibt `-Inf` mit Warnung, die Spalte würde NA;
- eine Zeile oder eine Spalte: `[, which_taxa]` in `mcmc_prepare.R:1454/1666` fällt auf einen Vektor
  zurück, `ncol` wäre `NULL`;
- `num_taxa ≠ ncol`;
- unendliche Werte und negative Null: Das Vorzeichen eines Null-Maximums ist ungebunden;
- nichtnumerische, boolesche oder ungleich lange Eingaben;
- nicht-boolesches `gibbs`.

## Unabhängige Referenz, Handrechnung und Gegenbeispiele

- [Fixture-Erzeuger](make_fixture.py) mit Handerwartungen als Brüche und [Fixture](synthetic_fixture.json)
- [Referenz](rational_reference.py): Listen und `Fraction`, ohne NumPy und ohne Modulimport;
  [Referenzwerte](rational_expectations.json)
- [Modulergebnisse](synthetic_result.json), [Prüfskript](check_sampler_preparation.py),
  [Arithmetik-/Gegenbeleg](arithmetic_check.json)

| Kunstfall | Matrix | `how_many_taxa` | Besonderheit |
| --- | --- | --- | --- |
| zeilenweise Abflachung | 3×4 | 4 | `1/6`, `1/3`, `2/3`; Mittel 7/3 usw. akkumulatorabhängig |
| NA in Maximum und Mittel | 4×3 | 3 | NA bleibt NA; ohne `na.rm` wären alle Mittel NA |
| Nullspalte | 3×3 | 3 | ganze Spalte NaN, Mittel 0 |
| Maximum 0 mit Negativwert | 3×3 | 3 | NaN, `-Inf`, NA; negatives Maximum dreht das Vorzeichen |
| Gibbs-Zweig | 2×5 | **2** | `floor(5/2)`, im Lauf tot |
| Gewichtszweig | 2×5 | **5** | gebundener Zweig, sonst gleiche Werte |
| Minimalfall | 2×2 | 2 | kleinste Matrix ohne Dimensionsverlust |
| ungenaue Zwischensumme | 3×2 | 2 | `0.1 + 0.2` nicht exakt: Mittel markiert |

Alle 8 Fälle stimmen **exakt** mit der Referenz und allen Handerwartungen überein.
`matrix(values, byrow = T)` stellt die normierte Matrix in allen Fällen wieder her (Leser 1276).

Gegenbeispiele (keine Kandidatenvarianten):

- Spaltenweise Abflachung `c(m)` ordnete in 2 bis 8 Positionen je Fall das Gewicht eines fremden
  Taxons zu.
- Die Rohwerte aus Zeile 64 und ein globales Maximum weichen in allen Fällen ab.
- Die auskommentierte Alternative in Zeile 72 ergäbe für NA-Spalten NA.

Das sind synthetische Kontrollen, keine historische Fehlerabschätzung.

## Prüfen

```powershell
.venv/Scripts/python.exe docs/thesis_research/literal_sampler_preparation_2026_11_02/bind_source.py
.venv/Scripts/python.exe docs/thesis_research/literal_sampler_preparation_2026_11_02/make_fixture.py
.venv/Scripts/python.exe docs/thesis_research/literal_sampler_preparation_2026_11_02/check_sampler_preparation.py
.venv/Scripts/python.exe -m pytest tests/test_historical_sampler_preparation.py tests/test_historical_tf_missing_priors.py tests/test_historical_gamma_prior_grid.py tests/test_historical_taxon_pruning.py tests/test_historical_tf_preparation.py tests/test_historical_phase1_chain.py tests/test_historical_cdf_discrepancy.py tests/test_historical_age_selection.py tests/test_historical_age_depth.py tests/test_historical_final_pdf.py tests/test_kinneret_phase1_spec.py -q -o cache_dir=.local/sampler_preparation_2026_11_02/pytest_cache --junitxml=.local/sampler_preparation_2026_11_02/pytest.xml
.venv/Scripts/python.exe tools/check_kinneret_phase1_spec.py
.venv/Scripts/python.exe tools/check_kinneret_phase1_spec.py --with-local
.venv/Scripts/python.exe docs/thesis_research/literal_sampler_preparation_2026_11_02/record_verification.py
```

`bind_source.py` braucht die privaten Quellpfade aus `.local/` und bricht ab, wenn ein Original von
seiner früheren Bindung abweicht.

Ergebnis: **118 neue Tests, mit den bisherigen 437 zusammen 555 bestanden**. Beide Phase-1-Prüfer
geben weiterhin Exit 2. [Prüfnachweis](verification.json).

Nicht ausgeführt:

- Gesamtsuite `pytest tests` (enthält Sampler-Unittests),
- `literal_tf_preparation_2026_11_02/verify_packet.py` und die zeitpunktgebundenen `tools/verify_*`
  (keine Regressionsprüfer).

## Grenzen

- Kein Originalfall und keine Originalzahlen: `taxa.appear.core` hängt an Alters-Tiefen-Modell, Pollendaten
  und der NA-Maske aus `weightedMedian` (offen).
- Keine R- oder Paketgleichheit behauptet. `colMeans` ist nur für markierte Mittel bitgenau festgelegt.
- Keine Laufbindung.
- Sampler-Schleife, Parameterstufen, `reconstruction`, `rmvnorm_rcpp`, `tf_x12_mean_current`, `combs` und
  alle Zufallsschritte bleiben unimplementiert.
- Alle elf Nachweisbündel bleiben offen; keine Variante gewählt.
- Keine Trainings, Sampler, Altersmodelle oder Fossilläufe.
- Frühere Pakete, Module, Fixtures und Tests sind bytegleich zum Ausgangscommit.
