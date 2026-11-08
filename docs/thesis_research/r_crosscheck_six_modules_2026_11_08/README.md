# R-3.5.0-Gegenprobe der sechs älteren Module

Stand 08.11.2026, Ausgangscommit `416d547` (PR-Kopf), bestehender Branch `dissertation-reconstruction`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).
Auftrag: Nutzerentscheidung „Ja, weiter prüfen“ nach der
[Übergabe nach dem gewichteten Median](../CLAUDE_NEXT_STEP_AFTER_WEIGHTED_MEDIAN_2026_11_08.md).
**Nur Gegenprobe auf Kunsteingaben; kein neues Rechenmodul, kein Originalfall.** Die sechs Module sind unverändert.

## Ergebnis

R 3.5.0 wertet die **Originalzeilen wörtlich** aus (Hashes der Originale geprüft; Python-Module und R laufen auf denselben
C99-Hex-Doubles). Es gibt zwei Eingabearten: die runden Fixtures der früheren Pakete und je 25 neue Stressfälle mit
ungenauen Doubles ([Erzeuger](make_stress.py), [Eingaben](stress_fixture.json)).

| Modul | Originalzeilen | Werte ohne Altersraster | bitgleich | Fixtures bitgleich | höchstens |
| --- | --- | ---: | ---: | ---: | ---: |
| Alters-/Tiefenübertragung | `check_ages.R:177–179, 230–232`; `mcmc_prepare.R:11–20, 83, 91–94` | 4.827 | 4.358 | 58/58 | 3 ulp |
| Altersraster und Zeilenauswahl | `mcmc_prepare.R:435–442, 1462–1467, 1522`; `mcmc_functions_v1.6.R:461–462, 470` | 6.535 | 5.303 | 5.222/5.222 | 2 ulp |
| Taxongewichtung und PDF-Summe | `mcmc_functions_v1.6.R:375, 411–424, 434–462, 543–547, 712–715, 756–760` | 15.407 | 10.807 | 154/154 | 4 ulp |
| Vergleich kumulierter Rohdichten | `mcmc_functions_v1.6.R:369–371, 485–490` | 2.827 | 2.797 | 78/78 | 2 ulp |
| TF-Normierung und Klimazellmaske | `mcmc_prepare.R:1581–1586, 1613–1618` | 2.580 | 2.000 | 58/58 | 3 ulp |
| Taxon-Pruning | `mcmc_prepare.R:51–80` | 9.790 | 9.653 | 1.494/1.494 | 3 ulp |
| **Summe** | | **41.966** | **34.918** | **7.064/7.064** | **4 ulp** |

Die Altersraster selbst (rund zwei Millionen Werte aus `seq`) sind ebenfalls alle bitgleich und deshalb hier
herausgerechnet, sie würden den Anteil sonst schönen. Ergebnis: [six_crosscheck_result.json](six_crosscheck_result.json).

- **Alle Fixture-Werte sind bitgleich**, ebenso alle NA-/NaN-/Inf-Stellen und Vorzeichen der Null (keine Abweichung).
- 7.048 Stresswerte weichen um 1–4 ulp ab. Nur **Summen und Matrixprodukte** sind betroffen:
  Normierung über `rowSums`/`sum`, `colMeans` und `%*%`.
- Masken, Raster, Indizes, Pruning-Entscheidungen, Spaltenmaxima und `rowCumsums` sind überall bitgleich.

## Ursache der Abweichungen, belegt

[explain_differences.py](explain_differences.py) rechnet die 24.822 Werte der betroffenen Schlüssel so nach, wie R 3.5.0
rechnet (`rowSums`, `sum`, `colMeans` je Addition in 80-Bit-`long double`; `%*%` als einfache Double-Schleife der
Referenz-BLAS). **Alle 24.822 stimmen bitgenau mit R überein** ([Nachweis](difference_causes.json)). Das Modul liegt dort bei
17.774.

Die Module summieren in Double und multiplizieren über die BLAS von NumPy. Das ist kein fachlicher Fehler; die früheren
Pakete prüften gegen eine Toleranz von `1e-14`. Es ist aber **nicht bitgleich zu R** und bleibt so markiert. Eine bitgleiche
Variante bräuchte eine R-nahe Summen- und Produktschicht; die gibt es nicht und wurde hier nicht eingeführt.

## Paketanbieter

[bind_provider.py](bind_provider.py) → [provider_rowcumsums.json](provider_rowcumsums.json), nur aus dem CRAN-Zwischenspeicher
(zusätzlich die sechs `proxy`-Tarballs geladen, im Speicher gelesen).

- `rowCumsums` (`mcmc_functions_v1.6.R:485–486`): im Fenster nur von **matrixStats** exportiert, davor auf dem Suchpfad
  niemand.
- Die Rechenanweisungen des C-Kerns sind in allen 13 Fassungen ein Text. Der Kern nutzt `long double`.
- Quellbefund: `rowCumsums(…, na.rm=T)` — `na.rm` ist kein Argument der Funktion und wird von `...` verschluckt.
  Fehlwerte werden also nicht übersprungen. Das Original ist nicht korrigiert.
- Base-Namen in den sechs Zeilenbereichen: Einzig `as.matrix` wird von einem angehängten Paket neu exportiert (`proxy`), in
  allen sechs Fassungen als `function(x, ...) base::as.matrix(x, ...)`, also wirkungsgleich.
- Nicht aufgelöst, alle hinter matrixStats: einzelne Fassungen von `verification`, `CircStats`, `smotefamily`, `fields`, `snow`
  mit `exportPattern`. Für `rowCumsums` ohne Belang. Base-Masken durch diese Pakete sind damit nicht ausgeschlossen.

`inside.range` (Zeilen 1613–1618) ist im Lauf durch eine explizite Kunstmaske ersetzt, die Bereichsprüfung selbst ist hier
nicht geprüft. Zeile 434 von `final_reconstruction` nutzt im Lauf `floor(num_taxa/2)` Spalten; die Gegenprobe legt das
Zielobjekt in der Länge der Kunst-IDs an.

## Prüfen

```powershell
.venv/Scripts/python.exe -I docs/thesis_research/r_crosscheck_six_modules_2026_11_08/bind_provider.py .local/evidence_r_env_2026_11_03/cran_cache .local/evidence_r_env_2026_11_03/cran_exports.json
.venv/Scripts/python.exe docs/thesis_research/r_crosscheck_six_modules_2026_11_08/make_stress.py
.venv/Scripts/python.exe docs/thesis_research/r_crosscheck_six_modules_2026_11_08/run_six_crosscheck.py .local/R/R-3.5.0/bin/x64/Rscript.exe .local/R/library_ms
.venv/Scripts/python.exe docs/thesis_research/r_crosscheck_six_modules_2026_11_08/explain_differences.py .local/r_crosscheck_six_modules_2026_11_08/output.tsv
.venv/Scripts/python.exe -m pytest tests/test_r_crosscheck_six_modules.py <die fünfzehn bisherigen Testdateien> -q
```

**Ergebnis:** siehe [Prüfnachweis](verification.json). Nicht ausgeführt: die Gesamtsuite, `verify_packet.py` und
`tools/verify_*`; Sampler, Ziehungen, Trainings, Altersmodelle, Fossilrekonstruktionen.

## Grenzen

- Kein Originalfall, keine Originalzahlen. Die Stresszahlen sind erfunden.
- R-Gleichheit gilt für R 3.5.0, Windows x86_64 mit `long double` = 16 Byte. Ein Rechner mit anderer Arithmetik oder einer
  anderen BLAS könnte historisch andere Bits geliefert haben.
- Für `rowCumsums` lief nur matrixStats 0.56.0.
- Alle elf Nachweisbündel bleiben offen; keine Variante gewählt. Die sechs Module und ihre Pakete sind bytegleich.
