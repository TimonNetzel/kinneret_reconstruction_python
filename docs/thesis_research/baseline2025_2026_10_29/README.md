# Moderne Shared37-Baseline: ausgeführt und geprüft

Stand 29.10.2026. Die vorausgehenden PRs #15, #16 und #17 sind nach `main`
übernommen; Ausgangscommit dieser Arbeit ist `e002f2b`.
Die moderne Vergleichsrechnung und ihre Ergebnisprüfung sind hier ausgeführt.
Claude muss diese Berechnung nicht erneut entwickeln. Der verbleibende
[Integrationsauftrag](./CLAUDE_TASK.md) ist entsprechend begrenzt.

## Was wurde festgelegt und gerechnet?

Die Konfiguration und der Evaluator wurden in Commit
[`e594aa9`](https://github.com/TimonNetzel/kinneret_reconstruction_python/commit/e594aa9dc98bda13f452fbe55a48aa468c560b67)
**vor dem ersten Ergebnislauf** festgehalten. Der festgelegte Hauptvergleich
ist `analog_k5`; k=1 und k=15 sind Sensitivitäten, keine Suche nach einem
nachträglich ausgewählten Gewinner.

- 7.634 moderne Proben aus der separaten publizierten 2025-Kohorte; 7.078
  erreichen mindestens 100 gemeinsame Zählungen ohne Pinus. 556 werden sichtbar
  ausgeschlossen. Keine zusätzlichen Klima-, Standort- oder Geographiefilter.
- Alle 37 Merkmale und die Nennerregeln aus dem
  [vorherigen Datenvertrag](../modern2025_2026_10_29/README.md) bleiben erhalten.
  Aus tatsächlichen Zählungen entsteht `sqrt(count / shared_count_sum)`.
  Der Hellinger-Abstand ist der euklidische Abstand dieser Wurzeln geteilt
  durch `sqrt(2)`. Keine Merkmalsauswahl oder Zielwertskalierung.
- Je fünf Testfolds für ein 5°- und ein 10°-Gitter, Ursprung 90°S/180°W.
  Sortierte belegte Blöcke werden mit PCG64/Seed 20261029 permutiert und
  abwechselnd auf die fünf Folds verteilt. Das verteilt Blöcke, nicht zwingend
  gleich viele Proben. Ein Fold kann mehrere getrennte Regionen enthalten.
- Gleiche gerundete Koordinaten bleiben zusammen. +180° und −180° werden für
  diese Gruppierung gleichgesetzt. Training innerhalb von 100 km um irgendeine
  Testprobe wird entfernt, auch über die Datumsgrenze hinweg. Verwendet wird
  ein sphärischer Großkreisabstand mit Radius 6.371,0088 km.
- Vorhersage: ungewichtetes Klimamittel der k nächsten **Pollen**-Analogproben.
  Koordinaten gehen nur in die Validierung ein. Bei exakt gleichen berechneten
  Abständen entscheidet die sortierte Proben-ID. Das Referenzmodell verwendet
  das Klimamittel desselben gepufferten Trainingsfolds, keine Testzielwerte.
- Keine Parameteranpassung auf Testdaten, keine fossilen Vorhersagen und keine
  Anpassung der bestehenden Kinneret-Kalibrierung.

5°/10° und 100 km sind ausdrücklich gewählte erste Designs. Es wurde keine
optimale Autokorrelationslänge geschätzt. Das Gradgitter hat je nach Breite
unterschiedliche physische Blockgrößen; der zusätzliche Puffer ist in km.

## Hauptbefund

Out-of-fold-Fehler über alle 7.078 zurückgehaltenen Proben, probengewichtet:

| Gitter | Modell | Tann RMSE, °C | Pann RMSE, mm/Jahr | Tann MSE-Skill | Pann MSE-Skill |
|---|---|---:|---:|---:|---:|
| 5° | Trainingsmittel | 7,411 | 423,6 | 0 | 0 |
| 5° | **5 Pollenanaloge** | **3,765** | **365,2** | **0,742** | **0,257** |
| 10° | Trainingsmittel | 7,514 | 417,2 | 0 | 0 |
| 10° | **5 Pollenanaloge** | **3,926** | **373,1** | **0,727** | **0,200** |

`MSE-Skill = 1 − MSE(Modell) / MSE(Trainingsmittel)`; dies ist kein neu
geschätztes R² und keine Wahrscheinlichkeit. Positiv bedeutet geringeren
quadratischen Vorhersagefehler als beim angegebenen Referenzmodell.

Für k=5 liegt der Temperatur-Bias bei −0,184/−0,304 °C und der Niederschlags-Bias
bei −35,4/−35,5 mm/Jahr (5°/10°; Vorhersage minus Beobachtung). Der deutliche
Gewinn bei Tann und der kleinere Gewinn bei Pann sind Befunde dieser modernen
Kohorte. Der Niederschlagsfehler bleibt groß.

Wird jede der 5.744 Koordinatengruppen gleich gewichtet, erhält k=5 einen
Tann-RMSE von 3,771/3,967 °C und Pann-RMSE von 343,0/351,5 mm/Jahr.
Die vollständigen RMSE-, MAE-, Bias- und Skill-Werte für beide Gewichtungen
stehen in `metrics_summary.csv`; foldweise Werte in `metrics.json`.
Die Trainingsmodelle verwenden weiterhin einzelne Proben: mehrfach beprobte
Orte können mehrere Analognachbarn stellen. Die ausgeglichene
Ergebnisgewichtung beseitigt diese Eigenschaft des Trainings nicht.

Die k-Sensitivität bleibt sichtbar: k=15 ergibt 3,626/3,815 °C und
358,3/366,5 mm/Jahr; k=1 ergibt 4,649/4,820 °C und 422,9/438,5 mm/Jahr.
Bei 10° ist die k=1-Niederschlagsvorhersage schlechter als das Trainingsmittel.
Das vorab festgelegte k=5 wird wegen dieser Ergebnisse nicht nachträglich
durch k=15 ersetzt. Eine spätere optimierte Modellwahl braucht eigene
Trainings-/Validierungsregeln, statt diese Testzahlen nochmals zu optimieren.

## Bedeutung für Europa und Grenzen

Die Quellkohorte ist eurasisch und umfasst auch Standorte an der Datumsgrenze;
das Gesamtergebnis ist keine ausschließlich europäische Güteangabe. Für das
bereits in der Datenprüfung verwendete Fenster 25–75°N, 15°W–45°E wurde deshalb
**nach dem Hauptlauf eine beschreibende Teilansicht** derselben Vorhersagen
berechnet: 5.223 Proben, ohne neues Training. Dies ist kein zusätzlicher
unabhängiger Test und war nicht Teil der vorab festgelegten Hauptauswertung.

| Fenster, probengewichtet | Tann RMSE, °C | Pann RMSE, mm/Jahr |
|---|---:|---:|
| 5°: Trainingsmittel → k=5 | 5,822 → 3,335 | 438,8 → 398,5 |
| 10°: Trainingsmittel → k=5 | 5,963 → 3,562 | 431,9 → 408,4 |

Insbesondere Pann sollte daher als begrenzte Baseline dargestellt werden.
Es gibt noch keine historische oder fossile externe Validierung, keine
Transferfunktion der ursprünglichen 279 Taxa und keine belastbaren
Unsicherheitsintervalle für Monticchio. Die 69 fossilen Proben wurden nicht
zur Modellwahl verwendet und nicht in Klimakurven umgerechnet.
Auch Taxonvergröberung, Teilzusammensetzung, Probenkontexte, Quellklima und
deren nicht unabhängig festgestellter Bezugszeitraum begrenzen die Übertragung.
Ein räumlicher Test innerhalb der modernen Quelle prüft keinen Datenbankwechsel,
keinen Vegetationswandel über 150 ka und keine Altersunsicherheit.

## Unabhängige Prüfung

`audit_baseline_shared2025.py` importiert den Evaluator nicht. Es liest dessen
CSV-Ausgaben und prüft mit einer separaten Haversine-Berechnung alle zehn
Fold-Aufteilungen und alle Trainings-/Pufferrollen. Der kleinste berechnete
sphärische Trainings-Test-Abstand ist 100,038 km; keine Koordinatengruppe oder Gitterzelle
liegt zugleich in Training und Test eines Folds.

Alle 56.624 Vorhersagezeilen sind eindeutig und vollständig. Die Zielwerte,
Trainingsmittel sowie sämtliche aggregierten Fehler beider Gewichtungen wurden
nachgerechnet. Für die vorab im Audit festgelegten ersten/mittleren/letzten
Testproben je Fold wurden 90 Analogvorhersagen zusätzlich mit direkter
NumPy-Distanzrechnung geprüft. Das sind Stichproben der Analogvorhersagen,
keine Behauptung, sämtliche Nachbarrangfolgen unabhängig neu berechnet zu haben.
Die regionalen Zahlen sind als nachgelagerte Diagnose im Prüfbericht markiert.

Zwei vollständige Evaluatorläufe waren in dieser Umgebung bytegleich,
einschließlich Manifest. Sechs neue Tests prüfen Transformation, ungültige
Zählungen, Distanzgleichstände, Datumsgrenze, Gruppentrennung und Ergebnisgewichte.
`verification.json` hält Befunde, Skript-Hashes und den Designcommit fest.
Die verwendete Umgebung war Python 3.12.14, NumPy 2.3.5 und SciPy 1.17.0.
Andere Bibliotheksstände können letzte Dezimalstellen bzw. numerische
Distanzgleichstände verändern; die Umgebung ist deshalb im Manifest enthalten.

## Reproduktion und Dateien

Zunächst die Shared37-Tabellen gemäß dem vorigen Datenvertrag erstellen.
Danach vom Repository-Stamm:

```bash
python -m unittest discover -s tools/thesis_import -p test_baseline_shared2025.py -v
python tools/thesis_import/evaluate_shared2025.py \
  --input-dir /path/to/shared37_output \
  --output-dir /path/to/new_baseline_run
python tools/thesis_import/audit_baseline_shared2025.py \
  --input-dir /path/to/shared37_output \
  --run-dir /path/to/new_baseline_run \
  --policy docs/thesis_research/modern2025_2026_10_29/shared_features.json \
  --report /path/to/new_baseline_verification.json
```

Der Evaluator benötigt die bereits vorhandenen Projektabhängigkeiten NumPy
und SciPy. Er prüft Quelltabellen und Merkmalsrichtlinie gegen das Manifest des
Präparers; ein vorhandenes Zielverzeichnis wird nicht überschrieben.

| Ausgabedatei | Inhalt |
|---|---|
| `oof_predictions.csv` | Je Blockgröße, Modell und Testprobe: beobachtete/vorhergesagte Zielwerte, Fold, nächste Trainingsentfernung und Analogabstände |
| `fold_assignments.csv` | Jede Probe einmal je Blockgröße mit Original-/kanonischer Koordinatengruppe, Gitterblock und Fold |
| `training_roles.csv` | Jede Probe je gehaltenem Fold genau einmal als `train`, `test` oder `buffer_excluded` |
| `metrics.json` | Gesamte und foldweise Fehler, beide Gewichtungen, Stichprobenzahlen und Zielgrößeneinheiten |
| `manifest.json` | Konfiguration, Softwarestände, Eingabe-/Code-Hashes und Hashes aller genannten Ausgaben |

Im Git-PR stehen Konfiguration, Code, kleine Ergebnistabellen, vollständige
Metriken, Prüfnachweis und `reference_manifest.json`. Die rund 9,8 MB großen
vollständigen Laufdateien werden reproduziert; sie sind nicht eingecheckt.
Die modernen Quelldaten behalten CC-BY-4.0. Die bestehende Importer-Referenz
und der ursprüngliche Rechenkern sind unverändert.

## Methodische Quellen

- Legendre & Gallagher (2001),
  [doi:10.1007/s004420100716](https://doi.org/10.1007/s004420100716):
  methodischer Kontext der Transformation von Zusammensetzungsdaten.
- Roberts et al. (2017),
  [doi:10.1111/ecog.02881](https://doi.org/10.1111/ecog.02881):
  Validierung bei räumlich strukturierten Daten. Die hier verwendeten
  Blockgrößen und der 100-km-Puffer sind unsere ausdrücklich dokumentierten
  Versuchsentscheidungen, keine aus diesem Artikel übernommene Optimalempfehlung.
- SciPy-Dokumentation zu
  [cdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)
  und [cKDTree.query](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query.html).

Quellen und Taxonentscheidungen der Eingaben stehen weiterhin im vorherigen
Datenvertrag. Für die nächste Integration genügt der kompakte Auftrag; es ist
keine erneute Recherche der gesamten Projektgeschichte nötig.
