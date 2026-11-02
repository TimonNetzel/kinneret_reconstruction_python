# Historischer Teilnachbau: Taxongewichtung und PDF-Summe

Stand 02.11.2026, Basis `74999b09ae5f7d7aaf784a7e2bf0cb2aba4005fa`,
Branch `dissertation-reconstruction`, [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).

Ein deterministischer Teil des erhaltenen `final_reconstruction` ist jetzt als
separates Python-Modul nachgebildet:
[historical_final_pdf.py](../../../tools/historical_final_pdf.py).
22 neue Tests vergleichen ihn mit unabhängig von Hand hergeleiteten rationalen
Sollwerten und gezielten Gegenbeispielen. Mit den 20 bestehenden Belegvertragstests
bestehen insgesamt 42 Tests. Das ist ein technischer Teilfortschritt in Phase 1,
**keine vollständige lokale Dissertations- oder Kinneret-Reproduktion**.

## Quelle und genauer Ausschnitt

Quelle ist `original:src/mcmc_functions_v1.6.R`, SHA-256
`b887144370e328fa016d1834ceebcd55622816d4b36838d33a26197110ab453c`.
Der vor/nach dem Lesen identische Hash stimmt mit dem bereits abgeschlossenen
[Nachbearbeitungsvertrag](../kinneret_postprocessing_2026_11_02/README.md) überein.
Die verwendeten Zeilen stehen in [source_binding.json](source_binding.json).
Keine Originaldatei wurde geschrieben, kein kompletter R-Code ausgeführt.

| Teilfunktion | Erhaltene R-Zeilen | Nachgebildetes Verhalten |
| --- | --- | --- |
| `column_max_taxa` | 411–416 | Jede Taxonspalte durch ihr eigenes Maximum teilen |
| `rw_coefficients` | 417–425, 450–455 | Gewichte in Taxonreihenfolge je Probenzeile anwenden, danach Zeilensumme normieren; entspricht `c(t(...))` und `matrix(...,byrow=T)` |
| `gibbs_coefficients` | 433–436 | Erst dieselbe Spaltenskalierung, dann ausgewählte Taxa und Zeilennormierung; Reihenfolge und Wiederholungen erhalten |
| `apply_taxon_densities` | 439–443, 458–462 | Koeffizientenmatrix mit numerischen TF-Zeilen multiplizieren; keine zusätzliche TF-Normierung |
| `unique_energy_ids` | 375 | Erste Position jedes exakt unterschiedlichen Energiewerts; kein Runden |
| `sum_pdf_branch` | 375, 428, 541–547, 711–715, 755–760 | Rohe Dichtezeilen der verbleibenden Ziehungen summieren, erst danach je Probe normieren |

Taxon- und Ziehungs-IDs sind ausdrücklich **einsbasiert**, passend zu R.
Die Energie wird zur Duplikaterkennung benutzt, nicht als Mischungsgewicht.
Die im Original berechnete Hilfsgröße `energy_weight` fließt nicht in diese
Dichtesumme ein. Frühere Ziehungsauswahl durch Prior/IAT/Pareto gehört nicht
zu `unique_energy_ids`; die Funktion behauptet keine Rekonstruktion dieser Auswahl.

Der explizite Name `sum_pdf_branch` bezeichnet nur diese erhaltene Quellvariante.
Er wählt sie nicht zur finalen historischen Kinneret-Fassung. Das textlich
beschriebene Mittel unabhängiger Zufallsgrößen, stochastisches `averaging_data`,
KDE, Splines, Quantile/Mediane, EMD, Klima-Klassen und Plotausgaben sind nicht
implementiert. Altersauswahl `reconst_ages`, `taxa_change` und Chronologie sind
ebenfalls außerhalb dieses Ausschnitts. Kein CLI-Runner, keine App-Integration,
kein Zugriff auf private Daten durch das neue Modul.

## Kleine unabhängige Rechenkontrollen

[synthetic_fixture.json](synthetic_fixture.json) enthält vier abstrakte Proben,
drei Taxa, vier abstrakte Dichtezellen und drei Gewichtungsvektoren. Alle Werte
sind künstlich; die Zellen besitzen keine historische Klimaskala. Die rechteckige
Pollentabelle prüft Reihenfolge und Transponierung. TF-Zeilensummen 2, 1 und 3
sind absichtlich verschieden: Eine unzulässige frühere Normierung würde damit
sichtbar. Energien `[4,4,9]` erhalten die Originalpositionen `[1,3]`.

Beispiel Probe 2: Die erste Gewichtung ergibt Taxonkoeffizienten
`[2/5,2/5,1/5]` und eine rohe Dichte `[4/5,2/5,0,3/5]`.
Mit der dritten Ziehung ist die rohe Summe `[9/5,13/20,0,27/20]` mit Masse `19/5`.
Die normierte Ausgabe lautet deshalb `[9/19,13/76,0,27/76]`.
Eine Normierung jeder Ziehung vor der Summe würde andere Werte ergeben und
wurde nicht eingebaut. Austausch der Energiewerte unter Erhaltung ihres
Duplikatmusters ändert das Ergebnis nicht. Das Ändern der übersprungenen
zweiten Dichte ändert die Ausgabe ebenfalls nicht.

[synthetic_result.json](synthetic_result.json) hält die Ergebnisse und Abweichungen
fest. Alle vier geprüften Stufen liegen innerhalb `1e-14` der rationalen Sollwerte.
Die Tests sind in [test_historical_final_pdf.py](../../../tests/test_historical_final_pdf.py).

```powershell
python -m pytest tests/test_historical_final_pdf.py tests/test_kinneret_phase1_spec.py -q
```

## Unterstützter Bereich und Grenzen

Das Modul akzeptiert ausschließlich endliche, nichtnegative und nichtleere
Datenfelder mit positiven Taxonmaxima und positiven benötigten Zeilensummen.
Energien müssen endlich sein und dürfen negativ sein. Keine Imputation,
Glättung, Nullersetzung, zusätzliche Normalisierung oder neue Kalibrierung.
Nicht unterstützte Eingaben werden ausdrücklich mit `UnsupportedSourceDomain`
abgewiesen; daraus darf keine Aussage über ein angeblich korrigiertes
R-Verfahren abgeleitet werden.

R-Verhalten für NA/NaN/Inf und `na.rm`, Division durch null, falsche
Vektorlängen mit Recycling sowie Dimensionsverlust beim Gibbs-Subset werden
nicht emuliert. Ein einziger ausgewählter Taxon oder eine einzige Probenzeile
würde dort R-Matrixdimensionen verlieren und wird hier nicht still repariert.
Die kontrollierte Eingabedomäne ist eine Grenze des Teilnachbaus, keine
methodische Änderung am Original. Eingabefelder werden nicht mutiert.

Geprüft sind die übersetzten Formeln und synthetischen Sollwerte. Ein Vergleich
gegen die historische R-/Paketlaufzeit wurde nicht ausgeführt; weder
Bitgleichheit noch die vollständige Semantik von `final_reconstruction` ist damit
nachgewiesen. `float64` und NumPy-Matrixprodukte können sich bei anderen Eingaben
in Rundung/Reduktionsreihenfolge von R unterscheiden.

## Ergebnis und nächster Einstieg

Der [Phase-1-Belegvertrag](../kinneret_phase1_spec_2026_11_02/README.md) bleibt
unverändert: elf offene Nachweisbündel, keine ausgewählte finale Laufvariante,
kein startbereiter vollständiger Fall. Neue Originalbasis/Chronologieliste,
numerische historische TFs und Laufzuordnung bleiben notwendig. Der Nutzer
kennt keinen weiteren Basisdatei-Speicherort; keine erneute Frage/Globalinventur.

Unabhängiger nächster technischer Phase-1-Einstieg: die historische
Alters-/Tiefenübertragung im bereits gebundenen `src/check_ages.R` als klar
begrenzten Ausschnitt lesen und mit synthetischen Matrizen prüfen. Listen-
und Matrixchronologie nicht gleichsetzen; keine neuen Altersmodelle oder
Fossilkurven berechnen. Das schließt die offene finale Chronologieliste nicht.

[verification.json](verification.json) dokumentiert die Prüfung.
App/CLI, Baseline, Referenzen, Originale und Monticchio-/Chronologie-Gates
bleiben erhalten. Keine Trainings, Sampler oder Fossilrekonstruktionen;
keine Verbesserungen aus Phase 2. Für spätere Plots weiter jung links, alt rechts.
