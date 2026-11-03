# Historischer Teilnachbau: kumulierte Rohdichten und EMD-Quellgröße

Stand 02.11.2026, Basis `d6e3343decbb1e74409a6c10e179da4e9da7be9f`,
Branch `dissertation-reconstruction`, [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).

**Abgeschlossen:** Der begrenzte Vergleich kumulierter Rohdichten aus dem
erhaltenen `final_reconstruction` ist in
[historical_cdf_discrepancy.py](../../../tools/historical_cdf_discrepancy.py)
nachgebildet. 43 neue synthetische Tests bestehen, mit den bisherigen
Teilpaketen **167 Tests grün**. Beide rationalen Scorekontrollen stimmen exakt.
Die Quelle nennt diese Größe `emd_tmp`/`emd_pann`; der Nachbau bezeichnet sie
bewusst als wörtliche kumulative Abweichung. Keine historische R-Gleichheit,
physikalisch skalierte Transportdistanz oder vollständige Fallreproduktion
ist dadurch nachgewiesen.

Der konkrete [nächste Auftrag für Claude](../CLAUDE_NEXT_STEP_2026_11_02.md)
enthält Einstieg, Modulkarte, Prüfkommandos und Abnahmekriterien.

## Quelle und Bindung

Drei bereits gebundene Skripte wurden ausschließlich gelesen. Die Hashes vor/
nach Lesen stimmen mit den früheren Quellenpaketen überein.
[source_binding.json](source_binding.json) enthält Bytegrößen und Zeilenauszüge;
private Wurzeln kommen ausschließlich aus `.local/my_reconst_paths.json`.

| Quelle | SHA-256 |
| --- | --- |
| `original:main_v1.2.R` | `82486f99bafff08af2b51475ed225501d26cec1a0322a286b97a248ebd2f0520` |
| `original:src/mcmc_functions_v1.6.R` | `b887144370e328fa016d1834ceebcd55622816d4b36838d33a26197110ab453c` |
| `original:src/mcmc_case_studies.R` | `37587ff08565bf284349e71a9ddbfd924d47fb786e0ccce5168a949be5b9c22e` |

`main_v1.2.R` ist an die
[ältere Eingabebindung](../kinneret_binding_2026_11_02/summary.json) gebunden,
die beiden anderen Skripte an den
[Zeilenauswahlbericht](../literal_age_selection_2026_11_02/source_binding.json).
Keine Rohdatencontainer, Originalposterioren oder neuen Referenzkurven wurden
für dieses Paket geladen oder berechnet.

| Ausschnitt | Quellzeilen | Verhalten |
| --- | --- | --- |
| `prepare_reference_pair` | `mcmc_functions_v1.6.R:369–370` | Länge der ersten Referenz gleich 1 ersetzt beide durch Nullmatrizen |
| `raw_row_cumsums` | 485–486 | Zeilenweise kumulative Summe der bereits berechneten Rohdichte |
| `cdf_discrepancy` | 487–489 | Summe absoluter Differenzen zur übergebenen Referenz, multipliziert mit `1/dims` |
| Fallstudienverbraucher | `mcmc_case_studies.R:309,317` | Beide Aufrufe übergeben `ecdf_ref1=1,ecdf_ref2=1` |
| Normierungszeitpunkt | `mcmc_functions_v1.6.R:755–760` | PDF-Normierung der später zusammengeführten Mischung liegt hinter dem Vergleich |

## Referenzargumente und zwei Zeitpunkte von `dims`

Der Platzhalterzweig prüft ausschließlich `length(ecdf_ref1) == 1`. Er prüft
weder den Wert 1 noch symmetrisch die Länge der zweiten Referenz. Bei endlicher
Einzeleingabe −17 oder einer 1×1-Matrix wird derselbe Zweig erreicht. Beide
Referenzen werden ersetzt; der vorher übergebene zweite Wert wird in diesem
Zweig nicht zur Numerik benötigt. Bei mehr als einem Element bleibt die erste
Referenz bestehen und beide Referenzmatrizen werden später direkt verglichen.
Sie werden nicht erneut kumuliert oder normiert.

Die beiden erhaltenen Fallstudienaufrufe lösen somit den Nullplatzhalter aus.
Der Kommentar „emd between old reconstruction method“ beweist hier keinen
Vergleich mit einer tatsächlich übergebenen älteren Rekonstruktions-CDF.
Dieser Quellbefund wählt auch keinen finalen Kinneret-Lauf aus.

Die Platzhaltermatrix hat Dimension

`nrow(rho.d.g.a) × dims`.

Die Verwendung von `dims` in Zeile 370 liegt **vor** der lokalen Zuweisung
`dims <- length(x1range.bck)` in Zeile 389. Für die Platzhalteranlage wird daher
eine auflösbare Größe aus der Umgebung benötigt. Der erhaltene Haupteinstieg
setzt `dims <- 300` in Zeilen 68 und 658. Das ist ein statischer Quellenbefund,
kein Beleg für eine konkrete historische Aufrufumgebung oder die Breite der
weiterhin fehlenden numerischen TFs.

Der Nachbau verlangt beim Platzhalter ausdrücklich `inherited_dims` und
`rho_rows`. Für die eigentliche Abweichung wird die spätere Breite separat als
`dims` übergeben. Er erfindet keine fehlende Umgebungsgröße und korrigiert keinen
Breitenkonflikt. Ein mit 5 Spalten angelegter Platzhalter passt beispielsweise
nicht zu einer späteren 4-spaltigen Rohdichte. Der begrenzte Nachbau weist das
ab, statt Referenzen neu zu skalieren oder R-Recycling nachzubauen.

## Formel, Dimensionen, Orientierung und Einheiten

Seien `A` die Zahl Alterszeilen und `G` die Zahl geordneter Klimazellen.
Die Rohdichte `D` und die Referenz `R` haben Dimension **A × G**; Zeilen sind
Alter, Spalten sind Klimazellen in ihrer bereits übergebenen Reihenfolge.
Für eine Ziehung berechnet die Quelle:

`C[a,k] = sum(l=1..k) D[a,l]`

`score[a] = (1/dims) × sum(k=1..G) abs(C[a,k] − R[a,k])`.

Im unterstützten konsistenten Bereich gilt `dims=G`. Die Ausgabe ist ein Vektor
mit **A** Einträgen je Ziehung. Die gesamte R-Funktion speichert die Werte in
Feldern **verbleibende Ziehungen × Alterszeilen** (406); Ziehungsselektion und
Zuordnung der `count`-Positionen gehören nicht zu diesem neuen Modul.

Der Ausschnitt fügt keine Normierung von `D`, `C` oder `R` hinzu.
`C[a,G]` ist die Rohmasse der Zeile, nicht automatisch 1. Insbesondere liegt die
spätere Normierung der PDF-Mischung hinter diesem Vergleich. Die getrennte
[PDF-Summe](../literal_final_pdf_2026_11_02/README.md) wird hier nicht vorgezogen.
Eine eventuell vorher bereits erfolgte TF-Normierung ist eine Eingabevoraussetzung
des umgebenden Pfads und wird nicht aus fehlenden Original-TFs rekonstruiert.

Kein Klimakoordinatenwert oder Zellbreitenfaktor kommt in dieser Formel vor.
`1/dims` mittelt Zellbeiträge; es ist nicht ein belegtes Temperatur- oder
Niederschlagsintervall. Die Quellgröße behält die Einheit der kumulierten
Arraywerte. Für diskrete Wahrscheinlichkeitsmassen ist sie dimensionslos;
bei anderen Rohwerten ist eine entsprechende Massenskalierung wirksam.
Ohne weitere Achsen-/Dichtebindung darf sie nicht in °C oder mm/a umgedeutet
werden. Die Nullreferenz ist auch keine normierte Wahrscheinlichkeits-CDF.

Zellreihenfolge ist wesentlich: Umkehren der Dichtespalten verändert die
kumulierten Werte und den Nullreferenzscore. Das Modul sortiert weder Zellen
noch Alter und verwendet keine zusätzlichen Koordinaten.

## Kleine unabhängige Rechenkontrollen

[synthetic_fixture.json](synthetic_fixture.json) enthält drei abstrakte
Alterszeilen mit vier Zellen:

| Rohdichte | Kumuliert | Score gegen Nullreferenz |
| --- | --- | ---: |
| `[2,0,0,0]` | `[2,2,2,2]` | `2` |
| `[0,1,0,0]` | `[0,1,1,1]` | `3/4` |
| `[0,0,3,1]` | `[0,0,3,4]` | `7/4` |

Die absichtlich verschiedenen Rohmassen 2, 1 und 4 machen eine verfrühte
Normierung sichtbar: Sie würde die Scores zu `1,3/4,7/16` verändern.
Mit den zusätzlich explizit angegebenen Referenzmatrizen ergeben sich die
handgeleiteten Werte `25/16,1/8,11/8`. Die kumulative Referenz wird dabei
unverändert eingesetzt. Ein weiterer Test mit künstlich ungleichen
Koordinatenabständen zeigt, dass ein mit Zellbreiten gewichtetes Integral ein
anderer Rechenschritt wäre; solche Breiten werden nicht eingebaut.

[synthetic_result.json](synthetic_result.json) hält die Ergebnisse und Fehler
fest. [Tests](../../../tests/test_historical_cdf_discrepancy.py) prüfen außerdem
den einseitigen Platzhaltertest, nicht ausgewertete zweite Argumente,
unterschiedliche übergebene Breiten, Nullzeilen, eine Klimazelle, fehlende
Mutation, Referenzform sowie Nichtendlichkeits-/Überlaufgrenzen.

```powershell
.venv/Scripts/python.exe docs/thesis_research/literal_cdf_discrepancy_2026_11_02/collect_evidence.py
.venv/Scripts/python.exe -m pytest tests/test_historical_cdf_discrepancy.py tests/test_historical_age_selection.py tests/test_historical_age_depth.py tests/test_historical_final_pdf.py tests/test_kinneret_phase1_spec.py -q -o cache_dir=.local/phase1_spec_v1/pytest_cache
```

[collect_evidence.py](collect_evidence.py) liest nur die drei bereits gebundenen
Skripte und erzeugt Auszüge und synthetische Arithmetik. Die Tests benötigen
keine privaten Originalquellen.
[verification.json](verification.json) dokumentiert die Prüfung und Erhaltung.

## Unterstützter Bereich und Grenzen

Unterstützt sind endliche nichtnegative, nichtleere 2D-Rohdichten, endliche
Referenzmatrizen in exakt passender Form und eine positive ganzzahlige Breite
entsprechend der Zellzahl. Eine Referenz muss im wörtlichen Vergleich nicht
monoton oder nichtnegativ sein; solche Daten werden nicht still zu einer
statistisch gültigen CDF repariert. Das ist keine Validierung ihrer fachlichen
Eignung. Die erste Referenz muss auch beim Platzhalter endlich sein; allgemeine
R-Längen-/NA-Semantik wird nicht beansprucht.

NA/NaN/Inf und `na.rm`, implizites Recycling, Achsenausrichtung, fehlende
Umgebungswerte, Überläufe, physikalische Zellbreiten, empirische Referenz-CDF-
Erzeugung und der spätere Köppen-Geiger-Klassenvergleich sind ausgeschlossen.
Nicht unterstützte Eingaben lösen `UnsupportedSourceDomain` aus.
Keine historischen Originalwerte wurden für die Numerik verwendet; kein R-
oder Paketlaufzeitvergleich. `float64` und Reduktionsreihenfolge können bei
anderen Eingaben von R abweichen. Originale wurden ausschließlich gelesen.

## Ergebnis und Übergabe an Claude

Die [Phase-1-Spezifikation](../kinneret_phase1_spec_2026_11_02/README.md) bleibt
unverändert: 26 Belegdateien, 19 Befunde, 50 Forschungsnachweise und elf offene
Nachweisbündel. Finale Basis `core_data_additional_sites_v1.2.rdata`, finale
Listenchronologie, numerische historische TFs und tatsächliche Laufzuordnung
fehlen weiterhin. Kein finaler Kinneret-Pfad gewählt; kein vollständiger Fall
reproduziert. Die Speicherortfrage ist beantwortet und wird nicht wiederholt.

Der [Claude-Startauftrag](../CLAUDE_NEXT_STEP_2026_11_02.md) beschreibt als
nächsten unabhängigen Schritt eine **rein synthetische Prüfung der Rechenkette**:
die bereits geprüften Module über ein gemeinsames künstliches Beispiel auf
Reihenfolge, Dimensionen und Normierungszeitpunkte prüfen. Nur Test und Belegpaket,
kein historischer Vollrunner und keine Nutzung fehlender Originaleingaben.

App/CLI, Baseline, Referenzen, Originale, bisherige Belegpakete und fallbezogene
Monticchio-/Chronologie-Gates bleiben erhalten. Keine Trainings, Sampler,
Altersmodelle oder Fossilrekonstruktionen. Phase 1 vor Phase 2; bei späteren
Plots jung links, alt rechts. PR #77 bleibt offen, kein automatischer Merge.
