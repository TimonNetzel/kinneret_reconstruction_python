# Historischer Teilnachbau: Alters-/Tiefenübertragung

Stand 02.11.2026, Basis `f7d97c1fad6852ce948cb868a16b10f7b1f83b8f`,
Branch `dissertation-reconstruction`, [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).

**Abgeschlossen:** Ein begrenzter deterministischer Ausschnitt der erhaltenen
Chronologieauswahl und Alters-/Tiefenübertragung ist in
[historical_age_depth.py](../../../tools/historical_age_depth.py) nachgebildet.
43 neue synthetische Testfälle bestehen; einschließlich Taxongewichtung/PDF-Summe
und Belegvertrag sind 85 Tests grün. Keine historische R-Laufzeitgleichheit,
keine neue Chronologie, keine vollständige Kinneret-Reproduktion und keine
finale Laufvariante sind damit nachgewiesen oder ausgewählt.

## Quellen und Abgrenzung

Die drei bereits gebundenen Originalskripte wurden ausschließlich gelesen.
SHA-256 vor/nach dem Lesen stimmt jeweils mit dem
[vorherigen Quellenbericht](../kinneret_binding_2026_11_02/summary.json) überein.
[source_binding.json](source_binding.json) enthält Hashes, Bytegrößen und
Zeilenauszüge; private Quellwurzeln kommen aus `.local/my_reconst_paths.json`.

| Quelle | SHA-256 | Bedeutung |
| --- | --- | --- |
| `original:src/check_ages.R` | `c7f92e6030d83540c14f29a2e9d0ad07c74104dfe86befebda921c1cffbd2729` | 175–191: Listenchronologie wählen; 195–208: Tiefenbehandlung; 223/230–232: Zeitfenster vor einmaliger Altersverschiebung |
| `original:src/mcmc_prepare.R` | `b897e84f8f2d72399de729592066341585974d521bf8164b185f20377ed5cc7e` | 3: ruft `check_ages.R` auf; 11–20: Tiefenauswahl, Normierung, Nullersetzung; 83/91–94: Pollennullen und Matrixprodukt |
| `additional:my_kinneret/src/bacon_kinneret.R` | `c9406d13083ddbb8248ca4b6ac24b43b42067d1044bbc833a5155274145b5f48` | 68–100: Listen-/Altersrasterschreiber; 105–146: anderer Schreiber mit Mittelwertmatrix unter demselben Dateinamen; hier keine Ausführung |

`check_ages.R` selbst enthält kein Pollen-Matrixprodukt. Der durch Zeile 3
explizit verbundene Verbraucher `mcmc_prepare.R` ist deshalb für den eigentlichen
Übertragungsschritt erforderlich. Die nachgebildeten Teile sind:

| Funktion | Belegte Zeilen | Begrenzung |
| --- | --- | --- |
| `select_list_age_grid` | `check_ages.R:177,179` | Explizit übergebene einsbasierte Listenposition; keine Standortwahl oder finale Auswahl |
| `shift_duplicate_ages_once` | `check_ages.R:230–232` | Exakte Duplikate nach dem ersten Vorkommen einmal um 0,5 erhöhen |
| `depth_kernel_at_samples` | `mcmc_prepare.R:11–20` | Bereits geordnete, eindeutige, ausgerichtete Probentiefen; kein Birkat-Ram-Sonderzweig |
| `pollen_at_ages` | `mcmc_prepare.R:83,91–94` | Bereits festgelegte Taxonspalten; Pollennullen vor Produkt ersetzen; keine Ausgabeimputation |

Es gibt keinen Runner, keine neue App-/CLI-Integration und keine automatisierte
Quellvariantenauswahl. Die Funktionen sind getrennte Ausschnitte; sie behaupten
keine vollständige Ausführung des umgebenden R-Skripts.

## Formeln, Dimensionen und Orientierung

Seien `A` die Zahl ausgewählter Alterszeilen, `G` die Zahl vollständiger
Tiefengitterspalten, `S` die Zahl Probentiefen und `T` die Zahl ausgewählter Taxa.
Die gespeicherte Eingabematrix `R` hat Dimension **A × G**: Zeilen sind Alter,
Spalten Tiefengitterwerte. Das ist die erhaltene Orientierung von
**rho(D gegeben A)**. Es ist keine Matrix rho(A gegeben D); weder Transponierung
noch Bayes-Umkehrung werden vorgenommen.

Für Probentiefe `d[j]` verwendet R die **einsbasierte** Spaltenposition

`k[j] = (d[j] − min(d)) + 1`.

Das setzt in diesem Ausschnitt ein Gitter mit Schrittweite einer Tiefeneinheit
und Ursprung `min(d)` voraus. Der Nachbau akzeptiert nur exakt ganzzahlige,
gültige Offsets; keine Rundung, Interpolation oder R-Abschneidung gebrochener
Indizes. Die Gleichheit mit einer tatsächlichen historischen Gitterachse muss
separat belegt werden. Es wird keine Achse geschätzt.

Zuerst werden nur die Probenspalten gewählt und ihre Masse je Alterszeile
normiert:

`Q[a,j] = R[a,k[j]] / sum_l R[a,k[l]]`.

Danach setzt Zeile 20 **exakte Nullen** auf das historische Literal `10e-10`,
also **1e-9**:

`W[a,j] = 1e-9 falls Q[a,j] = 0, sonst Q[a,j]`.

`W` hat Dimension **A × S**. Es folgt **keine zweite Normierung**. Bei `z`
Nullen ist die Zeilensumme rechnerisch `1 + z × 1e-9`, bis auf Rundung.
Die Auswahl normiert über die beobachteten Probentiefen, nicht über das ganze
Tiefengitter. Dieses Verhalten wird historisch nachgebildet, nicht korrigiert.

Die Pollentabelle `P` hat Dimension **S × T**, in exakt derselben Tiefenreihenfolge
wie die Spalten von `W`. Nach der hier ausgeschlossenen Taxonfilterung ersetzt
Zeile 83 exakte Pollennullen durch **1e-6**, nicht durch das Chronologieliteral:

`P'[j,t] = 1e-6 falls P[j,t] = 0, sonst P[j,t]`.

Die Übertragung lautet anschließend

`Y[a,t] = sum_j W[a,j] × P'[j,t]`, also **(A × S) × (S × T) = A × T**.

Das Produkt normiert weder Taxonzeilen noch Taxonspalten. Wegen der vorherigen
Nullersetzung sind die Gewichte auch nicht durchgehend exakt massenerhaltend.
Die spätere taxonweise Maximumskalierung aus
[final_reconstruction](../literal_final_pdf_2026_11_02/README.md) ist ein anderer
Schritt; sie wird hier nicht vorgezogen.

## Einheiten und zwei verschiedene Chronologieformen

Die [bestehende Probenbindung](../kinneret_binding_2026_11_02/sample_crosswalk.csv)
weist für den älteren Kinneret-Kandidaten Tiefen von 1 bis 1772 **cm** und mittlere
Alter von −60 bis 9015 **cal a BP** aus. Die 160 × 1772-Matrix hat Alterszeilen
an den 160 mittleren Probenaltern und Gitterspalten für Tiefen. Gleiche Zeilen-
und Probenanzahl macht die erste Achse nicht zu einer Tiefenachse.

Der Listen-Schreiber legt hingegen eine Liste verschiedener Matrizen und eine
passende Liste von Altersvektoren an: mittlere Probenalter, 50-Jahresraster,
500-Jahresraster (75/79–85/95–100). Der spätere Schreiber speichert die Matrix
mit flachem Altersvektor (132–146). Gleicher Dateiname ist kein Beleg, welche
Fassung historisch konsumiert wurde. Die erhaltene Mittelwertmatrix ersetzt
keine finale Liste. `[[2]]` auf Matrix oder flachem Vektor liefert in R einen
Einzelwert, kein zweites Raster. Der Nachbau weist diese Eingabeform ausdrücklich
ab, statt sie automatisch umzubauen.

Die Normierung verwendet diskrete Summen ohne `delta depth`, kein kontinuierliches
Integral. `W` ist dimensionslos; `Y` behält die numerische Einheit der
eingesetzten Pollentabelle. Eine Pollenzählung wird hier nicht automatisch zu
Prozenten. Die Original-Konstante 1e-6 wird in der jeweiligen Eingabeeinheit
hinzugefügt. Es erfolgt keine Umrechnung cm/m oder a/ka. Die Testgitter sind
abstrakt und werden nicht als historische Alter oder Tiefen ausgegeben.

## Altersverschiebung und bewusst ausgeschlossene Pfade

Die Duplikatmaske in `check_ages.R:230–232` wird auf dem ursprünglichen `age`
berechnet und einmal angewendet. `[100,100,100,100.5]` wird damit zu
`[100,100.5,100.5,100.5]`. Es bleiben Duplikate oder neue Kollisionen möglich;
keine Schleife stellt Eindeutigkeit her. `site.age` und die bereits vorher
berechneten Zeitfenster werden nicht mitverschoben; die Matrixwerte werden
nicht neu berechnet oder interpoliert. Die Konstante 0,5 trägt die Einheit des
Altersvektors, im gebundenen Kinneret-Kandidaten also a. Diese Unterscheidung
bleibt im Ausschnitt erhalten: Er gibt eine Kopie von `age` zurück.

Weitere Quellenpfade werden dokumentiert und **nicht still repariert**:

- `check_ages.R:176` überschreibt für Dead Sea `age.reso`; 187–190 bieten nur
  im Nicht-`add.sites`-Zweig einen Leerraster-Fallback. Beides ist ausgeschlossen.
- 195–199 benutzen eine Mengenmitgliedschaftsmaske, keine eindeutige geordnete
  Verknüpfung. `output$depth` wird unter einer Bedingung auch außerhalb des
  Ladezweigs gelesen. Ein passender historischer Workspace ist nicht bewiesen.
- 201 sortiert allein `depth`; die Pollentabelle wird dabei nicht mitpermutiert.
  203–208 entfernen spätere doppelte Tiefen und entsprechende Pollenzeilen.
  Der Nachbau führt diese Pfade nicht aus; ungeordnete oder doppelte Tiefen
  werden abgewiesen. Gleiche Dimensionen allein beweisen keine Ausrichtung.
- Der Birkat-Ram-Sonderzweig 24–34 glättet drei Spalten mit LOESS, normiert
  erneut und ersetzt nichtpositive Werte. Er gehört nicht zu diesem Ausschnitt.
- Taxonfilterung 51–80, vorherige NA→0-Behandlung und nachträgliche
  NA-Ausgabefüllung 96–101 sind ausgeschlossen. Es gibt keine Imputation,
  Neuschätzung, neue Kalibrierung oder Erweiterung zu NA/Inf-Semantik.

Unterstützt sind endliche, nichtnegative, nichtleere Matrizen mit positiver
gewählter Zeilenmasse und endlichem Produkt, gültige explizite Listenpositionen
sowie bereits ausgerichtete, streng steigende Tiefen mit ganzzahligen Offsets.
Nicht unterstützte Domänen lösen `UnsupportedSourceDomain` aus. Diese Grenzen
beschreiben die Übersetzung, keine behauptete Korrektur des Originals.

## Synthetische Kontrolle und Wiederholung

[synthetic_fixture.json](synthetic_fixture.json) enthält eine **4 × 5**-Matrix,
drei Probentiefen, zwei Taxa und rationale Sollwerte. Die ausgewählten R-Spalten
sind `[1,3,5]`. Große Werte in den ausgelassenen Spalten zeigen eine falsche
Normierung über das ganze Gitter. Vier Alterszeilen gegenüber drei Tiefen
machen die Orientierung im Produkt prüfbar. Nullzellen prüfen beide
unterschiedlichen Konstanten sowie den fehlenden zweiten Normierungsschritt.

Beispiel Zeile 1: ausgewählte Rohwerte `[1,2,1]` ergeben `[1/4,1/2,1/4]`.
Mit Pollenzeilen `[[2,0],[0,4],[6,2]]` entsteht nach Nullersetzung
`[4000001/2000000,10000001/4000000]`.
Zeile 2 erhält Gewichte `[1e-9,3/4,1/4]`; ihre Masse ist `1+1e-9`.
Eine anschließende Renormierung oder fehlende Pollen-Nullersetzung liefert
andere Werte. [synthetic_result.json](synthetic_result.json) enthält die
Ausgabe und Fehler gegen die von Hand hergeleiteten Brüche: höchstens
`2.23e-16` bei Toleranz `1e-14`.

[Tests](../../../tests/test_historical_age_depth.py) prüfen außerdem explizite
Listenpositionen, Matrix-/Vektorabwehr, ungültige Dimensionen und Tiefen,
Nullmasse, NA/Inf, Überlauf, fehlende Mutation, Zeilenskalierung und verbleibende
Altersduplikate. Kein historischer numerischer Originalwert wird als Testeingabe
verwendet. Kein R-Skript und keine R-/Bacon-/LOESS-Laufzeit wurden ausgeführt;
`float64`-Rundung und Reduktionsreihenfolge können von R abweichen.

```powershell
.venv/Scripts/python.exe docs/thesis_research/literal_age_depth_2026_11_02/collect_evidence.py
.venv/Scripts/python.exe -m pytest tests/test_historical_age_depth.py tests/test_historical_final_pdf.py tests/test_kinneret_phase1_spec.py -q -o cache_dir=.local/phase1_spec_v1/pytest_cache
```

[collect_evidence.py](collect_evidence.py) liest nur die drei bereits gebundenen
Skripte, prüft deren bisherige Hashes und exportiert Auszüge sowie synthetische
Arithmetik. Für die portablen Tests sind keine privaten Originalpfade nötig.
[verification.json](verification.json) dokumentiert Prüfung und Erhaltung.

## Ergebnis, Grenze und nächster Einstieg

Der [Phase-1-Belegvertrag](../kinneret_phase1_spec_2026_11_02/README.md) bleibt
unverändert: 26 Belegdateien, 19 Befunde, 50 Snapshotdateien und elf offene
Nachweisbündel. Finale Basisdatei `core_data_additional_sites_v1.2.rdata`,
Listenchronologie, numerische historische TFs und tatsächliche Laufzuordnung
fehlen weiterhin. Auch der nachgebildete Übertragungsschritt schließt diese
Voraussetzungen nicht. Der Nutzer kennt keinen weiteren Basisdatei-Speicherort;
keine wiederholte Frage, Gesamtinventur oder Ersatzdatenwahl.

Nächster unabhängiger technischer Einstieg in Phase 1: die bereits gebundenen
Verbraucher von `age` gegenüber `site.age` und `reconst_ages` in
`src/mcmc_functions_v1.6.R` gezielt auf Zeilen-/Zeitfensterauswahl verfolgen,
bevor die getrennten deterministischen Module überhaupt verbunden werden.
Einen belegten Ausschnitt synthetisch prüfen; keine Synchronisierung oder
Variantenwahl aus eigener Annahme. Vollständiger Fallnachbau bleibt auf neue
Originalbelege angewiesen.

App/CLI, Baseline, Referenzen, Originale und fallbezogene Monticchio-/Chronologie-
Gates sind erhalten. Keine Trainings, Sampler, Altersmodelle oder neuen
Fossilrekonstruktionen. Phase 2 bleibt nachgeordnet. Bei späteren Plots: jung
links, alt rechts. Bestehender PR bleibt offen; kein automatischer Merge.
