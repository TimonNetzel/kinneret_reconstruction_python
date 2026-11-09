# Kinneret: historische Ziehungsauswahl und Dichtekombination

Stand 02.11.2026; Phase 1, Ausgangscommit `edfd3f1`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).
Fortsetzung von [Fallmatrix](../local_case_matrix_2026_11_02/README.md),
[Proben-/Taxonbindung](../kinneret_binding_2026_11_02/README.md) und
[Referenz-/Ankervertrag](../kinneret_references_2026_11_02/README.md).

**Abgeschlossen:** Nachbearbeitung im Dissertationstext, Kinneret-Fallstudiencode,
verwandten Auswertungsfassungen und heutigem Einstieg getrennt gebunden.
Zwei deterministische synthetische Kontrollen belegen einen möglichen
Indexverlust nach der Filterung und die Nichtgleichheit von Dichtemischung
und Mittel unabhängiger Zufallsgrößen. **Keine Variante ist damit als finale
historische Kinneret-Ausführung identifiziert.** Originalposterioren, die zehn
ursprünglich ausgewählten Ziehungen und numerische Enddichten fehlen weiterhin.

Keine Originaldatei oder historische Codezeile geändert. Keine R-Ausführung,
IAT-Schätzung, Zufallsziehung, Dichteanpassung, Trainings, Sampler oder neue
Fossilrekonstruktion. Die synthetischen Beispiele sind technische Rechenkontrollen
mit kleinen Zahlen, keine Ersatzdaten oder Klimakurven.

## Quellenbindung

Wurzel `original` stammt ausschließlich aus der vorhandenen privaten
`.local/my_reconst_paths.json`. Es wurden die bereits bekannten Dateien gelesen
und acht daraus abgeleitete Dateipfade geprüft; keine neue Gesamtinventur.

| Quelle | Fundstelle / Rolle |
| --- | --- |
| `original:out/plots/mcmc/v5.0/diss_final.zip`, `thesis_mcmc_results.tex:209–220,274–293` | Textverfahren: Prior-IQR, IAT <1,3, Pareto-Nähe zu Prior-Mittelwerten, zehn Rekonstruktionen, Kombination durch Mittel unabhängiger Zufallsgrößen. Zugehörige lokale PDF-Fassung §3.2.2, S.61–64 / PDF67–70. |
| `original:main_v1.2.R:74–79,624–632,654–683` | Globale Kombinationsschalter; aktueller rwMH-Block 25000 Iterationen / Burn-in 5000 / Schritt 5; aktiv aufgerufener Nachbearbeiter und kommentierte Alternativen. |
| `original:src/mcmc_prepare.R:1295–1364` | IQR-Grenzen als Quantile von jeweils 10000 Prior-Ziehungen mit Seed 2022; hier nur Quelltext gelesen. |
| `original:src/mcmc_case_studies.R:8–43,67–213,232–246,282–332` | Zwei Vergleichsposterioren, Kinneret-Schwellen, Filter, Score-Deduplizierung, IAT-/Pareto-Schritte, Indexübernahme und Dichtemischung. |
| `original:src/mcmc_trimming.R:87–187`; `src/mcmc_algorithm_plots.R:1068–1123,1141–1168`; `src/independent_samples.R:4–33` | Gezielte Vergleichsfassungen: teilweise andere IAT-Schleifen, explizite Rückabbildung auf Originalindizes, andere Kombinationsschalter. Keine dieser Fassungen automatisch ausgewählt. |
| `original:src/mcmc_functions_v1.6.R:366–388,419–477,539–611,683–766,843–851` | `final_reconstruction`: weitere Deduplizierung, Taxonmischungen, Kombination und Ergebnisfelder. |
| `original:src/mcmc_trimming_posterior_trafo.R:94–113,168–261,308–336` | Heutiger aktiver Einstieg: Ausdünnung/Deduplizierung und KDE über gespeicherte Klima-Punktwerte. |
| `original:src/final_plotting.R:65–104` | Spline-/Quantilauswertung der Tiefen-/Altersdichten; zusätzlicher enger Anzeigequellkonflikt beim PANN-Tiefenmedian. |

Hash- und Statusnachweis: [contract.json](contract.json).
Die konkret zitierten Auszüge stehen in [source_spans.json](source_spans.json).
Alle Aussagen gelten für die genannten Quellstände; sie werden nicht auf alle
historischen Dissertationsergebnisse verallgemeinert.

## Vier Wege getrennt halten

| Weg | Auswahl der Ziehungen | Dichte-/Ergebnisbildung | Grenze |
| --- | --- | --- | --- |
| Dissertationstext | Prior-IQR → IAT unter 1,3 → Pareto-Nähe zu Prior-Mittelwerten → zehn einzelne Rekonstruktionen | Mittel unabhängiger Zufallsgrößen durch Faltung, Skalierung 1/N | Beschriebener historischer Zielweg; numerische Ziehungs-/Laufbindung fehlt. |
| Kinneret-Fallstudiencode | Schwellenfilter → gerundeter Score/Deduplizierung → einmalige IAT-Schrittweite → Prior-Pareto → zusätzlicher Proxy-Pareto → höchstens zehn | Setzt `averaging_pdfs=T`, `averaging_data=F`: summierte Dichtezeilen mit abschließender Normierung | Rückabbildung gefilterter Positionen kommentiert; andere Regel als Text. |
| Verwandte Auswertungsfassungen | `mcmc_trimming.R` und `mcmc_algorithm_plots.R` bilden Indizes ausdrücklich auf `best_combs` zurück; konkrete IAT-/Pareto-Details unterscheiden sich | Trimming schaltet auf Dichtemischung; Algorithmusvergleich auf Mittel unabhängiger Zufallsgrößen | Belegt Alternativen, aber keine Kinneret-Laufidentität. |
| Heutiger Haupteinstieg | Aktiver `mcmc_trimming_posterior_trafo.R`: feste Burn-in-/Thin-Auswahl, Score auf fünf Dezimalen, Deduplizierung; IQR-Block kommentiert | Pro Ziehung gewichtete Klima-Punktwerte aus `posterior$tf_sample`, danach KDE je Alter/Tiefe | Keine belegte Umsetzung des beschriebenen Zehner-Dichtewegs. |

Im derzeitigen Haupteinstieg sind die `source(...)`-Aufrufe für
`mcmc_case_studies.R` und `mcmc_trimming.R` kommentiert. Die aus der früheren
Quellenarbeit bekannten Vorbereitungsstopps und fehlenden Eingaben bestehen
weiter. Diese Tabelle beschreibt Verbraucherpfade, keinen durchgeführten Lauf.

## Kinneret-Fallstudiencode: tatsächliche Reihenfolge

1. Zwei Posterior-Dateien laden: Vergleich ohne Zusatzinformation und Hauptfall.
   Für `percent=FALSE`, `gibbs=FALSE`, Standortslot 3, Simulation 1 lauten sie
   `out/posterior/rw_posterior_add_3_1_case_studies_counts.rdata` und
   `out/posterior/rw_posterior_add_3_1_counts.rdata`.
2. Kinneret-spezifische Grenzen überschreiben: `iso_thresh=c(0.2,1)`,
   `spec_thresh=c(0.2,1)`, `iso_hi_thresh=c(0.05,1)` und angepasste
   Spektral-HI-Grenze; im zweiten Vergleich `pollen_thresh=c(0.5,1)`.
   Die heutigen Werte sind daher nicht einfach die unveränderten Prior-IQR.
3. Klima-/Proxy-Filter bilden. Der aktuelle Klimafilter verbindet die rezenten
   Grenzen mit dem LGM-Check. MH- und andere Checks werden zwar berechnet,
   hier aber nicht genauso in `best_combs` eingesetzt. Leere akzeptierte Menge
   führt zu `stop()`. Im Auswahlblock gibt es keinen expliziten Burn-in-Ausschluss;
   ob die geladenen Dateien bereits gethinnt waren, bleibt ohne Dateien offen.
4. `post_pollen[best_combs]` auf fünf Dezimalen runden, gleiche Scorewerte nur
   einmal behalten. Unter der Quellenbedingung wird `post_iso` als Fallback
   benutzt. Gleicher Score beweist keine identischen Parameter, Gewichte oder
   Rekonstruktionen; diese Deduplizierung bleibt als historischer Schritt erhalten.
5. Bei mehr als zehn verschiedenen Scorewerten `IAT` einmal schätzen,
   `ceiling(IAT)` als Abstand benutzen. Der verschiebende/revidierende IAT-Loop
   liegt im Zweig `length(energy_temp)>=100 & FALSE` und ist somit deaktiviert.
   Es gibt hier keinen aktiven Nachweis, dass danach IAT <1,3 erreicht ist.
   Bei höchstens zehn Scorewerten wird in diesem Abschnitt keine IAT berechnet.
6. Bei `recent` Pareto-Aufruf mit `low(tann_diff)*low(pann_diff)`,
   `top_level=10` und anschließendem Deckel von 20 Zeilen; danach zusätzliche
   Präferenzen für hohe Proxy-Scores, ggf. HI-Scores, und Deckel von zehn Zeilen.
   Die Quelle verwendet `rPref::psel`-Ausdrücke; deren Ausgabeordnung und genaue
   Paketsemantik sind nicht durch eine moderne Neuimplementierung ersetzt.
7. Gewichte/Taxonauswahl und `loglike` an `final_reconstruction` übergeben.
   Der Quellstand stoppt anschließend bei Zeile 332; der dortige Aufruf des
   finalen Plotters ist kommentiert. Spätere Plot-/Gewichtsabschnitte hinter
   dem Stopp belegen keine ununterbrochene vollständige Ausführung.

`IAT` wird im Haupteinstieg mit `library(LaplacesDemon) # IAT` verbunden,
`psel` mit `rPref`. Die separate Hilfsfassung `independent_samples.R` hat
einen aktiven Abbruchwert 1,05 oder beendet die Auswahl wegen eines zu kleinen
Restpools. `mcmc_trimming.R` deaktiviert dagegen bereits den äußeren
IAT-Zweig mit `if(length(energy_temp)>10 & FALSE)`; auch der innere Loop trägt
`& FALSE`. Keiner dieser Befunde wird zum behaupteten historischen
1,3-Laufnachweis umgedeutet; historische Paketversionen fehlen weiterhin.

## Indexverlust: deterministisch eingegrenzter Quellkonflikt

`best_combs` enthält ursprüngliche Ziehungsnummern. Die Deduplizierung von
`energy = ...[best_combs]` erzeugt dagegen Positionen **innerhalb dieser
gefilterten Liste**. Im Fallstudiencode ist die anschließende Zeile
`independent_samples <- best_combs[independent_samples]` auskommentiert.
Die so entstandenen Positionen werden anschließend direkt an ungefilterten
Posteriorfeldern benutzt.

Das vollständig synthetische Beispiel in [synthetic_checks.json](synthetic_checks.json)
verwendet zwölf Scorewerte und die vier akzeptierten Originalziehungen
`[4,7,9,12]`. Ihre Scores `[0.56,0.56,0.57,0.58]` ergeben nach Deduplizierung:

| Größe | Wert |
| --- | --- |
| Positionen im gefilterten Vektor | `[1,3,4]` |
| Dadurch bezeichnete ursprüngliche Ziehungen | `[4,9,12]` |
| Vom unveränderten Fallstudiencode vor Pareto direkt benutzte Indizes | `[1,3,4]` |
| Direkt benutzte Indizes außerhalb des Filters | `[1,3]` |

Die drei verschiedenen Scores umgehen in diesem Beispiel den IAT-Zweig,
es wurden keine IAT und keine Pareto-Ausgabe geschätzt. Das Beispiel belegt
einen möglichen Verlust der Filterbindung im erhaltenen Code, **keinen
nachgewiesenen Fehler einer konkreten Dissertationstabelle oder Abbildung**.
Die Vergleichsfassungen besitzen die explizite Rückabbildung; keine
Quellfassung und kein Originalposterior wurde hier korrigiert.

## Dichtemischung, Mittel von Zufallsgrößen und heutige KDE

`final_reconstruction` bereitet zunächst jede einzelne Rekonstruktionsdichte
als gewichtete Summe von Taxon-Transferfunktionen vor. Diese taxoninterne
Kombination ist von der folgenden Kombination mehrerer ausgewählter Ziehungen
zu unterscheiden.

- **PDF-Modus:** Dichtezeilen der ausgewählten Ziehungen summieren und am Ende
  zeilenweise normieren. Eine gleiche skalare Mischung folgt nur bei passenden
  Eingangsnormierungen. Die normalisierte Variable `energy` wird nicht als
  Gewicht in die Dichtesumme eingesetzt; `energy_weight` wird im Funktionskörper
  einmal definiert und danach nicht verwendet. Vorher werden nochmals exakt
  gleiche `energy`-Werte zusammengefasst.
- **Zufallsgrößenmittel:** Pro ausgewählter Dichte, Alter/Tiefe und Klimagröße
  25000 Werte ziehen, addieren, durch die Zahl der erhaltenen Ziehungen teilen
  und eine Kerndichte bilden. Für PANN folgt eine zusätzliche Spline-/Index-
  Rückabbildung auf das native Gitter. Das ist der erhaltene Monte-Carlo-Zweig,
  keine hier neu ausgeführte Faltung. Er benötigt vollständige TF-Achsen,
  Originaldichten, R-/KDE-Versionen und RNG-Zustand.
- **Conflation:** Normiertes punktweises Produkt als weitere Quellalternative.
  Die Ausgabeauswahl erfolgt in der Reihenfolge `averaging_pdfs`, sonst
  `averaging_data`, sonst `conflation`. Bei höchstens einer Auswahlzeile setzt
  die Funktion lokal den PDF-Modus. Globale Schalter allein beweisen daher
  nicht den tatsächlich verwendeten Kombinationsweg.

Die zweite synthetische Kontrolle verwendet ausschließlich exakte rationale
Wahrscheinlichkeiten: Eine normierte Verteilung mit Masse 1/2 bei 0 und 2
hat Mittelwert 1 und Varianz 1. Die gleiche Mischung aus zwei solchen
Verteilungen behält diese Momente. Das Mittel zweier unabhängiger Zufallsgrößen
hat dagegen Massen 1/4,1/2,1/4 bei 0,1,2 und Varianz 1/2. Damit sind die
beiden Quellenmodi nachweislich nicht austauschbar. Dies ist keine neue
Unsicherheitskorrektur; keine Variante wird fachlich bevorzugt oder übernommen.

Der heutige aktive Nachbearbeiter berechnet stattdessen Klima-Punktwerte je
bereits gespeicherter Ziehung und fasst diese per KDE zusammen. Seine rwMH-
Quellparameter liefern vor Score-Deduplizierung 4000 Kandidaten: 5001 bis
24996 in Fünferschritten. Das entspricht nicht automatisch zehn ausgewählten
Transferdichten oder der im Text beschriebenen Kombination.

## Quantile, Originalausgaben und konkrete Grenzen

`final_plotting.R` interpoliert Achsen/Dichtewerte auf 10000 Knoten und benutzt
`DescTools::Quantile` für die sichtbaren Quartile und Medianlinien. Der
PANN-Tiefenmedian liest in der erhaltenen Zeile 84 jedoch
`quantiles_tann_age[3]`; der Alterszweig benutzt sein PANN-Quantil separat.
Dies ist ein enger Anzeigequellkonflikt, keine hier vorgenommene Reparatur
und kein Beweis, dass die archivierte Altersabbildung diesen Fehler enthält.
Die genaue Interpolations-/Quantilfassung gehört zur späteren Laufbindung.

Die erhaltene Kinneret-`reconstruction.pdf` ist weiterhin bytegleich zum
Archivmitglied `figures/kinneret/reconstruction.pdf`. `mcmc_samples.pdf` und
`mcmc_samples.png` sind ebenfalls vorhanden; die PNG wurde visuell geprüft:
Altersdarstellung jung links/alt rechts, TANN/PANN und mehrere gezeichnete
Kurven. Weder daraus noch aus der Dateibenennung werden ursprüngliche
Ziehungsnummern, Auswahlanzahl, Dichtemethode oder numerische Posteriorarrays
rekonstruiert. Der berechnete Paper-Biomfall bleibt eine separate Referenz.

[posterior_paths.csv](posterior_paths.csv) prüft acht genau aus den vorhandenen
Fallstudienzweigen abgeleitete Dateinamen für Slot 3 / Simulation 1:
rw/gibbs × Counts/Prozent × Hauptfall/Vergleich. Keiner ist am expliziten
Originalpfad vorhanden. Vorhandene Kataloge führen den Paper-Posterior separat;
er wird nicht als Dissertationsposterior geöffnet oder eingesetzt. Dies ist
keine neue Gesamtsuche und keine Behauptung, solche Dateien seien auf keinem
anderen Datenträger vorhanden.

## Ergebnis und nächster Phase-1-Schritt

Die Quellen zeigen mehrere nicht identische Verfahren. Für die finale
Dissertationsabbildung ist weder die literal gespeicherte Auswahlfassung noch
die konkret verwendete Dichtekombination vollständig laufgebunden. Die
bekannten Lücken aus Basis-, TF-, Referenz- und Klimaankerpaket bleiben offen.
Der Nutzer kennt keinen weiteren Basisdatei-Speicherort; keine erneute Frage
hierzu und keine neue pauschale Suche.

**Nächster sinnvoller Einstieg:** Die inzwischen gebundenen Kinneret-Eingabe-,
Referenz- und Auswertungsverträge zu einer maschinenprüfbaren Phase-1-
Reproduktionsspezifikation zusammenführen. Erhaltene Textverfahren und
Quellvarianten dabei getrennt kennzeichnen; vollständige Originaleingaben,
Quelldatenbindung und Auswahl-/Kombinationsfassung als offene Voraussetzungen
führen. Ein späterer Lauf darf fehlende Angaben weder durch Paperdaten noch
moderne Artenmodelle oder eine eigenmächtig korrigierte Auswahl ersetzen.
Keine neue Kalibrierung oder Reliabilitätsansicht vorziehen.

[collect_evidence.py](collect_evidence.py) erzeugt die eng begrenzten Nachweise:

```powershell
.venv/Scripts/python.exe docs/thesis_research/kinneret_postprocessing_2026_11_02/collect_evidence.py
```

[verification.json](verification.json) dokumentiert Wiederholungs- und
Erhaltungsprüfungen. Quellunterschiede bleiben erhalten, methodische Kritik
bleibt im [Dissertationsabgleich](../../abgleich_thesis.md) getrennt.
App/CLI, Baseline, Forschungsnachweise und Monticchio-/Chronologie-Gates
unverändert. Keine neuen Trainings oder Sampler; Phase 1 bleibt vorrangig.
