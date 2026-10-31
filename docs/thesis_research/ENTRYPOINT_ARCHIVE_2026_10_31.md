# Archiv der bisherigen Einstiegsdokumente

Unveränderte Texte aus Forschungsstand `9ee2f84e41a04fafe5af203e809deeaaf329dba3`,
am 31.10.2026 vor der Neuordnung gesichert. Historische Statusangaben sind keine
aktuelle PR-Abfrage. Nur bei konkretem historischen Bedarf lesen. Die Texte
bleiben vollständig erhalten; Links der früheren Root-AGENTS wurden relativ angepasst.


## Original: AGENTS.md

# Kommunikation und Fortsetzung des Netzel-Projekts

Vom Nutzer am 30.10.2026 gewünscht; aus PR #26/#27 übernommen.

- Vor fachlicher Weiterarbeit PR-/Merge-Stand und
  [PROJECT_STATUS.md](./PROJECT_STATUS.md) prüfen.
- Zu Beginn und nach Arbeitspaketen kurz auf Deutsch berichten:
  **fertig / aktuell / als Nächstes**, mit Bezug zur taxonbasierten Multiproxy-App.
- Statusdatei mit Ergebnis und PR-Verweis fortschreiben; begonnen, blockiert
  und abgeschlossen unterscheiden. Fertige Analog-Baseline nicht als fertige
  taxonbasierte Hauptmethode oder Fossilrekonstruktion bezeichnen.
- Claude betreut CLI/App; Codex Kalibrierung, Quellen und unabhängige Prüfung.
  Bestehende Baseline und Claudes Integrationsdateien unverändert lassen.
- Keine stillen Quellenkorrekturen, automatischen Pilotfreigaben,
  Autorennachrichten. Merges nur mit ausdrücklichem Nutzerauftrag; die
  Zusammenführung von #18–#29 ist am 30.10.2026 ausdrücklich beauftragt.
  Diese technische Zusammenführung ist keine fachliche Pilotfreigabe.

Der Nutzer hat nach #27 die eigenständige Erstellung neuer Taxon-
Transferfunktionen parallel zur offenen Chronologieprüfung beauftragt.
Der Nutzer hat anschließend den Vergleich SMOTE + einfaches Netz nach seiner
Dissertation beauftragt. Aktueller Einstieg:
[CHAT_RESTART_SMOTE.md](./CHAT_RESTART_SMOTE.md).
Nach der beauftragten Zusammenführung `main` verwenden. #28 bleibt am
gesperrten Commit die wissenschaftliche Referenz für Daten und bisherige Modelle.
Aktuelle gemeinsame Übergabe: [HANDOFF.md](./HANDOFF.md).
Frühere getrennte Forschungsstände:
[PR #27](https://github.com/TimonNetzel/kinneret_reconstruction_python/blob/5fd3bad46d5108e8986c054171fba66a05c26589/docs/thesis_research/CHAT_RESTART_VARIABLES.md).


## Original: docs/thesis_research/PROJECT_STATUS.md

# Netzel-Projekt: kurzer Arbeitsstand

**Speicherprüfung 30.10.2026:** [Sicherungsnachweis der Chat-Arbeit](./CHAT_WORK_ARCHIVE_2026_10_30.md)
mit Dateiinventar und festen Ergebniscommits für #31/#32. Code, Modelle,
Parameter und Berichte liegen im Repository; vier große Laufausgaben sind bisher
nur durch Hashes und Reproduktionsbefehle gesichert. Das vollständige Bytearchiv
dieser vier Dateien bleibt offen. PRs weiterhin ungemergt.

Stand 30.10.2026. Vor Fortsetzung aktuellen PR-/Merge-Stand prüfen.
Ziel: überprüfbare App für taxonbasierte Multiproxy-Klimarekonstruktionen
europäischer Fundstellen, mit belegter Kalibrierung und Unsicherheiten.

| Bereich | Stand |
| --- | --- |
| Moderne Analog-Baseline #18 | Fertig; #18 auf main gemergt. Unveränderte Referenz f9acc16. |
| Erste sieben Artenfunktionen #28 | Fertiger experimenteller Trainingsstand aus Meusel/BfN + WorldClim 2.1; feste Referenz. |
| SMOTE + einfaches Netz #29 | **Berechnet und unabhängig geprüft:** sieben Arten, identische äußere räumliche Tests; deutlich bessere Balanced Accuracy, zugleich schlechtere rohe SMOTE-Wahrscheinlichkeiten im Brier. |
| Historische 279 Funktionen | Nicht wiederhergestellt; keine bitgleiche historische R-/caret-Reproduktion. |
| Räumliche Wahrscheinlichkeitskalibrierung #31 | **Berechnet und separat numerisch geprüft:** innere räumliche Kalibrierungsfolds, unabhängige äußere Bewertung je Fit; SMOTE-Brier verbessert, gegenüber rohem Netzensemble ohne SMOTE in 13/14 Fällen weiter schlechter. |
| Bedingte Kalibrierungssensitivität #32 | **Berechnet und separat geprüft:** 200 räumliche Block-Replikate bei festen Netzen; Band für 22/28 Vollgebiets-Ensembles, 104/140 äußere Ensemble-Folds. Keine nominell validierte Deckung. |
| Taxonverfahren insgesamt | Weitere Taxa, belegte Fossiltyp-Zuordnung, externe Bestätigung und Kalibrierungs-/Quellenunsicherheit offen. |
| Monticchio-P1 | Weiter offen: physischer LGM90D→MON16-Crosswalk samt Korrelationsunsicherheit fehlt. |
| Proben und Quellen aus #26/#27 | Unverändert: 28 numerische Kandidaten, 41 native Nichttreffer, 47 offene Herkunftsfragen. |
| Fossilrekonstruktion / Multiproxy-App | Kein fertiger Gesamtdurchlauf, keine neue fossile Klimakurve und keine P2-Freigabe. Claude betreut CLI/App; Integration #19 in den gemeinsamen Stand übernommen. |

**Aktueller Abschluss #32:** [Bedingte Kalibrierungssensitivität](./calibration_uncertainty_2026_10_30/README.md),
[PR #32](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/32),
Branch `codex/netzel-calibration-block-uncertainty`, abhängig von offenem #31 am
festen Head `9458011`; PR-Ziel ist dessen Forschungsbranch. Kein Mergeauftrag.
Vorabdesign `5a4a96f`, unverändert. 200 gepaarte geographische Block-Replikate,
nur innere Original-Kalibrierungszellen neu gewichtet, Netze/Normierung/SMOTE fest.
Äußere Labels nicht für Bänder oder Verfügbarkeitsauswahl verwendet.

22/28 Vollgebiets-Ensembles und 104/140 äußere Ensemble-Folds mit bedingtem
Sensitivitätsband; Abies in beiden Designs und Fagus 10° ohne Vollgebiets-Band.
Gate 190/200 schätzbare Replikate unverändert; fehlende Bänder sind keine Nullbreite.
95.876 Kalibrierungen / insgesamt 100.800 Mitglied×Replikat-Fälle, 1.589.440
Zellzeilen, 2.174.335 Blockmultiplizitäten und alle 168 Tabellenzeilen separat
geprüft. Zwei frische Endläufe: sieben bytegleiche Dateien, ohne Wiederherstellung.
Vorige I/O-Befunde und exakte Hülleneckpunkt-Korrektur ausdrücklich dokumentiert.
85 Repositorytests und sechs gezielte Tests bestanden; Browser-/Langtests ausgenommen.

**Als Nächstes:** belegte Pollentyp-Zuordnung und weitere Arten/Areale.
Vollständige Kalibrierungs-/Quellen-/Netzunsicherheit und Kreuztaxon-Kovarianz
bleiben offen. Keine nominell validierten 95%-Konfidenzintervalle, unabhängige
neue Region oder Fossilfreigabe. Monticchio-P1 und 28/41/47 unverändert.
Aktuelle Fortsetzung: [CHAT_RESTART_UNCERTAINTY.md](./CHAT_RESTART_UNCERTAINTY.md).
#31/#32 bleiben offen; Merges nur nach erneutem ausdrücklichem Auftrag.

## Historischer Abschluss #31 vor der Sensitivitätsanalyse

**Abschluss #31:** [Kalibrierungsbericht](./calibration_2026_10_30/README.md),
[PR #31](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/31),
Branch `codex/netzel-spatial-probability-calibration`, Basis aktuelles main nach
#30 (`7f1826c`). Design vor jedem Fit veröffentlicht in `afa0c8a` und unverändert.
504 Netzfits / 1.512 Starts; 502 geschätzte und zwei vorab deklarierte Identitäts-
Kalibratoren. 198.680 Labels, 397.360 Testzeilen und 2.279.093 synthetische
Elternpaare separat geprüft. Kleinster innerer/äußerer Abstand 100,093 km.

Artenmittel Brier 5°/10°: SMOTE roh 0,07129/0,07699 → kalibriert
0,03995/0,04408; Verbesserung in allen 14 Fällen. Rohes Ensemble ohne SMOTE
0,03939/0,04321, besser als kalibriertes SMOTE in 13/14 Fällen. Ohne SMOTE
verbessert Kalibrierung nur 3/14 Fälle und verschlechtert die Mittelwerte leicht.
Vier Varianten vollständig berichtet; keine automatische Variantenübernahme.
Verglichen werden dieselben drei Mitglieder je roh/kalibriert, keine direkten
Einzelfit-Ergebnisse aus #29. Bereits untersuchte Regionen sind keine vollständig
ungesehene Bestätigung; äußere #29/#31-Werte nicht zur Optimierung verwenden.

Elf Enddateien zweier vollständiger Trainingsausführungen bytegleich nach
explizit dokumentierter Wiederherstellung des letzten Checkpoint-Eintrags aus
der vollständigen Modellkopie desselben Laufs gegen den vorher gespeicherten
Manifest-Hash. Ein Lauf mit unvollständiger Gzip-Spur wurde verworfen;
[Ausführungshinweise](./calibration_2026_10_30/execution_notes.json).
79 Repositorytests und fünf neue Forschungstests bestanden; zwei Browser-/zwei
Langtests nicht ausgeführt. Reproduktion ohne alte Zwischendateien.

**Als Nächstes:** belegte Pollentyp-Zuordnung, weitere Taxa und Kalibrierungs-/
Quellenunsicherheit bearbeiten. Kein fossiler Gesamtdurchlauf oder Pilotfreigabe.
Claude behält CLI/App. Monticchio-P1 und 28/41/47 bleiben unverändert offen.
#31 bleibt offen und wird nur auf erneuten ausdrücklichen Auftrag gemergt.
Aktuelle Fortsetzung: [CHAT_RESTART_CALIBRATION.md](./CHAT_RESTART_CALIBRATION.md).

## Historischer Stand nach #29, vor Kalibrierung

**Fertig:** [SMOTE-/Netzvergleich](./smote_nnet_2026_10_30/README.md),
[PR #29](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/29),
ursprünglicher Ergebnisbranch `codex/netzel-smote-nnet-comparison`.
Die Ergebnisse aus #18–#29 werden auf ausdrücklichen Nutzerauftrag gemeinsam
auf `main` gesichert ([Integrations-PR #30](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/30)); [gemeinsame Übergabe](./HANDOFF.md) und
[Integrationsnachweis](./consolidation_2026_10_30/README.md). Vorabdesign veröffentlicht
in `d47129c6b8be8bd5e73701f6137ffb196dab9d91`, bevor neuronale Fits begannen.

Artenmittel der Balanced Accuracy bei Schwelle 0,5: ohne SMOTE 71,64/69,20 %,
mit SMOTE 90,41/89,74 % (5°/10°-Blöcke, 100-km-Puffer). Gewinn in allen 14
Art×Block-Fällen. Flächengewichteter Brier dagegen im Artenmittel 0,03975/0,04438
ohne SMOTE und 0,07747/0,08245 mit SMOTE: Verschlechterung in allen 14 Fällen.
Die vorab vorgesehene Prior-Odds-Diagnose hilft, garantiert aber keine Kalibrierung
und schlägt das Netz ohne SMOTE im Brier nur in zwei von 14 Fällen.

14 vollständige neue Artennetze und alle Foldparameter gespeichert.
198.680 Art-Zell-Labels, 397.360 Testzeilen mit 1.589.440 Wahrscheinlichkeiten,
1.274.548 synthetische Elternpaare und sämtliche 462 Optimierungsstarts separat
geprüft. Zwei vollständige Läufe in allen zehn Ergebnisdateien bytegleich.
Sechs neue Kerntests bestanden. Gemeinsame GIS-/Numerikbibliotheken benannt.

**Aktuell:** Die Klassifikationsverbesserung aus der Dissertation wurde auf den
neuen sieben Arten nachgebildet. Experimentelle Arealindikatoren sind noch
keine validierten fossilen Auftretenswahrscheinlichkeiten. Keine automatische
Variantenübernahme oder Pollentyp-Zuordnung. Die #28-Referenz wird ausdrücklich
am gesperrten Commit `8f02794434f561f5b30228360247935f3c025cc8` bezogen; die ursprünglichen getrennten
PR-Branches enthielten nicht automatisch alle Vorarbeiten. Der gemeinsame
Integrationsstand enthält jetzt alle zwölf PR-Historien.

**Als Nächstes:** neue trainingsinterne Wahrscheinlichkeitskalibrierung mit inneren
räumlichen Teilungen und unabhängiger äußerer Bewertung vorab festlegen; weitere
Arten/Areale und eine belegte Typ–Art-/Gattungs-Zuordnung bearbeiten. Äußere
Testergebnisse dieses PR nicht zur weiteren Optimierung verwenden. Moderne
Kalibrierung kann parallel zur weiterhin offenen Monticchio-Quellenfrage laufen.

Die frühere Merge-Sperre wurde für die Zusammenführung #18–#29 durch den
aktuellen ausdrücklichen Nutzerauftrag aufgehoben. Das ändert keine fachlichen
Freigaben: keine stillen Quellenkorrekturen, Pilotfreigaben oder Autorennachrichten.
Wissenschaftliche Berichte, Quellenregister, Modellparameter und Ergebniswerte
werden unverändert erhalten. Der gemeinsame Einstieg ist [HANDOFF.md](./HANDOFF.md).


## Original: docs/thesis_research/HANDOFF.md

# Netzel-Projekt: gemeinsamer Einstieg

**Speicherprüfung 30.10.2026:** [Sicherungsnachweis der Chat-Arbeit](./CHAT_WORK_ARCHIVE_2026_10_30.md)
mit Dateiinventar und festen Ergebniscommits für #31/#32. Code, Modelle,
Parameter und Berichte liegen im Repository; vier große Laufausgaben sind bisher
nur durch Hashes und Reproduktionsbefehle gesichert. Das vollständige Bytearchiv
dieser vier Dateien bleibt offen. PRs weiterhin ungemergt.

Stand 30.10.2026. Nutzerauftrag: alle bisherigen Arbeiten hochladen und mergen,
damit die Ergebnisse aus dem Chat erhalten bleiben. PR #18 ist gemergt;
[Integrations-PR #30](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/30)
hat #19–#29 mit vollständiger Historie zusammengeführt. Vor Fortsetzung den
aktuellen PR-/Merge-Stand prüfen. Die Kalibrierung #31 basiert auf diesem main;
solange #31 offen ist, liegen ihre neuen Ergebnisse auf dem Forschungsbranch
`codex/netzel-spatial-probability-calibration`, noch nicht auf main.
Neueste Folgearbeit: #32 auf `codex/netzel-calibration-block-uncertainty`,
abhängig von offenem #31. Solange beide offen sind, diesen Forschungsbranch
als vollständigen Ergebnisstand verwenden; keine neuen Merges ohne Auftrag.
[Integrationsnachweis und PR-Index](./consolidation_2026_10_30/README.md).

## Fertig

- Moderne Analog-Baseline #18 und Claudes CLI-/App-Integration #19.
- Quellen-/Chronologieprüfungen #20–#27 mit Berichten, gesperrten Quellen,
  Ergebnisdateien, Prüfwerkzeugen und unabhängigen Auditnachweisen.
- Sieben experimentelle quadratische Artenfunktionen #28 sowie je sieben
  einfache Netze ohne und mit SMOTE #29, mit Parametern und räumlicher Prüfung.
- Räumlich verschachtelte Wahrscheinlichkeitskalibrierung #31 vorab entworfen,
  berechnet und separat numerisch geprüft; [neuer Bericht](./calibration_2026_10_30/README.md).
- Bedingte räumliche Kalibrierungssensitivität #32 berechnet und separat geprüft;
  [Bericht](./calibration_uncertainty_2026_10_30/README.md), [aktuelle Fortsetzung](./CHAT_RESTART_UNCERTAINTY.md).
- Reproduktion der modernen Kalibrierung ohne alte Chat-Zwischendateien;
  [SMOTE-Anleitung](./smote_nnet_2026_10_30/REPRODUCE.md).

## Fachliche Bewertung und Grenzen

Mit dem nachvollziehbaren Vergleich sind wir als Forschungszwischenstand
zufrieden: Bei allen sieben Arten und beiden Blockgrößen verbessert SMOTE
gegenüber demselben Netz ohne SMOTE die Balanced Accuracy bei Schwelle 0,5.
Artenmittel: 71,64/69,20 % ohne und 90,41/89,74 % mit SMOTE (5°/10°).
Gleichzeitig verschlechtert sich überall der Brier-Score: 0,03975/0,04438 ohne
gegenüber 0,07747/0,08245 mit SMOTE. Das Netz ohne SMOTE schlägt die feste
quadratische Regression im Brier in 13 von 14 Art×Block-Fällen.

Die Verbesserung betrifft diese Klassifikationsprüfung mit fester Schwelle;
sie weist keine universelle Überlegenheit von SMOTE und keine fossil gültigen
Wahrscheinlichkeiten nach. Keine bitgleiche Wiederherstellung der historischen
279 Taxonfunktionen. Weitere Arten, belegte Pollentyp–Art-/Gattungs-Zuordnung,
externe Bestätigung und Kalibrierungs-/Quellenunsicherheit bleiben offen.

**Neuer Stand #31:** Drei räumliche Netz-/Kalibrator-Paare je äußerem Fold;
Transformationen/SMOTE nur im inneren Originaltraining, Kalibrierung nur auf
zurückgehaltenen Originalzellen. 100-km-Puffer innen und außen. Vorabdesign
`afa0c8a`, keine Auswahl nach äußeren Scores. SMOTE-Brier verbessert in 14/14
Fällen auf 0,03995/0,04408 (5°/10°); das rohe Ensemble ohne SMOTE ist in 13/14
Fällen besser (Mittel 0,03939/0,04321). Ohne SMOTE hilft die Kalibrierung nur
in 3/14 Fällen. Keine automatische Variantenübernahme. Bereits untersuchte
Regionen sind keine vollständig ungesehene Bestätigung.

504 Netzfits / 1.512 Starts, 502 Kalibratorlösungen und zwei deklarierte
Identitäten, sämtliche Vorhersagen und Elternspuren geprüft. Elf Endausgaben
zweier vollständiger Trainingsausführungen bytegleich nach dokumentierter
Wiederherstellung des letzten Checkpoint-Eintrags aus der vollständigen
Modellkopie desselben Laufs gegen den vorhandenen Manifest-Hash; ein anderer
Lauf mit unvollständiger Gzip-Spur verworfen. Keine Modellanpassungen deshalb.
Details und Grenzen: [Kalibrierungsübergabe](./CHAT_RESTART_CALIBRATION.md).

**Neuer Stand #32:** 200 gepaarte Block-Replikate bei unveränderten Netzen,
Normierungen und SMOTE; ausschließlich innere Kalibrierungszellen neu gewichtet.
Band nur ab 190/200 schätzbaren Replikaten: 22/28 Vollgebiets-Ensembles und
104/140 äußere Ensemble-Folds. Abies in beiden Designs und Fagus 10° ohne
Vollgebiets-Band. Fehlende Bänder werden nicht als Nullunsicherheit ausgegeben.
Es sind bedingte Bootstrap-Sensitivitätsbänder, keine nominell validierten
95%-Konfidenzintervalle oder vollständige Rekonstruktionsunsicherheit.

95.876 Kalibrierungen / 100.800 Mitglied×Replikat-Fälle, 1.589.440 Zellzeilen
und alle Blockmultiplizitäten/Quantile separat geprüft. Zwei frische Endläufe
mit sieben bytegleichen Dateien, keine Wiederherstellung erforderlich. Vorige
I/O-Fehler verworfen und dokumentiert. Exakte gespeicherte Hülleneckpunkte
werden im neuen Evaluator trotz konservativem Qhull-Randfehler eingeschlossen;
kein größerer Extrapolationspuffer, #31-Quellen und Parameter unverändert.
85 Repositorytests und sechs gezielte Tests bestanden. Details im neuen Bericht.

Monticchio-P1 bleibt blockiert: physischer LGM90D→MON16-Anschluss einschließlich
Korrelationsunsicherheit fehlt. Unverändert 28 numerische Kandidaten,
41 native Nichttreffer und 47 offene Provenienzfragen. Die 35 Plantago-Bilanzen
und neun Scrophulariaceae-/Digitalis-Zahlengleichheiten sind kein belegter
Umcodierungsnachweis. Keine stille Quellenkorrektur oder zusätzliche Freigabe.

## Als Nächstes

1. Weitere Areale/Arten, belegte Fossiltyp-Zuordnung und Kalibrierungs-/Quellen-
   unsicherheit bearbeiten. Ensemblevariation ist kein Unsicherheitsintervall.
2. Äußere #29/#31-Ergebnisse nicht zur weiteren Optimierung verwenden;
   für eine bestätigende Bewertung neue unabhängige Evidenz planen.
3. Parallel Originaleinreichung/Umcodierungsbelege und physischen MON16-Anschluss
   beschaffen: [präziser Quellenbedarf](./variable_provenance_2026_10_30/SOURCE_GAP.md).

Claude betreut CLI/App; Codex Kalibrierung, Quellen und unabhängige Prüfung.
Keine automatische Pilotfreigabe oder Autorennachricht. Die technische Zusammenführung #18–#29 über #30 war
ausdrücklich beauftragt, keine wissenschaftliche Abnahme der Fossilrekonstruktion.
Neue PRs einschließlich #31/#32 nur nach erneutem ausdrücklichem Auftrag mergen. Die Analog-Baseline wird nicht neu entwickelt.

Historische Übergaben bleiben im Repository und an ihren ursprünglichen
Commit-Refs erhalten. Aktueller Einstieg: [CHAT_RESTART_UNCERTAINTY.md](./CHAT_RESTART_UNCERTAINTY.md).
Vorige Kalibrierungsübergabe: [CHAT_RESTART_CALIBRATION.md](./CHAT_RESTART_CALIBRATION.md).
Historischer Einstieg in den SMOTE-Vergleich:
[CHAT_RESTART_SMOTE.md](./CHAT_RESTART_SMOTE.md),
[Ergebnisbericht](./smote_nnet_2026_10_30/README.md),
[PROJECT_STATUS.md](./PROJECT_STATUS.md).
Roharchive/Originalpublikationen werden durch die Zusammenführung nicht neu
weiterverteilt. Das ursprüngliche ScienceDirect-ZIP muss nur bei erneuter
Original-Rohdatenprüfung wieder angehängt werden; Quellenprüfsummen und
Reproduktionswege sind gespeichert.
