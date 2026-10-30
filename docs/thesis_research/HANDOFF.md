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
