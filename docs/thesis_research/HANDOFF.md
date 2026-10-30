# Netzel-Projekt: gemeinsamer Einstieg

Stand 30.10.2026. Nutzerauftrag: alle bisherigen Arbeiten hochladen und mergen,
damit die Ergebnisse aus dem Chat erhalten bleiben. PR #18 ist gemergt;
[Integrations-PR #30](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/30)
hat #19–#29 mit vollständiger Historie zusammengeführt. Vor Fortsetzung den
aktuellen PR-/Merge-Stand prüfen. Die Kalibrierung #31 basiert auf diesem main;
solange #31 offen ist, liegen ihre neuen Ergebnisse auf dem Forschungsbranch
`codex/netzel-spatial-probability-calibration`, noch nicht auf main.
[Integrationsnachweis und PR-Index](./consolidation_2026_10_30/README.md).

## Fertig

- Moderne Analog-Baseline #18 und Claudes CLI-/App-Integration #19.
- Quellen-/Chronologieprüfungen #20–#27 mit Berichten, gesperrten Quellen,
  Ergebnisdateien, Prüfwerkzeugen und unabhängigen Auditnachweisen.
- Sieben experimentelle quadratische Artenfunktionen #28 sowie je sieben
  einfache Netze ohne und mit SMOTE #29, mit Parametern und räumlicher Prüfung.
- Räumlich verschachtelte Wahrscheinlichkeitskalibrierung #31 vorab entworfen,
  berechnet und separat numerisch geprüft; [neuer Bericht](./calibration_2026_10_30/README.md).
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
Neue PRs einschließlich #31 nur nach erneutem ausdrücklichem Auftrag mergen. Die Analog-Baseline wird nicht neu entwickelt.

Historische Übergaben bleiben im Repository und an ihren ursprünglichen
Commit-Refs erhalten. Aktueller Einstieg: [CHAT_RESTART_CALIBRATION.md](./CHAT_RESTART_CALIBRATION.md).
Historischer Einstieg in den SMOTE-Vergleich:
[CHAT_RESTART_SMOTE.md](./CHAT_RESTART_SMOTE.md),
[Ergebnisbericht](./smote_nnet_2026_10_30/README.md),
[PROJECT_STATUS.md](./PROJECT_STATUS.md).
Roharchive/Originalpublikationen werden durch die Zusammenführung nicht neu
weiterverteilt. Das ursprüngliche ScienceDirect-ZIP muss nur bei erneuter
Original-Rohdatenprüfung wieder angehängt werden; Quellenprüfsummen und
Reproduktionswege sind gespeichert.
