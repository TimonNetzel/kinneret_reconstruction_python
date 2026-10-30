# Netzel-Projekt: gemeinsamer Einstieg

Stand 30.10.2026. Nutzerauftrag: alle bisherigen Arbeiten hochladen und mergen,
damit die Ergebnisse aus dem Chat erhalten bleiben. PR #18 ist gemergt;
der gemeinsame Integrationsstand führt #19–#29 mit vollständiger Historie
zusammen. Vor Fortsetzung den aktuellen PR-/Merge-Stand prüfen und `main`
verwenden. [Integrationsnachweis und PR-Index](./consolidation_2026_10_30/README.md).

## Fertig

- Moderne Analog-Baseline #18 und Claudes CLI-/App-Integration #19.
- Quellen-/Chronologieprüfungen #20–#27 mit Berichten, gesperrten Quellen,
  Ergebnisdateien, Prüfwerkzeugen und unabhängigen Auditnachweisen.
- Sieben experimentelle quadratische Artenfunktionen #28 sowie je sieben
  einfache Netze ohne und mit SMOTE #29, mit Parametern und räumlicher Prüfung.
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
Wahrscheinlichkeitskalibrierung und Kalibrierungsunsicherheit bleiben offen.

Monticchio-P1 bleibt blockiert: physischer LGM90D→MON16-Anschluss einschließlich
Korrelationsunsicherheit fehlt. Unverändert 28 numerische Kandidaten,
41 native Nichttreffer und 47 offene Provenienzfragen. Die 35 Plantago-Bilanzen
und neun Scrophulariaceae-/Digitalis-Zahlengleichheiten sind kein belegter
Umcodierungsnachweis. Keine stille Quellenkorrektur oder zusätzliche Freigabe.

## Als Nächstes

1. Kalibrierungsdesign vorab festlegen: innere räumliche Teilungen, sämtliche
   Transformationen/SMOTE/Kalibrierung ausschließlich trainingsintern;
   unabhängige äußere Bewertung. Äußere #29-Testergebnisse nicht zur weiteren
   Optimierung verwenden. Die wiederverwendeten Gebiete sind bereits untersucht;
   eine neue Auswertung nicht als vollständig ungesehene Bestätigung ausgeben.
2. Weitere Areale/Arten und belegte Fossiltyp-Zuordnung, anschließend Unsicherheit.
3. Parallel Originaleinreichung/Umcodierungsbelege und physischen MON16-Anschluss
   beschaffen: [präziser Quellenbedarf](./variable_provenance_2026_10_30/SOURCE_GAP.md).

Claude betreut CLI/App; Codex Kalibrierung, Quellen und unabhängige Prüfung.
Keine automatische Pilotfreigabe oder Autorennachricht. Technischer Merge ist
vom Nutzer ausdrücklich beauftragt, keine wissenschaftliche Abnahme der
Fossilrekonstruktion. Die Analog-Baseline wird nicht neu entwickelt.

Historische Übergaben bleiben im Repository und an ihren ursprünglichen
Commit-Refs erhalten. Einstieg in das letzte Forschungspaket:
[CHAT_RESTART_SMOTE.md](./CHAT_RESTART_SMOTE.md),
[Ergebnisbericht](./smote_nnet_2026_10_30/README.md),
[PROJECT_STATUS.md](./PROJECT_STATUS.md).
Roharchive/Originalpublikationen werden durch die Zusammenführung nicht neu
weiterverteilt. Das ursprüngliche ScienceDirect-ZIP muss nur bei erneuter
Original-Rohdatenprüfung wieder angehängt werden; Quellenprüfsummen und
Reproduktionswege sind gespeichert.
