# Netzel-Projekt: kurzer Arbeitsstand

Stand: 30.10.2026. Vor einer Fortsetzung PR-/Merge-Stand erneut prüfen.

**Ziel:** Eine App für taxonbasierte Klimarekonstruktionen europäischer
Fundstellen mit mehreren Proxys, nachvollziehbaren Quellen und Unsicherheiten.
Die moderne Analogmethode dient als Vergleich für die Hauptmethode.

| Bereich | Stand |
| --- | --- |
| Moderne Analogmethode | Fertig und an modernen Daten unabhängig geprüft, PR #18. Noch keine fossile Klimakurve berechnet. |
| Monticchio-Pilot | Vorbereitung, P1 offen. 69 Datenbank-Probenkennungen und exakte Tiefen bestätigt; physische Zuordnung zur neuen MON16-Chronologie samt Korrelationsunsicherheit fehlt. |
| Taxon-Transferfunktionen | Noch neu zu erstellen und zu prüfen; die ursprünglichen 279 Funktionen sind nicht wiederhergestellt. |
| Taxonbasiertes Multiproxy-Verfahren | Weiteres Entwicklungsziel; noch kein abgeschlossener Gesamtdurchlauf. |
| CLI/App | Claude betreut die Integration; OpenAI/Codex bearbeitet diese Dateien nicht gleichzeitig. |

**Aktuell bei OpenAI/Codex:** Ein echter gespeicherter Neotoma-Datenbestand
wurde gefunden und eingelesen. Datensatz **47525** enthält 924
Analyseeinheitsnamen. Für **28 der 69 Pilotproben** sind eindeutige numerische
Kandidaten mit Abschnittsbezeichnungen unabhängig geprüft; 41 bleiben ohne
vollständigen Treffer auf den 76 gemeinsam vergleichbaren Variablenschlüsseln.
Ihre 47 Zellabweichungen sind erhalten, einschließlich halbzähliger Werte.
Das ist noch kein bestätigter physischer LGM90D→MON16-Crosswalk.
[Neuer Archivbericht und Prüfung](./neotoma_workshop_2026_10_30/README.md),
[PR #26](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/26).

**Nächster konkreter Forschungsschritt:** Die 47 benannten Zellabweichungen
an ursprünglichen Variablen-/Aggregationsdefinitionen und Einreichungsdaten
klären. Daneben die nun greifbaren Abschnittsbezeichnungen mit den historischen
Kompositierungstabellen und danach mit MON16 samt Korrelationsunsicherheit
verbinden. Keine nachträgliche Toleranzlockerung, Quellenreparatur oder
unveränderte Wiederholung der gescheiterten API-/EPD-Abrufe. Notizen fehlen
weiter als Exportfelder; daraus keine leeren Datenbanktabellen ableiten.
P2 beginnt erst nach fachlicher P1-Abnahme bzw. ausdrücklich geänderter
Zieldefinition. Transferfunktionen bilden ein eigenes, noch nicht begonnenes
Arbeitspaket und können grundsätzlich parallel vorbereitet werden.

Zum Zeitpunkt dieser Aktualisierung waren PRs #18–#26 offen und ungemergt.
Arbeitsbasis ist weiterhin der Branch von #18. Keine automatische Freigabe
des Fossilpiloten, keine stillen Quellenkorrekturen, Autorennachrichten oder
Merges. Das Original-ScienceDirect-ZIP ist für erneute Originaldateiprüfungen
wieder anzuhängen.

Die [Kommunikationsregel](../../AGENTS.md) verlangt zu Beginn einer Fortsetzung
und nach einem Arbeitspaket einen knappen Überblick über fertig / aktuell /
als Nächstes. Dies ist eine dokumentierte Arbeitsregel, keine Hintergrundaufgabe.
