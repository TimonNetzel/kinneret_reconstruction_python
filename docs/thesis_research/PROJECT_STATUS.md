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

**Aktuell bei OpenAI/Codex:** Quellenbelege für die physische Zuordnung
LGM90D→MON16 prüfen. Für Neotoma-Pollendatensatz **47525** wurde der benötigte
Metadatenauszug anhand offizieller Tabellen-/Exportdefinitionen präzisiert.
Eine nur lesende Abfrage erhält Proben-/Analyseeinheitsnamen und Notizen;
drei Tests an künstlichen Daten bestanden. Die gelesenen Codeversionen sind
kein nachgewiesener gemeinsamer Serverstand. Es wurden weiterhin keine
echten Probenmetadaten aus 47525 bezogen. Neuer
[Metadatenbericht und Abfrage](./neotoma_metadata_path_2026_10_30/README.md),
vorheriger [Quellenbericht](./neotoma47525_source_lead_2026_10_30/README.md),
[PR #26](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/26).

**Nächster konkreter Forschungsschritt:** Den gezielt spezifizierten
Metadatenauszug für 47525 zusammen mit den Zählungen als echten Eingang
beschaffen. JSON-Namensfelder allein sind kein vollständiger Notizenauszug.
Die neue Frozen-Route war ebenfalls nicht lesbar; kein unveränderter weiterer
API-/EPD-Suchlauf. Bis zu einem neuen Eingang bleibt P1 blockiert. Falls der
Beleg nicht beschaffbar ist, die vorgesehene Pilotentscheidung vorbereiten.
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
