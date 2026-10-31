# Netzel-Projekt: nächster Einstieg

## Zuerst lesen und Stand prüfen

1. [Dauerhafte Arbeitsregeln](../../AGENTS.md) und [aktuellen Status](./PROJECT_STATUS.md)
   lesen. Git-Status, Branches, Worktrees und GitHub-PRs erneut prüfen; keine Merges.
2. Vollständiger gesicherter Forschungsstand: `codex/netzel-calibration-block-uncertainty`
   bei `9ee2f84e41a04fafe5af203e809deeaaf329dba3` einschließlich #31/#32.
   Am 31.10.2026 sind beide PRs offen, `main` enthält sie nicht. Der [Dokumentations-PR #33](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/33)
   (`codex/netzel-workflow-handoff`) baut darauf auf; Ergebnisse daraus nicht nochmals
   nach main übertragen oder als schon gemergt behandeln.
3. Die [Original-Namensprüfung](./original_pollen_names_2026_10_31/README.md)
   ist abgeschlossen; die morphologische/fossile Artbrücke bleibt offen.
   Für den nächsten Auftrag nur deren
   [SOURCE_GAP.md](./original_pollen_names_2026_10_31/SOURCE_GAP.md) und bei
   konkreten Namensfragen die sieben Zeilen von data/species_name_links.csv lesen.
   [Originalquellen-PR #35](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/35):
   Branch `codex/netzel-original-pollen-names` ab `98ef1bf` (#34);
   #34 bleibt die vorangegangene Inventur. Kein Merge beauftragt.

## Nächster Forschungsauftrag: Bestimmungsmethoden und Typdefinitionen

Gezielt die Methoden der im Originalarchiv genannten Publikation Watts,
Allen & Huntley (1996), DOI 10.1016/0277-3791(95)00093-3, sowie deren tatsächlich
zitierte Bestimmungsreferenzen prüfen. Je Modellart festhalten, ob ursprünglicher
Bestimmungsumfang, nur allgemeine Methodik oder kein datensatzbezogener Beleg
vorliegt. Danach artspezifische palynologische Belege mit exakter Fundstelle
für Grenzen und Überschneidungen der Typen nachladen.

Besonders beachten: Monticchio-Kommentare sind akzeptierte EPD-Namen. EMPD2
ordnet Betula pendula dem pubescens-type zu; bei Taxus gibt es zwei verschieden
akzeptierte Einträge mit demselben Originalnamen. Diese Datenbankbefunde nicht
als botanische Synonymie oder als Freigabe einer Artfunktion interpretieren.

Abschlusskriterium: belegte Methodenkette und kompakte Evidenztabelle für alle
sieben Arten; unterstützende, widersprechende und fehlende Belege vollständig
mit Seiten/Abschnitten/Objektkennungen sichern, Zugangslücken konkret benennen,
eigener PR. Nur gelesene Quellen als geprüft ausweisen. Keine automatische
Artzuordnung, Modelle/Fossilcounts/Gruppierungen unverändert; keine Auswahl
anhand äußerer Scores. Claude behält CLI/App; Analog-Baseline unverändert.

Der frühere Netzwerkblock für die beiden Originaldateien ist behoben.
Neue Originalprüfung benötigt keine alten Chat-Dateien; Bezugsbefehle und
Hashprüfungen in original_pollen_names_2026_10_31/REPRODUCE.md.
Die 2025-Originale bei konkretem Bedarf gesondert prüfen, keine Änderung der
Shared37-Referenz. Monticchio-P1 und weitere Arten bleiben getrennte Pakete.

## Übergabe an einen neuen Chat

Das Originalquellenpaket ist ein abgeschlossener, gesicherter Wechselpunkt.
Startauftrag: „Prüfe aktuellen Git-/PR-Stand, verwende den Forschungsstand auf
`codex/netzel-original-pollen-names` (#35), lies AGENTS.md und HANDOFF.md.
Bearbeite das dort begrenzte Paket zu Bestimmungsmethoden und Typdefinitionen;
Modelle/Counts/Baseline/CLI/App unverändert, eigener PR, kein Merge.“
Keine ungemergten Ergebnisse als schon auf main vorhanden behandeln.

## Nur bei konkretem Bedarf weiterlesen

| Anlass | Nachzuladen |
| --- | --- |
| #32-Bänder, Schätzbarkeit oder Aussagegrenzen prüfen | [Bericht](./calibration_uncertainty_2026_10_30/README.md), danach bei Methodenarbeit DESIGN.md und SOURCES.md desselben Ordners |
| Kalibrierung #31 oder SMOTE #29 prüfen | [#31-Bericht](./calibration_2026_10_30/README.md), [#29-Bericht](./smote_nnet_2026_10_30/README.md); jeweiliges Vorabdesign vor Methodenänderungen |
| Große Ausgaben benötigen | [Artefaktverzeichnis](./ARTIFACTS.md), erst dann dort verlinkte REPRODUCE.md; keine alten temporären Dateien voraussetzen |
| Monticchio-Quellenarbeit | [präziser Quellenbedarf](./variable_provenance_2026_10_30/SOURCE_GAP.md); physischer LGM90D→MON16-Anschluss bleibt separat blockiert |
| Historische Sicherung oder frühere Aufträge klären | [Chat-Archiv](./CHAT_WORK_ARCHIVE_2026_10_30.md), [JSON-Inventar](./CHAT_WORK_ARCHIVE_2026_10_30.json), [frühere Einstiegstexte](./ENTRYPOINT_ARCHIVE_2026_10_31.md), [Integration #18–#29](./consolidation_2026_10_30/README.md) |

Archive bleiben erhalten, sind aber keine Pflichtlektüre bei jedem Start.
