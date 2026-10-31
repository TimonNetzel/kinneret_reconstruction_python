# Netzel-Projekt: nächster Einstieg

Gesamtziel und Weg zum ersten diagnostischen lokalen Kernlauf stehen in der
[Taxon-Pilot-Roadmap](./TAXON_PILOT_ROADMAP.md). Aktuell ist das
[Literaturpaket zu sieben Bestimmungsumfängen](./pollen_methods_2026_10_31/README.md)
in [PR #37](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/37)
auf `codex/netzel-pollen-methods` ab `f7e5290` (#36) gesichert; Ziel ist der
#36-Branch, kein Merge.
Morphologische Referenzbefunde liegen vor; **W96s tatsächliche Methodik und
Bestimmungsreferenzen sind wegen fehlenden Volltextzugangs weiterhin offen**.
Keine fossile Artbrücke oder T0-Freigabe.

## Zuerst lesen und Stand prüfen

1. [Dauerhafte Arbeitsregeln](../../AGENTS.md) und [aktuellen Status](./PROJECT_STATUS.md)
   lesen. Git-Status, Branches, Worktrees und GitHub-PRs erneut prüfen; keine Merges.
2. Forschungsstand #31/#32: `codex/netzel-calibration-block-uncertainty`
   bei `9ee2f84e41a04fafe5af203e809deeaaf329dba3` einschließlich #31/#32.
   Am 31.10.2026 sind #31–#36 erneut als offen bestätigt; `main` bleibt `7f1826c`
   und enthält diese Kette nicht. [Git-/PR-Snapshot](./pollen_methods_2026_10_31/git_snapshot.json).
   Der [Dokumentations-PR #33](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/33)
   (`codex/netzel-workflow-handoff`) baut darauf auf; Ergebnisse daraus nicht nochmals
   nach main übertragen oder als schon gemergt behandeln.
3. Für den Quellenfolgeauftrag nur die neue
   [Evidenztabelle](./pollen_methods_2026_10_31/README.md),
   [Quellen/Fundstellen](./pollen_methods_2026_10_31/SOURCES.md) und den
   [präzisen Restbedarf](./pollen_methods_2026_10_31/SOURCE_GAP.md) lesen.
   Die [Original-Namensprüfung #35](./original_pollen_names_2026_10_31/README.md)
   ist abgeschlossen; deren Tabellen nur bei konkreten Namensfragen nachladen.
   Alle vorhandenen Worktrees waren beim Einstieg sauber und wurden erhalten;
   eigener Worktree `/workspace/netzel-pollen-methods`. Kein Merge beauftragt.

## Nächster Quellenauftrag: historische Methodenkette schließen

Lesbaren Methodenteil und Referenzverzeichnis von Watts, Allen & Huntley (1996),
DOI 10.1016/0277-3791(95)00093-3, beschaffen; bisher nur Titel/Abstract/Metadaten
gelesen. Danach **tatsächlich verwendete** Schlüssel/Sammlungen und Anwendung
je Fossilparameter mit Seite/Abschnitt prüfen. Nicht aus späteren Referenzen
rückwärts auf W96 schließen. Die sieben Literaturbefunde nicht erneut breit
inventarisieren; gezielt die in SOURCE_GAP genannten Grenzen ergänzen,
besonders originale Picea-abies-type/obovata-Definition und erhaltene Untertypen.

Besonders beachten: Monticchio-Kommentare sind akzeptierte EPD-Namen. EMPD2
ordnet Betula pendula dem pubescens-type zu; bei Taxus gibt es zwei verschieden
akzeptierte Einträge mit demselben Originalnamen. Diese Datenbankbefunde nicht
als botanische Synonymie oder als Freigabe einer Artfunktion interpretieren.

Die kompakte Evidenztabelle ist gesichert; die **Methodenkette als ursprüngliches
Abschlusskriterium bleibt offen**. Unterstützende, widersprechende und fehlende
Belege weiter mit Seiten/Abschnitten/Objektkennungen sichern, eigener PR.
Nur gelesene Quellen als geprüft ausweisen. Keine automatische
Artzuordnung, Modelle/Fossilcounts/Gruppierungen unverändert; keine Auswahl
anhand äußerer Scores. Claude behält CLI/App; Analog-Baseline unverändert.

Der frühere Netzwerkblock für die beiden Originaldateien ist behoben.
Neue Originalprüfung benötigt keine alten Chat-Dateien; Bezugsbefehle und
Hashprüfungen in original_pollen_names_2026_10_31/REPRODUCE.md.
Die 2025-Originale bei konkretem Bedarf gesondert prüfen, keine Änderung der
Shared37-Referenz. Monticchio-P1 und weitere Arten bleiben getrennte Pakete.
Ein eigenes vorab gesichertes T1-/synthetisches T2-Designpaket kann gemäß
Roadmap unabhängig vom Volltextzugang vorankommen; kein realer Kernlauf,
keine konkrete Modellvariante und keine geänderte Kernwahl dadurch genehmigt.

## Übergabe an einen neuen Chat

Das Literaturpaket ist ein gesicherter Wechselpunkt mit offen ausgewiesener
Originalmethodenlücke. Startauftrag: „Prüfe aktuellen Git-/PR-Stand, verwende
`codex/netzel-pollen-methods` auf dem #36-Stand, lies AGENTS.md und HANDOFF.md.
Schließe nur die in pollen_methods_2026_10_31/SOURCE_GAP.md dokumentierte
historische Methodenkette; bereits gelesene Literatur nicht neu inventarisieren.
Modelle/Resultate/Counts/Baseline/CLI/App unverändert, eigener PR, kein Merge.“
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
