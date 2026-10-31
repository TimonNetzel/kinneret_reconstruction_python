# Netzel-Projekt: nächster Einstieg

## Zuerst lesen und Stand prüfen

1. [Dauerhafte Arbeitsregeln](../../AGENTS.md) und [aktuellen Status](./PROJECT_STATUS.md)
   lesen. Git-Status, Branches, Worktrees und GitHub-PRs erneut prüfen; keine Merges.
2. Vollständiger gesicherter Forschungsstand: `codex/netzel-calibration-block-uncertainty`
   bei `9ee2f84e41a04fafe5af203e809deeaaf329dba3` einschließlich #31/#32.
   Am 31.10.2026 sind beide PRs offen, `main` enthält sie nicht. Der [Dokumentations-PR #33](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/33)
   (`codex/netzel-workflow-handoff`) baut darauf auf; Ergebnisse daraus nicht nochmals
   nach main übertragen oder als schon gemergt behandeln.
3. Aktuell ist die [Sieben-Arten-Inventur](./pollen_taxon_evidence_2026_10_31/README.md)
   abgeschlossen; die Originalprüfung und Artbrücke sind offen. Dafür nur
   [SOURCE_GAP.md](./pollen_taxon_evidence_2026_10_31/SOURCE_GAP.md) und die
   betroffenen Zeilen von inventory.csv / sources.json desselben Ordners lesen.
   [Inventur-PR #34](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/34):
   `codex/netzel-pollen-taxon-evidence` ab `6f35df1` (#33),
   zusätzlich zum vollständigen Forschungsstand. Kein Merge beauftragt.

## Nächster Forschungsauftrag: Original-Namensketten prüfen

Frisch bezogene Monticchio-Parameterdefinitionen (PANGAEA.739452) und die
EMPD2-p_vars-Namensketten (PANGAEA.909130) anhand der bestehenden Hashes prüfen:
für alle sieben Modellarten Originalnamen/-codes, akzeptierte Namen/-IDs,
Untertypen und Bestimmungsnotizen mit exakter Fundstelle sichern. Die sieben
Katalogkandidaten sind keine vollständigen Gattungspools. Taxus besonders als
Namens-vs.-Bestimmungsfrage prüfen, ohne eine Artzuordnung vorwegzunehmen.

Der aktuelle Zugriff auf doi.pangaea.de und store.pangaea.de wird vom Proxy
mit CONNECT HTTP 403 abgewiesen. Keine alten temporären Rohdateien voraussetzen.
Nach verfügbarem Zugriff die betroffenen Operationen neu prüfen, keine
Referenzhashes ersetzen. Anschließend gezielte palynologische Originalbelege
zu Art-/Typgrenzen recherchieren; vollständige Anforderungen im SOURCE_GAP.

Abschlusskriterium: prüfbare Original-Namensketten und verbleibende Beleglücken
je Art, eigener PR, keine automatische Verbindung einer Artfunktion mit
Gattungspollen. Keine neuen Fits, Modellwahl anhand äußerer Scores,
Fossilrekonstruktion, Pilotfreigabe oder CLI/App-Integration. Claude behält
CLI/App; Analog-Baseline bleibt unverändert. Weitere Taxa/Areale erst als
separates Paket abgrenzen. Die historische Fortsetzung nach #32 nur bei Fragen
zu den Kalibrierungs-/Unsicherheitsgrenzen nachladen.

## Nur bei konkretem Bedarf weiterlesen

| Anlass | Nachzuladen |
| --- | --- |
| #32-Bänder, Schätzbarkeit oder Aussagegrenzen prüfen | [Bericht](./calibration_uncertainty_2026_10_30/README.md), danach bei Methodenarbeit DESIGN.md und SOURCES.md desselben Ordners |
| Kalibrierung #31 oder SMOTE #29 prüfen | [#31-Bericht](./calibration_2026_10_30/README.md), [#29-Bericht](./smote_nnet_2026_10_30/README.md); jeweiliges Vorabdesign vor Methodenänderungen |
| Große Ausgaben benötigen | [Artefaktverzeichnis](./ARTIFACTS.md), erst dann dort verlinkte REPRODUCE.md; keine alten temporären Dateien voraussetzen |
| Monticchio-Quellenarbeit | [präziser Quellenbedarf](./variable_provenance_2026_10_30/SOURCE_GAP.md); physischer LGM90D→MON16-Anschluss bleibt separat blockiert |
| Historische Sicherung oder frühere Aufträge klären | [Chat-Archiv](./CHAT_WORK_ARCHIVE_2026_10_30.md), [JSON-Inventar](./CHAT_WORK_ARCHIVE_2026_10_30.json), [frühere Einstiegstexte](./ENTRYPOINT_ARCHIVE_2026_10_31.md), [Integration #18–#29](./consolidation_2026_10_30/README.md) |

Archive bleiben erhalten, sind aber keine Pflichtlektüre bei jedem Start.
