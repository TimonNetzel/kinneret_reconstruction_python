# Netzel-Projekt: nächster Einstieg

## Zuerst lesen und Stand prüfen

1. [Dauerhafte Arbeitsregeln](../../AGENTS.md) und [aktuellen Status](./PROJECT_STATUS.md)
   lesen. Git-Status, Branches, Worktrees und GitHub-PRs erneut prüfen; keine Merges.
2. Vollständiger gesicherter Forschungsstand: `codex/netzel-calibration-block-uncertainty`
   bei `9ee2f84e41a04fafe5af203e809deeaaf329dba3` einschließlich #31/#32.
   Am 31.10.2026 sind beide PRs offen, `main` enthält sie nicht. Der Dokumentations-PR
   `codex/netzel-workflow-handoff` baut darauf auf; Ergebnisse daraus nicht nochmals
   nach main übertragen oder als schon gemergt behandeln.
3. Für den folgenden Quellenauftrag nur die fachlichen Grenzen in
   [Fortsetzung nach #32](./CHAT_RESTART_UNCERTAINTY.md), Abschnitt
   „Aussagegrenzen und als Nächstes“, und den
   [#28-Bericht](./taxon_transfer_2026_10_30/README.md) gezielt lesen.
   Für Zuordnungsregeln anschließend die betroffenen Einträge in
   [Taxonentscheidungen](./taxon_decisions_2026_10_29/README.md) und
   [Variablenprovenienz](./variable_provenance_2026_10_30/README.md) nachladen.

## Nächster Forschungsauftrag: belegte Pollentyp–Taxon-Zuordnung

Ein begrenztes Quellenpaket für die sieben bereits modellierten botanischen Arten
anlegen: vorhandene Pollentyp-/Art-/Gattungsbezüge zunächst inventarisieren,
Originalbelege mit Fundstelle und Herkunft nachweisen und belegte, mehrdeutige und
fehlende Zuordnungen getrennt ausweisen. Keine automatische Verbindung einer
Artfunktion mit Gattungspollen. Fehlende Belege als präzise Quellenanforderung
sichern. Weitere Taxa/Areale erst nach dieser Bestandsaufnahme als eigenes Paket
abgrenzen; keine Auswahl nach äußeren Scores.

Abschlusskriterium: prüfbare Zuordnungstabelle und Quellen-/Lückenbericht mit
expliziten nicht zuordenbaren Fällen, gezielter Prüfung und eigenem PR. In diesem
Quellenpaket keine neuen Fits, Änderung der sieben Modelle, fossile Klimakurve,
Pilotfreigabe oder CLI/App-Integration. Claude behält CLI/App; Analog-Baseline
bleibt unverändert. Die Forschungsarbeit wird erst im anschließenden Paket begonnen.

## Nur bei konkretem Bedarf weiterlesen

| Anlass | Nachzuladen |
| --- | --- |
| #32-Bänder, Schätzbarkeit oder Aussagegrenzen prüfen | [Bericht](./calibration_uncertainty_2026_10_30/README.md), danach bei Methodenarbeit DESIGN.md und SOURCES.md desselben Ordners |
| Kalibrierung #31 oder SMOTE #29 prüfen | [#31-Bericht](./calibration_2026_10_30/README.md), [#29-Bericht](./smote_nnet_2026_10_30/README.md); jeweiliges Vorabdesign vor Methodenänderungen |
| Große Ausgaben benötigen | [Artefaktverzeichnis](./ARTIFACTS.md), erst dann dort verlinkte REPRODUCE.md; keine alten temporären Dateien voraussetzen |
| Monticchio-Quellenarbeit | [präziser Quellenbedarf](./variable_provenance_2026_10_30/SOURCE_GAP.md); physischer LGM90D→MON16-Anschluss bleibt separat blockiert |
| Historische Sicherung oder frühere Aufträge klären | [Chat-Archiv](./CHAT_WORK_ARCHIVE_2026_10_30.md), [JSON-Inventar](./CHAT_WORK_ARCHIVE_2026_10_30.json), [frühere Einstiegstexte](./ENTRYPOINT_ARCHIVE_2026_10_31.md), [Integration #18–#29](./consolidation_2026_10_30/README.md) |

Archive bleiben erhalten, sind aber keine Pflichtlektüre bei jedem Start.
