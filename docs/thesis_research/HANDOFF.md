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

## Vorbereiteter Folgeauftrag – wartet auf Nutzer-OK

**Nächstes empfohlenes Paket: T1-Design und begrenzter synthetischer T2-Prototyp.**
Der Nutzer hat am 31.10.2026 den Ablauf „Ergebnis sichern → Folgeauftrag
vorbereiten → mit OK fortsetzen“ festgelegt. Dieses Methodenpaket ist vorbereitet,
noch nicht begonnen oder durch die Ablaufentscheidung fachlich freigegeben.

- **Ziel:** eine vorab begründete Rekonstruktionsregel und einen kleinen
  Forschungsprototyp für bekannte synthetische Klimazustände schaffen.
- **Umfang:** zuerst Design mit Datenrollen, Beobachtungsmodell, mathematischer
  Kombination/Inversion, Abhängigkeiten, Klimaprior, Support und behandelter/
  fehlender Unsicherheit dokumentieren und vor wissenschaftlichen Anpassungen
  sichern. Danach nur den davon abgedeckten synthetischen Prototyp implementieren.
  Artenvorkommenswahrscheinlichkeit nicht mit fossiler Pollenhäufigkeit gleichsetzen.
- **Eingänge/Leseplan:** AGENTS.md, PROJECT_STATUS, diese Übergabe und
  TAXON_PILOT_ROADMAP T1/T2; danach gezielt taxon_transfer_2026_10_30/DESIGN.md
  und config.json für die Bedeutung der vorhandenen Funktionen sowie das neue
  Literaturpaket für die Quellen-/Artbrückengrenzen. Weitere Quellen nur zur
  konkreten Methodenbegründung nachladen, keine Pflichtlektüre der Archive.
- **Schutzumfang:** ausschließlich neues, getrenntes Forschungsdesign,
  Prototyp und synthetische Ausgaben sowie Status/Übergabe. Bestehende Modelle,
  Forschungsresultate/-register, Fossilcounts, Analog-Baseline und CLI/App
  unverändert. Keine Modellwahl anhand bekannter äußerer Testergebnisse,
  keine reale Kernrekonstruktion, Art-/Pilotfreigabe, Autorennachrichten oder Merges.
- **Abschluss:** Design vorab gesichert; bekannte synthetische Zustände,
  widersprüchliche Taxonsignale, fehlende Beobachtungen, Prior-only-Verhalten,
  Supportgrenzen und nicht identifizierbare Fälle nach vorher festgelegten
  Kriterien geprüft. Scheitern oder fehlende Identifizierbarkeit vollständig
  dokumentieren; keine Erfolgsgarantie. Kein SBC-Nachweis ohne geeignetes
  kohärentes generatives Modell. Eigenen PR, Ergebnisstatus und einen nächsten
  klaren Folgeauftrag sichern.

**OK-Bedeutung:** Ein kurzes OK zu diesem Folgeauftrag startet genau dieses
Paket nach erneuter Standprüfung im eigenen Worktree/Branch. Die W96-Methodenkette
bleibt ein gesondert blockierter Quellenauftrag; sie blockiert synthetische
Methodenarbeit nicht, ersetzt aber weiterhin keine realen Eingangsfreigaben T3/T4.

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

## Weiterhin blockierter Quellenauftrag: historische Methodenkette schließen

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

Ein neuer Chat ist vor dem Wechsel von Quellenprüfung zu Methodendesign sinnvoll,
aber nicht erforderlich. **Vorgeschlagener Startauftrag:** Wenn der Nutzer den
folgenden Prompt absendet, beauftragt er damit das vorbereitete Paket.

> Setze das Netzel-Projekt mit dem in HANDOFF.md vorbereiteten T1-/synthetischen
> T2-Paket fort. Verwende den Branch `codex/netzel-next-task-workflow`;
> letzter Forschungsstand darunter ist PR #37, Commit
> `fc52ce7ea2733ad11c39a59e629a01d14361fb8e`. Prüfe zuerst den aktuellen Git-/PR-/
> Merge-Stand, Branches und lokale Änderungen; erhalte ungesicherte Arbeit.
> Lies AGENTS.md, docs/thesis_research/HANDOFF.md, PROJECT_STATUS.md und
> TAXON_PILOT_ROADMAP.md. Führe genau den dort vorbereiteten Folgeauftrag aus:
> zuerst das begründete Design sichern, dann einen begrenzten synthetischen
> Forschungsprototyp prüfen. W96-Methodenteil und historische Artbrücken bleiben
> offen; keine reale Kernrekonstruktion oder Artfreigabe. Bestehende Modelle,
> Resultate, Fossilcounts, Analog-Baseline und CLI/App unverändert; keine
> Modellentscheidungen aus äußeren Scores. Claude betreut CLI/App. Eigener
> Worktree/Branch und PR, kein Merge und keine Autorennachrichten. Ohne
> Unteragenten beginnen, kurze deutsche Updates. Ergebnis und nächsten
> begrenzten Folgeauftrag im Repository sichern.

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
