> Aktualisierung 30.10.2026: Die Zusammenführung von #18–#29 ist ausdrücklich
> beauftragt. Gemeinsamer Einstieg: [HANDOFF.md](./HANDOFF.md); nach Merge `main`
> verwenden. Nachfolgend bleibt der historische Paketstand dokumentiert.

# Fortsetzung nach PR #26

Stand: 30.10.2026. Vor Weiterarbeit den aktuellen PR-/Merge-Stand erneut prüfen.
Diese Datei ergänzt den Übergabeindex aus PR #25; die Branches sind nicht kumulativ.

## Einstieg

1. Auf PR #26 `AGENTS.md`, `docs/thesis_research/PROJECT_STATUS.md` und
   `docs/thesis_research/neotoma_workshop_2026_10_30/README.md` lesen.
2. Auf PR #25 zusätzlich `docs/thesis_research/CHAT_RESTART.md`,
   `HANDOFF.md` und den dort verlinkten `MISSING_SOURCE_REQUEST.md` lesen.
   Gesicherter #25-Stand: `00c9f037642470c7d1a9a44e488f062109292c2d`.
   Die Berichte aus #23–#25 über diesen Index gezielt hinzunehmen.
3. #18–#26 waren beim letzten Check offen und ungemergt. Arbeitsbasis:
   Branch von #18, solange offen, sonst main. Vorarbeiten aus anderen PRs
   ausdrücklich über deren Ref beziehen, keine Dateien im Checkout voraussetzen.

## Ziel, Rollen und Abnahmen

Ziel ist eine aktualisierte Dissertation als App für taxonbasierte
Multiproxy-Klimarekonstruktionen europäischer Fundstellen.
Die fertige, unabhängig geprüfte moderne Analog-Baseline (#18) ist ein
Vergleichsverfahren; sie ist keine fertige taxonbasierte Hauptmethode.
Neue Taxon-Präsenz-/Transferfunktionen sind noch zu erstellen und zu prüfen;
die ursprünglichen 279 Funktionen wurden nicht wiederhergestellt.
Bisher ist keine fossile Klimakurve berechnet.

Claude betreut CLI/App. Codex betreut Datenrecherche und unabhängige Prüfung;
Claudes Integrationsdateien nicht gleichzeitig bearbeiten.
Keine Baseline-Neuentwicklung, stillen Quellenkorrekturen, automatischen
Fossilpilot-Freigaben, Autorennachrichten oder Merges.
P1 bleibt offen; P2 erst nach fachlicher Abnahme oder ausdrücklich
geänderter Zieldefinition. Die 69 Datenbank-Probenkennungen und exakten
Tiefen sind bestätigt. Physische LGM90D→MON16-Zuordnung und
Korrelationsunsicherheit fehlen weiterhin.

## Ergebnis dieses Chats

PR: https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/26
Branch: `codex/netzel-neotoma47525-source-lead`
Geprüfter Ergebniscommit: `63912e2c8d6820ce7c35e3740a7b85945b5800d3`.

Ein echtes öffentliches Archivobjekt aus dem offiziellen Neotoma-Workshop
wurde bezogen und geprüft, kein neuer Live-API-Export.
Monticchio enthält darin 13 Collection Units; LGM90D/26608 hat 245 Proben,
MONTICCHIO/47525 hat 924 Proben und 924 gespeicherte Analyseeinheitsnamen.
Die 69 Pilotproben ergeben auf 76 gemeinsam vergleichbaren nativen
Variablenschlüsseln 28 eindeutige numerische Kandidaten mit Abschnittsnamen.
Das beweist weder dieselbe physische Probe noch eine MON16-Tiefe.

41 Pilotproben bleiben ohne vollständigen Treffer.
Die nachgelagerte Diagnose benennt 47 Zellabweichungen:
Plantago 35, Scrophulariaceae 9, Abies 2, Sambucus 1.
Halbzählig gespeicherte Werte bleiben unverändert.
16 Quercus-Aggregationsunterschiede zur älteren Quelle und vier
Listenpositionswechsel sind ausdrücklich erhalten.
Fehlende Notiz-Exportslots beweisen keine leeren Datenbanktabellen.

Alle Ergebnis-/Diagnosetabellen, Quellen- und Hashnachweise,
Reproduktionsskripte, separate Prüfung und vier Negativtests sind in #26.
Zwei Produzentenläufe waren bytegleich; die separate Rohslot-Prüfung bestand.
Auf Ergebniscommit 63912e2 bestanden vier GitHub-CI-Jobs
(Python 3.11/3.13, push und pull_request).
Ein späterer reiner Übergabecommit ist davon zu unterscheiden.

## Nächster konkreter Quellenbeleg

Die 47 benannten Zellabweichungen anhand ursprünglicher
Variablen-/Aggregationsdefinitionen und Einreichungsdaten aufklären.
Die archivierten Abschnittsnamen mit historischen Kompositierungstabellen
und danach MON16 samt Korrelationsunsicherheit verbinden.
Keine unveränderte Wiederholung ausgeschöpfter API-/EPD-Prüfungen.
Die ältere 66/69-Prozentähnlichkeit aus #21 ist ein anderer Vergleich
und durch die 28 nativen Kandidaten weder ersetzt noch aufgelöst.

## Reproduzierbare Eingänge und Grenzen

Keine temporären Dateien aus diesem Chat voraussetzen.
Die öffentlichen Eingänge samt URLs, gesperrten Refs und SHA256 stehen in
`neotoma_workshop_2026_10_30/sources.json`; Reproduktion im zugehörigen README.

- RDS: NeotomaDB/Workshops, Ref
  `5caa58c9da2b23fd174ec3d785f3dd931298a3d6`,
  `2023_INQUA_pollen_vertebrates/data/itDownload.RDS`;
  2.495.691 Bytes, SHA256
  `03536b6f1d3974de77ff997f6d86674a78882b2ebe305bb4e135e38289a2a829`.
- Ältere Quelle: PANGAEA.739452; 69-Proben-Liste aus #25 unter
  `docs/thesis_research/epd_sample_identity_2026_10_30/sample_identity.csv`.
- Das RDS ist Dateninput, R-/Workshop-/Bchron-Programme nicht ausführen.
  Rohvektor vor Konvertierung nach Standort auswählen: 59 gleiche
  Listennamen würden sonst beim Umwandeln in ein Dictionary verloren gehen.
- Roharchive, installierte Abhängigkeiten und Chat-Zwischendateien sind
  nicht eingecheckt. Öffentliche Eingänge sind über URL/Hash wiederbeziehbar.
- Original-ScienceDirect-ZIP nur für erneute Original-Rohdatenprüfung
  wieder vom Nutzer anhängen lassen:
  `ScienceDirect_files_29Oct2026_22-15-14.667.zip`,
  SHA256 `2329fab3f4fc196174c70960cb9f1fddcbe9f9158ab099a3351193986f7ebdf8`.

Neue Ergebnisse auf eigenem Branch als PR dokumentieren.
Kurz auf Deutsch antworten und zu Beginn sowie nach Arbeitspaketen
erklären: fertig / aktuell / als Nächstes.
