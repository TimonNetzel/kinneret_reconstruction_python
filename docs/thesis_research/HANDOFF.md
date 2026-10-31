# Netzel-Projekt: nächster Einstieg

## Gesichertes Ergebnis: begrenztes EMPD2-Archiv-Herkunftspaket

Nach Nutzerauftrag am 31.10.2026 ausgeführt. Basis
`90d0350f20d8adc5d1feab09722cf02f0ebb8b69` aus
[PR #43](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/43).
Eigener Worktree `/workspace/netzel-empd2-lineage`, Branch
`codex/netzel-empd2-archive-lineage`; eigener PR folgt nach Inhaltscommit.
[ARCHIVE_LINEAGE](./empd2_archive_lineage_2026_10_31/ARCHIVE_LINEAGE.md),
[LINEAGE_EVIDENCE](./empd2_archive_lineage_2026_10_31/LINEAGE_EVIDENCE.json),
[DECISION_GAPS](./empd2_archive_lineage_2026_10_31/DECISION_GAPS.md).

**Archivauftrag begrenzt abgeschlossen; Erzeugungskette weiterhin offen.**
Zuerst offizielle DOI-Landingpage und verlinkte DataCite-Metadaten gelesen.
Die dort ausdrücklich zugeordnete PostgreSQL-Datei als genau zweites
Originalobjekt nur per Header und 32.768-Byte-Präfix geprüft, kein voller Dump.
PostgreSQL/pg_dump 11.2 und `count double precision` belegt; das ist kein
SQLite-Exportnachweis. Keine konkret angewandte Code-/Exportversion gefunden.
Nach den zwei Quellen gestoppt, keine Ersatzsuche. Messstatus/Nullunterdrückung,
N_original und Hierarchieanwendung offen; S_flag und n_ausgewählt getrennt.

Vorhandene SQLite nur gehasht, nicht geöffnet oder neu heruntergeladen:
21.327.872 Bytes, SHA-256
`5634e85d922ee0741c41a77d081010c9cc1c113966b283bc52e2051a9d3722e2`.
Die drei bekannten Zeugen nicht erneut ausgelesen; #43-Befunde übernommen:
108 gleiche Tupel, 7,4 erhalten, 36/40 Gruppen, sichtbarer Neuanlagepfad rundet
und überspringt bestehende Proben. Keine neue Matrix/Kohorte, Bereinigung,
Code-/Testausführung, Fits, Scores, Simulationen oder Kernläufe.

Einstieg: #31–#43 offen/ungemergt; main `7f1826c`, #43-Head wie beauftragt,
#43-PR-CI erfolgreich. Fünf vorherige Worktrees dieser Umgebung sauber erhalten.
Bestehende Designs/Modelle/Resultate/Register/Importer/Flags/Counts/Gruppen,
Baseline und CLI/App unverändert. Keine Unteragenten, Autorennachrichten oder
Merges. W96 und reale T0/T1/T3/T4/Monticchio-P1 bleiben offen.

## Genau ein Folgeauftrag – vorbereitet, wartet auf Nutzer-OK

**Eine dokumentarische Entscheidungsvorlage zum blockierten realen EMPD2-Counts-Einstieg erstellen.**

- **Ziel:** anhand vorhandener Belege gegenüberstellen: reales Counts-T1 bis
  zu archivgenauer Evidenz zurückstellen oder später ausdrücklich ein anderes,
  annahmenabhängiges Forschungsziel entwerfen lassen. Keine Wahl vorwegnehmen.
- **Ergebnis:** genau eine neue `COUNTS_DECISION_BRIEF.md` plus Status/Übergabe.
  Für beide Wege notwendige Entscheidung, minimale Evidenz, zulässige Aussagen
  und Stopbedingung nennen; N_original/S_flag/n_ausgewählt auseinanderhalten.
- **Leseplan:** AGENTS, diese Übergabe, PROJECT_STATUS; die drei Dateien des
  aktuellen Archivpakets. Dann nur `INPUT_CONTRACT.json` und `SOURCE_GAPS.md`
  aus `empd2_counts_readiness_2026_10_31`, Roadmap T1/T3; #43-RULE_EVIDENCE
  ausschließlich bei konkreter Regelbehauptung.
- **Grenzen:** keine weitere Recherche, neue Datenquelle, Daten-/Taxon-/
  Nennerauswahl, Matrix/Kohorte, Bereinigung, Code-/Test-/Modell-/Kernlauf.
  Bestehende Designs/Verträge/Register/Importer/Flags/Counts/Gruppen, Baseline
  und CLI/App unverändert. Keine Nullfüllung/Rundung oder reale Freigabe.
- **Sicherung:** eigener Worktree/Branch/PR, Quellen-/Link-/JSON-/Diffprüfung;
  keine Unteragenten, Autorennachrichten oder Merges. W96 und reale Gates offen.

[Konkreter Entscheidungspunkt und Abschlusskriterien](./empd2_archive_lineage_2026_10_31/DECISION_GAPS.md).
Ein OK beauftragt nur diese Vorlage, weder Zieländerung, Implementierung noch Lauf.
Der jetzt erledigte Archivauftrag benötigt kein weiteres OK.

## Zuerst Stand prüfen

1. Git-/PR-/Merge-/CI-Stand, Remote-Heads, Branches, Worktrees und lokale/
   unversionierte Änderungen prüfen; vorhandene Arbeit erhalten.
2. Aktuellen Quellenbranch verwenden, nicht main: die Kette ist ungemergt.
   Basiscommit bewusst wählen und nennen; eigenen Worktree anlegen.
3. Nur den obigen gezielten Leseplan bearbeiten, keine alten Läufe wiederholen.

## Separat blockiert oder später, kein zusätzlicher Folgeauftrag

W96: lesbaren Methodenteil und Referenzverzeichnis von Watts, Allen & Huntley
(1996), DOI 10.1016/0277-3791(95)00093-3, sowie tatsächlich verwendete
Bestimmungsschlüssel und ihren datensatzbezogenen Einsatz weiterhin nicht
belegt. [Präzise Grenze](./pollen_methods_2026_10_31/SOURCE_GAP.md).
Keine Rückprojektion späterer Referenzen. Namensübereinstimmung ist keine
botanische Synonymie oder Artfreigabe; Monticchio-Kommentare sind akzeptierte
EPD-Namen. Physischer LGM90D→MON16-Anschluss einschließlich Unsicherheit offen.
Klimanormalperiode unverifiziert. Synthetisches T1/T2 ersetzt keine reale Evidenz.

## Vorgeschlagener Startprompt für eine spätere Fortsetzung

> Setze das Netzel-Projekt auf codex/netzel-empd2-archive-lineage fort.
> Nutze den gesicherten Head/PR aus PROJECT_STATUS; Basis des Archivpakets war
> 90d0350f20d8adc5d1feab09722cf02f0ebb8b69 aus PR #43. Prüfe zuerst Git-/PR-/
> Merge-/CI-Stand, Branches, Worktrees und lokale Änderungen; erhalte Arbeit.
> Erstelle ausschließlich die in HANDOFF vorbereitete schriftliche
> Entscheidungsvorlage zum blockierten realen EMPD2-Counts-Einstieg nach dem
> gezielten Leseplan. Keine weitere Quellensuche oder wissenschaftliche
> Entscheidung/Implementierung, keine Datenwahl/Matrix/Kohorte/Bereinigung,
> Codeausführung/Fits/Scores/Simulationen/Kernläufe. Bestehende Designs,
> Modelle, Resultate, Verträge, Register, Importer, Flags, Counts/Gruppen,
> Baseline und CLI/App unverändert; W96 und reale Gates offen. Eigener
> Worktree/Branch/PR, keine Unteragenten/Autorennachrichten/Merges, kurze
> deutsche Updates. Ergebnis und genau einen nächsten begrenzten Auftrag sichern.

Absenden dieses Prompts oder ein OK im passenden Kontext autorisiert genau
jene Vorlage. Kein Chatwechsel erforderlich.
