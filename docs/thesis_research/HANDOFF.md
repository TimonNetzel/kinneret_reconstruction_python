# Netzel: historischer EMPD2-Erzeugungs-/Exportnachweis

## Paketabschluss, 01.11.2026

[A4_E_PROVENANCE_EVIDENCE_REVIEW](./A4_E_PROVENANCE_EVIDENCE_REVIEW.md#historischer-erzeugungs--und-exportnachweis-01112026)
ist um H1–H7 ergänzt. **Dokumentarisch abgeschlossen, reale Herkunftskette
weiter blockiert:** historischer Basiscontainer mit 5m-Vorgabe und unversionierter
latlon-utils-Installation; historischer WorldClim-2.0-Downloader; GitHub-Spur
eines PostgreSQL-/TSV-Exports vom 05.10.2020. Öffentliche Supplementbeziehung
v2.0/Zenodo↔PANGAEA belegt, konkrete Erzeugungs-/SQLite-Feldidentität unbelegt.
Keine Auflösung/Unterversion/Periode für PANGAEA festgelegt; R01–R03 teilweise
belegt, `unverified` bleibt. Begrenzt an der fehlenden Lauf-/Exportbindung gestoppt.

Basis/PR-#64-Head `7628f1267ff60ec24f191365826e87a6add9caa5` geprüft;
nach `65bd8b2` dort nur Sicherungsverweise. #64 offen/ungemergt und mergebar,
Head-CI [41287220663](https://github.com/TimonNetzel/kinneret_reconstruction_python/actions/runs/41287220663)
erfolgreich. Fünf vorgefundene Worktrees sauber erhalten; kein Reset/Stash/Merge.
Eigener Worktree `/workspace/netzel-historical-export-evidence`, Branch
`codex/netzel-historical-export-evidence`; eigener gestapelter PR gegen #64.
Sicherungsverweis wird nach Commit/PR eingetragen.

Nur Review ergänzt und HANDOFF/Status gepflegt. 14 gelesene Quelltexte gegen
Git-Blob-SHAs geprüft; sieben lokale Links/Anker und Drei-Dateien-Diff geprüft,
753 geschützte Git-Blobs/Dateimodi und alte Statushistorie unverändert. Keine Prüfung realer Datenbytes,
keine manuellen Projekt-/Modelltests oder Rechnungen; automatische CI separat.
Alle übrigen Fachdateien unverändert. Keine Gatefreigabe/A4-Umsetzung/A5/A6,
Unteragenten, Nachrichten oder Merges. Git-Fetch über freigegebenen Netzpfad
möglich; Recherche über Web-/GitHub-Anbindung, kein Nutzerdatei-Blocker.

## Genau ein Folgeauftrag: Probenzeit und Integration – vorbereitet, wartet auf Nutzer-OK

**Ziel:** öffentlich dokumentierte Bedeutung und Herkunft von Sammlung,
Probenalter, Altersunsicherheit und zeitlicher Integration in EMPD2 gegen
R04/R05 bewerten. Nach abgeschlossener R01–R03-Suche die separate Zeitseite
klären; keine Behauptung gemeinsamer Klimaperiodengeltung und kein R06-Nachweis.

**Eingänge/Leseplan:** Stand/PR-Head/CI prüfen, ungesicherte Arbeit erhalten;
eigener Worktree/Branch/PR ohne Merge. AGENTS, diese HANDOFF, nur neuester
PROJECT_STATUS-Abschnitt; Review nur historischer Abschluss/Stop;
A4_E_REOPENING_REQUIREMENTS nur R04/R05 und Herkunftsregeln;
A4_E_CLIMATE_TIME_GAPS nur Zeitnachweiskette; MODEL_CONTRACT Eingänge/Stop;
MODULAR_RECONSTRUCTION_PLAN §1, §2 nur A4, §4.
Dann selbst öffentliche Original-/Herausgebertexte suchen und gezielt lesen:
Davis 2020 Methoden/Metadaten, PANGAEA.909130-Metadatendefinitionen,
README/Metadatendokumentation am EMPD-data-v2.0-Commit
`737b76eb269b071422edded6ac332654ff83c103` sowie unmittelbar verknüpfte
historische Definitionstexte. Öffentlich auffindbare Dokumente beschafft Codex
nach AGENTS selbst ohne zusätzliches Recherche-OK. Kein Nutzer-Dokument nötig.

**Umfang:** nur A4_E_PROVENANCE_EVIDENCE_REVIEW.md um einen datierten
R04/R05-Abschnitt ergänzen, kurze HANDOFF-/Statuspflege. Je Aussage Quelle,
Stand, Fundstelle, Zeitbedeutung/Unsicherheit, Archiv-/Probenreichweite und Grenze.
Dokumentierte allgemeine Definition von tatsächlicher Zuordnung zu konkreten
Proben unterscheiden. Fehlende Integration nicht als Nullunsicherheit behandeln.

**Abnahme/Stop:** R04 und R05 getrennt belegt/teilweise belegt/unbelegt bewerten,
Restlücken konkret nennen. Nach relevanten auffindbaren Herausgeberpassagen
abschließen, auch bei Nichtbeleg; keine allgemeine Literatur-/Dokumentationsschleife.
Keine Wiederholung der R01–R03-Suche. Bei nur durch reale Proben-/Dateiprüfung,
Kontakt oder fachliche Auswahl schließbarer Lücke stoppen, keine Nutzerdatei
als künstlichen Blocker erfinden. Quellen-/Link-/Diff-/Blobprüfung,
Statushistorie erhalten, Commit/eigener PR; genau einen nächsten Auftrag
vorbereiten und Folgeprompt automatisch ausgeben.

**Schutz:** keine Rohdaten-/Datenbank-/Raster-/ZIP-/TSV-Zeilenprüfung oder
-beschaffung, keine Proben-/Daten-/Modellwahl, Joins, Reparatur, Implementierung,
manuellen Tests, Rechnungen, Fits oder Simulationen. Übrige Fachdateien,
Anforderungen, Verträge, Modelle/Resultate/Register, Importer/Flags,
Counts/Gruppen, Baseline und CLI/App unverändert. R06/übrige Gates offen;
keine A4-Umsetzung/A5/A6, Gatefreigabe, Unteragenten, Nachrichten oder Merges.

## Kopierfertiger Folgeprompt

```text
Setze Netzel auf codex/netzel-historical-export-evidence fort.
Sicherungsverweis wird nach Commit/PR eingetragen.
Prüfe aktuellen PR-Head/CI; erhalte ungesicherte Arbeit.
Eigener Worktree/Branch/PR, kein Merge.
Führe ausschließlich den in HANDOFF vorbereiteten R04/R05-Auftrag aus:
öffentliche EMPD2-Herausgeberbelege zu Probenzeit, Sammlung,
Altersunsicherheit und Integration selbst suchen und bewerten.
Leseplan, Umfang, Abnahme und Stop aus HANDOFF einhalten;
Recherche nach AGENTS ohne zusätzliches OK und ohne Nutzerdatei.
Nur A4_E_PROVENANCE_EVIDENCE_REVIEW.md ergänzen, HANDOFF/Status kurz pflegen.
Keine erneute R01–R03-Suche, keine R06-Freigabe.
Keine Rohdaten-/Datenbank-/Raster-/ZIP-/TSV-Zeilenprüfung, Auswahl,
Implementierung, manuellen Tests oder Rechnungen; übrige Fachdateien unverändert.
Keine A4-Umsetzung/A5/A6, Gatefreigabe, Unteragenten, Nachrichten oder Merges.
Bei fehlendem öffentlichen Nachweis konkrete Lücke nennen und stoppen.
Ergebnis prüfen, per Commit/eigenem PR sichern, genau einen Folgeauftrag
vorbereiten und Folgeprompt automatisch mitliefern. Kurz auf Deutsch berichten.
```
