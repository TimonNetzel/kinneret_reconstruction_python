# Netzel-Projekt: nächster Einstieg

## Gesichertes Ergebnis: EMPD2-Nenner-/Sparse-Quellenpaket

Nach Nutzer-OK hier im Chat am 31.10.2026 ausgeführt. Basis PR #42 bei
`e1e8ebe32f17487c4c91d2044db845112a8365bb`; Worktree
`/workspace/netzel-empd2-rules`, Branch `codex/netzel-empd2-denominator-rules`.
PR-/Inhaltscommit-Verweis steht in PROJECT_STATUS.
[Ergebnis](./empd2_denominator_rules_2026_10_31/DENOMINATOR_RULES.md),
[Regelbelege](./empd2_denominator_rules_2026_10_31/RULE_EVIDENCE.json),
[Quell-/Datenvergleich](./empd2_denominator_rules_2026_10_31/source_comparison.json).

Begrenzter Teilabschluss: Originalmethodik gelesen und offizielles v2.0-Release
am Commit `737b76eb269b071422edded6ac332654ff83c103` gepinnt; zwölf
Quelldateien per Git-Blobhash geprüft, kein fremder Code ausgeführt.
Drei bekannte Zeugen, 108 Messwert-Tupel zwischen Release und Sperr-SQLite
identisch; alle 98 bereitgestellten Prozentwerte mit S_flag=311/731/840 innerhalb
Quellpräzision kompatibel. Zehn leere Prozentfelder passen zum percent_values-Flag.
Dies ist keine globale Berechnungsregel oder N_original-Freigabe.

Der sichtbare Importpfad übernimmt positive Eingänge, rundet und überspringt
bestehende Proben. Erhaltenes 7,4 und 36 gegenüber 40 Gruppen zeigen, dass die
exakte SQLite-Erzeugungskette nicht belegt ist; kein Beweis, dass das Skript nie
benutzt wurde. Allgemeine archiveigene Nullregel, N_original und hierarchischer
Aggregationsalgorithmus bleiben offen. Kein Nullfüllen oder Runden angewandt.
Klimaperiode weiter unverified, W96 und reale T0/T1/T3/T4/Monticchio-P1 offen.

Kein Matrix-/Kohortenexport, Fits, Scores, Simulationen oder Tests; nur Quellen-
und Dokumentationsprüfung. Originaldatenbankhash vor/nach unverändert.
Alle bestehenden Designs/Modelle/Resultate/Register/Importer/Flags/Fossilcounts/
Gruppen, Baseline und CLI/App unverändert. Neun bisherige Worktrees sauber
geblieben; main `7f1826c`, #31–#42 bei Einstieg offen, #42-CI grün.
Keine Unteragenten, Merges oder Kontakte. Nicht gelesene Zenodo-Seite/großer
SQL-Dump nicht als geprüft behandeln; Details im Ergebnis.

## Genau ein vorbereiteter Folgeauftrag – wartet auf Nutzer-OK

**Erzeugungs-/Exportherkunft genau des gesperrten PANGAEA-SQLite-Originals prüfen.**

- **Zielobjekt:** DOI 10.1594/PANGAEA.909130, EMPD2.sqlite3, 21.327.872 Bytes,
  SHA-256 5634e85d922ee0741c41a77d081010c9cc1c113966b283bc52e2051a9d3722e2.
  Heutige Datenbank, ähnliches Release oder bloße Namensgleichheit genügen nicht.
- **Konkrete Frage:** Welche offiziell zugeordnete Version/Exportlogik erklärt
  erhaltenes 7,4, 40 Gruppen und fehlende Sample×Typ-Paare? Sind Messstatus,
  Nullunterdrückung und ursprünglicher Nenner für genau diese Datei belegbar?
- **Begrenzte Quellen:** zuerst offizielle Archivmetadaten/Dateibeschreibung
  dieses DOI; nur bei präzisem offiziellen Verweis ein zweites ursprüngliches
  Export-/Wörterbuch-/Codeobjekt. Schon gepinnte Repositorydateien lesend zur
  Abgrenzung nutzen. Keine breite Suche nach weiteren Datenbanken oder Paketen.
- **Leseplan:** AGENTS.md, PROJECT_STATUS, diese Übergabe und Roadmap T1–T3;
  DENOMINATOR_RULES.md, RULE_EVIDENCE.json, SOURCE_GAPS.md und source_comparison.json
  dieses Pakets. Voriger INPUT_CONTRACT und #35-Reproduktions-/Hashnachweis
  nur bei konkreter Vertrags-/Bytefrage. W96 bleibt gesondert blockiert.
- **Umfang:** nur neues ARCHIVE_LINEAGE.md, LINEAGE_EVIDENCE.json und verbleibendes
  DECISION_GAPS.md plus Status/Übergabe. Höchstens die drei bekannten Zeugen
  lesend prüfen, keine große SQL-Datenkopie oder weitere Messmatrix exportieren,
  keinen Code ausführen oder Daten bereinigen. Keine neue Proben-/Typauswahl.
- **Abschluss:** Quelle→Version→nachweislich angewandter Schritt→Archivfeld
  mit Objekt/Datei/Zeilen/Hash belegen oder getrennt sperren. N_original,
  S_flag/Prozentnenner und n_ausgewählt unterscheiden. Falls die zwei gezielten
  Quellen die Kette nicht liefern, hier stoppen und einen konkreten
  Entscheidungspunkt dokumentieren; keine Ersatzsuche oder Trainingsfreigabe.
- **Schutz/Sicherung:** bestehende Designs/Modelle/Resultate/Register/Importer/
  Flags/Fossilcounts/Gruppen, Baseline und CLI/App unverändert. Keine Fits,
  Scores, Simulationen, Kern-/CLI-/App-/Standardtests, Merges, Kontakte oder
  Unteragenten. Eigener Worktree/Branch/PR, Quellen-/Link-/Hash-/Diffprüfung,
  Ergebnis und genau einen nächsten Auftrag sichern.

Ein weiteres OK hier im Chat autorisiert genau dieses neue Archivpaket.
Der abgeschlossene Quellenauftrag hat keine neuen wissenschaftlichen Läufe
freigegeben. Kein Chatwechsel erforderlich.

## Zuerst Stand prüfen

1. Git-/PR-/Merge-/CI-Stand, Remote-Heads, Branches, Worktrees und lokale/
   unversionierte Änderungen prüfen; vorhandene Arbeit erhalten.
2. Aktuellen Quellenbranch verwenden; Inhaltscommit/PR in PROJECT_STATUS.
   main enthält die ungemergte Kette nicht automatisch.
3. Gezielten Leseplan befolgen; keine alten Import-/Modellläufe wiederholen.

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
Der vorab gesicherte synthetische T1-Kandidat samt begrenztem T2 ist jetzt
abgeschlossen; sein Ergebnis ersetzt keine historische Methodenkette.
Kein realer Kernlauf, keine reale Modellvariante und keine geänderte Kernwahl
sind dadurch genehmigt.

## Übergabe an einen neuen Chat

**Vorgeschlagener Startauftrag, wartet auf OK; auch hier im Chat ausführbar:**

> Setze das Netzel-Projekt auf `codex/netzel-empd2-denominator-rules` fort;
> letzter Inhaltscommit/PR stehen in PROJECT_STATUS.md, Basis war PR #42 bei
> e1e8ebe. Prüfe Git-/PR-/Merge-/CI-Stand, Branches, Worktrees und lokale
> Änderungen zuerst; erhalte ungesicherte Arbeit. Lies AGENTS.md, HANDOFF.md,
> PROJECT_STATUS.md und den dortigen gezielten Leseplan. Führe ausschließlich
> das vorbereitete Quellenpaket zur Erzeugungs-/Exportherkunft genau der
> gesperrten PANGAEA-EMPD2.sqlite3 aus. Erst offizielle Archivmetadaten, zweites
> Originalobjekt nur bei präzisem offiziellen Verweis; höchstens die drei
> bekannten Zeugen lesend prüfen. Keine neue Matrix/Kohorte, Datenbereinigung,
> Codeausführung, Fits, Scores, Simulationen oder Kernläufe. Nach zwei gezielten
> Quellen ohne Kette stoppen und Entscheidungspunkt dokumentieren.
> Bestehende Designs/Modelle/Resultate/Register/Importer/Flags/Fossilcounts/
> Gruppen, Baseline und CLI/App unverändert. W96 und reale Freigaben bleiben
> offen. Keine Nullfüllung/Rundung, Autorennachrichten/Merges/Unteragenten.
> Eigener Worktree/Branch/PR, Quellen-/Dokumentationsprüfung, kurze deutsche
> Updates; Ergebnis und genau einen nächsten Folgeauftrag im Repository sichern.

Absenden dieses Prompts autorisiert genau das neue begrenzte Archivpaket.

## Nur bei konkretem Bedarf weiterlesen

| Anlass | Nachzuladen |
| --- | --- |
| #32-Bänder, Schätzbarkeit oder Aussagegrenzen prüfen | [Bericht](./calibration_uncertainty_2026_10_30/README.md), danach bei Methodenarbeit DESIGN.md und SOURCES.md desselben Ordners |
| Kalibrierung #31 oder SMOTE #29 prüfen | [#31-Bericht](./calibration_2026_10_30/README.md), [#29-Bericht](./smote_nnet_2026_10_30/README.md); jeweiliges Vorabdesign vor Methodenänderungen |
| Große Ausgaben benötigen | [Artefaktverzeichnis](./ARTIFACTS.md), erst dann dort verlinkte REPRODUCE.md; keine alten temporären Dateien voraussetzen |
| Monticchio-Quellenarbeit | [präziser Quellenbedarf](./variable_provenance_2026_10_30/SOURCE_GAP.md); physischer LGM90D→MON16-Anschluss bleibt separat blockiert |
| Historische Sicherung oder frühere Aufträge klären | [Chat-Archiv](./CHAT_WORK_ARCHIVE_2026_10_30.md), [JSON-Inventar](./CHAT_WORK_ARCHIVE_2026_10_30.json), [frühere Einstiegstexte](./ENTRYPOINT_ARCHIVE_2026_10_31.md), [Integration #18–#29](./consolidation_2026_10_30/README.md) |

Archive bleiben erhalten, sind aber keine Pflichtlektüre bei jedem Start.
