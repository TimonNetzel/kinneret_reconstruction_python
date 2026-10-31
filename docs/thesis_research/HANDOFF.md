# Netzel-Projekt: nächster Einstieg

## Gesichertes Ergebnis: lesendes EMPD2-Counts-Readiness-Audit

Nach Nutzer-OK hier im Chat am 31.10.2026 ausgeführt. Basis PR #41 bei
`62ffeaf08ee7c7538f8d9c4cc99f78e7c0bdb394`; Worktree
`/workspace/netzel-empd2-readiness`, Branch `codex/netzel-empd2-counts-readiness`.
PR-/Inhaltscommit-Verweis steht in PROJECT_STATUS.
[Ergebnis](./empd2_counts_readiness_2026_10_31/MODERN_COUNTS_READINESS.md),
[Datenvertrag](./empd2_counts_readiness_2026_10_31/INPUT_CONTRACT.json),
[Originalbelege/Abfragen](./empd2_counts_readiness_2026_10_31/evidence.json).

Technische moderne Pollen-/Metadateneingänge sind vorhanden; **Counts-Training
ist noch nicht freigegeben**. Drei Zeugen/108 gespeicherte Messzeilen sowie
Schema-/Gruppen-/Code-Metadaten gelesen. Das cached EMPD2-Original ist vor/nach
read-only-Zugriff bytegleich zum Sperrhash, keine neuen Downloads.
Champvillair_a18: gebrochener Wert 7,4 ist Probenvolumen außerhalb flag-1-Gruppen;
244 historisch markierte Proben sind deshalb nicht automatisch 244 Proben mit
gebrochenen Pollenkornzahlen. Alte Flags/Referenzen bleiben unverändert.

Kein eigenes strukturiertes Proben-N im Schema. Arithmetische Gruppen- oder
Gesamtsummen sind keine bestätigten ursprünglichen Zählnenner; Gesamtwerte
können Volumen/Marker/weitere Messgrößen enthalten. Sparse-Nichttreffer bleiben
unaufgelöst, Text-NULL und Standortklasse X nicht repariert. Klimaperiode weiterhin
unverified, Typ-/Fossilbrücken weiterhin offen. Keine neue Matrix, Kohorte,
Simulation, Fits, Scores, Tests oder Kernrekonstruktion erzeugt/ausgeführt.

Vorherige Counts-T2-Ergebnisse und alle Designs/Modelle/Resultate/Register,
Fossilcounts/Gruppen, Baseline und CLI/App unverändert. Acht bisherige Worktrees
sauber erhalten; main `7f1826c`, #31–#41 bei Einstieg offen, #41-CI grün.
Keine Unteragenten, Merges oder Kontakte. Reales T1, W96/T0/Artbrücken,
T3/T4 und Monticchio-P1 bleiben offen.

## Genau ein vorbereiteter Folgeauftrag – wartet auf Nutzer-OK

**EMPD2s ursprüngliche Nenner- und Sparse-Verarbeitungsregeln gezielt belegen.**

- **Ziel:** die nun konkret belegte Lücke zwischen Tabellenflags und tatsächlichem
  Zähl-/Erfassungsprozess schließen, soweit zugängliche Originalstellen es tragen.
  Genaue Fragen/Begrenzung in [SOURCE_GAPS](./empd2_counts_readiness_2026_10_31/SOURCE_GAPS.md).
- **Begrenzte Quellen:** zuerst die registrierte EMPD2-Methodenpublikation
  Davis et al. (2020), DOI 10.5194/essd-12-2423-2020. Nur bei konkretem dortigem
  oder offiziellem Archivverweis ein zweites Objekt (ursprünglicher Code oder
  Datenwörterbuch) mit genauer Version lesen. Kein beliebiges späteres Paket
  als damals benutzt eintragen, keine breite Ersatzdatensuche/Autorenkontakte.
- **Regeln prüfen:** tatsächliche Anwendung von used_in_sum, percent_values und
  higher_groupid; belegte Bedeutung nicht gespeicherter Sample×Typ-Paare;
  originaler Zählnenner gegenüber gespeicherter arithmetischer Summe und bewusst
  ausgewähltem n. Klima-Periodenbeleg nur mit sichern, falls dieselbe gelesene
  Originalstelle ihn ausdrücklich liefert.
- **Leseplan:** AGENTS.md, PROJECT_STATUS, diese Übergabe, Roadmap T1–T3;
  MODERN_COUNTS_READINESS.md, INPUT_CONTRACT.json, SOURCE_GAPS.md und evidence.json
  dieses Pakets. Dann gezielt die Originalmethodik. Alter Importer und #35-
  Hash-/Reproduktionsnachweis nur für konkrete Datenvertrags-/Bytesfragen.
  W96-Quellenbedarf getrennt lassen, keine komplette neue Taxoninventur.
- **Umfang:** ausschließlich neues Quellen-/Regeldokumentpaket mit
  DENOMINATOR_RULES.md, RULE_EVIDENCE.json und verbleibendem SOURCE_GAPS.md.
  Quelle→Regel→Anwendung mit Seite/Abschnitt bzw. Code-Datei/Funktion/Zeile/Version
  belegen. Höchstens die drei vorhandenen Zeugen lesend nachvollziehen, keine
  neue Matrix, Probenkohorte, Harmonisierung oder fachliche Flagänderung.
- **Abschlusskriterien:** Nenner- und Sparse-Regeln belegt oder präzise weiter
  offen/blockiert; bibliographische Nennung nicht als Anwendung ausgeben.
  Zugangsblock/fehlenden Beleg erhalten. Keine angenommene Nullfüllung oder
  automatische Counts-Likelihood-Freigabe. Original-N, verfügbare Summe und
  bedingtes n explizit auseinanderhalten. Genau nächsten Auftrag vorbereiten.
- **Schutz/Sicherung:** bestehende Register/Importer/Flags/Designs/Modelle,
  Resultate/Counts/Gruppen, Baseline und CLI/App unverändert. Keine Fits, Scores,
  Simulationen, Kernel-/App-/Standardtests, Kernläufe, Merges, Kontakte oder
  Unteragenten. Eigener Worktree/Branch/PR, nur neue Dokumente und Status/Übergabe,
  Quellen-/Link-/Hash-/Diffprüfung. W96 und reale Freigabegates bleiben offen.

Ein erneutes OK hier im Chat autorisiert genau dieses neue Quellenpaket.
Der abgeschlossene Auditauftrag hat es noch nicht autorisiert. Kein Chatwechsel nötig.

## Zuerst Stand prüfen

1. Git-/PR-/Merge-/CI-Stand, Remote-Heads, Branches, Worktrees und lokale/
   unversionierte Änderungen prüfen; fremde Arbeit erhalten.
2. Aktuellen Auditbranch verwenden; Inhaltscommit/PR in PROJECT_STATUS, danach
   dürfen reine PR-/Übergabeverweise folgen. main enthält die Kette nicht automatisch.
3. Gezielten Leseplan verfolgen. Keine alten Import-/Modell-/Ergebnisläufe wiederholen.

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

> Setze das Netzel-Projekt auf `codex/netzel-empd2-counts-readiness` fort;
> letzter Inhaltscommit und PR stehen in PROJECT_STATUS.md, Basis war PR #41
> bei 62ffeaf. Prüfe Git-/PR-/Merge-/CI-Stand, Branches, Worktrees und lokale
> Änderungen zuerst; erhalte ungesicherte Arbeit. Lies AGENTS.md, HANDOFF.md,
> PROJECT_STATUS.md und den gezielten Leseplan. Führe ausschließlich den dort
> vorbereiteten Quellenauftrag aus: ursprüngliche EMPD2-Nenner- und Sparse-
> Regeln belegen, erste Quelle DOI 10.5194/essd-12-2423-2020, zweites Objekt
> nur bei konkretem offiziellen Verweis. Höchstens die drei bekannten Zeugen
> lesend nachvollziehen; keine neue Matrix/Kohorte/Harmonisierung oder Fits,
> Scores, Simulationen oder Kernläufe. Bestehende Designs, Register/Importer/
> Flags, Modelle/Resultate/Counts/Gruppen, Baseline und CLI/App unverändert.
> W96, Artbrücken und Monticchio-P1 bleiben offen. Keine automatische Nullfüllung,
> Counts-Freigabe, Autorennachrichten/Merges/Unteragenten. Eigener Worktree/
> Branch/PR, Dokumentationsprüfung, kurze deutsche Updates. Ergebnis und genau
> einen nächsten begrenzten Folgeauftrag im Repository sichern.

Absenden dieses Prompts autorisiert genau das neue begrenzte Quellenpaket.

## Nur bei konkretem Bedarf weiterlesen

| Anlass | Nachzuladen |
| --- | --- |
| #32-Bänder, Schätzbarkeit oder Aussagegrenzen prüfen | [Bericht](./calibration_uncertainty_2026_10_30/README.md), danach bei Methodenarbeit DESIGN.md und SOURCES.md desselben Ordners |
| Kalibrierung #31 oder SMOTE #29 prüfen | [#31-Bericht](./calibration_2026_10_30/README.md), [#29-Bericht](./smote_nnet_2026_10_30/README.md); jeweiliges Vorabdesign vor Methodenänderungen |
| Große Ausgaben benötigen | [Artefaktverzeichnis](./ARTIFACTS.md), erst dann dort verlinkte REPRODUCE.md; keine alten temporären Dateien voraussetzen |
| Monticchio-Quellenarbeit | [präziser Quellenbedarf](./variable_provenance_2026_10_30/SOURCE_GAP.md); physischer LGM90D→MON16-Anschluss bleibt separat blockiert |
| Historische Sicherung oder frühere Aufträge klären | [Chat-Archiv](./CHAT_WORK_ARCHIVE_2026_10_30.md), [JSON-Inventar](./CHAT_WORK_ARCHIVE_2026_10_30.json), [frühere Einstiegstexte](./ENTRYPOINT_ARCHIVE_2026_10_31.md), [Integration #18–#29](./consolidation_2026_10_30/README.md) |

Archive bleiben erhalten, sind aber keine Pflichtlektüre bei jedem Start.
