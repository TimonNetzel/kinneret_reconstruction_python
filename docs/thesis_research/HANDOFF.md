# Netzel-Projekt: nächster Einstieg

## Gesichertes Ergebnis: T1-Anschlussdesign für Zählkomposition und Typgemische

Nach ausdrücklichem OK am 31.10.2026 **hier im bestehenden Chat** ausgeführt.
Basis `35185534bf7f171c6f39537ebbc0449b28f5421a` aus PR #39; eigener Worktree
`/workspace/netzel-counts-design`, Branch `codex/netzel-counts-observation-design`.
Keine Unteragenten. PR-/Inhaltscommit-Verweis steht in PROJECT_STATUS.
[Ergebnis](./counts_observation_design_2026_10_31/README.md),
[Gleichungen und Datenvertrag](./counts_observation_design_2026_10_31/DESIGN_COUNTS.md),
[reale Eingangslücken](./counts_observation_design_2026_10_31/INPUT_GAPS.md).

Das begrenzte Designpaket ist abgeschlossen: Fluss→Typkomposition→bedingte
Multinomial-Likelihood einschließlich Missingness/Nenner, gemeinsamer latenter
Abhängigkeit und proper Priors definiert. Skalen-, Produktions-/Klima- und
Typgemisch-Mehrdeutigkeit algebraisch belegt. Eine reale Counts-Rekonstruktionsregel
bleibt wegen fehlender Brückeneingänge unvollständig; W96, T0/T3/T4 und
Monticchio-P1 weiterhin offen. Keine neuen Fits, Simulationen oder Tests ausgeführt;
nur Dokumentations-/Hash-/Diffprüfung. Bestehende wissenschaftliche Dateien,
Fossilcounts, Baseline und CLI/App unverändert. Kein Merge/Autorenkontakt.

Der binäre synthetische Vorgänger bleibt als
[PR-#39-Ergebnis](./t1_synthetic_t2_2026_10_31/README.md) unverändert erhalten.
Sein dokumentierter zu breiter Standardtest wird nicht wiederholt.
main bleibt `7f1826c`; #31–#39 bei Einstieg offen. Aktuellen PR-/CI-Stand separat
prüfen; technische CI-Freigabe ersetzt weder Bestimmungsbeleg noch Pilotfreigabe.

## Genau ein vorbereiteter Folgeauftrag – wartet auf Nutzer-OK

**Kleinen synthetischen Zählprototyp nach dem gesicherten Plan implementieren und prüfen.**

- **Ziel:** die neue bedingte Counts-Likelihood und korrekte Behandlung von
  Typgemisch, Rest/Nenner, Missingness und gemeinsamer Probenabhängigkeit
  rechnerisch prüfen. Keine echte Art und kein fossiler Datensatz beteiligt.
- **Festgelegter Umfang:** ausschließlich neues Forschungsmodul und gezielte Tests,
  neue synthetische Ausgaben und Ergebnisdokumentation. 52 gültige Posteriorszenarien
  nach [SYNTHETIC_PLAN](./counts_observation_design_2026_10_31/SYNTHETIC_PLAN.md)
  und [synthetic_plan.json](./counts_observation_design_2026_10_31/synthetic_plan.json):
  bekannte Zustände bei N=8/40, beide Priors, gemeinsamer-H-Vergleich bei exakt
  gleichen Einzelkorn-Marginalen, zusammengelegte Typen, fehlende Gruppen,
  bedingter ausgewählter Nenner, alle fehlend, N=0 und beobachtete Nullcounts.
  Zusätzliche Negativfälle nur als solche, keine 52 scheinbar erfolgreichen Fälle
  durch Weglassen unzulässiger Ergebnisse erzeugen.
- **Leseplan:** AGENTS.md, PROJECT_STATUS, diese Übergabe und Roadmap T1/T2.
  Danach DESIGN_COUNTS.md, INPUT_GAPS.md, SYNTHETIC_PLAN.md und JSON dieses Pakets.
  Bei numerischen Fragen ausschließlich #39-DESIGN/Produzent/Prüfer als Referenz
  lesen; deren Dateien nicht ändern oder deren Ergebniswerte neu erzeugen.
- **Vor Lauf sichern:** vorhandenen Plan erhalten; neuen ausführbaren Quellstand
  vor erstem Ergebnislauf committen. Bekannte Planlücke zuerst dokumentieren,
  keine wissenschaftlichen Parameter/Seeds/Kriterien an Resultate anpassen.
- **Prüf-/Abschlusskriterien:** normierte PMFs durch Enumeration (N=8: 495 Vektoren),
  alle 52×441 Massen separat ohne Produzentenimport nachrechnen, Missingness-
  Marginalisierung, probenweite Mischung, algebraische Invarianten und explizite
  ungültige/impossible Fälle prüfen. Identifizierbarkeit/Mehrdeutigkeit und alle
  Verbesserungen/Verschlechterungen berichten, kein SBC-/Deckungsnachweis.
  Die fünf festgelegten Produktionsdateien (observations.json, results.json,
  posterior.csv.gz, environment.json, manifest.json) im frischen Zweitlauf
  bytegleich reproduzieren und alle Dateibytes in Git sichern. Neue Pfade:
  tools/taxon_counts/, tests/test_synthetic_counts.py und neues Ergebnispaket
  docs/thesis_research/counts_synthetic_t2_2026_10_31/.
- **Gesperrt:** alte Designs/Modelle/Resultate/Register/Fossilcounts/Gruppen,
  Baseline und CLI/App; echte Modelle laden, reale Fits/Kernläufe, automatische
  Art-/Pilot-/Modellwahl, neue äußere Scoreentscheidungen, Autorennachrichten/Merges.
  Keine Unteragenten ohne ausdrücklichen Auftrag. Kein gesamter Repository-
  Standardtest: nur neue gezielte synthetische Tests und Dokument-/Diffprüfung.
- **Sicherung:** eigener Worktree/Branch und PR gegen den aktuellen Designbranch;
  PROJECT_STATUS mit vollständigem Ergebnis/Prüfstand, HANDOFF mit genau einem
  nächsten begrenzten Auftrag. W96 und reale Eingangslücken gesondert erhalten.

Ein kurzes OK hier im Chat autorisiert genau dieses vorbereitete synthetische
Paket nach Standprüfung. Dieses Design-OK hat es noch nicht beauftragt.
Kein neuer Chat erforderlich; ein Wechsel ist bei Bedarf möglich.

## Zuerst Stand prüfen

1. Git-Status einschließlich unversionierter Dateien, Worktrees und Branches,
   relevante Remote-Heads, PR-/Merge-/CI-Zustand prüfen; fremde Arbeit erhalten.
2. Aktuellen Designbranch verwenden, nicht main: die Kette #31 ff. ist nicht
   automatisch zusammengeführt. Letzter gesicherter Inhaltscommit/PR steht
   in PROJECT_STATUS. Danach dürfen reine PR-/Übergabeverweise folgen.
3. Nur den gezielten Leseplan des vorbereiteten Pakets lesen. Historische
   Archive, große Ausgaben und reale Daten nur bei eigenem konkretem Auftrag.

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

**Vorgeschlagener Startauftrag, wartet auf OK; hier im Chat ebenfalls ausführbar:**

> Setze das Netzel-Projekt auf `codex/netzel-counts-observation-design` fort.
> Letzter gesicherter Inhaltscommit und PR stehen in PROJECT_STATUS.md;
> Basis war PR #39 bei 3518553. Prüfe Git-/PR-/Merge-/CI-Stand, Worktrees,
> Branches und lokale Änderungen zuerst; erhalte ungesicherte Arbeit.
> Lies AGENTS.md, HANDOFF.md, PROJECT_STATUS.md und den dortigen gezielten
> Leseplan. Führe genau den vorbereiteten synthetischen Zählprototyp-Auftrag
> nach counts_observation_design_2026_10_31/SYNTHETIC_PLAN.md und JSON aus:
> 52 vorab festgelegte Posteriorfälle plus Struktur-/Negativprüfungen.
> Neuen Quellstand vor Ergebnislauf sichern. Bestehende Designs, Modelle,
> Resultate/Register, Fossilcounts/Gruppen, Baseline und CLI/App unverändert.
> W96, reale Artbrücken und Monticchio-P1 bleiben offen. Keine echten Modelle,
> realen Fits/Kernläufe, äußere Scoreauswahl, Autorennachrichten oder Merges.
> Eigener Worktree/Branch/PR, ohne Unteragenten; nur neue gezielte Tests,
> keinen Gesamtstandardtest. Kurze deutsche Updates; Ergebnisbytes/Prüfstand
> und genau einen nächsten Folgeauftrag im Repository sichern.

Absenden dieses Prompts autorisiert genau das oben begrenzte neue Paket.

## Nur bei konkretem Bedarf weiterlesen

| Anlass | Nachzuladen |
| --- | --- |
| #32-Bänder, Schätzbarkeit oder Aussagegrenzen prüfen | [Bericht](./calibration_uncertainty_2026_10_30/README.md), danach bei Methodenarbeit DESIGN.md und SOURCES.md desselben Ordners |
| Kalibrierung #31 oder SMOTE #29 prüfen | [#31-Bericht](./calibration_2026_10_30/README.md), [#29-Bericht](./smote_nnet_2026_10_30/README.md); jeweiliges Vorabdesign vor Methodenänderungen |
| Große Ausgaben benötigen | [Artefaktverzeichnis](./ARTIFACTS.md), erst dann dort verlinkte REPRODUCE.md; keine alten temporären Dateien voraussetzen |
| Monticchio-Quellenarbeit | [präziser Quellenbedarf](./variable_provenance_2026_10_30/SOURCE_GAP.md); physischer LGM90D→MON16-Anschluss bleibt separat blockiert |
| Historische Sicherung oder frühere Aufträge klären | [Chat-Archiv](./CHAT_WORK_ARCHIVE_2026_10_30.md), [JSON-Inventar](./CHAT_WORK_ARCHIVE_2026_10_30.json), [frühere Einstiegstexte](./ENTRYPOINT_ARCHIVE_2026_10_31.md), [Integration #18–#29](./consolidation_2026_10_30/README.md) |

Archive bleiben erhalten, sind aber keine Pflichtlektüre bei jedem Start.
