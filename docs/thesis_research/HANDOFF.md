# Netzel-Projekt: nächster Einstieg

## Gesichertes Ergebnis: synthetischer T1-Kandidat und begrenztes T2

Das autorisierte Paket aus PR #38 bei `d003fb9` wurde am 31.10.2026 auf
`codex/netzel-t1-synthetic-t2` im eigenen Worktree
`/workspace/netzel-t1-synthetic-t2` ausgeführt, ohne Unteragenten oder Merge.
[Ergebnis und Grenzen](./t1_synthetic_t2_2026_10_31/README.md),
[Vorabdesign](./t1_synthetic_t2_2026_10_31/DESIGN.md),
[Reproduktion](./t1_synthetic_t2_2026_10_31/REPRODUCE.md).
Design separat vor Implementierung/Lauf als `2b41c37` committed und gepusht.

108 synthetische Posteriore/38 Datensätze, alle geplanten Fälle berechnet;
29 neue Tests bestanden, separate Nachrechnung aller 1.681 Massen je Posterior
und bytegleicher frischer Zweitlauf bestanden. Widerspruch kann trotz engem
Posterior bestehen; Prior-only-/fehlende Dimension und Vorzeichenmehrdeutigkeit
sind sichtbar. Alle Varianten und verfehlten HPD-Wahrheiten berichtet.
Das ist **kein abgeschlossenes reales T1, kein SBC und keine fossile Freigabe**.
Der zusätzlich zu breit gestartete Standardtest wurde bei 56 bestanden/zwei
Browserfehlern abgebrochen; vorhandene kurze Regressionsketten mit realen
Testeingängen liefen teilweise. [Prüfstand](./t1_synthetic_t2_2026_10_31/TESTING.md).
Keine neue wissenschaftliche Kernkurve ausgewertet/gesichert; bestehende
Modelle, Resultate, Fossilcounts, Analog-Baseline und CLI/App unverändert.

[PR #39](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/39)
sichert das Paket gegen `codex/netzel-next-task-workflow` (#38), ohne Merge.
Letzter gesicherter Inhaltscommit: `953d1e995429a209b336cb83e89c41fda66a3f62`;
danach nur PR-/Übergabeverweise. CI war bei PR-Erstellung noch laufend,
MERGEABLE/UNSTABLE ist keine fachliche Freigabe.
Basis-/Remote-/Mergeprüfung: #31–#38 weiterhin offen; main bleibt `7f1826c`.
Kein ungemergtes Ergebnis als auf main vorhanden behandeln.

## Genau ein vorbereiteter Folgeauftrag – wartet auf Nutzer-OK

**T1-Anschlussdesign für Pollen-Zählkomposition und Typgemische, ohne neue Läufe.**

- **Ziel:** den noch fehlenden Modellvertrag zwischen kartiertem Areal,
  lokalem Vorkommen/Vegetation, Pollenfluss und beobachteter Zählkomposition
  mathematisch eingrenzen. Welche Größen wären aus welchen Eingängen überhaupt
  identifizierbar? Der binäre Spielzeugprototyp ist dafür keine fertige Brücke.
- **Erlaubter Umfang:** ausschließlich neue Designdokumente mit einer begründeten
  generativen Zählregel, konditionalen Abhängigkeiten und Datenrollen. Beobachtete
  Pollentypen von botanischen Arten unterscheiden; unbekannte Typ-/Art-Mischungen,
  Produktions-/Transportfaktoren und Restkategorie explizit führen. Verdeckte
  Vegetations-/Flussgrößen, Normalisierung zur Komposition und Stichprobenmodell
  mathematisch beschreiben; Voraussetzungen, nicht identifizierbare Parameter
  und minimale externe Belege mit Quellen/Annahmen trennen. Multinomial- oder
  zusätzliche Streuungsannahmen vorab begründen, keine Modellwahl durch Scores.
- **Konkrete Ergebnisse:** DESIGN_COUNTS.md mit Gleichungen und Datenvertrag;
  INPUT_GAPS.md mit vorhandenen/fehlenden Eingängen und Sperrgründen;
  festes Konzept für einen späteren kleinen synthetischen Zähltest mit bekannten
  Zuständen, Typgemischen, Nullcounts, Restkategorie, Gesamtzählung, Abhängigkeiten,
  Support und Unsicherheitsgrenzen. Diesen Test nur vorbereiten, nicht ausführen.
- **Leseplan:** zuerst AGENTS.md, PROJECT_STATUS und diese Übergabe sowie
  TAXON_PILOT_ROADMAP T1–T3. Danach Ergebnis/DESIGN/config dieses synthetischen
  Pakets; taxon_transfer_2026_10_30/DESIGN.md und config.json für die reale
  Funktionsbedeutung; pollen_methods_2026_10_31/README.md und SOURCE_GAP.md für
  Typgrenzen. Nur bei konkreter Gleichungs-/Datenfrage den Multiproxy-Plan oder
  weitere Originalquellen gezielt lesen; keine breite Wiederinventur.
- **Gesperrt:** bestehende Designs/Modelle/Resultate/Quellregister/Fossilcounts,
  Gruppierungen, Baseline, CLI/App; keine echte Artbrücke, reale Fits/Kernläufe,
  Integration, automatische Pilot-/Modellfreigabe, Autorennachrichten oder Merges.
  Keine neuen Unteragenten ohne ausdrücklichen Auftrag. Keine Wiederholung des
  gesamten Repository-Standardtests; Dokumentationsprüfung reicht hier.
- **Abschlusskriterien:** alle Likelihood-/Prior-/Mischungsterme einschließlich
  Normalisierung definiert; nicht identifizierbare Skalen/Typgemische algebraisch
  mit mindestens einem Gegenbeispiel belegt; fehlend gegenüber Nullcount sauber
  unterschieden; gemeinsame Support-/Unsicherheitsgrenzen und benötigte reale
  Belege genannt. Keine ungelöste Kartenlabel→Pollen-Brücke still als bekannt
  annehmen. Eigenen Worktree/Branch/PR, Status und genau nächsten Auftrag sichern.

Ein kurzes OK autorisiert genau dieses Designpaket nach erneuter Standprüfung.
Es erlaubt keine Implementierung oder neuen synthetischen/reellen Fits daraus.
W96 bleibt der **separat blockierte** Quellenauftrag unten; T0/T3/T4 und
Monticchio-P1 bleiben offen. Die nächste Designarbeit hebt diese Sperren nicht auf.

## Zuerst Stand prüfen

1. Arbeitsregeln und PROJECT_STATUS lesen, lokale/unversionierte Änderungen,
   Worktrees, Branches, relevante Remote-Heads und PR-/Mergezustände prüfen.
   Vorhandene Arbeit erhalten; kein Checkout/Reset fremder Worktrees.
2. Aktuellen gesicherten Inhaltscommit aus PROJECT_STATUS bzw. PR verwenden.
   #38-Basis `d003fb9` und Vorabdesign `2b41c37` sind historische Anker,
   nicht mit dem letzten Code-/Ergebniscommit verwechseln.
3. Für das vorbereitete Design nur den oben genannten Leseplan durcharbeiten.
   Alte Archive und große Ergebnisdateien erst bei konkretem Bedarf öffnen.

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

Ein Wechsel ist nach dem gesicherten Paketabschluss möglich, nicht erforderlich.
**Vorgeschlagener Startauftrag, noch nicht autorisiert:**

> Setze das Netzel-Projekt auf `codex/netzel-t1-synthetic-t2` aus PR #39 fort;
> letzter gesicherter Inhaltscommit 953d1e995429a209b336cb83e89c41fda66a3f62,
> danach nur PR-/Übergabeverweise (aktueller Branch-Head darf diese enthalten).
> Basis war PR #38 bei d003fb9, vorab gesichertes Design 2b41c37.
> Prüfe Git-/PR-/Merge-Stand, Branches, Worktrees und lokale Änderungen zuerst;
> erhalte ungesicherte Arbeit. Lies AGENTS.md, HANDOFF.md, PROJECT_STATUS.md,
> TAXON_PILOT_ROADMAP.md und den gezielten Leseplan der Übergabe.
> Führe ausschließlich den dort vorbereiteten Folgeauftrag aus:
> T1-Anschlussdesign für Pollen-Zählkomposition und Typgemische, ohne neue Läufe.
> Bestehende Modelle, Resultate, Register, Fossilcounts, Baseline und CLI/App
> unverändert. W96, reale Artbrücken und Monticchio-P1 bleiben offen.
> Keine äußeren Scores zur Wahl, keine Fits/Kernläufe, Autorennachrichten oder
> Merges. Eigener Worktree/Branch/PR, ohne Unteragenten, kurze deutsche Updates.
> Ergebnis und genau einen nächsten Folgeauftrag im Repository sichern.

Wenn der Nutzer diesen Prompt absendet, autorisiert er das oben begrenzte Paket.

## Nur bei konkretem Bedarf weiterlesen

| Anlass | Nachzuladen |
| --- | --- |
| #32-Bänder, Schätzbarkeit oder Aussagegrenzen prüfen | [Bericht](./calibration_uncertainty_2026_10_30/README.md), danach bei Methodenarbeit DESIGN.md und SOURCES.md desselben Ordners |
| Kalibrierung #31 oder SMOTE #29 prüfen | [#31-Bericht](./calibration_2026_10_30/README.md), [#29-Bericht](./smote_nnet_2026_10_30/README.md); jeweiliges Vorabdesign vor Methodenänderungen |
| Große Ausgaben benötigen | [Artefaktverzeichnis](./ARTIFACTS.md), erst dann dort verlinkte REPRODUCE.md; keine alten temporären Dateien voraussetzen |
| Monticchio-Quellenarbeit | [präziser Quellenbedarf](./variable_provenance_2026_10_30/SOURCE_GAP.md); physischer LGM90D→MON16-Anschluss bleibt separat blockiert |
| Historische Sicherung oder frühere Aufträge klären | [Chat-Archiv](./CHAT_WORK_ARCHIVE_2026_10_30.md), [JSON-Inventar](./CHAT_WORK_ARCHIVE_2026_10_30.json), [frühere Einstiegstexte](./ENTRYPOINT_ARCHIVE_2026_10_31.md), [Integration #18–#29](./consolidation_2026_10_30/README.md) |

Archive bleiben erhalten, sind aber keine Pflichtlektüre bei jedem Start.
