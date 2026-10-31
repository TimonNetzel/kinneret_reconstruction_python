# Netzel-Projekt: nächster Einstieg

## Gesichertes Ergebnis: begrenztes synthetisches T2 für Counts

Nach Nutzer-OK am 31.10.2026 hier im Chat ausgeführt. Basis PR #40 bei
`19a75426320d0143c09cd00c756bc31f97cef0d0`; neuer Worktree
`/workspace/netzel-counts-t2`, Branch `codex/netzel-synthetic-counts-t2`.
[Ergebnis](./counts_synthetic_t2_2026_10_31/README.md),
[Prüfergebnis](./counts_synthetic_t2_2026_10_31/verification.json),
[Reproduktion](./counts_synthetic_t2_2026_10_31/REPRODUCE.md).
PR-/Inhaltssicherung wird in PROJECT_STATUS geführt.

52/52 geplante gültige Posteriore aus 25 künstlichen Datensätzen berechnet,
zwei zusätzliche unmögliche Fälle explizit nicht schätzbar. 51 gezielte Tests,
separate Nachrechnung aller 52×441 Massen und zwölf PMF-Normierungen bestanden;
fünf Produktionsdateien im frischen Zweitlauf bytegleich. Quellstand vor erstem
Ergebnislauf gesichert (`5b7e67a`, Test-Syntaxreparatur `220ef55`). Kein anderer
Test-/Sampler-/App-Lauf gestartet und keine alten Ergebnisse neu erzeugt.

Mehr Counts verbessern in diesem kleinen base-Inventar den mittleren Fehler;
der informative Prior verschlechtert dort den mittleren Fehler, verkleinert aber
HPD-Sets. Das missspezifizierte je-Korn-H-Modell liegt bei einem einzelnen
Datensatz näher an der Wahrheit als die korrekte gemeinsame-H-Inferenz.
Beide berichtet, keine Modellwahl. Zusammengelegte Typen bleiben vorzeichen-
mehrdeutig; Priorabhängigkeit sichtbar. Kein SBC-/Deckungsnachweis.
Reale Brücken-/Produktions-/Taxonomie-/Chronologieunsicherheit bleibt offen.

Reales T1, W96/T0/Artbrücken, T3/T4 und Monticchio-P1 nicht freigegeben.
Bestehende Designs, Modelle, Resultate/Register, Fossilcounts/Gruppen,
Baseline und CLI/App unverändert. Sieben vorherige Worktrees erhalten;
main `7f1826c`, #31–#40 bei Einstieg offen, #40-CI grün. Keine Unteragenten,
Merges oder Autorennachrichten. Bisherige Design-/Quellenberichte unverändert.

## Genau ein vorbereiteter Folgeauftrag – wartet auf Nutzer-OK

**Einen modernen Counts-Kalibrierungsweg auf Basis EMPD2 lesend auditieren.**

- **Ziel:** über weitere Spielzeugläufe hinaus prüfen, ob die vorhandenen
  modernen Pollen-/Metadaten einen konkreten Datenvertrag für eine spätere
  direkte Typkompositions-Kalibrierung tragen könnten. Dies ist keine Übernahme
  dieser Methode und ersetzt keine Kartenlabel→Vegetation-Brücke oder fossile
  Typbestimmung. Die sieben bisherigen Arealfunktionen bleiben Referenzen.
- **Begrenzte Eingänge:** vorhandener EMPD2-Import/Quellennachweis und
  original_pollen_names_2026_10_31/REPRODUCE.md/Manifest. Nur falls nötig das dort
  identifizierte Original mit bestehendem Hashbeleg heranziehen; vorhandene Bytes
  bevorzugen, keine breite Suche nach Ersatzdaten. Schema/Metadaten und höchstens
  wenige genau bezeichnete Proben zur Count-/Nennerdarstellung lesen. Keine
  neuen Pollenmatrizen exportieren, Fossilcounts oder Gruppen ändern.
- **Konkrete Fragen:** Gibt es pro Probe ganzzahlige Typcounts und belegten
  Nenner oder nur Häufigkeiten? Welche Typkonzepte, Ausschlüsse, Missingness,
  Original-/akzeptierten Namensbeziehungen und Restkategorien sind erhalten?
  Gibt es belastbare Koordinaten/Zeitbezüge und einen dokumentierbaren modernen
  Klimaanbindungsweg? Welche Schritte würden neue unabhängige Information
  benötigen? Keine Existenz oder Eignung behaupten, bevor konkret gelesen.
- **Ergebnisse:** MODERN_COUNTS_READINESS.md mit exakten Tabellen-/Feld-/Quellen-
  fundstellen, INPUT_CONTRACT.json für verfügbare/fehlende Datenrollen und
  SOURCE_GAPS.md mit einem eng begrenzten nächsten Schritt. Eine direkte moderne
  Typkompositionsfunktion von Areal-, Präsenz- und Produktionsmodellen trennen.
  Trainings-/Bestätigungsrollen, Support und Unsicherheit nur prospektiv skizzieren;
  keine realen Fits, Scores, räumlichen Auswertungen oder Kernrekonstruktionen.
- **Leseplan:** AGENTS.md, PROJECT_STATUS, diese Übergabe, Roadmap T1–T3;
  aktueller T2-Bericht, danach counts_observation_design_2026_10_31/DESIGN_COUNTS.md
  und INPUT_GAPS.md. Anschließend gezielt bestehende EMPD2-Importerbeschreibung/
  Quellmanifest, #35-Reproduktionsnachweis und pollen_methods_2026_10_31/README.md
  samt SOURCE_GAP.md. Archive nur bei konkreter Herkunftsfrage; keine erneute
  komplette Namensinventur oder W96-Ersatzsuche.
- **Abschlusskriterien:** alle Aussagen über Eingänge an tatsächlich gelesenen
  Objekten belegen; Counts/Nenner/Typzuordnung und Klimarolle getrennt bewerten;
  Nichtverfügbarkeit klar sperren. Keine Prozent→Pseudo-Counts, breiter Typ→Art-
  Gleichsetzung oder Nutzung bekannter äußerer Scores. Reale Kalibrierungs-
  Bereitschaft darf ausdrücklich negativ ausfallen. Genau einen anschließenden
  Auftrag vorbereiten; keine neue Pipeline oder wissenschaftliche Freigabe.
- **Schutz/Sicherung:** ausschließlich neue Auditdokumente und Status/Übergabe,
  eigener Worktree/Branch/PR. Alle bestehenden Designs/Modelle/Resultate/Register,
  Fossilcounts/Gruppen, Baseline und CLI/App unverändert; keine Simulationen,
  Standardtests, Fits/Kernläufe, Autorennachrichten/Merges oder Unteragenten.
  Dokumentations-, Link-/Hash-/Diffprüfung genügt.

Ein erneutes OK hier im Chat autorisiert genau dieses lesende Auditpaket.
Der synthetische T2-Auftrag hat es noch nicht autorisiert. Kein Chatwechsel nötig.
W96 und reale T0/T1/T3/T4-Gates bleiben separat offen; keine Kernwahl geändert.

## Zuerst Stand prüfen

1. Lokale/unversionierte Änderungen, Worktrees, Branches und relevante
   Remote-Heads samt PR-/Merge-/CI-Zustand prüfen; fremde Arbeit erhalten.
2. Aktuellen Ergebnisbranch verwenden; letzter gesicherter Inhaltscommit/PR steht
   in PROJECT_STATUS. main enthält die ungemergte Kette nicht automatisch.
3. Nur den gezielten Leseplan des neuen Pakets verfolgen; keine Ergebnisläufe
   dieses abgeschlossenen Pakets oder der alten Modelle wiederholen.

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

**Vorgeschlagener Startauftrag, wartet auf Nutzer-OK; auch hier ausführbar:**

> Setze das Netzel-Projekt auf `codex/netzel-synthetic-counts-t2` fort; letzter
> gesicherter Inhaltscommit und PR stehen in PROJECT_STATUS.md. Basis war
> PR #40 bei 19a7542. Prüfe Git-/PR-/Merge-/CI-Stand, Branches, Worktrees und
> lokale Änderungen zuerst, erhalte ungesicherte Arbeit. Lies AGENTS.md,
> HANDOFF.md, PROJECT_STATUS.md und den dortigen gezielten Leseplan.
> Führe ausschließlich das vorbereitete lesende Audit eines modernen
> Counts-Kalibrierungswegs auf Basis EMPD2 aus. Nur gezielte Schema-/Metadaten-
> und wenige belegte Proben-/Nennerprüfungen; keine neue Matrixauswertung,
> Simulationen, Fits, Scores oder Kernläufe. Bestehende Designs, Modelle,
> Resultate/Register, Fossilcounts/Gruppen, Baseline und CLI/App unverändert.
> W96, reale Artbrücken und Monticchio-P1 bleiben offen. Keine äußere Scorewahl,
> Autorennachrichten/Merges/Unteragenten. Eigener Worktree/Branch/PR, nur
> Dokumentationsprüfung, kurze deutsche Updates. Ergebnis und genau einen
> nächsten begrenzten Folgeauftrag im Repository sichern.

Absenden dieses Prompts autorisiert genau dieses neue lesende Auditpaket.

## Nur bei konkretem Bedarf weiterlesen

| Anlass | Nachzuladen |
| --- | --- |
| #32-Bänder, Schätzbarkeit oder Aussagegrenzen prüfen | [Bericht](./calibration_uncertainty_2026_10_30/README.md), danach bei Methodenarbeit DESIGN.md und SOURCES.md desselben Ordners |
| Kalibrierung #31 oder SMOTE #29 prüfen | [#31-Bericht](./calibration_2026_10_30/README.md), [#29-Bericht](./smote_nnet_2026_10_30/README.md); jeweiliges Vorabdesign vor Methodenänderungen |
| Große Ausgaben benötigen | [Artefaktverzeichnis](./ARTIFACTS.md), erst dann dort verlinkte REPRODUCE.md; keine alten temporären Dateien voraussetzen |
| Monticchio-Quellenarbeit | [präziser Quellenbedarf](./variable_provenance_2026_10_30/SOURCE_GAP.md); physischer LGM90D→MON16-Anschluss bleibt separat blockiert |
| Historische Sicherung oder frühere Aufträge klären | [Chat-Archiv](./CHAT_WORK_ARCHIVE_2026_10_30.md), [JSON-Inventar](./CHAT_WORK_ARCHIVE_2026_10_30.json), [frühere Einstiegstexte](./ENTRYPOINT_ARCHIVE_2026_10_31.md), [Integration #18–#29](./consolidation_2026_10_30/README.md) |

Archive bleiben erhalten, sind aber keine Pflichtlektüre bei jedem Start.
