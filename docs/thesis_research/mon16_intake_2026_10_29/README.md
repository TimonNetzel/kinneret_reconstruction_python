# MON16-Supplemente eingegangen und unabhängig geprüft

Stand 29.10.2026, nach Upload des original benannten ScienceDirect-Archivs.
**Der Beschaffungsblocker aus [PR #22](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/22)
ist behoben. P1 ist noch nicht abgenommen:** Das Paket verbindet die
LGM90D-Pollenproben nicht numerisch mit der MON16-Tiefenskala und enthält
klärungsbedürftige Quellenwidersprüche. Kein neues Alter wurde übernommen.

Basis bleibt der offene PR #18 (`f9acc16`). Claude-PR #19 hat weiterhin den
bereits geprüften Head `c8977a5`. Baseline und Integrationsdateien unverändert.
Die ergänzenden Ergebnisse aus PR #20/#21 bleiben gültig.

## Eingang und Umfang

Der Nutzer hat `ScienceDirect_files_29Oct2026_22-15-14.667.zip` bereitgestellt:
2.928.099 Bytes, SHA256
`2329fab3f4fc196174c70960cb9f1fddcbe9f9158ab099a3351193986f7ebdf8`.
CRC-Prüfung bestanden, genau drei Mitglieder. Größen und SHA256 jedes Originals
stehen in [summary.json](./summary.json). Die unveränderten Originale verbleiben
beim Nutzer; hier werden Prüfcode und abgeleitete Diagnosen eingecheckt.
Ein erneuter Serverdownload oder eine Prüfung gegen Serverbytes wird nicht behauptet.

| Datei | Tatsächlicher Inhalt | Prüfumfang |
|---|---|---|
| `1-s2.0-S0277379126003252-mmc1.xlsx` | A-1: Tephraübersicht; A-2: 16 Datierungsanker; A-3: Alter–Tiefe-Tabelle | Alle drei Blätter maschinell gelesen, A-1/A-2/A-3 numerisch abgeglichen |
| `1-s2.0-S0277379126003252-mmc2.xlsx` | Acht Blätter B-1 bis B-8 mit Tephra-, Standard- und Glaschemiedaten | Vollständiges Blatt-/Zellen-/Formelinventar; B-1 für Kernetiketten punktuell abgeglichen, keine neue geochemische Klassifikation |
| `1-s2.0-S0277379126003252-mmc3.docx` | C-1: Mikroskopiebilder; C-2: Kaliumkurven über Komposittiefe und Modellalter | OOXML-Text/Medien und beide gerenderten Seiten visuell geprüft |

C-2 ist **keine numerische Transformationsdatei** von physischer auf tephrenfreie
Tiefe. Das DOCX enthält drei Bilder und keine eingebettete Datendatei. Die
frühere Beschaffungsspezifikation hatte C-2 nach dem Methodenverweis des Artikels
als möglichen Nachweis der bereinigten Stratigraphie angefragt. Der tatsächliche
Anhang liefert grafische K-Kurven, keine Liste entfernter Lagen oder Zuordnung
zu LGM90D. Keine Kurve wurde digitalisiert oder zur Tiefenanpassung verwendet.

## Alters- und Tiefenskalen

A-3 heißt ausdrücklich **MON16 composite depth (mm)**, nicht Bacon-Modelltiefe.
Die 13.024 Datenzeilen liegen bei 3,5–13.137,9 mm; Medianalter
1.351,2–29.074,6 **cal BP**. Die gesamten Randgrenzen reichen von
1.100,4 bis 29.525,8 cal BP. Tiefen, Median, Minimum und Maximum fallen nirgends
ab (numerische Vergleichstoleranz 0,000001). Es gibt 37 Schritte ungleich
1 mm. [depth_steps.csv](./depth_steps.csv) erhält sie einzeln; keine Resampling-
oder Lückenschlussannahme. Ein millimeterfeines Ausgabegitter belegt keine
entsprechend feine zeitliche Information.

Die Breiten `Age_max - Age_min` betragen 1,0 / 852,8 / 1.525,7 Jahre
(Minimum / Median / Maximum über **alle MON16-Zeilen**). Diese Statistik ist
nicht die Altersunsicherheit der 69 alten Pollenproben und nicht mit deren
bisherigen Intervallen gleichzusetzen. A-3 enthält nur drei marginale
Alterszusammenfassungen, keine zusammengehörigen Posterior-Kurven.

[Wulf et al. (2026)](https://doi.org/10.1016/j.quascirev.2026.110116),
Abschnitte 3.1.2 und 5.2, beschreibt Bacon auf tephrenbereinigter Stratigraphie,
IntCal20/Marine20 und 2-Sigma-Grenzen. A-2 bestätigt 1 historischen,
7 terrestrische 14C-, 1 marinen 14C- und 7 bereits bayesianisch modellierte
Anker. D/E sind als publizierte kalibrierte Alter/Unsicherheiten beschriftet.
Sie werden weder erneut kalibriert noch pauschal als rohe 14C-Alter,
Standardabweichungen oder unabhängige Messungen behandelt. Eine vollständige
Bacon-Eingabe, Transformationsliste, Einstellungen und gemeinsame Ziehungen
liegen im Paket nicht vor.

## Quellenbefunde vor jeder Übernahme klären

Zwei A-3-Zeilen verletzen ihre eigene Beschriftung `min <= median <= max`:

| Excel-Zeile | Komposittiefe mm | Median cal BP | Minimum | Maximum | Median über Maximum |
|---|---:|---:|---:|---:|---:|
| 137 | 135,5 | 1.437,6 | 1.435,2 | 1.437,2 | 0,4 Jahre |
| 138 | 136,5 | 1.438,4 | 1.437,3 | 1.438,3 | 0,1 Jahre |

Dies sind kleine, reale Tabelleninkonsistenzen. Die Prüfung bestätigt die
Originalwerte mit zwei Lesern; ihre Ursache ist unbekannt. Kein Clipping,
Spaltentausch oder stilles Reparieren.

Weitere besonders relevante Unterschiede:

| Lage | Originalzellen | Befund und Konsequenz |
|---|---|---|
| TM-5b | A-1!B21, A-2!B10 | `B2, 65.8–66.2 cm` gegenüber `C2, 104.9–105.2 cm`. B-1!B22 unterstützt das erste Etikett; das zweite steht bei A-1!B22 für TM-5c. Das legt einen Übertragungsfehler nahe, ersetzt keine Quellenkorrektur. |
| TM-6b / Mercato | A-1!D29, A-2!H11 | Oberkante 504,7 gegenüber 512,5 cm bei gleicher Unterkante 515,4 cm. Daraus folgen 107 gegenüber 29 mm Mächtigkeit. |
| TM-6b / Mercato | A-1!F29:H29, A-3!A5019:E5054 | Beschriftete A-3-Gruppe hat Median 8.869,12 statt 8.943; Grenzen 8.659,20–9.047,76 statt 8.758–9.117 cal BP. Eine spätere **unbeschriftete** A-3-Plateaustrecke um 512,5–515,4 cm hat etwa 8.942,52 / 8.758,18 / 9.117,12. Das ist ein Hinweis auf einen Etikett-/Tiefenkonflikt, keine Erlaubnis zum automatischen Verschieben. |
| TM-8-1b | A-1!K49, A-2!L16 | MON12-Unterkante 878,5 gegenüber 875,5 cm. Ohne geklärte Zuordnung kein präziser Tiefenanker. |
| TM-6-5a-c | A-1 Zeilen 38/39, A-2 Zeile 13 | A-1 trennt MON12- und MON16-Lagen, A-2 stellt sie nebeneinander. Der Artikel, Abschnitt 5.3.2, behandelt ihre Identität ausdrücklich als ungeklärt. Gleicher Name reicht nicht zum Join. |

[source_conflicts.csv](./source_conflicts.csv) enthält **17 Vergleichsflags**, nicht
17 behauptete wissenschaftliche Fehler: vier Unterschiede zwischen A-1/A-2,
eine Mächtigkeitsdifferenz bei TM-1 (26 gegenüber rechnerisch 28 mm), zwölf
Alterszellen aus beschrifteten A-3-Gruppen mit mehr als 1,1 Jahren Abstand
zu A-1. Die 1,1-Jahre-Schwelle ist ein deskriptiver Filter gegen einfache
Ganzzahlrundung, keine Validitäts- oder Unsicherheitsgrenze. Kleinere Abweichungen
sind damit nicht automatisch erklärt. Bei Y-3 beginnt die Beschriftung bereits
vor der A-1-Oberkante; [labelled_age_groups.csv](./labelled_age_groups.csv) erhält
beide Altersgruppen. Keine bevorzugte Version wird aus Nähe zu erwarteten Altern
gewählt. Die genannten Befunde betreffen die Übernahme hier, nicht eine
abschließende Beurteilung des gesamten publizierten Modells.

## Entscheidung für jede der 69 Proben

[sample_gate.csv](./sample_gate.csv) erhält alle bisherigen IDs, Pollen- und
Chronologietiefen, Alter und den Kandidatenstatus aus dem exakt gesperrten
[PR-#21-Stand](https://github.com/TimonNetzel/kinneret_reconstruction_python/blob/170d77f2bdbbe60217dc8342f34beb252ccb67de/docs/thesis_research/chronology_crosswalk_2026_10_29/sample_correspondence.csv).
66 numerische Pollen-Kandidaten und drei offene Zuordnungen bleiben bestehen.
**Für alle 69 bleiben MON16-Tiefe und MON16-Alter leer.** Kein Blatt liefert
Originaletiketten dieser Pollenproben oder einen numerischen LGM90D→MON16-Join.

Auch MON12-Tiefen werden nicht als LGM90D-Tiefen ausgegeben. Die begrenzte
zusätzliche Quellenprüfung bestätigt eine ältere Mehrkern-Kompositkonstruktion
(Wulf et al. 2012, Abschnitt 3, Manuskript S. 4–5); die gelesene Passage liefert
keinen Proben-Crosswalk. Sie beweist nicht, dass ein solcher nirgends existiert.
Sieben bisherige Pilot-Medianalter liegen unter dem jüngsten A-3-Medianalter.
Das ist eine Warnung gegen einfache Extrapolation, keine altersbasierte Auswahl
oder Feststellung, dass genau diese Proben im MON16 fehlen.

## Konkreter verbleibender Eingang

1. **Eine belegte numerische Zuordnung LGM90D→älteres Komposit/MON12→MON16**:
   Proben-/Kernabschnitte, Tiefennullpunkte, Markeridentität und
   Korrelationsunsicherheit. Eine direkte LGM90D→MON16-Zuordnung wäre ebenfalls
   geeignet. Die neue Tephratabelle deckt nur einen Teil dieser Kette ab.
2. **Quellenklärung für die oben genannten Tabellenkonflikte**, vor allem
   Mercato und die zwei ungültigen A-3-Intervalle. Eine korrigierte Autorenversion
   muss als neuer, separat gehashter Eingang geprüft werden.
3. Für eine reproduzierbare Neuberechnung bzw. zeitliche Unsicherheitsfortpflanzung:
   Original-Bacon-Projekt einschließlich tephrenfreier Transformation und
   zusammengehöriger Modellkurven. Ohne Kurven können nur ausgewiesene marginale
   Angaben geprüft werden. Die bekannten übertragenen Fen-AMS-Anker bleiben
   gesondert abhängig (PR #21), keine zusätzlichen unabhängigen Anker.

Keine Anfrage an Autoren wurde versandt. P1 bleibt bis zur Tiefenzuordnung und
Quellenklärung offen; kein automatischer P2-/Claude-Pilotauftrag. Ein Pilot mit
unveränderter historischer Chronologie wäre eine ausdrücklich zu entscheidende
Begrenzung des bestätigten Ziels, keine stillschweigende Freigabe.

## Reproduktion und unabhängige Prüfung

Benötigt werden das originale Nutzer-ZIP und die unten festgelegte Datei aus
PR #21. Keine Datei aus einem vergangenen Chat wird vorausgesetzt:

```bash
git show 170d77f2bdbbe60217dc8342f34beb252ccb67de:docs/thesis_research/chronology_crosswalk_2026_10_29/sample_correspondence.csv > /pfad/pr21_sample_correspondence.csv
python tools/thesis_import/audit_mon16_supplements.py \
  --archive /pfad/ScienceDirect_files_29Oct2026_22-15-14.667.zip \
  --correspondence /pfad/pr21_sample_correspondence.csv --output-dir /pfad/neuer_lauf
python tools/thesis_import/verify_mon16_supplements.py \
  --archive /pfad/ScienceDirect_files_29Oct2026_22-15-14.667.zip \
  --correspondence /pfad/pr21_sample_correspondence.csv --output-dir /pfad/neuer_lauf
python -m unittest discover -s tools/thesis_import -p 'test_*.py'
```

Falls der Commit lokal fehlt, zuvor den PR-#21-Branch fetchen. Produzent:
Python-Standardbibliothek, OOXML/ZIP und Decimal. Unabhängiger Leser:
`openpyxl 3.1.5` als optionale Prüfabhängigkeit, eigene Float-Rechnung, kein
Import des Produzenten. Er prüft alle 13.024 A-3-Zeilen, Intervalle und
Monotonie, sämtliche 37 Tiefensprünge, 873 beschriftete Zeilen, 16 Anker,
alle 17 gemeldeten Vergleichsflags sowie alle 69 unveränderten Probengates.
Das ist ein unabhängiger **Tabellen-/Ergebnis-Audit**, kein Nachrechnen des
originalen Bacon-Modells. Zwei vollständige Produzentenläufe einschließlich
Manifest bytegleich. Sechs neue Fehlerszenario-Tests; alle 26 auf dieser Basis
vorhandenen Werkzeugtests bestanden. [verification.json](./verification.json).

Das DOCX wurde mit LibreOffice zu zwei PDF-Seiten gerendert; die PNG-Stufe des
Standardrenderers scheiterte an fehlendem `libpoppler.so.160`. Beide PDF-Seiten
anschließend mit PyMuPDF gerastert und visuell geprüft. Quellen, Einheiten und
Reproduktionsgrenzen: [sources.json](./sources.json).
