# Sechs belegte Marker-Verbindungen und präzisierte Ankerherkunft

Stand 30.10.2026. **P1 hat jetzt eine belegte Verbindung auf Marker-Ebene.**
Ein vollständiger Crosswalk für Pollenproben ist weiterhin nicht freigegeben.
Die sechs L4–L9-Bezeichnungen aus der alten LGM90D-Tephratabelle lassen sich
über Original-Kernabschnitte und Komposittiefen mit den heutigen TM-Labels
verbinden. ETH-12464 ist einer Originalstudie und Probe zugeordnet. Bei
TM-8-1b stützt eine historische Originaltabelle die MON12-Tiefe aus A-1.

Eigener Branch ab weiterhin offenem PR #18 (`f9acc16`), Claude-PR #19
unverändert auf `c8977a5`. Dieser Bericht ergänzt
[PR #23](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/23),
ohne seine Befunde oder die Originaldateien umzuschreiben.

## Neu bezogene Originale

- **Wulf, Dissertation 2000 / Scientific Technical Report 01/03 (2001)**:
  [Original-PDF beim SUB-Göttingen-Archiv](https://webdoc.sub.gwdg.de/ebook/serien/tm-v/STR/0103.pdf).
  185 PDF-Seiten; Abschnitt 3.1 erklärt die L-Nomenklatur nach Narcisi.
  Anhang A, PDF-Seiten **138/139**, enthält ausdrücklich Aliase, physische
  Kernabschnitte und **Komposit-Basistiefen**. Beide Seiten visuell geprüft.
- **Hajdas et al., Radiocarbon 40(2), 803–807**,
  [DOI 10.1017/S0033822200018750](https://doi.org/10.1017/S0033822200018750):
  Verlagspdf über die `citation_pdf_url` des Artikels tatsächlich bezogen.
  Tabelle 1 auf PDF-Seite 3 / gedruckt **805** visuell geprüft. Der Druck
  nennt 1998, das Verlagsportal 1997 und die Onlinebereitstellung 2016;
  keine dieser Jahresangaben ist eine neue Probenahme.

Sechs Eingabedateien sind durch Größe/SHA256 in [evidence.json](./evidence.json)
gesperrt. Neue PDFs und Original-Supplemente werden hier nicht weiterverteilt.
Quellenwege, Fundstellen und begrenzte Suchergebnisse: [sources.json](./sources.json).

## Verbindung der Marker

Die Identitäten beruhen auf **publizierten Namen/Aliasen**, nicht auf ähnlichen
Altern oder einem Fit von Klimakurven. Wulf 2001 benennt die L-Aliase;
Wulf 2004 Tabelle 2, im [VARDA-Datensatz](https://varve.gfz-potsdam.de/database/details/94dd95c2-dacf-46fc-aa8f-f8be21c09a8e?datasetId=7f7eeb65-a58b-4cf5-84f8-a34442200254)
mit diesen Aliasen transkribiert, verbindet sie mit den TM-Namen. Die heutige
A-1-Tabelle bestätigt die alten Komposit-Basistiefen bis auf Anzeigerundung.

| Alter Name | TM-Name | EPD-gelistete Tiefe cm | Historische Kompositbasis cm | MON16 Ober-/Unterkante cm | Original-Anhang-A-Zeile |
|---|---|---:|---:|---:|---:|
| L4 | TM-6b / Mercato | 662 | 668,49 | 504,7 / 515,4 | 32 |
| L5 | TM-7b | 787 | 789,04 | 650,6 / 654,4 | 47 |
| L6 | TM-8 / NYT | 862 | 864,53 | 720,9 / 723,3 | 57 |
| L7 | TM-9 | 882 | 883,38 | 739,2 / 741,0 | 60 |
| L8 | TM-12 | 1.016 | 1.015,50 | 857,7 / 869,9 | 70 |
| L9 | TM-13 | 1.120 | 1.117,10 | 938,7 / 956,2 | 75 |

Die MON16-Werte geben **A-1 unverändert** wieder. Für Mercato widerspricht
ihnen weiterhin die A-2-Oberkante 512,5 cm; die erste Tabellenzeile ist keine
Auswahl zugunsten von A-1. [marker_links.csv](./marker_links.csv) führt diese
Sperre und sämtliche Kernabschnitte ausdrücklich mit.

Bei NYT gehören die Original-Anhang-Zeilen 55–57 zu Unterlagen desselben
Tephrapakets; Zeile 57 benennt L6 und dessen untere Basis. Die 7 mm dieser
einzelnen Unterlage dürfen nicht mit der Mächtigkeit des gesamten Pakets
gleichgesetzt werden. Die Tabellen erhalten ihre jeweilige Granularität.

**Die Differenzen zwischen EPD-gelisteter Tiefe und historischer Basis sind
kein Tiefenoffset:** Sie reichen von −2,90 bis +6,49 cm; die Tiefe eines
Tephraeintrags kann sich auf eine andere Position innerhalb der Lage oder
auf eine andere Kern-/Kompositskala beziehen. Diese Definitionen sind für
die EPD-Einträge noch nicht ausreichend belegt. Daher kein konstanter
Offset, keine lineare Interpolation zwischen sechs Punkten, keine
Extrapolation und keine Normalisierung nach ähnlichen Alterswerten.

## Konsequenz für die 69 Pollenproben

[sample_marker_context.csv](./sample_marker_context.csv) ordnet ausschließlich
die **Zahlen auf der publizierten nativen Tiefenachse** ein:

- 39 Proben zwischen gelisteten Markertiefen;
- 1 Probe mit exakt gleicher Tiefenangabe: `LGM90D:000052`, 8,62 m / L6;
- 23 Proben oberhalb, 6 unterhalb des Bereichs L4–L9.

Dies sind keine freigegebenen Zeitintervalle. Auch die Zahlenkoinzidenz
beweist weder dieselbe physische Probe noch eine Pollenentnahme im Tephra-
Material. Alle 69 MON16-Tiefen und -Alter bleiben leer. Bestehende
Kandidatenstatus aus PR #21 werden erhalten, keine Teilmenge automatisch
für P2 ausgewählt. Die sechs Marker verbessern den Nachweis, ersetzen aber
keine Proben-/Kernabschnittsliste und keine Korrelationsunsicherheit.

## Quellenkonflikte eingegrenzt

| Gegenstand | Neuer Nachweis | Entscheidung |
|---|---|---|
| ETH-12464 | Hajdas Tabelle 1: Probe **LGM7.1**, Komposittiefe **1.382,5 cm**, 23.140 ± 360 konventionelle 14C-Jahre BP. Die Methoden beschreiben terrestrisches Material aus korrelierten B/C/D-Kernen. | Studie, Materialkontext und Probe identifiziert. EPD führt dieselbe Messung bei 1.385,0 cm: Differenz **2,5 cm** bleibt ungeklärt. Kein zweiter unabhängiger Anker; einzelner Ursprungskern/Abschnitt dieser Aliquote nicht belegt. Keine Neukalibrierung. |
| TM-8-1b, MON12 | Wulf Anhang A Zeile **58**, D4-u, 3,5 cm: Kompositbasis **878,47 cm**. | Numerische historische Stützung für A-1!K49 = 878,5 cm. A-2!L16 = 875,5 cm weicht um −2,97 cm ab. Stützung der A-1-Tiefe, keine autorisierte Quellenkorrektur oder neue geochemische Identifizierung. |
| Mercato / MON16 | Die neue historische Quelle betrifft das ältere Profil. | Sie entscheidet **nicht** über MON16-Oberkante oder die abweichend beschriftete A-3-Altersgruppe. Offen. |
| Zwei A-3-Intervalle | Keine korrigierte Autorenversion gefunden/bezüglich dieser Zellen bezogen. | Originalbefund aus PR #23 bleibt bestehen; Ursache unbekannt. |

Die ETH-Messung wird damit gegenüber PR #21 genauer eingeordnet. Die dort
geprüften CAMS-Daten bleiben aus dem Fen-Kern übertragene Messungen;
sie werden ebenfalls nicht zusätzlich als unabhängige Direktdatierungen
der EPD-gelisteten Tiefen gezählt.

## P1-Entscheidung und nächster konkreter Eingang

P1 bleibt offen, **jetzt mit sechs identifizierten Markern statt einer
vollständig unbelegten Marker-Verbindung**. Es fehlt noch:

1. Für die Originalpollen: Proben-/Kernabschnittstabelle einschließlich
   Bezeichnung der nativen Tiefenskala, Nullpunkt und Anschluss an das
   ältere B/D/E-Komposit. Besonders klären, welche oberen Proben aus
   welchen Teilkernen stammen. Die vorhandenen Zahlenpaare reichen nicht.
2. Dokumentierte Korrelation zwischen diesen Abschnitten und MON16 samt
   Unsicherheit zwischen den Markern. Die sechs publizierten Marker können
   dabei als konkret benannte Prüfpunkte dienen.
3. Quellenklärung zu MON16-Mercato und den zwei A-3-Intervallen. Ein
   originales Bacon-Projekt würde zusätzlich die tephrenfreie Transformation
   und gemeinsame Altersunsicherheit prüfbar machen.

Ohne diese Eingänge kein weiterer unveränderter Suchlauf als vermeintliche
P1-Abnahme. Ein Pilot auf unveränderter historischer Chronologie oder ein
anderer Standort bleibt eine gesondert zu treffende Zielentscheidung.
Keine Autorenanfrage wurde versandt, keine Integrationsdatei bearbeitet.

## Reproduzieren und prüfen

Alle sechs Dateien mit den **genauen Namen aus `evidence.json`** in ein
Eingangsverzeichnis legen. Die beiden neuen PDFs über die Links in
`sources.json` beziehen, die zwei PANGAEA-Originale über deren dortige URLs.
Die XLSX stammt aus dem Nutzer-ZIP. Die Probentabelle wird ohne Altdateien
aus dem festgehaltenen Git-Stand erzeugt:

```bash
git show 170d77f2bdbbe60217dc8342f34beb252ccb67de:docs/thesis_research/chronology_crosswalk_2026_10_29/sample_correspondence.csv > /pfad/eingang/pr21_sample_correspondence.csv
python tools/thesis_import/audit_marker_crosswalk.py --source-dir /pfad/eingang --output-dir /pfad/neuer_lauf
python tools/thesis_import/verify_marker_crosswalk.py --source-dir /pfad/eingang --output-dir /pfad/neuer_lauf --evidence docs/thesis_research/marker_crosswalk_2026_10_30/evidence.json
python -m unittest discover -s tools/thesis_import -p 'test_*.py'
```

Falls der #21-Commit lokal fehlt, vorher dessen Branch fetchen. Produzent:
Standardbibliothek/Decimal und vorhandener XLSX-Leser. Unabhängige Prüfung:
PyMuPDF liest neun Anhangszeilen in fünf physischen Spalten erneut aus;
openpyxl prüft die sechs XLSX-Verbindungen; Fraction/Bisect prüft alle
69 nativen Positionen. Kein Import des Produzenten. Die ETH-Altersspalte
fehlt im PDF-Textlayer und wurde am Seitenbild gelesen; sie wird nicht
als maschinell aus dem PDF extrahiert ausgegeben.

Zwei vollständige Läufe einschließlich Manifest bytegleich. Vier neue
Grenzfalltests und alle **24** Werkzeugtests auf dieser #18-Basis unter
unittest/pytest bestanden. Die Tests aus den noch getrennten Forschungs-PRs
sind dadurch nicht entfernt. Prüfbericht: [verification.json](./verification.json).
