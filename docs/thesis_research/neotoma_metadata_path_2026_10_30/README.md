# P1: gezielter Neotoma-Metadatenbeleg statt unveränderter Abrufe

**Nachtrag 30.10.2026:** Die Archiveingabe und ihre Analyseeinheitsnamen sind
inzwischen tatsächlich geprüft: [neuer Archivbericht](../neotoma_workshop_2026_10_30/README.md).
Dieser Bericht erhält den vorherigen Code-/Schemabefund mit dessen Grenzen.

Stand 30.10.2026. Fortsetzung des Quellenpfads 47525 aus
[PR #26](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/26).
P1 bleibt offen; keine Datenbankproben auf MON16 übertragen.

## Konkreter neuer Befund

Der öffentlich versionierte Neotoma-Code macht die benötigten Felder und
die Grenzen eines JSON-Exports genauer benennbar. In der gelesenen Definition
von `doi.ndbdata` stehen `samplename`, `analysisunitid`, `analysisunitname`,
`thickness`, `igsn` und `depth` im Probenobjekt. **Die freien Notizen aus
`ndb.samples` und `ndb.analysisunits` fehlen in diesem Objekt**, obwohl beide
Tabellen solche Felder besitzen. Ein unbeschrifteter JSON-Eintrag wäre daher
kein Nachweis, dass sämtliche Herkunftsangaben der Datenbank leer sind.

Das ist ein **Quellcode-/Schemabefund**, kein Befund an den Proben von 47525.
Ob dort Namen oder Notizen vorhanden sind und was sie bedeuten, ist unbekannt.
Ein Name ist nicht automatisch ein ursprüngliches physisches Probenschild.

| Benötigte Information | Publizierter Speicherort | Gelesenes Proben-Exportobjekt |
| --- | --- | --- |
| Datenbankschlüssel, Probenname | `samples.sampleid`, `samples.samplename` | Enthalten |
| Analyseeinheit, Name | `analysisunits.analysisunitid`, `.analysisunitname` | Enthalten |
| Tiefe, Mächtigkeit | `analysisunits.depth`, `.thickness` | Enthalten; Schema nennt cm |
| IGSN | `analysisunits.igsn` | Enthalten; tatsächliche Belegung unbekannt |
| Laborbezeichnung, Probennotizen | `samples.labnumber`, `.notes` | Nicht enthalten |
| Notizen zur Analyseeinheit | `analysisunits.notes` | Nicht enthalten |
| Komposit-/Kernkontext | Dataset-/Collection-Unit-Namen, Handle, Ort, Notizen | Separat anfordern und zusammen mit Proben prüfen |

Die Schemabeschreibung nennt Tiefen normalerweise Mittelpunkte. Sie belegt
damit weder den tatsächlichen Tiefennullpunkt dieses Datensatzes noch seine
Identität mit LGM90D oder MON16. Datenbank-IDs sind keine physischen Kernetiketten.

## Versions- und Downloadgrenzen

Drei getrennte offizielle Codebestände wurden auf Commits gesperrt:
API `c57b83f`, Landingpage `cb8cd13`, Datenbankdefinitionen `359c92c`.
Sie sind **kein nachgewiesener gemeinsam deployter Serverstand**:
Der gelesene API-Handler erwartet unter `x.data.data` auch ein `site`-Objekt;
die gelesene `doi.ndbdata`-Definition baut am Ende nur `samples` auf.
Daher keine Behauptung, diese Kombination sei ausführbar oder beschreibe
vollständig die derzeitige Serverantwort. Die Live-Exportabdeckung bleibt
an tatsächlichen Downloadbytes zu bestätigen.

Der Landingpage-Downloadknopf ruft die bekannte `/downloads/{datasetid}`-API
auf und speichert `JSON.stringify(data.data)`. Er ist damit kein separat
belegtes Rohdatenarchiv. Die Datei ist neu serialisierter Antwortinhalt;
ein anderer Bytehash als der vollständigen HTTP-Antwort wäre allein noch
kein inhaltlicher Datenwiderspruch.

Zusätzlich wurde die im offiziellen Router nachgewiesene Route
`/v2.0/data/frozen/47525` einmal geprüft. Ihr Handler liest `doi.frozen.download`;
der Webleser konnte die Route nicht öffnen. Das beweist weder, dass ein
archivierter Stand fehlt, noch dass er die gesuchten Felder enthält.
Die Snapshot-Verweise lieferten HTTP 403. Kein neuer Rohdateneingang bezogen.
Der bereits erfolglose normale Download wurde nicht unverändert wiederholt.

## Jetzt konkret benötigter Eingang

[metadata_extract.sql](./metadata_extract.sql) ist eine nur lesende, auf
Datensatz **47525** begrenzte Abfrage für einen passenden Neotoma-Snapshot
oder einen fachlich autorisierten Datenhalter. Sie fordert Proben- und
Analyseeinheitsnamen, Laborfelder und **alle vier Ebenen der Notizen**
(Dataset, Collection Unit, Probe, Analyseeinheit) zusammen mit den Schlüsseln an.
Fehlende oder widersprüchliche Verknüpfungen bleiben sichtbar. Keine Abfrage
an eine Live-Datenbank ausgeführt, keine Anfrage versandt.

Mit dem Ergebnis zuerst Belegung und Bedeutung der physischen Etiketten
prüfen. Ein späterer Vergleich mit Entity 932 benötigt zusätzlich die
Zählungen und eine geprüfte Taxon-/Variablenzuordnung. Gleiche Namen, Alter,
Tiefen oder Reihenpositionen allein reichen nicht. Für MON16 bleibt darüber
hinaus ein belegter Abschnittsanschluss mit Korrelationsunsicherheit nötig.
Weder Exportcode noch SQL-Abfrage ersetzen die Original-Kompositierungsliste.

P1 ist bis zu einem solchen echten Eingang blockiert. Kein erneuter
unveränderter API-/EPD-Suchlauf als Fortschritt. Falls der Eingang nicht
beschaffbar ist, ist die bereits vorgesehene Entscheidung über Zurückstellen
des Piloten oder ausdrücklich begrenztes Weiterarbeiten vorzubereiten.
Die taxonbasierte Kalibrierung bleibt ein davon getrenntes Arbeitspaket.

## Prüfung und Reproduktion

[sources.json](./sources.json) nennt alle 15 gelesenen Quellcodedateien mit
Commit, Git-Blob-SHA, Bytezahl und SHA256. Die aus den gelesenen UTF-8-Inhalten
berechneten Git-Blob-Hashes stimmen mit den GitHub-Blobs überein. Die Originale
werden nicht als angebliche Monticchio-Daten weiterverteilt.

Die Abfrage wurde gegen diese Tabellen-/Spaltendefinitionen geprüft und auf
synthetischen SQLite-Tabellen ausgeführt: Filter gegen Fremddatensätze,
Erhalt von NULL und Notizen, sichtbare fehlende/widersprüchliche
Analyseeinheitsverknüpfungen und Dataset ohne Proben. **Kein PostgreSQL-
Integrationstest und keine Prüfung realer 47525-Beobachtungen.**

```sh
python -m unittest discover -s docs/thesis_research/neotoma_metadata_path_2026_10_30 -p test_metadata_extract.py -v
```

[verification.json](./verification.json) hält den abgeschlossenen Umfang fest.
Baseline, vorhandene Alter und Claudes CLI-/App-Dateien unverändert.
Keine Korrektur der MON16-Supplemente und keine Fossilpilot-Freigabe.
