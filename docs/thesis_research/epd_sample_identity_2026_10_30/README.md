# Monticchio: EPD-Probenidentität und Tiefenpräzision

Stand 30.10.2026. Fortsetzung von [PR #23](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/23)
und [PR #24](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/24),
auf der unveränderten modernen Baseline aus PR #18. Kein neuer Klimamodelllauf.

## Ergebnis

Der vollständige EPD-Snapshot vom 31.10.2017 bestätigt die Datenbankkennungen
und genaueren Probentiefen der 69 Pilotproben. Er enthält für Entity 932 weder
Kernabschnitte noch Bohrsegmente. Damit ist eine bisher offene Frage zur
Exportpräzision beantwortet; die physische Zuordnung LGM90D → MON16 bleibt offen.

| Prüfung | Ergebnis | Beleg |
| --- | --- | --- |
| Gesamter Pollenexport | 245 Proben × 135 numerische Spalten, 33.075 Zellen ohne Differenz | `numeric_fingerprints.json`, `variable_export_map.csv` |
| Archivwerte | 8.333 gespeicherte Zeilen, davon 48 explizite Nullen | `summary.json` |
| Pilotidentität | Entity 932, Datenbank-Proben 4–72; alle 69 Tiefen exakt gleich EpdAgeCut 2013 | `sample_identity.csv` |
| Tiefenpräzision | 19/245 Exporttiefen weichen um ±5 mm vom Archiv ab; darunter 8/69 Pilotproben | `depth_rounding.csv` |
| Physische Probenmetadaten | 0 Kernabschnitte, 0 Bohrsegmente; Notizen und Mächtigkeit aller 245 Proben NULL | `archive_table_status.csv`, `summary.json` |
| Chronologie | Zwei ältere Modelle; das als Standard markierte Modell ist rückkalibriertes 14C-Alter | `legacy_age_comparison.csv` |
| Entscheidung | Keine neuen MON16-Alter; P1 bleibt offen | `summary.json`, `MISSING_SOURCE_REQUEST.md` |

## Was genau bestätigt wurde

Die Nummer `sample_` ist der Datenbankschlüssel innerhalb einer Entity, kein
nachgewiesenes ursprüngliches Labor- oder Kernabschnittsetikett. Die früher aus
Exportzeilen gebildeten IDs `LGM90D:000004` bis `LGM90D:000072` können jetzt mit
diesem Archivschlüssel verknüpft werden. Der Beleg umfasst die gesamte
135-spaltige Zahlenmatrix, nicht nur die acht Fingerprint-Taxa aus PR #21.
Die drei dort offenen Zuordnungen zur älteren Prozentzusammenfassung werden
dadurch **nicht** aufgelöst: Das ist eine andere Quellenverknüpfung.

127 Exportvariablen werden über identische Archivcodes und acht über explizite,
datensatzbezogene Aliaspaare verbunden. Diese Zuordnung ändert keine Shared37-
Taxonomie. Die 135 Spalten umfassen auch Labor-/Mengenfelder; 33.075 ist daher
die Zahl geprüfter numerischer Zellen, keine Zahl pflanzlicher Einzelzählungen.
Fehlende Tupel im dünn gespeicherten Zählungsarchiv werden ausschließlich für
diesen Exportvergleich als numerische Null gelesen. Das ist keine Imputation
fehlender Messungen. Explizite Archivnullen bleiben erhalten, obwohl das ältere
Handbuch positive Werte beschreibt.

Besonders aufschlussreich ist `Euc [#]`: Hier gehört die Spalte zu Archivvariable
2602 `Euc(ct)` mit der Bezeichnung „Eucalyptus spike (counted)“, nicht zur
botanischen Eucalyptus-Variable 295. Das bestätigt den bestehenden Ausschluss
dieses Labormarkers; die Baseline und ihre Gruppendefinition bleiben unverändert.

## Tiefen und ihre Grenzen

Das EPD-Handbuch 2007 definiert P_SAMPLE.DepthCM als Probenmittelpunkt in cm
(PDF-Seiten 30–31; Tabelle beginnt auf gedruckter Seite 27). Die gespeicherten
Dezimalwerte werden unverändert erhalten, beispielsweise 513,50 cm → 5,135 m.
Der PANGAEA-Export hat daraus 5,13 m. Weitere Pilotfälle stehen in der Tabelle.

| EPD-Probe | Archivmittelpunkt (m) | Exporttiefe (m) |
| --- | --- | --- |
| 25 | 5,135 | 5,13 |
| 27 | 7,245 | 7,25 |
| 29 | 7,465 | 7,46 |
| 30 | 7,525 | 7,53 |
| 66 | 10,835 | 10,84 |
| 67 | 11,235 | 11,23 |
| 68 | 11,335 | 11,34 |
| 69 | 11,435 | 11,44 |

Ein Diagnoseformat `format(float(depth_cm) / 100, '.2f')` reproduziert alle
245 Exporttiefen. Exaktes dezimales Half-even-Runden dagegen verfehlt 13 Werte.
Das ist mit binärer Gleitkommaformatierung vereinbar, beweist aber nicht den
historischen Exportalgorithmus. Gerundete Exportwerte lassen sich nicht eindeutig
zurückrechnen. Die jetzt belegten Archivmittelpunkte dürfen als solche verwendet
werden, nicht als neu bestimmte MON16-Komposittiefen oder Tephraober-/unterkanten.

Das Handbuch beschreibt COREDRIV und SECTION als Orte für physische Segment-
und Abschnittsinformationen (PDF-Seite 21). Beide Tabellen sind im Gesamtdump
vorhanden und anderswo befüllt, haben aber keine Zeile für Entity 932.
Auch Analyseabstands- und Segmenttabellen liefern hier keine Zeilen.
ENTITY.C14DepthAdj ist NULL; das bedeutet fehlende Information und keinen
nachgewiesenen Nulloffset zwischen Datierungs- und Pollenachse.

## Ältere Chronologien nicht als Ersatz einlesen

Beide archivierten Modelle wurden am 02.06.2007 vorbereitet und heißen
„Linear interpolation“. Chronologie 1 (`EPD calibrated`, Standard N) bezeichnet
ihre Grundlage als Kalenderalter aus Laminationen. Der Modellname belegt keine
unabhängigen kalibrierten Radiokarbondatierungen. Chronologie 2
(`EPD uncalibrated`, **Standard Y**) beschreibt eine Rückkalibrierung in 14C-Jahre
und einen Wechsel zurück zu Laminations-Kalenderaltern mit einem Sprung von rund
4.000 Jahren um 20 ka. Ihre Werte bilden deshalb keine durchgängig einheitliche
Altersachse für diesen Pilot.

Der Unterschied zwischen Chronologie 1 und den beibehaltenen EpdAgeCut-Altern
2013 beträgt für die 69 Proben im Betrag median 332,55 Jahre und maximal
674,16 Jahre. Das ist ein Vergleich verschiedener Modelle, kein Fehlermaß und
kein Unsicherheitsintervall. `legacy_age_comparison.csv` dient zur Vermeidung
eines versehentlichen Austauschs. Weder die vorhandenen Cut-Alter noch ihre
Grenzen werden verändert. Neuere Uploads und die offenen Mercato-/Markerfragen
aus PR #23/#24 werden durch den älteren Dump nicht entschieden.

## Unabhängige Prüfung und Reproduktion

Die Quelle ist der von der [offiziellen EPD-Seite](https://epdweblog.org/browse-get-data/)
verlinkte vollständige Datenbanksnapshot. Der Pfad enthält `2025/02`, das Archiv
selbst heißt `20171031` und seine drei Dump-Mitglieder tragen dieses Datum.
Die mitgelieferte README enthält ältere Vorlagendaten und wird nicht zur
Datierung des Snapshots verwendet. Der deklarierte SQL-Zeichensatz ist LATIN1.
Die SQL-Dateien werden als Text gelesen, **niemals ausgeführt**.

Der Erzeuger liest die separaten Tabellendumps samt Schema. Der separate Prüfer
liest stattdessen den vollständigen komprimierten SQL-COPY-Strom; er importiert
keinen Erzeugercode und vergleicht mit exakten Brüchen. Er kontrolliert alle
13 Tabelleninventare, alle 33.075 Zahlenzellen, 245 Fingerprints, 69 Tiefen-/
Altersverknüpfungen, 19 Rundungsfälle und sieben Ausgabedatei-Hashes.
Zwei Erzeugerläufe sind über alle acht Ausgaben bytegleich. 24 Tests der
Import-/Auditwerkzeuge bestanden, darunter vier neue Parser-/Präzisionstests.
`verification.json` hält diese Prüfungen und die Code-Hashes fest.

Alle fünf Eingaben stehen mit exakter Downloadadresse, Größe und SHA256 in
`sources.json`; keine Datei aus diesem Chat ist für die Reproduktion nötig.
Der PR-#21-Eingang ist auf Commit `170d77f2bdbbe60217dc8342f34beb252ccb67de`
fixiert. Von der Repository-Wurzel aus, mit zwei neuen Ausgabeverzeichnissen:

```sh
python tools/thesis_import/audit_epd_sample_identity.py --source-dir /path/to/inputs --output-dir /path/to/output
python tools/thesis_import/verify_epd_sample_identity.py --source-dir /path/to/inputs --output-dir /path/to/output
python -m pytest -q tools/thesis_import
```

Der Erzeuger prüft die fünf Quellhashes vor der Verarbeitung. Das Handbuch ist
ein gesperrter Provenienzeingang, kein automatisch ausgelesener Datenbestand;
die genannten Tabellen-/Tiefenbeschreibungen wurden am PDF überprüft.
Roharchive und PDF werden hier nicht weiterverteilt. Quellenlizenzen bleiben
maßgeblich: insbesondere PANGAEA.739452 CC BY-NC 3.0 und PANGAEA.804596 CC BY 3.0.

Ein ergänzender Abruf des bereits bekannten Neotoma-Datensatzes 27166 bestätigte
nur das isolierte TM16b-Ereignis außerhalb des Piloten. Die zusätzliche
Standortabfrage lief in ein Zeitlimit; daraus wird keine Abwesenheit weiterer
Neotoma-Daten abgeleitet. Die Datenbankprüfung und ihre Befunde hängen davon
nicht ab.
