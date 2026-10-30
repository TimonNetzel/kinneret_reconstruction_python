# Monticchio: gespeicherte Probenmetadaten gefunden und geprüft

Stand 30.10.2026. Fortschreibung der Quellen- und Exportprüfung in PR #26.
**Neuer echter Eingang:** das öffentliche `itDownload.RDS` aus dem offiziellen
Neotoma-Workshopbestand. Ein vollständiger Live-API-Download ist weiterhin
nicht nachgewiesen. Die Ergebnisse gelten für die hier gesperrten Archivbytes.

## Ergebnis

| Prüfung | Ergebnis |
| --- | --- |
| Tatsächlich eingelesener Archivbestand | 59 Standorte; Monticchio gezielt vor der Klassenkonvertierung ausgewählt |
| Monticchio | 13 Collection Units / Pollendatensätze im gespeicherten Objekt |
| LGM90D | Neotoma **26608**, 245 Proben; keine Analyseeinheitsnamen im Export belegt |
| Neues untersuchtes Komposit | Neotoma **47525**, 924 Proben; **alle 924 mit Analyseeinheitsnamen** |
| Anschluss an die ältere Quelle | 245 Proben mit Tiefe in Exportpräzision und fünf unveränderten Gattungsfeldern abgeglichen: 1.225 Zahlenzellen gleich; vier unterschiedliche Listenpositionen erhalten |
| Vergleich der 69 Pilotproben mit 47525 | **28 eindeutige numerische Kandidaten**, 41 ohne vollständigen Treffer auf den 76 gemeinsam vergleichbaren Variablenschlüsseln |
| Diagnose der 41 offenen Fälle | Je ein nächster Eintrag; 35 Fälle mit einer und sechs mit zwei abweichenden Zellen. Insgesamt 47 ungeklärte Zellunterschiede |
| MON16 | Keine Tiefen oder Alter übernommen; physischer Anschluss und Korrelationsunsicherheit weiterhin nicht abgenommen |

Die 28 Kandidaten tragen gespeicherte Bezeichnungen wie `E1-u, 35.5`,
`D3-u, 50`, `B4-u, 45` und `D4-o, 90`. Damit sind erstmals in diesem
Quellenpfad konkrete Abschnittsbezeichnungen zu numerischen Kandidaten für
Pilotproben greifbar. Das ist noch **kein vollständig bestätigter physischer
Proben-Crosswalk**: Die Suche vergleicht nur einen Teil der Variablen, und
die Herkunft bzw. Einheit des Zahlenanteils in einem Etikett muss am
Methoden-/Kompositierungsbeleg geprüft werden. Ein Datenbankname ist nicht
automatisch ein unabhängig geprüftes ursprüngliches Probenschild.

## Der tatsächlich beschaffte Quellenstand

[NeotomaDB/Workshops, Commit 5caa58c](https://github.com/NeotomaDB/Workshops/blob/5caa58c9da2b23fd174ec3d785f3dd931298a3d6/2023_INQUA_pollen_vertebrates/data/itDownload.RDS)
enthält eine komprimierte R-Serialisierung: **2.495.691 Bytes**, SHA256
`03536b6f1d3974de77ff997f6d86674a78882b2ebe305bb4e135e38289a2a829`.
Git-Blob `eaa1e04645575ecd13ba5b2432e877ae57848db1` wurde aus den bezogenen
Originalbytes nachgerechnet. Die Workshopdatei `simple_workflow.Rmd` liest
diesen Bestand ausdrücklich statt eines neuen Downloads. Der Ordnername 2023
und ein späterer Verschiebecommit belegen nicht das genaue Abfragedatum.
Keine Identität mit dem aktuellen Neotoma-Serverstand behauptet.

Der normale GitHub-Dateileser lieferte für die große Binärdatei leeren Inhalt;
der Blobleser konnte sie nicht als UTF-8 dekodieren. Ein Browserdownload gelang,
war aber nicht in den Rechenarbeitsbereich synchronisiert. Der danach direkt
bezogene GitHub-Raw-Download lieferte die oben verifizierten Bytes. Leere
Connector-Antworten wurden nicht als leere Datenbank gewertet. Kein erneuter
unveränderter Neotoma-API-Abruf.

Die Quelle **47525**, DOI [10.21233/KVNC-0856](https://doi.org/10.21233/KVNC-0856),
bezeichnet sich als Komposit der Kerne P90, B, D, E, MF, J, K, L, M und O.
Sie ist weder das isolierte TM16b-Ereignis 27166 noch MON16.
Der im zweiten Workshop verwendete Datensatz **4275 ist MONT82**, nicht LGM90D
oder 47525. Seine dort demonstrativ veränderten Datierungskontrollen und sein
neues Bchron-Modell wurden nicht übernommen oder ausgeführt.

## Was genau verglichen wurde

**Erster Anschluss: ältere Quelle → Neotoma 26608.** Die erneut bezogene
PANGAEA.739452-Datei stimmt mit ihrem alten Quellhash überein. Abies, Alnus,
Artemisia, Betula und Fagus wurden unverändert mit den gleichnamigen nativen
Neotoma-NISP-Feldern verglichen. Zusammen mit der zweistelligen Exporttiefe
ergibt sich ein eindeutiger Anschluss aller 245 Zeilen. Die vier vertauschten
Listenpositionen bleiben in [legacy_source_alignment.csv](./legacy_source_alignment.csv)
sichtbar. Dies ist ein Fünf-Feld- und Präzisionsvergleich, kein neuer Vollvergleich
der 135 EPD-Spalten. Die 69 genauen Mittelpunkte und EPD-Kennungen stammen
unverändert aus dem festgehaltenen PR-#25-Eingang.

Bei **16 Quercus-Zeilen** unterscheidet sich der ältere PANGAEA-Wert vom
Neotoma-Feld `Quercus undiff.`. Die Differenz entspricht jeweils numerisch dem
separaten Neotoma-Wert `Quercus ilex`. Beide Angaben stehen unverändert in
[legacy_quercus_differences.csv](./legacy_quercus_differences.csv).
Dies wird als Quellen-/Aggregationsunterschied dokumentiert, nicht durch
Summieren oder Umbenennen repariert. Keine Änderung der Shared37-Taxonomie.

**Zweiter Anschluss: 26608 → 47525.** Die Suche über alle 924 Kompositproben
verwendet sämtliche unverändert gemeinsamen Variablenschlüssel aus Taxon-ID,
Einheit, Element, Kontext, Symmetrie und Elementtyp. Von 131 bzw. 182 nativen
Schlüsseln sind 76 vergleichbar. Das sind **keine 76 unabhängigen Pflanzentaxa**.
Ein Schlüssel mit verschiedenen Werten innerhalb derselben Probe wird global
ausgeschlossen: Indeterminable, Taxon-ID 312. Er wird weder überschrieben noch
summiert. Identische numerische Wiederholungen desselben Schlüssels werden
für den Vergleich nur einmal verwendet. Namen und ökologische Gruppen sind
keine zusätzlich gezählten Messungen.

Nicht gespeicherte Tupel gelten ausschließlich für diesen Vergleich der dünnen
numerischen Exporte als Null; ein tatsächlich fehlender Zahlenwert wird
abgewiesen. Keine Toleranzoptimierung, Umbenennung, Eltern-/Untertypenaddition
oder weitere Merkmalsauswahl zur Erhöhung der Trefferzahl. Gleiche Taxon-IDs
allein beweisen keine identische Aggregationspraxis beider Quellen.
**Alter, Tiefe und Etiketten werden bei der Kandidatensuche nicht verwendet.**

[sample_correspondence.csv](./sample_correspondence.csv) erhält alle 69
Entscheidungen, alte/neue Datenbankschlüssel und die Etiketten der 28 eindeutigen
Kandidaten. Die Komposittiefen sind native Angaben von **47525**, keine MON16-
Tiefen. Die variablen Tiefendifferenzen werden erhalten, nicht als konstanter
Offset oder Interpolationsfunktion verwendet. Alle MON16-Felder bleiben leer.

Die 41 fehlgeschlagenen Treffer werden nicht nachträglich akzeptiert.
Eine ausdrücklich **nachgelagerte Diagnose** sucht ihre jeweils kleinste Zahl
abweichender gemeinsamer Zellen, ohne Alter oder Tiefe. Die 47 Unterschiede
betreffen Plantago (35), Scrophulariaceae (9), Abies (2) und Sambucus (1).
[unmatched_diagnostics.csv](./unmatched_diagnostics.csv) und
[unmatched_variable_differences.csv](./unmatched_variable_differences.csv)
bewahren sie als ungeklärte Unterschiede, nicht als weitere Korrespondenzen.
Die zwei Abies-Diagnosezellen lauten 1 gegenüber 1,5 bzw. 0 gegenüber 0,5.
Solche halbzählig gespeicherten NISP-Werte werden nicht gerundet oder als
neu bestätigte ganzzahlige Rohzählungen ausgegeben; ihre Entstehung ist zu klären.
Der frühere 66/69-Vergleich aus PR #21 bleibt ein anderer Quellenvergleich.

## Metadatengrenzen und nächster Beleg

[field_availability.csv](./field_availability.csv) unterscheidet gespeicherte
NULL-Werte von überhaupt nicht exportierten Slots. Proben-/Analyseeinheitsnotizen
und Laborbezeichnungen fehlen als Slots dieses Objekts. Daraus folgt keine
Leere der zugrunde liegenden relationalen Tabellen. Der frühere Metadatenauftrag
bleibt insoweit relevant; die Behauptung, es lägen gar keine Etiketten für
47525 vor, ist durch diesen neuen Eingang überholt.

Jetzt gezielt die **47 benannten Zellunterschiede** anhand der ursprünglichen
Variablen-/Aggregationsdefinitionen und Einreichungsdaten klären. Weder die
Taxontoleranz lockern noch Eltern-/Untertypen nach gewünschtem Treffer addieren.
Daneben die belegten Abschnittsbezeichnungen mit den historischen Kern- und
Kompositierungstabellen verbinden und erst danach mit MON16 samt
Korrelationsunsicherheit. P1 bleibt offen; keine automatische Auswahl der
28 Proben für einen Fossilpilot und keine neue Klimakurve.

## Prüfung und Reproduktion

Der Produzent verwendet `rdata 1.1.0` für die Serialisierung und die Konvertierung
des ausgewählten Standorts. Der separate Prüfer liest S4-Slots, Vektoren und
Datentabellen direkt aus dem Parserobjekt; er verwendet weder den Produzenten
noch dessen Klassen-/Dataframe-Konverter. Die Zahlenprüfung erfolgt dort mit
exakten Brüchen der gespeicherten Zahlenwerte. **Gemeinsame Abhängigkeit:**
Beide Leser verwenden denselben R-Serialisierungsparser; kein völlig unabhängiger
Binärdecoder behauptet. R-, Workshop- und Altersmodellcode werden nicht ausgeführt.

Ein wichtiger Einlesefall wurde geprüft: Die 59 Standorte stehen in einem
R-Vektor mit 59 identischen Namen `site`. Eine ungeprüfte Umwandlung in ein
Python-Dictionary würde Einträge verlieren. Deshalb Auswahl im Rohvektor vor
Konvertierung; der Prüfer kontrolliert die 59 Elemente separat.

Zwei vollständige Produzentenläufe einschließlich Manifest sind bytegleich.
Der separate Leser kontrolliert 13 Inventarzeilen, 1.225 Quellzellen, vier
Positionswechsel, 16 Quercus-Differenzen, beide Variableninventare, alle 69
Entscheidungen, 28 Kandidatenetiketten und sämtliche 47 Diagnosezellen.
Vier neue Negativtests prüfen widersprüchliche/mehrfach gespeicherte Werte,
NULL gegen Null und unterschiedliche Messkontexte. [verification.json](./verification.json).

Die drei Eingaben gemäß [sources.json](./sources.json) beziehen. Die kleine
PR-#25-Tabelle kann gezielt vom dort fixierten Commit gelesen werden; kein
Original-ScienceDirect-ZIP und keine frühere Chat-Zwischendatei erforderlich.
Aus der Repository-Wurzel, mit neuen Ausgabeverzeichnissen:

```sh
python tools/thesis_import/audit_neotoma_workshop.py --rds /pfad/itDownload.RDS --pangaea /pfad/monticchio_739452.tab --prior /pfad/pr25_sample_identity.csv --output /pfad/neuer_lauf
python tools/thesis_import/verify_neotoma_workshop.py --rds /pfad/itDownload.RDS --pangaea /pfad/monticchio_739452.tab --prior /pfad/pr25_sample_identity.csv --output /pfad/neuer_lauf --report /pfad/pruefung.json
python -m unittest discover -s tools/thesis_import -p test_neotoma_workshop.py -v
```

Roh-RDS und Originalzählungen werden nicht im PR weiterverteilt. Die MIT-Lizenz
der Workshopsoftware wird nicht pauschal auf alle enthaltenen Quelldaten
übertragen; insbesondere PANGAEA.739452 trägt CC BY-NC 3.0. EPD, Neotoma und
Originalbeitragende bleiben als Quellen genannt. Baseline, vorhandene Alter und
Claudes CLI-/App-Dateien unverändert; keine Autorenanfrage oder Merge.
