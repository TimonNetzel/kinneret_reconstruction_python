# Monticchio: ursprüngliche Variablenstruktur und 47 Differenzen

Stand 30.10.2026. Fortsetzung von [PR #26](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/26)
auf eigener Basis des weiterhin offenen #18. **Nachgelagerte Quellen- und
Zahlendiagnose, keine neue Probenzuordnung oder Quellenkorrektur.**

## Ergebnis und Reichweite

| Befund an den bereits in #26 festgelegten Paaren | Ergebnis |
| --- | --- |
| Originalfelder Abies, vier Plantago-Felder, Scrophulariaceae, Sambucus | 483 Zellen (69 × 7) in EPD-Archiv, PANGAEA und altem Neotoma-Bestand gleich |
| Plantago | Bei 69/69 festen Paaren ist der neue Gattungswert gleich der Summe der vier alten Felder; das bilanziert alle 35 Plantago-Differenzzellen aus #26 |
| Scrophulariaceae | Bei 69/69 festen Paaren ist der alte Wert gleich dem neuen Digitalis-type-Wert; darunter die neun positiven Differenzfälle und 60 gemeinsame Nullfälle |
| Weitere Differenzen | Zwei Abies- und eine Sambucus-Zelle bleiben zahlenmäßig ungeklärt |
| Beleg einer historischen Umcodierung | Nicht vorhanden: ursprüngliche Einreichungsdatei, datensatzbezogene Umcodierungs-/Synonymiedokumentation und Einreichungshistorie fehlen |
| Probenstatus aus #26 | Unverändert 28 numerische Kandidaten und 41 ohne vollständigen nativen Treffer; P1 offen |

Die 44 numerischen Muster sind **keine 44 nachgewiesenen Quellenkorrekturen**.
Alle 47 Fälle bleiben hinsichtlich ihrer Entstehung ungeklärt. Insbesondere
werden die 35 und neun Diagnosezellen nicht durch neue Summen oder Namen in
der bisherigen Trefferentscheidung ersetzt. Gleiche Werte erlauben weder eine
Umbenennung noch den Schluss auf dieselbe physische Probe.

## Der neue konkrete Quellenbeleg

Der frisch wiederbezogene vollständige EPD-Snapshot vom 31.10.2017 wurde hier
gezielt auf **Variablendefinitionen und die sieben betroffenen Originalfelder**
geprüft. Das ist keine Wiederholung der ausgeschöpften Suche nach leeren
Kernabschnittstabellen. Archiv, RDS und PANGAEA stimmen mit den früheren
gesperrten Bytehashes überein; keine alten Chat-Dateien wurden vorausgesetzt.

[original_variable_definitions.csv](./original_variable_definitions.csv) erhält
die ursprünglichen EPD-Variablenkennungen, Namen, `accvar_`-Verweise und Notizen.
Die vier getrennten Plantago-Felder sind `Pla` (576), `Pla.l-T` (580),
`Pla.m/m` (583) und `Pla.c-T` (1478). Das ursprüngliche `Scoae` (763) verweist
über `accvar_` auf **918, Scrophulariaceae (colporate-reticulate)**. Der exportierte
Familienname erhält damit einen konkreteren historischen morphologischen
Kontext. Das ist **noch kein dokumentiertes Synonym zu Digitalis-type**.
Die 2017 gespeicherte Taxontabelle ist nicht mit der ursprünglichen
Einreichungsdatei oder einer datensatzbezogenen Umcodierungsanweisung gleichzusetzen.

Im gesperrten Neotoma-RDS stehen unter 26608 alle vier Plantago-Felder, unter
47525 nur das Gattungsfeld 480; die drei Untertypen sind dort nicht als
Variablenschlüssel gespeichert. Digitalis-type trägt dort Taxon-ID **4114**.
[affected_native_inventory.csv](./affected_native_inventory.csv) erfasst die
betroffenen Felder in beiden vollständigen Datensätzen. Das Objekt exportiert
hier keine ursprünglichen `variableid`-Schlüssel oder datensatzbezogenen
Synonymiebelege. Nicht gespeicherte Einträge sind nur im dünnen numerischen
Export als Null behandelt, nicht als bestätigte ursprüngliche Nullzählung.

Die allgemeine [Neotoma-Dokumentation, Abschnitte 14.6 und 14.8](https://open.neotomadb.org/manual/taxonomy-related-tables-1.html)
erläutert die Unterscheidung von Variablen und undifferenzierten Taxa. Diese
Konvention erklärt nicht automatisch, wie diese beiden Einreichungen
aggregiert wurden. Eine globale neue Taxonregel wird daraus nicht abgeleitet.

## Wie die Diagnose begrenzt wurde

[fixed_pair_diagnostics.csv](./fixed_pair_diagnostics.csv) enthält alle **69
bereits festgelegten Paare**: 28 Kandidaten und 41 jeweils eindeutige nächste
Diagnoseeinträge aus #26. Es erfolgt keine erneute Suche, Toleranzänderung oder
Neuauswahl nach Summen, Namen, Alter, Tiefe oder Abschnittsbezeichnungen.
Die dort neu mitgeführten 41 Abschnittsnamen beschriften ausdrücklich nur
Diagnoseeinträge; sie sind keine zusätzlich bestätigten physischen Zuordnungen.
Alle MON16-Felder bleiben leer und die ursprünglichen Status bleiben erhalten.

Die Interpretationen wurden nach Sichtung der Variableninventare und Werte
entwickelt, nicht vorregistriert. Plantago wird nur als feste Vier-Feld-Bilanz
ausgewiesen. Für Scrophulariaceae stehen auch die gespeicherten Werte für
Digitalis-type, Veronica-type und Pedicularis nebeneinander; kein automatischer
Familien-/Untertypenzusammenschluss. Im Sambucus-Fall erklärt auch der
separat ausgewiesene cf.-Wert die Differenz nicht.

[difference_evidence.csv](./difference_evidence.csv) bewahrt jede der 47
ursprünglichen Differenzzeilen einschließlich ihrer alten Werte und des Status
`unresolved_source_difference`; zusätzliche Spalten kennzeichnen nur das
Diagnosemuster. Die drei verbleibenden Zahlenunterschiede sind:

| Pilotprobe | Neotoma 26608 → Diagnose 47525 | Archivierter Name des Diagnoseeintrags | Feld | Alte / neue Zahl |
| --- | --- | --- | --- | --- |
| LGM90D:000009 | 269988 → 434216 | E1-o, 90 | Sambucus | 4 / 2 |
| LGM90D:000055 | 270034 → 434416 | D4-u, 30 | Abies | 1 / 1,5 |
| LGM90D:000067 | 270046 → 434441 | D5-u, 50 | Abies | 0 / 0,5 |

Halbzählig gespeicherte NISP-Werte bleiben unverändert; es wird weder ein
Rundungsfehler noch eine bestimmte Zähl-/Mittelungspraxis unterstellt.

## Einreichungsbeleg weiterhin offen

Vier neu gezielt gelesene offizielle Neotoma-Code-/Schemadateien benennen einen
passenderen nächsten Eingang: datensatzbezogene `synonymy` einschließlich
Beitrags-/Publikationsbezug, `datasetsubmissions` mit Datum, Typ und Notizen
sowie `datataxonnotes` an den Originalmessungen. Die Funktion
`ti.getdatasetvariablesynonyms` belegt den datensatzbezogenen Verknüpfungspfad.
[sources.json](./sources.json) sperrt die gelesenen Codebytes auf Commit und
Git-Blob; es ist **kein Befund an diesen Datenbanktabellen** und kein Nachweis
des gemeinsam deployten Live-Serverstands. Keine neue unveränderte API-Abfrage.

[SOURCE_GAP.md](./SOURCE_GAP.md) beschreibt den jetzt präzisierten fehlenden
Eingang. Ohne diesen bleiben auch die beiden plausiblen Aggregations- bzw.
Umcodierungsmuster unbestätigt. Danach sind die historischen Abschnitts- und
Kompositierungstabellen und der Anschluss an MON16 samt Korrelationsunsicherheit
weiter erforderlich. P1 wird durch diesen Bericht nicht gelöst.

## Prüfung und Reproduktion

Zwei vollständige Produzentenläufe liefern dieselben sechs Ausgabedateien
einschließlich Manifest bytegleich. Der separate Prüfer liest rohe S4-Slots
statt des Klassen-/Dataframe-Konverters und rechnet mit `Fraction` statt
`Decimal`. Er liest zudem den vollständigen SQL-COPY-Strom statt der getrennten
Tabellendumps. Er importiert den neuen Produzenten nicht. **Gemeinsame
Abhängigkeit:** der R-Serialisierungsparser `rdata 1.1.0`; kein unabhängiger
Binärdecoder behauptet. Vier neue Tests prüfen NULL, Bruchteile, widersprüchliche
Duplikate und verschiedene Messkontexte/Einheiten.

Geprüft sind 483 Originalzellen, alle 69 festen Paare samt Namen und erhaltenen
Status, sieben Variablendefinitionen, 22 Inventarzeilen und sämtliche 47
Differenzdiagnosen. [verification.json](./verification.json),
[summary.json](./summary.json), [manifest.json](./manifest.json).

Alle acht Eingänge unter den **lokalen Dateinamen** aus [sources.json](./sources.json)
in einen neuen Eingangsordner laden und SHA256 prüfen. Darunter sind zwei
explizit auf #26 fixierte Leser, die aus dessen eigenem Branch bezogen werden;
sie werden nicht als automatisch vorhandene Dateien vorausgesetzt. Python mit
`rdata==1.1.0`, NumPy und pandas verwenden. Aus der Repository-Wurzel:

```sh
python tools/thesis_import/audit_monticchio_variables.py --inputs /pfad/eingaenge --pr26-reader /pfad/eingaenge/read_workshop.py --output /pfad/neuer_lauf
python tools/thesis_import/verify_monticchio_variables.py --inputs /pfad/eingaenge --pr26-raw-reader /pfad/eingaenge/verify_workshop.py --output /pfad/neuer_lauf --report /pfad/pruefung.json
python -m unittest discover -s tools/thesis_import -p test_monticchio_variables.py -v
```

R-, Workshop-, SQL- und Altersmodellprogramme werden nicht ausgeführt.
Roharchive und Originalzählmatrizen werden nicht weiterverteilt. Das
Original-ScienceDirect-ZIP wurde nicht benötigt oder erneut geprüft. Baseline,
Shared37-Taxonomie, bestehende Alter und Claudes CLI/App bleiben unverändert.
Keine Pilotfreigabe, Autorennachricht oder Merge.
