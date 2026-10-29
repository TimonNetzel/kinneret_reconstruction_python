# Marker, Doppelzählung und Taxonentscheidungen

Stand 29.10.2026. Fortsetzung der Prüfliste aus PR #15, auf dessen Commit `e51a5c6dd80970d66430ca43c75eb9b731db6711`. Diese Prüfung legt Entscheidungen für den **fossilen Arbeitskatalog** fest. Sie erzeugt weder eine moderne Trainingsmatrix noch eine wissenschaftliche Modellfreigabe. Importer, Referenz und Kinneret-Modell bleiben unverändert.

## Entscheidungen für alle 135 Felder

| Aktion | Felder | Bedeutung |
|---|---:|---|
| `include` | 58 | Im Arbeitskatalog behalten, mit dokumentiertem Kandidaten für ein modernes Konzept |
| `pool` | 2 | Kandidat für das Zusammenführen gleichnamiger moderner ID-Namensräume; hier keine Werte zusammengeführt |
| `exclude` | 42 | Für diesen terrestrischen Piloten außerhalb des Umfangs, technisch/unbestimmbar oder als überlappende Untergliederung gesperrt |
| `unresolved` | 33 | Originalwerte erhalten; Konzeptgrenzen oder Zählkonvention weiter offen |

`fossil_decisions.csv` führt jede Quellspalte genau einmal mit Aktion, Begründung und gegebenenfalls modernen Kandidaten-IDs. `taxon_policy.json` macht die Auswahlregeln explizit; `build_taxon_decisions.py` prüft sie gegen die gesperrten Dateien. Auch ein `include` besitzt `scientific_status=not_training_approved`. Die ausgewählten modernen IDs sind keine vollständige Liste aller zu einem Gattungs- oder Familienkonzept gehörenden Originalcodes und dürfen nicht ungeprüft zu einer Trainingsspalte summiert werden.

## Eucalyptus als Marker: neue Primärevidenz

Der heruntergeladene [Neotoma-Datensatz 27166](https://data.neotomadb.org/27166), DOI [10.21233/ZEWE-M072](https://doi.org/10.21233/ZEWE-M072), beschreibt einen Abschnitt von LGM-90D, Drive D8-u, um die Tephra TM-16b. Seine Daten unterscheiden `Eucalyptus tablets` und den gezählten `Eucalyptus spike` ausdrücklich als **Laboratory analyses/LABO**. Die erste Probe enthält beispielsweise 42.500 Körner pro zugesetzter Tablette, eine Tablette und 585 gezählte Marker; diese Größen sind keine drei Pflanzenbeobachtungen.

Die [Methoden von Allen & Huntley (2018), Abschnitt 2](https://doi.org/10.1111/1365-2745.12998) beschreiben eine vor der chemischen Behandlung zugesetzte Suspension von Corymbia ficifolia, ehemals Eucalyptus ficifolia, zur Bestimmung von Akkumulationsraten. Das stützt die Markerinterpretation im Archivkontext. Es beweist keine zeilenidentische Zuordnung der 2018-Proben zur älteren PANGAEA-Tabelle. **Entscheidung:** `Euc [#]` vorsorglich aus dem Pflanzen-Klimasignal ausschließen; Marker-Risiko und Evidenzumfang erhalten. In den 69 Pilotproben betrifft das eine Summe von 5.821 im Eucalyptus-Feld.

Der neue Download ist **kein langer Rohzählungspilot**: zwölf Proben, 31.110–31.173 Kalenderjahre BP, also ein kurzes Ereignisfenster. Seine section-relative Tiefe darf nicht an das Kompositprofil angehängt werden. DOI, API-URL, Dateigröße und SHA256 stehen in `source_evidence.json`; die Datenlizenz dieses API-Snapshots wurde hier nicht unabhängig geklärt, die Rohdatei wird nicht weiterverteilt. Die Artikelseite war lesbar; lokale HTML-/Autoren-PDF-Downloads lieferten HTTP 403. Eine zusätzliche offene Arbeit von 1996 wurde gefunden, ihre gescannten Seiten klärten die Markerfrage nicht.

## Quercus: nicht Gattung und Untertypen addieren

In allen 23 der 245 Originalproben, in denen mindestens einer der beiden bestimmten Eichen-Untertypen positiv ist, gilt exakt:

`Que [#] = Que.r-T [#] + Que.i [#]`.

Die früher festgestellten 27 Gleichheiten über die gesamte Tabelle enthalten zusätzlich vier Nullfälle. Die Untersuchung der tatsächlich positiven Untertypen zeigt die für Doppelzählung relevante Beziehung.

**Entscheidung:** Für den Arbeitskatalog das ursprüngliche Gattungsfeld `Que [#]` behalten; die beiden Untertypfelder nicht zusätzlich als unabhängige Beobachtungen oder zur Zählsumme addieren. Ihre Werte bleiben im Originalimport erhalten. In den 69 Pilotproben werden dadurch 2.422 überlappende Untertypzählungen nicht noch einmal gezählt. Der Prüfcode stoppt, wenn der Nachweis bei einer anderen Quelldatei nicht mehr zutrifft.

Dies erzeugt noch kein modernes Quercus-Gattungsaggregat. Dort müssen alle geeigneten Unterkonzepte, Originalcodes und mögliche Gesamtsummen erst gemeinsam geprüft werden. Die bloße Ziel-ID 189 ist keine vollständige Quercus-Schließung. Untertypen dürfen außerdem nicht als sicher fehlend interpretiert werden, wenn nur eine breitere Bestimmung vorliegt.

## Mehrdeutige Namen: einige Kandidaten lassen sich eingrenzen

Die Archivkommentare enthalten laut PANGAEA die akzeptierten EPD-Namen. Deshalb wird ein exakt passender **moderner akzeptierter Name** einem bloßen Treffer auf einen alten Originalnamen vorgezogen. Dieser Schritt ist eine dokumentierte Katalogentscheidung, keine unabhängige taxonomische Validierung.

| Fossiles Feld | Katalogentscheidung | Begründung und Grenze |
|---|---|---|
| `Can [#]` | Kandidat 205, Humulus/Cannabis | Entspricht dem EPD-Kommentar; die auf Originalnamen beruhenden breiteren Cannabaceae/Urticaceae-Treffer und unbestimmte INUN-Varianten werden nicht gleichgesetzt. |
| `Urt-T [#]` | Kandidat 809, Urticaceae | Der breite gemeinsame Typ 335 wird nicht gleichzeitig beiden fossilen Spalten zugeordnet. |
| `Lar [#]` | Kandidat 123, Larix/Pseudotsuga | Entspricht dem vollständigen Archivkommentar; die abgekürzte Variante Larix/Pseudo bleibt außerhalb dieser Entscheidung. |
| `Pti [#]` | Kandidat 512, Potentilla-type | Eine durch den Originalnamen getroffene einzelne Art, Potentilla palustris, ersetzt den breiten Typ nicht. |
| `Ran-T [#]` | Kandidat 135, Ranunculaceae | Bewusst die breitere Archivzuordnung erhalten; nicht nachträglich eine Art oder einen engeren Ranunculus-Typ behaupten. |
| `Hed [#]` | Pool-Kandidat 51 und 4031 | Beide akzeptierten Namen Hedera. Im gesperrten Gesamtbestand gibt es keine Probe mit mehr als einem positiven Originalcode aus diesem Pool. Kein Pool wurde ausgeführt. |
| `Vts [#]` | Pool-Kandidat 122 und 4756 | Entsprechend für Vitis. Der Befund löst keine Fragen zu Landnutzung oder menschlichem Einfluss. |

Die Aliasentscheidungen Gramineae→Poaceae, Cruciferae→Brassicaceae, Umbelliferae→Apiaceae und die Cichorioideae-Bezeichnung beruhen auf der vorhandenen EMPD-Zuordnung. Sie werden als **Datenbank-Aliase** bezeichnet. Ähnlich klingende, enger oder weiter gefasste Morphotypen wie Anthemis/Matricaria, Caryophyllaceae mit ausgeschlossenen Untergruppen oder Ulmus gegenüber Ulmus/Zelkova bleiben offen.

Die [EMPD2-Methodenbeschreibung](https://doi.org/10.5194/essd-12-2423-2020) erläutert, dass akzeptierte Namen aus gröberer Hierarchie sowie Morphologie und geographischem Kontext entstehen können. Daraus folgt keine Garantie identischer Konzepte allein aus Namen oder IDs. Originalcodes und Originalnamen müssen bei jeder späteren Zusammenführung erhalten bleiben.

## Was bewusst offen oder ausgeschlossen bleibt

- **Cornaceae/Cornus:** Cornaceae ist in 13 Originalproben positiv. Nur eine davon enthält zugleich Cornus, dort jeweils den Wert 2. Das ist kein Beweis, dass die ganzen Spalten Kopien sind. Beide bleiben offen; keine automatische Summe und kein pauschaler Ersatz durch Maximum oder eine Einzelspalte.
- **Plantago, Polygonaceae/Rumex/Oxyria, Rosaceae und Leguminosae/Vicia:** Breite und bestimmte Typen können residuale oder zusammengefasste Werte darstellen. Die Quercus-Regel wird nicht auf diese Gruppen übertragen. Eine zufällige Gleichheit einzelner Proben begründet keine allgemeine Regel.
- **Cyperaceae:** Für diesen terrestrischen Pilotumfang ausgeschlossen; die modernen gleichnamigen IDs liegen in HERB und AQUA. Das ist eine explizite Umfangsentscheidung, keine Behauptung, jede Cyperaceae-Beobachtung sei aquatisch.
- **Sporen/Farne, Wasserpflanzen, Laborfelder und unbestimmbare Einträge:** Aus dem terrestrischen Katalog ausgeschlossen. `used_in_sum=1` der Ursprungsdatenbank definiert dafür keine fertige Pollensumme.
- **Unklare cf.-Bestimmungen und nicht identische Morphotypen:** Offen halten. Keine Gattung aus einem unsicheren Rest erschließen und keine fehlenden Trainingskandidaten als biologische Absenz behandeln.

Die maschinenlesbare Liste enthält alle 33 offenen Felder. Ausschlüsse sind Entscheidungen für den Pilotumfang, keine Aufforderung zum Löschen der Originaldaten.

## Auswirkung auf die 69 Pilotproben

`sample_selection_audit.csv` zerlegt die registrierten `[#]`-Werte pro Probe vollständig nach den vier Aktionen. Nach den ausdrücklichen Ausschlüssen beträgt der Anteil der `include`-/`pool`-Kandidaten an dieser **Prüfzählsumme** etwa 88,6–99,8 %, Median 94,6 %. Die Rechnung ist arithmetisch, kein Nachweis von 94,6 % Klimainformation oder von Modellgüte.

Der Nenner kann noch offene, möglicherweise überlappende Felder enthalten und ist **keine biologische Pollensumme**. Deshalb bleibt `biological_pollen_sum=null` in der Zusammenfassung. Kein Renormieren auf 100 %, kein Erfinden einer Restkategorie und keine moderne Trainingsmatrix aus dieser Prüftabelle.

## Konkreter nächster Datenvertrag

Der nächste Schritt ist nun enger begrenzt: moderne Originalcodes und Hierarchien für die behaltenen Katalogkonzepte schließen, breitere Bestimmungen als fehlende Typauflösung behandeln, eine nachvollziehbare Pollensumme festlegen und die Folgen der 33 offenen Felder als Sensitivität prüfen. Erst danach werden Standort-/Klimafilter und räumliche Validierung festgelegt. Ein fertiger Import und eine hohe Katalogzählungsquote ersetzen diese Entscheidungen nicht.

Die 18 bisherigen Mehrfachtreffer müssen nicht sämtlich als gleichartige Probleme weitergereicht werden: Marker und außerhalb des Umfangs liegende Gruppen sind gesperrt; fünf Namen wurden auf den ausdrücklichen Archivkommentar begrenzt; Hedera und Vitis haben klar dokumentierte Pool-Kandidaten. Die verbleibenden Konzept-/Hierarchieprobleme stehen gezielt in der Liste. Claude braucht dafür weiterhin keinen neuen Modellauftrag.

## Reproduktion

Das Skript ergänzt `tools/thesis_import/`; unveränderte Import- und Eignungs-Audit-Skripte werden verwendet. Den Neotoma-Snapshot über die URL in `source_evidence.json` beziehen und gegen Größe/SHA256 prüfen; bei einer neuen Datenbankversion einen neuen Stand registrieren.

```bash
python tools/thesis_import/build_taxon_decisions.py \
  --raw-dir /path/to/netzel_raw \
  --evidence-dir /path/to/neotoma_snapshot \
  --output-dir /path/to/new_taxon_audit
```

Ausgaben: vollständige Feldentscheidungen, probenweiser Zählaudit, Zusammenfassung und Prüfsummen. Keine Quelldatei und kein bestehendes Ausgabeverzeichnis wird verändert. Lizenzen der bisherigen Quelldateien gelten weiter; Originale werden nicht eingecheckt.
