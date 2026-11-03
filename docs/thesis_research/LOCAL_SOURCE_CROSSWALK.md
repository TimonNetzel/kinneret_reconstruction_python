# Quellenabgleich des lokalen Bestands

Stand 01.11.2026. Ergänzung zur [Bestandsaufnahme](LOCAL_WORKSPACE.md).
Originaldaten, vorhandene Referenzen und App-Integration werden nicht geändert.
Persönliche Quellpfade, vollständige Dateinamenlisten und neue Rohdaten bleiben
in `.local/`. Die nachfolgenden Befunde sind keine fachliche Pilotfreigabe.

## Fassungen und zitierbare Belege

Die vollständige lokal gefundene Dissertation mit Titelblatt September 2022
hat 176 PDF-Seiten. Die vier vorhandenen Repository-Ausschnitte haben
15, 8, 18 und 53 Seiten. Ein vollständiger Vergleich der durch Leerzeichen
normalisierten Seitentexte fand **keine exakt gleichen Seiten** zwischen dieser
Vollfassung und den Ausschnitten. Auch der begrenzte Vergleich mit vier weiteren
lokalen Kandidaten erbrachte keinen exakten Treffer für die erste Ausschnittseite.
Das belegt unterschiedliche Text-/Layoutfassungen; es beweist keinen
wissenschaftlichen Widerspruch und bestimmt nicht die endgültige Einreichfassung.
Beispielsweise ist die Abbildungsnummer im Einleitungstext zu Kapitel 2.5
unterschiedlich. Gedruckte Seiten beider Fassungen werden deshalb nicht vermischt.

Für den folgenden mathematischen Abgleich werden die vorhandenen Ausschnitte
als eindeutig benannte Quellen verwendet. Die lokale Vollfassung und vier
ausgewählte LaTeX-Archivmitglieder sind separat gehasht und zugeordnet.
Nachweise: `.local/thesis_version_comparison.json`,
`.local/thesis_candidate_match.json`, `.local/model_source_pages.json`,
`.local/selected_sources.json`, `.local/original_code_sources.json`.

| Aspekt | Dissertation: Abschnitt, Gleichung, gedruckte Seite / PDF-Seite im genannten Ausschnitt | Bestehende Implementierung / Ergebnis des Abgleichs |
| --- | --- | --- |
| Alters-Tiefen-Transformation | 2.3, Gl. 2.10; S. 27 / PDF 4 in `Phd_thesis_netzel(1).PDF` | `model.py::build_setup`, Zeilen 179–187: Auswahl der Probentiefen, zeilenweise Normierung und Matrixprodukt mit Spektrum bzw. AP. Das ist der vorhandene diskrete Ansatz. |
| Taxonmodell | 2.4, Gl. 2.13; S. 32 / PDF 9 im selben Ausschnitt | Dissertation nutzt Taxon-Transferfunktionen. Der Python-Paperfall besitzt stattdessen eine feste 33×3-Taxon-Biom-Zuordnung. Diese Modelle sind nicht identisch. |
| Referenzkurven | 2.4, Gl. 2.17; S. 34 / PDF 11 | Der Paketfall kalibriert gegen AP. Soreq- und MEDSTACK-Kanäle aus den Dissertationsbeispielen sind darin nicht enthalten. |
| Klimaanker | 2.4, Gl. 2.19; S. 35 / PDF 12 | Paket: rezenter Temperatur-Normalprior und Niederschlags-Gammaprior. Für den Baseline-Vergleich werden die vorhandenen Paperwerte aus den Eingangsdaten verwendet, keine neuen Anker festgelegt. |
| Transferfunktion und Klima | 2.5; S. 38 / PDF 1 in `Phd_thesis_netzel_ch2.5.PDF` | Dissertation: CRU 4.01, 1961–1990, Jahresmitteltemperatur und Jahresniederschlag. Paperfall: Wintertemperatur und Jahresniederschlag. Die Temperaturvariablen dürfen nicht gleichgesetzt werden. |
| Lern-/Testaufteilung, SMOTE | 2.5; S. 41 / PDF 4 im selben Ausschnitt | 70 % Training, verbleibende 30 % weiter geteilt; SMOTE nur im Training. Die lokal geprüfte R-Fassung zeigt ebenfalls 70/15/15 und trainingsseitiges SMOTE, hat aber aktive QDA-/Mehrklassenoptionen und eine logistische Niederschlagsnormierung. Sie ist kein Nachweis der exakten Parameter aller historischen Artennetze. |

## Neuer Codebeleg: Kombination der Taxa

Der vorhandene [Dissertationsabgleich](../abgleich_thesis.md) ließ die
Kombinationsregel offen und nannte ein Produkt als mögliche Lesart.
Im neu zugänglichen lokalen R-Quellstand lässt sich nun konkreter feststellen:

- `parameter_stage_weights`: Multiplikation der Taxoninformationen mit den
  gezogenen Gewichten, anschließend zeilenweise Normierung.
- Die letzte Definition von `reconstruction` im geprüften Funktionsskript
  bildet die Klimakurve über ein Matrixprodukt dieser Anteile mit gezogenen
  Taxon-Klimawerten.
- `final_reconstruction` bildet die Dichten über Matrixprodukte der
  normierten Taxonanteile mit den Taxon-Transferdichten. Für diesen Zweig ist
  damit eine gewichtete Summe/Mischung belegt; ein Produkt der Taxondichten
  wird dort nicht gebildet.

Die lokalen Zeilenbelege sind 1242–1284, 1781–1820 und 450–477 des gesperrten
Funktionsskripts. Es gibt mehrere Definitionen von `reconstruction`, alternative
Sampler und optionale Nachbearbeitung. Die Beobachtung gilt für diesen
Quellstand und diese Zweige, nicht automatisch für jeden Dissertationslauf.
Keine wissenschaftliche Methode wird dadurch stillschweigend ersetzt.

Im selben `parameter_stage_weights`-Zweig enthält der Dirichlet-Vorschlag den
Streuungsfaktor, die aktive Hastings-Dichtekorrektur jedoch nicht. Das entspricht
dem bereits im Paper-Code dokumentierten Problem. Der ursprüngliche R-Code wird
hier weder ausgeführt noch korrigiert; zuerst wird der erhaltene Paketfall geprüft.

## Eingabebestand: gesicherte Befunde und Grenzen

| Bestand | Struktur und Einheiten | Zuordnung / Verwendbarkeit |
| --- | --- | --- |
| Vorhandener Kinneret-Paperfall | 160 Tiefen, 33 Taxa, drei Biome; 181 Alter von -50 bis 8950 cal BP, Abstand 50 Jahre; Tiefe cm, Wintertemperatur °C, Jahresniederschlag mm/Jahr | Zusammenhängende vorhandene Eingaben, kurze Referenzkette und volle archivierte Referenz vorhanden. Paket-Eingabevalidierung bestanden. Als erstes technisches Beispiel geeignet. |
| Lokale Vegetationsraster | 16 CSVs mit 33 Spalten: Lon/Lat und BP0 bis BP15000 in 500-Jahres-Schritten; Klassen 0–4, `NA` und Eis-Sentinel `-9999` | 15 nichtleere Tabellen, eine nur mit Kopfzeile. Zwölf nichtleere Tabellen haben 3691 Zeilen, drei 3694; keine doppelten Koordinaten oder ungleich breiten Zeilen festgestellt. Koordinatenbezogen verbinden, keine gemeinsame Zeilenreihenfolge voraussetzen. |
| Raster-ZIP | Alle 16 untersuchten CSVs bytegleich zu ihren gleichnamigen Archivmitgliedern | Die leere Tabelle ist auch im Archiv leer. Einfaches Entpacken kann diese Lücke nicht beheben. Kein automatisches Auffüllen oder Ersetzen. |
| Zwei lokale Klimaraster | Je eine Zeitstufe, 97×149 Zellen; Temperatur °C, Niederschlags-Variablenattribut `mm/month` | Metadaten nennen **CRU TS4.00**, Ausgangsdaten 1901–2015. Temperatur: `timmean`; Niederschlag: `timmean -yearsum`, zusätzlich `frequency=year`. Daher ein nach Metadaten jährlich aggregiertes Mittel bei stehengebliebenem Monatsattribut; keine eigenmächtige Einheitenänderung. **Nicht** als CRU 4.01, 1961–1990 oder CRU 4.07 verwenden. |
| RData-Bestand | 13 Container; drei kleine Dateien erfolgreich gelesen | Große Container noch nicht vollständig dekodiert: auf diesem 8-GB-PC zu Beginn nur ca. 1,16 GB freier physischer Speicher. Komprimierte Dateigröße ist keine sichere RAM-Prognose. Keine pauschale Vollkopie/Vollkonvertierung. |
| Verbreitungsdaten in einem kleinen RData-Container | 149×97×275, Werte 0/1 und 1.302.125 fehlende Werte; Namenszuordnung 275×2 | Keine Koordinaten am Array gespeichert. Fortsetzung: ausdrücklicher R-Verbraucher und identische Fehlwertmasken beider Klimafelder stützen die lokale Zuordnung mit aufsteigenden Längen-/Breitengraden; Details im [Taxon-Methodenaudit](LOCAL_TAXON_METHOD_AUDIT.md). 16 Ebenen ohne Präsenz, Namen nicht eindeutig. Kein Nachweis der 279 fertigen historischen Transferfunktionen. |
| Historische R-Einstiege | 53 explizite erste Dateiparameter untersucht: 28 lokal vorhanden, acht lokal fehlend, 17 außerhalb des ausgewählten Bereichs | Dynamische Pfade, Bedingungen und vollständige Laufkonfiguration bleiben zusätzlich aufzulösen. Kein vollständiger ausführbarer Dissertationsfall damit nachgewiesen; externe Bereiche wurden nicht durchsucht. |

Detaillierte lokale Nachweise: `.local/input_validation.json`,
`.local/npz_structure.json`, `.local/epd_grid_validation.json`,
`.local/netcdf_metadata.json`, `.local/original_dependency_audit.json`.
Zusätzlich `.local/taxon_distribution_structure.json` und
`.local/R_container_headers.json`: alle 13 RData-Header nennen R 3.5.0 als
Schreibversion. Das ersetzt keine Aufzeichnung der damaligen Paketversionen.

## Fortsetzung: lokale Georeferenzierung und historische TF-Erzeugung

[LOCAL_TAXON_METHOD_AUDIT.md](LOCAL_TAXON_METHOD_AUDIT.md) ergänzt diesen Abgleich:
275 lokale Ebenen mit 9.718 verwendbaren Zellen; Positionen und R-Zeilenlabels
getrennt erhalten. Das historische Erzeugungsskript `all_trafo.R` wurde über den
Ausgabepfad im Haupteinstieg identifiziert. Seine globale Eingabedatei und die
fertigen TF-Container sind weiterhin nicht verfügbar. Der erhaltene Quellstand
setzt die Trainingstabelle nach der Hold-out-Aufteilung zurück und kombiniert
EMD-/BA-Ränge; keine stillschweigende Übernahme oder Behauptung eines geprüften
historischen Laufs. Die Art-/Pollentyp-Zusammenführung besitzt versionsabhängig
eigene Regeln und ist vom späteren gewichteten Rekonstruktionsschritt zu trennen.
Aktuelle lokale Nachweise: `.local/taxon_source_audit_v2/`.

## Zusatzfund nach Nutzerhinweis: globale Verbreitungsbasis

[MY_RECONST_SOURCE_RESULT.md](MY_RECONST_SOURCE_RESULT.md) ist die aktuelle
Fortsetzung nach Prüfung des zusätzlichen Ordners. Die zuvor dort noch nicht
gesuchte globale Datei wurde in zwei Fassungen gefunden. Ohrid hat 283 Ebenen;
der dokumentierte Löschpfad liefert 279 Positionen und die ausdrücklich im Code
stehende Juglans-Umbenennung. Keine bloße Ableitung aus gleich großen Arrays:
Koordinaten, geordnete Namensfolge und vollständige gemeinsame Rasterhashes geprüft.

Die 275 gemeinsamen globalen Ebenen sind untereinander gleich, unterscheiden sich
im lokalen geographischen Fenster jedoch an 95.006 Zellen vom bisherigen
regionalen Block. Die historischen fertigen TF-Container und test.nc-Klimadateien
fehlen weiter. Zwei geprüfte gespeicherte QDA-Modelle sind fundortbezogene
Mehrklassenfits. Keine automatische Übernahme als taxonbasierte Hauptmethode.
Aktueller Nachweis: `.local/my_reconst_source_audit_v3/`.

## Festgelegtes erstes Beispiel und Prüfplan

Der bestehende Kinneret-Paperfall wird zuerst als **vorläufiges technisches
Referenzbeispiel** reproduziert. Es handelt sich um reale vorhandene Eingaben,
keine synthetische Ersatzrekonstruktion. Der taxonbasierte vollständige
Dissertationslauf bleibt separat offen.

Vor Sichtung der neuen Vollergebnisse festgelegt in `.local/reference_design.json`:

1. 20.000 diagnostische Legacy-Iterationen; danach Original-Konfiguration mit
   einer Million Iterationen, Burn-in 250.000, Seed 2023.
2. Originalquelltext separat mit derselben lokalen Laufzeit bauen und gegen
   den Paket-Legacy-Kern prüfen. Archivierte Bitgleichheit separat berichten.
3. Bestehende korrigierte Variante bis v2: vier Ketten à zwei Millionen,
   Burn-in 500.000, fester Dir(0,5)-Prior und dieselben rezenten Paperanker.
   Nur Speicherung auf `thin=600` statt 150 begrenzen, damit insgesamt
   10.000 Nach-Burn-in-Ziehungen den Speicherbedarf überschaubar halten.
4. Konvergenz nach den vorhandenen Repositorykriterien prüfen:
   rangnormiertes Split-$\hat R\leq1{,}01$, Bulk-ESS ≥400. Weitere Parameter-
   und Klimakurvenprüfungen explizit ausweisen, keine Ein-Ketten-Diagnostik
   als unabhängige Mehrkettenprüfung bezeichnen.
5. Klimakurven mit einer separaten NumPy-Formel aus Posterior und Eingangsdaten
   nachrechnen, deterministische Wiederholung prüfen und Unterschiede zum
   vorhandenen Archiv bzw. zur unkorrigierten Baseline quantitativ berichten.
   Keine neue fachliche Toleranz aus dem gewünschten Ergebnis ableiten.

Alle Läufe und Berichte lokal unter `results/`; keine Rohdaten hochladen.
Vor weiteren Fossilfällen ist die fachliche Bewertung des konkreten Beispiels
erforderlich. Die bekannten Quellen-/Chronologie-Gates bleiben bestehen.
