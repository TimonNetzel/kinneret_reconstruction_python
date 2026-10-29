# Gemeinsame 37-Gruppen-Vergleichsdaten

Stand 29.10.2026. Folgearbeit zu PR #16, Commit `3a214f7ca2ca5ce869b4b8ed7db84344ec6961dd`.
Die Variante `published2025_shared37_v0` verbindet die publizierte moderne
90-Gruppen-Matrix mit den 69 chronologisch zugeordneten Monticchio-Proben.
Sie liefert gemeinsame Zähl- und Prozentmatrizen samt Klima, Koordinaten und
Altersgrenzen. Sie ist eine eigenständige Vergleichsvariante, keine Reproduktion
der ursprünglichen Netzel-Kalibrierung. Es wurden keine Modelle angepasst.

## Quellen und klare Versionsgrenze

- **Moderne Zählungen:** Zumaque et al. (2025),
  [PANGAEA 973997](https://doi.org/10.1594/PANGAEA.973997),
  `EMPD2_90taxa_7634sites.xlsx`, CC-BY-4.0. Verwendet wird ausschließlich
  das finale Blatt `1) EMPD2_90taxa_7634sites_cnts` mit 7.634 Proben.
- **Gruppierungsnachweis:** Supplement S1 zum
  [Datenartikel](https://doi.org/10.1016/j.dib.2025.111657),
  [öffentliche XML-Fassung](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12163141/fullTextXML).
  Das Supplement wird über die öffentliche Europe-PMC-API bezogen, anhand
  seiner DOCX-Tabellen gelesen und nicht als Rohdatei weiterverteilt.
- **Fossile Werte und Chronologie:** die drei bereits gesperrten Dateien
  `monticchio_739452`, `epd_entity_804591`, `epd_ages_cut_804596` aus dem
  bisherigen Quellenregister. Derselbe unveränderte Importer liefert die
  69 Tiefen-/Alterszuordnungen. Die anderen 176 Zählproben erhalten hier
  weiterhin keine künstlichen Alter.

Der Artikel beschreibt eine moderne Gruppierung von 840 über 192 auf 90 Taxa
und Klima aus WorldClim 2. Ein genauer Klima-Bezugszeitraum und Rasterstand
wurden hier nicht unabhängig festgestellt; die bereitgestellten Tann-/Pann-Werte
werden unverändert übernommen. Die alte SQLite-Datei und ihre Qualitäts-,
Standort- oder Taxon-IDs werden **nicht** mit dieser Kohorte vermischt.
`sources_2025.json` sperrt Downloads durch Bytezahl und SHA256. Für das DOCX
sind zusätzlich die Prüfsummen des gelieferten ZIP-Containers verzeichnet.

## Welche Prozentwerte werden berechnet?

Für jede Probe gilt:

`shared_count_sum = Summe der 37 ausgewählten Gruppen einschließlich Pinus`

`Prozent(G) = 100 × Zählung(G) / shared_count_sum`

Das ist eine ausdrücklich definierte **Teilzusammensetzung**, keine vollständige
biologische Pollensumme. Ausgelassene Gruppen sind außerhalb dieser Basis;
sie werden nicht als biologisch abwesend oder als Restkategorie ausgegeben.
Eine numerische Null bedeutet nur eine explizite Null in der Quellenmatrix.
Leere oder nichtnumerische Zählfelder führen zum Abbruch.

Die veröffentlichte Mindestzählung von 100 gilt für die 90-Gruppen-Summe
**ohne Pinus**. Die veröffentlichten Prozentwerte verwenden dagegen alle
90 Gruppen **mit Pinus**; der Präparer prüft diese Rechnung für jede Probe.
Unsere zusätzliche Spalte `count_threshold_pass` prüft mindestens 100 Zählungen
innerhalb der **37 gemeinsamen Gruppen ohne Pinus**. Sie ist ein dokumentiertes
Arbeitskriterium, keine Quellenbehauptung oder automatische Modellfreigabe.
Alle Proben bleiben im Export; es gibt keinen versteckten Schwellenfilter.

## Taxonomische Entscheidungen

`shared_features.json` nennt jede fossile Originalspalte und jeden modernen
Quellcode. `feature_evidence.csv` weist zu jedem modernen Code den publizierten
Gruppennamen und Beispiele seiner S1-Mitglieder nach. Die Zuordnung ist eine
bewusste Vergröberung von Pollenbestimmungen, keine Aussage identischer
botanischer Artenzusammensetzung.

| Gruppe | Entscheidung |
|---|---|
| Quercus | Modern `FQDEC + FQEVE`, fossil nur `Que`. Die zwei fossilen Untertypen werden nicht erneut addiert. Alle 23 positiven Untertypfälle der Originalquelle erfüllen die geprüfte Summenbeziehung. |
| Carpinus/Ostrya | Fossil `Car.b + Ost-T`, modern publizierte gemeinsame Gruppe `BCARP`. |
| Fraxinus; Ephedra | Jeweils beide getrennten fossilen Typen zu der publizierten Gattungsgruppe zusammenführen. |
| Cannabaceae/Urticaceae | Fossil `Can + Urt-T`, modern `CANURT`. Das breite publizierte Konzept enthält auch mehrdeutige ursprüngliche Typen; als gemeinsame Arbeitsgruppe, nicht als Artenidentität behandeln. |
| Poaceae | Nur `Graae` und `POACE`. Cereal-type bleibt außerhalb der Basis; keine Aussage zu Ackerbau. |
| Taxus; Cupressaceae | Explizite Vergröberung auf die publizierte moderne Gruppe; die fossile Bestimmungsauflösung wird dadurch nicht höher. |
| Eucalyptus/Laborfelder | Fossiles `Euc`, Spikekonzentration und Probenvolumen bleiben ausgeschlossen. Die Markerentscheidung gilt für diesen fossilen Archivkontext. |
| Weitere mehrdeutige Familien-/Gattungspaare | Unter anderem Cornaceae/Cornus, Rosaceae/Potentilla und Ulmus/Zelkova werden nicht durch ungeprüfte Gleichsetzung geschlossen. |

## Prüfungen und bekannte Quellenabweichungen

- Alle 90 modernen Zählspalten: explizite Zahlen, nichtnegativ; Originalsumme,
  Pinus-freie Summe und die vier Quell-Fußzeilen werden unabhängig nachgerechnet.
- Alle publizierten Prozentwerte werden gegen diese Zählungen geprüft.
  Zählungen werden nie aus gerundeten Prozenten rekonstruiert.
- Separate Koordinaten- und Klimablätter werden über eindeutige Probencodes
  abgeglichen. Das Koordinatenblatt schreibt `Longitude (°N)` und
  `Latitude (°E)`; die Werte stimmen numerisch mit Longitude/Latitude im
  kombinierten Blatt überein. Es erfolgt kein Vertauschen der Zahlen.
- Akzentunterschiede in wiederholten Probennamen werden nur bei ansonsten
  identischem Unicode-NFD-Namen akzeptiert und einzeln protokolliert.
  Die Blattverknüpfung verwendet eindeutige Probencodes und Quell-IDs.
- Das Zwischenblatt enthält 847 beschriftete Originaltaxonzeilen statt der
  im Artikel angegebenen 840. Die sieben letzten Namen stehen in `summary.json`.
  Dieser Befund wird weder still korrigiert noch zum Neuaufbau der finalen
  90-Gruppen-Matrix verwendet.
- Word-Tabellen übernehmen Gruppen ausschließlich über ausdrückliche
  vertikale Zellverbünde. Ein tatsächlich leeres Gruppenfeld wird nicht gefüllt.
- `coordinate_group_1e6deg` kennzeichnet gleiche auf sechs Dezimalstellen
  gerundete Koordinaten. Solche Proben müssen für spätere Datenaufteilungen
  zusammenbleiben; das ersetzt keine räumlich blockierte Validierung.

Die ausgeführten Ergebnisse und Datei-Prüfsummen stehen in `summary.json`
und `reference_manifest.json`. Die großen modernen Exporttabellen und Rohdateien
werden durch den folgenden Aufruf reproduziert und nicht eingecheckt.

| Ausgeführter Befund | Ergebnis |
|---|---:|
| Moderne Proben / gemeinsame Gruppen | 7.634 / 37 |
| Moderne Proben mit mindestens 100 gemeinsamen Zählungen ohne Pinus | 7.078 |
| Fossile Proben / davon über derselben Schwelle | 69 / 69 |
| Explizite Nullzellen in den 90 modernen Quellgruppen | 514.377 |
| Gebrochene Zählungen / Formeln in Quellzählfeldern | 0 / 0 |
| Geprüfte Prozent-, Koordinaten- und Klimazeilen | jeweils 7.634 |
| Koordinatengruppen / Proben in mehrfach belegten Gruppen | 6.192 / 1.968 |
| Akzentabweichungen in wiederholten Namensfeldern | 19 |

Die 556 modernen Proben unter der zusätzlichen gemeinsamen Zählschwelle
bleiben sichtbar im Export. Der größte absolute Fehler beim Nachrechnen der
publizierten Prozentwerte liegt unter `1.3e-14` Prozentpunkten.

`verification.json` dokumentiert eine zusätzliche Prüfung aller 285.011
exportierten Merkmalszellen, Prozentrechnungen, Schwellenflags und Prüfsummen.
Zwei vollständige Quellenläufe ergaben identische moderne Tabellen und
Zusammenfassungen. Drei Parser-Tests prüfen die Unterscheidung von Null und
Leerzelle, ausdrückliche Word-Zellverbünde und Akzentvergleich. Der Importer-Hash
ist unverändert; die alten Modell-/App-Tests wurden nicht erneut ausgeführt.
Fossile Altersgrenzen, Unsicherheitsklasse, Quelltiefe und Tiefenoffset sowie
der Status `candidate_depth_link` bleiben in beiden Exportvarianten erhalten.

## Reproduktion

1. Die drei alten Fossil-/Chronologiedateien gemäß
   `docs/thesis_research/source_audit_2026_10_29/sources.json` bereitstellen.
2. Die drei neuen Dateien gemäß `sources_2025.json` herunterladen.
   Der `supplementaryFiles`-Endpunkt liefert ein ZIP: daraus nur `mmc1.docx`
   in das Verzeichnis der neuen Quellen extrahieren. Originale unverändert lassen.
3. Vom Repository-Stamm aus ausführen (Python-Standardbibliothek genügt):

```bash
python -m unittest discover -s tools/thesis_import -p test_shared2025.py -v
python tools/thesis_import/prepare_shared2025.py \
  --raw-dir /path/to/old_sources \
  --published-dir /path/to/published2025_sources \
  --output-dir /path/to/new_shared37_output
```

Das Zielverzeichnis muss neu sein. Es entstehen je moderne/fossile
`*_shared_counts.csv` und `*_shared_percent.csv`, außerdem Evidenz,
Zusammenfassung und Manifest. In Prozentdateien sind nur die 37 Merkmalsfelder
umgerechnet; Metadatenfelder mit `count` behalten ihre ausdrücklich benannte
Zähleinheit. Die fossilen Ausgaben unterliegen weiterhin CC-BY-NC-3.0,
die modernen CC-BY-4.0; diese Quelldaten werden nicht unter die Code-Lizenz gestellt.

## Nächster Implementierungsschritt für Claude

Diese Variante separat laden; Schwellenfilter und geographischen Umfang sichtbar
festlegen und die Zahl verbleibender Proben ausgeben. Gleiche Koordinatengruppen
zusammenhalten und räumlich blockierte Validierung verwenden. Zunächst eine
einfache moderne Tann-/Pann-Vergleichsbasis bewerten, dann gegebenenfalls die
69 fossilen Proben mit ausgewiesenen Altersgrenzen anwenden. Modellwahl und
räumliche Blockgröße müssen dokumentiert werden; aus diesen Tabellen folgen
noch keine Rekonstruktionsgüte oder Klimaergebnisse. Bestehende Referenz und
ursprüngliche Kalibrierung nicht durch diese Variante ersetzen.
