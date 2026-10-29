# Quellenprüfung und Import-Arbeitspaket

Stand: 29.10.2026. Fortsetzung von PR #11 (`b02147261e3b4bb300ce1d4b109b98116ad1cb30`). Die Prüfung betrifft Datenquellen; Modell, Parameter und App wurden nicht geändert. Alle folgenden Auszählungen sind eigene Prüfungen der tatsächlich heruntergeladenen Dateien, keine aus Publikationstiteln abgeleiteten Abdeckungsbehauptungen.

## Ergebnis und Pilotentscheidung

| Quelle | Tatsächlicher Dateibefund | Entscheidung |
|---|---|---|
| EMPD2 SQLite | 8.746 Proben, 262.390 Messzeilen, 4.652 gemessene Originalcodes; 1.605 akzeptierte IDs | Geeignete Trainingsquelle für eine neue Pollen-Baseline; Filter und Taxonkonzepte zuerst festlegen |
| FloraWeb/Meusel | 2.111 Mappingzeilen, 2.085 unterschiedliche Taxon-IDs; Abies-alba-Beispiel mit Punkten und Polygonen | Nutzbare Grundlage für neue Präsenzfunktionen; kein Ersatz unter dem Namen der verlorenen Originalfunktionen |
| Ohrid 892362 | 308 Zeilen, 27 Spalten; davon 90 Proben bei 1,420–149,642 ka BP | Langer Pilot für ausdrücklich prozentbasierte Verfahren; kein Count-Modell |
| Monticchio LGM90D | 245 Proben; konservative Chronologie nur für 69 davon, 0,08575–20,22257 ka BP | Kurzer Rohzählungs-Pilot nach dokumentiertem Tiefenabgleich; kein 150-ka-Profil |
| Monticchio MONTWAW1 | 57 Proben, 18,65–27,45 m; keine passende Entity im geprüften Chronologiearchiv | Zurückstellen; nicht an LGM90D anhängen |
| Iznik | 111 Proben, 0,554–31,114 ka BP; 80 gebrochene Zählfelder | Erweiterungsregion Anatolien; Datierung und gebrochene Werte vor einem Count-Modell klären |

Empfehlung für Claude: zuerst EMPD2-Schema und Monticchio-Teilprofil importieren; Ohrid getrennt als langen Prozentdaten-Piloten erhalten. Damit sind technische Durchläufe möglich. Eine wissenschaftliche Freigabe oder ein kompletter belastbarer 0–150-ka-Rohzählungspilot ist hiermit nicht erreicht.

## Moderne Kalibrierung

**EMPD2:** [Primärarchiv](https://doi.org/10.1594/PANGAEA.909130), [SQLite-Datei](https://store.pangaea.de/Publications/ChevalierM-etal_2019/EMPD2.sqlite3), [Methodenbeschreibung](https://doi.org/10.5194/essd-12-2423-2020). Die veröffentlichte Lizenz ist CC BY 4.0. Die SQLite-Integritätsprüfung liefert `ok`. Alle 8.746 Metadatensätze sind als Nicht-Prozentdaten und nicht digitalisiert markiert; trotzdem enthalten 244 Proben zusammen 1.059 nicht ganzzahlige Messwerte. Keine negativen oder leeren Messwerte wurden in `p_counts.count` gefunden. Nicht runden und nicht ungeprüft eine diskrete Multinomial-Likelihood anwenden.

Die Datenbank umfasst `metadata`, `p_counts`, `p_vars`, `groupid` und `climate`. Gemessene Originalcodes, akzeptierte Taxon-IDs und Pflanzentaxa der Dissertation sind unterschiedliche Mengen. Die Koordinatenbox 25–75°N, 15°W–45°E enthält 6.064 Proben, einschließlich Erweiterungsregionen; sie definiert noch keinen europäischen Trainingssatz.

Der normale PANGAEA-TSV-Download zu 909130 ist eine **Klimatabelle**, keine Pollenmatrix. Er enthält 8.462 Zeilen. Gegenüber der SQLite-Datenbank gibt es 20 ausschließlich im TSV und 304 ausschließlich in SQLite vorkommende Proben-IDs. Daher zunächst die miteinander verknüpften Tabellen innerhalb der gesperrten SQLite-Version verwenden. Die Differenz ist dokumentiert, ihre Ursache noch nicht geklärt. Es wurde keine Identität mit dem GitHub-Release unterstellt; der gelesene GitHub-Stand ist `737b76eb269b071422edded6ac332654ff83c103`.

Die EMPD2-Publikation nennt **WorldClim2, 30 Bogensekunden, nächstgelegenen Gitterpunkt** als Klimazuordnung. Das ist eine andere Kalibrierungsbasis als CRU TS 4.01 mit 1961–1990 in der Dissertation. Die konkrete Referenzperiode der heruntergeladenen Klimatabelle wurde nicht separat dateiintern belegt; sie darf nicht stillschweigend als CRU bezeichnet werden. Die SQLite-Tabelle enthält für alle Proben nichtleere `t_ann` und `p_ann`; das bestätigt keine geographische oder fachliche Eignung jeder Probe.

Die [2025-Aufbereitung](https://doi.org/10.1594/PANGAEA.973997) bleibt eine getrennte Variante: 90 Taxa/7.634 Proben, zusätzliche niedrigere Nicht-Pinus-Summen und Zwischenfassungen laut Archivbeschreibung. Die XLSX-Datei wurde in dieser Sitzung nicht heruntergeladen oder vollständig geprüft. Sie ist kein Ersatz für den hier geprüften SQLite-Stand.

**FloraWeb/Meusel:** [Quellen- und Lizenznachweis](https://www.floraweb.de/inhalt/datenquellen.html), Datenrepository auf Commit `5ab8e638278c86035bf3c9c3224c90de4880327e`. Tatsächlich geprüft wurden Mapping, zwei README-Dateien und `taxon_id_1.geojson` (Abies alba). Die 2.111 Mappingzeilen enthalten 24 ausdrücklich als `duplicate_synonym=TRUE` markierte Zeilen; unterschiedliche Taxon-IDs belegen noch keine vollständige oder gültige Geometrie für jede Zeile. Die Mappingtabelle enthält keine Originalnamen der Gattung **Olea**. Der Dissertation-Beispieltaxon Olea europaea ist damit über diesen Mappingweg nicht beschafft.

Das Beispiel umfasst 13 Polygone und 29 Punkte, sämtlich als nativ und Vorkommen bezeichnet. Die README nennt EPSG:4326; die Datei kennzeichnet CRS84. Für den Import sind Koordinaten als Länge/Breite zu behandeln. Punkte haben keine dokumentierte Arealfläche und dürfen nicht ohne eigene Annahme zu flächigen Präsenzen gepuffert werden. Andere Dateien können erloschene, fragliche oder synanthrope Vorkommen enthalten. Außerhalb eines kartierten Areals ist ohne Prüfung des Kartenumfangs keine gesicherte Absenz gegeben.

Pflichtattribution laut Anbieter: Meusel & Jäger, *Vergleichende Chorologie der zentraleuropäischen Flora*, Bände I–III, 1965/1978/1992; digitalisiert bereitgestellt von Gunnar Seidler, Erik Welk und Helge Bruelheide, Universität Halle-Wittenberg. CC BY 4.0. Eine wissenschaftliche Kontaktaufnahme wird empfohlen; es wurde keine Nachricht versendet.

## Fossile Piloten und Chronologien

**Ohrid:** [Datensatz](https://doi.org/10.1594/PANGAEA.892362), [Corrigendum](https://doi.org/10.5194/bg-13-1423-2016-corrigendum). Archivdatei CC BY 3.0; Corrigendum CC BY 4.0. Die gesamte Datei reicht von 1,420 bis 501,828 ka BP. Innerhalb 0–150 ka beträgt der Medianabstand 1,638 ka und der größte Probenabstand 2,427 ka. Das belegt weder kontinuierliche Abdeckung bis heute noch eine 50-Jahres-Informationsauflösung.

Die 19 einzeln aufgeführten Pollentypen ergeben zusammen 77,02–178,41 %; in 265 der 308 Zeilen liegt die Summe über 100 %. Das ist kein Anlass zur automatischen Fehlerkorrektur: Das Corrigendum erklärt in Tabelle 1, dass die Basis für AP/NAP Pinus ausschließt, während der Archivheader für Pinus eine Summe mit Pinus angibt. Dazu sind nur ausgewählte Typen veröffentlicht. Bezugsgrößen pro Spalte erhalten; keine gemeinsame Schließung auf 100 %, keine künstlichen Zählungen. AP, modifiziertes AP und Vegetationsgruppen nicht zusätzlich zu denselben Einzeltypen als unabhängige Beobachtungen verwenden.

Das Corrigendum nennt eine Nachbearbeitung von 104 chemisch beeinträchtigten Proben und zusätzliche Nachzählungen, auch in MIS 4–2. Es belässt die ursprüngliche Chronologie. Das Archiv nennt zehn Tephralagen und Orbitaltuning. Die Prozenttabelle liefert Alter, aber keine zugehörigen Tiefen, Datierungsensembles oder Altersfehler pro Probe. Quantitative gemeinsame Altersunsicherheit bleibt offen; ein Vergleich mit orbital beeinflussten Reihen ist nicht chronologisch unabhängig.

**Monticchio:** [LGM90D-Zählungen](https://doi.org/10.1594/PANGAEA.739452), [Entity](https://doi.org/10.1594/PANGAEA.804591), [EpdAgeAll](https://doi.org/10.1594/PANGAEA.804595), [EpdAgeCut](https://doi.org/10.1594/PANGAEA.804596). Die Zählungen tragen **CC BY-NC 3.0**; die drei Chronologietabellen CC BY 3.0. Diese Unterschiede müssen auch bei Weitergabe abgeleiteter Dateien erhalten bleiben.

LGM90D besitzt 245 Zeilen und 136 Spalten. Die Datenfelder enthalten neben Pollen auch Sporen, unbestimmbare Reste und Volumen; die Spaltenzahl ist keine Taxonzahl. Die numerischen `[#]`-Felder sind nichtnegativ und ganzzahlig. Entity `932`, Event `LGM90D`, gehört zu Site `382`; die Entity-Tabelle begrenzt den nicht extrapolierten Tiefenbereich auf 0–12 m. EpdAgeAll enthält 245 Werte bis 76,21012 ka BP, EpdAgeCut dagegen nur 69 Werte bis 20,22257 ka BP. Die vollständige All-Kurve darf die ausgeschlossenen Tiefen nicht wieder als belastbare lange Chronologie einführen.

61 der 69 Cut-Tiefen stimmen exakt mit der Zähltabelle überein. Acht weitere liegen exakt 0,005 m daneben, passend zur unterschiedlichen Zahl der Nachkommastellen. Der Prüfcode weist unter dieser engen Toleranz für jede Cut-Tiefe genau einen Partner nach und verhindert Mehrfachzuordnungen. Dies ist ein dokumentierter **Zuordnungskandidat**, keine allgemeine Erlaubnis zum nächsten Nachbarn. Die vollständige Liste steht in `audit_results.json`. Nach diesem Kandidatenabgleich bleiben 176 Zählproben ausgeschlossen. Medianabstand der 69 Alter: 0,17627 ka; größte Lücke: 2,25895 ka.

`Age min/max` und Unsicherheitsklassen sind marginale Angaben, keine gemeinsamen Alters-Tiefen-Ziehungen. Ihre Wahrscheinlichkeitsbedeutung und die ursprünglichen Datierungen sind vor quantitativer Unsicherheitsfortpflanzung zu prüfen. Proben nicht unabhängig innerhalb der Grenzen stören. Die archivierte Höhe von 1.326 m wurde nur übernommen und nicht unabhängig bestätigt; nicht ungeprüft für die Klimazuordnung verwenden. MONTWAW1 hat im geprüften Entity-/Altersarchiv keinen passenden Eintrag; gleicher See bedeutet nicht identische Tiefe oder Chronologie.

**Iznik:** [Zählungen](https://doi.org/10.1594/PANGAEA.858056), [Originalartikel](https://doi.org/10.5194/cp-12-575-2016). Der erneut heruntergeladene Zählungsdatensatz ist SHA256-identisch mit dem Nachweis aus PR #11. Die Datei umfasst 111 Zeilen und 135 Spalten, einschließlich Summen und Nicht-Pollen-Daten. 80 `[#]`-Felder sind gebrochen. Der Artikel beschreibt die Verbindung mehrerer Kerne über die Avellino-Tephra und eine erweiterte Alterskurve nach Roeser; für den Abschnitt um 6,5 ka werden Reservoirprobleme ausdrücklich genannt (S. 586). Abbildung 4, S. 579, wurde visuell geprüft. Das heruntergeladene Supplement ist ein Pollendiagramm, kein gefundenes Altersensemble. Eine weitere Chronologiebeschaffung ist erforderlich; ein Alter in jeder Datenzeile ersetzt diese Prüfung nicht.

## Nachweise und nächster Schritt

- `sources.json`: 16 aktuelle, SHA256-gesperrte Dateien mit Lizenz, Rolle, Status und tatsächlichem Download-URL. Zusätzlich bleiben alle 16 historischen Manifest-Einträge aus PR #11 erhalten; davon wurden hier nur Iznik und EPD-Entity erneut mit identischer SHA256 bestätigt. Die übrigen sind ausdrücklich nicht in dieser Sitzung erneut verifiziert.
- `audit_results.json`: gemessene Tabellenumfänge, Zeitabstände, Rundungsabgleich und Problemfelder.
- `verify_sources.py`: reproduzierbare Dateiprüfung mit Python-Standardbibliothek; Download fehlender Dateien optional. Originaldateien werden nicht im Git-Branch mitgeführt.
- `import_spec.md` und `taxon_mapping_candidates.csv`: konkrete nächste Arbeit für Claude.

Offen bleiben ein fachlich freigegebenes Taxonmapping, die vollständige Abdeckung der verlorenen 279 Transferfunktionen, räumliche Trainingsvalidierung, gemeinsame Altersensembles und ein belastbarer langer Rohzählungspilot. `quality_checked` und `eligible` werden deshalb für keine Quelle wissenschaftlich vergeben. Eine erfolgreiche technische Prüfung ist nur ein Schritt zur späteren Modellfreigabe.
