# Arbeitsplan für Claude: europäische Multiproxy-Rekonstruktion 0–150 ka
Stand: 29. Oktober 2026. Planungsdokument; keine neue App oder neuen Modellläufe wurden hier implementiert.

## 1. Auftrag und Ausgangslage

Ziel ist eine interaktive Forschungs-App für quantitative Klima-Rekonstruktionen im europäischen Raum während der letzten etwa 150.000 Jahre. Grundlage sind Netzels Dissertation „Quantitative paleoclimate reconstructions in the European region based on multiple proxies“ (eingereicht 2022, veröffentlicht 2023) und die inzwischen weiterentwickelte Kinneret-Codebasis.

Verfügbar sind im Wesentlichen die Dissertation und öffentliche Kinneret-Dateien. Die ursprünglichen 279 Taxon-Transferfunktionen, sämtliche verwendeten Altersmodelle, die vollständige Standortauswahl und privat übermittelten Zusatzproben stehen nicht zur Verfügung. Die Beschaffung oder Neuerstellung dieser Bestandteile gehört zum Projekt; der Autor soll sie nicht als Voraussetzung liefern müssen.

Das Ziel ist eine dokumentierte Weiterentwicklung mit überprüfbarer Verwandtschaft zur Dissertation. Eine exakte Reproduktion aller damaligen Ergebnisse bleibt ein gesondertes, teilweise blockiertes Ziel. Neue Datenbankversionen, neu trainierte Funktionen und neue statistische Annahmen müssen als solche bezeichnet werden.

## 2. Entscheidung zum Repository

Empfehlung: eigenes Repository, beispielsweise `europe_multiproxy_reconstruction`, aus einem festgehaltenen Stand des Kinneret-Projekts ableiten. Kinneret bleibt die dokumentierte Referenz für die Publikation von 2025. Die neue App darf daraus geeignete Infrastruktur übernehmen.

Geprüfte öffentliche Ausgangsbasis:
- Repository: https://github.com/TimonNetzel/kinneret_reconstruction_python
- Festgehaltener Commit: `2d22f160d591b4d50e73fa6306a290f16d8599cb`
- Bestehender Multiproxy-Plan: https://github.com/TimonNetzel/kinneret_reconstruction_python/blob/2d22f160d591b4d50e73fa6306a290f16d8599cb/docs/plan_multiproxy.md
- Methodenabgleich: https://github.com/TimonNetzel/kinneret_reconstruction_python/blob/2d22f160d591b4d50e73fa6306a290f16d8599cb/docs/abgleich_thesis.md
- Standortplan: https://github.com/TimonNetzel/kinneret_reconstruction_python/blob/2d22f160d591b4d50e73fa6306a290f16d8599cb/docs/plan_standorte.md

Diese Dokumente und die README wurden gelesen. Die Implementierung und behaupteten numerischen Verbesserungen wurden in diesem Planungsschritt nicht unabhängig getestet.

Übernehmen: Konfigurationsmuster, Standortverwaltung, Lauf-Fingerprints, Diagnostik, Ergebnisexport und App-Grundgerüst, soweit sich diese bei einer Codeprüfung bewähren. Die aktuelle App betrachtet vorberechnete Ergebnisse; darauf lässt sich eine erste Version sinnvoll aufbauen.

Anpassen: Region, Klimaeingaben, Taxonomie, Pollenkonvertierung, Chronologien und wissenschaftliches Beobachtungsmodell. Das Kinneret-Modell mit Biomen und Wintertemperatur ist keine fertige taxonbasierte europäische Kalibrierung für Jahresmitteltemperatur. Die vorhandene Levante-Klimadatei deckt Europa nicht ab.

Zunächst ein Forschungs-Repository mit klar getrennten Modulen; eine zusätzliche gemeinsame Softwarebibliothek erst herauslösen, wenn beide Projekte tatsächlich dieselbe stabile Funktion benötigen. Kein vollständiger Neubau der Oberfläche, bevor der erste wissenschaftliche Durchlauf funktioniert.

## 3. Was an den bisherigen Plänen geändert werden muss

Der vorhandene Multiproxy-Plan setzt in D1–D6 teilweise Datenlieferungen des Autors voraus. Diese Annahme wird ersetzt:

| Fehlender Bestandteil | Beschaffungs- oder Ersatzweg | Freigabekriterium |
|---|---|---|
| D1: originale Taxon-Transferfunktionen | Öffentliche Repositorien und Supplements prüfen; parallel Pflanzenareale und moderne Klimadaten für ein neues Training beschaffen | Jede Funktion besitzt Trainingsquellen, Taxondefinition, Klimavariablen, Version und räumliche Validierung; kein erfundenes Original |
| D2: Pollentyp–Taxon-Zuordnung | Originalnamen aus Rohdaten erhalten; dokumentierte Zuordnung über EPD/Neotoma und EMPD erstellen | Mehrdeutige Typen bleiben markiert; Gattungen werden nicht willkürlich zu Arten |
| D3: Alters-Tiefen-Verteilungen | Datierungen und publizierte Ensembles suchen; andernfalls Chronologie neu modellieren, soweit Datierungen dies erlauben | Alterskonvention, Unsicherheit, Hiatus und gemeinsame Altersfehler dokumentiert |
| D4: Referenzreihen | SISAL, MEDSTACK, Soreq und weitere Primärarchive beziehen | Tatsächliche Dateien, Altersskala und Vorverarbeitung nachgewiesen |
| D5: Referenzvariable, Vorzeichen, Fenster | Aus Standortliteratur begründen und in Sensitivitätsläufen prüfen | Kein automatisches Vorzeichen allein aus maximaler Korrelation |
| D6: ursprünglicher Code | Veröffentlichte Repositorien prüfen; bei Nichterfolg Methoden nachvollziehbar neu implementieren | Unklare Formeln oder Kombinationen als offene Methodenentscheidung führen |

Die bereits vorgeschlagene Übertragung „Taxa wie Biome behandeln“ ist eine zu prüfende neue Modellvariante. Sie ist keine nachgewiesene exakte Umsetzung der Dissertation. Ebenso ist eine Korrelation oder Wavelet-Übereinstimmung noch kein kalibriertes Beobachtungsmodell für absolute Temperatur und Niederschlag.

## 4. Daten beschaffen: priorisierter Katalog

Statusbegriffe:
- **Vorhanden:** im bereits zusammengestellten Datenpaket enthalten, aber nicht automatisch wissenschaftlich freigegeben.
- **Öffentlich identifiziert:** Quelle und Inhalt geprüft; vollständiger Import, Qualitätsprüfung und Verknüpfung stehen noch aus.
- **Recherche offen:** geeigneter Kandidat, tatsächliche benötigte Rohdaten oder Chronologie noch nicht gesichert.

### 4.1 Fossile Pollen und Chronologien

| Priorität | Quelle und Zugang | Geplanter Einsatz und Grenze |
|---|---|---|
| P0 | EPD über Neotoma: https://www.neotomadb.org/data ; EPD-Zugang: https://epdweblog.org/browse-get-data/ | Rohzählungen, Proben-Tiefen und Metadaten zuerst beziehen. Europa-Pool systematisch nach Standort, realer Altersabdeckung und Chronologie filtern. Ein Datenbankeintrag allein garantiert keine Chronologie bis 150 ka. |
| P0 | LegacyPollen: https://doi.org/10.1594/PANGAEA.929773 | Ergänzender harmonisierter fossiler Pollenbestand. Identitäten mit EPD/Neotoma abgleichen, keine mehrfachen Profile zählen. Originalzählungen und harmonisierte Fassung getrennt erhalten. |
| P0 | LegacyAge: https://doi.org/10.1594/PANGAEA.933132 | Datierungen und harmonisierte Chronologien zuordnen. Nutzbarkeit pro Profil und Zeitbereich prüfen; keine generelle 150-ka-Abdeckung annehmen. |
| P0 | Giesecke et al. Altersdaten: https://doi.org/10.1594/PANGAEA.804597 | Teile bereits vorhanden. Kontrollpunkte und ältere EPD-Chronologien als Verbindung zur Dissertation. Nicht als Ersatz für fehlende tiefe Chronologie behandeln. |
| P0 | Ohrid: https://doi.org/10.1594/PANGAEA.892362 | Öffentlich geprüfte Taxon-Prozentwerte für die letzten 500 ka. Interessanter langer Pilot im Zielgebiet. AP und Vegetationssummen nicht zusätzlich zu ihren Bestandteilen als unabhängige Daten verwenden. Keine Rohzählungen aus Prozentwerten erfinden. |
| P1 | Van: https://doi.org/10.1594/PANGAEA.853776 und https://doi.org/10.1594/PANGAEA.853729 ; Multiproxy-Serie: https://doi.org/10.1594/PANGAEA.853731 | Abschnitte 10–110 und 110–135 ka identifiziert; Originalquellen teilweise vorhanden. Kerne, Chronologien und Überlappung separat prüfen. Anatolien als explizite Erweiterungsregion, nicht stillschweigend europäische Flächenabdeckung. |
| P1 | Tenaghi Philippon: https://doi.org/10.1594/PANGAEA.943593 ; XRF: https://doi.org/10.1594/PANGAEA.943592 | Öffentliche 500-ka-Daten enthalten aggregierte Pollengruppen und Geochemie. Als Vergleich oder eigens spezifiziertes Gruppenmodell verwendbar; diese Tabelle ist keine vollständige taxonspezifische Zählmatrix. |
| P1 | Monticchio LGM90D: https://doi.org/10.1594/PANGAEA.739452 ; MONTWAW1: https://doi.org/10.1594/PANGAEA.739556 | Öffentliche Rohzählungsprofile identifiziert. Passende Chronologien und tatsächliche Altersbereiche müssen noch nachgewiesen werden; die alten Kernnamen nicht ungeprüft zu einem 150-ka-Profil verbinden. |
| P1 | Kinneret: https://doi.org/10.1594/PANGAEA.884740 ; Iznik: https://doi.org/10.1594/PANGAEA.858056 ; Dead Sea: https://doi.org/10.1594/PANGAEA.900564 | Bereits identifizierte Fallstudien zum Test der Verbindung zur Dissertation und Kinneret-App. Kürzere oder abschnittsweise Reihen; zusätzliche damalige Privatproben fehlen teilweise. |
| P2 | Weitere Langarchive: Padul, Ioannina, La Grande Pile, Les Echets, Tenaghi-Rohmatrix und weitere Ohrid-Daten | Gezielte Recherche in Primärarchiven und Supplements. Vor Aufnahme müssen tatsächliche Tabelle, Taxa, Chronologie, Lizenz und Nutzungsrolle feststehen. Diese Kandidaten sind noch keine gesicherten Downloadbestände. |

Ohrid-Besonderheiten: Das öffentliche Archiv bezieht sich auf eine korrigierte Veröffentlichung; bei der Datenübernahme den Korrekturstand dokumentieren. Seine Chronologie enthält Tephra-Anker und Orbitaltuning. Ein Vergleich mit orbital beeinflussten Referenzen ist deshalb nicht vollständig chronologisch unabhängig. Die Veröffentlichung zur höher aufgelösten 160–70-ka-Rekonstruktion bezeichnete die entsprechenden höher aufgelösten Pollen 2019 als noch nicht öffentlich; eine spätere Freigabe muss gesondert gesucht werden.
Quellen: https://doi.org/10.5194/bg-13-1423-2016 und https://doi.org/10.5194/cp-15-53-2019

### 4.2 Moderne Kalibrierung: der wichtigste Ersatz für verlorene Dateien

**Weg A: taxonbasierte Weiterentwicklung der Dissertation.**
Digitalisierte Meusel-Areale prüfen und mit moderner Klimatologie verschneiden. Zugang:
- Quellenbeschreibung und Attribution: https://www.floraweb.de/inhalt/datenquellen.html
- Verlinkte Geodaten: https://gitlab.opencode.de/bfn/open-access/floraweb/floraweb-data-maps/-/tree/main/Data_in/Areale?ref_type=heads
- CRU TS 4.01 als historischer Bezug: https://doi.org/10.5285/58a8802721c94c66ae45c3baa4d814d0

FloraWeb bestätigt öffentliche Areal-Geodaten unter CC BY 4.0. Ob damit sämtliche 279 ursprünglichen Taxa abgedeckt werden, ist offen. Taxonomische Konzepte, Arealgrenzen und Rasterung können vom damaligen Input abweichen. Fehlende Vorkommensmeldungen sind keine gesicherten Absenzen. Verbreitung außerhalb Europas kann für eine unverzerrte Klimanische wichtig sein. Eine neue Kalibrierung muss vollständig versioniert werden; ein anderes CRU-Release darf nicht unbemerkt mit den damaligen 1961–1990-Bezügen vermischt werden.

**Weg B: moderne Pollenspektren als ergänzende Kalibrierung und Referenzmodell.**
- EMPD2 Rohzählungen: https://doi.org/10.1594/PANGAEA.909130
- Versioniertes Datenrepository: https://github.com/EMPD2/EMPD-data
- Dokumentation: https://doi.org/10.5194/essd-12-2423-2020
- Aufbereitete Fassung von 2025: https://doi.org/10.1594/PANGAEA.973997

Die Fassung von 2025 bietet 7.634 Proben mit 90 Taxa sowie eine zusätzliche Tabelle mit ursprünglichen 840 bzw. auf 192 gruppierten Taxa. Die 90-Taxa-Auswahl wurde für Südeuropa aufbereitet; sie sollte nicht ungeprüft ganz Europa vorgeben. Rohzählungen ermöglichen eine zusätzliche Modellfamilie für Pollenspektren. Diese benötigt ein eigenes Training und ist kein Ersatz unter identischem Namen für pflanzenbasierte Präsenzfunktionen.

Empfehlung: Beide Wege auf einem gemeinsamen, räumlich getrennten Validierungssatz vergleichen. Erst danach die Standardmethode wählen. Ein einfaches modernes Pollen-Klimamodell dient als überprüfbare Vergleichsbasis, während die aufwendigere Dissertation-Variante aufgebaut wird.

### 4.3 Höhlen- und Isotopendaten

| Priorität | Quelle | Verwendung und Einschränkung |
|---|---|---|
| P0 | SISALv3: https://doi.org/10.5287/ora-2nanwp4rk ; Mirror: https://www.ncei.noaa.gov/pub/data/paleo/speleothem/SISAL-v3/ | Daten bereits ausgewertet und als Kandidaten exportiert. δ18O, vielfach δ13C, teilweise Spurenelemente; Datierungen und Hiatusinformationen mitnehmen. V3 ist eine Aktualisierung, kein nachgewiesener Originalstand der Dissertation. |
| P0 | SISAL-Altersensembles: https://doi.org/10.5281/zenodo.10726619 | Verfügbarkeit und Verknüpfung pro Entity prüfen. Ensembles nicht als voneinander unabhängige Messungen zählen. |
| P0 | MEDSTACK: https://doi.org/10.1594/PANGAEA.790006 | Vorhandene öffentliche Primärquelle; genaue Dissertation-Vorverarbeitung ungesichert. Langzeitreferenz. Das archivierte Zeitraster von 1.000 Jahren begrenzt den Informationsgehalt. |
| P1 | Soreq, Grant 2012: https://doi.org/10.25921/mf4a-k406 | Öffentliche Reihe bis etwa 161 ka; verschiedene Alterszuordnungen erhalten und Chronologieabhängigkeit prüfen. Regionaler Nahost-Bezug, keine allgemeine europäische Temperaturkalibrierung. |
| P1 | Soreq/Peqiin 2003: https://doi.org/10.25921/8a4v-ac50 | Weitere lange Isotopenarchive; mit SISAL-Einträgen und Kompositen auf Duplikate prüfen. |
| P1 | NGRIP: https://doi.org/10.25921/g2m5-qd24 | Großräumiger Vergleich bzw. begründete Kopplung; ungefähr 123 ka, somit nicht bis 150 ka. Vorhandene cal-BP-Spalten nicht erneut von b2k nach BP umrechnen. |
| P1 | LR04: https://www.lorraine-lisiecki.com/stack.html ; https://doi.org/10.1594/PANGAEA.704257 | Globaler Eisvolumen-/Tiefsee-Kontext; kein direkter lokaler Thermometer für Europa. |
| P2 | Van δ18O und Ca/K: https://doi.org/10.1594/PANGAEA.853763 ; https://doi.org/10.1594/PANGAEA.853764 | Same-core-Multiproxy-Pilot 110–135 ka; gemeinsame Chronologie und Archivabhängigkeit modellieren. |

SISAL-Stand dieses Projekts: Die frühere Auswahl umfasst 201 aktuelle Profile aus 106 Höhlen in 25–75°N und 15°W–45°E, jeweils mindestens 20 numerische δ18O-Werte und eine ursprüngliche Altersspanne über 2.000 Jahre. Davon überlappt die minimale/maximale Altersspanne bei 194 Profilen aus 105 Höhlen das Zielintervall 0–150 ka.

| Intervall | Profile mit überlappender Altersspanne | Höhlen |
|---|---:|---:|
| 0–11,7 ka | 102 | 71 |
| 11,7–30 ka | 54 | 36 |
| 30–70 ka | 38 | 26 |
| 70–115 ka | 66 | 42 |
| 115–150 ka | 54 | 36 |

Dies ist eine eigene Metadatenauszählung, **keine Anzahl durchgehend nutzbarer oder unabhängiger Reihen**. Ein Profil kann in mehreren Zeilen erscheinen; ein Altersbereich kann große Lücken enthalten. Die Box enthält auch Teile von Nordafrika, Anatolien und der Levante. Für die eigentliche Auswahl müssen Messpunkte im jeweiligen Fenster, Hiatus, Datierungsqualität, Einzugsgebiet und Abhängigkeiten geprüft werden.

Die ursprüngliche 150-km-Regel kann als Vergleich dienen. Das Auswahlverfahren sollte zusätzlich Überlappung, Chronologie, reale Auflösung und regionale Feuchtequellen berücksichtigen. Nähe allein macht eine Höhle nicht zum geeigneten Proxy für Temperatur oder Jahresniederschlag.
SISAL-Dokumentation: https://doi.org/10.5194/essd-16-1933-2024

### 4.4 Zusätzliche Daten nur mit klarer Rolle

- **Weitere lokale Proxies:** Chironomiden, Seesediment-Geochemie, Biomarker, Seespiegel und unabhängige Temperaturindikatoren gezielt für Pilotregionen suchen. Einbeziehen erst mit quantifizierbarem Beobachtungsmodell oder als ausdrücklich qualitative Prüfung.
- **Modellfelder:** PMIP-Experimente für verfügbare Zeitfenster: https://pmip.lsce.ipsl.fr/protocols_and_data/model_database . 6 ka, 21 ka und heutige Felder liefern keinen belegten kontinuierlichen Prior über 150 ka. Eine transiente Simulation erst nach Prüfung von Zeitabdeckung, Forcings, Region und Variablen einsetzen.
- **Kurze unabhängige Vergleichsdaten:** Temp12k https://doi.org/10.25921/4ry2-g808 ; MARGO https://doi.org/10.1594/PANGAEA.733406 . Nur im jeweiligen Zeitfenster und für die tatsächlich beobachtete Variable verwenden.
- **Pollenbasierte Klimaprodukte:** LegacyClimate https://doi.org/10.1594/PANGAEA.930512 und vergleichbare Produkte als Methodenvergleich. Gemeinsame Eingangspollen verhindern eine unabhängige Validierung.
- **CO2 und Orbit:** mögliche Randbedingungen bzw. Sensitivitätsvariablen. Vegetationsreaktionen unter glazialem CO2, Migration und menschlichem Einfluss können die Übertragbarkeit moderner Beziehungen verändern. Keine zusätzliche unabhängige Proxyinformation aus einem Tuning-Ziel erzeugen.

## 5. Automatisierbare Datenpipeline

Die Recherche darf breit sein; die statistische Aufnahme bleibt begründet. Jeder Fund erhält eine Rolle und einen dokumentierten Zustand.

Vorgeschlagene Dateien:
- `data_catalog/sources.yml`: Quellen, Versionen, URLs und Status.
- `data_catalog/records.parquet`: Kerne/Entities, Koordinaten, Proxys, reale Altersabdeckung.
- `data_catalog/dependencies.csv`: gleiche Probe, gleicher Kern, gleiche Höhle, Komposit–Komponente, gemeinsame Chronologie und Tuning-Ziele.
- `data_catalog/taxon_mapping.csv`: Originaltyp, harmonisierter Typ, zulässige Zuordnungen, Begründung und ungelöste Fälle.
- `data_catalog/gaps.md`: fehlende Dateien, alternative Quellen und Auswirkung auf den nächsten Meilenstein.
- `data/raw/`: unveränderte, versionierte Originaldateien; große Dateien extern ablegen und durch Prüfsummen referenzieren.
- `data/processed/`: reproduzierbar erzeugte Tabellen.
- `configs/sites/`, `configs/models/`, `runs/` und `app/`: Standorte, Modellannahmen, Ergebnisse und Oberfläche.

Mindestfelder pro Quelldatei:
`source_id, doi, citation, version, retrieved_at, landing_url, download_url, license, sha256, byte_count, retrieval_status`.

Mindestfelder pro Datenreihe:
`record_id, original_id, site_id, archive_group, lat, lon, elevation, proxy, unit, isotope_standard, value_type, depth_unit, age_unit, age_origin, chronology_id, age_model_method, hiatuses, role, quality_flags`.

Statuskette:
`discovered → downloaded → parsed → chronology_linked → quality_checked → eligible`.
Separat kennzeichnen: `missing`, `restricted`, `metadata_only`, `digitized`, `excluded`. Eine erfolgreiche HTTP-Anfrage gilt erst nach Prüfung des Dateiinhalts als Download.

Wichtige Regeln:
1. Originalwerte, Taxonnamen und Altersangaben erhalten.
2. Intern Jahre vor 1950 verwenden; ka/BP/b2k nur anhand expliziter Metadaten umrechnen. Negative heutige BP-Werte können zulässig sein.
3. Zählungen, Prozentwerte und binäre Präsenz getrennt behandeln. Keine künstlichen Zählungen für eine Count-Likelihood.
4. Datierungsensembles gemeinsam pro Kern/Entity ziehen; keine unabhängige Altersstörung jeder Probe.
5. Hiatus und fehlende Messungen erhalten. Außerhalb der realen Abdeckung keine stillschweigende Extrapolation.
6. Medianabstand und große Lücken berechnen; Rechengitter und effektive zeitliche Auflösung separat führen.
7. Komposite und Originalreihen deduplizieren; AP nicht zusätzlich zum gleichen Pollenspektrum als unabhängigen Proxy zählen.
8. Neue Quellenstände als neue Version einlesen. Veröffentlichte Ergebnisstände behalten ihren Daten-Lock.
9. Reproduzierbare Downloader dürfen öffentlich zugängliche Dateien automatisch beziehen. Zugriffsbeschränkte Quellen als Lücke dokumentieren; Kontaktentwürfe gegebenenfalls vorbereiten, aber keine Nachrichten automatisch versenden.

## 6. Wissenschaftliches Vorgehen und Freigaben

### Phase A: Dateninventar und Softwarebasis
Zuerst Ausgangscommit, Dissertation und vorhandenes Datenpaket registrieren. Bestehende Tests und einen verfügbaren kleinen Kinneret-Referenzlauf prüfen. Vorhandene Originalergebnisse nicht als bereits unabhängig validiert darstellen.

**Ergebnis:** neues Projektgerüst, maschinenlesbares Quellenregister, explizite Lückenliste und nachvollziehbare Herkunft der übernommenen Software.

### Phase B: Kalibrierung und Pilotdaten
EMPD2 sowie Areale und Klimadaten importieren. Taxonomische Kompatibilität prüfen. Parallel einen langen Standort und einen gut datierten kürzeren Standort aufbereiten. Ohrid ist ein Kandidat für die lange Prozentwert-Variante; ein Count-Modell benötigt zusätzlich echte Zählungsdaten.

**Ergebnis:** ein vollständiger Durchlauf von Originaldateien zu modellfähigen Eingaben; Tabellen mit tatsächlicher Altersabdeckung und Auflösung. Fehlende Original-Transferfunktionen blockieren nicht das gesamte Projekt, sondern führen zur klar benannten Neukalibrierung.

### Phase C: Pollen allein als Referenz
Zielvariablen zuerst Jahresmitteltemperatur (TANN, °C) und Jahresniederschlag (PANN, mm/Jahr). Wintertemperatur nur als gesonderte Variable. Zunächst Pollen ohne zusätzliche Referenzkurven rekonstruieren. Moderne Kalibrierung räumlich blockiert prüfen; zufällig verteilte Nachbarproben sind kein ausreichend unabhängiger Test.

Taxon-Präsenzfunktionen, Verknüpfung mehrerer Taxa und ein trainiertes Pollenspektren-Modell explizit unterscheiden. Die mathematische Likelihood und die zu schätzende Größe müssen vor dem großen Lauf feststehen.

**Ergebnis:** nachvollziehbare Baseline, Fehlermaße, Unsicherheitsintervalle und Kennzeichnung von Proben ohne geeignete moderne Analoga.

### Phase D: ein zusätzlicher Proxy
Zuerst eine geeignete Höhle oder eine andere begründete Referenz ergänzen. Die Dissertation-Variante mit Korrelations-/Wavelet-Gewichtung kann als Vergleich implementiert werden. Ein erweitertes Beobachtungsmodell benötigt eine vollständig spezifizierte Wahrscheinlichkeitsverteilung und Fehlerstruktur.

Das im aktuellen Plan vorgeschlagene standardisierte AR(1)-Referenzmodell muss unter anderem auf unregelmäßige Abstände, Altersfehler, Skalierung und Identifizierbarkeit geprüft werden. Sein Kopplungsparameter ist nicht ohne zusätzliche Annahmen das empirische R². Auch die Verknüpfung eines Isotopensignals mit absolutem lokalen Klima muss begründet sein.

**Ergebnis:** dokumentierter Vergleich Pollen allein / Pollen plus Proxy / Referenzmodell allein; Nachweis, welchen Teil des Ergebnisses die Referenz tatsächlich bestimmt.

### Phase E: Chronologie und Mehrfachproxies
Altersensembles bereits im Pilot berücksichtigen, sobald eine zeitliche Kopplung genutzt wird. Ein nachträglich verbreitertes Werteintervall ersetzt keine zeitlich korrelierten Altersfehler. Gemeinsam genutzte Kerne, Höhlen und Komposite als abhängige Beobachtungen behandeln. δ18O-Vorzeichen und Klimabezug sind regional bzw. prozessbezogen.

**Ergebnis:** Robustheitsläufe über Altersmodelle, Vorzeichen, Referenzauswahl und Glättung. Originale und modifizierte Chronologie getrennt anzeigen; kein verstecktes Tuning auf die gewünschte Übereinstimmung.

### Phase F: räumliche Erweiterung und App
Zuerst Standortrekonstruktionen, dann regionale Modelle. Karten dürfen nur die durch Daten und Modellannahmen gestützten Bereiche zeigen. Für 0–150 ka wird die räumliche Abdeckung zeitlich stark variieren; eine lückenlos gefärbte Europakarte wäre kein Qualitätsnachweis.

**Ergebnis:** interaktive Karte und Zeitreihen mit Standort-/Proxy-Auswahl, Quellenansicht, Unsicherheit, echten Zeitlücken, Auflösungsangabe und Szenarienvergleich. Räumliche Interpolation und rein priorgetriebene Bereiche sichtbar kennzeichnen.

## 7. Verbindliche Prüfung vor einer wissenschaftlichen Freigabe

- Moderne Kalibrierung mit räumlich getrennten Testdaten; Temperatur- und Niederschlagsfehler getrennt ausweisen.
- Für vollständig generative Modellvarianten Simulationen mit bekanntem Klima, Lücken, Altersfehlern und Proxyabhängigkeiten verwenden. Parameter-Recovery und Intervallabdeckung prüfen. Simulation-based calibration nur dort, wo Prior und Likelihood tatsächlich generativ definiert sind.
- Mehrere Ketten, Konvergenzdiagnostik, effektive Stichprobengröße und numerische Stabilität prüfen. Grenzwerte vor dem Vergleich festlegen; Konvergenz allein belegt keine wissenschaftliche Richtigkeit.
- Leave-one-archive-out und vollständig zurückgehaltene Archive bzw. Zeitabschnitte prüfen, soweit fachlich sinnvoll. Standorte mit gemeinsamem Datierungsziel sind nicht automatisch unabhängig.
- Zeigen, ob zusätzliche Proxies die Vorhersage verbessern oder nur die erwartete Referenzform erzwingen.
- Keine Auswahl der „besten“ Posterioren allein wegen Nähe zu einem Prior. Keine künstliche Unsicherheitsverkleinerung durch nachträgliches Mitteln abhängiger Verteilungen.
- Quellen, Taxonmapping, Seeds, Softwarestand, Datenprüfsummen und Konfiguration jedes freigegebenen Laufs exportieren.
- Erst nach diesen Prüfungen von einer validierten Rekonstruktion sprechen; zuvor „experimenteller Lauf“ oder „Prototyp“.

## 8. App-Betrieb und Rechenaufwand

Für die erste Version: Python-Rechenkern, reproduzierbare Batch-Läufe und eine interaktive Ansicht vorberechneter Resultate. Die Oberfläche kann bestehende Szenarien sofort umschalten. Neue MCMC-Rechnungen laufen zunächst separat; später kann ein Rechendienst mit Warteschlange und Ergebnis-Cache ergänzt werden.

LLM-Aufwand und numerischer Rechenaufwand getrennt planen. Ein leistungsfähigeres Sprachmodell beschleunigt nicht automatisch den MCMC-Kern. Zuerst Laufzeit, RAM, Speicher und effektive Stichproben pro Sekunde für einen Standort und wenige Altersensemble-Ziehungen messen. Anschließend anhand der Zielgenauigkeit hochskalieren. Viele fein gerasterte Zeitpunkte bedeuten keine entsprechend hoch aufgelösten Beobachtungen.

Es ist derzeit nicht seriös, Stundenkosten oder eine Fertigstellungsdauer für ganz Europa zu versprechen: Kalibrierungsaufwand, nutzbare Langarchive und Laufzeit der erweiterten Likelihood sind noch offen. Die Arbeitspakete A–D erzeugen die Informationen für eine belastbare Aufwandsschätzung.

## 9. Rollenverteilung mit Claude

OpenAI unterstützt die Quellenrecherche, liest Primärpublikationen und liefert überprüfbare Datennachweise, offene Lücken sowie Reviews wissenschaftlicher Annahmen. Claude übernimmt im GitHub-Projekt Importer, Modellimplementierung, Oberfläche und Tests auf Grundlage kleiner, klarer Arbeitspakete. Fachliche Entscheidungen und Freigaben bleiben beim Projektverantwortlichen.

Empfohlene OpenAI-Einstellung: GPT-6.1 Sol mit High für laufende Recherche, Datenprüfung und Code-Reviews; Extra High für die Verbindung mehrerer Quellen und schwierige Modellentscheidungen. Max gezielt für besonders schwierige Einzelprobleme. Das ist eine Empfehlung für diesen Workflow, keine Garantie wissenschaftlicher Richtigkeit. Astra ist für den Projektstart nicht erforderlich.
Offizielle Modellinformationen: https://learn.chatgpt.com/docs/models und https://learn.chatgpt.com/docs/model-selection

## 10. Direkt an Claude übergebbarer Startauftrag

Arbeite auf Grundlage dieses Plans und der Dissertation. Ziel ist eine europäische Multiproxy-App für 0–150 ka. Der Autor besitzt keine verlässlichen Kopien der ursprünglichen Taxon-Transferfunktionen, Altersensembles oder vollständigen Standortzuordnungen.

1. Prüfe die im Plan genannten vorhandenen Projektdokumente gegen den tatsächlichen Code. Liste wiederverwendbare Komponenten und standortspezifische Annahmen.
2. Lege für das neue Projekt eine dokumentierte Ableitung des oben genannten Ausgangscommits zugrunde. Erhalte einen ausführbaren Kinneret-Referenzfall.
3. Implementiere zuerst das Quellenregister und Datenstatusmodell. Übernimm bestehende Downloads aus dem Datenpaket anhand der enthaltenen Prüfsummen; keine unkontrollierten Doppelimporte.
4. Baue Importer für EMPD2, EPD/Neotoma, PANGAEA und SISAL. Priorisiere moderne Kalibrierung sowie ein vollständiges langes und ein gut datiertes kürzeres Pilotarchiv.
5. Erzeuge einen Bericht über fehlende Rohzählungen, Chronologien, Taxonzuordnungen und Transferfunktionen. Lege für jeden Engpass einen öffentlichen Ersatzweg oder den Status „offen“ fest.
6. Liefere zuerst eine Pollen-Baseline mit TANN/PANN und geprüfter Altersbehandlung. Prozentwert-Daten und Count-Daten benötigen passende getrennte Beobachtungsmodelle.
7. Ergänze erst danach einen Proxy und vergleiche dessen Beitrag mit der Baseline. Behandle Korrelationsgewichtung als gesonderte experimentelle Variante.
8. Baue die vorhandene App-Ansicht nach dem ersten reproduzierbaren Durchlauf aus. Zeige Datenherkunft, Unsicherheit, Lücken, echte Auflösung und Modellstatus.
9. Dokumentiere je Arbeitspaket Änderungen, Annahmen, sinnvolle Prüfungen und offene wissenschaftliche Entscheidungen. Keine Daten oder Originalparameter erfinden.
10. Erstelle keine flächendeckende 150-ka-Standardrekonstruktion, bevor die standortbezogene Validierung und die räumliche Datenabdeckung dies rechtfertigen.

Erster überprüfbarer Lieferumfang: Quellenregister, Importprotokoll, Kalibrierungs- und Pilotdaten, Lückenbericht sowie ein reproduzierbarer Baseline-Durchlauf. Ein schönes Dashboard ohne belegte Datengrundlage erfüllt diesen Meilenstein nicht.

## 11. Vorhandene Übergabematerialien

Aus der vorangehenden Recherche stehen bereit:
- `Netzel_Analyse_Datenverzeichnis.md`: Dissertation-Auswertung mit Quellen, Fallstudien, Vergleichsdaten, Grenzen und SISAL-Kandidaten.
- `Netzel_Datenpaket.zip`: dokumentierte Originaldownloads, Prüfsummen und exportierte SISAL-Kandidaten. Es ist ein Quellenbestand, kein vollständiges Reproduktionsarchiv.
- Dissertation: https://hdl.handle.net/20.500.11811/10758

Im Datenpaket enthaltene Kandidaten müssen die hier beschriebenen Prüfungen durchlaufen. Die heute zusätzlich identifizierten EMPD-Aufbereitungen, Ohrid- und weiteren Langarchivquellen sind nicht allein durch ihre Nennung automatisch in dieses ältere Paket aufgenommen worden.

