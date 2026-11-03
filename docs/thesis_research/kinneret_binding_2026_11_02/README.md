# Kinneret: historische Eingaben und Taxonbindung

Stand 02.11.2026; Fortsetzung der [Fallmatrix](../local_case_matrix_2026_11_02/README.md),
Phase 1 auf `dissertation-reconstruction`, Ausgangscommit `756865b`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).

**Abgeschlossen:** Der ältere Kinneret-Eintrag ist an 160 Proben und zwölf
Taxonspalten der vorhandenen Pollentabelle gebunden. Alle vier Tiefenvektoren,
die beiden verglichenen Altersvektoren und der vollständige numerische
Chronologieblock stimmen exakt überein. Für die zwölf Spalten sind 1.920
Werte exakt nachgerechnet. **Der finale Dissertations-Basiseintrag ist damit
noch nicht wiederhergestellt.** Die fehlende v1.2-Datei, eine andere Taxonreihenfolge,
abweichende Plotvarianten und die erwartete Listen-Chronologie verhindern eine
stille Übernahme des älteren Kandidaten.

Es wurden nur explizite Quellen gelesen und einfache Identitätsprüfungen
berechnet. Kein R-Skript, Training, Sampler, Bacon-Lauf, TF-Zusammenführung oder
Fossilrekonstruktionslauf wurde ausgeführt. Die App und alle Modelle bleiben
unverändert. Die privaten Quellwurzeln werden wie bisher aus
`.local/my_reconst_paths.json` aufgelöst; alle Pfade unten sind relativ dazu.

## Quellen und Ergebnisse

| Beleg | Konkrete Quelle | Geprüfter Befund |
| --- | --- | --- |
| A | `original:in/core_data_all_andrea_taxa_additional_sites.rdata` | Vier Standorte; Kinneret explizit benannt in Slot 3, 160×12 unter `all.available.taxa$taxa.spectrum`. Alte Struktur mit `qualitativ_reconstruction`, flachem Altersvektor und Chronologiematrix. SHA256 `c4782878d4b19c48d1e0651f4d6828e335079622cdc6b521be2811b12b36e888`. |
| B | `additional:my_kinneret/data/in/core/pollen_counts_epd_taxanomy.csv` | 160 Datenzeilen, 60 Taxonspalten plus Metadaten; Tiefen cm. 3.701 leere Taxonzellen ausdrücklich erfasst, keine Originalzelle verändert. |
| C | `additional:my_kinneret/data/in/core/core_data.rdata` | 160×35 Taxonspektrum und 35×3 Biomzuordnung; `taxa.probs`, `biomes.appear` usw. Dieser Einzelcontainer ist ein Biomzweig und enthält nicht den gesuchten v1.2-Dissertations-Basiseintrag. |
| D | `additional:my_kinneret/data/out/age_depth/rho_depth_given_age.rdata` | `output`: 160×1772-Matrix, 160 Tiefen und 160 mittlere Alter von −60 bis 9015 cal a BP. Keine Liste der verschiedenen Altersraster. |
| E | `original:main_v1.2.R:107,143–154,433–436,569–590,656` | Lädt die fehlende `in/core_data_additional_sites_v1.2.rdata`; übernimmt das Taxonspektrum aus Slot 3 und ersetzt aus B nur die Baumreferenz. Setzt Taxonnamen auf drei Positionen; benutzt Ohrid-Pool mit dokumentiertem 283→279-Pfad und `age.reso <- 2`. |
| F | `original:src/check_ages.R:7–19,174–208`; `src/mcmc_prepare.R:11–34,49–83,1450–1595` | Fehlwert-/Tiefen-/Altersbehandlung, Listen-Auswahl, Normalisierung und spätere Namens-/TF-Zuordnung. Mehrfachzuordnung enthält Zufallsziehungen; hier nur gelesen, nicht ausgeführt. |
| G | `additional:my_kinneret/src/bacon_kinneret.R:68–100,105–146`; `my_kinneret/main.R:40,45–49` | Ein Quellzweig speichert Alterslisten samt 50-/500-Jahresrastern; ein späterer Zweig überschreibt denselben Dateinamen mit einer Matrix an mittleren Probenaltern. Der heutige Einstieg lädt Klimacode und Biomrekonstruktion. Codebefund, kein Beweis einer konkreten historischen Ausführungsfolge. |
| H | `original:src/mcmc_case_studies.R:1117–1151,1215–1245`; drei ausdrücklich genannte Kinneret-Gewichts-PDFs | Plotbeschriftungen werden nachträglich positionsbezogen geändert. Das archivierte `taxa_weights.pdf` zeigt elf Labels, `gibbs_taxa_weights.pdf` zehn je erster Tafel. Diese sind keine invertierbare Arten-/TF-Liste. |
| I | Vorhandener Nachweis `.local/my_reconst_source_audit_v3/global_taxon_crosswalk.csv` | Der abgeschlossene globale Namens-/Positionsvergleich wird wiederverwendet. Keine Raster erneut geladen oder inventarisiert. Drei exakte Groß-/Kleinschreibungszuweisungen aus E an den tatsächlich dort verwendeten Ohrid-Pool gebunden. |

Aktuelle Dateihashes, Blockhash, Plotlabels, Fehlpfade und Prüfergebnisse:
[summary.json](summary.json). Der dort gespeicherte Dateistand wurde vor und
nach der Prüfung verglichen; sämtliche expliziten Quellen unverändert.

## 160 Proben und Chronologie

[sample_crosswalk.csv](sample_crosswalk.csv) enthält je Zeile die vier
unverändert gelesenen Tiefen aus A/B/C/D und die beiden Alter aus A/D,
mit Quelle und Ergebnis über die obige Tabelle/summary.json gebunden.
Alle Tiefen stehen in exakt derselben Reihenfolge, sind eindeutig und
stimmen ohne Sortieren, Runden, Interpolieren oder Nächste-Nachbar-Suche überein.
Die Gleichheit ist nicht allein aus gleicher Zeilenzahl abgeleitet.

Zusätzlich wurde der im Metadatenleser übersprungene große numerische Block
von A gegen die aus D gelesene Matrix geprüft: gleicher XDR-Double-Hash in
R-/Fortran-Reihenfolge,
`e39d0dafe1f099bed0c2608eaf0e48fce039bd2a9cd6d6135d4597a17b24da96`.
Damit sind sämtliche 283.520 Matrixwerte identisch, nicht nur die Dimensionen.

**Verbleibende Grenze:** A und D sind eine Matrix mit flachem Altersvektor.
F erwartet bei `add.sites` dagegen `rho_depth_given_age[[age.reso]]` und
`age[[age.reso]]`; E setzt für Kinneret `age.reso=2`. Das ist im historischen
Listenvertrag das zweite Altersraster. Auf einem Matrix-/Vektorobjekt wäre
`[[2]]` ein einzelner Wert. Deshalb ist die vorhandene Datei nicht ungeprüft
als finale 50-Jahres-Chronologie verwendbar. G belegt sowohl den ursprünglichen
Listen-Schreiber als auch das spätere Überschreiben desselben Dateinamens.
Die fehlende vorgeladene Basisdatei könnte die richtige Listenfassung enthalten;
das ist ohne ihren tatsächlichen Inhalt offen.

Die Tabellenwerte sind Quellenmetadaten, keine neu geschätzten Alter. Die
Bacon-Kette, Datierungskalibrierung und Chronologie wurden nicht erneut berechnet.
Der [Monticchio-Gate](../HANDOFF.md) bleibt fallbezogen erhalten.

## Zwölf ältere Taxonspalten zahlenidentisch zugeordnet

[older_taxon_crosswalk.csv](older_taxon_crosswalk.csv) hält alte Spaltenposition,
alten Namen, CSV-Spalte und Pollentyp samt allen 160 Vergleichen fest.
Die mathematisch exakt passende Regel lautet:

`gespeicherter Anteil = CSV-Zählwert / Zeilensumme aller 60 Taxonspalten`.

Leere CSV-Zellen wurden **nur in dieser Identitätsprüfung** mit null angesetzt;
die Zahl der leeren Zellen bleibt dokumentiert. Alle zwölf zugeordneten Spalten
stimmen damit ohne Toleranz überein. Das beweist die Zahlenbeziehung dieses
älteren gespeicherten Stands, keine Ausführung eines bislang fehlenden
Erzeugungsskripts und keine fachlich bestätigte Typ→Art-Zuweisung.

| Alte Position | CSV-Pollentyp | Gespeicherter älterer Taxonname |
| --- | --- | --- |
| 1 | Olea europaea | Olea europaea |
| 2 | Quercus ilex-type | Quercus evergreen |
| 3 | Quercus | Quercus cerris |
| 4 | Phillyrea | Phillyrea |
| 5 | Pinus | Pinus cembra |
| 6 | Betula | Betula |
| 7 | Juglans regia | Juglans sp. |
| 8 | Vitis | Vitis sylvestris |
| 9 | Fagus | Fagus orientalis |
| 10 | Alnus | Alnus |
| 11 | Tilia | Tilia |
| 12 | Chenopodiaceae/Amaranthaceae | Chenopodium glaucum |

Die gedruckten Taxonnamen oben dokumentieren den gespeicherten alten Stand;
sie sind keine neuen botanischen Zuordnungsentscheidungen. Insbesondere
Quercus→Q. cerris und Pinus→P. cembra werden nicht in den finalen Fall übernommen.

**Wichtiger Nennerunterschied:** AP+NAP ist nicht die hier passende Normierung.
Schon für Olea ergibt diese Alternative eine maximale absolute Abweichung
von 0,06666096305485447 auf der Anteilsskala. Auch die ältere
`qualitativ_reconstruction` ist nicht identisch mit Bs `tree_percentage`.
E ersetzt im jüngeren Einstieg die qualitative Referenz ausdrücklich aus B.
Diese Unterschiede werden erhalten; keine Vereinheitlichung von Pollen- und
Referenznennern vorgenommen.

## Warum der ältere Stand nicht die finale Basistabelle ersetzt

Der konkrete jüngere Code E erwartet andere Namen auf denselben Positionen:

| Position vor späterer Auswahl | Name im älteren A | Aktive Zuweisung in E | Gesicherter Befund |
| --- | --- | --- | --- |
| 5 | Pinus cembra, Zahlen aus CSV-Pinus | Olea Europaea, Z.148 | Ungeprüfte Übernahme würde Pinus-Zahlen mit Olea beschriften. Vorherige Z.147 `Atriplex Halimus` wird direkt überschrieben. |
| 6 | Betula, Zahlen aus CSV-Betula | Phillyrea Latifolia, Z.149 | Kein gültiger Crosswalk durch gleiche Positionsnummer. |
| 8 | Vitis sylvestris, Zahlen aus CSV-Vitis | Pistacia Lentiscus, Z.150 | Die fehlende finale Grundreihenfolge ist entscheidend. |

Darüber hinaus unterscheidet sich die Referenzstruktur (`qualitativ_reconstruction`
in A gegenüber `qualitativ_data$qualitativ_reconstruction` in E) und das
Altersinterface. Die vier Standorte in A sind Iznik/Birkat Ram/Kinneret/En Gedi;
die jüngere Basis setzt Prespa/En Gedi/Kinneret/Birkat Ram/Iznik/Ohrid voraus.
Dass Kinneret zufällig in beiden Fassungen Slot 3 hat, beweist keine Laufidentität.

Auch C löst das Problem nicht: sein 35-Taxa-/3-Biom-Objekt hat eine andere
Aufgabe. Der berechnete Paper-Biomfall und die sieben modernen Artenmodelle
werden weiterhin nicht als taxonbasierte Dissertationsreproduktion ausgegeben.

## Was die Gewichtsplots belegen und was offenbleibt

Gezielt gelesen wurden unter `original:out/plots/case_studies/kinneret/`:

- `taxa_weights.pdf`: elf Labels; byteidentisch zum Archivmitglied
  `figures/kinneret/taxa_weights.pdf` in `diss_final.zip`.
- `gibbs_taxa_weights.pdf`: zehn Labels der ersten Tafel; Olea/Phillyrea als
  Gattungsbeschriftungen, keine entsprechende Chenopodiaceae-Zeile.
- `rw_diff_taxa_weights.pdf`: elf Taxonlabels, darunter `Atriplex Halimus`;
  die X-Achsenbeschriftung wurde ausdrücklich nicht als Taxon mitgezählt.

H ersetzt für Kinneret Label 1 durch `Ericales (tetrad)`, 2 durch `Juglans regia`,
5 durch `Pinus`, 6 durch `Pistacia`, 9 durch `Chenopodiaceae`.
Das sind Anzeigeänderungen. Insbesondere Atriplex/Chenopodiaceae können
unterschiedliche Beschriftungen desselben Positionsplatzes sein; die Labels
allein beweisen keine verschiedenen numerischen Funktionen. Umgekehrt ist die
elf- bzw. zehnteilige Anzeige nicht mit der alten Zwölferspaltenliste gleichzusetzen.
Die gespeicherte Textreihenfolge ist kein gesicherter ursprünglicher TF-Index.

## Drei konkrete TF-Abhängigkeiten, vollständige Teilmenge weiter offen

[explicit_tf_requirements.csv](explicit_tf_requirements.csv) bindet ausschließlich
die drei expliziten Zuweisungen aus E an den dort tatsächlich geladenen
Ohrid-Namenspool. Hier wird nicht der Ohrid-Löschpfad auf die Kinneret-
Verbreitungsdatei übertragen.

| Exakter Verbrauchername | Poolposition nach dem belegten Ohrid-Pfad | Separater kleingeschriebener Name / Position |
| --- | --- | --- |
| Olea Europaea | 142 | Olea europaea / 141 |
| Phillyrea Latifolia | 149 | Phillyrea latifolia / 148 |
| Pistacia Lentiscus | 162 | Pistacia lentiscus / 161 |

Diese benachbarten Namen werden nicht gleichgesetzt. Für jede wirklich
benötigte Funktion fehlen weiterhin `prod.x1`, `prod.x2` und die geprüften
`x1range.bck`/`x2range.bck` im historischen
`mixed_partial_arts_emd.rdata` samt Info-/Taxonreihenfolge. Die Poolpositionen
sind Namens-/Verbraucherbindungen, keine Belege vorhandener numerischer TFs.
**Drei explizite Abhängigkeiten sind keine vollständige Dreitaxon-Rekonstruktion.**
Eine endgültige Gesamtteilmenge samt Mehrfachzuordnungen und Ausschlüssen lässt
sich erst aus dem finalen Basiseintrag und seiner tatsächlichen Auswahl ableiten.

## Konkrete Lücke und nächster Einstieg

**Benötigt wird jetzt:** die im Dissertationscode aktiv geladene
`in/core_data_additional_sites_v1.2.rdata`, insbesondere `core.data[[3]]`,
oder das Skript mit den Eingaben, das exakt diesen Stand erzeugt hat.
Auch Gs gesonderter Verbraucherpfad `in/core_data_kinneret.rdata` fehlt im
geprüften Originalordner. Für den finalen Basiseintrag werden mindestens
geordnete Taxonnamen/-spektren, Proben-/Tiefenliste, Referenzstruktur und
Chronologielisten mit ihren Altersrastern benötigt.

Der Nutzer hat am 02.11.2026 bestätigt, dass ihm kein weiterer Speicherort
bekannt ist. Das belegt keine vollständige Abwesenheit auf allen Datenträgern;
die Quelle bleibt im geprüften Bestand offen. Die bisherigen Verzeichnisse
und abgeschlossenen negativen Dateinamens-/Archivprüfungen werden nicht erneut
pauschal durchsucht. Damit bleibt
**die finale Eingabe-/TF-Bindung offen**; das abgeschlossene Teilpaket ist eine
verifizierte Quellenbindung, keine Lauf- oder Ersatzmodellfreigabe.

**Nächster unabhängig möglicher Arbeitsschritt:** den Kinneret-spezifischen
Referenz-/Ankervertrag an den bereits benannten Verbrauchern festhalten:
AP-Maske bis 6500 a BP, Auswahl und Vorverarbeitung von Soreq/MEDSTACK,
Csa-Anker und deren tatsächlich konsumierte CRU-/Isotopendateien. Die fehlende
Basisdatei darf dabei nicht als wiederhergestellt gelten. Keine neue Suche
ohne einen konkreten neuen Pfad und keine fachliche Ersatzmodellentscheidung.

Nach Erhalt einer passenden Basisquelle: zuerst deren Kinneret-Eintrag gegen die
160 Proben prüfen, Taxon-/Chronologiereihenfolge samt Ausschlüssen binden,
dann die vollständige historische TF-Anforderung formulieren. Danach folgen
wie geplant Referenz-/CRU-/Ankerbindung und historische Nachbearbeitung.
Keine neue Kalibrierung, Reliabilitätsansicht oder methodische Korrektur vorziehen.

[collect_evidence.py](collect_evidence.py) erzeugt die Nachweise ausschließlich
lesend aus den expliziten Quellen; Aufruf im bestehenden Checkout:

```powershell
.venv/Scripts/python.exe docs/thesis_research/kinneret_binding_2026_11_02/collect_evidence.py
```

[verification.json](verification.json) dokumentiert Vergleichs- und
Erhaltungsprüfungen. Rohzählwerte, numerische TFs und Originalcontainer bleiben
lokal; exportiert sind die abgeleitete Proben-/Spaltenbindung und Quellenmetadaten.
Plots bleiben jung links, alt rechts; Monticchio-/Chronologie-Gates, Baseline,
Referenzdaten und Originale sind unverändert erhalten.
