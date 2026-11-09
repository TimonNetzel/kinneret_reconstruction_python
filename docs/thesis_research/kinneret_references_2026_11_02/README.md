# Kinneret: Referenz- und Klimaankerpfad

Stand 02.11.2026; Phase 1, Ausgangscommit `44092d3`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).
Fortsetzung der [Fallmatrix](../local_case_matrix_2026_11_02/README.md) und
[Proben-/Taxonbindung](../kinneret_binding_2026_11_02/README.md).

**Abgeschlossen:** Der textlich beschriebene Referenz-/Ankervertrag und die
konkreten Verbraucher sind mit getrennten Belegen dokumentiert. Die 160
AP-Prozentwerte sind nachgerechnet, zwölf explizite Ladepfade geprüft und
mehrere Quellkonflikte eingegrenzt. **Die numerischen Referenz-/Ankereingaben
und die Zuordnung zum finalen Dissertationslauf bleiben offen.** Insbesondere
sind die heute gespeicherten R-Skripte keine nachgewiesene ausführbare
Konfiguration der Kinneret-Abbildung.

Originale, App/CLI, Baseline und Modelle sind unverändert. Keine R-Ausführung,
neue Kalibrierung, Zufallsziehungen, Waveletberechnung, Trainings, Sampler,
Altersmodelle oder Fossilrekonstruktionen. Alle neuen Rechnungen sind
Prozentidentitäten, Metadaten- und Kontrollflussprüfungen; keine Klimakurven.

## Quellen und Nachweise

Die privaten Quellwurzeln `original`/`additional` werden über die vorhandene
`.local/my_reconst_paths.json` aufgelöst. Keine neue Verzeichnisinventur.

| Quelle | Geprüfte Rolle |
| --- | --- |
| `original:out/plots/mcmc/v5.0/diss_final.zip`, Mitglieder `thesis_kinneret.tex:28–43,140–146` und `thesis_dead_sea1.tex:150–160` | Lokaler textlicher Referenzvertrag: drei Referenzfamilien, anthropogenes Fenster, rezenter Klimaanker. Archiv und Mitglieder separat gehasht. Zugehörige lokale PDF-Fassung: Kinneret S.90–92 / PDF96–98; Holozän-/Levante-Regel S.74 / PDF80. Fassungen weiter getrennt wie in der Fallmatrix. |
| `original:main_v1.2.R:143–154,325–350,388–401,654–675` | AP-Quelle, anthropogenes Fenster, vier Klimadateien, Reihenfolge der Isotopen-Ladevorgänge und Hauptschalter. |
| `original:src/mcmc_prepare.R:635–730,751–907,1014–1046,1138–1204,1247–1304,1729–1735,1776–1818,2057` | Referenzauswahl, Masken, Überschreibungen, Zeitabgleich, Ankerparameter und aktiver Stopp. |
| `original:src/check_ages.R:219–231`; `src/prior_climate.R:6–130` | Zeitfenster mit Alter ≤0, Klassen-/Rasterauswahl und alternative Ankerzweige. |
| `original:src/mcmc_functions_v1.6.R:1843–1855,1935–2027,2317–2329`; `src/mcmc_conduction.R:153–196` | Tatsächliche Verwendung der Masken/Referenzen, Korrelationen, Ankerdichten und Vorverarbeitung. |
| `original:src/mcmc_prep_v1.2.R:1–30,764–832,980–990` | Gezielte ältere Vergleichsfassung: anderer Kinneret-Höhlenindex, keine heutige 220,5-ka-Testüberschreibung. Keine automatische Wahl dieser Variante. |
| `additional:my_kinneret/data/in/core/pollen_counts_epd_taxanomy.csv` | Die bereits gebundene 160-zeilige AP/NAP-/Prozentreferenz. |
| `additional:dead_sea_reconstruction_R/data/in/speleo_data.rdata` | Ein aus einem konkreten kommentierten Schreiberpfad abgeleiteter gespeicherter Kandidat; späterer anderer Fall, kein Kinneret-Ersatz. |
| [Vorhandene Klimazeitgrenzen](../research_snapshot_2026_11_02/historical_provenance/climate_time_bounds.json) und [Quellenverzeichnis](../Netzel_Analyse_Datenverzeichnis.md) | Bereits geprüfte regionale Klimafassungen und öffentliche Referenzfamilien. Keine Wiederholung der früheren Gesamt-/Modellprüfungen. |

[contract.json](contract.json) enthält die strukturierten Befunde und Hashes,
[source_spans.json](source_spans.json) die gezielten, mit Zeilennummern gebundenen
Quellauszüge. Diese sind Belege, keine auszuführende Konfiguration.

## Vertrag je Kanal

| Kanal | Dissertationstext | Erhaltener Verbraucher | Status |
| --- | --- | --- | --- |
| AP/NAP | Referenz für PANN nur vor 6,5 cal ka BP; jüngerer anthropogener Bereich ausgeschlossen | CSV-`tree_percentage`, Tiefen-Alters-Matrixprodukt; `!inside.range(pollen.age,c(-60,6500))` wird direkt in der Korrelation benutzt | **belegt** für Text und statischen Code; numerische altersbezogene Kurve wegen fehlender finaler Chronologieliste weiterhin **offen** |
| Soreq | WPS-Referenz für PANN über den Kinneret-Fall; Bildunterschrift nennt Bar-Matthews et al. 2003 | Erst räumliche Auswahl/Pareto, dann Kinneret-Index 105 nur im Mehrkandidatenzweig; später unconditional Index 170 innerhalb des aktivierten Speläothemzweigs und Überschreiben von `site.age` | **widersprüchliche Quellstände**; tatsächliche Reihenidentität, Spektraleingabe und historische Ausführung **fehlen** |
| MEDSTACK | Temperaturbezogene Referenz, Colleoni et al. 2012 | Csa gehört zu Kategorie 1; Auswahl aus `d18o.age$splined.series[[length(d18o.age$loc_nr)-2]]`, Vorzeichenwechsel, Zeitabgleich über gemeinsame Alter | Auswahlregel **belegt**, Identität dieses letzten Containerabschnitts als konkrete MEDSTACK-Datei/-Fassung **fehlt** |
| Rezenter Klimaanker | Csa-Klassifikation am jüngsten Horizont; CRU4.01/1961–1990 als textlicher Klimakontext | Bei mindestens einem Alter ≤0 lokale Klimazeitreihe anstatt Klassenverteilung; sonst Klassenfilter der globalen Klimamittel. Mindest-SD 2 °C / 150 mm/a; Normalprior für TANN und Gammaprior für PANN | Zweige **belegt**, nicht automatisch mit dem textlichen Csa-Anker gleichzusetzen; exakte Zahlen und globale Klima-/Zeitachsen **fehlen** |

TANN bleibt Jahresmitteltemperatur in °C, PANN Jahresniederschlag in mm/a.
Pollenanteile stehen hier in %, Isotope in den Textbelegen als δ18O in ‰ VPDB;
Korrelationen und deren Quadrate sind dimensionslos. Keine direkte Umrechnung
von Isotopenwerten in °C oder mm. Keine Übernahme der TDJF-Paperanker oder
moderner WorldClim-Ersatzdaten.

## AP: Quelle, Maske und tatsächlicher Verbraucher

Die 160 CSV-Werte erfüllen `tree_percentage = 100*AP/(AP+NAP)` mit maximal
`6,394884621840902e-14` Prozentpunkten Rundungsabweichung. Dieser Nenner gehört
zur Referenz; er ersetzt nicht die im älteren Taxonspektrum nachgewiesene
Normierung über die Summe aller 60 Taxonspalten.

`main_v1.2.R:144` übernimmt die nicht fehlenden Prozentwerte in den geladenen
Kinneret-Basiseintrag. `mcmc_prepare.R:639–663` transformiert die Tiefenreferenz
mit `rho.d.g.a`, sofern der entsprechende Kernreferenzzweig erreicht wird.
Dessen Aktivierung hängt auch von `inside.giesecke` ab; sie ist ohne die
ursprünglichen geladenen Daten nicht allein aus dem Standortnamen bewiesen.

Die anthropogene Alterszone wird als `c(-60,6500)` gesetzt. Im Hauptpfad bleiben
`hi_ages_time = !inside.range(...)` und `hi_time` zwei verschiedene Größen:
`add.sites` setzt `hi_time <- FALSE`, löscht aber die Altersmaske nicht.
Die letzte `reconstruction`-Definition benutzt die Maske weiterhin direkt in
`post_pollen`, sowohl im Mehrreferenz- als auch im Einzelreferenzzweig.
`hi_time=FALSE` deaktiviert nur den zusätzlichen zeitlichen
`post_iso_hi`-Kanal; dort wird stattdessen 0,95 gesetzt.

Für Kinneret setzt die Vorbereitung `tann <- FALSE`: AP und Speläothem-WPS
werden gegen PANN ausgewertet. Die MEDSTACK-Zeitreferenz bleibt im Code
explizit mit TANN verknüpft. Negative oder fehlende Korrelationen werden im
erhaltenen Zweig durch `1e-10` ersetzt und anschließend quadriert. Das bleibt
historischer Implementierungsbefund, keine neue Bewertungsentscheidung.

**Randwert offen:** Die genaue Ein-/Ausschlusskonvention von `inside.range`
bei exakt 6500 a BP benötigt die historische Paket-/Funktionsbindung. Der
Bericht bewahrt daher den Originalausdruck und setzt keine eigene Grenzregel.
Bei `case_studies=TRUE` wird die anthropogene Zone vorher auf NULL gesetzt;
zusätzlich werden bei Kinneret die beiden Isotopenquellen deaktiviert. Dieser
Vergleichspfad ist nicht die Konfiguration des beschriebenen Hauptfalls.

## Soreq: Auswahl, Testüberschreibungen und gespeicherter Kandidat

Die Vorbereitung prüft die ersten 170 Koordinateneinträge von `d18o.age` auf
Entfernung ≤150 km. Im Mehrkandidatenzweig folgen zeitliche Abdeckung,
mittlere Auflösung und Entfernung als Pareto-Kriterien; Abdeckung ≤0,5 kann
`sources$speleothem` deaktivieren. Die Kinneret-Zuweisung `which.site=105`
liegt **nur in diesem Mehrkandidatenzweig**. Bei einem einzigen Kandidaten wird
stattdessen dessen Index verwendet. Die Override-Zuweisung aktualisiert weder
die zuvor berechnete Abdeckung noch beweist sie einen konkreten Publikationsnamen.

Sobald der Speläothemzweig aktiv ist, überschreibt der gespeicherte Code:

- `site.age <- seq(0,220500,by=500)` und entfernt die R-Indizes 6:29.
- `iso.ref <- d18o.age$splined.series[[170]]`, unabhängig vom vorher gewählten Index.

Das sind 418 Alterswerte mit einer Lücke zwischen den behaltenen Werten
2000 und 14500 a BP. Die vorhandene Kinneret-Chronologie wird dadurch nicht
als 0–9-ka-Reihe weitergeführt; `age` bleibt daneben ein anderes Objekt.
Spätere Mitgliedschaftsabgleiche benutzen das veränderte `site.age`.
Dieser Quellstand ist daher nicht allein durch Entfernen fehlender Ladepfade
als belegter Kinneret-Dissertationslauf zu behandeln. Der aktive Stopp in
`mcmc_prepare.R:2057` ist eine weitere Grenze; kein R-Skript wurde ausgeführt.

Weitere statisch gebundene Schritte: Auswahl des Referenzfensters und der
Schrittweite, Standardisierung, Achtpunkt-Gleitfilter über `fftfilt`,
Rücktransformation und Rücksetzen der ersten acht Isotopenwerte. Morlet-/WPS-
Parameter und Cone-of-Influence-Maske werden erzeugt; für Kinneret wird keine
anthropogene Teilung des WPS vorgenommen (`hi_spec=FALSE`). Hier wurden weder
Filter noch WPS erneut berechnet.

Die gezielt verglichene ältere `mcmc_prep_v1.2.R` setzt für Kinneret dagegen
Index 37 und enthält die 220,5-ka-Überschreibung nicht. Kommentare können
Höhlennamen andeuten, ersetzen aber nicht die fehlenden `d18o.age`-Metadaten.
**37, 105 und 170 bleiben getrennte positionsgebundene Belege.** Keine der
Fassungen wird ohne Eingabe-/Ausgabebindung zur richtigen historischen Version erklärt.

Der erhaltene spätere Container
`dead_sea_reconstruction_R/data/in/speleo_data.rdata` enthält tatsächlich
`which_site=170`, 368 Isotopen-/Alterszeilen von 0 bis 183500 in 500er-Schritten,
eine 368×55-COI-Maske und 1024×55-Waveletbasis. Der konkrete kommentierte
Schreiberpfad in `mcmc_prepare.R` machte ihn zu einem gezielten Prüfobjekt.
Er belegt einen gespeicherten anderen Zweig, keine Identität der Kinneret-
Referenz. Insbesondere wurde aus diesem Fund kein Ersatz-Soreq-Eingang erzeugt.

## MEDSTACK: Kategorie und Reihenidentität auseinanderhalten

Die Csa-Kategorie ist in `med.kgz` enthalten und führt, unter dieser
Klassifikation, zu `which.site=1`. Der Verbraucher liest die Reihe über die
**Länge von `d18o.age$loc_nr`**, nicht über einen festen MEDSTACK-Namen oder
einen verifizierten Dateinamen. Vor der Korrelation wird das Isotopenvorzeichen
für Kategorien 1 und 2 gewechselt. Gemeinsame Alter werden über `%in%`
ausgewählt; die vorbereiteten `splined.series` müssen deshalb bereits passende
Zeitachsen und Vorverarbeitung enthalten. Der Verbraucher beweist nicht,
welcher externe Rohdatensatz diese Werte erzeugt hat.

Der bereits verzeichnete öffentliche MEDSTACK-Kandidat PANGAEA.790006 sowie
die öffentlichen Soreq-Familien von 2003/Grant2012 bleiben Quellenkandidaten
laut [bestehendem Downloadmanifest](../download_manifest.json). Sie werden
nicht mit den fehlenden historischen Workspaces gleichgesetzt oder neu geladen.

## Klimaanker: tatsächlich konsumierte Variablen und Verzweigung

`prior_climate.R` benötigt `pre`, `tmp`, `lon`, `lat` aus den globalen Mitteln
sowie `stn` für die Maske im noch geöffneten Niederschlags-NetCDF. Die
Klassenkarte stammt in diesem Zweig aus `climatezones`/`LookupCZ` des
R-Pakets `kgc`. Dass daneben eine `kgc_distribution_ww.rdata` vorhanden ist,
beweist keine identische Paketkarte, Klassifikation oder Ankerverteilung.

`check_ages.R:223` bestimmt die rezenten Zeitpunkte über `site.age <= 0`.
Sind solche Punkte vorhanden, werden lokale Werte aus
`cru4_tann_v1.2.nc`/`cru4_pann_v1.2.nc`, Zeitindizes 50:115, verwendet.
Die Lokalindizes stammen aus dem festgelegten 2,5°-Feld; bei fehlender
Temperaturzeitreihe gibt es eine Suche in den acht Nachbarpositionen und der
Mittelposition. Die tatsächlichen Koordinatenarrays und diese Indexbindung
sind ohne die Dateien nicht prüfbar. Der bekannte ältere Kinneret-Kandidat
hat zwei Alter ≤0 (−60 und −23); das schließt nicht den finalen Listenvertrag.

Fehlen solche rezenten Punkte, werden die globalen Klimamittel nach Klasse
und regionalem Ausschnitt gewählt. Beide Zweige schätzen Normalparameter;
anschließend gelten SD-Untergrenzen von 2 °C bzw. 150 mm/a. Die Vorbereitung
leitet Gamma-Shape/Rate aus den Niederschlagsparametern ab und setzt bei
rezentem Start `rate=shape/mean.pann`. Der eigentliche Sampler verwendet
`dnorm` für TANN und `dgamma` für PANN. **Es wurden keine Parameter neu angepasst.**
Das spätere Flag `recent` prüft Alter ≤11000 und ist eine andere Bedingung
als der Ankerquellen-Zweig bei Alter ≤0.

Der Kommentar „1950–2015“ zu 50:115 beweist ohne Zeitachse keinen Kalenderbezug.
Die Dissertation nennt CRU4.01/1961–1990; die bereits gelesenen regionalen
TS4.00-Fassungen belegen dagegen Mittel 1901–2015 bzw. 1961–1990. Diese werden
nicht unter anderem Namen als fehlende globale Ankereingaben eingesetzt.
Das Standortdiagramm 18,5 °C / 472,1 mm/a ist ebenfalls kein Nachweis der
verwendeten Prior-Mittelwerte und Standardabweichungen.

## Verfügbarkeit, Grenzen und nächster Einstieg

[dependencies.csv](dependencies.csv) enthält zwölf exakt benannte Ladepfade mit
Quellzeile und Rolle. Alle zwölf fehlen am jeweiligen, aus dem vorhandenen
Originalstand aufgelösten Verbraucherpfad. Dies ist keine Behauptung einer
vollständigen Laufwerkssuche. Frühere negative Bestands-/Archivprüfungen wurden
weiterverwendet. Verfügbare Rohquellenfamilien oder spätere Fallcontainer gelten
weiterhin nicht als identische ursprüngliche Workspaces.

Für einen numerisch geschlossenen Vertrag fehlen insbesondere:

1. Finaler Kinneret-Basiseintrag, Listenchronologie und historische numerische TFs.
2. Die geladenen Isotopen-Workspaces samt `d18o.age`-Namens-/Positionsfolge,
   Koordinaten, Roh-/Spline-Altersachsen und Vorverarbeitungsfassung. Erst damit
   lassen sich 37/105/170 und der MEDSTACK-Endabschnitt tatsächlich identifizieren.
3. Vier originale globale CRU-Eingaben mit Metadaten/Zeitachsen und die
   historische `kgc`-/Funktions-/Paketbindung; konkrete Ankerparameter.
4. Eine zur archivgebundenen Kinneret-Ausgabe gehörige, von Testüberschreibungen
   getrennt belegte Ausführungsfassung. Kein stilles Entfernen oder Reparieren.

Der Nutzer kennt keinen weiteren Speicherort der finalen Basisdatei; diese
Frage wird nicht erneut gestellt. **Als nächster unabhängiger Schritt** lässt
sich die historische Nachbearbeitung gezielt vergleichen: tatsächliche Auswahl
von Ziehungen, Prior-/IAT-/Pareto-Schritte und Dichtekombination in
`mcmc_case_studies.R` gegenüber dem Dissertationstext und dem aktuellen
Trimming-Einstieg. Nur die bereits bekannten Fassungen und gespeicherten
Nachweise verfolgen, keine neuen Sampler-/Rekonstruktionsläufe.

[collect_evidence.py](collect_evidence.py) liest die ausdrücklich angegebenen
Quellen und wiederholt die Metadaten-/Prozentprüfungen. Aufruf im vorhandenen Checkout:

```powershell
.venv/Scripts/python.exe docs/thesis_research/kinneret_references_2026_11_02/collect_evidence.py
```

Die Kontrollflusstabelle in contract.json setzt nur benannte Schalterwerte ein;
sie ist kein R-Lauf und enthält keine berechneten Klima- oder Isotopenreihen.
[verification.json](verification.json) dokumentiert Wiederholung und Erhaltung.
Originalbestände, beide vorherigen Quellenpakete, 50 Forschungsnachweise,
Baseline und Monticchio-/Chronologie-Gates bleiben erhalten. Plots jung links,
alt rechts; Phase 2 bleibt nachgeordnet.
