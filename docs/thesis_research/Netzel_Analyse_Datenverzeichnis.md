# Netzel 2022/2023: Analyse und Datenverzeichnis

Stand: 29. Oktober 2026. Die Angaben zu Seiten beziehen sich auf die gedruckte Seitenzählung der Dissertation. In der heruntergeladenen PDF liegt die gedruckte Seite 1 auf PDF-Seite 11.

## 1. Identifizierte Arbeit und Umfang der Prüfung

**Timon Netzel: Quantitative paleoclimate reconstructions in the European region based on multiple proxies.** Dissertation, Rheinische Friedrich-Wilhelms-Universität Bonn. Eingereicht September 2022, Promotion am 14.12.2022, veröffentlicht 2023 als Bonner Meteorologische Abhandlungen, Heft 95, vi + 179 Seiten.

- Hochschulnachweis: https://hdl.handle.net/20.500.11811/10758
- Original-PDF: https://bonndoc.ulb.uni-bonn.de/xmlui/bitstream/handle/20.500.11811/10758/6986.pdf?sequence=2&isAllowed=y
- URN: urn:nbn:de:hbz:5-69864

Der vollständige PDF-Text wurde heruntergeladen und durchsucht. Für dieses Verzeichnis wurden insbesondere die Proxy- und Methodenkapitel 2.1–2.6, die Datengrundlagen und Fallstudien in Kapitel 3, Kapitel 4, das Literaturverzeichnis und die Anhänge A/B ausgewertet. Die Standortkarte Abb. 3.3 und die Modelltabelle 2.3 wurden zusätzlich visuell geprüft.

**Ergebnis:** Die wesentlichen Datenquellen und die ausdrücklich beschriebenen Fallstudien sind identifiziert. Ein lückenloser, datei- und probenidentischer Nachbau aller ursprünglichen Läufe ist mit den gefundenen öffentlichen Materialien noch nicht gesichert. Es fehlen insbesondere eine vollständige Liste der tatsächlich verwendeten EPD- und SISAL-IDs, die Zuordnung Höhle–Pollenprofil, eindeutige Datenbankstände und einige persönlich übermittelte Zusatzdaten. Das beigefügte Datenpaket ist daher ein dokumentierter Quellenbestand und Erweiterungspool, kein vollständiges Reproduktionsarchiv der Dissertation.

## 2. Was das Modell mit den Daten macht

Die Dissertation rekonstruiert vor allem **Jahresmitteltemperatur TANN** und **Jahresniederschlag PANN**. Die später veröffentlichte Kinneret-Arbeit von 2025 rekonstruiert dagegen Wintertemperatur und Jahresniederschlag; ihre Eingabedateien dürfen deshalb nicht unverändert mit denen der Dissertation gleichgesetzt werden.

1. Pollenprofile und moderne Pflanzenverbreitungen bilden zusammen mit modernem Klima die Grundlage probabilistischer Transferfunktionen. Der ML-Vergleich umfasst 279 Pflanzentaxa (S. 49).
2. Datierungen und ihre Fehler werden in Alters-Tiefen- bzw. Alters-Distanz-Beziehungen berücksichtigt. Zentral ist die Transformation P(Tiefe | Alter), nicht nur eine einzelne mittlere Alterskurve (S. 24–29).
3. Zusätzliche Pflanzen- und Isotopenkurven steuern die Rekonstruktion über Übereinstimmung im Zeitbereich und/oder in Wavelet-Leistungsspektren. Diese Referenzen beeinflussen unter anderem die posterioren Taxagewichte (S. 29–38).
4. Moderne Klimaverteilungen und ausgewählte zusätzliche Klimaanker begrenzen absolute Temperatur- und Niederschlagswerte. Ein Isotopenwert wird dabei nicht allgemein direkt in °C oder mm umgerechnet.
5. Für räumliche Klimafelder werden lokale Rekonstruktionen mit einem CMIP5/PMIP3-Ensemble kombiniert (S. 45–48, 123–138).

Die Arbeit beschreibt ungefähr 600 Pollenprofile bzw. mehr als 600 MCMC-Simulationen und schließlich **186 akzeptierte lokale Rekonstruktionen**. Das sind keine 600 oder 186 voneinander unabhängigen Isotopenreihen. Für die räumlichen Beispiele werden 20 lokale Rekonstruktionen für HIST, 152 für 6 ka und 15 für 21 ka verwendet (S. 123, 129, 134). Überlappungen zwischen diesen Gruppen sind möglich.

## 3. Datenquellen, die tatsächlich in die Rekonstruktion eingehen

„Verwendet“ bezeichnet hier die im Methodentext oder in Ergebnisschritten nachgewiesene Verwendung. Ein gefundenes öffentliches Archiv ist nicht automatisch die exakt gleiche Datei, die Netzel ursprünglich eingelesen hat.

| ID | Datenquelle | Verwendung und Fundstelle | Bezug und Einschränkung |
|---|---|---|---|
| U01 | European Pollen Database, EPD | Pollen aus See- und Moorsedimenten; regionale Grundlage, S. 21–22 und 52–53 | Heute über Neotoma: https://www.neotomadb.org/data und https://epdweblog.org/browse-get-data/ . Die damalige vollständige Profilauswahl ist nicht veröffentlicht gefunden. |
| U02 | Giesecke et al. 2013: Alterskontrollpunkte und Chronologien | Grundlage neuer Bacon-Modelle der EPD-Profile, S. 21–25 | https://doi.org/10.1594/PANGAEA.804597 . Untertabellen: Control_P 804606, Entity 804591, EpdAgeAll 804595, EpdAgeCut 804596. Der öffentliche Datensatz heißt „823 EPD sites“; die Dissertation nennt im Zusammenhang mit der Literatur 1036 Standorte. Diese Zahlen und Teilmengen dürfen nicht gleichgesetzt werden. |
| U03 | Moderne Pflanzenareale, darunter Meusel/Weinert/Jäger | Präsenz/Absenz und Klima-Transferfunktionen; Beispiel Olea europaea, S. 38–40, 49 | Die Dissertation verweist für das Beispiel auf Meusel et al. 1974. Ein vollständiger Download der tatsächlich verwendeten 279 Arealraster wurde nicht gefunden. Heutige digitalisierte Meusel-Geodaten: https://www.floraweb.de/inhalt/datenquellen.html . Diese sind eine mögliche Ersatz-/Erweiterungsquelle, kein belegter identischer Input. |
| U04 | CRU TS 4.01, Harris & Jones 2017 | Moderne Klimatologie 1961–1990, Transferfunktionen, Klimaanker, Klimadiagramme und Prüfung räumlicher Ergebnisse, S. 35, 38–39, 57–58, 70–71, 123–128 | https://doi.org/10.5285/58a8802721c94c66ae45c3baa4d814d0 . Gesamtdaten 1901–2016, 0,5°-Raster. Benötigt werden insbesondere Temperatur und Niederschlag. |
| U05 | Aus modernen Daten abgeleitete Köppen-Geiger-Klimaklassen | Auswahl von Klimaverteilungen für Einzugsgebiete und Klimaanker; S. 35, 57, 70–72, 112 | Abgeleitete Information, kein zusätzlicher unabhängiger Proxy. Chen & Chen 2013 erläutern die Klassifikation; die in der Dissertation gezeigten Klassen beziehen sich auf die gewählte CRU-Referenzperiode. |
| U06 | MEDSTACK, östliches Mittelmeer, planktonisches δ18O | Temperaturbezogene Referenzkurve; u. a. S. 52, 58–59, 80–81, 90–91, 104–105, 112–113, 117–118 | Netzel zitiert Colleoni et al. 2012, https://doi.org/10.1016/j.epsl.2011.12.020, und erläutert den Stack nach Wang et al. 2010. Öffentliches Primärarchiv: https://doi.org/10.1594/PANGAEA.790006 . Die heruntergeladene δ18O-Stackspalte umfasst 5331 Werte von 0 bis 5330 ka, in 1-ka-Schritten. Eine Identität mit Netzels vorverarbeiteter Datei ist nicht nachgewiesen. |
| U07 | MEDSTACK-Einzelreihen | Herkunft des Stacks; wichtig für mögliche Zerlegung und Vermeidung von Doppelzählung | Im Archiv 790006 stehen KC01B, MD84-641, ODP 160-967 und Aufschlussdaten neben den Stackspalten. Zugehörige Daten-DOIs: 789990, 789994, 789998, 790005. Die δ13C-Spalten sind zusätzlich verfügbar; ihre Assimilation durch Netzel ist damit nicht belegt. |
| U08 | LR04, benthisches δ18O | Überregionale Referenz und teilweise Grundlage bestehender Chronologien; S. 9–13, 52, 80 | Originalautor und Download: https://www.lorraine-lisiecki.com/stack.html ; https://www.lorraine-lisiecki.com/LR04stack.txt . Datensammlung: https://doi.org/10.1594/PANGAEA.704257 . 5,3 Mio. Jahre. Die Originalquelle beschreibt 57 Kerne; die Dissertation nennt an einer Stelle 52. Für Quellenangaben ist die Originalbeschreibung maßgeblich. |
| U09 | NGRIP/NorthGRIP, Eis-δ18O | Überregionale Referenz; S. 9–13 und 52; mittelbar in der Van-Chronologie, S. 105 | https://doi.org/10.25921/g2m5-qd24 ; Direktdatei: https://www.ncei.noaa.gov/pub/data/paleo/icecore/greenland/summit/ngrip/isotopes/ngrip-d18o-50yr-noaa.txt . Die ursprüngliche Serie nutzt ss09sea; heutige NOAA-Datei enthält sowohl b2k-Grenzen als auch BP1950-Mittelpunkte. |
| U10 | SISAL-Speläothem-δ18O und Datierungen | Zusätzliche Referenzen aus nahegelegenen Höhlen und Alters-Distanz-Modelle; S. 15–18, 24–25, 52–53 | Die genaue verwendete SISAL-Version und vollständige Entity-Liste bleiben offen. Ältere Versionen: https://www.ncei.noaa.gov/pub/data/paleo/speleothem/SISAL/ . SISALv3 ist eine spätere Erweiterungsquelle, nicht der ursprüngliche Datenbankstand. |
| U11 | Soreq, Bar-Matthews et al. 2003 | Ausdrücklich verwendete Isotopenreferenz der holozänen Levante-Beispiele; Abb. 3.19, 3.29, 3.34 | https://doi.org/10.25921/8a4v-ac50 ; Soreq-Datei: https://www.ncei.noaa.gov/pub/data/paleo/speleothem/israel/soreq_2003-noaa.txt . Das Archiv enthält zusätzlich Peqiin; das ist keine automatische Bestätigung ihrer Verwendung in diesen Beispielen. |
| U12 | Soreq, Grant et al. 2012 | Lange Isotopenreferenz für das Tote Meer, ausdrücklich in Abb. 3.24, S. 81 | https://doi.org/10.25921/mf4a-k406 ; https://www.ncei.noaa.gov/pub/data/paleo/contributions_by_author/grant2012/grant2012soreq.txt . Enthält U/Th-Daten, frühere und modellierte Alter sowie δ18O vor und nach chronologischer Modellierung. Die genaue eingelesene Spaltenauswahl muss vor einer Reproduktion festgestellt werden. |
| U13 | Ascunsă, POM2, Drăguşin et al. 2014 | Reales Beispiel der Alters-Distanz-Transformation, Abb. 2.12–2.14, S. 25–28 | https://doi.org/10.5194/cp-10-1363-2014 . Im heutigen SISALv3 Entity 161. Die Verwendung im Methodenbeispiel belegt noch keine bestimmte Zuordnung zu einem der regionalen Pollenprofile. |
| U14 | Giesecke et al. 2017: interpolierte Vegetationskarten | Abgeleitete thermophile Baumreferenz anstelle standortspezifischer AP-Kurven; S. 67–69 | https://doi.org/10.1594/PANGAEA.865120 ; Gesamtsammlung https://doi.org/10.1594/PANGAEA.865121 . 16 Taxa, 0–15 ka, 500-Jahres-Schritte, 0,5°-Raster. Netzel vergleicht sieben thermophile mit neun weiteren Taxa. Abhängigkeit von EPD-Daten beachten. |
| U15 | Simonis et al. 2012 / DECVeg, Pollen und Makroreste | Zusätzliche 50 Paläostandorte für die 6-ka-Sensitivitätsanalyse, S. 133, Anhang B.1 | https://doi.org/10.1016/j.quaint.2012.02.050 . Weitzel et al. 2019 verweist ausdrücklich auf die Synthese im Repository https://bitbucket.org/nils_weitzel/spatial_reconstr_repo . Die dortige Datei `Data/shl_12.RData` wurde heruntergeladen; `Main.R` bezeichnet sie ausdrücklich als Simonis-2012-Proxydatensatz. Die Identität mit Netzels tatsächlich verwendeter Datei und deren Aufbereitung ist noch nicht geprüft. |
| U16 | CMIP5/PMIP3-Modellensemble | Räumlicher Klima-Prior für HIST, 6 ka und 21 ka, Tabelle 2.3, S. 46 | Siehe folgende Tabelle. Datenzugang: https://pmip.lsce.ipsl.fr/protocols_and_data/model_database . Exakte Run-IDs, Dateiversionen und vollständige Dateiliste fehlen im gefundenen Archiv. |

Die Höhlenauswahl in S. 53 folgt einer konkreten Regel: Höhlen innerhalb von **150 km** werden anhand mittlerer Altersauflösung, gemeinsamem Zeitbereich und Entfernung verglichen; gewählt wird eine Entity über ein Pareto-Optimum. Die namenlosen Punkte in Abb. 3.3 reichen nicht aus, um diese Zuordnung exakt wiederherzustellen.

### Verwendete Klimasimulationen

| Modellbezeichnung in Tabelle 2.3 | Simulierte Jahre HIST / MH / LGM |
|---|---:|
| CCSM4 | 156 / 301 / 101 |
| MRI-CGCM3 | 156 / 100 / 100 |
| CNRM-CM5 | 156 / 100 / 200 |
| MPI-ESM-P | 156 / 100 / 100 |
| GISS-E2-R | 156 / 100 / 100 |
| IPSL CM5ALR, im Archiv üblicherweise IPSL-CM5A-LR | 156 / 500 / 200 |
| FGOALS-g2 | 156 / 685 / 100 |

Die Variablen sind TANN und PANN, mit bilinearer Überführung auf 2,5° × 2,5°. Zur Beschaffung eignen sich Filter für `historical`, `midHolocene`, `lgm` und die zugrundeliegenden Temperatur-/Niederschlagsvariablen. Eine bestimmte Ensemblekennung wie `r1i1p1` ist ohne weitere Belege nicht als Netzels tatsächliche Auswahl festzuschreiben. Treibhausgase, Orbitalparameter, Eisschilde, Topographie und Küstenlinien sind Randbedingungen dieser Simulationen; das macht sie nicht zu separat assimilierten lokalen Messreihen.

## 4. Die ausdrücklich untersuchten Sedimentarchive

| Archiv | In der Dissertation untersuchter Zeitraum | Nachgewiesene Daten und Verarbeitung | Originaldaten / Zugangsstatus |
|---|---|---|---|
| Prespa, Co1215 | ungefähr 0–18 ka | Pollen nach Panagiotopoulos et al. 2013; zusätzliche persönlich übermittelte Proben; Datierungen/Tephren nach Aufgebauer et al. 2012. MEDSTACK und AP als Temperaturreferenzen. S. 57–69. | Publikation https://doi.org/10.1016/j.quaint.2012.05.048 ; Datierungsstudie https://doi.org/10.1016/j.quaint.2012.02.015 . Kein eigenständiger vollständiger öffentlicher Download der von Netzel ergänzten Matrix gefunden. |
| Totes Meer, En Gedi, 21-m-Kern | ungefähr 0–10 ka | Pollen nach Litt et al. 2012; AP und Soreq-Referenz. Anthropogene Beeinflussung ab etwa 6,5 ka wird gesondert behandelt. S. 71–79. | https://doi.org/10.1016/j.quascirev.2012.06.012 . Exakte Rohmatrix und alle ursprünglichen Aufbereitungsdateien nicht gefunden. |
| Totes Meer, ICDP 5017-1-A | ungefähr 14,5–147 ka | Pollen nach Chen & Litt 2018 und Miebach et al. 2019; 14C-, U/Th- und teilweise LR04-Altersanker. MEDSTACK und Soreq/Grant als Referenzen. Die AP/NAP-Kurve wird für diese lange Fallstudie ausdrücklich nicht als Klima-Referenz verwendet. S. 80–89. | 88–14 ka: https://doi.org/10.1594/PANGAEA.900564 . Älterer Teil: Publikation https://doi.org/10.1016/j.quascirev.2018.03.029 ; vollständiger separater Rohdownload hierfür nicht gefunden. |
| Kinneret / See Genezareth, KI10-Komposit | ungefähr 0–9 ka | Pollen nach Schiebel & Litt 2018, persönlich übermittelte Ergänzungen; Altersmodell nach Miebach et al. 2022. MEDSTACK, AP und Soreq. S. 89–96. | Basis: https://doi.org/10.1594/PANGAEA.884740 . Ergänzende Studie https://doi.org/10.1016/j.gloplacha.2022.103774 . Ein späteres Kinneret-Codebeispiel ist offen, aber nicht identisch mit der gesamten Dissertation. |
| Birkat Ram, zwei Komposite | ungefähr 0–30 ka | Schwab et al. 2004 und Schiebel 2013. Jüngerer Teil bis etwa 6,25 ka aus höher aufgelöstem altem Kern; Verknüpfung über magnetische Suszeptibilität. MEDSTACK, AP und Soreq. S. 96–103. | https://doi.org/10.1016/j.quascirev.2004.05.001 ; Schiebel: https://hss.ulb.uni-bonn.de/2013/3270/3270.htm . Vollständige kombinierte Rohmatrix nicht gefunden. |
| Van, VAN04-2 | jüngerer Teil der 0–130-ka-Rekonstruktion | Pollen nach Litt et al. 2009; Verbindung mit Ahlat-Ridge-Profil bei etwa 20 ka. S. 103–105. | https://doi.org/10.1594/PANGAEA.874345 ; alternative EPD-Version https://doi.org/10.1594/PANGAEA.894270 . Diese zwei Versionen nicht als unabhängige Profile zählen. |
| Van, Ahlat Ridge, jüngeres Glazial | ungefähr 10–110 ka | Pollen nach Pickarski et al. 2015; gemeinsames Altersmodell nach Stockhecke et al. 2014. S. 103–110. | https://doi.org/10.1594/PANGAEA.853776 . |
| Van, Ahlat Ridge, letztes Interglazial | ungefähr 110–135 ka | Pollen nach Pickarski et al. 2015; Pinus wird aus der AP-Referenz entfernt. S. 103–110. | Pollen https://doi.org/10.1594/PANGAEA.853729 ; Gesamtpaket mit weiterem Multiproxy-Material https://doi.org/10.1594/PANGAEA.853731 . |
| Van: Datierungsbasis | Altersmodell reicht weiter zurück als die diskutierte Rekonstruktion | Varven, 14C, Tephren und klimatische Synchronisierung; Netzel speist Daten bis zur ersten Lücke bei ungefähr 260 ka in Bacon ein. S. 105. | Stockhecke et al. 2014: https://doi.org/10.1016/j.quascirev.2014.04.008 . NGRIP/GLTsyn/SpeleoAge sind mittelbare Chronologiereferenzen und keine voneinander unabhängigen lokalen Van-Proxys. |
| Iznik, LC-IZN-Komposit | ungefähr 0,55–31 ka, jüngste Probe ca. 1400 n. Chr. | Pollen und gemäßigte Baum-/Strauchtaxa nach Miebach et al. 2016; MEDSTACK. Zusätzlicher LGM-Niederschlagsanker aus modernem zentralanatolischem BSk-Klima. S. 111–116. | https://doi.org/10.1594/PANGAEA.858056 ; Datierungen im Supplement https://doi.org/10.5194/cp-12-575-2016-supplement . |
| Ohrid, DEEP / ICDP 5045-1 | ausdrücklich 1165–1365 ka, also etwa 1,2–1,4 Mio. Jahre | Pollen nach Panagiotopoulos et al. 2020; Tephren, Orbitaltuning und zwei magnetische Umkehrungen nach Wagner et al. 2019. MEDSTACK und gemäßigte Baumpollen ohne Pinus. S. 116–122. | Pollenstudie https://doi.org/10.1016/j.quascirev.2019.106044 ; Datenfamilie https://doi.org/10.1594/PANGAEA.896848 und https://doi.org/10.1594/PANGAEA.896847 . Letzterer Datensatz enthält aggregierte Pollenwerte, nicht automatisch das vollständige Taxonspektrum der Fallstudie. |

Bei Ohrid werden moderne Analogareale für Relikttaxa wie Cedrus libani, Liquidambar, Pterocarya, Zelkova, Carya und Tsuga berücksichtigt (S. 117, 120). Die teilweise persönlich kommunizierte Zuordnung ist ein eigener Reproduktionspunkt. Das heutige Klima-Areal-Verhältnis über mehr als eine Million Jahre als konstant zu behandeln ist eine starke Modellannahme.

## 5. Vergleichsdaten und illustrative Daten

Diese Quellen sind nicht pauschal zusätzliche unabhängige Modellinputs. Manche beruhen auf denselben Pollenprofilen, andere werden nur zur Interpretation oder zum Vergleich verwendet.

| Quelle | Funktion in der Arbeit | Bezug |
|---|---|---|
| Nordatlantischer Drift-Eis-/IRD-Record nach Bond; Waveletvergleich nach Debret et al. 2007 | Abb. 2.7, S. 17–18: Vergleich des Spektrums mit Soreq. Keine nachgewiesene allgemeine Assimilation als zusätzliche Referenz in allen Läufen. | NOAA https://doi.org/10.25921/bgvh-nb23 ; https://www.ncei.noaa.gov/access/paleo-search/study/8671 . Die Dissertation führt die Science-Arbeit als Bond et al. 2002, der NOAA-Nachweis und DOI als 2001. |
| Soreq-Arbeiten 1997, 1999, 2007, 2011 sowie Orland et al. 2009 | Interpretation von Ereignissen und Niederschlag; konkrete Eingangskurven sind gesondert oben genannt | Orland https://doi.org/10.1016/j.yqres.2008.08.005 ; Bar-Matthews 2003 https://doi.org/10.1016/S0016-7037(02)01031-1 . |
| Jeita, Cheng et al. 2015 | Regionaler Vergleich, Spektralperioden und nord-/südlevantinische Unterschiede; S. 15–18 und 138 | https://doi.org/10.25921/attj-1g71 ; NOAA Studie 20446. Gute Erweiterungsquelle, aber nicht als konkreter Input eines benannten Netzel-Laufs nachgewiesen. |
| Villars, Poleva und weitere Speläothemstudien | Literaturbegründung zu D/O-, Bond- und 8,2-ka-Ereignissen; S. 15–18 | In SISAL auffindbar; siehe Kandidatenverzeichnis. Literaturerwähnung ist kein Beleg einer bestimmten Modellzuordnung. |
| Waltgenbach et al. 2020, deutsche Höhlen | Literatur und Altersmodellkontext, S. 16, 24 | https://doi.org/10.1016/j.gloplacha.2020.103266 . Die kurzen Ereignisfenster sind nicht automatisch für lange Waveletspektren geeignet. |
| Deininger et al. 2016, Demény et al. 2021, Niggemann et al. 2003 | Literatur zu Kohärenz europäischer Höhlenreihen und Nordatlantik-Verbindungen, S. 16–17 | Belegen den Interpretationsrahmen. Nicht als vollständige getrennte Rohdatensammlung im Modellinventar ausgegeben. |
| Hazan et al. 2005: Kinneret-/Lisan-Seespiegel | Prüfung der Niederschlagsinterpretation, u. a. S. 90–96 | https://doi.org/10.1016/j.yqres.2004.09.004 . Öffentliche Publikation gefunden; eigenständige vollständige maschinenlesbare Originalkurve nicht gesichert. |
| Migowski et al. 2006, Stein et al. 2010, Torfstein et al. 2013: Dead-Sea-/Lisan-Seespiegel | Hydroklimatischer Vergleich, insbesondere Unterschied zwischen Niederschlag und verfügbarer Feuchte | Migowski https://doi.org/10.1016/j.yqres.2006.06.010 ; Torfstein 2013 https://doi.org/10.1016/j.quascirev.2013.02.015 . Für eine Integration werden Originalpunkte, Fehler, Pegeldatum und Wasserbilanz benötigt. |
| Miebach et al. 2022: Kinneret-Karbonatisotope und Holzkohle | Vergleich und Datierungskontext; δ13C wird zur Interpretation diskutiert, kein belegter zusätzlicher δ13C-Modellkanal in der Dissertation | https://doi.org/10.1016/j.gloplacha.2022.103774 ; Holzkohle-Morphotypen später archiviert unter https://doi.org/10.1594/PANGAEA.957251 . |
| Neumann et al. 2007 und Thoma 2017 | Frühere quantitative Birkat-Ram-/Kinneret-/Prespa-Rekonstruktionen, u. a. S. 102 | Neumann https://doi.org/10.1007/s00334-006-0046-x . Thoma: https://hdl.handle.net/20.500.11811/7204 ; Daten vollständig nicht gesichert. Geteilte Pollenbasis beachten. |
| Mauri et al. 2015 | Vergleich der räumlichen Holozänfelder, S. 131–134 | https://www.ncei.noaa.gov/access/paleo-search/study/18317 ; Autorenbeschreibung und Downloadzugang https://wp.unil.ch/davisgroup/datadownloads/ . NetCDF, 1° und 1000 Jahre. |
| Bartlein et al. 2010/2011 | Vergleich bei 6 und 21 ka, S. 132, 134–138 | https://www.ncei.noaa.gov/access/paleo-search/study/9897 ; Artikel mit CSV-/NetCDF-Supplement https://doi.org/10.1007/s00382-010-0904-1 . |
| Simonis et al. 2012, Stolzenberger 2017, Weitzel et al. 2019 | Vergleich räumlicher Rekonstruktionen und Methoden; Simonis zusätzlich tatsächlich in einer Sensitivitätsanalyse verwendet | Simonis-DOI oben; Weitzel https://doi.org/10.5194/cp-15-1275-2019 ; Stolzenberger https://hdl.handle.net/20.500.11811/7299 . Die Doppelrolle wird ausdrücklich getrennt. |
| Ludwig & Hochman 2022 | Vergleich der LGM-Wasserbilanz mit regionalen Simulationen, S. 137–138 | https://doi.org/10.1088/1748-9326/ac46ea . Als regionales Modellargument diskutiert, nicht eines der sieben Ensemblemitglieder aus Tabelle 2.3. |
| Abb. 1.1 aus Litt et al. 2014 | Übernommene Übersichtsgrafik: Einstrahlung 40°N, Grönland-δ18O, antarktisches CO2, marine Isotope, AP von Van, Urmia und Tenaghi Philippon | Illustration, S. 2. Die Abbildung belegt keine Assimilation von Urmia, Tenaghi Philippon, CO2 oder Insolation als eigenständige zusätzliche Modellinputs. |
| Bohn et al. 2007 und Goring et al. 2016 | Moderne Vegetationsanaloge bei der Interpretation bestimmter Klimaklassen, S. 66, 113, 120 | Kontextdaten zur ökologischen Interpretation, kein dokumentierter eigenständiger Assimilationskanal. |

## 6. Öffentliches Code- und Datenarchiv: wichtige Abgrenzung

Die beiden Zenodo-Pakete wurden tatsächlich heruntergeladen und ihre Dateien geprüft:

- Python v0.1.0: https://doi.org/10.5281/zenodo.8214290
- R v0.1.0: https://doi.org/10.5281/zenodo.8214297
- Zugehörige spätere Publikation: https://doi.org/10.5194/cp-21-357-2025

Im Python-Paket liegen insbesondere:

| Datei | Tatsächlich gefundener Inhalt |
|---|---|
| `core_data.npz` | 160 Tiefenproben × 33 Pollentaxa, 33 × 3 Biomzuordnungen, Taxonnamen, AP-Werte |
| `age_depth_relationship.npz` | `depth_given_age` mit 181 × 1772 Werten, 181 regelmäßige Alter, Tiefen- und mittlere Altersinformation für 160 Proben |
| `nnet_fit_params.npz` | Gewichte eines neuronalen Netzes, Temperatur-/Niederschlagsgitter, Normierungsparameter und moderner Klimaanker |
| `terrain_colors.npz` | Farbdefinitionen für Darstellungen; keine zusätzliche Paläoklima-Messreihe |

Diese Pakete enthalten das **Kinneret-Beispiel einer späteren Publikation**. Sie sind kein belegtes Komplettarchiv der 279 Pflanzenareale, sämtlicher EPD-Profile, der damaligen SISAL-Auswahl oder aller CMIP/PMIP-Dateien. Zudem verwendet der spätere Artikel ein Biommodell und Wintertemperatur, während die Dissertation ihre regionalen Ergebnisse überwiegend als TANN darstellt. Aktuelle GitHub-Versionen können inzwischen weiterentwickelt sein; für die obigen Inhaltsangaben wurde die unveränderliche Zenodo-Version untersucht.

## 7. Sinnvolle zusätzliche Daten für eine Erweiterung

Die folgende Priorisierung ist eine fachliche Einschätzung aus Modellstruktur, Datenverfügbarkeit und regionaler Aussagekraft. Sie ist keine Behauptung, dass Netzel diese Daten schon verwendet hat.

| Priorität | Zusätzliche Daten | Sinnvolle Verwendung | Fundstelle |
|---|---|---|---|
| Hoch | SISALv3, δ18O und weitere Kanäle | Räumlich und zeitlich passende Höhlenprofile; getrennte regionale Beobachtungsmodelle | https://doi.org/10.5287/ora-2nanwp4rk ; Beschreibung https://doi.org/10.5194/essd-16-1933-2024 . Im Paket sind 201 vorgeprüfte Profile mit Messwerten und Datierungen enthalten. |
| Hoch | SISALv3-Altersmodellensembles | Datierungsunsicherheit gemeinsam propagieren statt nur ein mittleres Alter zu verwenden | https://doi.org/10.5281/zenodo.10726619 . Vollständige Ensembles sind verlinkt, nicht im Paket mitgespeichert; die tabellierten Chronologien und Unsicherheiten der Datenbank sind enthalten. |
| Hoch | δ13C, Mg/Ca, Sr/Ca, Ba/Ca, U/Ca, P/Ca und Sr-Isotope derselben Höhlen | Hydrologie, Boden, Vegetation und Karstprozesse besser unterscheiden; gemeinsame Fehler und gemeinsame Datierung modellieren | SISALv3. Diese Kanäle sind nicht alle direkte Niederschlagsmesser. |
| Hoch | Cloşani, Rumänien, seit etwa 20 ka | Regionales Multiproxy-Beispiel aus δ18O, δ13C und Spurenelementen, besonders interessant für Ost-/Südosteuropa | https://doi.org/10.1594/PANGAEA.971689 . Vier Unterdatensätze: 972606, 972607, 971690, 972608. Veröffentlichung nach der Dissertation. |
| Hoch | Jeita und Peqiin zusätzlich zu Soreq | Räumliche Unterschiede der Levante prüfen, statt überall dieselbe Soreq-Referenz vorauszusetzen | Jeita https://doi.org/10.25921/attj-1g71 ; Peqiin zusammen mit Soreq https://doi.org/10.25921/8a4v-ac50 . |
| Hoch | See-Karbonat-δ18O/δ13C und Sedimentgeochemie von Van und Ohrid | Lokale/regionale Hydroklima-Referenzen mit eigenem Prozessmodell | Van https://doi.org/10.1594/PANGAEA.853763 und https://doi.org/10.1594/PANGAEA.853764 ; Ohrid https://doi.org/10.1594/PANGAEA.896847 . |
| Hoch | Unabhängige Temperaturarchive, vor allem Chironomiden und Biomarker aus Temperature12k | Temperatur und Feuchte besser trennen; kalibrierte quantitative Daten als zusätzliche Likelihood oder Anker | https://doi.org/10.25921/4ry2-g808 ; https://www.ncei.noaa.gov/access/paleo-search/study/27330 . Europäische Einzelrecords auswählen, Saison und gemeinsame Rohdaten beachten. |
| Hoch | Lake-level-/Wasserbilanzdaten von Kinneret, Dead Sea/Lisan und weiteren Seen | Verfügbare Feuchte und P–E, nicht ungeprüft PANN | Hazan/Migowski oben. Lokale Einzugsgebiete, Abfluss, Grundwasser und spätere Wasserentnahmen berücksichtigen. |
| Mittel | LegacyPollen 1.0 und LegacyAge 1.0 | Größerer, taxonomisch harmonisierter Pollenpool und dokumentierte Chronologien | https://doi.org/10.1594/PANGAEA.929773 ; https://doi.org/10.1594/PANGAEA.933132 . LegacyAge umfasst 1075 europäische Standorte; Überschneidungen mit EPD/Neotoma vermeiden. |
| Mittel | LegacyClimate 1.0 | Sensitivitätsvergleich und Validierung pollenbasierter Ergebnisse | https://doi.org/10.1594/PANGAEA.930512 ; Methoden https://doi.org/10.5194/essd-15-2235-2023 . Nicht als unabhängig zählen, wenn dieselben Pollenprofile bereits eingehen. |
| Mittel | Iso2k | Weitere δ18O-/δ2H-Archive für die letzten etwa 2000 Jahre | https://doi.org/10.25921/57j8-vs18 ; https://www.ncei.noaa.gov/pub/data/paleo/reconstructions/iso2k/ . Für lange pleistozäne Rekonstruktionen zeitlich ungeeignet. |
| Mittel | Old World Drought Atlas | Baumringbasierte Sommerfeuchte für Europa/Mittelmeer in der Common Era; besonders zur Prüfung jüngerer Abschnitte | https://doi.org/10.25921/rjm6-mq74 ; https://www.ncei.noaa.gov/access/paleo-search/study/19419 . Sommer-PDSI ist weder Jahresniederschlag noch Jahresmitteltemperatur. |
| Mittel | MARGO-SSTs | LGM-Vergleich und Trennung mariner Temperatur- von Isotopen-/Eisvolumensignalen | https://doi.org/10.1594/PANGAEA.733406 ; https://www.ncei.noaa.gov/access/paleo-search/study/12034 . SST nicht direkt als lokale Lufttemperatur einsetzen. |
| Mittel | PMIP4/CMIP6 statt oder neben PMIP3/CMIP5 | Neue räumliche Klima-Priors, zusätzliche Zeitfenster und Unsicherheitstests | https://pmip.lsce.ipsl.fr/protocols_and_data/model_database . Das sind Modellinformationen, keine zusätzlichen Beobachtungen. |
| Ergänzend | Antarktisches CO2, z. B. Bereiter et al. 2015, 800 ka | Externes Forcing bzw. Kovariate im Prior | https://doi.org/10.25921/n8y4-bp27 . Gasalter und Eisalter unterscheiden. Die Serie reicht nicht bis zum frühen Ohrid-Zeitfenster bei 1,3 Ma. |
| Ergänzend | Orbitalparameter und Insolation, z. B. La2004 | Zeit-, Saison- und Breitengrad-spezifisches Forcing | https://doi.org/10.1051/0004-6361:20041335 ; https://vo.imcce.fr/insola/earth/online/earth/La2004/index.html . Bei orbital getunten Chronologien entsteht eine Abhängigkeit, die nicht als unabhängige Bestätigung behandelt werden darf. |
| Nach Modellerweiterung | Holzkohle, Landnutzungsindikatoren, archäologische Nutzungsphasen | Vorzugsweise als Indikatoren menschlicher Störung der Vegetation | Kinneret-Holzkohle https://doi.org/10.1594/PANGAEA.957251 . Nicht als reine Klimakurve behandeln. |

## 8. Systematische SISALv3-Auswahl im Datenpaket

Es wurde der offizielle NOAA-Spiegel der SISALv3-Datenbank ausgewertet:

https://www.ncei.noaa.gov/pub/data/paleo/speleothem/SISAL-v3/sisalv3_database_mysql_csv.zip

Die darin enthaltenen CSV-Tabellen tragen das Datum 04.03.2024. Die Auswahl ist reproduzierbar über folgende Kriterien:

- Untersuchungsfenster **25–75°N, 15°W–45°E**. Das ist ein geografisches Rechteck und keine politische Europa-Abgrenzung; es umfasst auch Teile Nordafrikas und der Levante.
- `entity_status = current`. Dadurch werden als veraltet markierte Datenbankversionen nicht übernommen.
- Mindestens 20 numerisch vorhandene δ18O-Werte je Profil.
- Spannweite der originalen Alter der δ18O-Proben größer als 2000 Jahre.

Das Rechteck enthält zunächst 123 Höhlenstandorte und 263 aktuelle Entities. Nach den Mindestkriterien bleiben **201 Profile an 106 Standorten**. Diese Auswahl ist ein erster Filter. Sie prüft weder einen Abstand von 150 km zu einem bestimmten Pollenprofil noch den tatsächlich gemeinsamen Zeitraum mit ihm. Auch das Überschreiten einer Zeitspannweite von 2000 Jahren beweist keine kontinuierliche Ablagerung.

| Inhalt der gefilterten Auswahl | Profile mit Messwerten | Zahl numerischer Messwerte |
|---|---:|---:|
| δ18O | 201 | 136597 |
| δ13C | 171 | 81809 |
| Mg/Ca | 27 | 7196 |
| Sr/Ca | 20 | 6035 |
| Ba/Ca | 16 | 5199 |
| U/Ca | 9 | 3852 |
| P/Ca | 10 | 3049 |
| Sr-Isotope | 9 | 89 |

**32 Profile** enthalten mindestens einen zusätzlichen Element- oder Sr-Isotopenkanal. Kategorien überlappen. Insbesondere ist die Summe der Zeilen keine Anzahl unabhängiger Klimaarchive. Für **79 Profile** gibt es explizite Hiatus-Einträge; auch ein Profil ohne solchen Eintrag kann Lücken enthalten.

Anhand bloßer Min-/Max-Alter umfassen 63 Profile den Zeitpunkt 6 ka und 25 Profile 21 ka. Dies sind obere Grenzen möglicher Kandidatenzahlen: Die Zeitpunkte können innerhalb einer Lücke liegen. Für 1,3 Ma findet sich unter diesen ausgewählten Profilen kein abdeckendes Profil; hier können die Höhlenreihen die frühpleistozäne Ohrid-Fallstudie nicht unmittelbar ergänzen.

Beispiele für gut auffindbare Profile; Zeitspannen sind Min/Max und kein Nachweis kontinuierlicher Abdeckung:

| Höhle / Profil | SISALv3 Entity-ID | Ungefähre Altersspanne, ka BP1950 | Bemerkung |
|---|---:|---:|---|
| Ascunsă POM2 | 161 | −0,032 bis 8,169 | Methodenbeispiel der Dissertation; δ18O und δ13C |
| Ascunsă POM1 | 582 | 30,728 bis 47,177 | Zusätzliches glaziales Profil |
| Soreq-composite185 | 690 | −0,050 bis 183,950 | Im heutigen SISAL als 2003er Datenfamilie; nicht automatisch die Grant-Chronologie |
| Soreq 2N-HR | 353 | 4,384 bis 33,748 | Höher aufgelöste zusätzliche Isotopeninformation |
| Jeita-1 | 58 | 1,137 bis 12,288 | Regionaler Levante-Kandidat |
| Jeita-2 | 59 | 13,330 bis 20,367 | Glazial-/Deglazial-Kandidat |
| Bunker BuStack | 244 | −0,057 bis 10,724 | Holozän, Deutschland; Stack nicht zusätzlich zu seinen Komponenten unabhängig zählen |
| Bleßberg BB-1 | 564 | 0,150 bis 5,615 | Holozän, Deutschland |
| Bleßberg BB-3 | 565 | 5,249 bis 11,991 | Ergänzender holozäner Abschnitt |
| Poleva PP10 | 428 | 2,360 bis 74,080 | Rumänien; Lückenprüfung besonders wichtig |
| Sofular SO-1 | 305 | −0,056 bis 50,275 | Nordanatolien; Wasserquelleneffekte prüfen |
| Antro del Corchia CC_stack | 669 | 108,002 bis 139,999 | Letztes Interglazial/Termination; Überschneidung mit Einzelstalagmiten |
| Villars Vil-car1 | 32 | 1,055 bis 178,002 | δ18O, δ13C und zusätzliche Elementkanäle; keine durchgehende Abdeckung unterstellen |

## 9. Was im Datenpaket tatsächlich enthalten ist

### Öffentlich heruntergeladene Originaldateien

16 unverändert übernommene Datendateien bzw. Archive:

1. MEDSTACK-Archiv mit Einzelreihen und Stackspalten.
2. Soreq 2003.
3. Soreq/Grant 2012 einschließlich Datierungsinformationen.
4. NGRIP 50-Jahres-Mittel.
5. LR04-Stack.
6. Kinneret-Basispollenprofil.
7. Iznik-Pollenprofil.
8. Dead-Sea-Pollenprofil 88–14 ka.
9. Van VAN04-2.
10. Van glaziales Pollenprofil.
11. Van interglaziales Pollenprofil.
12. Ohrid-Chronologie und Multiproxy-/aggregierte Pollendaten.
13. EPD-Alterskontrollpunkte nach Giesecke.
14. Zugehörige EPD-Entity-Metadaten.
15. ZIP mit interpolierten Giesecke-Vegetationsdateien.
16. Simonis-/DECVeg-Synthese `shl_12.RData` aus dem verlinkten Weitzel-Repository (Commit `ebec70dc88a6e8b923ea53e457d72180c432e9f2`).

`manifest.json` nennt für jede Datei Original-URL, Zugriffsdatum, Bytezahl, SHA-256 und die Einschränkung ihrer Beziehung zum Dissertationseingang. Die Originalheader enthalten Zitate, Parameterdefinitionen und Lizenzangaben. Diese beibehalten und bei Weiterverwendung Datenarchiv sowie Originalpublikation zitieren.

### Extrahierter SISALv3-Erweiterungspool

`SISALv3/kandidaten.json` enthält alle 201 Profilmetadaten einschließlich Standort, Entity-ID, Datierungszahl, Altersbereich, Zahl der Messwerte je Kanal, Hiatuszahl und Herkunftsreferenzen.

Die Dateien `SISALv3/proben_entity_<ID>.jsonl` enthalten zusammen **136856 Probenzeilen**. Eine Zeile enthält das originale `sample`-Objekt sowie die dazugehörigen vorhandenen Chronologie- und Messwertobjekte. Die originalen Zellwerte werden als Strings beibehalten; `NA` bezeichnet fehlende Information. Es wurde keine Interpolation, Umkalibrierung, Normierung oder Änderung der Isotopenwerte vorgenommen. Einige Proben tragen weitere Kanäle, aber keinen numerischen δ18O-Wert, weshalb die Probenzahl von der δ18O-Messwertzahl abweicht.

Weitere JSONL-Tabellen enthalten Entity-Metadaten, Standorte, **3360 Datierungszeilen**, Lamineninformationen, Lücken-/Hiatusangaben und Originalreferenzen. Die tabellierten SISAL-Chronologieangaben sind enthalten, die separaten vollständigen Altersmodellensembles nicht. Das Datenbank-Vollarchiv einschließlich Höhlenbildern ist nicht Teil des Pakets.

Einträge mit publizierten Kompositen, mehreren Stalagmiten derselben Höhle oder wiederverwendeten Messreihen sind weiterhin möglich. `current` ist ein Versionsstatus und kein Beleg statistischer Unabhängigkeit. Quellen ohne eigenständigen Daten-DOI sind über die SISAL-Gesamtquelle und die Publikationsreferenzen nachvollziehbar.

## 10. Empfehlungen für eine wissenschaftlich sinnvolle Integration

Diese Empfehlungen sind eine eigene methodische Bewertung; es wurde kein erweitertes Rekonstruktionsmodell gerechnet.

1. **Zuerst Reproduktion festlegen.** Verwendete Profil-IDs, ursprüngliche Datenbankstände, Pollensummen, Taxonsynonyme, Kernzusammenführungen und Ausschlüsse sichern. Die persönlichen Ergänzungen bei Prespa/Kinneret sowie die 279 Arealraster sind besonders wichtig.
2. **Pro Zeitfenster und Region auswählen.** Holozän, LGM, letztes Interglazial und frühes Pleistozän brauchen unterschiedliche Datenpools. Die 201 Höhlenprofile sind nicht für alle diese Fenster gleichzeitig geeignet. Die ursprüngliche 150-km-Regel kann als Ausgangspunkt dienen; Einzugsgebiet und Feuchtequelle sind zusätzliche Kriterien.
3. **Datierungen gemeinsam propagieren.** BP1950, b2k, ka und Jahre unterscheiden. In der aktuellen NOAA-NGRIP-Datei ist `age_calBP` bereits auf 1950 bezogen; diese Spalte nicht nochmals um 50 Jahre verschieben. Soreq/2003 und Soreq/Grant benötigen getrennte Provenienz. Klimatisch synchronisierte oder orbital getunte Chronologien als abhängig kennzeichnen.
4. **Keine künstliche Auflösung erzeugen.** Die Dissertation verwendet ein 50-Jahres-Gitter und diskutiert einen Tiefpass für Perioden unter 400 Jahren sowie mindestens 2000 Jahre lange Reihen (S. 18, 28). Das MEDSTACK-Primärarchiv hat 1000-Jahres-Schritte. Ein feineres Rechengitter erzeugt daraus keine neue Jahrhundertinformation. Wavelets und Vergleichsfilter müssen zur echten Auflösung und Altersunsicherheit passen.
5. **Ein Proxy–Klima-Modell pro Datentyp verwenden.** δ18O kann Temperatur, Niederschlagsquelle, Saisonalität, Eisvolumen und lokale Hydrologie enthalten. δ13C und Spurenelemente können Vegetation, Bodenprozesse, prior calcite precipitation und Verweilzeit abbilden. Ein regional begründetes Vorzeichen und ein Unsicherheitsmodell sind nötig.
6. **Abhängige Informationen zusammenfassen.** Pollen, AP/NAP und interpolierte EPD-Baumkarten stammen teilweise aus denselben Daten. Gleiches gilt für einen Stack und seine Einzelreihen, mehrere SISAL-Entities derselben Höhle und mehrere Kanäle desselben Stalagmiten. Die in S. 34 vereinfachend angenommene Unabhängigkeit der Referenzen darf bei einem größeren Pool nicht ungeprüft fortgeschrieben werden. Geeignet sind Archiveffekte, gemeinsame Chronologieensembles und multivariate Fehlerstrukturen.
7. **PANN und Feuchteverfügbarkeit trennen.** Hohe Seespiegel oder Waldanteile können bei gleichem oder niedrigerem Niederschlag auch aus geringerer Verdunstung entstehen. Für die Levante ist ein zusätzlicher P–E-/Wasserbilanzbaustein besonders sinnvoll. Netzel nennt Verdunstung selbst als Erweiterung (S. 142).
8. **Wirklich unabhängige Prüfungen reservieren.** Dieselben Pollenprofile nicht zugleich als Eingang, abgeleiteten Klimaanker und unabhängigen Validierungstest verwenden. CRU wird bereits zur Kalibrierung und Verankerung genutzt; die spätere CRU-Übereinstimmung ist deshalb eine Konsistenzprüfung und keine vollständig unabhängige Validierung. Geeignet sind zurückgehaltene Archive, standortweises Cross-Validation und Sensitivitätsläufe ohne einzelne Referenzfamilien.
9. **Spektrale Übereinstimmung ergänzen.** Ähnliche Leistungsspektren können gemeinsame Perioden zeigen, ohne gleiche Phasenlage oder gleiche physikalische Ursachen zu beweisen. Ein R²-Maß allein sichert das Vorzeichen des Zusammenhangs nicht. Zeitbereich, Vorzeichen, Phasenlage, Chronologie und Prozesswissen zusammen prüfen.

Als erster Erweiterungsschritt bietet sich ein standortbezogener Pool aus **Pollen + lokalem Höhlen-δ18O + δ13C/Spurenelementen + unabhängigem Temperaturproxy + einer hydrologischen Information** an. CO2, Insolation und Simulationen sollten als eigene Forcing-/Prior-Komponenten geführt werden. Die Anzahl eingelesener Reihen ist kein Qualitätsmaß; entscheidend sind zusätzlicher Informationsgehalt, räumliche Repräsentativität und transparente Unsicherheit.

## 11. Noch offene Punkte für einen exakten Nachbau

- Vollständige damalige EPD-Profilauswahl einschließlich verworfener Läufe und Gründe.
- Liste der 186 akzeptierten Rekonstruktionen mit ursprünglichen IDs und Ergebnisdateien.
- SISAL-Version, verwendete Entity-IDs, Proxyspalten und Zuordnung zu jedem Pollenprofil.
- Originaldatei und Aufbereitung von MEDSTACK; das identifizierte Wang-Archiv ist eine gesicherte Primärdatenquelle, aber kein nachgewiesener Byte-für-Byte-Input.
- Tatsächliche Soreq-Datei/Spaltenauswahl für jedes Zeitfenster.
- Persönlich übermittelte Pollenproben für Prespa und Kinneret sowie vollständige moderne Areal-/Taxon-Matrix.
- Identität der DECVeg-Matrix im verlinkten Weitzel-Repository mit Netzels tatsächlich verwendeter 50-Standorte-Matrix und deren Anpassung an seine Transferfunktionen.
- Ensemble-Mitgliedskennungen, Downloadversionen und Vorverarbeitung aller sieben CMIP5/PMIP3-Modelle.
- Daten zu einzelnen älteren Fallstudien, für die bisher nur Publikation, EPD-Zugang oder Aggregationen gefunden wurden.

Die offenen Punkte sind im Verzeichnis absichtlich sichtbar: Eine Literaturquelle zu finden, Originaldaten zu beschaffen und den tatsächlich verwendeten Modellinput nachzuweisen sind drei unterschiedliche Arbeitsschritte.


## Anhang A. Alle 201 SISALv3-Erweiterungskandidaten

Dies ist die zusätzliche Auswahl nach Abschnitt 8, keine Liste der ursprünglichen Thesis-Inputs. Zeitspannen bezeichnen minimale und maximale originale Probenalter in ka BP1950; Lücken, überlappende Komposite und Abhängigkeiten müssen separat geprüft werden. Vollständige Metadaten und Referenzen stehen in `SISALv3/kandidaten.json` im Datenpaket.

| Höhle | Profil | Entity-ID | Altersspanne (ka BP) | δ18O-Werte | Weitere Kanäle |
|---|---|---:|---:|---:|---|
| Abaliget cave | ABA_1 | 105 | 123.303–143.456 | 339 | d13C |
| Abaliget cave | ABA_2 | 106 | 140.274–160.598 | 440 | d13C |
| Antro del Corchia | CC-1_2018 | 665 | 128.484–139.964 | 184 | d13C |
| Antro del Corchia | CC-5_2018 | 666 | 108.097–139.980 | 1334 | d13C |
| Antro del Corchia | CC-7_2018 | 667 | 121.308–128.592 | 310 | d13C |
| Antro del Corchia | CC-28_2018 | 668 | 108.000–118.425 | 604 | d13C |
| Antro del Corchia | CC_stack | 669 | 108.002–140.000 | 1538 | d13C |
| Antro del Corchia | CC27 | 670 | 2.442–5.993 | 247 | d13C |
| Ascunsa cave | POM2 | 161 | -0.032–8.169 | 150 | d13C |
| Ascunsa cave | POM1 | 582 | 30.728–47.177 | 740 | d13C |
| B7 cave | STAL-B7-7 | 560 | -0.047–17.551 | 227 | d13C |
| Baradla cave | BAR-II#L | 159 | 109.194–129.003 | 58 | d13C |
| Baradla cave | BAR-II#B | 160 | 108.758–128.125 | 501 | d13C |
| Baschg cave | BA-1b | 70 | 75.492–80.897 | 778 | — |
| Baschg cave | BA-1 | 71 | 80.982–89.489 | 1261 | — |
| Baschg cave | BA-5 | 643 | 84.956–90.300 | 279 | d13C |
| Baschg cave | BA-7 | 644 | 80.902–86.918 | 387 | d13C |
| Beatus cave | EXC3 | 397 | 100.839–110.038 | 1974 | — |
| Beatus cave | EXC4 | 398 | 77.470–107.247 | 1063 | — |
| Bigonda cave | BG2 | 883 | 110.195–130.510 | 289 | d13C, Mg_Ca, Sr_Ca, Ba_Ca, U_Ca, P_Ca |
| Bigonda cave | BG4 | 884 | 104.472–132.474 | 469 | d13C, Mg_Ca, Sr_Ca, Ba_Ca, U_Ca, P_Ca |
| Bleßberg cave | BB-1 | 564 | 0.150–5.615 | 1451 | d13C |
| Bleßberg cave | BB-3 | 565 | 5.249–11.991 | 506 | d13C |
| Bourgeois–Delaunay cave | BDinf | 162 | 121.339–128.151 | 358 | d13C |
| Buca della Renella | RL4_2018 | 381 | 1.024–7.277 | 758 | d13C |
| Bue Marino cave | BMS1 | 195 | 110.207–112.881 | 295 | d13C |
| Bulgarian Orlova Chuka cave | ocz-6 | 566 | 112.355–129.018 | 763 | d13C |
| Bunker cave | BuStack | 244 | -0.057–10.723 | 2677 | — |
| Buraca Gloriosa | BG41 | 583 | 73.230–82.606 | 37 | d13C |
| Buraca Gloriosa | BG66 | 584 | 185.269–222.448 | 279 | d13C |
| Buraca Gloriosa | BG67 | 585 | 104.320–146.537 | 43 | d13C |
| Buraca Gloriosa | BG611 | 586 | 53.143–109.788 | 80 | d13C |
| Buraca Gloriosa | BG6LR | 587 | 1.319–87.440 | 961 | d13C |
| Béke cave | BNT-2 | 595 | 0.454–4.348 | 261 | d13C |
| Ceremosnja cave | CC-1 | 165 | -0.048–2.426 | 100 | d13C |
| Cesare Battisti cave | CB25 | 400 | 121.282–125.124 | 72 | d13C |
| Cesare Battisti cave | CB39 | 401 | 120.287–127.717 | 199 | d13C |
| Cesare Battisti cave | CB47 | 402 | 123.556–126.412 | 164 | d13C |
| Chaara cave | Cha2_2019 | 588 | -0.058–9.662 | 657 | — |
| Chaara cave | Cha1 | 589 | -0.051–11.622 | 1573 | — |
| Chauvet cave | Chau-stm6 | 166 | 11.415–34.183 | 210 | d13C |
| Clamouse cave | Cla4 | 211 | 74.460–187.405 | 77 | d13C |
| Clamouse cave | CL26 | 212 | 0.142–11.179 | 232 | d13C |
| Closani cave | C09-2 | 390 | -0.036–3.793 | 1832 | d13C, Mg_Ca, Sr_Ca, Ba_Ca, U_Ca |
| Cobre cave | PA-8 | 561 | 100.606–116.342 | 252 | d13C, Mg_Ca, Sr_Ca, Ba_Ca, U_Ca, P_Ca |
| Cova da Arcoia | ESP03 | 310 | 0.340–9.440 | 100 | d13C |
| Coves de Campanet | CAM-1 | 569 | 0.152–121.015 | 448 | d13C |
| Crag cave | CC3 | 196 | -0.047–10.132 | 145 | d13C |
| Crovassa Azzurra | CA | 562 | 72.640–238.168 | 737 | d13C |
| Cueva de Asiul | ASR | 248 | 0.489–12.161 | 1191 | — |
| Cueva de Asiul | ASM | 249 | -0.062–7.777 | 828 | — |
| Cueva Rosa | Artemisa_R | 750 | 100.638–122.997 | 48 | d13C, Mg_Ca, Sr_Ca |
| Cueva Rosa | Angelines | 751 | 65.049–117.735 | 189 | d13C, Mg_Ca, Sr_Ca |
| Cueva Victoria | Vic-III-4 | 570 | 6.988–14.926 | 196 | d13C |
| Cueva Victoria | Vic-III-1 | 697 | 45.383–95.162 | 440 | d13C |
| Cueva Victoria | Vic-III-3 | 698 | 48.696–93.255 | 274 | d13C |
| Cueva Victoria | SR01t | 699 | 49.138–86.522 | 120 | d13C |
| Dim cave | Dim-E2 | 168 | 9.738–13.094 | 88 | d13C, Mg_Ca, Sr_Ca, Ba_Ca, U_Ca, Sr_isotopes |
| Dim cave | Dim-E3 | 169 | 12.575–89.714 | 442 | d13C, Mg_Ca, Sr_Ca, Ba_Ca, U_Ca, Sr_isotopes |
| Dim cave | Dim-E4_2023 | 693 | 12.020–14.555 | 49 | d13C, Mg_Ca, Sr_Ca, Ba_Ca, U_Ca, Sr_isotopes |
| Dim cave | Dim1 | 754 | 128.299–132.151 | 421 | d13C, Sr_isotopes |
| Ejulve cave | HOR | 250 | 2.708–6.072 | 266 | d13C |
| Ejulve cave | ARTEMISA | 251 | 218.976–257.426 | 236 | d13C, Mg_Ca |
| Ejulve cave | Andromeda | 702 | 37.117–119.346 | 735 | d13C, Mg_Ca |
| El Soplao cave | SIR-1 | 563 | 7.965–14.670 | 329 | d13C |
| Entrische Kirche cave | TKS | 252 | 113.389–126.889 | 1340 | d13C |
| Frasassi cave system - Grotta Grande del Vento | FR16 | 411 | 96.724–112.958 | 544 | d13C |
| Gassel Tropfsteinhöhle | GAS-12 | 656 | 77.015–81.899 | 743 | d13C |
| Gassel Tropfsteinhöhle | GAS-13 | 657 | 76.914–80.324 | 679 | d13C |
| Gassel Tropfsteinhöhle | GAS-22 | 658 | 105.246–107.971 | 525 | d13C |
| Gassel Tropfsteinhöhle | GAS-25 | 659 | 83.938–91.349 | 625 | d13C |
| Grete-Ruth Shaft | HUN-14 | 662 | 102.918–111.314 | 705 | d13C |
| Grotta di Carburangeli | CR1 | 277 | 0.948–8.374 | 1232 | d13C |
| Grotta di Ernesto | ER76 | 279 | 2.511–7.969 | 1220 | d13C |
| Grotte de Piste | GP2 | 591 | 2.537–11.416 | 588 | — |
| Gruta do Casal da Lebre | GCL6 | 571 | 161.417–186.962 | 55 | d13C |
| Gueldaman cave | stm2 | 173 | 4.039–6.191 | 224 | d13C |
| Gueldaman cave | stm4 | 174 | 3.193–5.793 | 201 | d13C |
| Han-sur-Lesse cave | Han-stm1 | 73 | 4.778–10.949 | 135 | d13C |
| Han-sur-Lesse cave | Han-9 | 75 | 106.500–125.343 | 834 | d13C |
| Herbstlabyrinth cave | TV1_2021 | 775 | 0.114–2.453 | 431 | d13C, Mg_Ca, P_Ca |
| Hölloch im Mahdtal | HOL-7 | 230 | 40.105–48.664 | 530 | d13C |
| Hölloch im Mahdtal | HOL-comp | 235 | 49.063–64.498 | 4150 | — |
| Hölloch im Mahdtal | HOL-10 | 236 | 110.844–131.765 | 418 | d13C |
| Hölloch im Mahdtal | HOL1 | 704 | 11.441–14.940 | 199 | d13C |
| Hölloch im Mahdtal | HOL22 | 705 | 10.218–15.211 | 629 | d13C |
| Hüttenbläserschachthöhle | HBSH-1 | 783 | 0.554–6.091 | 300 | d13C, Mg_Ca, Sr_Ca, Ba_Ca, P_Ca |
| Hüttenbläserschachthöhle | HBSH-3 | 784 | 1.286–10.503 | 354 | d13C, Mg_Ca, Sr_Ca, Ba_Ca, P_Ca, Sr_isotopes |
| Hüttenbläserschachthöhle | HBSH-4 | 785 | 1.223–11.379 | 572 | d13C, Sr_isotopes |
| Hüttenbläserschachthöhle | HBSH-5 | 786 | 8.200–10.932 | 120 | d13C, Mg_Ca, Sr_Ca, Ba_Ca, P_Ca, Sr_isotopes |
| Jeita cave | Jeita-1 | 58 | 1.137–12.288 | 1600 | d13C |
| Jeita cave | Jeita-2 | 59 | 13.330–20.367 | 120 | d13C |
| Jerusalem west cave | AF12 | 152 | -0.016–168.714 | 308 | Sr_isotopes |
| Kanaan cave | Kanaan_MIS5 | 81 | 83.125–128.847 | 204 | d13C |
| Kanaan cave | Kanaan_MIS6 | 82 | 154.455–193.498 | 808 | d13C |
| Karaca cave | K1 | 416 | 5.904–77.267 | 309 | d13C |
| Katerloch cave | K1 | 199 | 7.080–10.324 | 690 | d13C |
| Katerloch cave | K3 | 200 | 7.787–10.027 | 1323 | d13C |
| Katerloch cave | K2 | 794 | 124.957–128.509 | 275 | d13C |
| Katerloch cave | K4 | 795 | 126.793–129.600 | 246 | d13C |
| Klapferloch cave | PFU6 | 201 | -0.047–2.943 | 2814 | d13C |
| Klaus Cramer cave | KC1 | 417 | 63.719–72.014 | 850 | — |
| Kleegruben cave | SPA_126 | 280 | 47.396–55.966 | 1633 | d13C |
| Kleegruben cave | SPA_49 | 281 | 47.816–58.266 | 1591 | d13C |
| Kocain cave | Ko-1 | 887 | -0.050–3.316 | 369 | d13C, Mg_Ca, Sr_Ca, P_Ca |
| Korallgrottan cave | K11 | 202 | -0.055–3.792 | 413 | d13C |
| Korallgrottan cave | K1 | 450 | 5.922–8.896 | 73 | d13C |
| La Garma cave | GAR-01_drill | 650 | -0.026–13.345 | 354 | d13C |
| La Garma cave | GAR-01_laser_d18O | 651 | 10.001–13.864 | 411 | — |
| La Mine cave | Min-stm1 | 176 | 5.316–23.168 | 78 | d13C |
| La Vallina | Gael_2022 | 803 | 121.849–140.013 | 217 | d13C |
| La Vallina | Gloria | 805 | 78.772–194.086 | 70 | d13C, Mg_Ca, Ba_Ca |
| La Vallina | Garth | 806 | 111.282–140.009 | 478 | d13C |
| La Vallina | Luna | 808 | 10.163–12.850 | 54 | d13C |
| Labyrintgrottan cave | L4 | 122 | 7.348–9.565 | 46 | d13C |
| Lancaster Hole | LH-70s-1 | 50 | 3.456–12.718 | 370 | d13C |
| Lancaster Hole | LH-70s-2 | 51 | 0.262–9.736 | 469 | d13C |
| Lancaster Hole | LH-70s-3 | 52 | 0.946–8.463 | 367 | d13C |
| Larshullet cave | L03 | 123 | 0.130–3.921 | 273 | d13C |
| Leány cave | Leany | 177 | 4.739–10.543 | 189 | d13C |
| Limnon cave | KTR-2 | 671 | 6.688–12.413 | 177 | d13C |
| Lithophagus cave | LFG-2 | 558 | 54.935–131.935 | 42 | d13C |
| Ma'ale Efrayim cave | ME-12 | 218 | 16.548–66.948 | 394 | d13C |
| Manita peć cave | MP-2 | 815 | 5.326–93.308 | 177 | d13C |
| Manita peć cave | MP-3 | 816 | 63.676–105.923 | 228 | d13C |
| Mavri Trypa cave | S1 | 347 | 1.296–4.687 | 354 | d13C |
| Meravelles cave | Maat | 821 | 14.672–18.611 | 250 | d13C, Mg_Ca, Sr_Ca |
| Milchbach cave | MB-2 | 255 | 3.249–6.830 | 614 | d13C |
| Milchbach cave | MB-3 | 256 | 1.987–9.026 | 1084 | d13C |
| Milchbach cave | MB-5 | 257 | 3.890–7.246 | 1140 | d13C |
| Mizpe Shelagim | MS-composite | 822 | 4.424–86.400 | 279 | d13C, Sr_isotopes |
| Molinos cave | MO-1 | 216 | 4.727–11.335 | 168 | d13C, Mg_Ca |
| Molinos cave | MO-7 | 217 | 3.253–6.812 | 141 | d13C, Mg_Ca |
| Murada | Indiana | 826 | 113.685–125.496 | 93 | d13C, Mg_Ca |
| Neotektonik cave | M37-1-16A | 827 | 115.218–131.226 | 1810 | d13C |
| Neotektonik cave | M37-1-16C | 828 | 124.382–127.163 | 430 | d13C |
| Neotektonik cave | M37-1-23A | 829 | 129.349–132.167 | 45 | d13C |
| Nova Grgosova cave | NG-7 | 830 | 5.636–7.828 | 358 | d13C |
| Nova Grgosova cave | NG-3 | 831 | -0.052–10.023 | 481 | d13C |
| Okshola cave | FM3 | 95 | -0.047–7.515 | 289 | d13C |
| Okshola cave | Oks82 | 96 | 5.006–10.328 | 149 | d13C |
| Ostolo cave | OST2 | 832 | 10.813–18.713 | 903 | d13C |
| Ostolo cave | OST1 | 833 | 8.526–18.682 | 73 | d13C |
| Ostolo cave | OST3 | 834 | 10.503–17.948 | 331 | d13C |
| Pentadactylos | Pentadactylos-1 | 835 | 110.892–174.599 | 685 | d13C |
| Peqiin cave | PEK_composite | 452 | 14.000–248.200 | 566 | d13C |
| Pere Noel cave | PN-95-5_2018 | 714 | 1.789–12.877 | 130 | d13C, Mg_Ca, Sr_Ca, Ba_Ca |
| Piani Eterni karst system | MN1 | 425 | 84.033–105.480 | 563 | d13C |
| Piani Eterni karst system | GG1 | 426 | 106.118–289.234 | 252 | d13C |
| Piani Eterni karst system | IS1 | 427 | 352.740–374.429 | 149 | d13C |
| Pindal cave | Candela_2023 | 715 | 11.640–29.340 | 422 | d13C |
| Pindal cave | Candela_Base | 716 | 21.915–25.087 | 85 | d13C |
| Pindal cave | Candela_Main | 717 | 13.457–20.409 | 166 | d13C |
| Pindal cave | Laura | 719 | 14.259–16.405 | 122 | d13C |
| Pippikin Pot cave | YD01 | 129 | 4.206–9.479 | 483 | d13C |
| Poleva cave | PP10 | 428 | 2.360–74.080 | 154 | d13C |
| Pot au Feu | JUD | 893 | 68.544–115.100 | 229 | d13C, Mg_Ca |
| Pozzo Cucù cave | PC | 838 | 26.606–106.020 | 2658 | d13C |
| Qadisha cave | Qad_1 | 720 | 3.248–6.814 | 67 | d13C |
| Qadisha cave | Qad_2 | 721 | -0.036–9.133 | 161 | d13C |
| Sa balma des quartó cave | Multieix | 843 | 0.106–2.250 | 303 | d13C, Mg_Ca, Sr_Ca, Ba_Ca |
| Sant'Angelo cave | SA1 | 847 | 6.769–47.820 | 1002 | — |
| Sarma cave | SAR-12-1 | 848 | 2.984–22.408 | 288 | d13C, Sr_Ca |
| Savi cave | SV1 | 894 | 9.261–15.288 | 333 | d13C, Mg_Ca, Sr_Ca, Ba_Ca, U_Ca, P_Ca |
| Savi cave | SV7 | 895 | 10.466–13.172 | 276 | d13C, Mg_Ca, Sr_Ca, Ba_Ca, U_Ca, P_Ca |
| Schafsloch cave  | MF-3 | 260 | 130.050–137.390 | 318 | d13C |
| Schneckenloch cave | SCH-5 | 206 | 115.340–134.085 | 2150 | d13C |
| Schneckenloch cave | SCH-7 | 207 | 111.589–118.315 | 754 | — |
| Schneckenloch cave | SCH-6 | 663 | 75.004–78.121 | 331 | d13C |
| Schratten cave | M6-73-3 | 849 | 112.431–121.120 | 584 | d13C |
| Sieben Hengste cave | 7H | 133 | 14.620–29.873 | 15254 | — |
| Sieben Hengste cave | 7H-2 | 134 | 17.137–29.940 | 2804 | — |
| Sieben Hengste cave | 7H-3 | 135 | 14.639–23.537 | 1477 | — |
| Sieben Hengste cave | 7H-12 | 728 | 117.456–133.621 | 1439 | d13C |
| Skala Marion cave | MAR_L | 136 | 1.481–5.534 | 209 | d13C |
| Sofular cave | SO-1 | 305 | -0.056–50.275 | 3976 | d13C |
| Sofular cave | SO-17A | 429 | 86.190–122.930 | 1306 | — |
| Sofular cave | SO-2 | 456 | -0.060–59.510 | 1111 | — |
| Sofular cave | SO-4 | 687 | 1.080–307.030 | 2022 | — |
| Sofular cave | SO-6 | 688 | 93.320–133.200 | 645 | — |
| Sofular cave | SO-14B | 689 | 475.910–670.000 | 298 | — |
| Soreq cave | 2N-HR | 353 | 4.384–33.748 | 1223 | — |
| Soreq cave | Soreq-composite185 | 690 | -0.050–183.950 | 1387 | d13C |
| Soylegrotta cave | SG95 | 137 | -0.043–4.141 | 169 | d13C |
| Soylegrotta cave | SG93 | 435 | 0.131–10.421 | 308 | d13C |
| Spannagel cave | COMNISPA II | 142 | -0.013–9.931 | 4343 | — |
| Spannagel cave | SPA146 | 730 | 182.336–191.571 | 493 | — |
| Strašna peć cave | SPD-2 | 568 | 1.376–3.808 | 77 | d13C |
| Tausoare cave | 1152 | 457 | 20.011–68.313 | 740 | d13C |
| Ursilor cave | PU-2 | 184 | -0.050–7.068 | 39 | d13C |
| Villars cave | Vil-stm6 | 27 | -0.043–8.657 | 101 | d13C |
| Villars cave | Vil-stm9 | 28 | 31.438–82.855 | 296 | d13C |
| Villars cave | Vil-stm11 | 29 | 5.361–15.875 | 83 | d13C |
| Villars cave | Vil-stm14 | 30 | 28.893–52.156 | 143 | d13C |
| Villars cave | Vil-stm27 | 31 | 31.341–49.663 | 191 | d13C |
| Villars cave | Vil-car1 | 32 | 1.055–178.002 | 605 | d13C, Mg_Ca, Sr_Ca, Ba_Ca |
| Villars cave | Vil-stm1 | 33 | -0.038–2.333 | 336 | d13C |
| White Scar cave | WSC-97-10-5 | 150 | 7.348–11.191 | 160 | d13C |
| Wintimdouine | WIN2 | 868 | -0.037–4.490 | 201 | — |
| Wintimdouine | WIN3 | 869 | 0.047–11.574 | 1566 | — |
| Wintimdouine | WIN_composite | 870 | -0.003–11.562 | 1708 | — |
