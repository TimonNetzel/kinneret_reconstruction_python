# Fallmatrix: lokale Dissertationsrekonstruktionen

Stand 02.11.2026; Phase 1, Ausgangscommit `865d155`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).

**Ergebnis:** Die sechs Standorte aus Kapitel 3.3 ergeben sieben Arbeitsfälle,
weil die beiden Rekonstruktionen des Toten Meeres getrennte Eingaben und
Referenzen besitzen. Kein Fall ist derzeit mit vollständig gebundenen
Originaleingaben numerisch reproduzierbar. **Kinneret ist der empfohlene erste
Klärungsfall**, noch keine Laufempfehlung: eine vorhandene 160-zeilige
Pollentabelle, 160 Chronologieeinträge und zwei byteidentische Abbildungen im
Dissertationsarchiv ergeben den günstigsten gezielten Einstieg. Die genaue
historische Taxon-/TF-Zuordnung und der ausführbare Laufstand fehlen weiterhin.

## Leseregeln und Quellenbindung

**belegt** bedeutet: genau der genannte Text-, Code- oder Dateibefund ist geprüft.
Es bedeutet keine vollständige Laufidentität. **fehlt** bezeichnet eine konkret
nicht geschlossene Abhängigkeit; bei Dateien die Abwesenheit am genannten
Verbraucherpfad, keine Behauptung einer vollständigen Laufwerkssuche.
**widersprüchlich** bezeichnet nicht zusammenpassende Stände, Dimensionen oder
aktive Einstellungen. Es wird keine Variante still ausgewählt oder korrigiert.
Jede Tabellenzelle nennt ihren eigenen Status; die Fallfreigabe bleibt überall offen.

Quellwurzeln `original` und `additional` werden ausschließlich über die vorhandene
private `.local/my_reconst_paths.json` aufgelöst. Relative Pfade unten gehören
zu diesen Wurzeln, nicht zum Checkout. Historische Linux-Pfade müssen vor einem
späteren Lauf explizit gebunden werden. Keine Originale wurden verschoben oder
geändert. Die vorhandenen Kataloge und abgeschlossenen Audits wurden verwendet;
der neue Leser untersucht nur ausdrücklich genannte Dateien.

| Kürzel | Konkreter Beleg und Reichweite |
| --- | --- |
| D | `original:out/plots/mcmc/v5.0/diss_final.pdf`, 176 PDF-Seiten, Titelblatt September 2022, SHA256 `1770993ad65dd5024789568bc8f124f3b46ccc68ce36e4506a76170590209de8`. Kapitel 3.3: gedruckt 70–123, PDF 76–129. Hier maßgebliche lokale Fassung; nicht als endgültige Einreichfassung ausgegeben. |
| E | [Repository-Ausschnitt Kapitel 3.3](../../paper/Phd_thesis_netzel_ch3.3.PDF), 53 PDF-Seiten, gedruckt 70–122. Separater Text-/Layoutstand. Seiten und Abbildungsnummern nicht mit D vermischen. Vollständige SHA256 in [evidence.json](evidence.json). |
| M | D §3.2.2, gedruckt 61–64 / PDF 67–70: historische Auswahl und Kombination der Ziehungen. [Repository-Ausschnitt §3.2](../../paper/Phd_thesis_netzel_ch3.2.PDF) bleibt eigene Fassung. |
| C | `original:main_v1.2.R`, SHA256 `82486f99bafff08af2b51475ed225501d26cec1a0322a286b97a248ebd2f0520`; `src/check_ages.R`, `src/mcmc_prepare.R`, `src/mcmc_functions_v1.6.R`, `src/mcmc_conduction.R`, `src/mcmc_trimming_posterior_trafo.R`, `src/prior_climate.R`. Genaue Hashes und gelesene Zeilen in evidence.json. Statische Quellfassung, keine belegte historische Commit-/Seed-/Paketbindung an alle Abbildungen. |
| F | [Zweiter TF-Plot](../HISTORICAL_SECOND_TF_PLOT_RESULT.md), [Konsolen-/EMD-Bindung](../HISTORICAL_CONSOLE_TF_LINK_RESULT.md), [Taxonverfahren](../LOCAL_TAXON_METHOD_AUDIT.md), [globale Verbreitungsbasis](../MY_RECONST_SOURCE_RESULT.md). Abgeschlossene Prüfungen werden nicht wiederholt. |
| P | [Lokaler Quellenabgleich](../LOCAL_SOURCE_CROSSWALK.md), [Quellenverzeichnis](../Netzel_Analyse_Datenverzeichnis.md), [portabler Forschungsstand](../research_snapshot_2026_11_02/README.md). Publikations-/Downloadkandidaten sind nicht automatisch historische Eingaben. |
| N | [Neue eng begrenzte Metadatenprüfung](evidence.json): sieben CSVs, fünf vorhandene Chronologiecontainer, explizite fehlende Verbraucherpfade, 14 Original-/Archiv-Abbildungsvergleiche und getrennte Seitenprüfungen für D/E. CSV-Spaltenlisten sind Kandidaten, keine finalen TF-Indizes. |

## 1. Verifizierte Fallliste und Zeitfenster

Alter in cal ka BP; jung links, alt rechts. Textfenster, Rohquellenumfang und
numerische Endpunkte eines gespeicherten Kandidaten sind unterschiedliche Angaben.
Die öffentliche Fassung wird durch P erschlossen; D und E sind hier direkt geprüft.
Die bekannte Fassungsfrage bleibt offen ([Versionsabgleich](../LOCAL_SOURCE_CROSSWALK.md)).

| Fall | Dissertationsfenster | Fundstelle D: gedruckt / PDF | Fundstelle E: gedruckt / Ausschnitt-PDF | Grenze |
| --- | --- | --- | --- | --- |
| Totes Meer Holozän | etwa 0–10 | 73–80 / 79–86; §3.3.1 „Holocene“ | 72–79 / 3–10 | Jüngste beschriebene Schicht ca. 1955 AD; nicht automatisch Alter 0. |
| Totes Meer Glazial | etwa 14,5–147; erläuterte älteste Phase 130,9–147,3 | 81–89 / 87–95; „Late Penultimate to Last Glacial“ | 80–89 / 11–20 | Kein heutiger Klimaanker; 88–14-ka-Download deckt den älteren Teil nicht ab. |
| Kinneret | etwa 0–9 | 90–97 / 96–103; §3.3.2 | 89–96 / 20–27 | Datei N: −60 bis 9015 a BP; diese Werte sind kein Nachweis des finalen Altersrasters. |
| Birkat Ram | etwa 0–30 | 98–104 / 104–110; §3.3.3 | 96–102 / 27–33 | Kein 300-ka-Fall: Schreibweise „30,0000“ in D ist mit Kontext 30.000 und der Phase 30–18 ka zu lesen. Exakte numerische Endpunkte offen. |
| Van | Kurzbeschreibung letzte 130 ka; besprochene Phasen bis 133,9 ka, Referenzgrafik bis etwa 135 ka | 105–112 / 111–118; §3.3.4, insbesondere S.107 | 103–110 / 34–41 | Nicht auf 130 ka abschneiden; Eingaben zur Datierung bis zur Lücke bei ca. 260 ka sind kein Rekonstruktionsfenster. |
| Iznik | etwa 0,55–31; jüngste Schicht ca. 1400 AD, älteste besprochene Phase 28,4–31,1 | 113–118 / 119–124; §3.3.5 | 111–116 / 42–47 | Kandidat N: 512–30925 a BP. Text-/Dateigrenzen erst nach Laufbindung angleichen. |
| Ohrid | ausdrücklich 1165–1365, Frühes Pleistozän | 119–123 / 125–129; §3.3.6 | 116–122 / 47–53 | Kandidat N: 1165841–1366341 a BP; kein Holozänfall und kein automatisch identisches Auswahlfenster. |

Prespa wird in §3.2 als Methoden-/Nachbearbeitungsbeispiel behandelt und ist
kein zusätzlicher §3.3-Fall dieser Matrix. Monticchio gehört ebenfalls nicht in
diese Liste; sein eigener Chronologie-Gate bleibt unverändert offen.

## 2. Fossildaten, Chronologie und benötigte Taxa

Alle CSV-Zeilen-/Spaltenzahlen sind neu lesend geprüft, keine Zahlen aus einem
Training. Vollständige Kandidaten-Spaltennamen, Literal-Tiefen und Hashes stehen
fallweise in N. Leere Spalten sowie AP/NAP/Prozentsummen sind nicht als Taxa gezählt.
Die endgültige nach Auswahl und Zuordnung benötigte TF-Menge ist **für alle Fälle
noch nicht belegt**. Auch passende Zeilenzahlen beweisen weder Tiefenidentität
noch dieselbe Proben- oder Pollentypauswahl.

| Fall | Fossil-/Polleneingabe und Provenienz | Chronologie und konkrete Grenze | Taxa/Zuordnung im erhaltenen Code |
| --- | --- | --- | --- |
| Totes Meer Holozän | **belegt:** Litt et al. 2012, En-Gedi-Kern, 21 m. `additional:my_gedi/data/in/core/en_gedi_counts.csv`: 58 Zeilen, 14 Taxonspalten; Tiefe im Header cm. D S.73–74; C Z.120–140. | **belegt:** 20 Radiokarbondaten, Bacon, Hiatus um 16 m (D S.74). Datei `my_gedi/data/out/age_depth/rho_depth_given_age.rdata`: Matrix 57×1949, 57 Alters-/Tiefeneinträge, 405–10126 a BP. **widersprüchlich:** 58 Pollenzeilen, 57 Chronologieeinträge und fehlende rezente Schicht gegenüber D. Auswahl-/Ausschlussbindung fehlt; nicht still erste Probe entfernen. | **belegt:** Kandidaten u.a. Quercus ithaburensis/calliprinos, Pistacia lentiscus, Olea europaea, Phillyrea, Pinus halepensis, Artemisia, Chenopodium. C Z.133–139 setzt Namen/Aliaslisten anhand der geladenen Basistabelle, u.a. Position 13 `Atriplex Halimus`. **fehlt:** dafür passende geordnete historische Basistabelle und numerische TFs. |
| Totes Meer Glazial | **belegt:** Chen & Litt 2018 und Miebach et al. 2019, ICDP 5017-1-A. `my_gedi2/data/in/core/taxa_spectrum.csv`: 422 Zeilen, 19 Kandidatenspalten. D S.82; C Z.185–227. P: PANGAEA.900564 ist nur ein Teil. | **belegt:** jüngere 14C-Daten, U/Th und LR04-Anker oberhalb etwa 120 ka (D S.82). **fehlt:** exakt aktiv geladene `my_gedi2/data/in/age_depth/rho_depth_given_age_higher_resol.rdata` (C Z.221). Gleichnamige Datei ohne `_higher_resol` vorhanden; kein Ersatznachweis. Tiefeneinheit/Transformation am Lauf binden. | **belegt:** C Z.194–200 nennt u.a. `Alnus glutinosa`; `check_ages.R:103–113` entfernt aktiv Betula, Thalictrum minus, Cedrus, Tilia, Alnus, Artemisia, Pinus halepensis, Filipendula und Fraxinus ornus. **fehlt:** Prüfung dieser Auswahl an der tatsächlichen Namensliste/TF-Reihenfolge; CSV-Kandidaten sind keine fertige 19-Taxon-Rekonstruktion. |
| Kinneret | **belegt:** Schiebel & Litt 2018 plus persönlich übermittelte Ergänzungen (Litt/Miebach), KI10-Komposit 17,8 m; D S.90–92. `my_kinneret/data/in/core/pollen_counts_epd_taxanomy.csv`: 160 Zeilen, 60 Kandidatenspalten, Header cm, erste/letzte Tiefe 1/1772. | **belegt:** terrestrische Pflanzen, Holzkohle, reservoirkorrigiertes Bulk-14C nach Miebach et al. 2022. `my_kinneret/data/out/age_depth/rho_depth_given_age.rdata`: Matrix 160×1772, 160 Alters-/Tiefeneinträge, −60–9015 a BP. **fehlt:** Bindung dieser Datei an den geladenen Dissertations-Basiseintrag, Auswahlregeln und endgültiges Altersraster. | **belegt:** C Z.143–154 liest aus der CSV nur die Baumreferenz; Taxonspektrum und Taxonnamen werden aus dem fehlenden Basiscontainer übernommen. Position 5 wird nacheinander `Atriplex Halimus` und `Olea Europaea`, Position 6 `Phillyrea Latifolia`, Position 8 `Pistacia Lentiscus`. **widersprüchlich:** diese Indizes sind keine Zuordnung zur vorliegenden 60-Spalten-CSV. **fehlt:** geordnete finale Pollen→TF-Tabelle. |
| Birkat Ram | **belegt:** Schwab et al. 2004 / Schiebel 2013, 543-cm- und 10,96-m-Komposite. `my_birkat_ram/data/in/core/pollen_counts_composite.csv`: 161 Zeilen, 14 Kandidatenspalten. D S.98–100. | **belegt:** magnetische Verknüpfung 516/527,7 cm, jüngerer Kurzkern bis ca. 6250 a BP, geringe Sedimentation 10–17 ka. **widersprüchlich:** aktiv geladene `data/out/age_depth/rho_depth_given_age.rdata` enthält drei Listenvarianten mit Matrizen 140×540, 174×540, 18×540; Altersenden 8675/8650/8500 a BP, Tiefe nur 2,5–542. Das belegt nicht das 30-ka-Komposit. Hiatusvarianten sind Katalogkandidaten, nicht ausgewählt. | **belegt:** C Z.229–261, Slot 8 statt jüngerer Slot 4; Olea-Aliasliste, Pistacia/Phillyrea umbenannt, Position 12 `Chenopodium`, Division durch AP+NAP. `mcmc_prepare.R:24–34` glättet Spalten 144–146 per loess. **fehlt:** gültige Komposit-Chronologie und konsistente Spalten-/TF-Bindung. Kein neuer Glättungsfit ausgeführt. |
| Van | **belegt:** VAN04-2 nach Litt et al. 2009 und Ahlat Ridge nach Pickarski et al. 2015a/b. `my_van/data/in/core/taxa_spectrum.csv`: 350 Zeilen, 21 Kandidatenspalten. Verbindung bei etwa 20 ka; D S.105–107. | **belegt:** gemeinsame Varvenchronologie, 14C, Tephren, NGRIP/GLTsyn-SpeleoAge-Tuning; fehlende Fehler historisch durch mittlere benachbarte Unsicherheit ersetzt; Bacon-Eingaben bis ca. 260 ka (D S.107). **fehlt:** exakt geladene `my_van/data/in/age_depth/rho_depth_given_age.rdata` (C Z.288). Die 2023er 0–600-ka-Datei im Katalog ist kein belegter Ersatz der Dissertationsfassung. | **belegt:** C Z.264–290 setzt Position 12 auf `Pinus nigra`; AP aus den Spalten 3,5,13,15,16,9 und nicht aus allen Bäumen. Pinus bleibt im Spektrum, wird aus AP entfernt. **fehlt:** finale Arten-/Gattungszuordnung und historische numerische TFs für den tatsächlich verwendeten Ausschnitt. |
| Iznik | **belegt:** Miebach et al. 2016, mehrere Kerne/Komposit. Vorhanden `my_iznik/data/in/core/pollen_counts_epd_taxanomy.csv`: 111 Zeilen, 106 Kandidatenspalten, Header m. **fehlt:** C Z.161 lädt stattdessen `core_information.csv`; diese Datei existiert am Verbraucherpfad nicht. | **belegt:** Roeser et al. 2016, zwei Tephren und reservoirkorrigierte/unkorrigierte 14C-Daten. Vorhandene `my_iznik/data/out/age_depth/rho_depth_given_age.rdata`: Matrix 111×1764, 111 Einträge, 512–30925 a BP; Tiefenwerte 50,8–1814 gegenüber CSV 0,508–18,14. **fehlt:** dokumentierte m→cm-/Probenbindung an die fehlende konsumierte CSV; gleiche Zeilenzahl reicht nicht. | **belegt:** C Z.163–174: Division durch `Pollen_sum`, Fagus orientalis/sylvatica, Pinus sylvestris/nigra, Ulmus/Zelkova als Aliaslisten sowie Juglans regia. **widersprüchlich:** deren Positionsindizes dürfen nicht auf die vorhandene 106-Spalten-CSV übertragen werden. **fehlt:** konsumierte historische Spaltenliste, numerische TFs und Zuordnung. |
| Ohrid | **belegt:** Panagiotopoulos et al. 2020, DEEP-Komposit / ICDP 5045-1. `my_ohrid/data/in/core/core_data_counts.csv`: 103 Zeilen, 46 Kandidatenspalten. D S.119–121. `core_data_ohrid.rdata` zusätzlich vorhanden, noch keine automatische Basisdateisubstitution. | **belegt:** Tephren, Orbitaltuning, zwei paläomagnetische Umkehrungen nach Wagner et al. 2019. Vorhandene `data/out/age_depth/rho_depth_given_age.rdata`: Matrix 103×4673, 103 Einträge; 1165841–1366341 a BP, Tiefe 40056–44728 gegenüber CSV 400,56–447,28. **widersprüchlich:** Endpunkte passen nicht exakt zum Textfenster; Einheit/Filter/Chronologievariante bleibt offen. | **belegt:** Relikttaxa Cedrus libani, Liquidambar (D schreibt „orientalist“), Pterocarya fraxinifolia, Zelkova carpinifolia, Carya, Tsuga; Vergleich mit/ohne diese Gruppe. F belegt ausschließlich für Ohrid den 283→279-Pfad. Picea omorika ohne numerische Funktion nicht durch anderes Picea ersetzen. **fehlt:** vollständige Fossiltyp→Art-/TF-Bindung für beide Varianten, teilweise persönlich kommuniziert. |

## 3. Transferfunktionen und gemeinsame numerische Abhängigkeiten

**belegt:** D §2.5 beschreibt binäre historische Taxonfunktionen mit TANN und PANN,
CRU TS4.01, Referenz 1961–1990. C lädt aktiv
`out/transfer_functions/mixed_partial_arts_emd.rdata` (Z.569), übernimmt aus der
ersten Funktion `x1range.bck`/`x2range.bck` und taxonweise `prod.x1`/`prod.x2`
(Z.570–590), jeweils maximal auf eins normiert. Zu jedem tatsächlich behaltenen
Taxon werden beide numerischen Marginalen, gültige Achsen und derselbe geordnete
Namensindex benötigt. **fehlt:** dieser Container und seine Info-Datei sowie die
vollständige Lauf-/Taxonbindung. Der historische 279er-Stand ist kein Nachweis,
dass jeder lokale Fall 279 Taxa benötigt.

F belegt zwei grafische Sammlungen mit je 278 zugeordneten Taxontiteln und die
Auslassung von Picea omorika. Diese Grafiken sind keine numerischen Marginalarrays.
Die 167 strukturell gelesenen caret-Fits aus 78 verschiedenen Byteinhalten
ersetzen den fehlenden TF-Container ebenfalls nicht. Die sieben modernen
Artenmodelle bleiben experimentelle Vergleichsmodelle; keine Substitution.

**belegt, Quellstand:** `all_trafo.R:45–52` überschreibt CRU-Objekte durch
`tann_test.nc`/`pann_test.nc`, mittelt Indizes 61:90 (Z.106–113), standardisiert
Temperatur und transformiert Niederschlag über Wurzel/Gamma-CDF/Normalquantil
(Z.217–229). 300 Knoten je transformierter Achse, Temperatur zeilenweise,
Niederschlag spaltenweise, rücktransformiertes Niederschlagsgitter im Allgemeinen
nicht äquidistant (Z.407–411,469–529). Das bleibt historischer Rechenweg; keine
nachträgliche Quadratur-/Normierungskorrektur. **fehlt:** originale globale
Klimadateien mit Zeitachse; 61:90 beweist ohne sie keine Kalenderjahre.

## 4. Klima, Referenzen und Anker je Fall

Für **alle sieben Fälle belegt:** Ziel TANN in °C, PANN in mm/a; Alter in cal a BP
bzw. Darstellung cal ka BP; Pollenreferenzen in %, Isotope δ18O in ‰ VPDB,
Wavelet-Leistung als abgeleitete Spektralinformation. Isotope sind keine direkt
in °C/mm umgerechneten Messwerte. TANN niemals mit TDJF des Paperfalls verwechseln.

**fehlt für alle:** exakte konsumierte globale CRU-Arrays, Zeitachsen und vollständige
Referenz-/Ankerparameter. C Z.325–347 lädt `cru_tann_ww.nc`/`cru_pann_ww.nc`
sowie `cru4_tann_v1.2.nc`/`cru4_pann_v1.2.nc`; Kommentar „1950–2015“ zu 50:115
ist ohne Zeitachse kein Nachweis. Erhaltene regionale Mittel 1901–2015 und
1961–1990 (F) werden nicht als identische globale Eingaben eingesetzt.
`in/kgc_distribution_ww.rdata` ist vorhanden (P), aber numerische
`prob_c_kgc.rdata` und konsumierte Isotopencontainer fehlen am geprüften Pfad (N).
Moderne WorldClim-Dateien werden nicht eingesetzt.

| Fall | Historische Referenzkurven und Fenster | Klimaanker / konkrete Grenzen |
| --- | --- | --- |
| Totes Meer Holozän | **belegt:** AP/NAP→PANN nur älter als 6,5 ka; Soreq nach Bar-Matthews et al. 2003, Korrelation der WPS→PANN über den gesamten Holozänfall. Kein MEDSTACK-Kanal im beschriebenen Hauptfall. D S.73–74, Abb.3.20. | **belegt:** rezenter BSh-Anker; nächster Rasterpunkt etwa 20 °C / 230 mm/a ist eine Klimabeschreibung, kein einzelner Priorparameter. **fehlt:** Verteilungsparameter, konkrete Zellen und Bindung der jüngsten Probe. |
| Totes Meer Glazial | **belegt:** MEDSTACK→TANN im Zeitbereich; Soreq nach Grant et al. 2012, WPS→PANN. AP ausdrücklich kein Klimareferenzkanal. D S.81–82, Abb.3.25. | **belegt:** keine Klimaanker für 14,5–147 ka. LR04/U-Th sind Chronologieanker, keine Temperatur-/Niederschlagspriors. **fehlt:** exakte Referenzvorverarbeitung und Laufkonfiguration ohne rezente Anker. |
| Kinneret | **belegt:** MEDSTACK→TANN; AP/NAP→PANN nur vor 6,5 ka; Soreq-WPS→PANN über den Fall. D S.90–92, Abb.3.30. δ13C und Seespiegel dienen der Interpretation, kein zusätzlich belegter Modellkanal. | **belegt:** jüngster Csa-Anker; Standortdiagramm 18,5 °C / 472,1 mm/a (D Abb.3.29). **fehlt:** numerische Ankerverteilung und konsumierte Referenzcontainer. Keine Paperanker übernehmen. |
| Birkat Ram | **belegt:** gleiche drei Referenzfamilien wie Kinneret, entsprechende Levante-Maske nach 6,5 ka; geringe Sedimentation 10–17 ka beachten. D S.99–100, Abb.3.35. **fehlt:** exakt verwendeter Soreq-Ausschnitt über das gesamte 30-ka-Fenster. | **belegt:** moderner Klimakontext Csa, 16,1 °C / 623,4 mm/a; jüngerer Anker folgt dem beschriebenen Kinneret-Verfahren. **fehlt:** konkrete Priorparameter und Verbindung mit dem langen Komposit. |
| Van | **belegt:** MEDSTACK→TANN; AP ohne Pinus→TANN und PANN (warm/feucht), gemeinsame Chronologie beachten. D S.106–107, Abb.3.39. C setzt anthropogenen Bereich 0–4000 a BP (Z.286). | **belegt:** moderner Dsb-Kontext 8,5 °C / 551,7 mm/a. Ein rezent erreichender Fall kann den jüngsten Anker verwenden; **fehlt:** verifizierte aktive Anker-/AP-Maske des historischen Laufs, kein zusätzlicher LGM-Anker allein aus dem Standortdiagramm ableiten. |
| Iznik | **belegt:** MEDSTACK sowie AP der gemäßigten Bäume/Sträucher wie bei Van; C Maske 0–4800 a BP. D S.113–115, Abb.3.44. | **belegt:** Csb am jüngsten Horizont trotz ca.1400 AD; zusätzlicher PANN-Anker bei 21 ka aus zentralanatolischem BSk-Klima. **widersprüchlich:** `past_climate_module <- F` im erhaltenen Einstieg C Z.670; Vorbereitung hat BSk+Dsb im Fenster 35–40°E/36–45°N (Z.318–327), nicht nur BSk. Hauptfall/Varianten nicht unbemerkt vertauschen. **fehlt:** numerische LGM-PDF und eindeutig zugeordnete Einstellung. |
| Ohrid | **belegt:** MEDSTACK und gemäßigte AP ohne Pinus→TANN, D S.120–121, Abb.3.48; kein belegter Soreq-Kanal. Niederschlag folgt Taxon-TFs und Modellkopplung, keine zusätzliche PANN-Referenz erfinden. | **belegt:** keine Klimaanker; heutiges Csb-Diagramm 9,2 °C / 791,3 mm/a ist Kontext. Beide Relikttaxonvarianten ohne heutige Anker unterscheiden. **fehlt:** konsumierte MEDSTACK-/AP-Zeitreihen und finale Chronologieauswahl. |

Öffentliche Kandidaten aus P bleiben nachvollziehbar: MEDSTACK
[PANGAEA.790006](https://doi.org/10.1594/PANGAEA.790006),
Kinneret [PANGAEA.884740](https://doi.org/10.1594/PANGAEA.884740),
Totes Meer jüngerer Glazialteil [PANGAEA.900564](https://doi.org/10.1594/PANGAEA.900564),
Iznik [PANGAEA.858056](https://doi.org/10.1594/PANGAEA.858056),
Van [PANGAEA.874345](https://doi.org/10.1594/PANGAEA.874345),
[853776](https://doi.org/10.1594/PANGAEA.853776),
[853729](https://doi.org/10.1594/PANGAEA.853729),
Ohrid [PANGAEA.896848](https://doi.org/10.1594/PANGAEA.896848).
Das sind übernommene Bezüge des abgeschlossenen Quellenverzeichnisses, keine
neuen Downloads oder neu behauptete Identitätsnachweise.

## 5. Historische Codefassung, Rechenweg und Nachbearbeitung

**belegt als Textverfahren M, für jeden Fall zunächst zu erhalten:**

1. Datierungen/Bacon und diskrete Tiefen-Alters-Transformation, Taxon-/Pollentyp-
   Zuordnung sowie Referenzfenster festhalten. C `check_ages.R:174–231` wählt
   Altersvarianten, filtert Tiefen, behandelt Duplikate; ein RData-Matrixfund allein
   schließt diesen Vertrag nicht. `mcmc_prepare.R:11–34` normiert und ersetzt
   Nullen; Birkat erhält historische zusätzliche Glättung.
2. Historische Taxonfunktionen mit gezogenen Taxongewichten verbinden.
   C `parameter_stage_weights` und letzte `reconstruction`-Definition sowie
   `final_reconstruction` belegen gewichtete Summen/Mischungen, kein still
   angenommenes Produkt aller Taxondichten (P; C Z.1242–1284,1781–1820,450–477).
3. Proxy-Pool und Klimaanker nach den fallbezogenen Fenstern verwenden.
   D §3.2.1 nennt MG 12500/rwMH 50000 Iterationen, zehn Ketten und multivariates
   Gelman-Rubin <1,1. Das sind Textangaben; Seed, tatsächlich verwendete Ketten,
   Burn-in, Pakete und Zahlen je Originalabbildung sind **nicht laufgebunden**.
4. M beschreibt Auswahl in Prior-IQR, Ausdünnung bis IAT <1,3, Pareto-Auswahl
   der zehn den Prior-Mittelwerten nächsten Ziehungen. Anschließend Dichten
   durch Faltung zum Mittel der Zufallsgrößen kombinieren, Faktor 1/N, für alle
   folgenden lokalen Rekonstruktionen. Diese Reihenfolge nicht durch volle
   Posterior-Auswertung oder andere Kombination ersetzen.

**widersprüchlich/offen:** Der erhaltene Haupteinstieg setzt zwar
`averaging_data <- T` (C Z.74–78), ruft jedoch eine Bearbeitungsfassung auf,
deren aktive Z.96–103 nur ausdünnen und nach auf fünf Dezimalen gerundeten
Kriterien deduplizieren; der anschließende Trimmingblock Z.105–113 ist
kommentiert. Sie enthält zudem aktive `stop()`-Marken. Die Speicherzeilen für
Posterior-Dateien in `mcmc_conduction.R:427–463` sind kommentiert.
Das ist kein durchgängig ausführbarer, nachgewiesener Dissertationslauf.
Die für jede Abbildung verwendete historische Auswahlfassung muss gefunden
oder als begrenzter textgebundener Nachbau eigens ausgewiesen werden.

C adressiert die Fälle als Slots 2 En Gedi, 3 Kinneret, 7 Dead Sea, 8 Birkat-
Komposit, 9 Van, 5 Iznik und 6 Ohrid; Slot 4 ist der jüngere Birkat-Fall und im
aktuellen Stationsvektor ausgeschlossen. Das belegt historische Rollen, keine
übertragbaren EPD-IDs. Die aktiv geladene Basisdatei
`in/core_data_additional_sites_v1.2.rdata` fehlt (C Z.107, N). Ähnlich benannte
ältere Container werden nicht ohne Struktur-/Quellvergleich eingesetzt.

**Methodische Kritik bleibt getrennt:** Die frühere Bewertung von Prior-Auswahl,
Faltung, Hastings-Korrektur, SMOTE und TF-Gitter bleibt in
[Dissertationsabgleich](../../abgleich_thesis.md) und F bestehen. Hier ist keine
Korrektur, neue Kalibrierung oder neue Diagnose implementiert. Quellkonflikte
oben sind Reproduktionshindernisse, keine Erlaubnis, eine bessere Methode
in den historischen Pfad einzubauen.

## 6. Gespeicherte Originalausgaben und Vergleichsgrenzen

N vergleicht jeweils nur zwei ausdrücklich genannte Dateien mit dem Mitglied
des bereits bekannten `diss_final.zip`; 11 von 14 Paaren sind **byteidentisch**.
Damit ist ihre Bindung an das lokale Dissertationsarchiv belegt. Es ist kein
Beweis, dass die PDFs im danebenliegenden D-PDF ohne Veränderung eingebettet
wurden, oder dass zugrunde liegende numerische Posteriorarrays erhalten sind.

| Fall | Gespeicherter Pfad unter `original:out/plots/case_studies/` | Archivmitglied unter `figures/` | Status und Vergleichsziel |
| --- | --- | --- | --- |
| Totes Meer Holozän | `gedi/reconstruction.pdf`, `gedi/reference_curves.pdf` | `gedi/` jeweils gleicher Dateiname | **belegt:** beide vorhanden; Referenz bytegleich. **widersprüchlich:** Rekonstruktions-PDF nicht bytegleich zum Archiv. Archivfassung für Zielvergleich gesondert halten; kein Beweis numerisch verschiedener Kurven allein aus Hashunterschied. |
| Totes Meer Glazial | `gedi2/reconstruction.pdf`, `gedi2/reference_curves.pdf` | `dead_sea/` jeweils gleicher Dateiname | **belegt:** alle vorhanden. **widersprüchlich:** beide Paarvergleiche nicht bytegleich. Keine ungeprüfte Identität der Ausgabevarianten. |
| Kinneret | `kinneret/reconstruction.pdf`, `kinneret/reference_curves.pdf` | `kinneret/` | **belegt:** beide bytegleich. Später grafisches Ziel D Abb.3.31 / E Abb.3.30. |
| Birkat Ram | `birkat2/reconstruction.pdf`, `birkat2/reference_curves.pdf` | `birkat/` | **belegt:** beide bytegleich. `birkat/` im Originalbestand enthält zusätzlich einen anderen jüngeren Fall; nicht verwechseln. Ziel D Abb.3.36 / E Abb.3.35. |
| Van | `van/reconstruction.pdf`, `van/reference_curves.pdf` | `van/` | **belegt:** beide bytegleich. Ziel D Abb.3.40 / E Abb.3.39. |
| Iznik | `iznik/reconstruction.pdf`, `iznik/reference_curves.pdf` | `iznik/` | **belegt:** beide bytegleich. `iznik_iso/`, `reconstruction_old.pdf` sind andere Kandidaten. Ziel D Abb.3.45 / E Abb.3.44. |
| Ohrid | `ohrid/reconstruction.pdf`, `ohrid/reference_curves.pdf` | `ohrid/` | **belegt:** beide bytegleich. Ziel D Abb.3.49 / E Abb.3.48; Varianten mit/ohne Relikttaxa noch numerisch zu binden. |

**fehlt für alle sieben:** vollständig gebundene numerische Originalausgabe
inklusive Ketten, ausgewählter zehn Ziehungen, endgültiger Dichte-/Quantilarrays
und ihrer Eingabe-/Codekonfiguration. Die vorhandenen Kataloge weisen lokale
Fallgrafiken nach, keinen damit geschlossenen Zahlenvergleich. Modellcontainer
und Paper-Posteriorarchive sind keine lokalen Dissertationsposterioren.
Der [bereits berechnete Paper-Biomfall](../LOCAL_REFERENCE_RESULT.md) bleibt
technische Referenz, keine vollständige taxonbasierte Dissertationsreproduktion.

## 7. Ergebnis, entscheidende Lücken und nächster Einstieg

**Noch kein vollständig belegbar reproduzierbarer Fall.** Ein fehlender
numerischer TF-Vertrag verhindert jeden der sieben Läufe; auch ein Fall ohne
Klimaanker benötigt Transferfunktionen und passende Referenzen/Chronologie.
Deshalb ist Ohrid oder der glaziale Tote-Meer-Fall keine Abkürzung.

**Als nächstes Kinneret gezielt erschließen:**

1. Den konkret geladenen Eintrag `core.data[[3]]` aus der fehlenden
   `core_data_additional_sites_v1.2.rdata` an den schon vorhandenen lokalen
   Kinneret-Einzelcontainer und seine Erzeuger binden. Nur diese gezielten
   Kandidaten prüfen: `my_kinneret/data/in/core/core_data.rdata`,
   `my_kinneret/data/out/age_depth/rho_depth_given_age.rdata` und die zugehörigen
   bereits katalogisierten Vorbereitungsskripte. Ergebnis muss eine geordnete
   Tabelle Probe/Tiefe/Alter/Pollentyp/historischer TF-Name/-Index samt Ausschlüssen
   sein. Die 60 Rohspalten nicht als 60 verwendete Arten ausgeben.
2. Erst daraus die **tatsächlich für Kinneret benötigte TF-Teilmenge** formulieren
   und den konkreten numerischen historischen Container samt Info/Namensfolge
   verfolgen. Vorherige negative Gesamt-/ZIP-Prüfungen weiterverwenden.
   Falls keine neue Quelle existiert, die konkret benötigten Originaldateien
   benennen; keine Approximation aus Plots und kein Neutraining starten.
3. Soreq-/MEDSTACK-/AP-Verbraucher, Csa-Anker inklusive CRU-Zeitachse und die
   zur bytegebundenen Kinneret-Grafik gehörige Nachbearbeitungsfassung schließen.
   Numerische Originalausgaben/ausgewählte Ziehungen suchen, damit zunächst
   Zahlen und erst danach grafische Darstellungen verglichen werden können.

Kinneret wird wegen der vorhandenen zusammenpassenden Eingabedimensionen und
gebundenen Archivgrafiken priorisiert, nicht wegen eines günstigen
Rekonstruktionsergebnisses. Dieser Vorschlag benötigt jetzt keine fachliche
Ersatzmodellentscheidung und erteilt keine Lauf-/Pilotfreigabe.

**Abgeschlossen:** quellengebundene Matrix, Fassungs-/Fensterprüfung,
gezielte Metadaten-/Chronologieprüfung, Archivgrafikbindung und Einstieg.
**Offen:** historischer Datenvertrag und numerische Reproduktion. Phase 2
bleibt nachgeordnet. Keine Trainings, Sampler, Fossilrekonstruktionen, neue
Kalibrierung oder Reliabilitätsansicht. Baseline, Referenzdaten, Originale und
Monticchio-/Chronologie-Gates sind erhalten; kein Merge.

## Prüfung und Wiederholung des Nachweises

[collect_evidence.py](collect_evidence.py) verwendet ausschließlich die expliziten
Pfade, liest RData mit dem bereits installierten `rdata` und prüft die PDF-Seiten
mit dem vorhandenen MuPDF-Runtime. Es führt kein R aus und enthält keine
Verzeichnisdurchläufe. Aufruf aus diesem Checkout:

```powershell
.venv/Scripts/python.exe docs/thesis_research/local_case_matrix_2026_11_02/collect_evidence.py
```

N hält Dateihashes vor/nach dem Lesen, Containerdimensionen, Altersbereiche,
CSV-Kandidatennamen und Archivpaarvergleiche fest; keine Rohzählwerte, vollständigen
RData-/PDF-Dateien oder persönlichen absoluten Pfade werden veröffentlicht.
[verification.json](verification.json) dokumentiert die abgeschlossenen Prüfungen.
