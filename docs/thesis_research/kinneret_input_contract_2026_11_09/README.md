# Kinneret: Eingabevertrag des Falls (Slot 3, `add.sites <- T`)

Stand 09.11.2026, Ausgangscommit `ad0882a`, bestehender Branch `dissertation-reconstruction`, [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).

**Abgeschlossen:** Für den erhaltenen Nachbaupfad steht jetzt maschinenprüfbar, welche Eingaben der Kinneret-Fall braucht, wo der Code sie liest und wie weit jede gebunden ist: 48 Eingaben (5 gebunden, 10 teilweise gebunden, 6 Quellkonflikte, 27 fehlend) und 18 Bedingungen.

**Kein Originalfall ist reproduziert, kein Nachweisbündel ist geschlossen, keine Variante gewählt.** „Gebunden“ heißt hier: am Verbraucherpfad vorhanden oder als Literal im Code. Es heißt nie: an den historischen Lauf gebunden.

Das Paket baut auf dem [Phase-1-Belegvertrag](../kinneret_phase1_spec_2026_11_02/README.md), der [Proben- und Taxonbindung](../kinneret_binding_2026_11_02/README.md), dem [Referenzvertrag](../kinneret_references_2026_11_02/README.md), der [Nachbearbeitung](../kinneret_postprocessing_2026_11_02/README.md) und den [Laufzeitbelegen](../r_runtime_crosscheck_2026_11_03/README.md) auf. Es ersetzt keines davon und ändert keines.

- Maschinenlesbar: [input_contract.json](input_contract.json), erzeugt von [bind_input_contract.py](bind_input_contract.py). Das Skript prüft die sechs aktiven Quellen per Hash und jeden Zeilenbezug gegen den Zeilenanfang und bricht bei einer Abweichung ab.
- Tests: `tests/test_kinneret_input_contract.py`. Sie halten Stand und Zeilenbezüge jeder Eingabe fest.
- `diss` wurde nur gelesen. Keine R-Skripte, keine Ziehung, kein Sampler, kein Training.

## Was dabei neu belegt wurde

1. **Standortname: Quellkonflikt mit Folgen.** Der ältere Container nennt den Standort `Kinneret`, alle Schalter in `mcmc_prepare.R` prüfen `Lake Kinneret`, und `main_v1.2.R` setzt den Namen für Slot 3 nicht. Der Wert im finalen Container ist unbekannt. Er entscheidet über den Levante-Zweig in Zeile 1209: Unter jedem anderen Namen laufen mit den älteren Koordinaten `multiple <- FALSE` und `sources[4] <- 0`, die marine Reihe fällt also weg. Der Dissertationstext nennt drei Referenzfamilien, was zu `Lake Kinneret` passt; das bindet den Wert nicht.
2. **`inside.giesecke` ist entscheidbar.** Die Giesecke-Dateien liegen am Verbraucherpfad. Das Gitter endet im Süden bei 34,75° N mit einer einzigen Zelle bei 24,75° E, also ist die Raumbedingung für 32,79° N und 35,53° E falsch. Damit wird Zeile 636 erreicht und `sources$core` gesetzt. Das ist die einzige Auswertung von Quellwerten in diesem Paket: ein Min/Max-Vergleich auf zwei Spalten, abhängig von den älteren Koordinaten, kein Modellschritt.
3. **`climatezones` kommt nicht aus der daneben geladenen Datei.** `in/kgc_distribution_ww.rdata` enthält nur `kgc.distribution.ww`, das im aktiven Pfad niemand liest. Der Name `climatezones` wird in keiner der 95 R-Dateien von `local_reconstr_all_epd` zugewiesen.
4. **`d18o.age` hat in `local_reconstr_all_epd` keinen Erzeuger.** Es kommt nur aus einem der fehlenden Workspaces. Welcher das ist, ist nicht belegt.
5. **Der Speläothemzweig passt nicht zu einem Holozänkern.** Ist `sources$speleothem` 1, ersetzt `mcmc_prepare.R:825–826` `site.age` durch 418 Werte mit einer Lücke von 2000 bis 14500 a BP, und Zeile 828 setzt Reihe 170. Das bleibt ein offener Quellenbefund und wird nicht repariert.
6. **Aktiver Abbruch.** `mcmc_prepare.R:2057` ist ein aktives `stop()`. Im erhaltenen Quellstand wird `mcmc_conduction.R` also nicht erreicht.

**Präzisierung einer früheren Aussage**, ohne das eingefrorene Paket zu ändern: In den [Laufzeitbelegen](../r_runtime_crosscheck_2026_11_03/README.md) steht, `tann <- FALSE` in Zeile 1248 gelte „in jedem Kinneret-Lauf“. Das setzt den exakten Namen `Lake Kinneret` voraus. Mit den älteren Koordinaten ist `tann` zwar unter beiden Namen FALSE, die übrigen Schalter unterscheiden sich aber (Tabelle unten).

## Stand je Eingabe

Kurznamen: `main` = `main_v1.2.R`, die übrigen = `src/mcmc_*.R`, `src/check_ages.R`, `src/prior_climate.R`. Fehlt ein Pfad, ist das die Prüfung genau dieses Pfads, keine Aussage über alle Datenträger.

### Basiseintrag von Slot 3

| Eingabe | Stand | Gelesen in | Herkunft in `diss` | Warum |
|---|---|---|---|---|
| Finaler Sammelcontainer der Zusatzorte (`core_container`) | fehlend | `main` 107, 146, 654 | `src/local_reconstr_all_epd/in/core_data_additional_sites_v1.2.rdata` (fehlt)<br>`src/local_reconstr_all_epd/in/core_data_all_andrea_taxa_additional_sites.rdata` (vorhanden) | Die Datei fehlt am Verbraucherpfad. Der ältere Sammelcontainer hat vier Standorte und eine andere Struktur. |
| Standortname in Slot 3 (`site_name`) | **Quellkonflikt** | `check_ages` 7; `prepare` 790, 881, 1145, 1209, 1248 | `src/local_reconstr_all_epd/in/core_data_all_andrea_taxa_additional_sites.rdata` (vorhanden)<br>`src/local_reconstr_all_epd/in/core_data_additional_sites_v1.2.rdata` (fehlt) | Der ältere Container nennt „Kinneret“, alle Schalter prüfen „Lake Kinneret“. main_v1.2.R setzt den Namen für Slot 3 nicht. Der Wert des finalen Containers ist unbekannt. |
| Pollenspektrum (Proben × Taxa) (`taxa_spectrum`) | **Quellkonflikt** | `main` 153; `check_ages` 14, 15; `prepare` 83, 93, 1435, 1454 | `src/local_reconstr_all_epd/in/core_data_all_andrea_taxa_additional_sites.rdata` (vorhanden)<br>`src/my_reconst/my_kinneret/data/in/core/pollen_counts_epd_taxanomy.csv` (vorhanden)<br>`src/my_reconst/my_kinneret/data/in/core/core_data.rdata` (vorhanden) | Der ältere Stand hat 160×12 und ist mit der CSV zahlengleich. Seine Spalten 5, 6 und 8 tragen aber andere Namen, als main_v1.2.R dort setzt. Anders als bei Prespa, En Gedi und Ohrid ersetzt der Code die Kinneret-Zahlen nicht aus der CSV. |
| Taxonnamen und Reihenfolge (`taxon_names`) | **Quellkonflikt** | `main` 147, 148, 149, 150, 152; `prepare` 1452 | `src/local_reconstr_all_epd/in/core_data_all_andrea_taxa_additional_sites.rdata` (vorhanden) | main_v1.2.R überschreibt die Positionen 5, 6 und 8. Im älteren Stand stehen dort Pinus cembra, Betula und Vitis sylvestris. Die finale Grundreihenfolge fehlt. |
| Tiefenvektor (`depth`) | teilweise | `check_ages` 178, 195; `prepare` 14 | `src/local_reconstr_all_epd/in/core_data_all_andrea_taxa_additional_sites.rdata` (vorhanden)<br>`src/my_reconst/my_kinneret/data/in/core/pollen_counts_epd_taxanomy.csv` (vorhanden)<br>`src/my_reconst/my_kinneret/data/out/age_depth/rho_depth_given_age.rdata` (vorhanden) | 160 Tiefen stimmen in vier erhaltenen Quellen exakt überein. Der finale Container fehlt. |
| Altersraster age[[2]] (`age_grid`) | **Quellkonflikt** | `main` 656; `check_ages` 179, 223; `prepare` 728 | `src/my_reconst/my_kinneret/data/out/age_depth/rho_depth_given_age.rdata` (vorhanden)<br>`src/my_reconst/my_kinneret/src/bacon_kinneret.R` (vorhanden) | Der Verbraucher erwartet eine Liste. bacon_kinneret.R schreibt zuerst eine Dreierliste, deren zweites Element das 50-Jahres-Raster ist, und überschreibt denselben Dateinamen später mit einem Mittelwertvektor. |
| Dichte Tiefe gegeben Alter, rho[[2]] (`rho_depth_given_age`) | **Quellkonflikt** | `check_ages` 177; `prepare` 14, 648 | `src/my_reconst/my_kinneret/data/out/age_depth/rho_depth_given_age.rdata` (vorhanden)<br>`src/my_reconst/my_kinneret/src/bacon_kinneret.R` (vorhanden) | Erhalten ist eine 160×1772-Matrix. Der Verbraucher erwartet das zweite Listenelement, einen anderen Datentyp. |
| Koordinaten lat/lon (`coordinates`) | teilweise | `prior_climate` 9, 10; `prepare` 568, 755, 1209 | `src/local_reconstr_all_epd/in/core_data_all_andrea_taxa_additional_sites.rdata` (vorhanden) | Der ältere Container hat 32,79° N und 35,53° E. Der Wert des finalen Containers ist nicht belegt. |
| Land und Höhe (`country_elevation`) | teilweise | `prepare` 1036, 1080 | `src/local_reconstr_all_epd/in/core_data_all_andrea_taxa_additional_sites.rdata` (vorhanden) | Der ältere Container hat Israel und −210. Beide Felder steuern im Slot-3-Pfad keinen Rechenwert, müssen aber vorhanden sein. |
| AP-Referenz (tree_percentage) (`ap_reference`) | gebunden | `main` 143, 144; `prepare` 639, 648 | `src/my_reconst/my_kinneret/data/in/core/pollen_counts_epd_taxanomy.csv` (vorhanden) | Die CSV liegt am passenden Pfad und wird vom Code selbst in Slot 3 geschrieben, unabhängig vom fehlenden Container. Die Alterskurve hängt zusätzlich an rho[[2]]. |
| Fenster menschlicher Einfluss (`human_impact`) | gebunden | `main` 154; `prepare` 711, 716, 878 | – | Das Literal c(-60,6500) steht in main_v1.2.R. Offen bleibt nur die Randwertregel von inside.range. |

### Transferfunktionen

| Eingabe | Stand | Gelesen in | Herkunft in `diss` | Warum |
|---|---|---|---|---|
| Numerische Transferfunktionen (`tf_container`) | fehlend | `main` 569, 577, 578, 588, 589 | `src/local_reconstr_all_epd/out/transfer_functions/mixed_partial_arts_emd.rdata` (fehlt)<br>`src/local_reconstr_all_epd/src/all_trafo.R` (vorhanden) | Die Datei fehlt. Nur die Schreibzeile in all_trafo.R ist erhalten. Die sieben modernen Artenmodelle sind kein Ersatz. |
| Klimaachsen x1range.bck, x2range.bck (`tf_axes`) | fehlend | `main` 570, 571; `prepare` 1445, 1446, 1934; `prior_climate` 159 | `src/local_reconstr_all_epd/out/transfer_functions/mixed_partial_arts_emd.rdata` (fehlt) | Die Achsen stehen im fehlenden Container. |
| Namenspool taxa.distribution.ww$names (`taxa_name_pool`) | teilweise | `main` 433, 434, 435, 436; `prepare` 1460, 1573 | `src/my_reconst/my_ohrid/data/in/climate_data/taxa_distribution_ww.rdata` (vorhanden)<br>`src/my_reconst/my_kinneret/data/in/climate_data/taxa_distribution_ww.rdata` (vorhanden) | Eine Datei mit passendem Pfadsuffix liegt vor. Gleichheit mit dem verlorenen Automount-Original ist nicht belegt, und die Zuordnung Name zu TF-Index braucht den fehlenden Container. Die Kinneret-Kopie ist ein anderer Stand. |
| tf.x1.current, tf.x2.current (abgeleitet) (`tf_current`) | fehlend | `prepare` 1574, 1575, 1535, 1581, 1582, 1631, 1664, 1897, 1899; `conduction` 146 | `src/local_reconstr_all_epd/out/transfer_functions/mixed_partial_arts_emd.rdata` (fehlt) | Abgeleitet aus dem fehlenden Container. Der Nachbau hat dafür nur Kunstwerte. |

### Klima

| Eingabe | Stand | Gelesen in | Herkunft in `diss` | Warum |
|---|---|---|---|---|
| CRU-PANN global, Stationsmaske (`cru_pann_ww`) | fehlend | `main` 325, 327, 329; `prior_climate` 6, 156 | `src/cru_nh/cru_pann_ww.nc` (fehlt) | Die Datei fehlt. Regionale Ausschnitte sind kein Ersatz. |
| CRU-TANN global (`cru_tann_ww`) | fehlend | `main` 326, 330; `prior_climate` 154 | `src/cru_nh/cru_tann_ww.nc` (fehlt) | Die Datei fehlt. Regionale Ausschnitte sind kein Ersatz. |
| CRU4-PANN Zeitreihe 50:115 (`cru4_pann`) | fehlend | `main` 343, 346; `prior_climate` 71 | `src/cru_nh/cru4_pann_v1.2.nc` (fehlt) | Die Datei fehlt. Ohne Zeitachse ist 50:115 kein belegter Kalenderbezug. |
| CRU4-TANN Zeitreihe 50:115 (`cru4_tann`) | fehlend | `main` 344, 347; `prior_climate` 70 | `src/cru_nh/cru4_tann_v1.2.nc` (fehlt) | Die Datei fehlt. Ohne Zeitachse ist 50:115 kein belegter Kalenderbezug. |
| Köppen-Geiger-Tabelle climatezones (`climatezones`) | fehlend | `main` 360; `prior_climate` 33, 39, 52 | – | Der Code lädt climatezones nicht aus der Datei daneben: Sie enthält nur kgc.distribution.ww. In den R-Quellen wird der Name nirgends zugewiesen. Als Anbieter kommt das Paket kgc infrage; dessen Fassung ist ungebunden und wurde hier nicht geprüft. |

### Isotope und Referenzen

| Eingabe | Stand | Gelesen in | Herkunft in `diss` | Warum |
|---|---|---|---|---|
| Workspace reference_speleo_iso (`reference_speleo_iso`) | fehlend | `main` 388 | `src/local_reconstr_all_epd/in/reference_speleo_iso.rdata` (fehlt) | Die Datei fehlt. Welche der Ladedateien d18o.age liefert, ist nicht belegt. |
| SISAL speleo_info.csv (`sisal_speleo_info`) | teilweise | `main` 391 | `src/climate_modelling/speleothems/speleo_data/speleo_info.csv` (vorhanden) | Unter climate_modelling liegt ein Namenskandidat ohne Versionsbeleg. Im aktiven Pfad gibt es keinen Leser. |
| SISAL ages_info.csv (`sisal_ages_info`) | teilweise | `main` 392 | `src/climate_modelling/speleothems/speleo_data/ages_info.csv` (vorhanden) | Unter climate_modelling liegt ein Namenskandidat ohne Versionsbeleg. Im aktiven Pfad gibt es keinen Leser. |
| SISAL d18_d13.RData (`sisal_d18_d13`) | fehlend | `main` 394 | `src/stalage_speleodatabase/sisal/SISAL_V2/d18_d13.RData` (fehlt) | Die Datei fehlt. Welche der Ladedateien d18o.age liefert, ist nicht belegt. |
| Workspace isotope_info (`isotope_info`) | fehlend | `main` 397 | `src/local_reconstr_all_epd/in/isotope_info.rdata` (fehlt) | Die Datei fehlt. Welche der Ladedateien d18o.age liefert, ist nicht belegt. |
| Workspace isotope_speleo_other (`isotope_speleo_other`) | fehlend | `main` 401 | `src/local_reconstr_all_epd/in/isotope_speleo_other.rdata` (fehlt) | Die Datei fehlt. Welche der Ladedateien d18o.age liefert, ist nicht belegt. |
| Objekt d18o.age (`d18o_age`) | fehlend | `prepare` 751, 752, 766, 767, 794, 828, 1038 | – | Das Objekt wird in keiner R-Datei von local_reconstr_all_epd erzeugt, sondern nur aus einem der fehlenden Workspaces geladen. Koordinaten, Namen, Alter und Reihen fehlen. |
| Index der Soreq-Reihe (`soreq_index`) | **Quellkonflikt** | `prepare` 790, 828; `prep_v1.2` 773 | `src/local_reconstr_all_epd/src/mcmc_prep_v1.2.R` (vorhanden) | Drei Quellstände nennen 37, 105 und 170. Ohne d18o.age sind sie keiner Reihe zuzuordnen. |
| Marine Referenzreihe (MEDSTACK) (`medstack_series`) | fehlend | `prepare` 1038, 1039, 1044 | – | Die Auswahlregel ist belegt, die Reihe selbst fehlt. Der Code wählt über die Länge von d18o.age$loc_nr, nicht über einen Namen. |
| Giesecke-Gitter (nur Koordinaten) (`giesecke_grid`) | gebunden | `main` 407, 412; `prepare` 565, 566 | `src/local_reconstr_all_epd/in/epd_giesecke_2016/quant_files/epdQuant_Abi.csv` (vorhanden) | Die CSV-Dateien liegen am Verbraucherpfad. Im Slot-3-Pfad liest der Code nur Lat und Lon, die Inhaltsspalten nicht. |

### Geladen oder geöffnet, ohne belegten Einfluss

| Eingabe | Stand | Gelesen in | Herkunft in `diss` | Warum |
|---|---|---|---|---|
| Insolationsreihe si.txt (`insolation`) | fehlend | `main` 367 | `src/insolation/si.txt` (fehlt) | Die Datei fehlt. sol.irr hat außerhalb von main_v1.2.R:367–381 keinen aktiven Leser. |
| station_ids_ref (`station_ids`) | gebunden | `main` 431 | `src/local_reconstr_all_epd/in/station_ids_ref.rdata` (vorhanden) | Die Datei liegt vor und enthält station_ids_ref. Das Objekt hat im aktiven Pfad keinen Leser. |
| kgc_distribution_ww.rdata (`kgc_rdata`) | gebunden | `main` 359 | `src/local_reconstr_all_epd/in/kgc_distribution_ww.rdata` (vorhanden) | Die Datei liegt vor und enthält kgc.distribution.ww. Das Objekt hat im aktiven Pfad keinen Leser, die Tabelle climatezones steckt nicht darin. |
| prob_c_kgc.rdata (`prob_c_kgc`) | fehlend | `main` 432 | `src/local_reconstr_all_epd/out/transfer_functions/prob_c_kgc.rdata` (fehlt) | Die Datei fehlt. Ihre Objektnamen sind unbekannt, ein Einfluss ist daher weder belegt noch ausgeschlossen. |
| IRD-Reihe bond2001.csv (`bond2001`) | fehlend | `check_ages` 238; `prepare` 872, 873 | `src/other_isotopes/bond2001.csv` (fehlt) | Die Datei fehlt. ird_data wird im aktiven Pfad nur im nicht erreichbaren Zweig mcmc_prepare.R:872 gelesen. |
| PMIP3-Eismaske (`pmip3_mask`) | fehlend | `prepare` 451, 462, 468 | – | Die Datei fehlt, wird aber geöffnet. Das Ergebnis der Prüfung in Zeile 462 wird in Zeile 468 unbedingt überschrieben. |

### Laufzeit und Pakete

| Eingabe | Stand | Gelesen in | Herkunft in `diss` | Warum |
|---|---|---|---|---|
| R-Fassung des Laufs (`r_version`) | teilweise | `main` 644 | – | Die Eingabecontainer stammen aus R 3.5.0. Die Fassung des Laufs ist nicht belegt; die Quellen wurden bis 07.07.2023 geändert. |
| rdirichlet, ddirichlet (`pkg_laplacesdemon`) | teilweise | `main` 38; `functions` 1248, 1269 | – | Im Fenster 2018-04-23…2023-07-07 liefert LaplacesDemon beide Funktionen, in allen vier Fassungen textgleich. Die installierte Fassung ist unbekannt. |
| weightedMedian (`pkg_matrixstats`) | teilweise | `main` 27; `prepare` 1631 | – | Im Fenster liefert allein matrixStats die Funktion. Die installierte Fassung ist unbekannt. |
| r_truncnorm, d_truncnorm (`pkg_rcppdist`) | teilweise | `functions` 1566 | – | Im Fenster gab es auf CRAN genau RcppDist 0.1.1. Die installierte Fassung ist unbekannt. |
| rtruncnorm aus dem Paket truncnorm (`pkg_truncnorm`) | fehlend | `prepare` 1886, 1957, 1958 | – | Die Funktion setzt die Startwerte tf_sample_tann und tf_sample_pann und verbraucht Zufallszahlen. Das Paket ist nicht gebunden. |
| inside.range (`pkg_spatstat_utils`) | fehlend | `main` 50; `prepare` 716 | – | Bestimmt die Masken und die Randwertregel bei 6500 a BP. Das Paket ist nicht gebunden. |
| psel (Pareto-Auswahl) (`pkg_rpref`) | fehlend | `main` 49; `prepare` 779 | – | Wird im Mehrkandidatenzweig der Höhlen gerufen; die Ausgabeordnung ist nicht gebunden. |
| wt und rcpp_wt_bases_morlet (`pkg_biwavelet`) | fehlend | `main` 52; `prepare` 901, 934 | – | Nur im Speläothemzweig. rcpp_wt_bases_morlet ist in keiner R-Quelle definiert; der Anbieter ist nicht gebunden und wurde hier nicht geprüft. |
| fftfilt (`pkg_signal`) | fehlend | `main` 24; `functions` 2325; `prepare` 868 | – | Nur im Speläothemzweig über smoothing. Das Paket ist nicht gebunden. |
| fitdistr (`pkg_mass`) | fehlend | `main` 62; `prior_climate` 106; `prepare` 1916 | – | Setzt die Anker und die Parameter der Transferfunktionen. Das Paket ist nicht gebunden. |
| LookupCZ, RoundCoordinates, CZUncertainty (`pkg_kgc`) | fehlend | `main` 22; `prior_climate` 37, 40 | – | Bestimmen die Klimaklasse des Standorts. Das Paket ist nicht gebunden. |
| arma::randn in rmvnorm_rcpp (`pkg_armadillo`) | fehlend | `main` 30; `functions` 1720; `conduction` 133 | – | Verbraucht beim Samplerstart Zufallszahlen. Paket und Compiler des Laufs sind nicht gebunden. |

Pfade anderer Slots in `main_v1.2.R` (14 Stück, drei fehlen) stehen in `other_slot_paths_not_affecting_kinneret`. Sie müssen beim Laden existieren, beeinflussen Slot 3 aber nicht. Die literalen Schalter stehen in `settings_literal`.

## Offene Bedingungen

| Bedingung | Stand | Gelesen in | Hängt an | Befund |
|---|---|---|---|---|
| sources$giesecke bleibt 0 (`sources_giesecke`) | belegt | `prepare` 555, 591 | – | Der Zweig 591 verlangt !add.sites oder giesecke_prespa. Beides ist im gebundenen Quellstand falsch. |
| inside.giesecke ist FALSE (`inside_giesecke`) | bedingt | `prepare` 568, 572, 575, 587 | `coordinates`, `giesecke_grid` | Das Giesecke-Gitter endet im Süden bei 34,75° N mit einer einzigen Zelle bei 24,75° E. Mit den älteren Koordinaten ist die Raumbedingung falsch. Der finale Containerwert ist nicht belegt. |
| sources$core (`sources_core`) | bedingt | `prepare` 636, 661, 662 | `coordinates`, `giesecke_grid`, `ap_reference`, `age_grid`, `rho_depth_given_age` | Der Zweig 636 wird erreicht, solange inside.giesecke falsch ist; dann wird sources$core gesetzt. Offen bleibt der Varianztest in Zeile 662, der die Alterskurve braucht. |
| sources$speleothem (`sources_speleothem`) | **offen** | `prepare` 759, 772, 773, 813, 814 | `d18o_age`, `age_grid`, `coordinates`, `pkg_spatstat_utils` | Hängt an den Höhlenkoordinaten und Altersachsen in d18o.age. Das Objekt fehlt. Der Dissertationstext nennt Soreq-WPS für Kinneret; das bindet den Code nicht. |
| sources$marine_ice_core (`sources_marine`) | bedingt | `prepare` 1072, 1212 | `site_name`, `coordinates`, `giesecke_grid` | Zeile 1072 setzt den Wert ohne Bedingung auf 1. Zeile 1212 setzt ihn für Standorte außerhalb der Namensliste wieder auf 0. Das hängt am Standortnamen. |
| multiple und spectral (`multiple_spectral`) | **offen** | `prepare` 1189, 1195, 1200, 1203 | `site_name`, `d18o_age` | Folgen aus den vier Quellenschaltern. Die Tabelle source_flag_table führt alle Fälle auf. |
| wt_power erreichbar (`wt_power_reachable`) | **offen** | `prepare` 947, 966; `functions` 2021, 2023 | `d18o_age` | Läuft genau dann, wenn sources$speleothem 1 ist: in der Vorbereitung und in jeder Iteration über spectral. Das bleibt offen, solange d18o.age fehlt. |
| site.age wird im Speläothemzweig ersetzt (`site_age_override`) | bedingt | `prepare` 825, 826, 828 | `d18o_age`, `soreq_index` | Ist sources$speleothem 1, ersetzt Zeile 825–826 site.age durch 418 Werte mit einer Lücke von 2000 bis 14500 a BP, und Zeile 828 setzt Reihe 170. Beides passt nicht zu einem 0–9-ka-Kern und ist ohne Laufbindung nicht als Kinneret-Lauf zu lesen. |
| recent (`recent`) | bedingt | `prepare` 728, 1183, 1184, 1264, 1730; `conduction` 146; `functions` 1844 | `age_grid` | recent ist genau dann FALSE, wenn kein Alter ≤ 11000 vorkommt. Die 160 älteren Mittelalter reichen von −60 bis 9015. Das finale Raster age[[2]] fehlt. |
| Zweig des Klimaankers (`anchor_branch`) | bedingt | `check_ages` 223; `prior_climate` 69, 109 | `age_grid`, `cru4_pann`, `cru4_tann`, `cru_pann_ww`, `cru_tann_ww`, `climatezones` | Bei mindestens einem Alter ≤ 0 nimmt der Code die lokale CRU4-Zeitreihe, sonst die Klassenmittel. Das ältere Altersraster hat Werte ≤ 0, das finale fehlt. Das ist eine andere Bedingung als recent. |
| tann (`tann_flag`) | bedingt | `prepare` 1208, 1209, 1210, 1249; `functions` 1939 | `site_name`, `coordinates`, `giesecke_grid` | Bei „Lake Kinneret“ setzt Zeile 1249 FALSE. Bei einem anderen Namen setzt mit den älteren Koordinaten schon Zeile 1210 FALSE, schaltet dann aber auch multiple und die marine Reihe ab. |
| hi_time ist FALSE (`hi_time`) | belegt | `prepare` 713, 715, 718 | `human_impact` | Im Zusatzorte-Zweig setzt Zeile 718 hi_time zurück. Die Altersmaske hi_ages_time bleibt bestehen. |
| hi_spec (`hi_spec`) | bedingt | `prepare` 879, 881, 882 | `site_name`, `d18o_age` | Zeile 881 schließt „Lake Kinneret“ aus. Bei einem anderen Namen würde hi_spec TRUE. Der Zweig läuft nur bei sources$speleothem 1. |
| stad_interstad ist FALSE (`stad_interstad`) | belegt | `prepare` 468 | `pmip3_mask` | Zeile 468 setzt den Wert unbedingt. Die PMIP-Maske beeinflusst das Ergebnis deshalb nicht. |
| Mehrfachzuordnung von Taxa (`taxa_merge`) | **offen** | `prepare` 1460, 1470, 1557, 1567 | `taxon_names`, `taxa_name_pool`, `tf_container` | Ob ein Name fehlt, hängt an der finalen Taxonliste. Der Zweig zieht ohne eigenen Seed (1557–1558) und überschreibt alle Rechenvarianten mit der letzten. |
| Fehlende Transferfunktionen (`tf_missing`) | **offen** | `prepare` 1664, 1678 | `tf_container`, `taxon_names` | Welche Taxa ohne Funktion sind, hängt am fehlenden Container. |
| Aktiver Abbruch vor dem Sampler (`active_stop`) | belegt | `prepare` 2057; `main` 675, 676 | – | mcmc_prepare.R endet mit einem aktiven stop(). Im erhaltenen Quellstand wird mcmc_conduction.R daher nicht erreicht. Für den historischen Lauf muss eine andere Fassung gegolten haben; welche, ist nicht belegt. |
| Zufallszustand beim Samplerstart (`rng_start`) | **offen** | `prepare` 1354, 1898; `conduction` 133 | `r_version`, `pkg_truncnorm`, `pkg_armadillo`, `tf_current`, `taxon_names`, `age_grid` | Belegt ist nur die Kette: letzter Reset set.seed(2023) in Zeile 1898, danach weitere Ziehungen. Die Startwerte in 1709–1763 liegen schon davor, nach set.seed(2022) in Zeile 1354. Der Zustand hängt an RNGkind, R-Fassung, Paketen und allen fehlenden Eingaben. |

## Quellenschalter je Fall

Symbolische Spur von `mcmc_prepare.R:1189–1203`, `1208–1214`, `1248–1252`, keine Rechnung. Die Reihenfolge zählt: Zeile 1189 liest die Quellen, bevor Zeile 1212 `sources[4]` ändert. Zeilen mit `Lake Kinneret` stimmen mit den eingefrorenen `control_flow_traces` des Referenzvertrags überein.

| Name in Slot 3 | `core` | `speleothem` | aktive Quellen (1189) | `multiple` | `spectral` | marine Reihe nach 1212 | `wt_power` läuft |
|---|---|---|---|---|---|---|---|
| Lake Kinneret | 1 | 1 | [2, 3, 4] | ja | ja | 1 | ja |
| Lake Kinneret | 1 | 0 | [2, 4] | ja | nein | 1 | nein |
| Lake Kinneret | 0 | 1 | [3, 4] | nein | ja | 1 | ja |
| Lake Kinneret | 0 | 0 | [4] | nein | nein | 1 | nein |
| Kinneret | 1 | 1 | [2, 3, 4] | nein | ja | 0 | ja |
| Kinneret | 1 | 0 | [2, 4] | nein | nein | 0 | nein |
| Kinneret | 0 | 1 | [3, 4] | nein | ja | 0 | ja |
| Kinneret | 0 | 0 | [4] | nein | nein | 0 | nein |

**`wt_power` läuft genau dann, wenn `sources$speleothem` 1 ist.** Das hängt an `d18o.age` und bleibt offen.

## Zufallszustand beim Samplerstart

Nicht belegt, und aus den Quellen allein auch nicht bestimmbar. Festgehalten ist die Kette in `rng_chain`:

- Letzter Reset vor dem Sampler ist `set.seed(2023)` in `mcmc_prepare.R:1898`, je Taxon. Danach folgen weitere Ziehungen: `sample` (1899), `rtruncnorm` (1957–1958), `rnorm` und `rgamma` (1979–1980) und `rmvnorm_rcpp` (`mcmc_conduction.R:133`).
- Die Startwerte `post_pollen` bis `means_diff_pann` (1709–1763) liegen davor, also nach `set.seed(2022)` in Zeile 1354.
- `RNGkind` und `RNGversion` kommen in den sechs aktiven Dateien nicht vor. Generatorwahl und `sample`-Verfahren hängen damit an der unbekannten R-Fassung.
- Wie viele Zufallszahlen die Ziehungen verbrauchen, hängt an Paketen und an fehlenden Eingaben wie `num_taxa` und den Transferfunktionen.
- `set.seed(2019)` in `main_v1.2.R:1055` liegt hinter der Stationsschleife (651–684) und betrifft den Fall nicht; die Variable `seeds` aus Zeile 65 hat keinen aktiven Leser.

## Bezug zu den elf Nachweisbündeln

Alle elf bleiben offen. Das Paket ordnet ihnen nur die konkreten Eingaben zu.

| Nachweisbündel (Phase 1) | Stand dort | Eingaben hier |
|---|---|---|
| Finaler Kinneret-Basiseintrag (`final_base`) | fehlt | `core_container`, `site_name`, `taxa_spectrum`, `taxon_names`, `depth`, `coordinates`, `country_elevation` |
| Finale Listenchronologie und Proben-/Altersraster (`final_chronology`) | widerspruechlich | `depth`, `age_grid`, `rho_depth_given_age` |
| Numerische historische TFs und finale Taxonzuordnung (`historical_tf`) | fehlt | `taxon_names`, `tf_container`, `tf_axes`, `taxa_name_pool`, `tf_current` |
| Ursprüngliche Isotopen-Workspaces (`isotope_workspaces`) | fehlt | `reference_speleo_iso`, `sisal_speleo_info`, `sisal_ages_info`, `sisal_d18_d13`, `isotope_info`, `isotope_speleo_other`, `d18o_age`, `medstack_series` |
| Globale Klimaeingaben und Paket-/Klassenbindung (`cru_kgc_inputs`) | fehlt | `cru_pann_ww`, `cru_tann_ww`, `cru4_pann`, `cru4_tann`, `climatezones`, `pkg_kgc` |
| Historische Soreq-/MEDSTACK-Vorbereitung (`reference_variant`) | widerspruechlich | `d18o_age`, `soreq_index`, `medstack_series` |
| Historische Klimaankerquelle und Parameter (`anchor_variant`) | widerspruechlich | `cru4_pann`, `cru4_tann` |
| Originalposterioren und ausgewählte Ziehungs-IDs (`posterior_selection`) | fehlt | Ausgabeseite, keine Eingabe des Samplers |
| Finale Auswahl-/Kombinationsfassung (`postprocessing_variant`) | widerspruechlich | Ausgabeseite, keine Eingabe des Samplers |
| Historische Laufzeit und Zufallszustände (`runtime_rng`) | fehlt | `r_version`, `pkg_laplacesdemon`, `pkg_matrixstats`, `pkg_rcppdist`, `pkg_truncnorm`, `pkg_spatstat_utils`, `pkg_rpref`, `pkg_biwavelet`, `pkg_signal`, `pkg_mass`, `pkg_kgc`, `pkg_armadillo` |
| Numerische Originalausgaben und Abweichungsvergleich (`numeric_original_output`) | fehlt | Ausgabeseite, keine Eingabe des Samplers |

## Prüfen

```powershell
.venv/Scripts/python.exe docs/thesis_research/kinneret_input_contract_2026_11_09/bind_input_contract.py
.venv/Scripts/python.exe -m pytest tests/test_kinneret_input_contract.py -q
.venv/Scripts/python.exe tools/check_kinneret_phase1_spec.py
```

Das Belegskript braucht `diss` und `.local/my_reconst_paths.json`. Die Tests brauchen beides nicht; nur der Zeilenabgleich mit den Originalen wird ohne `diss` übersprungen. Der Phase-1-Prüfer endet erwartungsgemäß mit Exit 2. Die tatsächlich ausgeführten Läufe stehen in [verification.json](verification.json).

## Grenzen

- Kein Originalfall, keine Originalzahlen außer Metadaten des älteren Containers und der Gitterkoordinaten.
- Der Vertrag beschreibt den erhaltenen Quellstand (Änderungsdatum bis 07.07.2023), nicht den historischen Lauf.
- Nicht geprüft: der Anbieter von `rcpp_wt_bases_morlet` und die Pakete `truncnorm`, `spatstat.utils`, `rPref`, `biwavelet`, `signal`, `MASS` und `kgc` samt Fassungen. Sie stehen als fehlend in der Liste.
- Die Speicherortfrage wird nicht wiederholt. Baseline, Gates, Artenmodelle und Paperreferenz bleiben unverändert.
