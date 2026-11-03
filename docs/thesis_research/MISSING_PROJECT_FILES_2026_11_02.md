# Benötigte, noch fehlende Projektdateien und vorhandene Kopien

Stand 02.11.2026, Basis `26d06cfc3530eef5c65f53e270812894b5d84838`,
Branch `dissertation-reconstruction`, [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).

> **Neue Nutzerinformation nach dem Abgleich:** In der nächsten Sitzung
> wird ein zusätzlicher Ordner `climate_modelling` verfügbar. Er ist hier
> **noch nicht geprüft**. Die folgenden Fehlbefunde gelten für den bisherigen
> Suchumfang; Claude soll sie zuerst mit dem neuen Ordner abgleichen.
> [Aktuelle Fortsetzung und vollständige Übergabe](CLAUDE_RESUME_CLIMATE_MODELLING_2026_11_02.md).

**Die Idee ist umgesetzt:** Dateien werden aus den erhaltenen Verbrauchern
abgeleitet, mit lokalen Beständen/Kopien abgeglichen und nach ihrer wirklichen
Rolle priorisiert. Ein alter `/automount/`-Pfad bedeutet Verlust des damaligen
Datenträgers; eine lokale Kopie kann trotzdem vorhanden sein. Daher wird erst
abgeglichen und nur bei verbleibender Lücke ein Wiederaufbau geplant.

Die geprüfte Liste enthält **39 dateibezogene Anforderungen** in drei Prioritäten
und **acht separate historische Kinneret-Ergebnisvarianten**. Das sind nicht
47 universell erforderliche Eingaben. **16 direkte Ladelücken** betreffen den
erhaltenen lokalen R-Einstieg: **zwölf RData-/NC-Dateien und vier TXT-/CSV-Dateien**.
Bedingte Varianten und erzeugte Ergebnisse bleiben getrennt. Der bestehende
Python-Paperfall und die App sind dadurch nicht plötzlich unbenutzbar.

## A. Direkte Ladelücken des erhaltenen lokalen R-Einstiegs

Alle folgenden referenzierten Namen wurden in den geprüften Ordnern und
ZIP-Inhaltsverzeichnissen nicht gefunden. „Erforderlich“ meint hier: Der
unveränderte erhaltene R-Code lädt die Datei. Das beweist noch nicht ihre
fachliche Notwendigkeit in jeder künftigen Implementierung oder eine finale
historische Laufvariante. Deren Auswahl bleibt offen.

| ID | Datei | Benötigt für | Beleg |
| --- | --- | --- | --- |
| F01 | `core_data_additional_sites_v1.2.rdata` | Finale Basis einschließlich Standort-/Taxon-/Chronologieobjekten | `original:main_v1.2.R:107` |
| F02 | `mixed_partial_arts_emd.rdata` | Historische numerische Taxon-TFs | `original:main_v1.2.R:569` |
| F03 | `cru_pann_ww.nc` | Globales PANN-Mittel und Stationsmaske stn | `original:main_v1.2.R:325` |
| F04 | `cru_tann_ww.nc` | Globales TANN-Mittel | `original:main_v1.2.R:326` |
| F05 | `cru4_pann_v1.2.nc` | Zeitaufgelöste PANN-Ankerdaten | `original:main_v1.2.R:343` |
| F06 | `cru4_tann_v1.2.nc` | Zeitaufgelöste TANN-Ankerdaten | `original:main_v1.2.R:344` |
| F07 | `reference_speleo_iso.rdata` | Speläothem-Referenzworkspace | `original:main_v1.2.R:388` |
| F08 | `d18_d13.RData` | SISAL-d18O/d13C-Workspace | `original:main_v1.2.R:394` |
| F09 | `isotope_info.rdata` | Isotopen-Metadatenworkspace | `original:main_v1.2.R:397` |
| F10 | `isotope_speleo_other.rdata` | Gemeinsamer Isotopen-/Speläothem-Workspace | `original:main_v1.2.R:401` |
| F11 | `prob_c_kgc.rdata` | Klima-/Köppen-Geiger-Zuordnung aus Transferfunktionsumfeld | `original:main_v1.2.R:432` |
| F12 | `pmip3_21k_v0.nc` | PMIP3-LGM-Eismaske mask2 mit lon/lat | `original:src/mcmc_prepare.R:451` |
| F13 | `si.txt` | Insolationsreihe | `original:main_v1.2.R:367` |
| F14 | `speleo_info.csv` | SISAL-Höhlenmetadaten | `original:main_v1.2.R:391` |
| F15 | `ages_info.csv` | SISAL-Altersmetadaten | `original:main_v1.2.R:392` |
| F16 | `bond2001.csv` | IRD-/Bond-Referenz | `original:src/check_ages.R:238` |

Besonders wichtig: Die PMIP3-Datei **`pmip3_21k_v0.nc`** wird in der im Juli 2023
bearbeiteten `mcmc_prepare.R:451` geöffnet, bevor später `stad_interstad <- FALSE`
gesetzt wird. Das Abschalten des späteren Zweigs überspringt diesen Dateizugriff
nicht. Gesuchtes Objekt ist `mask2` samt `lon`/`lat`; hier ist ein konkreter
Automount-Datenverlust mit bislang fehlender lokaler Namenskopie belegt.

Die vier globalen CRU-Dateien sind nicht durch vorhandene regionale CRU-
Ausschnitte ersetzt. Globale Maske/Achsen, Temperaturgröße, Version, Zeitraum
und Ankerzeitachse sind weiter an ihre Verbraucher zu binden. Zeitindizes
50:115 sind ohne Zeitachse keine bewiesenen Kalenderjahre.

Bei F01 existiert ein älterer Sammelcontainer, aber nicht die benannte finale
v1.2-Datei. Auch die erhaltene Kinneret-Mittelwertmatrix ersetzt keine erwartete
Liste mehrerer Altersraster. Diese Befunde sind keine reine Pfadkorrektur.

## B. Fall-, Referenz-, Erzeuger- oder Nachbearbeitungszweig

Diese Dateien dürfen nicht pauschal als Voraussetzung für jeden Kinneret-Lauf
behandelt werden. Für den jeweils genannten Zweig ist ein Verbraucher belegt;
seine finale historische Verwendung ist nicht in allen Fällen gebunden.

| ID | Datei | Benötigt für | Beleg |
| --- | --- | --- | --- |
| F17 | `mixed_partial_arts_emd_info.rdata` | Numerische TF-Informationen und Qualitäts-/Zuordnungsdaten | `original:src/all_trafo.R:544` |
| F18 | `tann_test.nc` | Klimaraster im erhaltenen TF-Erzeuger: TANN | `original:src/all_trafo.R:47` |
| F19 | `pann_test.nc` | Klimaraster im erhaltenen TF-Erzeuger: PANN | `original:src/all_trafo.R:48` |
| F20 | `core_data_ext.rdata` | Erweiterte EPD-Kerndaten | `original:main_v1.2.R:90` |
| F21 | `core_data_all_giesecke_v1.2.rdata` | Giesecke-Prozentbasis | `original:main_v1.2.R:86` |
| F22 | `core_data_all_giesecke_counts.rdata` | Giesecke-Zählbasis | `original:main_v1.2.R:88` |
| F23 | `core_information.csv` | Iznik-Kerninformationen | `original:main_v1.2.R:161` |
| F24 | `rho_depth_given_age_higher_resol.rdata` | En-Gedi-Chronologie mit höherer Auflösung | `original:main_v1.2.R:221` |
| F25 | `rho_depth_given_age.rdata` | Van-Chronologie am referenzierten Standortpfad | `original:main_v1.2.R:288` |
| F26 | `pollen_g_climate_all_giesecke_v4.0.rdata` | Älterer/globaler Pollen-Klima-Sammelstand | `original:main_v1.2.R:858` |
| F27 | `cru_pann2_ww.nc` | Jährliche globale PANN-Datei für Posterior-Transformation | `original:src/mcmc_trimming_posterior_trafo.R:461` |
| F28 | `cru_tann2_ww.nc` | Jährliche globale TANN-Datei für Posterior-Transformation | `original:src/mcmc_trimming_posterior_trafo.R:462` |
| F29 | `cru_ymonmean_pann.nc` | Monatsmittel PANN | `original:src/mcmc_case_studies.R:1514` |
| F30 | `cru_ymonmean_tann.nc` | Monatsmittel TANN | `original:src/mcmc_case_studies.R:1515` |
| F31 | `cru_ymonmean_usa_pann.nc` | USA-Monatsmittel PANN | `original:src/mcmc_case_studies.R:1572` |
| F32 | `cru_ymonmean_usa_tann.nc` | USA-Monatsmittel TANN | `original:src/mcmc_case_studies.R:1573` |

Bei **F25 (`rho_depth_given_age.rdata` für Van)** gibt es gleichnamige Dateien
anderer Seen. In der aktuellen Suche gibt es keine Datei unter passendem
`my_van`-Pfad. Das ist eine fallbezogene Zuordnungslücke, kein Beleg, dass jede
Datei dieses Namens fehlt. F24 ist die ausdrücklich benannte höhere En-Gedi-
Auflösung; andere En-Gedi-Chronologien liegen lokal vor.

F20–F22 gehören zum Nicht-`add.sites`-/EPD-Zweig. F27–F32 betreffen besondere
Nachbearbeitungs-/Fallstudienfassungen. Nicht automatisch neu erzeugen, bevor
der jeweilige benötigte Zweig festgehalten ist.

## C. Gespeicherte Erzeuger-/Vergleichsartefakte

Auch diese Namen fehlen im abgeglichenen Bestand. Sie gehören zu bestimmten
TF-Erzeuger-, Wiederanlauf- oder Vergleichsfassungen; vorhandene Save-/Erzeuger-
Zeilen sind in der CSV vermerkt. Sie machen die Artefakte nicht automatisch zu
externen Eingaben, die schon vor jeder neuen Rechnung vorliegen müssten.

| ID | Datei | Benötigt für | Beleg |
| --- | --- | --- | --- |
| F33 | `all_nnet_transfer_functions.rdata` | Gespeicherte Arten-Netzmodelle | `original:src/all_trafo.R:560` |
| F34 | `all_qda_transfer_functions.rdata` | Gespeicherte Arten-QDA-Modelle | `original:src/all_trafo.R:566` |
| F35 | `mixed_tf_info.rdata` | TF-Metainformationen älterer Erzeugerfassung | `original:src/all_trafo.R:699` |
| F36 | `fits.rdata` | Gespeicherte Fit-Sammlung | `original:src/all_trafo.R:888` |
| F37 | `taxon_map.rdata` | Taxonkarte/Zuordnung | `original:src/all_trafo.R:1178` |
| F38 | `mixed_tf_info_partials.rdata` | TF-Metadaten einer Vergleichsfassung | `original:src/all_transferfunctions_v0.9.R:146` |
| F39 | `mixed_tf_partial_v1.1.rdata` | Numerische TFs einer Vergleichsfassung | `original:src/all_transferfunctions_v0.9.R:147` |

Die sieben modernen experimentellen Artenmodelle sind kein Nachweis dieser
historischen Sammelcontainer. Trainings oder Sampler wurden nicht gestartet.

## D. Historische Ergebnisdateien für Auswahl-/Ergebnisabgleich

Alle acht bereits im Nachbearbeitungsvertrag benannten Kinneret-Varianten
wurden erneut per Dateiname im aktuellen Index abgeglichen: keine lokale
Namenskopie gefunden. Keine davon wird als finale Variante gewählt.

- `out/posterior/rw_posterior_add_3_1_counts.rdata`
- `out/posterior/rw_posterior_add_3_1_case_studies_counts.rdata`
- `out/posterior/rw_posterior_add_3_1.rdata`
- `out/posterior/rw_posterior_add_3_1_case_studies.rdata`
- `out/posterior/gibbs_posterior_add_3_1_counts.rdata`
- `out/posterior/gibbs_posterior_add_3_1_case_studies_counts.rdata`
- `out/posterior/gibbs_posterior_add_3_1.rdata`
- `out/posterior/gibbs_posterior_add_3_1_case_studies.rdata`

Das sind historische Posterior-/Vergleichsausgaben, keine acht zusätzlichen
externen Grunddaten vor einer Neuberechnung. Neue Ziehungen wären neu erzeugte
Ergebnisse und nicht automatisch die verlorenen Originalposterioren.
Weitere alte Zusammenfassungen, Laufzeit-/Skill- und Plotzwischenstände sind
im vollständigen Suchregister verzeichnet, nicht als universelle Pflichtdateien.

## E. Vorhandene Dateien: nicht unnötig neu erstellen

Ein konkreter positiver Fund stützt die Nutzervermutung:

- Der alte Automount-Pfad auf **`my_ohrid/data/in/climate_data/taxa_distribution_ww.rdata`**
  hat eine pfadpassende lokale Datei im Zusatzordner. 3.703.315 Bytes, SHA-256
  `89a5e35906e2bd50ae5c85881269aecc59992f91744c6f879cc481e7ff875073`;
  unverändert zum früheren strukturellen Audit. Der erhaltene globale
  283→279-Quellpfad ist bereits dokumentiert. Kein neuer Originalfund nötig.
- Auch **`my_kinneret/data/in/climate_data/taxa_distribution_ww.rdata`** liegt vor
  (3.603.177 Bytes, SHA-256
  `73efc9c9b9ba03d8e964db9b69f592d8aa707f7e2c72d45798e3549aec1e3dd8`).
  Es ist ein anderer erhaltener Stand, keine beliebig austauschbare Variante.
- `in/kgc_distribution_ww.rdata` und `in/station_ids_ref.rdata` sind im
  Dissertationsordner vorhanden.
- Die Standort-CSV-Dateien von Prespa, En Gedi, Kinneret, Birkat Ram und Ohrid
  sowie mehrere Pollentabellen/Van-Überlappungstabellen wurden unter passendem
  Standort-Pfadsuffix im Zusatzordner gefunden. Alte absolute Pfade können
  falsch sein, während die Datei lokal tatsächlich existiert.
- Mehrere Standortchronologien, regionale Klimaausschnitte und der ältere
  Sammelcontainer sind vorhanden. Ihre Datenverträge und historischen Varianten
  bleiben separat zu prüfen; sie sind keine stillen Ersatzbelege für A.

Die beiden kleinen Verbreitungscontainer wurden gegen ihre früheren Hashes
geprüft. Es wurden keine großen RData-/NetCDF-Felder dekodiert. Für andere
Namens-/Pfadkandidaten bedeutet „vorhanden“ noch keine Gleichheit mit dem
verlorenen Automount-Original oder dem finalen historischen Lauf.

## F. `src` nach 2021: Priorität und Datumsgrenzen

369 erhaltene R-Dateien aus beiden Quellordnern wurden lesend untersucht.
**35 Dissertations-`src`-Dateien und 149 Zusatz-`src`-Dateien** haben ein erhaltenes
Änderungsdatum ab 01.01.2022. Ihre vollständigen Zeitstempel, Hashes und
Zuordnungen stehen in [source_dates.csv](required_files_2026_11_02/source_dates.csv).
Ältere indirekte Abhängigkeiten bleiben erfasst. Relevante Dissertationsdateien:

| Datei | Erhaltenes Änderungsdatum (UTC) |
| --- | --- |
| `src/mcmc_prepare.R` | 2023-07-07 |
| `src/mcmc_functions_v1.6.R` | 2023-07-07 |
| `src/mcmc_trimming_posterior_trafo.R` | 2023-04-17 |
| `src/mcmc_conduction.R` | 2023-04-04 |
| `src/final_plotting.R` | 2023-04-04 |
| `src/check_ages.R` | 2023-03-30 |
| `src/mcmc_case_studies.R` | 2023-03-28 |
| `src/all_trafo.R` | 2023-03-21 |
| `src/prior_climate.R` | 2023-02-23 |
| `src/all_transferfunctions_v0.9.R` | 2023-02-06 |
| `src/tf_qda_un-smote.R` | 2022-12-05 |
| `src/all_transferfunctions_v1.R` | 2022-05-26 |

`main_v1.2.R` trägt Änderungsdatum 01.07.2023. Erzeugungszeit auf dem heutigen
Windows-Dateisystem liegt häufig im November 2026 und kann schlicht das
Kopieren abbilden. **Historische Erstellung nach 2021 ist damit nicht bewiesen.**
Änderungszeiten dienen zur Priorisierung, nicht als Nachweis einer ausgeführten
Dissertationsfassung. Die späteren Paper-/Biomzweige von 2023/2024 sind deshalb
sichtbar, aber nicht automatisch finale taxonbasierte Dissertationsquellen.

## Umfang, Grenzen und nutzbare Listen

Abgeglichen wurden die beiden privat konfigurierten Quellordner, der vorhandene
Checkout außerhalb privater/Laufzeitverzeichnisse und Inhaltsverzeichnisse von
18 ZIP-Dateien. Index: **584 Datendateien**, davon **257 RData/RDS/NC-Dateien**.
Die Prüfung ist keine Suche auf dem gesamten PC. Der Chat-Artefaktdienst listete
keine gesonderten Upload-Artefakte; nicht erreichbare frühere Anhänge können
hier nicht als geprüft gelten. Die beiden zugänglichen lokalen Quellordner
wurden tatsächlich geprüft.

Der statische Suchlauf enthält 1178 Dateileseverweise: 264 am angenommenen
Projektpfad vorhanden, 142 mit lokalen Namenskandidaten, 615 ohne Treffer und
157 dynamische Familien ohne Kandidaten. Das sind **Aufrufstellen**, keine
Anzahlen notwendiger oder eindeutiger Dateien. 13 weitere Leseausdrücke konnten
nicht zu einem Dateinamen mit Datenendung aufgelöst werden.
Kommentare werden ausgeschlossen, verschachtelte Zeichenketten/Aufrufe erfasst.
Bedingungen, spätere Überschreibungen, `setwd`, Paket-internes Laden und tatsächliche
Funktionsausführung werden nicht vollständig ausgewertet. Eine statisch gesetzte
Zeichenkonstante kann eine Variante abbilden. Die kuratierte Priorität ist daher
von den Rohverweisen getrennt. Umbenannte Container und Objektgleichheit unter
anderen Dateinamen sind durch diese Metadatenprüfung nicht vollständig ausgeschlossen.

- [Priorisierte Dateiliste als CSV](required_files_2026_11_02/priority_requirements.csv)
  mit Rollen, Bedingungen, Fundstatus, Kopiekandidaten und Erzeugerhinweisen.
- [Alle Dateileseverweise](required_files_2026_11_02/dependency_calls.csv),
  [ungefundene Verweise](required_files_2026_11_02/missing_references.csv) und
  [nicht auflösbare Ausdrücke](required_files_2026_11_02/unresolved_reads.csv).
- [Erzeuger-/Speicherverweise](required_files_2026_11_02/producer_calls.csv),
  [statischer Source-Graph](required_files_2026_11_02/source_edges.csv) und
  [Suchumfang](required_files_2026_11_02/summary.json).
- [Lokale Verbreitungsdateien mit geprüftem Hash](required_files_2026_11_02/preserved_copies.json)
  und [erneuter Posteriorpfadabgleich](required_files_2026_11_02/posterior_variants_recheck.json).
- [Methodik und Wiederholung](required_files_2026_11_02/README.md).

**Planfolge:** Erst konkrete lokale Kopien samt Struktur/Version binden; nur
verbleibende Lücken anhand belegter Erzeuger und benötigter Eingaben neu erstellen.
Keine automatische Rekonstruktion, Variantenwahl oder Freigabe. Originale,
Baseline, App/CLI und eingefrorene Belege bleiben erhalten. Alle elf bisherigen
Nachweisbündel bleiben offen; ein Dateifund schließt sie nicht automatisch.
