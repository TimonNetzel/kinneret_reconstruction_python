# climate_modelling: neuer Quellenabgleich

Stand 02.11.2026, Ausgangscommit `705cf7b`, bestehender Branch
`dissertation-reconstruction`, [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).
Dieses Paket ergänzt den [eingefrorenen Zwei-Quellen-Abgleich](../required_files_2026_11_02/README.md).
Es ersetzt weder dessen damalige Befunde noch den Phase-1-Belegvertrag.

## Ergebnis für die beiden ausdrücklich genannten Ladepfade

| Verbraucher | Neuer Befund | Grenze |
| --- | --- | --- |
| `original:main.R:128–130` | `pollen/transfer_functions/input_data/data_out/taxa_distribution_ww.rdata` vorhanden; SHA-256 `73efc9c9b9ba03d8e964db9b69f592d8aa707f7e2c72d45798e3549aec1e3dd8`. Bytegleich zur bereits belegten Kinneret-Kopie. | Geografische Verbreitungsbasis, keine numerischen historischen Klima-TFs und kein Beleg finaler Ausführung. |
| `original:main_v1.2.R:858,860` | Keine Namenskopie von `pollen_g_climate_all_giesecke_v4.0.rdata`, auch nicht im ZIP-Verzeichnis. Ältere unversionierte/v1.4-Dateien sind vorhanden. | Nicht als v4.0 einsetzen. Kein finaler Basiseintrag oder historische Laufbindung dadurch geschlossen. |

Die Verbreitungsdatei enthält `taxa.distribution.ww` mit `dist`, `lon`, `lat`,
`names`: **720 × 360 × 275**, Längengitter −179,75 bis 179,75 und Breitengitter
−89,75 bis 89,75, jeweils Schritt 0,5. Die Achsen tragen keine gespeicherten
Einheitenattribute; keine Einheitenkorrektur vorgenommen. Die 570.240.000 Bytes
des numerischen Verbreitungsfeldes wurden gestreamt/gehasht und nicht als Array
materialisiert; nur 11.060 Bytes kleiner numerischer Metadaten behalten.
Der Hash unterscheidet sich von der bereits erhaltenen Ohrid-Fassung
(`89a5e359…`). **275 Taxa sind kein Nachweis der 279 historischen TFs.**

## Abgleich der Fehlenden-/Kopienliste

[Alle 39 Anforderungen plus acht Posteriorvarianten und main.R-Einstieg](requirement_crosswalk.csv)
sind ausdrücklich gegen den **neuen** Bestand geprüft. Nur **F14 und F15**
haben neue Namenskandidaten unter `speleothems/speleo_data/`:

- `speleo_info.csv`: 455 Datensätze, 15 Spalten, 455 eindeutige `entity_id`.
- `ages_info.csv`: 6.849 Datensätze, sieben Spalten, 455 eindeutige `entity_id`.

Die Spalten enthalten passende Höhlen-/Altersmetadaten. Der Verbraucher
`main_v1.2.R:391–392` nennt jedoch **SISAL_V2** in einem anderen Pfad. Die
neuen CSVs tragen erhaltene mtime vom 04.07.2019; dies belegt keine konkrete
SISAL-Version. Kein SISAL-V2-Versionsbeleg/Referenzhash ist gebunden, Einheiten
sind in den CSV-Kopfzeilen nicht deklariert. Daher **Kopiekandidaten mit offener
Version**, keine freigegebenen Ersatzdateien und keine geschlossenen Gates.
`read.csv(..., sep = ,)` bleibt im Original unverändert; hier wurde kein R-Code
ausgeführt oder dessen Einleseverhalten repariert.

Die übrigen **37 priorisierten Namen** und **alle acht benannten Posteriorvarianten**
haben auch in dieser zusätzlichen Quelle keine Namenskopie. Damit verbleiben
für die 16 direkten Ladelücken **14 ohne Namenskopie** und **zwei mit ungeklärter
Versionszuordnung**. Das ist keine Liste universeller Eingaben und schließt
umbenannte äquivalente Objekte nicht aus.

Insbesondere bleiben finale v1.2-Basis/Listenchronologie, `mixed_partial_arts_emd*`,
die benannten globalen CRU-Dateien, Isotopenworkspaces und `pmip3_21k_v0.nc`
ungebunden. Die erhaltene Mittelwertmatrix wird nicht zur Liste mehrerer
Altersraster erklärt. Keine finale Kinneret-Variante ausgewählt; alle elf
Nachweisbündel bleiben offen.

## Versionskonflikte und ausdrücklich ausgeschlossene Ersatzannahmen

| Kandidat | Prüfbarer Unterschied | Folgerung |
| --- | --- | --- |
| `pollen_speleothems/mcmc_recon/Data/pollen_g_climate_all_giesecke.rdata` und gleichnamige Datei in `verification/Data/my/` | Beide 180.541.931 Bytes, SHA-256 `d8b2f90b…`, bytegleich. | Erhaltene Kopien eines unversionierten Stands; kein v4.0-Nachweis. |
| `…/Data/pollen_g_climate_all_giesecke_v1.4.rdata` gegenüber `verification/Data/my/…v1.4.rdata` | 65.335.056 / 56.092.869 Bytes, SHA-256 `80fda992…` / `7f01ac82…`; verschieden trotz gleichen Namens. | Getrennte Varianten; keine davon auswählen oder umbenennen. Große Objekte nur gehasht, nicht dekodiert. |
| `verification/Data/my/pmip3_21k_sftlf_v0.nc` | `sftlf(lat=180,lon=360)`, Einheit `%`, kein `mask2`. | Kein Ersatz für die von `mcmc_prepare.R:451` verlangte `pmip3_21k_v0.nc` mit `mask2`. Keine Landfraktion in Eismaske umdeuten. |
| `…/data_in/cru/pann_cru4.nc`, `tann_cru4.nc` | Regionales Raster 149 × 97, eine Zeitposition; `pre` in `mm/month`, `tmp` in `degrees Celsius`. | Keine globale CRU-/Ankerfassung und keine stillschweigende Umrechnung zu PANN in mm/a. |

Vollständige Hashes: [candidate_hashes.json](candidate_hashes.json).
Geprüfte Objekte, Achsen, Einheiten und Grenzen:
[candidate_structures.json](candidate_structures.json).
Die beiden zusätzlichen regionalen CRU-Header sind dort ebenfalls gehasht.

## Tatsächlicher Umfang

Der bereitgestellte Sitzungsordner wurde über den privaten Konfigurationsschlüssel
`climate_modelling_source_root` absolut gebunden. Sein Pfad bleibt in
`.local/my_reconst_paths.json`; keine privaten absoluten Pfade oder Originalcontainer
werden exportiert. **Der neue Collector liest diesen Schlüssel tatsächlich und
läuft ausschließlich über die zusätzliche Quelle.** Der alte Scanner mit zwei
festen Wurzeln wurde weder geändert noch erneut ausgeführt.

- 1.828 Dateien per Pfad/Größe/mtime indexiert, darunter 395 RData, 343 NetCDF,
  vier CSV; [Metadatenindex](file_metadata.csv).
- 161 R-Dateien gelesen/gehasht; zehn mit mtime nach 2021. Im neuen Baum heißt
  keine relevante Quellunterstruktur wörtlich `src`: sieben dieser zehn Dateien
  liegen in `Source`, zwei sind `Main.R`, eine in `create_Data`. Auch diese wurden
  priorisiert geprüft, keine Beschränkung auf das Verzeichniswort `src`.
  [Quellhashes und Datumsbefunde](source_dates.csv).
- 111 gezielte statische Lese-/Schreibstellen zu Klima-/Isotopen-/Sammelständen;
  [Aufrufregister](targeted_source_calls.csv). Die aktuelle Quellenprüfung liefert
  keinen v4.0-Erzeugerbeleg. Eine Titelzeichenkette `v4.0` in `verification/Main.R`
  benennt Rekonstruktionsausgaben, nicht die gesuchte Sammeldatei.
- Ein ZIP ausschließlich über sein Inhaltsverzeichnis geprüft, nichts extrahiert;
  [Archivnachweis](archive_directories.json). Keine priorisierte Namenskopie darin.
- Die 1.178 alten Leseverweise nur anhand ihrer statischen Dateinamen gegen den
  neuen Index ergänzt: [57 Aufrufstellen mit Kandidaten](prior_reference_new_candidates.csv).
  Davon waren 24 vorher bereits am erwarteten Pfad vorhanden und 29 hatten schon
  Kopiekandidaten. Die vier vorher fehlenden Aufrufstellen sind F14/F15 in
  `main_v1.1.R` und `main_v1.2.R`; **keine 57 neu geschlossenen Lücken**.
- Acht relevante Dateien vollständig gehasht, zusätzlich zwei regionale CRU-Dateien
  bei der Headerprüfung. Zwei CSVs gelesen, ein RData-Container begrenzt gestreamt,
  drei NetCDF-Header/kleine Achsen gelesen. Große Pollencontainer nicht dekodiert;
  andere Datenfelder nicht durchsucht. Keine pauschale PC- oder Altbestandsinventur.

[summary.json](summary.json) dokumentiert Suchumfang und Fehlerliste (leer).
Dateizeitstempel dienen zur Priorisierung, nicht als historische Erstellungs-
oder Ausführungsbelege. Statische Parser lösen keine Bedingungen, Funktionsaufrufe,
`setwd` oder dynamische Dateifamilien vollständig auf. Das Fehlen eines Namens beweist keine
Abwesenheit gleichwertiger Daten unter anderen Namen.

## Prüfung und Fortsetzung

23 vorhandene Parser-/RData-Lesertests bestanden. Beide Phase-1-Prüfungen
(portabel und gezielt lokal) behalten erwarteten Exitcode 2 mit elf offenen
Voraussetzungen; ihr lokaler Umfang bleibt die bisherige 40-Pfade-Liste und
wird nicht als climate_modelling-Scan ausgegeben. Die früheren 200 numerischen
Teiltests wurden hier nicht erneut gerechnet. 133 Dateien der bisherigen Pakete und
Startaufträge sind bytegleich zum gesicherten Ausgangscommit erhalten; 110 relative
Dokumentationslinks geprüft.
[Prüfung, Nachweiserhaltung und Paketmanifest](verification.json).

```powershell
.venv/Scripts/python.exe docs/thesis_research/climate_modelling_audit_2026_11_02/collect_evidence.py
.venv/Scripts/python.exe docs/thesis_research/climate_modelling_audit_2026_11_02/inspect_candidates.py
.venv/Scripts/python.exe -m pytest tests/test_required_files_audit.py tests/test_local_rdata_metadata.py tests/test_local_rdata_stream.py -q -o cache_dir=.local/climate_modelling_audit_2026_11_02/pytest_cache
.venv/Scripts/python.exe tools/check_kinneret_phase1_spec.py
.venv/Scripts/python.exe tools/check_kinneret_phase1_spec.py --with-local
.venv/Scripts/python.exe docs/thesis_research/climate_modelling_audit_2026_11_02/verify_evidence.py
```

Keine Trainings, Sampler, Altersmodelle, R-Skripte oder Fossilrekonstruktionen
starten. Originale, Baseline, App/CLI, alte Belegpakete/Startaufträge und
fallbezogene Monticchio-/Chronologie-Gates bleiben erhalten. Phase 1 vor Phase 2,
Plots jung links, alt rechts. Nächster abgegrenzter Auftrag:
[Fortsetzung nach dem Abgleich](../NEXT_STEP_AFTER_CLIMATE_AUDIT_2026_11_02.md).
