# Getestetes Importpaket für Claude

Eigenständige Vorarbeit zur Integration; Python >=3.11, ausschließlich Standardbibliothek. Auf PR #12 aufgebaut. Der bestehende Kinneret-Rechenkern wird weder importiert noch geändert.

## Ausführen

Vom Repository-Hauptverzeichnis, nachdem dieses Paket unter `tools/thesis_import/` liegt:

```bash
python -m unittest discover -s tools/thesis_import -p 'test_*.py' -v
python tools/thesis_import/import_data.py \
  --raw-dir /path/to/netzel_raw \
  --output-dir /path/to/new_import \
  --download \
  --reference tools/thesis_import/reference_run.json
```

Das Zielverzeichnis muss neu sein. Ohne `--download` erfolgt kein Netzwerkzugriff. Es werden nur vier fehlende Quellen aus dem vorhandenen Register heruntergeladen: EMPD2-SQLite, Monticchio LGM90D, EPD-Entity und EpdAgeCut. Vor Verarbeitung müssen Größe und SHA256 stimmen. Vorhandene Quelldateien werden nicht überschrieben. `--catalog` erlaubt einen expliziten Registerpfad; der Standard ist `docs/thesis_research/source_audit_2026_10_29/sources.json`.

Der gespeicherte Referenzlauf enthält die erwarteten Kennzahlen und Prüfsummen aller Ausgaben. Mit `--reference` führt jede Abweichung zu einem Fehler, bevor das Zielverzeichnis bereitgestellt wird. Unterschiede prüfen, den Referenzstand nicht automatisch ersetzen. Eine Änderung am Importer ändert ebenfalls dessen im Manifest gespeicherte Prüfsumme.

## CSV-Schnittstelle, Version 1

Alle Dateien: UTF-8, Komma als Trennzeichen, Header, LF-Zeilenenden, Dezimalpunkt. Numerische Fehlwerte sind leere Felder, keine Nullen. Textspalten behalten Originaltexte. Schlüssel als Zeichenfolgen lesen. Keine Taxonaggregation und keine Ergänzung von Sparse-Zeilen.

| Datei | Inhalt und Schlüssel |
|---|---|
| `empd_samples.csv` | 8.746 Zeilen; `sample_id` eindeutig; Koordinaten, ursprüngliche Standortqualität, Probentyp, Alter, TANN/PANN und Qualitätsflags |
| `empd_measurements.csv` | 262.390 Zeilen; (`sample_id`, `original_taxon_code`) eindeutig; `count`, `is_fractional` |
| `empd_taxa.csv` | 5.230 Originalcodes; `original_taxon_code`, Originalname, akzeptierte ID/Name, `group_id`; einschließlich Codes ohne Messungen |
| `empd_groups.csv` | 40 ursprüngliche Gruppendefinitionen einschließlich `used_in_sum` und `percent_values`; keine neu festgelegte Pollensumme |
| `monticchio_samples.csv` | 69 Zuordnungskandidaten; `sample_id=LGM90D:<ursprüngliche Datenzeile mit sechs Stellen>`; beide Tiefen, Offset, Alter/Altersgrenzen in BP-Jahren, Unsicherheitsklasse |
| `monticchio_values.csv` | 9.315 Zeilen; (`sample_id`, `original_column`); Originalwert und Einheit, fehlend-/gebrochen-Markierung; auch Sporen, unbestimmbare Reste und Volumen bleiben erhalten |
| `monticchio_excluded.csv` | Alle 176 nicht zugeordneten Datenzeilen mit Tiefe und Ausschlussgrund |
| `report.json` | Umfänge, Flags, Zeitabdeckung, Tiefenabgleich und Status |
| `manifest.json` | Quell- und Ausgabefingerprints, Datenlizenzen sowie Importer-SHA256 |

Die endgültigen Header sind in den CSVs bzw. im kurzen Importer festgelegt. EMPD-Koordinaten, Höhe, Alter und Klima erhalten zusätzlich `*_raw`-Spalten. So bleibt beispielsweise das Originaltoken `"NULL"` sichtbar, während das zugehörige numerische Feld leer ist. `tann_c` und `pann_mm_per_year` bleiben als WorldClim2 gekennzeichnet; die konkrete Referenzperiode ist `unverified`. Kein CRU-Label ableiten.

## Gesicherte Sonderfälle

- 1.059 gebrochene Zählwerte aus 244 EMPD-Proben bleiben unverändert. Kein Runden und keine Freigabe für eine diskrete Count-Likelihood.
- **Korrektur gegenüber dem ursprünglichen Prüfbericht in PR #12:** 529 Klimazuordnungen fehlen als Text `"NULL"`; SQL `IS NULL` erkennt sie nicht. Nur 8.217 Proben besitzen numerisches TANN/PANN. 284 Proben haben Text-NULL-Koordinaten und Standortklasse X; 4.136 eine unbekannte Standortklasse `"NULL"`. Alle Proben bleiben für die Prüfung erhalten, Qualitätsflags kennzeichnen diese Fälle. Das ist kein Trainingssatz mit 8.746 freigegebenen Proben.
- EMPD `age_bp` kann negativ sein. Fehlende Werte werden nicht ersetzt. Proben- und Klimazeilen dürfen beim Join nicht verschwinden; verwaiste Messzeilen führen zum Abbruch.
- Monticchio: Entity 932/Event LGM90D, zulässige Tiefe aus Entity-Tabelle. Genau ein Partner pro Cut-Tiefe bei maximal 0,005 m; keine Wiederverwendung, keine Mehrdeutigkeit und keine Extrapolation. Erwartet: 61 exakte und acht gerundete Treffer, 85,75–20.222,57 BP-Jahre.
- Monticchio-Altersgrenzen sind marginale Angaben, keine gemeinsamen Altersensembles. Die Tiefe bleibt ein Zuordnungskandidat. Die 135 Datenfelder pro Probe sind keine Liste von 135 unabhängigen Pollentaxa.

Die elf Tests prüfen diese entscheidenden Datenverträge und Fehlerfälle mit synthetischen Fixtures. Zusätzlich wurde der vollständige Originaldatenimport ausgeführt und ein zweiter Lauf gegen den Referenzstand geprüft.

## Was Claude noch übernehmen soll

Nur den Projektzugang bzw. CLI-Wrapper und passende Integrationstests ergänzen. Die Importlogik übernehmen und zunächst unverändert gegen die Referenz prüfen. `CLAUDE_TASK.md` ist der kurze Arbeitsauftrag; die langen Forschungsberichte braucht Claude nur bei konkreten fachlichen Fragen.

Keine Modellfreigabe: Standortfilter, vollständige Taxonkonzept-Harmonisierung, Pollensummen, räumliche Trainingsvalidierung und Behandlung gebrochener Zählungen sind nachfolgende fachliche Entscheidungen. Sämtliche EMPD-Daten tragen `scientific_status=not_assessed`, Monticchio `candidate_depth_link`. Ohrid und FloraWeb sind nicht Teil dieses Importers.

Datenlizenzen gelten weiter: EMPD2 CC BY 4.0; Monticchio-Zählungen CC BY-NC 3.0; EPD-Chronologien CC BY 3.0. Die großen Originale und vollständigen CSV-Exporte werden nicht im Git-Branch mitgeführt; sie sind aus den gesperrten Quellen reproduzierbar.
