# Import-Arbeitspaket für Claude

Stand 29.10.2026. Forschungsdatenprüfung zu PR #11. Kein Auftrag, ungeprüfte Daten bereits zu assimilieren.

## 1. Gesperrte Originaldateien beziehen

`sources.json` enthält tatsächliche Download-URLs, Größen und SHA256-Werte. Große Originaldateien bleiben außerhalb dieses Dokumentations-Branches. Vom Verzeichnis dieser Datei aus:

```bash
python verify_sources.py --raw-dir /path/to/netzel_raw --output /path/to/audit_results.json --download
```

Der Download ergänzt nur fehlende Dateien. Bei Prüfsummenabweichung stoppen und einen neuen Quellenstand registrieren; bestehende Originale nicht ersetzen. Ein Download ist keine Modellfreigabe. Für Monticchio die CC-BY-NC-3.0-Lizenz der Zählungen separat von den CC-BY-3.0-Chronologien berücksichtigen.

## 2. EMPD2 als neue Pollen-Baseline vorbereiten

Datei `empd2.sqlite3`, in sich konsistente Tabellen statt eines Mischimports mit dem abweichenden PANGAEA-Klima-TSV verwenden. Beobachtete Schemas:

| Tabelle | Schlüssel und benötigte Felder |
|---|---|
| `metadata` | `samplename` PK; `longitude`, `latitude`, `elevation`, `locationreliability`, `sampletype`, `samplecontext`, `agebp`, `ispercent`, `isdigitised`, `empd_version` |
| `p_counts` | PK (`samplename`, `var_`); `count` als NUMERIC, nicht INTEGER erzwingen |
| `p_vars` | `var_` PK; `acc_var_`, `original_varname`, `acc_varname`, `groupid` |
| `groupid` | `groupid`; `groupname`, `higher_groupid`, `used_in_sum`, `percent_values` |
| `climate` | `samplename` PK; `t_ann`, `p_ann`, Monats- und Saisonwerte |

Join-Startpunkt:

```sql
SELECT c.samplename, c.var_, c.count, v.original_varname,
       v.acc_var_, v.acc_varname, v.groupid,
       m.latitude, m.longitude, m.locationreliability,
       m.ispercent, m.isdigitised, k.t_ann, k.p_ann
FROM p_counts AS c
JOIN p_vars AS v USING (var_)
JOIN metadata AS m USING (samplename)
JOIN climate AS k USING (samplename);
```

Originalcodes immer erhalten. Erst nach Konzeptprüfung über `acc_var_` gruppieren; ähnliche Namen sind kein eindeutiger Schlüssel. Summendefinitionen mit `groupid` prüfen. Nicht jeden Datenbankcode als unabhängiges Pflanzentaxon verwenden. Abwesende Sparse-Zeilen erst nach Klärung des Datenprotokolls zu Nullen machen.

Abnahme des ersten Imports: 8.746 eindeutige Proben und 262.390 Messzeilen vor Filtern; keine verlorenen Messzeilen im Join. 1.059 gebrochene Zählfelder aus 244 Proben ausdrücklich markieren. Eine diskrete Count-Likelihood darf diese Werte ohne begründete Behandlung nicht verwenden. `agebp` bedeutet Jahre relativ zu 1950; negative Werte erhalten. Standortqualität `X` ausschließen; andere Qualitätsklassen, Meeresproben, Höhenfehler und räumliche Doppelungen vor Training getrennt auswerten. Die bisherige Boxauswahl liefert nur Kandidaten.

`t_ann` in °C und `p_ann` als Jahresniederschlag führen. Klimaherkunft WorldClim2 speichern. Eine CRU-basierte Neukalibrierung benötigt eigene Extraktion, Periode und Version. Räumliche Trainings-/Validierungsblöcke vor Modellwahl festlegen; nahe Proben nicht zufällig auf beide Mengen verteilen.

## 3. Monticchio als kurzen Zählungs-Piloten importieren

PANGAEA-Metadatenblock bis einschließlich `*/` getrennt erhalten, anschließend UTF-8-TSV lesen. `Depth sed [m]` ist keine Altersangabe. Zählungen aus 739452, Entity 804591 und **EpdAgeCut 804596** verbinden. Identität: `Event=LGM90D`, `ID=932`, Site `382`. MONTWAW1 bleibt separat.

Eine dokumentierte Zuordnung mit Decimal-Tiefen und maximal 0,005 m Abweichung liefert 69 eindeutige Partner, davon 61 exakt und acht mit Rundungsabweichung. Die Liste `results.monticchio_chronology.join` in `audit_results.json` ist der zu prüfende Zuordnungsvorschlag. Nicht nur auf abgerundete Fließkommazahlen joinen; Toleranz und Offset pro Probe speichern. Bei neuer Quellenversion den vollständigen Eindeutigkeitscheck erneut ausführen. Die übrigen 176 Zählproben für diesen Pilot ausschließen, Originale behalten.

Erwartete Abdeckung nach dieser Auswahl: 0,08575–20,22257 ka BP, größte Probenlücke 2,25895 ka. Multiplikation von ka BP mit 1.000 liefert die internen BP-Jahre; keine b2k-Verschiebung. `Age min`, `Age max` und Unsicherheitsklasse unverändert übernehmen. Das sind keine Altersensembles. Einen gemeinsamen chronologischen Prozess erst nach Prüfung der Datierungen oder publizierten Ensembles spezifizieren.

Summen für terrestrische Pollen fachlich definieren: Sporen, Volumen, Spike-/Konzentrationsfelder und unbestimmbare Reste sind keine automatisch einzubeziehenden Taxa. Abies ist kein eindeutiger Nachweis von Abies alba. Das beigefügte Mapping enthält nur konservative Konzeptentscheidungen, keine vollständige Harmonisierung.

## 4. Ohrid als getrennten Prozentdaten-Piloten erhalten

739452 und 892362 brauchen unterschiedliche Beobachtungsmodelle. Die Ohrid-Tabelle enthält `Age [ka BP]`, 19 ausgewählte Einzeltypen sowie AP und weitere aggregierte Spalten. Keine Tiefen oder vollständige Zählmatrix aus diesem Export ableiten. Für das Zielintervall 0–150 ka werden 90 Zeilen erwartet; keine Extrapolation von 1,42 ka bis heute oder vom letzten Messpunkt bis 150 ka.

Pro Spalte Originalname, Einheit und Pollensumme speichern. Corrigendum Tabelle 1: AP-/NAP-Basis ohne Pinus; Pinus-Header im Archiv: mit Pinus. Diese Definitionen nicht zu einer gemeinsamen 100-%-Komposition schließen. Rohdaten enthalten nur ausgewählte Taxa. Für eine ausgewählte Teilkomposition wäre ein explizites, fachlich begründetes Beobachtungsmodell erforderlich. AP und aus denselben Pollen berechnete Gruppen nicht als unabhängige Zusatzdaten zählen.

Chronologierolle und Abhängigkeiten speichern: zehn Tephren, biogeochemisches Orbitaltuning, gemeinsame DEEP-Chronologie. Punktalter nicht unabhängig zufällig verschieben. Fehlende Altersensembles bleiben ein Blocker für die quantitative gemeinsame Altersunsicherheit.

## 5. FloraWeb als separaten Kalibrierungsweg vorbereiten

Commit `5ab8e638278c86035bf3c9c3224c90de4880327e`, `Data_in/Areale/Mapping_areale.csv`. Felder: `name_choro_orig`, `name_use_id_orig`, `taxon_id`, `displayname`, `duplicate_synonym`. `taxon_id` und `name_use_id` nicht austauschen. Duplikatmarkierung respektieren; keine Abdeckung aus bloßer Zeilenanzahl folgern.

GeoJSON-Beispiel: `Data_in/Areale/geojson/taxon_id_1.geojson`. Feature-Properties: `geom_id`, `name_choro_orig`, `occurrence_status`, `occurrence_origin`, `taxon_id`, `name_use_id_orig`. Punkt- und Polygongeometrien getrennt behandeln; Länge/Breite. Prüfungen für Polygon-Gültigkeit, Datumsgrenze, Rasterung, Kartierungsumfang und fachliche Absenzen fehlen noch. Nicht alle Features als gesicherte native Präsenz behandeln. Fehlendes Olea im geprüften Mapping als Lücke führen.

## Abnahme und nächste Recherche

Der nächste PR sollte Importstatistik, ausgeschlossene Proben, Original-/Zieltaxa und ungelöste Zuordnungen exportieren. Keine Änderungen am bestehenden Kinneret-Datenlock. Vor Modelltraining braucht es eine vollständige Taxonkonzepttabelle, eine erklärte Behandlung gebrochener Zählungen und räumliche Validierung. Für einen langen Rohzählungspiloten anschließend eigenständige Langarchive und passende Datierungsdaten beschaffen. Iznik bleibt ein regionaler Zusatzkandidat mit Reservoir- und Kompositprüfung.
