# Plan: weitere Standorte (Birkat Ram, Totes Meer)

Stand 29.10.2026. Vom Autor gewünscht: analoge Rekonstruktionen für **Birkat Ram** und das **Tote Meer** mit
derselben Biom-Transferfunktion und eigenen Taxa-Gewichten je Standort. Die Unsicherheiten der Alters-Tiefen-Modelle
stammen aus der Dissertation Netzel (2022). Die Daten liefert der Autor nach (Abschnitt 2); bis dahin lassen sich
Schritt 1 und 2 umsetzen.

Grundlage ist das Standort-Register aus App Schritt 5 (`docs/app_konzept.md`, Abschnitt 13): ein Standort ist ein
Ordner `data/<ordner>/` mit den Eingangsdaten und `standort.toml`.

## 1. Ausgangslage

**Schon standortfähig:** Register mit Lage, Höhe, Biomen und CRU-Einstellungen; `cru-prior`/`cru-extract` mit
`--standort`; Lauf-Katalog mit `standort` je Lauf; App mit Standort-Wähler ab zwei Standorten. Beide neuen Standorte
liegen im vorhandenen CRU-Ausschnitt `data/in/cru_levante.npz` (29–36,5° N, 32,5–39,5° O). Biome und Transferfunktion
sind dieselben wie bei Kinneret, daher passen auch Abbildungen und Bericht (Biome des Papers).

**Noch auf Kinneret festgelegt:**

| Stelle | Heute | Nötig |
|---|---|---|
| `config.py`, `CRU_PRIOR_50KM` | Voreinstellung des Priors des rezenten Klimas (11,09 ± 0,87 °C, 521,8 ± 41,4 mm) gilt für Kinneret | Prior je Standort (z. B. im Register, berechnet mit `cru-prior --standort`); davon hängt auch ab, welcher Lauf als Standardlauf eines Standorts gilt (`katalog_index`) |
| `data.py`, `ARBOREAL_TAXA`, `CULTIVATED_TAXA` | feste Taxa-Listen; bestimmen die Baumpollenkurve in $R^2$ und die Variante „AP ohne Kulturbäume“ | je Standort ins Register (`gehoelz`, `kultur`) |
| `nnet_fit_params.npz`, `recent_climate` | Prior des Papers für Kinneret (10,74 ± 2,48 °C, 556 ± 135 mm), in derselben Datei wie die Gewichte des Netzes | Transferfunktion gemeinsam nutzen (eine Datei, auf die das Register verweist), `recent_climate` je Standort |
| Umwandlung der Rohdaten | `core_data.npz` und `age_depth_relationship.npz` liegen fertig vor, ohne Werkzeug | Werkzeug, das sie aus Tabellen erzeugt (Abschnitt 3, Schritt 2) |
| Jüngste Altersstufe | Der Prior des rezenten Klimas gilt für die Altersstufe −50 cal BP (Gl. 7 und 8 des Papers, `docs/abgleich_paper.md`) | prüfen, wie das Modell einen Kern behandelt, der nicht bis zur Gegenwart reicht (Totes Meer) |
| Namen | Paketname `kinneret_recon`, App-Titel „Kinneret-Rekonstruktion“, Datei `kinneret-app.html`, Cache-Name | App-Titel und Dateinamen allgemein; Paketname bleibt |

**Formate der Eingangsdaten (Kinneret, zum Vergleich):**

- `age_depth_relationship.npz`: `depth_given_age` (Alter × Tiefe auf 1-cm-Raster, hier 181 × 1772) = $\Pr(\text{Tiefe} \mid \text{Alter})$,
  `age_regular` (50-Jahre-Raster, −50 … 8950 cal BP), `depth` (Probentiefen, 160), `age_mean` (mittleres Alter je Probe).
- `core_data.npz`: `taxa_spectrum_depth` (Proben × Taxa, Anteile), `biomes_assign` (Taxa × Biome, 0/1),
  `taxa_names`, `ap_depth` (Baumpollen in % je Probe).
- `nnet_fit_params.npz`: Gewichte des Netzes, Klimaraster `temp_range`/`pann_range`, `normal_params`,
  `recent_climate`.

## 2. Daten vom Autor (ausstehend)

1. **Alters-Tiefen-Modelle (Netzel 2022)** je Standort, möglichst als Stichproben des Altersmodells (Alter je Tiefe,
   viele Ziehungen) oder direkt als $\Pr(\text{Tiefe} \mid \text{Alter})$. Liegt nur die Dissertation (PDF) vor, muss das
   Altersmodell neu gerechnet werden (eigenes Teilprojekt, vorher planen).
2. **Pollendaten** je Kern: Anteile je Tiefe und Taxon (Tabelle), Probentiefen, Gesamtpollensumme oder AP in %;
   Kernbezeichnung und Quelle (Publikation, Datenbank).
3. **Biomzuordnung** der Taxa (wie bei Kinneret oder je Standort eigene Liste), Gehölz- und Kulturbaum-Taxa.
4. **Lage und Höhe** (Koordinaten, Seespiegel bzw. Höhe), gewünschtes Einzugsgebiet für den CRU-Prior (Kinneret: 50 km,
   Gauß-Gewichtung).
5. **Ablage:** Daten im Haupt-Repo wie bei Kinneret (`data/<ordner>/`, je Standort wenige MB) oder im privaten
   Daten-Repo.

## 3. Schritte

1. **Code auf mehrere Standorte umstellen** (ohne neue Daten möglich). Prior des rezenten Klimas, Gehölz- und
   Kulturbaum-Taxa und Verweis auf die Transferfunktion ins Register; `recent_climate` je Standort; App-Titel und
   Dateinamen allgemein. **Kinneret bleibt bitgleich:** Fingerabdrücke aller 27 Läufe (`kinneret-recon katalog`),
   `standort_kinneret.json` und `app.json` unverändert; Tests grün.
2. **Werkzeug zur Umwandlung** (`kinneret-recon standort-daten` o. ä.): aus Tabellen (CSV) und den Stichproben des
   Altersmodells `core_data.npz` und `age_depth_relationship.npz` erzeugen, mit Prüfungen (Anteile summieren auf 1,
   Tiefen innerhalb des Rasters, jede Zeile von $\Pr(\text{Tiefe} \mid \text{Alter})$ normiert). Probe: Kinneret aus
   denselben Tabellen nachbauen und mit den vorhandenen Dateien vergleichen (soweit die Rohtabellen vorliegen).
3. **Birkat Ram:** Daten umwandeln, `standort.toml`, CRU-Prior (`cru-prior --standort birkat_ram`), Standardlauf
   (Konfiguration und `lauf.toml`), Diagnostik prüfen (R-hat, ESS), Ketten im Daten-Repo sichern. Erste Bewertung:
   Gewichte, CI-Verhältnis der Transferfunktionen, Plausibilität gegen die Literatur.
4. **Totes Meer:** wie Birkat Ram; zusätzlich die jüngste Altersstufe klären (Abschnitt 1), ggf. längeres
   Altersraster.
5. **Vergleich der Standorte:** in der App eine Ansicht oder Erweiterung der Sensitivität (gleiche Faktoren je
   Standort), Bericht im Prüfbericht; erst nach Schritt 3 und 4 genauer planen.

**Offen zu entscheiden (mit dem Autor):** Prior der Taxa-Gewichte je Standort (Vorschlag: hierarchisch wie Standard
v3); Sensitivitätsläufe je Standort (Vorschlag: zunächst nur Standardlauf, dann CRU-Radius und Gewichte-Prior);
Altersraster und Kalibrierungszeitraum je Standort.
