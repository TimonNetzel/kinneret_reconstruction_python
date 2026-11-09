# Zusatzquelle my_reconst: Ergebnis des Quellenabgleichs

Stand 01.11.2026. Der Nutzer hat den zusätzlichen lokalen Ordner `my_reconst`
ausdrücklich zur Prüfung benannt. Fortsetzung im vorhandenen Checkout auf
`dissertation-reconstruction`, Basis `7f1826c` nach
[PR #30](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/30).
Alle Originalbestände bleiben ausschließlich lesend.

**Die globale Verbreitungsgrundlage wurde gefunden.** Der historische Codepfad
von 283 zu 279 Taxonpositionen lässt sich nun an der tatsächlich vorhandenen
Ohrid-Datei nachvollziehen. Die fertigen 279 Transferfunktionen und die externen
Klimadateien des zuletzt gefundenen Erzeugers sind weiterhin nicht wiederhergestellt.

## 1. Geprüfter Umfang

Metadateninventar: **2.687 Dateien / 3.101.923.547 Bytes**, darunter 193 RData-
Dateien und 49 NetCDF-Dateien. Das ist keine vollständige inhaltliche Prüfung
aller Dateien. Gezielt untersucht wurden zwei globale Verbreitungscontainer,
zwei kleine gespeicherte Modellcontainer und die zugehörigen R-Quellabschnitte.
Die neun ZIPs mit zusammen 159 Mitgliedern wurden nur im Inhaltsverzeichnis
nach den sechs gesuchten Dateinamen durchsucht.

| Gesuchte Quelle | Ergebnis im benannten Ordner |
| --- | --- |
| `taxa_distribution_ww.rdata` | Zwei unterschiedliche Dateien unter `my_kinneret/data/in/climate_data/` und `my_ohrid/data/in/climate_data/` gefunden. |
| `tann_test.nc`, `pann_test.nc` | Keine exakten Dateinamenstreffer, auch nicht in den ZIP-Verzeichnissen. |
| `mixed_partial_arts_emd.rdata` und zugehörige Infodatei | Keine exakten Dateinamenstreffer. |
| `all_nnet_transfer_functions.rdata` | Kein exakter Dateinamenstreffer. |

Andere Dateinamen und noch nicht dekodierte Container können weitere Inhalte
enthalten. Insbesondere werden die 193 RData-Dateien nicht pauschal als
historische Taxonmodelle ausgegeben. Externe Ordner außerhalb des vom Nutzer
genannten Bereichs wurden nicht durchsucht.

## 2. Zwei globale Fassungen mit eigenen Koordinaten

Beide enthalten das Objekt `taxa.distribution.ww` mit `dist`, `lon`, `lat`,
`names`. Die globalen Achsen sind ausdrücklich gespeichert und aufsteigend:
720 Längengrade von −179,75 bis 179,75 °E und 360 Breitengrade von −89,75
bis 89,75 °N, jeweils in 0,5°-Schritten. Endliche Rasterwerte sind binär.

| Fassung | Form Länge × Breite × Taxon | Unkomprimierter numerischer Hauptblock |
| --- | --- | --- |
| Kinneret | 720 × 360 × 275 | 570.240.000 Bytes |
| Ohrid | 720 × 360 × 283 | 586.828.800 Bytes |

Die Kinneret-Namen entsprechen positionsgenau der ersten Namensspalte des
bisherigen lokalen 275er-Blocks, einschließlich der wiederholten Namen.
Ohrid enthält dieselbe geordnete Folge mit acht zusätzlich eingefügten Namen:
Carya glabra, Carya tomentosa, Cedrus libani, Liquidambar orientalis,
Pterocarya fraxinifolia, Tsuga canadensis, Tsuga diversifolia und Zelkova.
Nach Entfernen dieser zusätzlichen Einträge stimmt die Reihenfolge exakt.
Die **275 gemeinsamen vollständigen globalen Rasterebenen** sind mittels
kanonischer Binärprüfsummen identisch; es wurden nicht nur Namen oder Maße verglichen.

Der kommentierte Ohrid-Quellabschnitt Z. 113–174 beschreibt diese acht
Relikt-Ergänzungen samt anschließendem Sortieren; Z. 176 lädt die erhaltene
Datei. Kommentierter Erzeugungscode ist kein Ausführungsprotokoll. Der aktive
Kinneret-Zweig lädt Z. 168 die eurasische NetCDF-Quelle und Z. 211 den globalen
Container; der dazwischenliegende Rastererzeugungsvorschlag ist auskommentiert.

## 3. Konkreter Pfad 283 → 279 und Umbenennung

Der bereits geprüfte historische Haupteinstieg `main_v1.2.R`, Z. 433–436,
referenziert ausdrücklich die **Ohrid-Datei**, entfernt Positionen 8, 10,
152 und 153 und benennt danach Position 110 um. Mit der gefundenen Datei:

| Entfernte Position in der ursprünglichen Ohrid-Liste, R-Zählung ab 1 | Erhaltener Name |
| --- | --- |
| 8 | Abies nebrodensis |
| 10 | Abies numidica |
| 152 | Phoenix Dactylifera |
| 153 | Picea abies |

Danach verbleiben **279 Positionen**. Position 110 heißt vor der ausdrücklichen
Code-Umbenennung `Juglans sp.` und danach `Juglans regia`. Der lokale Nachweis
bewahrt ursprüngliche Position, Namen und resultierende Position; die Originaldatei
wurde nicht verändert. Das ist eine rekonstruierbare Namens-/Positionsliste,
noch kein Beweis für 279 trainierte oder gültige Funktionen in einem bestimmten Lauf.
Vor der Juglans-Umbenennung enthalten diese 279 Positionen **277 unterschiedliche
Namen**: Groenlandia densa und Zannichellia palustris bleiben jeweils zweimal
vertreten. Die Positionsliste wird deshalb nicht automatisch dedupliziert.

**Die Kinneret-Datei ist kein Ersatz für diesen Codepfad.** Dort bezeichnen
Positionen 152/153 Pinus brutia und Pinus cembra. Nach denselben vier Löschungen
blieben nur 271 Positionen; Position 110 wäre Juniperus phoenicea. Eine automatische
Übertragung der Indexoperationen würde daher andere Taxa treffen.

## 4. Globale Raster sind nicht identisch mit dem lokalen 275er-Block

Alle 149 × 97 Koordinaten des bisherigen lokalen Klimagitters wurden exakt
in den globalen Achsen wiedergefunden; keine Nächste-Nachbar-Zuordnung,
Interpolation oder Spiegelung. Nach positionsgesichertem Ausschneiden:

- Fehlwertmasken stimmen in allen 275 gemeinsamen Ebenen überein.
- **27 von 275 Ebenen** stimmen vollständig mit dem lokalen Block überein.
- **95.006 endliche 0/1-Zellen unterscheiden sich** über die 275 Ebenen.
  Beispiel: Ulmus minor an lokaler Position 266 hat 3.984 abweichende Zellen.
- Die gleichnamigen regionalen RData-Dateien im neuen Ohrid-/Kinneret-Ordner
  sind dagegen bytegleich zum bisherigen lokalen Block.

Damit handelt es sich nicht einfach um zwei auf unterschiedliche geographische
Fenster zugeschnittene identische Rasterfreigaben. Welche Bearbeitung die
Wertunterschiede verursacht hat, ist noch nicht belegt. Es wird keine Verschiebung,
Neuzuordnung oder Datenkorrektur daraus abgeleitet. Für historische Reproduktion
muss die im konkreten Codepfad referenzierte globale Fassung erhalten bleiben.

## 5. Gefundene Modelle sind teilweise fundortbezogene Mehrklassenmodelle

Zwei kleine Ohrid-Dateien wurden strukturell erfolgreich gelesen:
`best_fit_2D_v1.6.rdata` und
`qda_2D_unsmoted_nh_unknown_relict_v2.6.rdata`.
Beide enthalten ein `best.fit` der Klassen `train`/`train.formula`, Methode `qda`,
mit Klassen `taxa1` bis `taxa6` plus `unknown`, dazu Normierungsparameter und
`logloss.df`. Sie enthalten jeweils **einen Mehrklassenfit**, keine Sammlung von
279 unabhängigen binären Artenfunktionen.

Auch der gespeicherte Dateiname `transferfunction.rdata` belegt eine solche
Sammlung nicht: `my_ohrid/src/ml_competition_ww.R`, Z. 555–556, speichert unter
diesem Namen `best.fit <- all.models$nnet` plus vier Normierungsparameter.
Ein erster Leseversuch dieses größeren caret-Containers endete an dem von der
installierten Python-Bibliothek nicht unterstützten `RObjectType 252`.
Er gilt ausdrücklich **nicht als erfolgreich dekodiert**; eine R-Laufzeit ist
hier nicht verfügbar. Keine Modelle geladen/ausgeführt oder neu trainiert.

## 6. Prüfungen und lokale Artefakte

Das neue Werkzeug `tools/audit_my_reconst_sources.py` liest die globalen
Hauptblöcke mit `tools/local_rdata_stream.py` ebenenweise. Ein vollständiges
globales Array wird nicht im RAM aufgebaut und nicht exportiert. Der Leser
nutzt die vorhandene `rdata`-Objektlogik, ersetzt große numerische Vektoren
vorübergehend durch ausdrücklich unbrauchbare leere Platzhalter und liest
die Rasterebenen anhand der erhaltenen Dimensionen separat. R-Code wird nicht
ausgeführt. Parser und Prüfsummen sind im Bericht festgehalten.

**Sieben Lesertests bestanden:** RData-Versionen 2/3, mit/ohne gzip, jeweils
gegen den regulären vollständigen Leser; außerdem Abbruch bei abgeschnittenem
Payload, verändertem Payload-Hash und unpassenden Dimensionen.
**Zehn Quellenprüfungen bestanden**, darunter genaue Namensreihenfolge,
gemeinsame Rasterhashes, 283→279, binäre Werte und unveränderte Metadaten beider
Originalbestände sowie zehn gezielte Datei-Prüfsummen. Rasterabweichungen werden
als Befund ausgewiesen und nicht als bestandener Gleichheitsnachweis umetikettiert.

Die abschließende unabhängige Kontrolle bestätigt zusätzlich die lückenlose
279er-Positionsfolge, Juglans an Position 110, die ausdrücklich hypothetische
Kennzeichnung der Kinneret-Indexoperation, den stabilen Quellkatalog und 16
zuvor festgehaltene Quellen-/Referenzhashes. Neun Abschlussprüfungen bestanden;
Artefakthashes und Linkprüfung unter `.local/my_reconst_verification.json`.

Aktueller lokaler Nachweis: `.local/my_reconst_source_audit_v3/` mit `audit.json`,
`source_inventory.csv`, `source_spans.json`, `global_taxon_crosswalk.csv` und
`global_plane_summary.csv`. Persönliche Pfade stehen nur in
`.local/my_reconst_paths.json`. Frühere Zwischenstände bleiben unter
`.local/my_reconst_source_audit/` und `.local/my_reconst_source_audit_v2/`.
Die aktuelle CSV kennzeichnet Indexoperationen auf Kinneret ausdrücklich als
rein hypothetischen Vergleich, der nicht angewendet werden darf.

```powershell
.\.venv\Scripts\python.exe tools/audit_my_reconst_sources.py --config .local/my_reconst_paths.json --output .local/my_reconst_source_audit_v3
```

Für eine Wiederholung einen neuen Zielordner wählen; vorhandene Ziele und
Ausgaben in einem der Originalbestände werden verweigert.

**Weiter offen:** historische fertige Taxon-TFs samt Version, externe globale
Klimadateien und deren tatsächliche Zeitachse, Lauf-/Paketprovenienz sowie die
Herkunft der 95.006 Rasterabweichungen. Der neue Namens-Crosswalk ermöglicht
gezieltere weitere Quellenarbeit, hebt aber keine fachlichen Fossil-/Chronologie-
Gates auf. Paperreferenz und Sieben-Arten-Ergebnisse bleiben erhalten; keine
langen Rechenläufe wiederholt, keine neuen Klimakurven und keine App-Änderungen.
Darstellungsvorgabe unverändert: jung links, alt rechts.

## Fortsetzung desselben Tages: Herkunft und weitere Modellcontainer

Der neue [Herkunfts- und Provenienzbericht](HISTORICAL_PROVENANCE_RESULT.md)
ergänzt diesen Stand: 95.006 Rasterabweichungen auf die vorgelagerte
RData-/NetCDF-Fassung eingegrenzt; tatsächlich gespeicherter 279er-
Verbreitungsstand vor Juglans-Umbenennung vollständig geprüft; sechs binäre
QDA-Teilmodelle und 15 QDA-Paarmodelle strukturell gefunden. Klimazeitgrundlagen
1901–2015 und 1961–1990 durch gespeicherte Zeitgrenzen unterschieden.
Historische 279er-TFs, globale test.nc-Dateien und konkreter Lauf weiterhin offen.
Aktuelle Nachweise und nächster Einstieg: neuer Bericht und
`.local/provenance_continuation_v1/final_verification.json`. Frühere Befunde
oben bleiben als vorangegangener Prüfstand vollständig erhalten.
