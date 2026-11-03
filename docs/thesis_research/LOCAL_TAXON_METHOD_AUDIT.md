# Lokaler Taxon- und Methodenabgleich – 01.11.2026

**Aktueller Nachtrag:** Nach Nutzerhinweis wurde die zusätzliche Quelle geprüft.
Die globale Verbreitungsdatei ist inzwischen gefunden; der historische Ohrid-
Codepfad 283→279 ist konkret nachvollzogen. Fertige TF-Container und test.nc
bleiben offen. Die folgenden Befunde beschreiben den vorherigen Suchumfang;
aktueller Gesamtbefund: [MY_RECONST_SOURCE_RESULT.md](MY_RECONST_SOURCE_RESULT.md).

Fortsetzung auf dem vorhandenen Branch `dissertation-reconstruction`, Basis
`7f1826c`. [PR #30](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/30)
erneut über GitHub als gemergt bestätigt. Vorhandene lokale Änderungen erhalten.
Keine neuen Modellfits, Samplerläufe, Fossilrekonstruktionen oder Uploads.

**Ergebnis:** Der 275-Ebenen-Block lässt sich dem lokalen europäischen Klimagitter
mit belegter Orientierung zuordnen. Der historische globale TF-Zweig ist jetzt
bis zu seinem Erzeugungsskript zurückverfolgt, seine Eingaben und Ergebnisse
fehlen jedoch weiterhin im untersuchten Bestand. Beide Zweige sind nicht austauschbar.

## 1. Geographische Zuordnung des lokalen 275er-Blocks

Zwei voneinander verschiedene Belege ergänzen die bisher allein bekannte Form:

1. `src/climate_input.R`, Z. 6–21, lädt ausdrücklich den kleinen Verbreitungsblock
   und genau die beiden lokalen NetCDF-Dateien. Z. 77–104 entnimmt den Klimafeldern
   die mit `which(taxa.appear[,,j] == 1)` bestimmten Zellen. Es wird keine
   Koordinatensuche oder Achsenspiegelung dazwischengeschaltet.
2. Alle 275 Ebenen haben dieselben **4.735 fehlenden Zellen**. Ihre Maske stimmt
   zellgenau mit beiden Klimafeldern überein. Die drei Achsenspiegelungen ergeben
   jeweils 5.330, 6.940 und 8.362 abweichende Maskenzellen.

| Merkmal | Nachgerechneter Befund |
| --- | --- |
| Arrayachsen im lokalen R-Verbraucher | Länge, Breite, Taxonposition: 149 × 97 × 275 |
| Längengrade | −12,75 bis 61,25 °E, aufsteigend in 0,5°-Schritten |
| Breitengrade | 27,25 bis 75,25 °N, aufsteigend in 0,5°-Schritten |
| Verwendbare Zellen je Ebene | 9.718; beide Klimafelder dort endlich |
| Werte des Taxonblocks | ausschließlich 0, 1 und fehlend |
| NetCDF-Anordnung beim Python-Lesen | `time, lat, lon`; einzelne Zeitstufe ausdrücklich nach `lon, lat` transponiert |

Damit ist die Zuordnung **für diesen lokalen Verbraucherzweig gestützt** und
die bisher offene Achsenorientierung geklärt. Ein Erzeugungsskript des binären
Blocks, die Kartendigitalisierung und die Übereinstimmung mit einer historischen
globalen Datenfreigabe sind damit nicht nachgewiesen. Die Klimametadaten bleiben
CRU TS4.00 / Ausgangsdaten 1901–2015; der im Text genannte CRU-4.01-Zeitraum
1961–1990 wird dadurch nicht ersetzt. Das Niederschlagsattribut `mm/month` und
die jährliche Aggregationshistorie bleiben unverändert dokumentiert.

## 2. Namen und Positionen sind kein eindeutiger Taxonschlüssel

275 Arrayebenen entsprechen **272 unterschiedlichen Zeichenketten in der ersten
und 271 in der zweiten Namensspalte**. Keine Namenszelle fehlt; 17 Zeilenpaare
unterscheiden sich zwischen den Spalten. Die R-Zeilenlabels sind nicht identisch
mit den fortlaufenden Arraypositionen. Der lokale CSV-Nachweis bewahrt beides.

| Wiederholter Name | Arraypositionen, R-Zählung ab 1 | Präsenzzellen der jeweiligen Ebenen |
| --- | --- | --- |
| Groenlandia densa | 99, 100 | 222, 359 |
| Picea abies | 149, 150 | 2.699, 2.229 |
| Zannichellia palustris / zweite Spalte Zannichellia-type | 273, 274 | 1.542, 4.518 |
| Ulmus minor, nur zweite Spalte | 263, 266 | 2.404, 3.893 |

Die gleichnamigen Paare besitzen jeweils unterschiedliche binäre Raster.
Ein namensbasierter Join mit anschließender Dublettenlöschung würde Informationen
verlieren. Umgekehrt ergeben die 275 Ebenen nur **233 unterschiedliche binäre
Raster**; identische Raster sind kein Nachweis identischer Taxa.
**16 Ebenen haben keine Präsenzzelle** im vorliegenden Ausschnitt. Das belegt
weder globale Abwesenheit noch eine trainierte historische Transferfunktion.

Schreibweisen, Synonymkandidaten und Ebenen werden nicht korrigiert oder
zusammengeführt. Insbesondere ist die zweite Namensspalte ohne weiteren Beleg
kein geprüfter allgemeiner Arten-/Pollentyp-Crosswalk.

## 3. Historische Transferfunktionen: konkreter Erzeuger gefunden

`main_v1.2.R`, Z. 569, lädt `mixed_partial_arts_emd.rdata`.
**`src/all_trafo.R`, Z. 528–529, schreibt genau diese Datei und ihre Infotabelle.**
Das ist ein konkreter Codepfad, aber noch kein Beweis, dass ein bestimmter
Dissertationslauf genau diesen erhaltenen Skriptstand verwendet hat.

Dieser Erzeuger lädt eine externe globale Verbreitungsdatei und entfernt
Positionen 8, 10, 152 und 153 (Z. 119–126). Der Haupteinstieg benennt außerdem
Position 110 als Juglans regia um (Z. 433–436). Diese Operationen werden nicht
auf den lokalen 275er-Block übertragen. Eine numerische Beziehung zwischen
275 Ebenen und den bisher dokumentierten 279 historischen Funktionen ist kein
belegter Crosswalk.

Die erwarteten historischen Modellcontainer fehlen an den geprüften relativen
Pfaden. Auch die Verzeichnisse aller **neun lokalen ZIPs mit zusammen 1.324
Einträgen** enthalten keinen exakten Dateinamenstreffer für die im Audit
gesuchten fehlenden RData-Abhängigkeiten. Es wurden nur ZIP-Verzeichnisse
gelesen. Andere Objektnamen in großen Containern und externe Verzeichnisse
sind damit nicht ausgeschlossen; diese wurden nicht pauschal dekodiert/durchsucht.

Zusätzlich wurde der vollständige GitHub-Dateibaum am gesperrten Basiscommit
`7f1826c` lesend abgefragt: 670 Einträge, nicht abgeschnitten, keine Treffer für
die sechs konkret gesuchten globalen Daten-/TF-Dateinamen. Das betrifft nur den
versionierten Dateibaum, keine Releases oder externen Speicherorte. Nachweis:
`.local/remote_source_tree_check.json`. Ein weiterer lokaler Fundort wurde beim
Nutzer erfragt; eine Antwort wird nicht vorausgesetzt.

## 4. Unterschiedliche Trainingsstände ausdrücklich trennen

Der Dissertationsausschnitt `Phd_thesis_netzel_ch2.5.PDF`, gedruckte S. 40–43
(PDF 3–6), beschreibt Normierung, binäre Artenmodelle, 70/15/15, SMOTE nur
im Training und ein einfaches `nnet` mit einer verborgenen Schicht. Der lokale
LaTeX-Archivtext `thesis_tf_results.tex`, Z. 32–99, beschreibt Auswahl nach
Balanced Accuracy und den Vergleich mit/ohne SMOTE. Die Fassungsgrenzen aus
[LOCAL_SOURCE_CROSSWALK.md](LOCAL_SOURCE_CROSSWALK.md) gelten weiterhin.

| Erhaltener R-Quellstand | Konkretes Verhalten vor dem ersten `stop()` |
| --- | --- |
| `all_transferfunctions_v1.1.R` | Binäres `nnet`, Seed 2019, fünffache CV mit `logLoss`, SMOTE K=1; Normalstandardisierung für T, Logistic-CDF der Wurzel von P; Ausgabe enthält Modell und 300×300-Prognosegitter. |
| `all_transferfunctions_v1.2.R` | Zehn Fits: fünf Algorithmen mit/ohne SMOTE, CV-Ziel `logLoss`, nachgelagerte Wahl über BA-Spalte 16; Gamma-CDF der Wurzel von P; Wiederaufnahme ab Taxon 27 mit geladenen Zwischenständen; Ausgabe eindimensionale Randkurven. |
| `all_trafo.R` | Seed 2023, künstliche Absenzpunkte, Gamma-CDF, `nnet` size=6/decay=0,015, CV-Ziel `Balanced_Accuracy`; Auswahl aus den ersten fünf SMOTE-Fits über 0,75 EMD-Rang + 0,25 BA-Rang; Ausgabe `mixed_partial_arts_emd.rdata`. |

**Konkrete Abweichungen im gefundenen Erzeuger, statisch am Code belegt:**

- Z. 45–48 überschreibt die zuerst geöffneten CRU-Dateihandles mit externen
  `tann_test.nc` und `pann_test.nc`. Z. 106–113 mittelt die Zeitindizes 61:90.
  Ohne deren Zeitmetadaten belegt das weder Kalenderjahre 1961–1990 noch CRU 4.01.
- Z. 75–99 erzeugt künstliche Absenzdaten; Z. 191–194 fügt sie **vor** der
  Aufteilung hinzu. Die als Tests bezeichneten Tabellen sind in diesem Skript
  deshalb nicht nachweislich auf reale unveränderte Beobachtungen begrenzt.
- Normierungsparameter werden vor der Aufteilung geschätzt (Z. 213–223).
  Das trifft auch auf die beiden anderen hier verglichenen Trainingsskripte zu.
- Z. 243–249 erzeugt zunächst die 70/15/15-Tabellen. **Z. 264 setzt die
  Trainingstabelle wieder auf `df.ML.compet.orig` zurück.** Z. 281/322 verwendet
  diese vollständige Tabelle für das Netz ohne SMOTE; Z. 273 übergibt sie auch
  an SMOTE. Bei Ausführung dieses Pfads wäre die äußere Trennung aufgehoben.
  Eine dort berechnete Testkennzahl wäre keine unabhängige Hold-out-Bewertung.
- Z. 425–463 berücksichtigt beim EMD-Ranking nur Fits 1:5, obwohl zehn Fits
  vorliegen. Gewichtet werden **Rangplätze**, nicht rohe EMD- und BA-Werte.
  Beim BA-Fallback werden auch die zweiten Testhälften herangezogen.
- `for(k in length(cut_offs))` (Z. 379/396) durchläuft nur den letzten Index,
  also die Schwelle 0,9. Das ist keine Schleife über alle neun Schwellen.
  Die daraus gespeicherte Zusatzspalte ist von BA-Spalte 16 zu unterscheiden,
  die Z. 457 für das Ranking verwendet.

Diese Befunde betreffen die erhaltenen Quellen. Ohne Originaleingaben,
gespeicherte Modelle, Paketversionen und Laufprovenienz werden sie **nicht**
als nachgewiesene Fehler sämtlicher publizierter Ergebnisse ausgegeben.
Der unabhängig räumlich geprüfte Sieben-Arten-Vergleich aus PR #29 bleibt
unverändert; er wird nicht durch diese historischen Trainingsskripte ersetzt.

## 5. Drei unterschiedliche Kombinationsebenen

Der frühere Befund einer gewichteten Summe gilt für die Rekonstruktion mit
bereits gewählten Taxonfunktionen. Dazu kommt nun eine vorgelagerte Unterscheidung:

- `mcmc_prep_v1.2.R`, Z. 1322–1370: mehrere einem Pollentyp zugeordnete
  Funktionen werden summiert und normalisiert.
- `mcmc_prepare.R`, Z. 1543–1568: derselbe Mehrfachzuordnungszweig berechnet
  zunächst Summe und Produkt. **Beide werden anschließend überschrieben:**
  Aus jeder gültigen Quellfunktion werden 1.000 Klimawerte gezogen, durch die
  Zahl der zugeordneten Funktionen geteilt und addiert; eine Kerndichte dieses
  Mittels wird als letzte Funktion gespeichert. Das ist eine andere Operation
  als eine Mischung oder ein Produkt von Dichten.
- `mcmc_functions_v1.6.R`, Z. 1242–1284 / 1781–1820 und 450–477: anschließend
  gewichtete Taxonanteile bzw. Matrixprodukte zur Klimakurve/finalen Dichte,
  wie bereits im Quellenabgleich belegt.

Die Art-/Pollentyp-Zusammenführung, Kombination zwischen Taxa und Kombination
von Posteriorziehungen dürfen nicht als eine einzige Regel behandelt werden.
Es wird keine dieser Varianten stillschweigend in die App übernommen.

## 6. Reproduzierbarer Nachweis und Fortsetzung

```powershell
.\.venv\Scripts\python.exe tools/audit_local_taxon_sources.py --config .local/paths.json --output .local/taxon_source_audit_v2
```

Der gezeigte Zielordner ist bereits vorhanden; zum Wiederholen einen neuen
Zielordner unter `.local/` wählen. Das Werkzeug verweigert Überschreiben und
Ausgaben im Originalbestand. Es dekodiert nur den ausdrücklich ausgewählten
kleinen RData-Block, führt kein R aus und startet keine Trainings-/Samplerläufe.

Aktueller Nachweis: `.local/taxon_source_audit_v2/` mit `audit.json`,
`taxon_inventory.csv` und gehashten `source_spans.json`. Acht Prüfungen bestanden:
binäre Werte, gemeinsame Masken, gleiche Klimakoordinaten, zwei Maskenabgleiche
als gemeinsamer Prüfpunkt, endliche Klimawerte, Originalmetadaten während des
Audits unverändert, elf Quellenhashes unverändert und Abgleich zum vorhandenen
1.783-Dateien-Katalog bestanden. Der frühere Audit ohne `all_trafo.R` bleibt
als Zwischenstand unter `.local/taxon_source_audit/` erhalten.

Zusätzliche unabhängige Kurzprüfung: **neun Prüfungen bestanden**. Alle 275
Ebenenzählungen skalar nachgerechnet, Namen/Positionen erhalten, 233 Raster per
Arrayvergleich bestätigt, gleichnamige Paare als unterschiedliche Raster geprüft,
Schreiben in Quelle/Unterordner/Quell-Elternordner und Überschreiben verweigert.
Alle drei Auditdateien beim Wiederholen bytegleich; 16 zuvor gesperrte Quellen-
und Referenzhashes unverändert. Nachweis mit Artefakthashes:
`.local/taxon_source_verification.json`; wiederholbarer Prüfer
`.local/verify_taxon_audit.py`. Keine erneute Kern-Testsuite, da Kern unverändert.

**Erledigt:** lokale Orientierung, Positions-/Namensinventar, Verbindung vom
historischen Verbraucher zum Erzeuger und methodische Trennung mit Zeilenbelegen.
**Offen:** globale Eingaben samt Zeitachse, historische Modelle/TF-Container,
deren geordnete Taxonliste und Lauf-/Paketprovenienz; große RData-Inhalte nur
bei konkretem Bedarf speicherschonend prüfen. Eine weitergehende Suche außerhalb
des ausgewählten Originalbestands ist nicht erfolgt.

Nächster fachlicher Schritt bleibt ein belegter globaler Taxon-/TF-Crosswalk.
Für neue Modelle gilt weiterhin das vorab festzulegende trainingsinterne
Kalibrierungsdesign mit unabhängigen äußeren räumlichen Tests. Keine neue
Fossilfreigabe; Monticchio-/Chronologie-Gates bleiben bestehen. Vorhandene
Referenzläufe werden weiterverwendet. Altersplots: jung links, alt rechts.
