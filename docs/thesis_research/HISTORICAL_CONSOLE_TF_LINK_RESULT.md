# Konsolenspur und grafische Taxon-TF-Ausgaben numerisch verbunden

Stand 02.11.2026. Vorhandener Checkout `dissertation-reconstruction`, Basis
`7f1826c` nach [PR #30](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/30).
Abgeschlossenes Quellenarbeitspaket. Die vollständige numerische
Dissertationsrekonstruktion bleibt offen; keine neue fachliche Freigabe.

## 1. Vollständiger Abgleich der Konsolen-Zeilenzahlen

Der bereits strukturell geprüfte gespeicherte 279er-Ohrid-Stand
`my_ohrid/data/in/climate_data/taxa_distribution.rdata` wurde ebenenweise
anhand seiner eigenen Koordinaten gelesen. Jede Ebene enthält auf der
Nordhalbkugel 54.025 endliche binäre Zellen. Es wurde kein Kinneret-Indexpfad
umgeschrieben und kein Raster ersetzt.

Der Quellkandidat `src/all_transferfunctions_v1.R` verwendet die Nordhalbkugel,
eine klassenweise 70-%-Trainingsaufteilung und `SMOTE(...)$syn_data`, ergänzt
um die unknown-Klasse. Die archivierte smotefamily-Fassung 1.3.1 enthält in
`syn_data` ausschließlich synthetische Minderheitsbeobachtungen. Die originalen
positiven Beobachtungen werden in diesem Quellzweig nicht wieder angefügt.

Für Präsenzzahl p und Absenzzahl n ergibt der geprüfte Kandidat:

- P = ceil(0,7 × p), N = ceil(0,7 × n)
- d = floor((N − P) / P), im tatsächlich betretenen SMOTE-Zweig
- protokollierte Zeilenzahl = N + d × P

Beispiel der ersten Position: p = 333, n = 53.692, P = 234, N = 37.585,
d = 159, Zeilenzahl = **74.791**. Dieser Wert steht tatsächlich im Protokoll.
Es wurden nur Kardinalitäten berechnet, keine synthetischen Klimawerte erzeugt.

| Vergleichshypothese | Passende Zeilenzahlen von 271 |
| --- | ---: |
| Nordhalbkugel, 70 %, nur synthetische Positive + unknown | **271** |
| Weltgitter, sonst gleicher Weg | 0 |
| Nordhalbkugel, originale positive Trainingswerte zusätzlich enthalten | 0 |
| Nordhalbkugel ohne SMOTE | 7 |
| Nordhalbkugel ohne Hold-out-Aufteilung, sonst gleicher Weg | 0 |

Alle 271 ausgegebenen Positionen betreten unter diesen Kandidatenzahlen den
im Code bedingten SMOTE-Zweig. Alle vier Modellmeldungen pro Position haben
dieselbe passende Zeilen-/Spaltenzahl: insgesamt 1.084 Datenformausgaben mit
fünf Spalten. Die acht Positionen ohne Modellmeldung sind genau die Ebenen
mit höchstens zehn Präsenzzellen:

| Gespeicherte Position | Präsenzzellen |
| --- | ---: |
| 9 | 10 |
| 16 | 4 |
| 142 | 9 |
| 149 | 6 |
| 151 | 1 |
| 162 | 9 |
| 167 | 6 |
| 206 | 8 |

Das ist ein vollständiger numerischer Nachweis eines zum Quellzweig passenden
Kardinalitätspfads. Es ist keine eindeutige Identifikation aller Eingabewerte:
251 verschiedene ausgegebene Zeilenzahlen bedeuten auch Kollisionen. Gleiche
Zellzahlen beweisen weder identische räumliche Zellmitgliedschaften noch
identische Klimawerte. Seeds, Aufteilungsindizes und die konkret installierte
Paketfassung folgen nicht aus den Zählungen.

Format-/Rechenquellen sind der lokal gesicherte
[caret-Quellstand 6.0-84](https://raw.githubusercontent.com/cran/caret/6.0-84/R/createDataPartition.R)
und das [CRAN-Archiv smotefamily 1.3.1](https://cran.r-project.org/src/contrib/Archive/smotefamily/smotefamily_1.3.1.tar.gz).
Die Archivfassung wurde ausschließlich als Text gelesen, nicht installiert
oder ausgeführt. Ihre Verfügbarkeit im Jahr 2019 belegt nicht ihre Verwendung
im konkreten historischen Lauf.

## 2. Variablenanalyse und spätere Transferfunktionen getrennt eingegrenzt

Der passende Fünf-Variablen-Quellzweig erstellt vier DALEX-Explainer und
speichert `var.imp.logloss`, `var.imp.rmse`, `var.imp.acu`.
Die vier Speicheranweisungen für einzelne nnet-/MDA-/QDA-/GBM-Fits sind dort
auskommentiert. Der Konsolentext ist daher ein belastbarer Hinweis auf einen
Variablenanalyse-Schritt; er wird nicht als erhaltene TF-Modellsammlung ausgegeben.

Der heutige Quelltext enthält weiterhin die aktive Überschreibung
`trafo.input <- titanic_imputed`. Diese Zuweisung und das fehlende Laufmanifest
verhindern eine Behauptung, dass genau die heutige unveränderte Datei ausgeführt
wurde. Der passende Klimadaten-Vorbereitungszweig lässt sich deutlich enger
eingrenzen als die vollständige ausgeführte Quellfassung.

Sechs erhaltene Variablenwichtigkeits-PDFs wurden vollständig gelesen und
visuell geprüft. Sie tragen die fünf Beschriftungen PANN, PWIN, TANN, TSUM,
TWIN. Ihre nativen Metadaten nennen R 3.5.0 und den 09.06.2021. Diese Angaben
beziehen sich auf die PDF-Erstellung, nicht auf einen belegten Trainingstermin.
Aus den Boxplots wurden keine numerischen Importance-Arrays zurückgewonnen.

Die unveränderten Einzelbefunde der bereits gelesenen Modellcontainer ergeben
167 caret-Fits mit jeweils zwei gespeicherten Prädiktoren: 165 mit
`t.djf.norm` / `p.ann.norm`, zwei mit `t.ann.norm` / `p.ann.norm`.
Darunter befindet sich kein gespeicherter Fünf-Prädiktoren-Fit. Es wurde keine
erneute Modellinventur durchgeführt.

## 3. 278 vorhandene TF-Konturgrafiken mit Titel und Punktzahl abgeglichen

Zusätzlich gezielt gelesen:
`local_reconstr_all_epd/out/plots/trafo/2D/all_trafos_emd.pdf`,
278 Seiten, SHA256
`e19b69a8574818e83c66f6c91c2e61d196cd39d2a28ad4074c2ac186eceb454c`.

Auf **allen 278 Seiten** wurde der eigentliche fett gesetzte Titel und die Zahl
der im PDF-Inhaltsstrom gespeicherten grünen Punktzeichen gelesen.
Jede Kombination stimmt mit genau einer Position des gespeicherten
279er-Ohrid-Datenstands und deren nördlicher Präsenzzahl überein.
Die 278 zugeordneten Positionen sind verschieden und ergeben genau alle
Positionen außer 151. Die Titel einschließlich ihrer Wiederholungen stimmen
vollständig überein; es gab keine unscharfe Namensangleichung.
Die PDF-Seitenreihenfolge wurde nicht als Taxonindex verwendet.

Position 151 trägt im gespeicherten Stand den Namen `Picea omorika` und hat
nur eine Präsenzzelle. Der spätere Quellkandidat `src/all_trafo.R` überspringt
explizit `length(taxa.precip[[1]]) == 1`; unter dem vorhandenen Datenstand ergibt
dies genau 278 Ausgaben. Die sieben weiteren seltenen Positionen, die im
Fünf-Variablen-Protokoll fehlen, sind in den TF-Konturgrafiken enthalten.
Beispiel: `Aesculus hippocastanum`, Position 16, vier Präsenzzeichen, PDF-Seite 76.

**Juglans separat:** PDF-Seite 12 heißt tatsächlich `Juglans sp.` und enthält
2.926 Präsenzzeichen, passend zu gespeicherter Position 110. Die grafische
Ausgabe liegt damit hinsichtlich ihrer Beschriftung vor der im Haupteinstieg
codierten Umbenennung zu `Juglans regia`. Daraus wird kein Kinneret-Pfad abgeleitet.

Die Titel-/Punktzahlbindung stützt die Herkunft der grafischen Ausgaben durch
Quellcode und tatsächliche PDF-Inhalte. Sie beweist nicht die Identität der
verwendeten Klimadateien oder aller Zellwerte. Keine Konturlinie wurde als
exakte Modellwahrscheinlichkeit übernommen. Die ursprünglichen 300er-
Marginalarrays und RData-Transferfunktionen sind dadurch nicht wiederhergestellt.

Drei vollständige Seiten einschließlich der seltensten dargestellten Art wurden
zusätzlich visuell geprüft. Poppler ließ die Punkt-Standardschrift in den
Vorschaubildern aus; ein zweiter, nur im neuen lokalen Nachweisordner
bereitgestellter MuPDF-Renderer zeigt die gespeicherten grünen Zeichen.
Die numerische Zählung verwendet die originalen Inhaltsströme und keinen
Screenshot. Original-PDFs und Projektumgebung wurden nicht verändert.

## 4. Konkrete Verbindung zum Rekonstruktions-Einstieg

`src/all_trafo.R` berechnet im späteren Zweig zwei Klimaprädiktoren,
vergleicht Modelle und gewichtet EMD-Rang und Balanced-Accuracy-Rang mit
0,75 / 0,25. Der gespeicherte Quellstand enthält weiterhin den bereits
festgestellten Rücksatz auf die vollständige Trainingstabelle. Kein neues Fit
wurde ausgeführt und diese Regeln wurden nicht still korrigiert.

Dieser Zweig schreibt `mixed_partial_arts_emd.rdata` mit den Feldern
`prod.x1`, `prod.x2`, `x1range.bck`, `x2range.bck` sowie eine separate
Info-Datei. `main_v1.2.R` lädt ausdrücklich denselben TF-Pfad, liest dieselben
Felder und baut daraus `tf.x1` / `tf.x2`. Das ist eine konkrete Schreib-/Lese-
Verbindung. Ein tatsächlich gespeicherter numerischer Container dieses Pfads
fehlt weiterhin in den bisher untersuchten Beständen.

Die aus expliziten historischen Verweisen abgeleiteten lokalen Verzeichnisse
`diss/reconstructions/cru_nh`, `diss/src/cru_nh` und die beiden entsprechenden
`climate_modelling/pollen/transfer_functions/input_data/data_out`-Pfade
existieren nicht. Das ist eine Prüfung dieser konkreten Pfade, keine Aussage
über sämtliche Laufwerke oder frühere Rechner.

## 5. Prüfung, Erhalt und nächster Einstieg

59 Tests bestanden, darunter 23 neue Fälle zur klassenweisen Rundung,
SMOTE-Kardinalität, Konsolen-Fehlstellen und konservativen PDF-Inhaltszählung.
Die bisherigen 36 Lesertests wurden mitgeprüft. Keine R-Ausführung,
Modelltrainings, Sampler oder Fossilrekonstruktionen. Keine Git-Veröffentlichung.

Maßgebliche Evidenz liegt unter `.local/console_provenance_v1/`:
`cardinality/console_cardinality.json`, `emd_plot_pages.json`,
`source_bindings.json`, `stored_model_predictor_crosscheck.json`,
`package_source_manifest.json`, `pdf_inspection.json`, `pytest_v2.xml`
und `final_verification.json`. Der erste `cardinality_probe.json` ist eine
vorläufige Diagnose; die maßgebliche Wiederholungsprüfung ist der neue
Kardinalitätsaudit. Vorherige Nachweise und Dokumentabschnitte bleiben erhalten.

**Als Nächstes:** Am jetzt materiell belegten 278-Seiten-TF-Plotstand ansetzen.
Den zweiten erhaltenen Sammelplot `all_trafos.pdf` und die zugehörigen
Quellvarianten gezielt gegen denselben Datenstand prüfen; Plotkoordinaten und
Klimaskalen können zusätzliche Datenkonsistenztests liefern. Die Suche nach
den numerischen `mixed_partial_arts_emd`-Ausgaben und den globalen Klimaeingaben
mit Zeitachse bleibt erforderlich. Grafisch abgeleitete Näherungen benötigen
eine getrennte fachliche Entscheidung, bevor sie fehlende TF-Arrays ersetzen
könnten; hier wurde kein solcher Ersatz vorgenommen.

Der regionale Rastererzeuger, die 95.006 Rasterabweichungen und die globale
Klimazeitachse bleiben offen. Paper- und Sieben-Arten-Ergebnisse, App/CLI,
Baseline sowie Monticchio-/Chronologie-Gates bleiben bewahrt. Jung links, alt rechts.
