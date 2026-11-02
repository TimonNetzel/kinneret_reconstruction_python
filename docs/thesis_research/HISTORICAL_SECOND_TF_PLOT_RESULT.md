# Zweiter TF-Sammelplot: Punktkoordinaten, Klimaskalen und Quellvarianten

Stand 02.11.2026. Abgeschlossenes Quellenarbeitspaket im vorhandenen Checkout
`dissertation-reconstruction`, Basis `7f1826cffddc8893ba6354cd3cca3fa43954b9be`.
[PR #30](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/30)
ist über das GitHub-Plugin erneut als gemergt mit genau diesem Merge-Commit
bestätigt. Keine vollständige numerische Dissertationsrekonstruktion und
keine neue fachliche Freigabe.

## 1. Zweiter Sammelplot vollständig zugeordnet

Gezielt gelesen wurde
`local_reconstr_all_epd/out/plots/trafo/2D/all_trafos.pdf`, 278 Seiten,
SHA256 `66b1db70abf1c61308c4b9fe7e2bfbc4c528153f4754c23c97b0b4a030beca69`.
Der vorhandene Rohinhalts-Audit wurde mit dem bereits belegten
`cardinality/console_cardinality.json` wiederverwendet. Keine neue
Modellinventur und kein erneutes Lesen der 78 Modell-Byteinhalte.

Alle 278 tatsächlichen Titel und grünen Punktzahlen passen jeweils eindeutig
zu einer Position des gespeicherten 279er-Ohrid-Stands. Die Positionsmenge
ist genau 1–279 ohne Position 151 (`Picea omorika`, eine Präsenzzelle).
`Juglans sp.` steht auch hier auf Seite 12, mit 2.926 Punkten, Position 110;
die spätere Umbenennung zu `Juglans regia` bleibt ein eigener Schritt.

Die Zuordnung wurde zuerst aus Titel und Punktzahl berechnet. Erst danach
wurde festgestellt, dass auch die Seitenreihenfolge beider Sammelplots im
Ergebnis übereinstimmt. Diese Reihenfolge ist kein Taxonindex; der Ohrid-Pfad
wird nicht auf Kinneret übertragen.

## 2. Alle gespeicherten Punktkoordinaten verglichen

Beide PDFs wurden zusätzlich vollständig mit dem vorhandenen MuPDF-Runtime
unter `.local/console_provenance_v1/pdf_runtime` gelesen. `get_texttrace`
erhält auch überlagerte Zeichen. Füllung und Umriss des PDF-Modus `Tr=2`
wurden getrennt gezählt und positionsweise geprüft, damit kein Punkt doppelt
gezählt wird. Titel, beide Punktzahlen und grüne Farbe stimmen auf allen
556 gelesenen Seiten mit dem Rohinhalts-Audit überein.

| Prüfung der 278 zugeordneten Seitenpaare | Ergebnis |
| --- | ---: |
| Vollständig identische Punktkoordinaten samt Reihenfolge | 23 Seitenpaare |
| Identische Punktmultimenge, Überlagerungen erhalten | dieselben 23 |
| Vollständig identische Achsentexte und Achsenvektoren | 269 |
| Zweite, minimal verschobene X-Achsenfassung im zweiten Plot | 9 |
| Identische gespeicherte Konturvektoren | 0 |
| Unterschiedliche Zahl gespeicherter Konturpfade | 135 |

Ein unabhängiger zweiter Koordinatencheck liest die tatsächlichen
`/F1 ... 2 Tr ... Tm (l) Tj`-Befehle aus den dekomprimierten PDF-Inhalten.
Er rechnet mit ganzen Hundertsteln eines PDF-Punkts statt mit gerenderten
Pixeln oder MuPDF-Gleitkommapositionen. Alle Befehle haben in diesen PDFs
zwei Nachkommastellen; die Zahl passt auf jeder Seite zur unabhängigen
Füllungs-/Umrisszählung.

Insgesamt **961.568** in Zeichnungsreihenfolge gepaarte Punkte:

| X-Verschiebung zweiter Plot minus EMD, PDF-Punkt | Anzahl |
| --- | ---: |
| −0,01 | 16.515 |
| 0 | 922.453 |
| +0,01 | 22.388 |
| +0,02 | 212 |

Die gespeicherte Y-Koordinate ist bei **allen 961.568 Paaren exakt gleich**.
Die größte X-Abweichung beträgt 0,02 PDF-Punkt. Auf der gemeinsamen
Temperatur-Tickskala entspricht diese Zeichnungsdistanz ungefähr 0,0031 °C.
Das ist eine Aussage über die PDF-Präzision, keine Genauigkeitsangabe zu den
ursprünglichen Klimadaten. Die Ursache der kleinen Unterschiede ist nicht
nachgewiesen; sie wird nicht pauschal als Rundungsfehler erklärt.

Die neun Seiten mit leicht verschiedenen X-Achsenvektoren sind 54, 104, 111,
120, 121, 150, 189, 213 und 227. Die Tickpositionen unterscheiden sich dort
um höchstens 0,01 PDF-Punkt; die Beschriftungen bleiben gleich. Im EMD-Plot
haben alle 278 Seiten dieselbe Achsenfassung.

Die enge Übereinstimmung der Beobachtungspunkte stützt eine gemeinsame
grafische Klimadatenbasis auf gespeicherter Präzision. Sie beweist weder
identische Dateien noch dieselben vollständigen Werte, Zellmitgliedschaften,
Zeitachsen, Seeds oder Modelle. Auch bei gleicher Punktzahl wird keine
räumliche Zellzuordnung aus dem Zeichnungsindex abgeleitet.

Die Konturen sind als gespeicherte Vektoren auf keiner Seite identisch.
Das beweist nicht 278 verschiedene ausgewählte Modelle: minimale
Koordinaten-/Beschriftungsänderungen können ebenfalls einen Vektorvergleich
ändern. Auf Seite 1 (`Groenlandia densa`) sind die unterschiedlichen
Konturverläufe zusätzlich klar sichtbar. Aus keinem der beiden PDF-Stände
wurden Modellwahrscheinlichkeiten oder TF-Arrays zurückgewonnen.

## 3. Tatsächliche Achsen und Transformationen

Die tatsächlichen Beschriftungen lauten `TANN [°C]` und `PANN [mm]`.
Alle Seiten tragen Temperaturticks −30 bis +30 in Zehnerschritten und
Niederschlagsticks 0 bis 10.000 in Schritten von 2.000. Das sind Tickwerte,
nicht die behaupteten Grenzen der fehlenden 300er-Arrays. Die Achsen sind
im dargestellten physischen Klimaraum linear; der PDF-Rahmen enthält
zusätzlichen Rand. Keine Gleichsetzung von Plotrand und Datenextremum.

Der spätere Erzeuger `src/all_trafo.R` bindet den Weg konkreter:

- Zeilen 45–52: CRU-Dateien werden zunächst geöffnet, dann werden die
  Variablen `temp` und `precip` durch `tann_test.nc` und `pann_test.nc`
  überschrieben. Gelesen werden `tmp`, `pre`, `lon` und `lat`.
- Zeilen 55–99: künstliche unknown-Klimapunkte entstehen aus unbelegten
  Histogrammfeldern vor der anschließenden zeitlichen Mittelung.
- Zeilen 106–113: Mittel über die **Zeitindizes 61:90**. Ohne die fehlende
  Zeitachse sind dies keine bewiesenen Kalenderjahre 1961–1990.
- Zeilen 175–204: Die grünen Plotpunkte stammen aus den ursprünglichen
  Präsenz-Klimawerten; künstliche unknown-Punkte werden separat ergänzt.
- Zeilen 217–229: Temperatur wird affin standardisiert. Niederschlag wird
  über Quadratwurzel, angepasste Gamma-CDF und Normalquantil transformiert;
  Nullwerte werden zuvor durch `1e-06` ersetzt.
- Zeilen 407–411: Je 300 gleichabständige Knoten im transformierten Raum;
  Rücktransformation `x1*sd1+mean1` und `qgamma(pnorm(x2),shape1,rate1)^2`.
  Das native Niederschlagsgitter ist damit im Allgemeinen nicht äquidistant.
- Zeilen 469–505: Temperatur läuft in Blöcken, Niederschlag innerhalb des
  Blocks. `matrix(..., byrow=TRUE)` erzeugt passend Zeile=Temperatur,
  Spalte=Niederschlag. Die Kontur wird gegen die rücktransformierten Achsen
  gezeichnet; die grünen Punkte sind die untransformierten Präsenzwerte.

Die Marginalen summieren die Matrixzeilen bzw. -spalten und normieren auf
Maximum eins. Diese Summen verwenden Knoten im transformierten Raum,
keine im Code sichtbare Gewichtung mit nativen Niederschlagsabständen.
Das ist ein statischer Implementierungsbefund; keine nachträgliche
Methodenkorrektur oder Aussage über jeden historischen Lauf.

Die PDF-Beschriftung allein belegt nicht die Einheiten-/Aggregationsmetadaten
der fehlenden NetCDF-Eingaben. Insbesondere wird aus `pre` oder `PANN` kein
ungeprüfter Monats-/Jahresfaktor eingeführt. Bereits belegte regionale Mittel
1901–2015 und 1961–1990 werden nicht als globale Eingaben eingesetzt.

## 4. Quellvarianten und Haupteinstieg

Gezielt geprüft wurden sechs vorhandene `all_trafo`-/`all_transferfunctions`-
Varianten im Original-Quellverzeichnis, `main_v1.2.R` sowie die erhaltene
BBM-Vergleichsfassung `my_kinneret_bbm/src/all_trafo_verific_v1.R`.
Alle acht Quellen sind mit Hash, Treffern und relevanten Zeilenbereichen
im neuen Nachweis `source_variant_bindings.json` gebunden.

Nur `src/all_trafo.R:493` schreibt in dieser Auswahl ausdrücklich
`all_trafos_emd_<i>.pdf`. Ein expliziter Schreiber für den zweiten Sammelplot
wurde darin nicht gefunden. Auch der exakte historische Zusammenführungsbefehl
für die Einzel-PDFs ist damit nicht belegt.

`all_transferfunctions_v1.2.R:330–399` enthält eine ältere 2D-Variante mit
Auswahl nach Balanced Accuracy und denselben vier Marginal-/Achsenfeldern,
schreibt jedoch `mixed_tf.rdata`. `all_transferfunctions_v0.9.R` schreibt
`mixed_tf_partial_v1.2.rdata` aus separaten eindimensionalen Fits. Diese
Ausgaben sind nicht aufgrund ähnlicher Felder mit dem zweiten Plot gleichgesetzt.
Die beiden Variablenanalyse-Fassungen speichern Wichtigkeitsarrays.

Der Haupteinstieg `main_v1.2.R:568` erwähnt `mixed_partial_arts.rdata`
als auskommentierte BA-only-Variante. Die **aktive** Zeile 569 lädt
`mixed_partial_arts_emd.rdata`, passend zum aktiven EMD-Schreiber in
`all_trafo.R:528–529` und dessen separater Info-Datei. Der zweite Plot kann
allein durch diesen Kommentar oder seinen Dateinamen keinem exakten
BA-only-Lauf zugewiesen werden.

Der aktive Verbraucher übernimmt die beiden Achsen aus `transfer.functions[[1]]`
und alle Marginalen taxonweise, danach normiert er nochmals auf Maximum eins.
Ein älterer Spline-Abgleich im Abschnitt 508–554 ist auskommentiert. Ohne
die numerischen Container ist nicht geprüft, ob alle taxonweisen Achsen
exakt zur ersten Achse passen. Die nahezu gleichen Plotachsen ersetzen
diesen fehlenden Array-Abgleich nicht.

Der Haupteinstieg öffnet außerdem die gesonderten globalen Mitteldateien
`cru_tann_ww.nc` / `cru_pann_ww.nc` sowie `cru4_tann_v1.2.nc` /
`cru4_pann_v1.2.nc` für die jüngere Prior-Klimareihe. Diese Verbraucherpfade
beweisen nicht die Identität mit den vom TF-Erzeuger überschreibend geladenen
test.nc-Dateien. Der bestehende Trainingsrücksatz und die Rangwahl 0,75 EMD /
0,25 Balanced Accuracy bleiben dokumentiert und unverändert.

## 5. Prüfung, Erhalt und nächster konkreter Einstieg

31 gezielte Tests bestanden: acht neue Fälle für Koordinaten, überlagerte
Punkte, getrennte Füllung/Umriss, Achsenrichtung, Farbbindung und
Fehlersperren sowie die bisherigen 23 Konsolen-/Punktlesertests.
Die bereits abgeschlossene Modellprüfung wurde nicht wiederholt.
Sechs vollständige MuPDF-Kontrollseiten wurden visuell geprüft: beide
Fassungen der Seiten 1 und 54 sowie zweiter Plot Seiten 76 und 278.
Die grünen Zeichen sind sichtbar; Popplers bekannte Schriftgrenze umgangen.

Maßgebliche neue Evidenz: `.local/tf_plot_comparison_v1/` mit
`second_plot_pages.json`, `vector_comparison.json`,
`fixed_decimal_coordinates.json`, `source_variant_bindings.json`,
`missing_dependency_followup.json`, `pytest.xml` und `final_verification.json`.
Die vorherigen Nachweise bleiben unverändert. Gezielte Original-Hashes,
Prüfsummen der bisherigen lokalen Dateien und der früheren Nachweise sowie
die vorhandenen Metadatenlisten werden im Abschluss erneut verglichen.
Keine erneute Gesamtinventur, Installation, R-Ausführung, Trainings-/Sampler-
oder Fossilrekonstruktionsläufe. Kein Commit, Push oder Merge.

**Weiter offen:** Die numerischen `mixed_partial_arts_emd.rdata` und
`mixed_partial_arts_emd_info.rdata`, globale test.nc-/CRU-Eingaben samt
Zeitachse und vollständige Laufbindung fehlen weiterhin. Frühere negative
Katalog-/Archivbefunde wurden wiederverwendet; die vier konkreten historischen
Verzeichniskandidaten existieren weiterhin nicht. Kein Anspruch einer
neuen vollständigen Laufwerkssuche. Die 95.006 regionalen Rasterabweichungen
und der vorgelagerte Rastererzeuger bleiben offen.

**Als Nächstes:** Die jetzt gebundenen Präsenzpunktwolken ermöglichen einen
gezielten Konsistenzvergleich mit den bereits erhaltenen regionalen
Klimamitteln und ihren nachgewiesenen Zeitgrundlagen, zunächst über tatsächlich
koordinatengebundene Kandidaten und PDF-Präzisionsgrenzen. Das wäre ein Test
von Eingabekandidaten, keine numerische TF-Wiederherstellung. Parallel nur
neue konkrete Quellenpfade für die fehlenden Container oder die exakte
BA-only-Quellfassung verfolgen; abgeschlossene Gesamtprüfungen nicht wiederholen.
Grafische Näherungen erfordern eine eigene fachliche Entscheidung, bevor sie
numerische TFs ersetzen könnten; hier wurde kein Ersatz vorgenommen.

Paper- und Sieben-Arten-Ergebnisse, App/CLI, Baseline sowie
Monticchio-/Chronologie-Gates bleiben bewahrt. Jung links, alt rechts.
