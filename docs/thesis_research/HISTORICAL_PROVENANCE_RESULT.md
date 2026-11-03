# Historische Raster-, Taxonmodell- und Klimaprovenienz

Stand 01.11.2026. Fortsetzung im vorhandenen Checkout, Branch
`dissertation-reconstruction`, unveränderte Basis `7f1826c` nach
[PR #30](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/30).
Dies ist ein abgeschlossenes Quellenarbeitspaket; die historische
279er-Taxonmethode und ihr konkreter Lauf sind weiterhin nicht vollständig reproduziert.

## 1. Herkunft der 95.006 Rasterabweichungen enger eingegrenzt

Die 275 Namen stimmen geordnet zwischen regionalem RData, eurasischem RData
und globalem Kinneret-Container überein. Die vollständigen eurasischen
RData-/NetCDF-Arrays sind wert- und fehlwertgleich; die beiden eurasischen
NetCDF-Kopien aus Prespa und Ohrid sind bytegleich. Das regionale NetCDF
stimmt auf seinen gesamten gemeinsamen Koordinaten ebenfalls exakt mit dem
eurasischen NetCDF überein. Es handelt sich um geprüfte Werte und Koordinaten,
nicht um eine Zuordnung allein nach Form oder Taxonnamen.

| Vergleich | Verschiedene endliche Zellen | Fehlwertmasken |
| --- | ---: | --- |
| Globaler Ausschnitt gegen bisheriges regionales RData | 95.006 | gleich |
| Eurasische Präsenz-Einfügung gegen regionales RData, auf beobachteter globaler Maske | 95.006 | gleich |
| Globaler Ausschnitt gegen dieselbe Einfügung | 0 | gleich |

Von den 95.006 Abweichungen sind 63.285 Zellen global 1 / regional 0 und
31.721 global 0 / regional 1. Alle liegen innerhalb des eurasischen
Eingabefensters. Die vier regionalen Längengrade westlich von −10,75° tragen
keine weitere Abweichung bei. Die Abweichungen liegen damit bereits zwischen
der regionalen RData-Fassung und der NetCDF-/eurasischen Eingabefassung vor;
sie entstehen im regionalen Fenster nicht erst beim Einbau in das Weltgitter.
Welcher frühere Export, Zuschnitt oder Bearbeitungsschritt die regionale
RData-Fassung erzeugte, ist weiterhin nicht durch einen Erzeuger belegt.
Die zuvor bestätigten 27 vollständig gleichen Ebenen bleiben gültig.

Der rohe Vergleich des regionalen NetCDF mit regionalem RData auf gemeinsamen
Koordinaten ergibt separat 93.721 Wert- und 202.950 Maskenabweichungen. Diese
Zahlen haben einen anderen räumlichen und maskenbezogenen Vergleichsumfang
und ersetzen nicht den 95.006-Befund.

**Zusätzlicher globaler Randbefund:** Bei direktem Vergleich eurasischer
Eingangswerte mit der globalen Fassung gibt es 76 endliche Wertabweichungen:
74 Präsenzzellen bei 179,75° Ost und zwei Zellen mit Eingangswert 2.
Letztere liegen in der eurasischen Position 152, `Pinus brutia`, bei
43,75° E / 36,75° N und 43,25° E / 37,25° N. Der historische Einfügecode
wählt ausdrücklich nur `== 1`; er übernimmt Werte 2 deshalb nicht.
Die R-Indexformel mit `floor(k / length(lon))` führt bei einem 1-basierten
Zeilenendindex zum Längengradindex 0. Eine Nachbildung, die die letzte
Längenspalte auslässt und nur Werte 1 einfügt, stimmt in allen 275 globalen
Ebenen vollständig mit dem gespeicherten Container überein.

Diese Nachbildung verwendet ausdrücklich die **beobachtete globale Maske**.
Die externe unabhängige Klimamaske ist weiterhin nicht vorhanden. Der
numerische Gleichheitsnachweis identifiziert einen zum Code passenden
Erzeugungsweg, aber keinen protokollierten historischen Lauf. Keine Werte,
Schwellen oder Zuordnungen wurden korrigiert oder ersetzt.

## 2. Tatsächlich gespeicherter 279er-Verbreitungsstand

Zusätzlich gelesen: `my_ohrid/data/in/climate_data/taxa_distribution.rdata`.
Das Objekt heißt `taxa.distribution` und enthält nur `distribution`, `names`,
`lon`, `lat`, mit 720 × 360 × 279 Werten und eigenen globalen Achsen.
Alle 279 kanonischen Ebenenprüfsummen entsprechen exakt dem bereits geprüften Ohrid-
283er-Container nach Entfernung der Positionen 8, 10, 152, 153. Auch die
Namensfolge stimmt exakt, **vor** der späteren Juglans-Umbenennung.
Gespeicherte Position 110: `Juglans sp.`; 277 unterschiedliche Namen.

Damit ist der 279er-Datenzwischenstand materiell belegt. Diese Datei enthält
keine Transferfunktionssammlung und ist kein Laufprotokoll. Die Umbenennung
zu `Juglans regia` bleibt ein separater Code-Schritt aus `main_v1.2.R`.
Keine entsprechende Indexoperation wurde auf Kinneret angewendet.

## 3. Modellcontainer: Struktur statt Dateinamensannahme

Gezielter Umfang: **109 RData-Dateien aus Transferfunktions- und
Modellvergleichsordnern, 78 verschiedene Byteinhalte**. Bytegleiche Dateien
wurden nur einmal strukturell geprüft; ihre individuellen Hashes wurden
festgehalten. Ergebnis auf Ebene der verschiedenen Byteinhalte:

| Strukturprüfung | Anzahl |
| --- | ---: |
| Vollständig vom Metadatenleser gelesen | 65 |
| Leserfehler, nicht klassifiziert | 9 |
| 45-Sekunden-Grenze, nicht klassifiziert | 4 |

In vollständig gelesenen Byteinhalten wurden 77
caret-Fitobjekte containerbezogen gezählt, ohne zusätzliche Deduplizierung
der Modellgewichte: 6 mit zwei Klassen und
71 mit mehr als zwei Klassen. Das ist keine
Vollprüfung sämtlicher 193 RData-Dateien des Bestands und kein Nachweis, dass
nicht gelesene Container keine Taxonfunktionen enthalten. Rohmodelle ohne
caret-Klasse werden durch diesen Zähler nicht automatisch klassifiziert.

**Neue positive Funde:**

- `my_kinneret/.../qda_fit_smoted_partial.rdata`: `best.fit` enthält sechs
  QDA-Fits, jeweils Klassen `taxa1`, `unknown` und Prädiktoren
  `t.djf.norm`, `p.ann.norm`.
- `my_kinneret/.../comb_qda_fit_smoted_partial.rdata`: `comb.best.fit` enthält
  15 QDA-Fits, jeweils zwei generische Taxonklassen plus `unknown`, mit
  denselben beiden Klimaprädiktoren.
- Der zuvor bei Typ 252 blockierte Ohrid-Container `transferfunction.rdata`
  ist nun strukturell lesbar: ein caret/nnet-Mehrklassenfit für sechs Taxa
  plus `unknown`, ebenfalls mit Wintertemperatur und Jahresniederschlag.

Der vorhandene Erzeuger `my_kinneret/src/ml_competition_partial.R`, Z. 6–45,
149 und 311–395, schreibt die beiden QDA-Dateipfade ausdrücklich. Seine
Teilmodelle verwenden je eine ausgewählte Klasse gegen die gemeinsame
`unknown`-Klasse; sie benennen die positive Klasse jeweils in `taxa1` um.
Der gelesene Klimavorbereitungszweig definiert `unknown` über
`taxa.not.appear == 0`, nach möglichen Taxonverschmelzungen. Das entspricht
nicht automatisch der vollständigen Abwesenheitsmenge jeder einzelnen Art.
Die 15 Paarmodelle werden im Code aus Zweierkombinationen erzeugt.

Gesicherter Befund sind Objektstruktur, gespeicherte Klassen und Prädiktoren
sowie der dazu vorhandene Quellzweig. Eine konkrete Laufzuordnung oder eine
nachträgliche Artzuweisung allein aus `taxa1` oder einer Listenposition folgt
daraus nicht. Keine dieser Sammlungen wird als die gesuchte historische
279er-Artenfunktionssammlung eingesetzt.

## 4. Klimagrundlagen und Zeitachse

51 NetCDF-Dateien beider Originalbestände geprüft: 50 mit dem vorhandenen
NetCDF3-Leser lesbar; `taxapool_relict.nc` bleibt damit nicht dekodiert.
Globale `tann_test.nc` / `pann_test.nc` sowie die referenzierten globalen
CRU-Eingaben sind in diesen Beständen nicht wiederhergestellt.

| Erhaltene Dateien | Dokumentierte Zeitgrundlage | Rechenweg laut Header |
| --- | --- | --- |
| `in/tann_cru4.nc`, `in/pann_cru4.nc` | 1901–2015 | Temperaturmittel; Niederschlags-Jahressummen, danach Mittel |
| Europäische CRU-Dateien | 1961–1990 | Explizite Jahresauswahl; Tann- bzw. DJF-Mittel; Niederschlag Jahresmittel × 12 |

Die ausgelesenen Zeitgrenzen stützen diese unterschiedlichen Zeitgrundlagen:
regional 16.01.1901 bis 16.12.2015, europäisch 16.01.1961 bis 16.12.1990,
jeweils `days since 1900-1-1`, gregorianisch. Das sind die tatsächlich
gespeicherten Monatsstützpunkte der Bounds, keine neu erfundenen Jahresgrenzen.
Alle diese erhaltenen Klimamittel enthalten jeweils **einen** Zeitschritt.

Im späteren `all_trafo.R` überschreiben die `test.nc`-Öffnungen die zuvor
geöffneten globalen CRU-Dateien; anschließend werden Positionen `61:90`
gemittelt. Ohne die wirklichen Eingabedateien und deren Zeitachse darf dies
nicht automatisch in Kalenderjahre übersetzt werden. Die erhaltenen
Ein-Zeitschritt-Mittel sind dafür kein unmittelbarer Ersatz.

Die Header tragen bei Niederschlag teilweise weiterhin `mm/month`, obwohl
ihre Verarbeitungshistorie eine Jahressumme oder Multiplikation mit 12
beschreibt. Diese Inkonsistenz bleibt sichtbar; keine stille Einheitenkorrektur.
Wintertemperatur-Modelle werden nicht als Tann-Modelle behandelt.

## 5. Lauf- und Paketprovenienz: Teilnachweise und Grenze

In lesbaren Modellheadern gespeicherte R-Schreiberversionen: 3.5.0.
Die Namespace-Referenzen enthalten unter anderem `caret` mit dem gespeicherten
Versionsfeld `6.0-84`. Schreiberversion und Namespace-Felder sind konkrete
Serialisierungsmetadaten, aber keine vollständige Trainingsumgebung, kein
Sitzungsprotokoll und kein Beweis des verwendeten Seeds.
Die gelesenen Quellstände setzen unter anderem 2019 bzw. im späteren
`all_trafo.R` 2023. Quelltextwerte werden nicht als ausgeführte Laufparameter
umetikettiert.

18 ZIP-Archive beider Bestände zusätzlich im Speicher geprüft: 78 ausgewählte
R-/Python-/Shell-/Text-/Markdown-Dateien gelesen, ohne Treffer für die gezielt
gesuchten Rasterobjekte, Erzeugungsschritte oder Laufprotokollbegriffe. Keine
Archive entpackt oder Programme ausgeführt; keine Aussage über ungelesene
Binärmitglieder daraus abgeleitet.

Der neue Metadatenleser bewahrt die besonderen R-Marker 252, 250 und 248
getrennt und führt keine Umgebungen, Packages oder Closures aus. Grundlage
für diese Marker und Bytecode-Grenzen ist der
[R-Serialisierungsquelltext](https://github.com/wch/r-source/blob/trunk/src/main/serialize.c).
Unvollständige Bytecode-Kompatibilität bleibt als Lesergrenze sichtbar.
Große ausgelassene Zahlenblöcke sind keine verwendbaren Modellwerte.

## 6. Nachweise, Erhalt und nächster Einstieg

Aktueller lokaler Nachweis: `.local/provenance_continuation_v1/`.
Maßgeblich sind `raster_v2/`, `models_v2/`, `climate/`, die ergänzenden
Quellstellen-/Zeitgrenzen-/279er-/Wert-2-Nachweise und `final_verification.json`.
Der erste Modellscan wurde nach einer Lesergrenze abgebrochen; `models/`
ist ausdrücklich kein abgeschlossener Nachweis. `raster/` und frühere
Testordner bleiben als Zwischenstände erhalten.

21 gezielte Leser- und Klassifikationstests bestanden. Die unabhängige
Abschlussprüfung kontrolliert die Artefaktsummen, Original-Metadaten und
-Prüfsummen sowie die ursprünglichen Checkout-Dateien und die vollständig
erhaltenen früheren Dokumentabschnitte. Konkrete Ergebnisse stehen im
Abschluss-JSON; nicht dekodierte Container gelten nicht als bestandene
wissenschaftliche Identifizierung.

Reproduzierbare Hauptschritte (jeweils einen **neuen** Ausgabeordner wählen):

```powershell
.\.venv\Scripts\python.exe -B tools/audit_historical_provenance.py --config .local/my_reconst_paths.json --stage raster --output .local/NEUER_RASTER_AUDIT
.\.venv\Scripts\python.exe -B tools/audit_historical_provenance.py --config .local/my_reconst_paths.json --stage models --output .local/NEUER_MODELL_AUDIT
.\.venv\Scripts\python.exe -B tools/audit_historical_provenance.py --config .local/my_reconst_paths.json --stage climate --output .local/NEUER_KLIMA_AUDIT
```

**Nächster Einstieg:** Zuerst diesen Bericht und das Abschluss-JSON lesen.
Danach gezielt die verbliebenen Leserfehler/Zeitgrenzen bearbeiten und den
früheren regionalen Rasterexport sowie die globalen Klima-/TF-Laufdateien
weiter identifizieren. Für den bekannten 279er-Datenstand ist die noch
nicht erfolgte Juglans-Umbenennung ausdrücklich zu beachten. Für die
sechs Teilmodelle fehlen weiterhin eine belastbare Art-/Listenbindung und
vollständige Klima-/Laufprovenienz. Kein automatischer Austausch von Rastern,
Klimamitteln oder Modellfamilien.

Paperreferenz, geprüfte Sieben-Arten-Ergebnisse, App/CLI und Baseline wurden
bewahrt. Keine Fits, Sampler oder Fossilrekonstruktionen ausgeführt, keine
Commits, Pushs oder Merges. Monticchio-/Chronologie-Gates unverändert;
Darstellungsvorgabe weiterhin jung links, alt rechts.


## 7. Fortsetzung: Modell-Lesergrenzen abgeschlossen, 01.11.2026

Die oben dokumentierten neun Leserfehler und vier Zeitgrenzen sind inzwischen
gezielt aufgelöst. Alle 78 verschiedenen Byteinhalte des bisherigen Suchumfangs
sind strukturell gelesen; keine R-Vorhersagevalidierung. Neu hinzu kommen eine
fitinterne Klassenabweichung, bytegleiche regionale Kopien und ein vorhandenes
279-Positionen-Konsolenprotokoll ohne vollständige Laufbindung.
Die vorigen Abschnitte bleiben als historischer Arbeitsstand erhalten.
Maßgebliche Fortsetzung mit Grenzen und nächstem Einstieg:
[Modell-Leser und weitere Quellenprovenienz](HISTORICAL_MODEL_READER_RESULT.md).
