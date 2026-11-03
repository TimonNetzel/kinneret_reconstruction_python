# Historische Modell-Lesergrenzen und weitere Quellenprovenienz

Stand 01.11.2026. Fortsetzung im vorhandenen Checkout auf
`dissertation-reconstruction`, Basis `7f1826c` nach
[PR #30](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/30).
Kein neuer PR, Commit, Push oder Merge.

## Ergebnis und Geltungsbereich

**Abgeschlossen:** Die neun bisherigen Modell-Leserfehler und vier
Zeitüberschreitungen sind bearbeitet. Alle 13 verschiedenen Byteinhalte mit
17 Dateialiasen sind jetzt vollständig strukturell lesbar. Die 65 bereits
abgeschlossenen Befunde wurden aus ihren unveränderten Einzeldateien übernommen;
keine Wiederholung der gesamten Inventur. Damit sind im bisherigen Suchumfang
109 Dateien / 78 verschiedene Byteinhalte strukturell gelesen, ohne verbliebene
Leserfehler oder Zeitgrenzen.

Das ist **keine** Validierung der R-Vorhersagen, Modellgüte oder historischen
Taxonzuordnung. Ausgelassene große Zahlenblöcke bleiben keine verwendbaren
Modellgewichte. Die Auswahl ist weiterhin keine Vollprüfung aller 193
RData-Dateien des Zusatzbestands.

| Containerbezogen gezählte caret-Fits | Bisher | Neu gelesen | Gesamt |
| --- | ---: | ---: | ---: |
| Binär | 6 | 0 | 6 |
| Mehrklassen | 71 | 90 | 161 |
| Gesamt | 77 | 90 | 167 |

Keine zusätzliche Deduplizierung der Modellgewichte über Container hinweg.
Die sechs binären QDA-Teilmodelle und 15 QDA-Paarmodelle bleiben die bisherigen
Befunde mit ihrer gemeinsamen unknown-Klasse. Die neuen Funde sind keine
historische Sammlung von 279 unabhängigen Artenfunktionen.

## Leseränderungen und Nachweise

1. Gewöhnlich serialisierter, verschachtelter R-Bytecode startet eine eigene
   Wiederholungstabelle. Nur kompakte Bytecode-Konstanten teilen die äußere
   Tabelle. Der bisherige Leser übersprang bei einem gewöhnlichen inneren
   Bytecodeobjekt die neue Tabellenlänge und behandelte etwa 53 als Objekttyp.
   Der gesicherte alte Leser scheitert am unabhängigen Wire-Fixture weiterhin
   mit `ValueError: 53 is not a valid RObjectType`; die Korrektur liest den
   inneren Code und das unmittelbar folgende Objekt exakt.
2. Einzelne XDR-Integer werden direkt und mit Kürzungsprüfung gelesen. Sie
   verbrauchen nicht mehr das Budget für aufbewahrte numerische Nutzdaten.
3. Die Modellauswertung liest Klassen- und Namensattribute gezielt, ohne
   beliebige Formelattribute oder Umgebungen durch den allgemeinen Konverter
   zu schicken. Nicht ausgewählte Attribute bleiben als R-Struktur erhalten.
4. Ein begrenzter Cache teilt identische, kurze, attributfreie CHAR-Blätter
   einschließlich ihrer Serialisierungsflags. Nachgewiesene lange, wiederholte
   Zeilennamenvektoren wurden zuvor millionenfach als getrennte Objekte gezählt.
   Der Cache hält höchstens 65.536 Einträge mit je höchstens 128 Nutzbytes.
   Objektgrenze 2.000.000 und Zahlenbudget 32.000.000 Byte bleiben erhalten;
   zusätzlich begrenzen 10.000.000 Parse-Aufrufe und 180 Sekunden je Retry den Lauf.

Grundlage der Bytecode-Unterscheidung ist
[R serialize.c, ReadItem / ReadBC / ReadBC1 / ReadBCConsts / ReadBCLang](https://github.com/wch/r-source/blob/trunk/src/main/serialize.c).
Das ist eine Formatquelle, kein Nachweis der historischen Trainingsumgebung.
Keine R-Funktion, Closure, Paketinitialisierung oder Modellvorhersage wurde ausgeführt.

36 gezielte Tests bestanden, darunter unabhängige verschachtelte Wire-Fixtures,
Referenzidentität, Stream-Fortsetzung, Kürzungen, Flag-Erhalt, Cache-Verdrängung,
Objekt-/Arbeitsgrenzen und Standardwerte gegen den regulären RData-Leser.
Der größte offene Container benötigt mit dem letzten Leser 47,36 Sekunden;
3.816.068 Parse-Aufrufe, davon 2.762.757 wiederverwendete CHAR-Objekte,
4.742.848 Byte aufbewahrte numerische Daten. Ein bloßes Beibehalten der früheren
45-Sekunden-Grenze hätte diesen Fall weiterhin ausgeschlossen.

## Neue Modellbefunde und eine innere Klassenabweichung

Die neun früher gescheiterten Byteinhalte enthalten jeweils neun
Mehrklassenfits: LDA, QDA, PDA, MDA, FDA, SVM radial, nnet, RRF und GBM.
Zwei davon sind bytegleiche Aliaspaare in den Gedi-/Kinneret-BBM-Ordnern und
tragen `biome1`–`biome3` plus `unknown`; sie werden nicht als Artenmodelle umgedeutet.
Alle 90 neu gelesenen Fits haben die gespeicherten Prädiktoren
`t.djf.norm`, `p.ann.norm`: Wintertemperatur und Jahresniederschlag.

Die vier früheren Zeitgrenzen ergeben sechs Mehrklassenfits in
`my_birkat_ram/.../all_models_smoted_2D_neumann.rdata` und je einen nnet-Fit in
`my_kinneret/.../best_fit_2D_smoted_v1.6.rdata`,
`my_ohrid/.../best_fit_2D_v2.6.rdata` und `my_ohrid/.../nnet_2D.rdata`.
Die beiden Ohrid-Dateien besitzen jeweils bytegleiche Prespa-Aliase.
Diese drei nnet-Fits tragen sechs generische Taxonklassen plus `unknown`.
Der Birkat-Ram-Container enthält neun generische Taxonklassen plus `unknown`.
Keine Artbindung wird aus Listenposition oder Dateinamen hergestellt.

**Gespeicherte Klassenabweichung:** Im Birkat-Ram-Container
`all_models_smoted_2D_7.rdata` haben acht caret-Fits `taxa1`–`taxa7` plus
`unknown`, der GBM-Fit dagegen zusätzlich `taxa8`.
Die Klassenlisten werden fitweise erhalten; keine automatische Harmonisierung
oder gemeinsame Auswertung. Ob hier ein älterer Fit gespeichert wurde, bleibt
unbelegt. Der Dateiname entscheidet diese Frage nicht.

## Regionaler Rastererzeuger und globale Eingaben

385 bereits katalogisierte R-/Python-Quellen wurden gezielt nach regionalen
Objekten, expliziten Schreibstellen, globalen Klimaeingaben, TF-Containern und
Laufprovenienz durchsucht; rbacon-Paketquellen wurden ausgelassen. Kein
expliziter Schreiber des regionalen Objekts unter den geprüften Mustern gefunden.
Dynamische, anders benannte oder nicht vorhandene Erzeuger sind dadurch nicht ausgeschlossen.

Neun ausgewählte regionale Container besitzen nur zwei verschiedene Byteinhalte:
Das ursprüngliche regionale `taxa_old_new_mediterr.rdata` ist in allen fünf
zusätzlich geprüften Kopien bytegleich, mit dem 149 × 97 × 275-Block.
Die drei `taxa_mediterranean.rdata`-Kopien sind untereinander bytegleich,
enthalten jedoch einen eigenen 40 × 40 × 15-Block mit Koordinaten.
Sie werden nicht allein wegen ähnlicher Namen als Vorstufe des 275er-Blocks zugeordnet.
Keine dieser Kopien liefert einen belegten früheren Rastererzeuger.
Der Befund von 95.006 Abweichungen und der bekannte Ohrid-Pfad bleiben unverändert.

Die konkreten globalen `tann_test.nc` / `pann_test.nc` und die fünf im
unten genannten Quellzweig referenzierten globalen CRU-Dateien fehlen weiterhin
in den bisherigen Katalogen. Vorhandene Ein-Zeitschritt-Mittel ersetzen sie nicht.
Aus `61:90` wird weiterhin keine unbelegte Kalenderjahreszuordnung abgeleitet.

## Tatsächlich vorhandene Konsole, aber keine vollständige Laufbindung

`local_reconstr_all_epd/logfile_trafo.txt` ist ein materiell vorhandenes
Konsolenprotokoll mit 22.262 Zeilen. Es enthält genau die ausgegebenen
Positionen 1–279. Bei 271 Positionen folgen jeweils Modellmeldungen für
`nnet`, `mda`, `qda`, `gbm`, insgesamt 1.084 Meldungen mit
`caret`, Versionsausgabe `6.0.84`. Die ausgegebenen Daten besitzen jeweils
fünf Spalten; Zeilenzahlen variieren zwischen 37.818 und 75.610.
Die Positionen 9, 16, 142, 149, 151, 162, 167 und 206 haben keine solchen Meldungen.
Das bedeutet weder acht nachgewiesene Trainingsfehler noch 271 erhaltene TF-Dateien.

Als **Quellkandidat** passt `src/all_transferfunctions_v1.R` zu der Reihenfolge
der vier DALEX-Ausgaben und einem Zweig mit fünf Klimaprädiktoren. Dieser
Quellstand referenziert Ohrid-Verbreitung nach dem separaten Löschpfad sowie
`cru_tann_ww.nc`, `cru_wintertemp_ww.nc`, `cru_summertemp_ww.nc`,
`cru_pann_ww.nc`, `cru_pwin_ww.nc`.

**Grenze:** Im heutigen Quellstand überschreibt eine aktive Zuweisung
`trafo.input <- titanic_imputed` die zuvor vorbereiteten Klimadaten vor dem Fit.
Der Konsolentext enthält weder diese Eingabebindung noch ein Seed-, Dateihash-
oder Sitzungsmanifest. Deshalb wird nicht behauptet, dass genau dieser heutige
Quellstand das Protokoll erzeugte. Die im Code stehende 2019 ist kein aus dem
Protokoll nachgewiesener Seed. Aus den gedruckten Zahlen werden keine Arten
zugeordnet; der Ohrid-Pfad wird weiterhin nicht auf Kinneret übertragen.

## Maßgebliche lokale Evidenz und nächster Einstieg

Nachweisordner: `.local/provenance_reader_followup_v1/`.
Maßgeblich für das Ergebnis sind `model_summary.json`, die dort je SHA256
referenzierten Einzeldateien aus `models_v2/` und `models_v5/`,
`regional_container_copies.json`, `targeted_source_search.json`,
`training_console_trace.json`, `training_console_candidate_source.json`,
`regression_demonstration.json`, `pytest_results.json` und `final_verification.json`.
Frühere Fehlversuche in `models/`, `models_v3/`, `models_v4/` und
`combined_after_v2.json` sind erhaltene Zwischenstände, keine endgültige Auswertung.
Insbesondere dient `combined_after_v2.json` nur der Auswahl des letzten offenen Falls;
die maßgeblichen Gesamtsummen stehen in `model_summary.json`.

**Als Nächstes:** Die Leserreparatur ist abgeschlossen. An der materiell
vorhandenen Konsolenspur und ihren fehlenden Eingabebindungen weiterarbeiten:
frühere Fassung des Fünf-Klimavariablen-Zweigs, tatsächliche globale CRU-/test.nc-
Dateien mit Zeitachsen und historische TF-/Variablenwichtigkeits-Ausgaben mit
explizitem Laufbezug suchen. Für den regionalen Export gezielt die in älteren
Quellen genannte vorgelagerte Sammlung
`climate_modelling/pollen/transfer_functions/input_data/data_out` verfolgen,
sobald sie als konkrete zusätzliche Quelle verfügbar ist. Eine weitere Wiederholung
der 78 Modellprüfungen löst diese Quellenlücken nicht.

Gesamtprovenienz weiterhin teilweise offen. Keine neue fachliche Freigabe,
kein Training, Sampler oder Fossilfall. Paper- und Sieben-Arten-Ergebnisse,
Monticchio-/Chronologie-Gates, App/CLI und Baseline bewahrt. Jung links, alt rechts.
