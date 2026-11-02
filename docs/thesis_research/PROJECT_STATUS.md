# Netzel-Projekt: kurzer Arbeitsstand

## GitHub-Sicherung und Claude-/App-Übergabe, 02.11.2026

Auf Nutzerauftrag auf GitHub gesichert: [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
Branch `dissertation-reconstruction`, erster Sicherungscommit `973fddb`.
Der PR bleibt für die Claude-Fortsetzung offen; `main` ist unverändert.
[Aktueller Einstieg](CLAUDE_HANDOFF_2026_11_02.md), automatisch auffindbar
auch über `CLAUDE.md`. 50 ausgewählte Nachweise/Ergebnisdateien, rund 2,8 MB,
stehen im [portablen Paket](research_snapshot_2026_11_02/README.md); persönliche
Pfade nur in Exportkopien ersetzt. Originale, Runtimes und vollständige
Samplerarchive bleiben lokal. 67 betroffene Tests erneut bestanden.

Nächster App-Schritt: vorhandene Architektur prüfen und eine lesende
Forschungsstands-/Quellenansicht mit vorhandenen, experimentellen und fehlenden
Ergebnissen ergänzen. Baseline, App/CLI und sieben Arten bleiben die Grundlage.
Die fehlenden historischen numerischen TFs und fachlichen Gates werden
dadurch nicht aufgehoben. Keine neuen Trainings, Sampler oder Fossilfälle.
Plots jung links, alt rechts. Basis weiterhin PR #30 / `7f1826c`.


## Zweiter TF-Sammelplot und Klimakoordinaten geprüft, 02.11.2026

**Abgeschlossen – Quellenarbeitspaket:** Auch der zweite Sammelplot enthält
278 durch tatsächlichen Titel und grüne Punktzahl eindeutig zugeordnete
Ohrid-Ausgaben, genau ohne Position 151 (`Picea omorika`). `Juglans sp.` bleibt
Position 110; PDF-Seitenordnung ist kein Taxonindex und kein Kinneret-Pfad.

**Neu belegt:** 961.568 Punktpaare unabhängig aus beiden PDF-Inhaltsströmen
verglichen. Sämtliche Y-Koordinaten sind exakt gleich; X weicht höchstens
0,02 PDF-Punkt ab. 23 komplette Punktwolken sind exakt gleich. 269 Achsenfassungen
sind gleich, neun minimal in X verschoben. Konturvektoren sind auf allen Seiten
nicht identisch; das beweist keine 278 verschiedenen Modelle. Keine numerischen
TFs oder exakte Identität der globalen Klimadateien daraus abgeleitet.

**Quellbindung:** Temperatur affine Standardisierung, Niederschlag
Quadratwurzel/Gamma-CDF/Normalquantil; Rücktransformation auf native Plotachsen.
300er-Niederschlagsgitter im Allgemeinen nicht äquidistant. Zeitindizes 61:90
sind ohne Zeitachse keine bewiesenen Kalenderjahre. Aktiver EMD-Schreiber und
Haupteinstieg bleiben verbunden; der exakte BA-only-Erzeuger des zweiten Plots
ist trotz Vergleich von acht Quellen weiterhin nicht identifiziert.

**Geprüft:** 31 gezielte Tests, unabhängiger Dezimalcheck und sechs vollständige
visuelle Kontrollseiten; neuer Abschluss unter `.local/tf_plot_comparison_v1/`.
Originale und bisherige lokale Änderungen erhalten. Keine erneute Modellprüfung,
Trainings, Sampler, Fossilläufe, Commits, Pushs oder Merges. GitHub bestätigt
[PR #30](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/30)
und Basis `7f1826c`. [Ergebnis und Grenzen](HISTORICAL_SECOND_TF_PLOT_RESULT.md).

**Als Nächstes:** Punktwolken gezielt gegen koordinatengebundene erhaltene
regionale Klimamittel und deren Zeitgrundlagen testen; nur neue konkrete
Quellenansätze für die numerischen EMD-Container, globale test.nc/CRU-Dateien
und BA-only-Laufbindung verfolgen. Numerische TFs, globale Zeitachse,
95.006 Rasterabweichungen und früherer Rastererzeuger bleiben offen.
Paper, sieben Arten, App/CLI, Baseline und fachliche Gates bewahren.
Jung links, alt rechts.

## Konsolenspur und 278 grafische TF-Ausgaben verbunden, 02.11.2026

**Abgeschlossen – Quellenabgleich:** Der gespeicherte 279er-Ohrid-Stand erklärt
mit Nordhalbkugel, klassenweisen 70 % und dem synthetischen SMOTE-Ausgabezweig
alle 271 Konsolen-Zeilenzahlen und acht Auslassungen vollständig. Der passende
Fünf-Variablen-Quellzweig berechnet Variablenwichtigkeiten; seine Modellspeicherung
ist auskommentiert. Heutige Titanic-Überschreibung verhindert exakte Laufbindung.

**Neu materiell belegt:** 278 Seiten im erhaltenen EMD-TF-Plotcontainer.
Jeder tatsächliche Titel und jede gespeicherte grüne Punktzahl passt eindeutig
zu einer Position des gespeicherten Ohrid-Stands. Alle Positionen außer 151
(`Picea omorika`, eine Präsenz) sind enthalten, passend zur späteren Auslassungsregel.
Juglans-Ausgabe heißt noch `Juglans sp.`. Keine PDF-Seitenordnung als Taxonindex,
keine Übertragung des Ohrid-Pfads auf Kinneret, kein Ersatz numerischer TF-Arrays.

**Geprüft:** 59 Tests und unabhängige Abschlussprüfung unter
`.local/console_provenance_v1/`. Originale und lokale Vorarbeiten erhalten;
kein Training, Sampler, neuer Fossilfall, Commit, Push oder Merge.
Basis weiterhin [PR #30](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/30) / `7f1826c`.
[Ergebnisse, Grenzen und Nachweise](HISTORICAL_CONSOLE_TF_LINK_RESULT.md).

**Als Nächstes:** zweiten erhaltenen TF-Sammelplot und Varianten gezielt
gegen den belegten Plotstand prüfen; Klimakoordinaten und tatsächliche numerische
`mixed_partial_arts_emd`-Ausgaben weiter verfolgen. Globale Klimaeingaben,
Zeitachse und regionaler Rastererzeuger bleiben offen. Paper, sieben Arten,
App/CLI, Baseline und fachliche Gates bewahren. Jung links, alt rechts.

## Modell-Lesergrenzen abgeschlossen, Quellen fortgesetzt, 01.11.2026

**Abgeschlossen:** Neun Leserfehler und vier Zeitgrenzen gezielt aufgelöst:
13 verschiedene Byteinhalte / 17 Dateialiasen jetzt strukturell gelesen.
65 frühere Einzelbefunde unverändert übernommen; damit im bisherigen Umfang
109 Dateien / 78 Byteinhalte vollständig strukturell lesbar. Containerbezogen
167 caret-Fits: sechs binäre, 161 Mehrklassenfits. Keine Gleichsetzung mit
279 unabhängigen Taxonfunktionen und keine validierten R-Vorhersagen.

**Neu:** Ein gespeicherter GBM-Fit hat eine zusätzliche Taxonklasse gegenüber
den acht anderen Fits desselben Containers; Klassenlisten nicht harmonisiert.
Regionale RData in fünf Zusatzkopien bytegleich. Ein Konsolenprotokoll enthält
Positionen 1–279 und bei 271 Positionen Meldungen für vier Modelle, aber keine
vollständige Taxon-/Klima-/Laufbindung. Der passende heutige Quellkandidat enthält
eine aktive Titanic-Eingabeüberschreibung; daher keine exakte Laufzuordnung.

**Geprüft:** 36 gezielte Tests; Nachweise und unabhängiger Abschluss unter
`.local/provenance_reader_followup_v1/`. Originale und vorherige lokale Änderungen
erhalten. Keine Fits, Sampler, Fossilrekonstruktionen oder Git-Veröffentlichungen.
Basis unverändert nach [PR #30](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/30).
[Ergebnis und Grenzen](HISTORICAL_MODEL_READER_RESULT.md).

**Als Nächstes:** konkrete Eingabe-/Ausgabebindung der Konsolenspur,
frühere Fünf-Klimavariablen-Quellfassung, globale Klima-/TF-Dateien und
vorgelagerten regionalen Rasterexport suchen. Leserreparatur abgeschlossen;
Gesamtprovenienz offen. Paper, Sieben-Arten-Ergebnisse, App/CLI, Baseline und
Monticchio-/Chronologie-Gates bewahren. Jung links, alt rechts.

## Quellen- und Laufprovenienz fortgesetzt, 01.11.2026

**Abgeschlossen – Quellenarbeitspaket:** Die 95.006 regionalen Rasterabweichungen
liegen bereits zwischen regionalem RData und der NetCDF-/eurasischen Fassung;
im regionalen Fenster entspricht die globale Präsenz-Einfügung exakt der
eurasischen Eingabe. Der frühere regionale Export bleibt als Ursprungsschritt
offen. Global sind zusätzlich 74 Randspaltenverluste und zwei durch `== 1`
ausgeschlossene Wert-2-Zellen eingegrenzt; keine Rasterkorrektur.
Ein tatsächlich gespeicherter 279er-Ohrid-Verbreitungsstand stimmt in allen
Ebenen mit dem dokumentierten Löschpfad überein, vor der Juglans-Umbenennung.

109 Modellkandidaten / 78 verschiedene Byteinhalte geprüft: 65 vollständig
strukturell gelesen, 9 Leserfehler, 4 Zeitgrenzen. Gefunden sind sechs
binäre QDA-Teilmodelle und 15 QDA-Paarmodelle, mit Wintertemperatur und einer
gemeinsamen unknown-Klasse; keine Gleichsetzung mit 279 Artenfunktionen.
Erhaltene Klimamittel belegen unterschiedliche Zeitgrundlagen 1901–2015 bzw.
1961–1990. Die globalen test.nc-Eingaben und konkrete Laufprovenienz bleiben offen.

**Geprüft:** 21 gezielte Tests; Abschlussnachweis unter
`.local/provenance_continuation_v1/final_verification.json`. Originale und
vorhandene lokale Änderungen erhalten; keine Fits, Sampler, neuen Fossilfälle
oder Git-Veröffentlichungen. PR #30 / Basis `7f1826c` unverändert.
[Ergebnis, Grenzen und nächster Einstieg](HISTORICAL_PROVENANCE_RESULT.md).
**Als Nächstes:** gezielt verbleibende Modell-Lesergrenzen, regionalen
Rastererzeuger und globale Klima-/TF-Laufquellen klären. Paper, Sieben-Arten-
Ergebnisse, Monticchio-/Chronologie-Gates und App/CLI-Baseline bewahren.
Jung links, alt rechts.

## Lokaler Einstieg 01.11.2026

Neue lokale Bestandsaufnahme und sicherer Arbeitsbereich auf Branch
`dissertation-reconstruction`, Basis `main` bei `7f1826c` nach geprüftem Merge
von [PR #30](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/30).
Schritt 1 der Startdatei ausgeführt, Quellenabgleich begonnen. Originalbestand
unverändert; Katalog und persönliche Pfadkonfiguration lokal ausgeschlossen.
Fortsetzung am selben Tag: portablen Compiler lokal eingerichtet, Kern und
Original separat gebaut. Lokaler Originalvergleich über 20.000 Iterationen
bitgleich; archivierter Vergleich mit anderer C++-Laufzeit weiterhin fehlgeschlagen.
Diagnostischer und vollständiger Originallauf sowie korrigierter Vierkettenlauf
abgeschlossen. Beide Posterior-Sampler vollständig bitgleich wiederholt;
48 CSV-Felder unabhängig nachgerechnet, 362 Klimadiagnosen bestanden.
Alle 43 korrigierten Parameter erfüllen R̂ ≤1,01 und Bulk-ESS ≥400.
Zwei bestehende Tests bleiben als Laufzeit-/Kurzkettenbefunde dokumentiert.
Konkretes Ergebnis und Grenzen: [LOCAL_REFERENCE_RESULT.md](LOCAL_REFERENCE_RESULT.md).
Keine neue fachliche Freigabe; vor weiteren Fossilfällen Bewertung erforderlich.
Darstellungsvorgabe: Alter immer von jung (links) nach alt (rechts).
Aktueller Vergleichsbericht entsprechend aktualisiert; keine neuen Samplerläufe.
Der Paperfall ist keine vollständige Reproduktion der taxonbasierten Dissertation.
Befunde, Umgebungsprüfung und nächster Plan:
[LOCAL_WORKSPACE.md](LOCAL_WORKSPACE.md). Kein neuer PR oder Upload beauftragt.

## Quellenfortsetzung 01.11.2026: Taxonverfahren

**Abgeschlossen:** lokalen 275-Ebenen-Verbreitungsblock anhand des ausdrücklichen
R-Verbrauchers und identischer Fehlwertmasken dem europäischen 0,5°-Klimagitter
zugeordnet; Achsen aufsteigend Länge/Breite. 9.718 verwendbare Zellen je Ebene,
16 Ebenen ohne Präsenz, 233 verschiedene binäre Raster. Namen nicht eindeutig:
272/271 unterschiedliche Einträge in den beiden Spalten; keine Zusammenführung.

**Neu belegt:** `all_trafo.R` schreibt die vom Haupteinstieg erwartete historische
TF-Datei. Dieser Quellstand enthält einen Rücksatz auf die vollständige
Trainingstabelle nach der Hold-out-Aufteilung und eine zusätzliche EMD-Rangwahl.
Das ist ein statischer Befund, kein Beweis für die Ausführung in jedem
Dissertationslauf. Verschiedene Regeln zur Art-/Pollentyp-Zusammenführung
von der späteren gewichteten Taxonkombination getrennt.

**Weiter offen:** globale Eingaben, historische TF-Container, geordnete Taxonliste
und Laufprovenienz. Keine exakten Dateinamenstreffer für die gesuchten fehlenden
RData-Abhängigkeiten in den neun lokalen ZIP-Verzeichnissen. Acht Auditprüfungen
bestanden; 1.783 Originaldateien laut Metadatenkatalog und elf gehashte Quellen
unverändert. Keine Fits oder Sampler wiederholt; keine neuen Fossilfälle.
[Nachweis und Fortsetzung](LOCAL_TAXON_METHOD_AUDIT.md). Ergebnisse aus PR #29,
Paperreferenz, App und bestehende fachliche Gates unverändert. Kein neuer PR.

## Zusatzquelle my_reconst geprüft, 01.11.2026

Nutzerhinweis auf weiteren Originalordner abgearbeitet: 2.687 Dateien inventarisiert.
**Gefunden:** globale Verbreitungscontainer Kinneret (275 Ebenen) und Ohrid
(283 Ebenen), mit eigenen 720×360-Koordinaten. Ohrid enthält acht zusätzliche
Relikttaxa. Die 275 gemeinsamen globalen Ebenen sind exakt gleich. Der explizit
auf Ohrid bezogene historische Lösch-/Umbenennungspfad liefert 279 Positionen;
dieser Pfad ist nicht auf Kinneret übertragbar. Fertige historische TFs weiterhin offen.

**Wichtige Grenze:** Globaler Ausschnitt und bisheriger regionaler 275er-Block
haben 95.006 verschiedene binäre Zellen bei identischen Fehlwertmasken; nur
27 Ebenen sind exakt gleich. Keine Datenkorrektur oder Substitution. Zwei
kleine historische Ohrid-Modelle sind jeweils ein QDA-Mehrklassenfit für sechs
Taxa plus unknown, keine Sammlung von 279 binären Artenfunktionen.

**Geprüft:** sieben Lesertests und zehn Quellenprüfungen bestanden. Große
Raster nur ebenenweise gelesen, Originalbestände unverändert. Kein Training,
Sampler oder neuer Fossilfall. Die konkreten TF-/test.nc-Dateinamen fehlen
weiter im neuen Ordner und seinen neun ZIP-Verzeichnissen.
[Ergebnis und aktuelle Nachweise](MY_RECONST_SOURCE_RESULT.md).
Bestehende Ergebnisse aus PR #29/#30, App und fachliche Gates bleiben erhalten.

## Bisheriger wissenschaftlicher Stand

Stand 30.10.2026. Vor Fortsetzung aktuellen PR-/Merge-Stand prüfen.
Ziel: überprüfbare App für taxonbasierte Multiproxy-Klimarekonstruktionen
europäischer Fundstellen, mit belegter Kalibrierung und Unsicherheiten.

| Bereich | Stand |
| --- | --- |
| Moderne Analog-Baseline #18 | Fertig; #18 auf main gemergt. Unveränderte Referenz f9acc16. |
| Erste sieben Artenfunktionen #28 | Fertiger experimenteller Trainingsstand aus Meusel/BfN + WorldClim 2.1; feste Referenz. |
| SMOTE + einfaches Netz #29 | **Berechnet und unabhängig geprüft:** sieben Arten, identische äußere räumliche Tests; deutlich bessere Balanced Accuracy, zugleich schlechtere rohe SMOTE-Wahrscheinlichkeiten im Brier. |
| Historische 279 Funktionen | Globale Verbreitungsbasis und konkreter Ohrid-Pfad 283→279 gefunden; fertige TFs und bitgleiche historische R-/caret-Reproduktion weiterhin offen. |
| Taxonverfahren insgesamt | Weitere Taxa, Fossiltyp-Zuordnung, unabhängige Wahrscheinlichkeitskalibrierung und Unsicherheiten offen. |
| Monticchio-P1 | Weiter offen: physischer LGM90D→MON16-Crosswalk samt Korrelationsunsicherheit fehlt. |
| Proben und Quellen aus #26/#27 | Unverändert: 28 numerische Kandidaten, 41 native Nichttreffer, 47 offene Herkunftsfragen. |
| Fossilrekonstruktion / Multiproxy-App | Kein fertiger Gesamtdurchlauf, keine neue fossile Klimakurve und keine P2-Freigabe. Claude betreut CLI/App; Integration #19 in den gemeinsamen Stand übernommen. |

**Fertig:** [SMOTE-/Netzvergleich](./smote_nnet_2026_10_30/README.md),
[PR #29](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/29),
ursprünglicher Ergebnisbranch `codex/netzel-smote-nnet-comparison`.
Die Ergebnisse aus #18–#29 werden auf ausdrücklichen Nutzerauftrag gemeinsam
auf `main` gesichert ([Integrations-PR #30](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/30)); [gemeinsame Übergabe](./HANDOFF.md) und
[Integrationsnachweis](./consolidation_2026_10_30/README.md). Vorabdesign veröffentlicht
in `d47129c6b8be8bd5e73701f6137ffb196dab9d91`, bevor neuronale Fits begannen.

Artenmittel der Balanced Accuracy bei Schwelle 0,5: ohne SMOTE 71,64/69,20 %,
mit SMOTE 90,41/89,74 % (5°/10°-Blöcke, 100-km-Puffer). Gewinn in allen 14
Art×Block-Fällen. Flächengewichteter Brier dagegen im Artenmittel 0,03975/0,04438
ohne SMOTE und 0,07747/0,08245 mit SMOTE: Verschlechterung in allen 14 Fällen.
Die vorab vorgesehene Prior-Odds-Diagnose hilft, garantiert aber keine Kalibrierung
und schlägt das Netz ohne SMOTE im Brier nur in zwei von 14 Fällen.

14 vollständige neue Artennetze und alle Foldparameter gespeichert.
198.680 Art-Zell-Labels, 397.360 Testzeilen mit 1.589.440 Wahrscheinlichkeiten,
1.274.548 synthetische Elternpaare und sämtliche 462 Optimierungsstarts separat
geprüft. Zwei vollständige Läufe in allen zehn Ergebnisdateien bytegleich.
Sechs neue Kerntests bestanden. Gemeinsame GIS-/Numerikbibliotheken benannt.

**Aktuell:** Die Klassifikationsverbesserung aus der Dissertation wurde auf den
neuen sieben Arten nachgebildet. Experimentelle Arealindikatoren sind noch
keine validierten fossilen Auftretenswahrscheinlichkeiten. Keine automatische
Variantenübernahme oder Pollentyp-Zuordnung. Die #28-Referenz wird ausdrücklich
am gesperrten Commit `8f02794434f561f5b30228360247935f3c025cc8` bezogen; die ursprünglichen getrennten
PR-Branches enthielten nicht automatisch alle Vorarbeiten. Der gemeinsame
Integrationsstand enthält jetzt alle zwölf PR-Historien.

**Als Nächstes:** neue trainingsinterne Wahrscheinlichkeitskalibrierung mit inneren
räumlichen Teilungen und unabhängiger äußerer Bewertung vorab festlegen; weitere
Arten/Areale und eine belegte Typ–Art-/Gattungs-Zuordnung bearbeiten. Äußere
Testergebnisse dieses PR nicht zur weiteren Optimierung verwenden. Moderne
Kalibrierung kann parallel zur weiterhin offenen Monticchio-Quellenfrage laufen.

Die frühere Merge-Sperre wurde für die Zusammenführung #18–#29 durch den
aktuellen ausdrücklichen Nutzerauftrag aufgehoben. Das ändert keine fachlichen
Freigaben: keine stillen Quellenkorrekturen, Pilotfreigaben oder Autorennachrichten.
Wissenschaftliche Berichte, Quellenregister, Modellparameter und Ergebniswerte
werden unverändert erhalten. Der gemeinsame Einstieg ist [HANDOFF.md](./HANDOFF.md).
