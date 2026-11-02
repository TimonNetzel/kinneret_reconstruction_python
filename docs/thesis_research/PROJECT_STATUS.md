# Netzel-Projekt: kurzer Arbeitsstand

## Datei-Abgleich gesichert; neuer Ordner für Claude angekündigt, 02.11.2026

**Ergebnis:** [Notwendige Dateien, Lücken und lokale Kopien](MISSING_PROJECT_FILES_2026_11_02.md)
mit 39 priorisierten Anforderungen und acht getrennten Posteriorvarianten.
16 direkte Ladelücken des erhaltenen lokalen R-Einstiegs: zwölf RData/NC und
vier TXT/CSV. 369 R-Dateien lesend geprüft; 35 Dissertations- und 149 Zusatz-
`src`-Dateien mit mtime nach 2021 priorisiert. Dateizeiten sind keine gesicherte
Erstellungsgeschichte. Index von 584 Datendateien (257 RData/RDS/NC) und 18 ZIP-
Inhaltsverzeichnissen im bisherigen abgegrenzten Bestand. Keine gesamte PC-Suche.

**Positive Kopien:** Die über Automount referenzierte Ohrid-Verbreitungsdatei
liegt lokal vor. Zwei kleine Verbreitungscontainer gegen frühere Hashes bestätigt.
Automount-Verlust bedeutet daher nicht automatisch fehlender Dateiinhalt.
[Quellenregel entsprechend präzisiert](SOURCE_RECOVERY_PLAN.md).

**Neueste Nutzerinformation, noch ungeprüft:** In der nächsten Sitzung steht
zusätzlich `climate_modelling` zur Verfügung. Nutzer möchte zeitnah zu Claude
wechseln. [Vollständige Übergabe und erster Auftrag](CLAUDE_RESUME_CLIMATE_MODELLING_2026_11_02.md)
sichern den Chat-Arbeitsstand und diesen Hinweis. Claude soll den neuen Ordner
zuerst mit `main.R`/`main_v1.2.R` und der Liste abgleichen. Genauer Pfad und Inhalt
hier noch nicht bekannt; frühere Fehlbefunde gelten nur für bisherigen Suchumfang.

**Prüfung/Grenzen:** Acht neue synthetische Parserprüfungen bestanden. Frühere
200 historische Teiltests in diesem Datei-Abgleich nicht erneut gerechnet.
Originale und eingefrorene Nachweise unverändert; keine R-Ausführung, Trainings,
Sampler, Altersmodelle oder Fossilläufe. Elf Nachweisbündel bleiben offen,
keine finale Kinneret-Variante. Bestehender Branch/PR #77, kein Merge.

## Nutzerangabe zum Originalverlust dauerhaft verankert, 02.11.2026

**Neue gesicherte Kontextinformation:** Alle Dateien mit historischem Pfadpräfix
`/automount/` lagen laut Nutzer auf einer gelöschten Festplatte. Die dortigen
Originale sind verloren; benötigte Dateien müssen neu erstellt werden.
[Verbindlicher Wiederaufbauplan](SOURCE_RECOVERY_PLAN.md) sowie Hinweise in
`CLAUDE.md`, `AGENTS.md`, Arbeitsplan und Claude-Übergabe sichern die Regel für
künftige Sitzungen. Quelle ist die Nutzerangabe, keine neue Dateiinventur.

**Planfolge:** Für konkret an diesen Präfix gebundene Dateien Erzeugercode,
benötigte Eingaben/Parameter, Achsen und Prüfmaßstäbe gezielt erfassen und
Wiederaufbau vorbereiten. Nicht erneut nach dem Originaldatenträger/Speicherort
fragen oder pauschal danach suchen. Andere Fehlpfade nicht ohne Beleg zuordnen.
Neue Dateien als neu erzeugt kennzeichnen; keine automatische Originalidentität,
finale Variantenwahl oder Schließung der elf Nachweisbündel.

**Sicherung/Grenze:** Dokumentationsänderung auf Basis `91436c4` in
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
kein Merge. Eingefrorene Belege, Startaufträge und Numerik unverändert. Keine
Neuberechnungen, Trainings, Sampler, Altersmodelle oder Fossilläufe gestartet.
Nächster technischer Schritt bleibt die begrenzte TF-Vorbereitung.

## Gemeinsame synthetische Rechenkette geprüft, 02.11.2026

**Abgeschlossen:** [Gemeinsame Fixture, Bruchrechnung und Grenzen](literal_phase1_chain_2026_11_02/README.md)
prüfen die vier unveränderten Teilmodule über ihre Schnittstellen. Chronologie
4×5 → Probentiefenkern 4×3 → Pollen 4×2 → RW-Koeffizienten/TF-Produkt →
Roh-CDF-Scores für Original-IDs 1 und 3 → Rohsumme und normierte Kunst-PDF.
Nur Test-/Belegcode; kein historischer Runner und keine App-/CLI-Erweiterung.

**Prüfung:** 33 neue Tests, mit Vorpaketen 200 grün. Unabhängige skalare
`Fraction`-Rechnung ohne NumPy-/Modulimporte bestätigt sämtliche numerischen
Zwischenstufen bis auf höchstens 2.23e-16 (Toleranz 2e-14). Energiegrößen werden
keine Mischungsgewichte; nur verbleibende Original-IDs erhalten Scores.
Gegenbeispiele erkennen verfrühte Normierung, ausgelassene Nullersetzung,
Achsen-/Masken-/Platzhalterkonflikte und semantisch vertauschte TF-Zeilen.
Drei Skripthashes entsprechen vor/nach Lesen den bisherigen Bindungen.

**Bewusste Grenze:** Die TF-Massen 2 und 3 sind künstliche Schnittstelleneingaben;
sie belegen keinen historischen TF-Vorbereitungslauf. Normierung jeder Ziehung
vor der Summe verändert die Kunst-PDF maximal um 0.00282, stilles Normieren
der gelieferten TF-Zeilen um 0.09716. Keine historische R-Laufzeitgleichheit,
keine Originalnumerik, keine Wiederherstellung fehlender TFs oder Freigabe.

**Claude kann direkt fortsetzen:** [Neuer konkreter Startauftrag](CLAUDE_NEXT_STEP_TF_PREPARATION_2026_11_02.md)
mit portablen, hashgebundenen Originalauszügen, Formeln, Beispiel, Prüfkommandos
und Abnahmekriterien. Nächster begrenzter Schritt: TF-Zeilennormierung und
anschließendes Nullsetzen ausdrücklich maskierter Klimazellen aus
`mcmc_prepare.R:1581–1586,1613–1618`. Keine unbelegte `inside.range`-Randkonvention
wählen, keine neue TF schätzen oder umgebende Zufalls-/Medianpfade ausführen.
Der bisherige Startauftrag bleibt als eingefrorener Nachweis unverändert.

**Weiter offen:** Elf Nachweisbündel, finale Basis/Listenchronologie, numerische
historische TFs und Laufzuordnung. Kein vollständiger Fall reproduziert und
keine finale Kinneret-Variante gewählt. Keine Speicherortfrage oder Inventur.

**Sicherung:** Basis `674bf13`, bestehender Branch in
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
kein Merge. App/CLI, Baseline, Originale, Referenzen, bisherige Pakete, Module
und fallbezogene Monticchio-/Chronologie-Gates erhalten. Keine Trainings,
Sampler, Altersmodelle oder Fossilrekonstruktionen. Phase 1 vor Phase 2;
Plots jung links, alt rechts.

## Kumulative Rohdichte-/EMD-Teilprüfung und Claude-Auftrag, 02.11.2026

**Abgeschlossen:** [Quellen, Formeln und Grenzen](literal_cdf_discrepancy_2026_11_02/README.md)
belegen den Nullreferenzzweig und kumulierten Rohdichtevergleich aus
`final_reconstruction`; separates Modul `tools/historical_cdf_discrepancy.py`.
Die erhaltenen Fallstudienaufrufe übergeben zweimal Skalar 1. Der einseitige
Längentest der ersten Referenz ersetzt beide Referenzen durch Nullmatrizen.
Verglichen wird vor späterer Mischungsnormierung mit `sum(abs(C−R))/dims`;
kein Klima-Zellbreitenfaktor und keine zusätzliche Normierung.

**Wichtige Grenze:** Die Platzhalteranlage verwendet `dims` vor der lokalen
Zuweisung. Der Haupteinstieg setzt zwar 300, aber historische Aufrufumgebung
und finale TF-Breite fehlen. Geerbte und spätere Breite werden ausdrücklich
getrennt übergeben; Konflikte nicht still korrigiert. Nullreferenzscore ist
kein belegter Abstand zu einer älteren Rekonstruktions-CDF oder in °C/mm.

**Prüfung:** 43 neue synthetische Tests, mit Vorpaketen 167 grün. Rationale
Scorekontrollen exakt gleich; drei gezielt gelesene Skripte vor/nach Lesen
identisch zu bisherigen Hashbindungen. Keine Originalnumerik oder historische
R-/Paketlaufzeitgleichheit. Nichtendlichkeit, Recycling, Achsenreparatur und
allgemeine R-Längensemantik außerhalb der dokumentierten Domäne bleiben aus.

**Claude kann fortsetzen:** [Konkreter Startauftrag](CLAUDE_NEXT_STEP_2026_11_02.md)
enthält Modulkarte, Prüfkommandos, erwarteten Ausgang und Abnahmekriterien.
Nächster begrenzter Schritt: vorhandene Module über ein gemeinsames künstliches
Beispiel auf Reihenfolge, Achsen und Normierungszeitpunkte prüfen. Nur neues
Testmodul/Belegpaket, kein historischer Vollrunner und keine App-/CLI-Erweiterung.
Exakte Energieduplikate vor aktiven Ziehungen bestimmen, Original-IDs erhalten.

**Weiter offen:** Elf Nachweisbündel, finale Basis/Listenchronologie, numerische
historische TFs und tatsächliche Laufzuordnung. Keine finale Kinneret-Variante
gewählt, kein vollständiger Dissertationsfall reproduziert. Keine Speicherort-
frage, Globalinventur oder Ersatzdatenwahl.

**Sicherung:** Basis `d6e3343`, bestehender Branch in
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
kein Merge. App/CLI, Baseline, Originale, Referenzen, bisherige Belegpakete und
fallbezogene Monticchio-/Chronologie-Gates erhalten. Keine Trainings, Sampler,
Altersmodelle oder Fossilrekonstruktionen. Phase 1 vor Phase 2; Plots jung links,
alt rechts.

## Historische Altersraster und Zeilenauswahl geprüft, 02.11.2026

**Abgeschlossen:** [Quellen, Zustände, Formeln und Grenzen](literal_age_selection_2026_11_02/README.md)
verfolgen `age`, `site.age` und `reconst_ages` durch vier bereits gebundene
Skripte. Separates Modul `tools/historical_age_selection.py`: exakte Perioden-
rastermitgliedschaft, vorläufige/wirksame Maske des erhaltenen Kontrollflusses
und boolesche RW-Zeilenauswahl vor dem Matrixprodukt. Kein Runner.

**Belegt:** `age %in% grid` ist kein Intervalltest; erste Altersdifferenz
bestimmt die Rasterweite. Die erhaltene Vorbereitung setzt zunächst bei
`taxa.merge` eine Holozänmaske, überschreibt sie im wegen `& FALSE` erreichten
Alternativzweig wieder mit allen Alterszeilen. Das gilt nur bei erfolgreichem
Erreichen dieser Zeilen, nicht als historische Laufbindung. Gibbs verwendet
an dieser Stelle keine `reconst_ages`-Maske. Drei Masken-/Schaltgrößen sind
freie Umgebungsvariablen von `final_reconstruction`, keine formalen Parameter.
`site.age` bleibt separat und kann im Speläothemzweig überschrieben werden.

**Prüfung:** 39 neue synthetische Tests, mit den bisherigen Teilpaketen 124
Tests grün. Rechteckige Produkte und rationale Sollwerte stimmen in der
Prüfung exakt überein. Vier gelesene Quellskripte entsprechen vor/nach dem
Lesen ihren bisherigen Hashbindungen. Kein historischer R-Laufzeitvergleich.
Halbe/ganzzahlige Rastereinheiten und strikt passende Boolmasken begrenzen
den Nachbau; keine stillen R-Recycling-/Dimensionsverlustreparaturen.

**Grenzen:** Elf Nachweisbündel unverändert offen. Finale Originalbasis,
Listenchronologie, numerische historische TFs und tatsächliche Laufzuordnung
fehlen weiterhin. Kein finaler Kinneret-Pfad gewählt, keine Ersatzdaten und
kein vollständiger Dissertationsfall reproduziert. Keine Synchronisierung
von `age`/`site.age` oder Aktivierung des deaktivierten Taxonwechselzweigs.

**Nächster Einstieg:** Kumulativen Dichte-/EMD-Ausschnitt im gebundenen
`src/mcmc_functions_v1.6.R:366–370,485–490` gegen Fallstudienaufrufe mit skalaren
Referenzargumenten lesen: Normierungszeitpunkt, `dims`-Faktor, Referenzform und
Einheiten belegen und synthetisch prüfen. Keine neue Referenzkurve oder
Kalibrierung; getrennte Module noch nicht zu einem Vollrunner verbinden.

**Sicherung:** Basis `6241f41`, bestehender Branch in
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
kein Merge. App/CLI, Baseline, Originale, Referenzen, bisherige Belegpakete und
fallbezogene Monticchio-/Chronologie-Gates erhalten. Keine Trainings, Sampler,
Altersmodelle oder Fossilrekonstruktionen. Phase 1 vor Phase 2, Plots jung links,
alt rechts. Keine wiederholte Speicherortfrage oder pauschale Inventur.

## Historischer Alters-/Tiefenteil synthetisch geprüft, 02.11.2026

**Abgeschlossen:** [Quellen, Formeln und Grenzen](literal_age_depth_2026_11_02/README.md)
binden `check_ages.R` an den tatsächlich rechnenden Verbraucher
`mcmc_prepare.R`. Getrenntes Modul `tools/historical_age_depth.py`: explizite
Listenposition, einmalige Altersduplikatverschiebung, Auswahl der Tiefenspalten,
Zeilennormierung, historische Nullersetzungen und Pollen-Matrixprodukt.
Orientierung rho(D|A): Alter × Tiefengitter, dann Alter × Probentiefe;
Produkt mit Probentiefe × Taxon ergibt Alter × Taxon. Keine finale Auswahl.

**Prüfung:** 43 neue synthetische Tests bestanden, zusammen mit Taxongewichtung/
PDF-Summe und Belegvertrag 85 Tests grün. Rechteckige Matrizen und rationale
Sollwerte prüfen Orientierung und Reihenfolge; maximaler Fehler 2.23e-16.
Historisches `10e-10` nach Normierung bleibt 1e-9 ohne Renormierung;
Pollennullen werden separat zu 1e-6. Altersverschiebung um 0,5 kann Duplikate
übriglassen und verändert `site.age` nicht. Drei Quellhashes stimmen vor/nach
Lesen mit der vorherigen Bindung überein. Kein R-Laufzeitvergleich.

**Grenzen:** Mittelwertmatrix ersetzt keine Liste mehrerer Altersraster.
Ungeordnete/doppelte oder nicht passend indizierbare Tiefen werden nicht still
repariert. Standortüberschreibungen, Fallback, Birkat-Ram-LOESS und NA-Füllung
sind ausgeschlossen. Alle elf Nachweisbündel bleiben offen; finale Originalbasis,
Listenchronologie, historische TFs und Laufbindung fehlen weiterhin.
Kein vollständiger Fall reproduziert, keine neue Chronologie oder Variante.

**Nächster Einstieg:** Verbraucher von `age` gegenüber `site.age` und
`reconst_ages` im bereits gebundenen `src/mcmc_functions_v1.6.R` gezielt auf
Zeilen-/Zeitfensterauswahl verfolgen und einen belegten Ausschnitt synthetisch
prüfen, bevor getrennte Module verbunden werden. Keine neue Gesamtinventur,
Speicherortfrage, Ersatzdatenwahl oder Phase 2.

**Sicherung:** Basis `f7d97c1`, bestehender Branch in
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
kein Merge. App/CLI, Baseline, Originale, Referenzen, bisherige Belegpakete und
fallbezogene Monticchio-/Chronologie-Gates erhalten. Keine Trainings, Sampler,
Altersmodelle oder Fossilrekonstruktionen. Plots weiter jung links, alt rechts.

## Historischer deterministischer Teilnachbau, 02.11.2026

**Abgeschlossen:** [Taxongewichtung und PDF-Summe](literal_final_pdf_2026_11_02/README.md)
als getrenntes Modul `tools/historical_final_pdf.py`. Belegte Zeilen der erhaltenen
`final_reconstruction`-Fassung nachgebildet: Spaltenmaximum, RW-/Gibbs-Koeffizienten,
TF-Matrixprodukt, exakte Energieduplikate und anschließende PDF-Zeilensumme.
Kein Runner, keine App-Integration, keine automatische finale Variantenwahl.

**Prüfung:** 22 neue Tests mit handgeleiteten rationalen Sollwerten und
Grenzfällen bestanden; zusammen mit den bisherigen 20 Belegvertragstests
42 Tests grün. Uneinheitliche TF-Massen und rechteckige Proben-/Taxonmatrix
prüfen Normierungszeitpunkt und Reihenfolge. Energie wird nicht zum
Mischungsgewicht. Quellenhash vor/nach Lesen gleich dem Vorpaket. Kein R-
Laufzeitvergleich: nur Formeln im dokumentierten endlichen, nichtnegativen
Eingabebereich geprüft. NA-/Null-/Dimensionsverlustpfade nicht still repariert.

**Grenze:** Die elf offenen Nachweisbündel und finale Laufbindung bleiben
unverändert offen. Dies ist Schritt 4 als isolierter technischer Ausschnitt,
keine abgeschlossene historische Fallreproduktion und keine Ersatzdatenwahl.
**Nächster unabhängiger technischer Einstieg:** Alters-/Tiefenübertragung aus
dem bereits gebundenen `src/check_ages.R` isolieren und mit synthetischen
Matrizen prüfen. Finale Listenchronologie nicht durch Mittelwertmatrix ersetzen;
keine neuen Altersmodelle oder Fossilkurven. Neue Originalbelege für einen
vollständigen Lauf bleiben notwendig, Speicherortfrage nicht wiederholen.

**Sicherung:** Basis `74999b0`, bestehender Branch in
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
kein Merge. App/CLI, Baseline, Originale, Referenzen und Monticchio-/Chronologie-
Gates erhalten. Keine Trainings, Sampler oder Fossilrekonstruktionen; Phase 1
bleibt vorrangig. Für spätere Plots jung links, alt rechts.

## Kinneret: Phase-1-Belegvertrag maschinenprüfbar, 02.11.2026

**Abgeschlossen:** [Spezifikation, Prüfer und Grenzen](kinneret_phase1_spec_2026_11_02/README.md)
führen Eingabe-, Referenz- und Nachbearbeitungsbelege zusammen. Lesender
Prüfer bestätigt 26 eingefrorene Dateien, 19 explizite Befunde und alle 50
Snapshotdateien. Vier Quellgruppen bleiben getrennt, keine finale Variante
gewählt. Elf Nachweisbündel offen; kein Fall vollständig reproduzierbar.

**Geprüft:** 20 gezielte Integritätstests bestanden. Portabler und lokaler
CLI-Aufruf ergeben erwarteten Exitcode 2 (offene Voraussetzungen). Der gezielte
lokale Hashabgleich von 40 bereits dokumentierten Pfaden bestätigt 28 Dateien
unverändert und zwölf Pfade weiterhin fehlend. Keine neue Inventur und keine
Prüfung aller globalen Abhängigkeiten. Neu auftauchende Dateien schließen
keinen Nachweis automatisch. Prüfer ist kein Runner oder App-Freigabesystem.

**Nächster Einstieg/Grenze:** Finalen Originaleingabestand und Laufzuordnung
beschaffen, beginnend mit Basis/Chronologieliste und numerischen TFs, dann an
den eingefrorenen Vertrag binden. Ohne neue Originalbelege kein historischer
Vollnachbau. Der Nutzer kennt keinen weiteren Basisdatei-Speicherort; nicht
erneut danach fragen oder pauschal suchen. Ein Nachbau mit Ersatzdaten wäre
eine gesonderte, noch nicht getroffene fachliche Entscheidung. Keine automatische
Fortsetzung mit Verbesserungen, Kalibrierung oder Phase 2.

**Sicherung:** Basis `00d7d2a`, bestehender Branch in
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
kein Merge. Originalquellen nur gelesen; Vorpakete/Snapshot und App/CLI,
Baseline, Referenzen sowie Monticchio-/Chronologie-Gates erhalten. Keine neuen
Trainings, Sampler oder Fossilrekonstruktionen. Jung links, alt rechts.

## Kinneret: Nachbearbeitungsfassungen gebunden, 02.11.2026

**Abgeschlossenes Quellenpaket:** [Ziehungsauswahl und Dichtekombination](kinneret_postprocessing_2026_11_02/README.md)
trennt Dissertationstext, Fallstudiencode, Vergleichsfassungen und aktuellen
Trimming-Einstieg. Zwei deterministische synthetische Kontrollen belegen den
möglichen Verlust der Originalindizes nach Filterung und den Unterschied
zwischen Dichtemischung und Mittel unabhängiger Zufallsgrößen. Keine Original-
ziehungen verändert und keine Klimakurven berechnet.

**Konkret belegt:** Die Rückabbildung auf `best_combs` ist im Fallstudiencode
kommentiert, in Vergleichsfassungen aktiv. Der Fallstudien-IAT-Refinementloop
ist deaktiviert; der aktuelle `mcmc_trimming.R` schaltet sogar den ganzen
IAT-Zweig ab. Fallstudiencode nutzt PDF-Mischung statt des textlich beschriebenen
Zufallsgrößenmittels; der heutige aktive Einstieg bildet eine KDE über gespeicherte
Klima-Punktwerte. Keine dieser Fassungen ist als finaler Kinneret-Lauf identifiziert.

**Weiter offen:** Alle acht explizit geprüften Kinneret-Posteriorpfade fehlen;
Original-IDs/zehn gewählte Ziehungen und numerische Enddichten sind nicht
wiederhergestellt. Basis-, TF-, Referenz-/CRU- und Laufbindungslücken bleiben
bestehen. Grafik-Bytegleichheit zum Dissertationsarchiv bestätigt nur das
Ausgabeartefakt, keine Auswahl-/Kombinationsfassung. Keine stillen Reparaturen.

**Als Nächstes:** Die gebundenen Eingabe-, Referenz- und Nachbearbeitungsbefunde
zu einer maschinenprüfbaren Phase-1-Reproduktionsspezifikation zusammenführen.
Textverfahren und Quellvarianten getrennt halten; unbekannte Originaleingaben,
Indizes, Achsen, Paket-/RNG- und Ausführungsfassung als offen ausweisen.

**Prüfung/Sicherung:** Quellenhashes vor/nach dem Lesen gleich, synthetische
Kontrollen ohne Zufall, Nachweise wiederholbar; Vorpakete und Forschungssnapshot
erhalten. Basis `edfd3f1`, bestehender Branch in
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
kein Merge. Keine R-Ausführung, IAT-Schätzung, Dichteanpassung, Trainings,
Sampler oder Fossilrekonstruktionen. App/CLI, Baseline, Originale und Monticchio-/
Chronologie-Gates erhalten; Phase 1 vorrangig. Jung links, alt rechts.

## Kinneret: Referenz-/Ankerpfad gebunden, 02.11.2026

**Abgeschlossenes Quellenpaket:** [Referenz-/Ankervertrag](kinneret_references_2026_11_02/README.md)
trennt Dissertationstext und erhaltene Verbraucher. 160 AP-Prozentwerte als
100×AP/(AP+NAP) geprüft; zwölf konkrete Ladepfade fehlen weiterhin. Source-
Auszüge, Hashes, Verzweigungen und ein gezielt gelesener späterer Speläothem-
Container sind dokumentiert; keine neue Referenz- oder Klimakurve berechnet.

**Wichtige Befunde:** Die AP-Altersmaske bleibt bei `hi_time=FALSE` wirksam.
Soreq-Auswahl 105 wird im neueren aktiven Speläothemzweig durch 170 und eine
0–220500-a-Zeitachse mit Lücke überschrieben; die ältere Vorbereitung setzt
für Kinneret 37. Keine dieser Positionen ist ohne den fehlenden Workspace als
konkrete Reihe identifiziert. MEDSTACK wird über Klassen-/Endposition gewählt.
Bei Altern ≤0 verwendet der Ankerpfad eine lokale Zeitreihe, sonst Klassen-
Klimamittel. Zeitindizes 50:115 sind ohne Zeitachse keine belegten Kalenderjahre.
Die heutigen Quellstände sind damit keine laufgebundene finale Konfiguration.

**Weiter offen:** finale Basis und TFs, ursprüngliche Isotopen-/CRU-Workspaces,
Paket-/Zeitachsenbindung, numerische Anker und tatsächliche historische
Ausführungsfassung. Der Nutzer kennt keinen weiteren Basisdatei-Speicherort;
keine wiederholte pauschale Suche oder Ersatzdatenübernahme.
**Als Nächstes:** unabhängig davon die historische Nachbearbeitung in
`mcmc_case_studies.R` gegen Dissertation und aktuellen Trimming-Einstieg binden:
Ziehungsauswahl, Prior-/IAT-/Pareto-Schritte und Dichtekombination.

**Geprüft und Sicherung:** Wiederholbare Metadaten-/Prozentprüfung, Quellen vor/
nach dem Lesen unverändert, Vorpakete und Forschungssnapshot erhalten. Fortsetzung
auf bestehendem Branch in [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
Basis `44092d3`; kein Merge. Keine R-Ausführung, Fits, Wavelets, Altersmodelle,
Sampler oder Fossilrekonstruktionen. App/CLI, Baseline, Referenzdaten und alle
Monticchio-/Chronologie-Gates bleiben erhalten; Phase 1 zuerst. Jung links, alt rechts.

## Kinneret: ältere Proben-/Taxonbasis gebunden, 02.11.2026

**Abgeschlossenes Teilpaket:** [Kinneret-Eingabebindung](kinneret_binding_2026_11_02/README.md)
mit 160-zeiligem Proben-Crosswalk, zwölf exakt gebundenen älteren Taxonspalten
(1920 Werte), identischem numerischem Chronologieblock (283520 Werte) und
drei expliziten aktuellen TF-Namens-/Poolzuweisungen. Eingaben nur gelesen;
keine neuen Altersmodelle, TF-Ziehungen oder Fossilkurven berechnet.

**Grenzen konkretisiert:** Der Einzelcontainer ist ein 35-Taxa-/3-Biomzweig.
Der ältere Sammelstand besitzt zwölf andere Taxonspalten; die jüngeren
positionsbezogenen Umbenennungen würden dort u.a. Pinus als Olea beschriften.
Die erhaltene mittlere Chronologiematrix ersetzt keine vom finalen Code
erwartete Liste mehrerer Altersraster. Gewichtsplots besitzen elf bzw. zehn
Anzeigeeinträge und überschreiben Taxonlabels; daraus folgt kein eindeutiger
finaler TF-Crosswalk. Die vollständige historische TF-Teilmenge bleibt offen.

**Quelllücke:** `core_data_additional_sites_v1.2.rdata` samt finalem Kinneret-
Eintrag oder dessen genauer Erzeuger fehlt weiter. Der Nutzer kennt laut
Antwort vom 02.11.2026 keinen weiteren Speicherort. Keine erneute pauschale
Inventur oder ungefragte Ersatzrekonstruktion. **Als Nächstes unabhängig davon:**
Kinnerets Referenz-/Ankervertrag anhand der bekannten Verbraucher und vorhandenen
Nachweise konkretisieren: AP-Maske, Soreq/MEDSTACK, Csa und konsumierte CRU-
Zeiträume. Die offene Basisdatei dabei nicht als erledigt behandeln.

**Geprüft und Sicherung:** Quellenhashes vor/nach dem Lesen gleich;
Spaltenidentitäten ohne Toleranz, Chronologie als kompletter XDR-Block,
Plotlabels und Archivgleichheit geprüft. Nachweise im Paket. Fortsetzung auf
bestehendem Branch in [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
Basis `756865b`; kein Merge. Baseline, Referenzen, vorherige Fallmatrix,
App/CLI, sieben Arten und Monticchio-/Chronologie-Gates bleiben erhalten.
Keine Trainings oder Sampler; Phase 1 bleibt vorrangig. Jung links, alt rechts.

## Lokale Dissertationsfälle: Fallmatrix abgeschlossen, 02.11.2026

**Abgeschlossen:** [Quellengebundene Fallmatrix](local_case_matrix_2026_11_02/README.md)
für sieben Fälle an sechs Standorten, Totes Meer Holozän/Glazial getrennt.
Lokale Vollfassung und Repository-Ausschnitt mit eigenen Seiten-/Abbildungsbelegen
geprüft. Ohrid ausdrücklich 1165–1365 ka; Van nicht still auf 130 ka begrenzt.
Fossil-/Chronologiekandidaten, Taxa/TFs, Klima, Referenzen, Anker, historische
Code-/Nachbearbeitungsfassungen und Originalausgaben fallweise zugeordnet.

**Ergebnis und Grenzen:** Noch kein Fall vollständig numerisch reproduzierbar.
Historische numerische TFs, geordnete Zuordnung, globale Klima-/Referenzeingaben
und genaue Laufbindung bleiben offen. Gezielte Prüfungen belegen zusätzliche
Chronologie-/Versionslücken bei En Gedi, Birkat Ram, Van, Iznik und Ohrid.
Elf von 14 Original-/Archivgrafikpaaren bytegleich; drei abweichende Stände am
Toten Meer separat erhalten. Grafiken sind keine numerischen Originalposterioren.

**Als Nächstes:** Kinneret als ersten Klärungsfall erschließen: 160 Pollenzeilen,
160 Chronologieeinträge und beide bytegebundene Archivgrafiken liegen vor.
Den historischen `core.data[[3]]`-Basiseintrag an lokale Einzelcontainer und
Erzeuger binden, Proben-/Tiefen-/Pollentyp-/TF-Reihenfolge festhalten und erst
daraus die benötigte numerische TF-Teilmenge ableiten. Keine pauschale Inventur,
Neukalibrierung oder neue Reliabilitätsansicht vorziehen.

**Geprüft und gesichert:** Explizite Quellen vor/nach dem Lesen unverändert;
getrennte PDF-Seiten-/Abbildungsprüfung, strukturierte RData-/CSV-Metadaten,
Archivhashvergleich und Dokumentationslinks geprüft. Nachweise im Matrixpaket.
Ausgangspunkt `865d155`, sauberer Checkout und identischer offener PR #77
vor Arbeitsbeginn bestätigt. Sicherung auf bestehendem Branch in
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
kein Merge. Keine Trainings, Sampler oder Fossilrekonstruktionen. Paper-Biomfall,
sieben moderne Arten, App/CLI, Baseline, Referenzdaten und Monticchio-/Chronologie-
Gates bleiben erhalten. Phase 1 fortsetzen; Plots jung links, alt rechts.

## Priorität geändert: lokale Dissertation zuerst, Verbesserungen danach, 02.11.2026

**Abgeschlossen – Planänderung auf ausdrücklichen Nutzerauftrag:**
[Verbindlicher Arbeitsplan](LOCAL_DISS_RECONSTRUCTION_PLAN.md) mit zwei Phasen.
Phase 1 bindet Originaleingaben und historischen Ablauf der lokalen
Dissertationsrekonstruktionen, reproduziert zunächst einen belegbaren Fall
und erschließt danach die weiteren lokalen Fälle. Bekannte methodische Kritik
bleibt dokumentiert; Verbesserungen werden getrennt gegen den gesicherten
historischen Referenzstand geprüft. Neue Kalibrierung und zusätzliche
Reliabilitätsansicht sind keine Voraussetzungen und keine nächsten App-Aufträge.

**Als Nächstes:** Quellengebundene Fallmatrix der lokalen Dissertationsfälle
anhand vorhandener Berichte und Belege erstellen: Daten, Chronologie, TFs,
Zuordnungen, Klima-/Referenzeingaben, Codefassung, Originalausgaben und Lücken.
Danach ersten Referenzfall nach belegter Eingabeverfügbarkeit festlegen.
Paper-Biomfall und sieben moderne Artenmodelle bleiben erhaltene Vorarbeiten;
sie sind keine abgeschlossene historische Taxonreproduktion.

**Grenzen und Sicherung:** Originaleingaben und numerische historische TFs
bleiben teilweise offen. Monticchio-/Chronologie-Gates bleiben fallbezogen
erhalten. Keine Trainings, Sampler oder Fossilfälle durch diese Planänderung.
Reine Dokumentationsänderung; Einstiege, Prioritäten und lokale Links geprüft.
Ausgangsstand `4b75ead`, Checkout sauber und mit GitHub identisch; Sicherung auf
bestehendem Branch in [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
kein Merge. Jung links, alt rechts. Frühere „Als Nächstes“-Angaben unten
beschreiben den damaligen Stand und werden durch diesen Plan nachgeordnet.


## App: äußere Foldwerte erschlossen, 02.11.2026

**Abgeschlossen:** Im Artenvergleich zwischen Übersicht und Foldwerten wechseln,
Art und Blockgröße wählen, fünf äußere Testfolds mit BA/Brier und Testzellzahlen
lesen. Beide Offlineformate und Sprachen; keine Neuberechnung oder Modellwahl.
280 Foldzeilen und 56 Gesamtzeilen an gespeicherte Prüfsummen gebunden.
Alle tatsächlichen Folds enthalten beide Klassen; hypothetisch fehlende Klassen
bleiben in Negativkontrollen ausdrücklich undefiniert. Folds sind keine
unabhängigen Wiederholungen oder fossilen Unsicherheitsintervalle.

**Geprüft:** 23 App-/Datentests, acht gezielte Offline-Browserfälle mit beiden
Sprachen, sämtliche Fold-/Gesamtwerte und Zellzahlen direkt abgeglichen.
Fehlende Dateien, Klassen und widersprüchliche Daten geprüft. Standortwechsel
und visuelle Mobil-/Desktopkontrolle bestanden. Vollständige Browserregression
mit acht Durchläufen und beiden Gestenprüfungen bestanden. Alle 50 portablen
Forschungsnachweise erhalten.
[Ergebnis, Grenzen und Nachweise](app_folds_2026_11_02/README.md).

**Sicherung:** Bestehender Branch und [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
Ausgangsstand `2f6d0eb`, kein Merge. Keine Trainings, Sampler oder Fossilfälle.
Baseline, Referenzdaten und Monticchio-/Chronologie-Gates bleiben erhalten.
**Als Nächstes:** Schema, Gewichtung und leere Bins der gespeicherten
Reliabilitätstabelle prüfen; rein diagnostische Ansicht erwägen. Keine neue
Kalibrierung; Altersplots jung links, alt rechts.


## App: Sieben-Arten-Vergleich umgesetzt, 02.11.2026

**Abgeschlossen:** Lesende Ansicht „Artenvergleich“ mit Balanced Accuracy und
Brier nebeneinander je Art, 5°-/10°-Blockwahl und beiden Blockdesigns im
ungewichteten Artenmittel. Netze ohne/mit SMOTE standardmäßig sichtbar;
Regression #28 und Prior-Diagnose ergänzbar. Experimenteller Status,
Gewichtung, Testdesign und sechs feste Quellenlinks sichtbar. Beide
Offlineformate, Deutsch/Englisch, keine neuen Fits oder Fossilläufe.

**Geprüft:** 15 App-/Datentests, acht gezielte Offline-Browserfälle mit beiden
Sprachen; 56 Original-Metrikzeilen und acht Artenmittel direkt abgeglichen.
Fehlende/veränderte Metriken sichtbar behandelt, Mobil/Desktop-Bilder geprüft.
Vollständige Browserregression (acht Fälle, zwei Gestenprüfungen) und
Standortwechsel bestanden. Alle 50 bisherigen Forschungsnachweise unverändert. [Ergebnis und Nachweise](app_species_2026_11_02/README.md).

**Sicherung:** Bestehender Branch und [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
kein Merge. Ausgangsstand `2d2fdfd`, Checkout sauber und mit GitHub identisch.
Baseline, Referenzdaten und Monticchio-/Chronologie-Gates bleiben erhalten.
**Als Nächstes:** Schema und undefinierte Werte der gespeicherten Foldscores
prüfen, dann eine rein deskriptive Detailansicht erwägen. Keine neuen Trainings,
Sampler oder Fossilrekonstruktionen; Altersplots jung links, alt rechts.


## App: Forschungsstand und Quellen umgesetzt, 02.11.2026

**Abgeschlossen:** Neue lesende Ansicht „Forschungsstand“ in Ordner- und
Ein-Datei-App, offline und Deutsch/Englisch. Fünf Bereiche trennen vorhandene,
experimentelle und fehlende Ergebnisse; Berichte sind an einen festen Commit
gebunden. Fehlende oder veränderte Nachweise werden sichtbar behandelt.
Keine neuen Modelle, Sampler oder Fossilrekonstruktionen.

**Geprüft:** 11 App-/Daten-/Buildtests bestanden; zwei normale Browserwrapper
übersprungen, durch explizite Browserprüfungen ergänzt. Acht vollständige
Browserdurchläufe plus zwei Gestenprüfungen und vier abschließende
Nachweis-/Sprachprüfungen erfolgreich. Mobil/Desktop, Hell/Dunkel, beide
Offlineformate; Bilder visuell geprüft. Bisherige Ansichten/Exporte erhalten.
Das portable Paket mit 50 Nachweisdateien bleibt unverändert. Die CI-Korrektur
`954746c` ist auf GitHub erfolgreich geprüft.
[App-Ergebnis und Nachweise](app_research_2026_11_02/README.md).

**Gesichert:** Fortsetzung im selben Branch und [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).
Abgeschlossene, geprüfte Arbeitspakete auf Nutzerwunsch jeweils hochladen;
keine automatischen Merges. Basis, Originale und wissenschaftliche Gates bewahren.
**Als Nächstes:** Vorhandene Sieben-Arten-Metriken in der App gemeinsam mit
Balanced Accuracy und Brier verständlich darstellen, ohne neue Trainings-
oder Fossilläufe. Jung links, alt rechts.


## CI-Testabhängigkeit korrigiert, 02.11.2026

Die frische GitHub-CI von [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77)
scheiterte beim Einsammeln der neuen Lesertests an fehlendem `rdata`.
Das Test-Extra enthält jetzt die lokal geprüfte Version `rdata==1.1.0`,
da die Leser deren interne Parserstrukturen verwenden. App-Laufzeitabhängigkeiten
und wissenschaftliche Ergebnisse bleiben unverändert. 67 betroffene Tests
lokal erneut bestanden; der neue GitHub-Lauf prüft beide Python-Versionen.

Die weitere Arbeit erfolgt hier im Chat; abgeschlossene, geprüfte Pakete werden
auf dem bestehenden Branch gesichert. Nächster App-Schritt bleibt die lesende
Forschungsstands-/Quellenansicht.


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
