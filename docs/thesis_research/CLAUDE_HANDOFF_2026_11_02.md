# Claude-Übergabe: Dissertation zur überprüfbaren App weiterführen

> **Aktueller Einstieg nach Altersraster-/Zeilenauswahlprüfung, 02.11.2026:**
> [Quellen, Formeln, Kontrollfluss und Grenzen](literal_age_selection_2026_11_02/README.md).
> Separates Modul `tools/historical_age_selection.py`, 39 neue Tests und mit
> Vorpaketen 124 grün. Raster verwenden exakte Mitgliedschaft und erste
> Altersdifferenz. Vorläufige Holozänmaske wird im erhaltenen Kontrollfluss
> wieder auf alle Zeilen gesetzt; `& FALSE` hält Taxonwechsel ausgeschaltet.
> Das beweist keinen historischen Ausführungserfolg. `age`/`site.age` bleiben
> getrennt; Masken sind freie Variablen von `final_reconstruction`.
> Vier Quellhashes vor/nach Lesen gleich. Keine R-Laufzeitgleichheit/finale
> Laufbindung; elf Nachweisbündel offen. Nächster unabhängiger Einstieg:
> kumulative Dichte/EMD in `src/mcmc_functions_v1.6.R:366–370,485–490` gegen
> Fallstudienaufrufe mit skalaren Referenzargumenten belegen und synthetisch
> prüfen. Keine neue Referenzkurve, Kalibrierung oder Verbindung zum Vollrunner.
> Phase 1 vorrangig; ältere Einstiege unten sind Verlauf. Keine Speicherortfrage,
> neuen Trainings/Sampler/Altersmodelle/Fossilläufe oder automatische Merges.

> **Aktueller Einstieg nach Alters-/Tiefenteilprüfung, 02.11.2026:**
> [Quellen, Formeln, synthetische Prüfung und Grenzen](literal_age_depth_2026_11_02/README.md).
> `tools/historical_age_depth.py` bildet begrenzte Teile aus `check_ages.R`
> und seinem Verbraucher nach. 43 neue Tests; mit bisherigen Paketen 85 grün.
> rho(D|A)-Orientierung, 1e-9 nach Normierung ohne Renormierung, Pollennullen
> 1e-6 und einmalige Altersverschiebung getrennt dokumentiert. Drei Quellhashes
> unverändert. Keine historische R-Gleichheit oder finale Laufbindung.
> Mittelwertmatrix ersetzt keine Listenchronologie; elf Nachweisbündel offen.
> Nächster unabhängiger technischer Einstieg: `age`/`site.age`/`reconst_ages`
> im gebundenen `src/mcmc_functions_v1.6.R` auf Zeilen-/Zeitfensterauswahl
> prüfen, bevor getrennte Module verbunden werden. Keine stillen Reparaturen,
> Originalersatzwahl, neue Läufe oder Speicherortfrage. Frühere Einstiege unten
> sind Verlauf. Phase 1 bleibt vorrangig; PR #77 offen, kein Merge.

> **Aktueller Einstieg nach deterministischem Teilnachbau, 02.11.2026:**
> [Taxongewichtung/PDF-Summe, Tests und Grenzen](literal_final_pdf_2026_11_02/README.md).
> `tools/historical_final_pdf.py` bildet ausschließlich belegte deterministische
> Ausschnitte der erhaltenen R-Funktion nach. 22 neue Tests, mit Belegvertrag
> zusammen 42 bestanden. Keine R-Laufzeitgleichheit oder finale Laufbindung
> behauptet. Dokumentierter endlicher Eingabebereich; keine stillen Reparaturen.
> Nächster unabhängiger technischer Teil: Alters-/Tiefenübertragung aus dem
> bereits gebundenen `src/check_ages.R` lesen und synthetisch prüfen. Keine
> neue Chronologie und keine Gleichsetzung von Matrix-/Listenformen. Für einen
> vollständigen Fall bleiben Originalbasis/TFs und Laufzuordnung notwendig.
> Keine Speicherortfrage wiederholen, keine Ersatzdatenwahl oder Phase 2.
> Frühere Einstiege unten sind Verlauf.

> **Aktueller Einstieg nach Phase-1-Belegvertrag, 02.11.2026:**
> [Spezifikation, Prüfung und Grenzen](kinneret_phase1_spec_2026_11_02/README.md).
> `python tools/check_kinneret_phase1_spec.py` prüft 26 Belege, 19 Befunde und
> 50 Snapshotdateien. Erwarteter Exitcode 2: elf Nachweisbündel offen,
> kein startbereiter historischer Fall. `--with-local` liest nur die bereits
> dokumentierten Original-/Zusatzpfade. 20 gezielte Tests bestanden.
> Als Nächstes neue Originalbelege für finale Basis/Chronologieliste und TFs
> sowie Laufzuordnung an den Vertrag binden. Weiterer Speicherort unbekannt;
> keine wiederholte Frage oder pauschale Suche. Keine Quellvariante automatisch
> wählen, keine Ersatzdaten oder Phase-2-Arbeit voraussetzen. Kein Runner,
> keine neue App-Ansicht, keine Trainings/Sampler/Fossilläufe. Frühere Einstiege
> unten beschreiben den Verlauf.

> **Aktueller Einstieg nach Nachbearbeitungsabgleich, 02.11.2026:**
> [Auswahl, Kombination und Grenzen](kinneret_postprocessing_2026_11_02/README.md).
> Quellvarianten und zwei synthetische Rechenkontrollen sind dokumentiert;
> Originalposterioren/finale Laufbindung bleiben offen. Indexverlust und
> unterschiedliche Dichtemodi nicht still korrigieren oder harmonisieren.
> Als Nächstes die Eingabe-, Referenz- und Nachbearbeitungsverträge zu einer
> maschinenprüfbaren Phase-1-Spezifikation zusammenführen. Keine neuen Läufe.
> Frühere Einstiege folgen als Verlauf.

> **Aktueller Einstieg nach Referenz-/Ankerabgleich, 02.11.2026:**
> [Vertrag, Quellstellen und Grenzen](kinneret_references_2026_11_02/README.md).
> AP-Maske, Soreq-/MEDSTACK-Verbraucher und Ankerzweige sind dokumentiert;
> finale numerische Eingaben/Laufbindung bleiben offen. Aktive Testüberschreibungen
> nicht still entfernen oder ältere Daten substituieren. Nächster unabhängiger
> Einstieg: historische Ziehungsauswahl und Dichtekombination anhand
> `mcmc_case_studies.R`, Dissertation und aktuellem Trimming-Einstieg binden.
> Keine neuen Läufe, keine pauschale Inventur. Ältere Einstiege folgen als Verlauf.

> **Aktueller Einstieg nach Kinneret-Teilbindung, 02.11.2026:**
> [Ergebnis, Crosswalks und Grenzen](kinneret_binding_2026_11_02/README.md).
> 160 Proben und zwölf ältere Taxonspalten exakt gebunden; finale v1.2-Basis
> und numerische TFs fehlen. Der Nutzer kennt keinen weiteren Speicherort.
> Ältere Biom-/Sammelcontainer nicht ersetzen. Als nächsten unabhängigen Schritt
> Kinnerets Referenz-/Ankervertrag anhand bekannter Verbraucher konkretisieren.
> Keine neuen Rechenläufe oder pauschale Inventur. Frühere Einstiege folgen als Verlauf.

> **Aktueller Einstieg nach abgeschlossener Fallmatrix, 02.11.2026:**
> [Sieben Fälle, Quellen, Grenzen und Einstieg](local_case_matrix_2026_11_02/README.md).
> Noch kein vollständig numerisch reproduzierbarer Dissertationsfall.
> Als Nächstes gezielt Kinnerets historischen Basiseintrag und geordnete
> Proben-/Taxon-/TF-Zuordnung an die vorhandenen 160 Pollen-/Chronologieeinträge
> binden. Beide Kinneret-Fallgrafiken sind bytegleich zum Dissertationsarchiv.
> Keine neue Gesamtinventur oder Rechenläufe. Die folgenden älteren Hinweise
> zur noch zu erstellenden Matrix beschreiben den vorherigen Stand.

> **Neue Priorität auf Nutzerauftrag, 02.11.2026:** Zuerst die lokalen
> Rekonstruktionen der Dissertation nachbauen, danach verbessern.
> [Verbindlicher Arbeitsplan](LOCAL_DISS_RECONSTRUCTION_PLAN.md).
> Nächster Auftrag: quellengebundene Fallmatrix, anschließend belegbaren ersten
> Originalfall festlegen. Neue Kalibrierung und zusätzliche Reliabilitätsansicht
> gehören in Phase 2. Die folgenden Berichte bleiben gültig; ihre früheren
> App-Aufträge und Prioritäten sind nachgeordnet.


> Fortschritt 02.11.2026: Die unten als erster Schritt beschriebene
> Forschungsstands-/Quellenansicht ist jetzt umgesetzt und geprüft.
> [Aktuelles App-Ergebnis](app_research_2026_11_02/README.md).
> Auch der gespeicherte Sieben-Arten-Vergleich mit Balanced Accuracy und Brier
> ist umgesetzt: [Ergebnis und nächster Einstieg](app_species_2026_11_02/README.md).
> Die Foldansicht ist ebenfalls umgesetzt und geprüft:
> [Aktuelles Ergebnis und Einstieg](app_folds_2026_11_02/README.md).
> Der damals vorgeschlagene Reliabilitätseinstieg ist mit der neuen
> Priorisierung in Phase 2 verschoben. Zuerst den Dissertationsnachbau verfolgen.
> Die Fortsetzung erfolgt derzeit hier; geprüfte Pakete jeweils in PR #77 sichern.


Stand 02.11.2026. **Aktueller Einstieg** für die Fortsetzung nach den lokalen
Quellen-/Modellprüfungen vom 01.–02.11.2026. Branch `dissertation-reconstruction`,
Basis `7f1826cffddc8893ba6354cd3cca3fa43954b9be` nach
[PR #30](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/30).
[Sicherungs-PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77)
sammelt die bisherigen lokalen Vorarbeiten; erster Sicherungscommit `973fddb`.
Der offene PR ist über seinen Head-Branch vollständig für Claude erreichbar.
Vor Beginn Branch/HEAD und PR-Stand prüfen; nicht ungeprüft nur `main` verwenden.

## Auftrag und Rollen

Ziel ist eine App für nachvollziehbare taxonbasierte Multiproxy-
Klimarekonstruktionen. Der Nutzer hat jetzt Commit, Upload und eine dauerhafte
GitHub-Übergabe beauftragt. Frühere Aussagen „kein Commit/Push“ in den
Ergebnisberichten beschreiben deren damaligen Erstellungsstand. Kein neuer
fachlicher Pilot ist damit freigegeben. Keine Subagenten; kurze deutsche
Fortschrittsmeldungen. Fragen nur bei fehlenden Eingaben oder fachlichen
Entscheidungen, die die vorhandenen Befunde nicht entscheiden.

Die vorhandene App/CLI, Analog-Baseline und sieben experimentellen
Artenmodelle sind die nutzbare Entwicklungsbasis. Claude übernimmt die
App-/CLI-Fortsetzung. Die wissenschaftlichen Grenzen bleiben sichtbar.

## Zuerst gezielt lesen

1. `AGENTS.md` und den neuesten Abschnitt von [PROJECT_STATUS.md](PROJECT_STATUS.md).
2. [Zweiter TF-Plot](HISTORICAL_SECOND_TF_PLOT_RESULT.md), danach nur bei Bedarf
   [Konsolen-/EMD-Bindung](HISTORICAL_CONSOLE_TF_LINK_RESULT.md).
3. [Portabler Forschungsstand](research_snapshot_2026_11_02/README.md),
   insbesondere `summary.json` und `manifest.json`.
4. Für App-Aufbau die bestehende Projekt-README und
   `src/kinneret_recon/app/`; für moderne Modelle den bestehenden
   [Sieben-Arten-Vergleich](smote_nnet_2026_10_30/README.md).
5. Den bisherigen [gemeinsamen Einstieg](HANDOFF.md) für Baseline,
   Chronologie-Gates und die erhaltenen PRs #18–#29.

## Was tatsächlich vorliegt

- Alle 78 verschiedenen Modell-Byteinhalte aus 109 Dateien strukturell gelesen.
  167 caret-Fits: sechs binäre, 161 Mehrklassenfits. Das sind **keine**
  wiederhergestellten 279 unabhängigen Artenfunktionen.
- Der gespeicherte 279er-Ohrid-Verbreitungsstand erklärt alle 271
  Konsolen-Zeilenzahlen und acht Auslassungen über Nordhalbkugel,
  klassenweise 70 % und den synthetischen SMOTE-Zweig plus unknown.
  Die heutige Titanic-Überschreibung verhindert eine genaue Laufzuordnung.
- Beide erhaltenen TF-Sammelplots enthalten je 278 durch tatsächlichen Titel
  und Punktzahl eindeutig gebundene Ausgaben. Nur Ohrid-Position 151,
  `Picea omorika` mit einer Präsenzzelle, fehlt. Juglans heißt dort
  `Juglans sp.`, Position 110; Umbenennung zu `Juglans regia` separat behandeln.
- 961.568 gezeichnete Punktpaare: Y-Koordinaten exakt gleich, X-Abweichung
  höchstens 0,02 PDF-Punkt. Die Konturvektoren unterscheiden sich.
  Keine exakten numerischen Marginalarrays wurden daraus gewonnen.
- EMD-Erzeuger und Haupteinstieg binden dieselben vier TF-Felder und denselben
  Dateipfad. BA-only-Quellfassung und exakte globale Klimadateien bleiben offen.
- Paperfall: lokal berechnet und geprüft; Original/Paket unter gleicher
  Laufzeit sowie Posterior-Wiederholungen exakt gleich. Der plattformübergreifende
  archivierte Bitvergleich bleibt eine dokumentierte Grenze. Der Paperfall ist
  kein vollständiger taxonbasierter Dissertationsdurchlauf.
- Sieben Arten: SMOTE verbessert die geprüfte Balanced Accuracy, verschlechtert
  aber die rohen Brier-Werte. Keine validierten fossilen Wahrscheinlichkeiten.
- 67 betroffene Tests sind vor der Veröffentlichung erneut bestanden.
  Historische Abschlussprüfungen und neue portable Prüfsummen liegen bei.

## Was GitHub enthält und was lokal bleiben muss

Die neuen Werkzeuge, Tests, Berichte, Konfigurationen und 50 ausgewählte
Nachweise/Ergebnisdateien werden mit dieser Übergabe versioniert. Das Paket ist
rund 2,8 MB groß. Absolute lokale Benutzerpfade sind in den Exportkopien durch
klar benannte Platzhalter ersetzt; Original- und Exporthashes stehen im Manifest.
Die große Vektorvergleichsdatei ist verlustfrei als JSON-Gzip gespeichert.

Originale RData-/NetCDF-/PDF-Bestände, Compiler, virtuelle Umgebung sowie
vollständige Posterior-/Prior-/Samplerarchive liegen **nicht** neu auf GitHub.
Die bisherigen lokalen Dateien bleiben erhalten. Die portablen Nachweise und
Paper-Ergebnistabellen reichen für Befundprüfung und die nächste App-Aufgabe.
Ein frischer Cloud-Checkout hat keinen Zugriff auf die Originalordner.

Für eine erneute Rohquellenprüfung benötigt Claude lokalen Dateizugriff oder
gezielt bereitgestellte Eingaben. Nicht nur wegen fehlender lokaler Laufdateien
Trainings/Sampler erneut starten. Frühere `.local/...`-Links in den Berichten
sind historische Originalpfade; das Exportmanifest zeigt verfügbare Kopien.
Die alten Erhaltungsverifier prüfen den damaligen Git-/Dateizustand und sind
nach Übergabe-Commit oder einem frischen Checkout keine portable Testsuite.

## Historischer erster App-Arbeitsschritt (inzwischen abgeschlossen)

Die weitere Reihenfolge bestimmt der [neue Arbeitsplan](LOCAL_DISS_RECONSTRUCTION_PLAN.md).
Der folgende Abschnitt dokumentiert den bereits abgeschlossenen App-Einstieg.

Die App soll den gesicherten Forschungsstand nutzbar machen. Zunächst die
vorhandene Architektur und Tests lesen, dann eine kleine, lesende Ansicht
„Forschungsstand / Quellen“ nach dem vorhandenen App-Muster ergänzen:

- Erhaltene Analog-Baseline, sieben experimentelle Artenmodelle,
  historische grafische TF-Belege und fehlende numerische TFs deutlich
  beschriften. Quellen und gespeicherte Nachweise verlinken.
- Vorhandene Artefakte verwenden. Die neue `summary.json` ist ein
  Forschungsstatus, keine Transferfunktion und keine Vorhersageeingabe.
- Status „vorhanden“, „experimentell“ und „fehlt“ verständlich darstellen.
  Keine Behauptung einer fertigen 279-Taxon-Rekonstruktion und keine neue
  fossile Klimakurve. Keine persönlichen Dateipfade in der Produktansicht.
- Bestehende Läufe, Baseline und CLI-Verhalten bewahren; über das bestehende
  Build-/Datenmuster integrieren, passende App-Prüfungen ausführen.

Damit entsteht ein konkret überprüfbarer App-Fortschritt, ohne fehlende
wissenschaftliche Eingaben vorzutäuschen. Danach den nächsten abgegrenzten
App-/Datenvertrag mit dem Nutzer festlegen, falls eine fachliche Entscheidung
nötig wird.

## Verbindliche Grenzen und spätere Quellenarbeit

Keine neuen Fossilrekonstruktionen ohne fachliche Freigabe. Monticchio-P1 und
Chronologie-Gates bleiben offen: physischer LGM90D→MON16-Anschluss samt
Korrelationsunsicherheit fehlt. 28 Kandidaten, 41 Nichttreffer und 47
Provenienzfragen bleiben erhalten. Keine stillen Quellenkorrekturen,
Taxonharmonisierung, Autorenanfragen oder neuen Trainings-/Samplerläufe.
Plots immer jung links, alt rechts.

Numerische `mixed_partial_arts_emd.rdata` und
`mixed_partial_arts_emd_info.rdata`, globale `tann_test.nc` / `pann_test.nc`
und referenzierte CRU-Eingaben mit Zeitachse fehlen weiter. Zeitindizes 61:90
sind ohne Zeitachse keine bewiesenen Kalenderjahre. Die 95.006 regionalen
Rasterabweichungen und der frühere Rastererzeuger bleiben offen.
PDF-Seitenreihenfolge ist kein Taxonindex; Ohrid-Löschpfad nicht auf Kinneret
übertragen. Keine grafischen Näherungen als ungefragter numerischer TF-Ersatz.

Für spätere Quellenarbeit wäre der nächste neue Test ein koordinatengebundener
Vergleich der belegten Punktwolken mit bereits vorhandenen regionalen
Klimamitteln unter deren Zeitgrundlagen und PDF-Präzision. Keine erneute
Gesamtinventur oder Wiederholung der abgeschlossenen Modellprüfung.
