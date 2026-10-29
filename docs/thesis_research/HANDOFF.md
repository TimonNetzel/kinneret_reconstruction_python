# Übergabe: Dissertation, Datenrecherche und Europa-App

## Fortschreibung: gemeinsame Vergleichsdaten, 29.10.2026

Dieser Abschnitt ist der aktuelle Einstieg; die ursprüngliche Übergabe darunter bleibt als Ausgangsstand erhalten.

- Importer integriert (PR #14); danach Eignungsaudit (#15) und fossile Taxonentscheidungen (#16). Die noch offenen Folge-PRs bauen aufeinander auf und sind nicht als bereits gemergt vorauszusetzen.
- Neu: [37 gemeinsame Pollengruppen und publizierte moderne Daten von 2025](./modern2025_2026_10_29/README.md) auf Branch `codex/netzel-shared-modern2025`. Dort stehen Quellen, reproduzierbarer Präparer, Taxonregeln, Prüfbericht und der nächste Claude-Auftrag.
- Tatsächlich vorbereitet: 7.634 moderne Proben und 69 chronologisch zugeordnete Monticchio-Proben; 7.078 moderne und alle 69 fossilen Proben erreichen die zusätzliche Schwelle von 100 gemeinsamen Zählungen ohne Pinus. Keine stillschweigende Filterung.
- Der Nenner ist die Summe der 37 gemeinsamen Gruppen einschließlich Pinus. Das ist eine eigene Teilzusammensetzung, keine vollständige biologische Pollensumme und keine Wiederherstellung der verlorenen 279 Transferfunktionen. Die neue publizierte Kohorte wird nicht mit der alten SQLite-Kohorte vermischt.
- Claude kann nun diese separate Vergleichsvariante einlesen und eine moderne Tann-/Pann-Baseline räumlich validieren; Auswahl, Filter und Validierung sind zu dokumentieren. Gleiche Koordinatengruppen zusammenhalten. Erst anhand der Validierung über eine fossile Anwendung entscheiden. Es liegen noch keine Modellgüte oder Klimaergebnisse vor.
- Der 69-Proben-Pilot reicht nur bis rund 20,2 ka; das Europa-Ziel 0–150 ka ist weiterhin nicht durch diesen Piloten abgedeckt. Altersgrenzen und die Art der Tiefenzuordnung bleiben erhalten, vollständige Altersensembles fehlen.


Stand: 29. Oktober 2026. Einstiegspunkt für Claude und einen neuen OpenAI-Chat.

## Ziel und Vereinbarungen

- Dissertation: Timon Netzel, *Quantitative paleoclimate reconstructions in the European region based on multiple proxies*, eingereicht 2022, veröffentlicht 2023: https://hdl.handle.net/20.500.11811/10758
- Ziel: wissenschaftlich überprüfbare Weiterentwicklung als interaktive App für ungefähr 0–150.000 Jahre BP im europäischen Raum. Flächendeckende durchgehende Daten sind nicht vorausgesetzt.
- Der Autor besitzt im Wesentlichen die Dissertation und öffentliche Kinneret-Dateien. Ursprüngliche Taxon-Transferfunktionen, vollständige Standortzuordnungen und Altersensembles müssen recherchiert oder dokumentiert neu erstellt werden.
- OpenAI betreut Datenrecherche, Quellenregister und wissenschaftliche Prüfung; Claude entwickelt Importer, Modell und App.
- Änderungen auf getrennten Branches und als Pull Requests. Vor Beginn den aktuellen Repository-Stand lesen und gemeinsame Dateien nicht gleichzeitig bearbeiten.
- Empfehlung: eigenes Europa-Repository aus einem dokumentierten Kinneret-Stand ableiten. Ein solches neues Repository wurde noch nicht angelegt. Dieser Branch übergibt zunächst die Forschungsergebnisse im bestehenden Repository.
- Nutzer bevorzugt kurze Chatantworten; ausführliche Ergebnisse in Dateien dokumentieren.

## Dateien auf diesem Branch

1. [Plan und Claude-Arbeitsauftrag](./Claude_Plan_Europa_0_150ka.md): priorisierte Beschaffung, Ersatz fehlender Daten, Modellprüfung, Meilensteine und Übergabeauftrag.
2. [Dissertation-Auswertung und Datenverzeichnis](./Netzel_Analyse_Datenverzeichnis.md): tatsächlich beschriebene Daten, Vergleichsreihen, Erweiterungskandidaten und offene Reproduktionslücken.
3. [Download-Nachweise](./download_manifest.json): URLs, Prüfsummen und Größen von 16 zuvor heruntergeladenen Originaldateien. Die angegebenen Dateipfade beziehen sich auf das damalige Datenpaket, nicht auf Dateien in diesem Git-Branch.
4. [SISAL-Kandidaten](./sisal_candidates.json): 201 Profile aus 106 Höhlen, regionale Auswahl und Herkunft. Dies ist ein Erweiterungspool, keine Liste der in der Dissertation tatsächlich verwendeten Profile.

Die fünf Dateien liegen unter `docs/thesis_research/` auf Branch `codex/thesis-data-handoff`. Die GitHub-Verbindung kann nach erneuter Freigabe Branches erstellen und wird für diese dokumentierte Übergabe verwendet. Der frühere Schreibversuch war mit HTTP 403 abgewiesen worden; diese Blockade besteht beim erneuten Branch-Test nicht mehr. Vor weiteren Änderungen den aktuellen Branch-/PR-Stand prüfen.

Die großen Originaldateien und SISAL-Probenexporte sind nicht Bestandteil dieses Dokumentations-Branches. Sie lagen im vorherigen Chat als `Netzel_Datenpaket.zip` vor. Der neue Chat darf nicht voraussetzen, dass temporäre Chatdateien verfügbar bleiben. Die öffentlichen Quellen im Bericht und Manifest erlauben erneute Beschaffung; aktuelle Dateien erneut gegen die gespeicherten Prüfsummen prüfen. Bei Abweichungen neue Version dokumentieren.

## Bereits erledigt

- Vollständige Dissertation im vorherigen Chat heruntergeladen und relevante Methoden-, Daten- und Ergebniskapitel ausgewertet; einzelne Abbildungen visuell geprüft.
- Primärquellen und ausgewählte Downloads in Datenbericht und Manifest dokumentiert.
- SISALv3-Kandidaten aus öffentlich verteilten Tabellen exportiert. Davon überlappen 194 Profile aus 105 Höhlen anhand ihrer Altersspannen das Zielintervall; dies belegt weder Kontinuität noch Unabhängigkeit.
- Öffentliche Kinneret-README und `docs/abgleich_thesis.md`, `docs/plan_multiproxy.md`, `docs/plan_standorte.md` gelesen. Noch kein unabhängiger Code-Audit und keine Modellläufe durch OpenAI.
- Lesezugriff und beim erneuten Test Branch-Schreibzugriff bestätigt. Dieser Branch übergibt Forschungsergebnisse und Arbeitsstand. Am Rechenkern und an Modellparametern wurden für diese Übergabe keine Änderungen vorgenommen.

## Zentrale wissenschaftliche Entscheidungen

- Jahresmitteltemperatur TANN der Dissertation und Wintertemperatur TDJF des Kinneret-Modells unterscheiden; nicht stillschweigend austauschen.
- Verlorene 279 Taxon-Transferfunktionen sind ein zentraler Engpass. Öffentliche Meusel-Arealgeodaten plus moderne Klimatologie sowie EMPD2-Rohzählungen sind Ersatz-/Erweiterungswege, keine nachgewiesenen identischen Originalinputs.
- Pollen-Prozentwerte nicht in erfundene Zählungen umwandeln. Ohrid 892362 enthält öffentliche Taxon-Prozentwerte; Tenaghi 943593 enthält vor allem aggregierte Gruppen.
- Korrelations-/Wavelet-Gewichtung und generative Likelihood getrennt benennen; das neue Modell muss erst spezifiziert und validiert werden.
- Altersensembles, Hiatus und gemeinsame Chronologien berücksichtigen. AP/NAP aus denselben Pollen sowie Komposit und Komponenten nicht als unabhängige Daten mehrfach zählen.
- Ein 50-Jahres-Rechengitter schafft keine 50-jährige Informationsauflösung. MEDSTACK liegt im gefundenen Archiv auf einem 1.000-Jahres-Raster.
- Die existierenden Multiproxy-Pläne setzen teilweise Datenlieferung durch den Autor voraus. Diese Voraussetzung ist durch öffentliche Beschaffung oder Neukalibrierung zu ersetzen.

## Unmittelbar nächste Aufgabe

Zuerst fehlende moderne Kalibrierungsdaten und einen langen Pilotstandort prüfen. Dafür:

1. Aktuellen Branch/PR und vorhandene Projektdokumente lesen; vor Änderungen aktuellen Repository-Stand prüfen. Falls diese Übergabe inzwischen gemergt ist, die Dateien auf `main` verwenden.
2. Quellenregister auf Basis von Bericht und Manifest aufbauen. Status `gefunden`, `heruntergeladen`, `Chronologie verknüpft`, `geprüft`, `modellfähig` getrennt führen.
3. EMPD2 und FloraWeb/Meusel-Geodaten tatsächlich auf Dateiformat, Taxonumfang, Klimabezug und Nutzbarkeit prüfen.
4. Ohrid sowie einen Standort mit echten Rohzählungen und belastbarer Chronologie als Piloten vergleichen. Vollständige 150-ka-Abdeckung nicht allein aus einem Publikationstitel ableiten.
5. Für Claude ein kleines Import-Arbeitspaket mit tatsächlichen URLs, Dateischemata, Taxonmapping, Alterskonventionen und offenen Lücken liefern.

Danach Pollen-Baseline, ein zusätzlicher Proxy, Robustheitsprüfung und erst anschließend regionale Erweiterung. Keine automatische wissenschaftliche Freigabe allein aufgrund plausibler Kurven.

## Starttext für den neuen Chat

> Wir setzen das Netzel-Dissertationsprojekt fort. Lies im Repository https://github.com/TimonNetzel/kinneret_reconstruction_python auf Branch `codex/thesis-data-handoff` zuerst `docs/thesis_research/HANDOFF.md`, dann Plan und Datenverzeichnis in demselben Ordner. Ziel: europäische Multiproxy-App für 0–150 ka. Claude entwickelt Code und App; du recherchierst und prüfst die Daten. Original-Transferfunktionen und Altersensembles fehlen. Prüfe zuerst moderne Kalibrierungsquellen und geeignete Pilotdaten und dokumentiere Quellen und Lücken auf einem eigenen Branch. Antworte im Chat kurz. Falls diese Übergabe inzwischen gemergt ist, verwende die Dateien auf `main`.
