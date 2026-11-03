# Claude: hier nach dem Wechsel fortsetzen

Stand 02.11.2026. Dieser Einstieg hat Vorrang vor älteren „nächster Schritt“-
Angaben. Nutzer möchte jetzt zeitnah zu Claude wechseln; Arbeit und die neue
Information zum Ordner `climate_modelling` sind deshalb im Repository gesichert.
Deutsch, kurze Fortschrittsmeldungen, keine Subagenten.

## Git-/Arbeitsbasis

Repository `TimonNetzel/kinneret_reconstruction_python`, Branch
`dissertation-reconstruction`, [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
offen und nicht gemergt. Basis der aktuellen Dateiprüfung ist `26d06cf`.
Den Übergabecommit zeigt `git log -1 -- docs/thesis_research/CLAUDE_RESUME_CLIMATE_MODELLING_2026_11_02.md`.
Vor Beginn tatsächlichen Git-/PR-Stand und lokale Änderungen prüfen und erhalten.
Nur den bereits vom Nutzer benannten vorhandenen Checkout verwenden; keine
neuen Klone/Worktrees oder automatischen Merges.

## Neueste Nutzerinformation – noch nicht untersucht

Der Nutzer hat weitere Dateien gefunden. In der **nächsten Sitzung** sind sie
in einem neuen Ordner **`climate_modelling`** verfügbar. Ein absoluter Pfad,
Dateiliste oder Dateninhalt liegt hier noch nicht vor. Der Ordner wurde in
dieser Sitzung nicht durchsucht, geladen oder an die Konfiguration gebunden.
Keinen bereits abgeschlossenen Abgleich dieses neuen Bestands behaupten.

Der Nutzer vermutet einen Bezug zu `main.R`. Belegte Einstiegspunkte:

- `original:main.R:128` lädt
  `../../climate_modelling/pollen/transfer_functions/input_data/data_out/taxa_distribution_ww.rdata`.
- `original:main_v1.2.R:858` verweist auf den älteren Sammelstand
  `.../climate_modelling/pollen_speleothems/mcmc_recon/Data/pollen_g_climate_all_giesecke_v4.0.rdata`.
- Weitere Erzeuger-/Verbraucherpfade sind im neuen Suchregister mit Quelle und
  Zeile erfasst. Ein aktueller `load`-Text ist kein Nachweis einer finalen
  historischen Ausführung; spätere Test-/Biomvarianten getrennt halten.

**Erste Aufgabe für Claude:** Den tatsächlich bereitgestellten Ordner lokalisieren
und ausschließlich lesend, bedarfsgebunden gegen die Dateiliste prüfen. Nutze
seinen in der nächsten Sitzung zugänglichen Pfad; keine pauschale PC-Suche.
Nur wenn der neue Ordner trotz Bereitstellung nicht zugänglich/identifizierbar
ist, konkret nach dessen neuem Pfad fragen. Nicht nochmals nach dem unbekannten
Erzeuger/Speicherort der schon diskutierten finalen Basisdatei fragen.

## Dauerhafte Quellenregel mit Präzisierung

Historische `/automount/...`-Pfade lagen auf einer gelöschten Festplatte.
**Lokale Kopien können trotzdem vorhanden sein**, auch in bereitgestellten
lokalen Dateien und jetzt möglicherweise `climate_modelling`.
Daher nicht alle solchen Dateien sofort neu erzeugen: erst Pfad-/Namenskandidaten,
Hashes, Objektstruktur, Achsen, Einheiten und Version/Laufbezug prüfen. Nur die
verbleibenden notwendigen Lücken zur kontrollierten Neuerstellung einplanen.
Eine neu erzeugte Datei ist kein automatisch identisches Original.
Maßgeblich: [Quellen-/Wiederaufbauplan](SOURCE_RECOVERY_PLAN.md).

## Was im Chat abgeschlossen und gespeichert wurde

| Commit | Gesichertes Paket |
| --- | --- |
| `f7d97c1` | Deterministische historische Taxongewichtung/TF-Produkt/PDF-Summe, 22 synthetische Tests |
| `6241f41` | Alters-/Tiefenteil, Listen-/Matrixformen, Nullersetzungen und Altersduplikate, 43 Tests |
| `d6e3343` | Exakte Altersrastermitgliedschaft und vorläufige/wirksame Zeilenauswahl, 39 Tests |
| `674bf13` | Kumulierte Rohdichte-/EMD-Quellgröße, Nullreferenz und geerbtes/späteres dims, 43 Tests |
| `91436c4` | Gemeinsame Kunstrechnung mit unabhängigen exakten Brüchen, 33 weitere Tests; mit Belegvertrag insgesamt 200 grün |
| `26d06cf` | Nutzerangabe zum gelöschten Automount-Datenträger und Wiederaufbauplanung |
| Dieser Übergabecommit | Quellengebundene Fehlenden-/Kopienliste, Parserprüfungen und neue climate_modelling-Information |

Vorarbeiten: quellengebundene Fallmatrix für Totes Meer Holozän/Glazial getrennt,
Kinneret, Birkat Ram, Van, Iznik und Ohrid; teilweise ältere Kinneret-Eingabe-,
Referenz- und Nachbearbeitungsbindung. Maschinenprüfbarer Phase-1-Belegvertrag:
26 Belegdateien, 19 Befunde, 50 Forschungsnachweise; **elf Nachweisbündel offen**,
keine finale Laufvariante gewählt. Noch kein vollständiger taxonbasierter
Dissertationsfall reproduziert und keine historische R-Laufzeitgleichheit bewiesen.

App/CLI, Analog-Baseline, Paperreferenz, sieben experimentelle Artenmodelle,
Originale und fallbezogene Monticchio-/Chronologie-Gates bleiben erhalten.
Frühere Startaufträge und Hashpakete sind eingefroren und werden nicht geändert.
Die noch nicht bearbeitete TF-Vorbereitung ist unter
[früherem konkretem Arbeitsauftrag](CLAUDE_NEXT_STEP_TF_PREPARATION_2026_11_02.md)
beschrieben; **zuerst den neuen Bestand abgleichen**, danach den nächsten
belegten technischen Teil planen.

## Ergebnis des begonnenen Dateiauftrags

Zuerst lesen: [Benötigte und noch fehlende Projektdateien](MISSING_PROJECT_FILES_2026_11_02.md),
[Methodik/Suchumfang](required_files_2026_11_02/README.md) und
[priorisierte CSV](required_files_2026_11_02/priority_requirements.csv).

Bisheriger, tatsächlich geprüfter Bestand:

- 369 R-Dateien; 35 Dissertations- und 149 Zusatz-`src`-Dateien mit mtime nach
  2021. Historische Erstellung ist damit nicht bewiesen; heutige Erstellungszeiten
  können Kopierzeiten sein. Source-Dateien nach 2021 besonders priorisieren.
- 584 Datendateien, davon 257 RData/RDS/NC, sowie 18 ZIP-Inhaltsverzeichnisse in
  den beiden Quellordnern und dem vorhandenen Checkout. Keine gesamte PC-Suche.
- 39 kuratierte Anforderungen: 16 direkte Ladelücken (zwölf RData/NC und vier
  TXT/CSV), 16 bedingte, sieben Erzeuger-/Vergleichsartefakte. Zusätzlich acht
  historische Posteriorvarianten ohne Namenskopie. Nicht alle sind universelle
  Eingaben; Bedingungen und erzeugte Ergebnisse sind getrennt.
- Ein positiver Fund ist bereits gesichert: Automount-pfadpassende
  `my_ohrid/data/in/climate_data/taxa_distribution_ww.rdata` existiert lokal.
  Diese und die Kinneret-Fassung stimmen mit früheren Audit-Hashes überein.
  Keine automatische Gleichsetzung der beiden Fassungen.
- Unter den weiterhin fehlenden benannten Dateien: finale v1.2-Basis,
  `mixed_partial_arts_emd*.rdata`, globale CRU-/Ankerdateien, Isotopen-Workspaces
  und `pmip3_21k_v0.nc`. **Dies sind Vorbefunde vor climate_modelling**, keine
  endgültige Behauptung über dessen Inhalt.

Die Rohlisten enthalten statische Aufrufstellen, auch bedingte/alte Varianten.
Sie sind kein vollständiger R-Interpreter oder historischer Abhängigkeitsgraph.
Namensgleichheit allein beweist keine Datei-/Objektidentität.
Acht neue synthetische Parserprüfungen bestanden; die früheren 200 numerischen
Teiltests wurden im Dateiauftrag nicht neu gerechnet. Keine Originaldatenfelder
oder R-Skripte ausgeführt, keine Trainings/Sampler/Neurekonstruktionen gestartet.

## Konkreter Abgleich im neuen Ordner

1. `climate_modelling` als tatsächlich vorhandene neue Quelle in der privaten
   Konfiguration/Arbeitsnotiz mit korrektem Pfad aufnehmen. Der vorhandene
   Scanner hat zwei Quellwurzeln fest im Code; ein neuer Konfigurationseintrag
   allein prüft den Ordner noch nicht. Den neuen Abgleich ausdrücklich erweitern
   und seinen Suchumfang dokumentieren. Originale nur lesen;
   keine großen Container pauschal vollständig laden oder kopieren.
2. Zuerst Namen/Pfadenden der priorisierten Anforderungen und `main.R`-Bezüge
   abgleichen; Kandidaten auch in Unterordnern und vorhandenen Archivverzeichnissen
   gezielt berücksichtigen. Keine erneute globale Altinventur.
3. Gefundene Kandidaten anhand Hash/Metadaten und vorhandener Strukturleser
   gezielt prüfen. Verbreitungsbasis, trainierte Modelle, numerische TFs,
   Referenzworkspaces und erzeugte Ausgaben voneinander unterscheiden.
4. Neues Belegpaket mit Suchumfang, Kopien/Versionskonflikten und verbleibenden
   Lücken anlegen. Die bisherige Liste ist ein datierter Snapshot; nicht still
   alte Hashnachweise oder die elf offenen Voraussetzungen umschreiben.
5. Prüfen, ob konkrete Nachweisbündel wirklich geschlossen werden können.
   Nur anschließend Wiederaufbauplan/aktuelle Dateiliste und nächsten Auftrag
   fortschreiben; kein ungeprüfter Start einer historischen Fallrechnung.

## Prüfkommandos und Grenzen

```powershell
.venv/Scripts/python.exe -m pytest tests/test_required_files_audit.py -q -o cache_dir=.local/required_files_2026_11_02/pytest_cache
.venv/Scripts/python.exe tools/check_kinneret_phase1_spec.py
```

Acht Parserfälle bestanden. Der Belegprüfer erwartet Exitcode 2 bei intakten
Belegen und offenen Voraussetzungen. Lokale Prüfungen benötigen die private
`.local/my_reconst_paths.json`; keine privaten absoluten Pfade oder Original-
container auf GitHub exportieren. Kein automatischer Merge.

Zuerst Phase 1, danach methodische Verbesserungen. Keine neue Kalibrierung,
Trainings, Sampler, Altersmodelle oder Fossilrekonstruktionen aufgrund eines
bloßen Dateifunds starten. Ältere Biomcontainer, der Paper-Biomfall und moderne
Artenmodelle ersetzen keine historischen taxonbasierten Eingaben.
Plots jung links, alt rechts; fallbezogene Gates bewahren.
