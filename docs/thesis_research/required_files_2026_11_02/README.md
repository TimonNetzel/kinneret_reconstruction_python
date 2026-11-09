# Quellengebundener Abgleich notwendiger und fehlender Dateien

Stand 02.11.2026, Basis `26d06cfc3530eef5c65f53e270812894b5d84838`, PR #77.
Hauptergebnis: [priorisierte Dateiliste](../MISSING_PROJECT_FILES_2026_11_02.md).
Aktuelle Fortsetzung: [Claude-Übergabe mit climate_modelling](../CLAUDE_RESUME_CLIMATE_MODELLING_2026_11_02.md).

**Neue Nutzerinformation nach diesem Abgleich:** In der nächsten Sitzung wird
ein zusätzlicher Ordner `climate_modelling` bereitgestellt. Er wurde hier noch
nicht geprüft. Sämtliche Fehlbefunde sind ein Snapshot des bisher zugänglichen
Bestands, keine Aussage über den neuen Ordner oder den gesamten PC.

## Tatsächlich ausgeführter Umfang

369 R-Dateien aus den beiden privat konfigurierten Quellordnern lesend geprüft.
184 `src`-Dateien besitzen mtime nach 2021 (35 Dissertations-, 149 Zusatzdateien).
Dateizeiten sind Priorisierung, kein Nachweis historischer Erstellung oder
Ausführung; heutige Erstellungszeiten können Kopierzeiten sein.

Metadatenindex der beiden Quellordner und des vorhandenen Checkout außerhalb
privater/Laufzeitverzeichnisse: 584 Datendateien, davon 257 RData/RDS/NC;
zusätzlich 18 ZIP-Inhaltsverzeichnisse. Keine Rohdatenfelder dekodiert,
kein R-Code, Training, Sampler oder Rekonstruktionslauf ausgeführt.
Der Chat-Artefaktdienst meldete keine separaten Upload-Artefakte.

Der Scanner erfasst 1178 Leseverweise: 264 am angenommenen Projektpfad vorhanden,
142 mit lokalen Namenskandidaten, 615 ohne Treffer, 157 dynamische Familien ohne
Kandidaten; dazu 13 nicht zu Datenpfaden aufgelöste Leseausdrücke. Das sind
Aufrufstellen, keine Anzahlen notwendiger Dateien. Branches, Funktionsaufrufe,
setwd und Paket-internes Laden sind nicht vollständig ausgewertet.

Die separat geprüfte Kuration nennt 39 Anforderungen (16 direkter lokaler
R-Einstieg, 16 bedingt, sieben Erzeuger-/Vergleichsartefakte) und acht frühere
Kinneret-Posteriorvarianten. P0 bezeichnet explizite Ladelücken im unveränderten
R-Einstieg, keine universelle fachliche Pflicht jedes möglichen App-Pfads.

## Dateien und Wiederholung

- `collect_evidence.py`: Metadaten-/Quellscanner; Originale ausschließlich lesen.
- `curation.json` und `build_priority_list.py`: überprüfte Verbraucherstellen
  auf die aktuelle Suchausgabe abbilden; keine automatische finale Variantenwahl.
- `priority_requirements.csv`: benutzbare priorisierte Liste mit Bedingungen.
- `dependency_calls.csv`, `missing_references.csv`, `unresolved_reads.csv`:
  vollständiges statisches Suchregister innerhalb des angegebenen Umfangs.
- `producer_calls.csv`, `source_edges.csv`, `source_dates.csv`: Erzeugerhinweise,
  statischer Source-Graph, Zeitstempel und Quellenhashes.
- `preserved_copies.json`: zwei kleine Verbreitungscontainer gegen frühere
  strukturelle Audit-Hashes bestätigt, ohne Identität zum verlorenen Lauf zu behaupten.
- `posterior_variants_recheck.json`: acht benannte Ergebnisvarianten ohne
  Namenskopie im bisherigen Index; keine Auswahl einer finalen Variante.
- `summary.json` und `verification.json`: Umfang, Grenzen und Sicherungsprüfung.

```powershell
.venv/Scripts/python.exe docs/thesis_research/required_files_2026_11_02/collect_evidence.py
.venv/Scripts/python.exe docs/thesis_research/required_files_2026_11_02/build_priority_list.py
.venv/Scripts/python.exe -m pytest tests/test_required_files_audit.py -q -o cache_dir=.local/required_files_2026_11_02/pytest_cache
```

Acht synthetische Parserprüfungen bestanden: Kommentare/Strings, verschachtelte
und mehrzeilige Argumente, Namespace-/Dateiparameter, Schreib-/Leseunterscheidung,
Pfadpräfixe, dynamische Variablen, unbeweisbare Bedingungen und Escape-Zeichen.
Die bisherigen 200 historischen Teilprüfungen wurden in diesem Datei-Audit nicht
neu gerechnet; ihre eingefrorenen Nachweise bleiben erhalten.

Die Skripte benötigen die bestehende private `.local/my_reconst_paths.json`.
Der vollständige Metadatenindex bleibt in `.local/required_files_2026_11_02/`.
Absolute Benutzerpfade sind in den Exporten durch Platzhalter ersetzt.
Der neue climate_modelling-Ordner ist darin noch nicht enthalten; erst nach
seiner tatsächlichen Bereitstellung mit explizitem Pfad ergänzen, keinen
Speicherort erfinden oder einen unveränderten Altindex als neuen Scan ausgeben.
