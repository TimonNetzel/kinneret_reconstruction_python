Setze das Netzel-Projekt in https://github.com/TimonNetzel/kinneret_reconstruction_python fort.

Verwende das vorbereitete Paket unter `tools/thesis_import/` aus dem Importpaket-PR, einschließlich seiner Vorgänger PR #11 und #12. Prüfe den aktuellen Merge-Stand und arbeite auf einem eigenen Branch. Lies zunächst nur `tools/thesis_import/README.md` und diese kurze Aufgabe.

Der Importer ist fertig. Führe seine elf Tests und den vollständigen Import mit `--reference tools/thesis_import/reference_run.json` aus. Verwende ein neues Ausgabeverzeichnis. Referenzabweichungen untersuchen; den Referenzstand nicht automatisch aktualisieren.

Deine Aufgabe: Binde den Importer über einen passenden Projektzugang oder CLI-Wrapper ein, ergänze einen Integrationstest und dokumentiere den Aufruf. Übernimm die Importlogik zunächst unverändert. Vollständige Prüf-JSONs und Forschungsberichte nur bei konkreten Fragen lesen.

Qualitätsflags, Text-NULL-Behandlung, gebrochene Zählungen und Chronologie-Ausschlüsse erhalten. Keine Taxonharmonisierung, kein Training, keine Modellfreigabe und keine Änderung am Kinneret-Datenlock.

Stelle die Integration als eigenen PR bereit. Antworte im Chat kurz; ausführliche Befunde gehören in Dateien.
