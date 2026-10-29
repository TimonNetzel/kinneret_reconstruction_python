# Claude: vorhandene Baseline integrieren

Die moderne Shared37-Baseline wurde bereits berechnet und unabhängig geprüft.
Lies diese Datei, `metrics_summary.csv` und bei Schemafragen die README in
diesem Ordner. Die früheren Forschungsberichte müssen für diesen Auftrag
nicht erneut ausgewertet werden. Verwende den aktuellen Stand des Baseline-PRs
bzw. nach dessen Merge `main`.

1. Einen dünnen Projekt-/CLI-Zugang zum vorhandenen
   `tools/thesis_import/evaluate_shared2025.py` und dessen Konfiguration
   ergänzen, entsprechend dem bestehenden Zugang zum Thesis-Importer.
   Rechenregeln und Referenzdateien übernehmen; kein neues Modell entwickeln.
2. Die vorhandenen Metriken für die App nutzbar laden. Falls eine Ansicht
   ergänzt wird: **moderne räumliche Validierung** mit Blockgröße, 100-km-Puffer,
   Probenzahl, Tann-/Pann-Einheiten und Trainingsmittel als Vergleich beschriften.
   `analog_k5` ist das vorab festgelegte Hauptmodell; k=1/15 sind Sensitivitäten.
   Die regionale Diagnose ist eine nachgelagerte Teilansicht desselben Tests.
3. Dünne Integrationsprüfungen, reproduzierbaren Aufruf und eigenen PR liefern.
   Keine großen Roh- oder Vorhersagedateien in den Chat ausgeben. Bei einem
   erneuten Lauf Ausgabepfad, Manifest-Hash und kompakte Metriken berichten.

Bekannte Grenze: globaler k=5-Tann-RMSE etwa 3,8–3,9 °C, Pann etwa
365–373 mm/Jahr; im betrachteten Europa-Fenster ist der Niederschlagsgewinn
kleiner. Das sind moderne Testfehler, keine fossilen Fehlerbänder.
Noch keine Monticchio-Klimakurve oder 150-ka-Rekonstruktion in die App einbauen.
Importer, ursprüngliche Kalibrierung und deren Referenz unverändert lassen.
