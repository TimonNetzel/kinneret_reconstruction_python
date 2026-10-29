# Claude: moderne Shared37-Baseline

Arbeite auf dem aktuellen Stand von PR #17 bzw. nach dessen Merge auf main.
Lies zuerst diese Datei und `docs/thesis_research/modern2025_2026_10_29/shared_features.json`.
Nutze die README desselben Ordners nur für Quellenbeschaffung und Detailfragen;
die früheren Forschungsberichte müssen für diesen Auftrag nicht neu ausgewertet werden.

## Vorhandene Grundlage

`tools/thesis_import/prepare_shared2025.py` erzeugt moderne/fossile Zähl- und
Prozenttabellen. Aufruf und Download-Hashes stehen in der genannten README.
Erwartet: 7.634 moderne Proben, 37 Merkmale, davon 7.078 mit
`count_threshold_pass=1`; 69 fossile Proben. Zuerst moderne Daten verwenden.
Zählungen nicht aus Prozentwerten zurückrechnen und die alte SQLite-Kohorte
nicht hinzufügen.

## Begrenzter Implementierungsauftrag

1. Separate Vergleichsvariante einlesen. Die 37 Merkmalsnamen/-reihenfolge aus
   der JSON-Richtlinie übernehmen. Für Prozenttabellen gilt der Nenner
   `shared_count_sum` einschließlich Pinus; Metadaten mit `count` bleiben Zählungen.
2. Den vorhandenen Schwellenflag anwenden und ausgeschlossene Proben zählen.
   Zunächst den bereitgestellten modernen Gesamtraum verwenden. Zusätzliche
   Filter müssen ausdrücklich konfiguriert und bilanziert sein.
3. Eine einfache dokumentierte Tann-/Pann-Baseline räumlich blockiert validieren.
   `coordinate_group_1e6deg` muss vollständig in einem Fold bleiben.
   Gruppierung gleicher Koordinaten allein genügt nicht: räumliche Blöcke,
   Blockgröße, Fold-Zuordnung und Zufallsseed in den Ergebnissen speichern.
   Skalierung, Auswahl und etwaige Parameteranpassung nur in Trainingsdaten.
4. Für beide Zielgrößen Vorhersagen je zurückgehaltener Probe sowie RMSE,
   MAE und Bias ausgeben. Als Vergleich jeweils den Mittelwert des
   Trainingsfolds verwenden. Einheiten: Tann °C, Pann mm/Jahr.
5. Reproduzierbaren Aufruf, Konfiguration, Daten-/Policy-Hashes und Prüfungen
   zu Merkmalen, Probenverlusten und räumlicher Fold-Trennung mitliefern.
   Eigenen PR erstellen; Ergebnis und verbleibende Grenzen knapp dokumentieren.

Dieser Meilenstein endet mit moderner Validierung. Fossile Rekonstruktion und
App-Anbindung folgen anhand dieser Ergebnisse. Bestehende Kinneret-Kalibrierung,
Importer und Referenz nicht ersetzen. Keine neuen Klimadaten, keine
Wiederherstellung der verlorenen 279 Transferfunktionen behaupten.
