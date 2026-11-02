# Quellenverlust, mögliche lokale Kopien und Wiederaufbauplan

Stand 02.11.2026. Quelle: Nutzerangaben im Chat vom 02.11.2026 einschließlich
der späteren Präzisierungen zu lokalen Kopien und `climate_modelling`.
Dauerhafte Planungsgrundlage für Codex und Claude.

## Verbindliche Quellenlage

Der historische Datenträger hinter **`/automount/...`** wurde gelöscht. Die dort
referenzierten Originalpfade sind nicht mehr verfügbar. **Dateien können aber
zuvor auf den lokalen PC kopiert worden sein** und in den bereitgestellten
lokalen Ordnern/Dateien noch existieren. Der Pfadpräfix allein beweist deshalb
nicht, dass der benötigte Dateiinhalt überall fehlt oder neu erzeugt werden muss.

Zuerst konkrete lokale Kopien und deren Datenvertrag prüfen. Nur verbleibende
notwendige Lücken anhand belegter Erzeugungsschritte kontrolliert neu erstellen.
Nicht erneut nach dem gelöschten Datenträger oder dem schon als unbekannt
beantworteten Speicherort/Erzeuger der finalen Basisdatei fragen.
Andere Fehlpfade nicht ohne Herkunftsbeleg diesem Verlust zuordnen.

**Neueste Nutzerinformation:** In der nächsten Sitzung wird ein weiterer Ordner
`climate_modelling` verfügbar. Er wurde hier noch nicht untersucht. Priorität
hat nun der gezielte Abgleich mit `main.R`/`main_v1.2.R` und der
[Dateiliste](MISSING_PROJECT_FILES_2026_11_02.md).
[Claude setzt hier fort](CLAUDE_RESUME_CLIMATE_MODELLING_2026_11_02.md).
Der genaue neue Pfad ist erst bei Bereitstellung zu binden; keine Dateifunde
oder geschlossenen Voraussetzungen vorwegnehmen.

## Vorgehen je tatsächlich benötigter Datei

1. Verbraucher, historischen Pfad, Rolle, Format, Objektstruktur, Achsen,
   Einheiten und relevante Quellfassung festhalten. `src` mit erhaltenem
   Änderungsdatum nach 2021 priorisieren; Kopierzeit nicht als Erstellung ausgeben.
2. In den bereitgestellten Quellen passende Kopien/Archivmitglieder gezielt
   abgleichen. Namensgleichheit ist nur ein Kandidat; Provenienz, Hash,
   Struktur und Variante/Laufbezug prüfen. Unterschiedliche Standortdateien
   oder regionale/globale Klimaversionen nicht still gleichsetzen.
3. Nur nach verbleibender Lücke Erzeugercode, benötigte Eingaben, Parameter,
   Versions-/Paketstand und gegebenenfalls RNG/Seed zuordnen. Bereits vorhandene
   Daten nicht unnötig neu erstellen. Keine pauschale PC-Suche.
4. Einen abgegrenzten Wiederaufbau mit Prüfmaßstab und Grenzen planen.
   Fachlich nicht ableitbare Daten-/Verfahrenswahl bleibt eine konkrete offene
   Entscheidung; bestehende Laufgrenzen beachten.
5. Neu erzeugte Artefakte als **neu erzeugt** mit Hash/Provenienz kennzeichnen.
   Keine automatische Identität mit verlorenen Originalen, historischen
   Posterioren oder R-Laufzeitgleichheit behaupten.

Die [aktuelle Dateiliste](MISSING_PROJECT_FILES_2026_11_02.md) enthält einen
gezielt geprüften bisherigen Bestand. Ihr Abgleich ist vor Bereitstellung von
`climate_modelling` entstanden und muss um diesen neuen Bestand ergänzt werden.
Ein konkreter positiver Gegenbeleg zum pauschalen Verlust liegt schon vor:
Die unter Automount referenzierte Ohrid-Verbreitungsbasis ist lokal erhalten.

## Erhaltung und fachliche Grenzen

Eingefrorene Quellenberichte, Hashmanifeste, Spezifikationen und frühere
Startaufträge bleiben unverändert. Neue Nutzerangaben ergänzen ihren damaligen
Stand. Ein Fund schließt ein Nachweisbündel erst nach Prüfung der konkreten
Voraussetzungen und nachvollziehbarer Aktualisierung der Spezifikation.
Alle elf bisherigen Nachweisbündel bleiben vorerst offen; keine finale
Kinneret-Variante gewählt. Die neue Quelleninformation startet keine Trainings,
Sampler, Altersmodelle oder Fossilrekonstruktionen.

Diese Regel ist in `CLAUDE.md`, `AGENTS.md`,
[Arbeitsplan](LOCAL_DISS_RECONSTRUCTION_PLAN.md),
[Projektstatus](PROJECT_STATUS.md) und
[Claude-Übergabe](CLAUDE_HANDOFF_2026_11_02.md) verlinkt.
Phase 1 vor Phase 2; Originale nur lesen, App/CLI/Baseline und fallbezogene
Gates erhalten, keine automatischen Merges, Plots jung links und alt rechts.
