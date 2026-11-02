# Bekannter Datenträgerverlust und Wiederaufbauplan

Stand 02.11.2026. **Quelle: ausdrückliche Nutzerangabe im Chat vom 02.11.2026.**
Diese Angabe ist eine dauerhafte Planungsgrundlage für Codex und Claude.

## Gesicherte Nutzerangabe

Alle Dateien, deren historischer Pfad mit **`/automount/`** beginnt, lagen auf
einer inzwischen gelöschten Festplatte. Die dortigen Originaldateien sind
verloren; benötigte Dateien müssen erneut erstellt werden.

Solche Pfade bedeuten deshalb **bekannter Originalverlust / Wiederaufbau nötig**,
nicht lediglich einen momentan nicht eingehängten Datenträger oder einen noch
zu erfragenden Speicherort. Nicht wieder nach dieser Festplatte oder einem
weiteren Speicherort dieser Originale fragen und keine pauschale Suche danach
wiederholen.

Die Aussage gilt anhand des belegten Pfadpräfixes. Andere fehlende Dateien und
rein relative Pfade nicht ohne Herkunftsbeleg diesem Verlust zuordnen.
Eine bereits konkret gesicherte Kopie oder ein Export kann mit seiner eigenen
Provenienz weiterverwendet werden; seine Identität mit dem verlorenen Original
muss jeweils belegt sein.

## Verbindliche Konsequenz für Phase 1

Für jede tatsächlich benötigte und an `/automount/` gebundene Datei wird die
Aufgabe von „Original noch finden“ zu **„belegten Erzeugungsweg nachvollziehen
und Datei kontrolliert neu erstellen“** geändert. Dies ist keine pauschale Wahl
von Ersatzdaten oder eine methodische Verbesserung in Phase 2.

Der Wiederaufbau wird fall- und dateibezogen vorbereitet:

1. Historischen Pfad, erwartete Rolle, Format, Objektstruktur, Achsen und Einheiten
   anhand vorhandener Verbraucher/Quellenberichte festhalten.
2. Erhaltenen Erzeugercode, benötigte Eingaben, Parameter, Versions-/Paketstand
   und gegebenenfalls RNG-/Seed-Abhängigkeiten konkret zuordnen. Fehlende
   Voraussetzungen als solche ausweisen; keine neue Gesamtinventur.
3. Prüfen, ob sich die Datei mit belegten erhaltenen Eingaben deterministisch
   wiederherstellen lässt. Fachlich nicht ableitbare Daten-/Verfahrenswahl oder
   fehlende Laufzuordnung bleibt eine konkrete offene Entscheidung.
4. Einen abgegrenzten Wiederaufbau samt Konfiguration, Provenienz, Prüfungen
   und Grenzen planen. Bestehende Einschränkungen für Trainings, Sampler,
   Altersmodelle und Fossilrekonstruktionen vor einem tatsächlichen Lauf beachten.
5. Neu erzeugte Dateien ausdrücklich als **neu erzeugt** kennzeichnen und ihre
   Hashes sowie Abweichungen sichern. Sie sind nicht automatisch bytegleiche
   Originale oder ein Nachweis der historischen numerischen Laufgleichheit.

Ein erfolgreich erzeugtes Artefakt schließt ein offenes Nachweisbündel erst,
wenn dessen konkrete Voraussetzungen geprüft und die Reproduktionsspezifikation
nachvollziehbar fortgeschrieben wurden. Der Datenträgerverlust selbst ist keine
wissenschaftliche Freigabe und wählt keine finale Kinneret-Variante.

## Umgang mit bisherigen Nachweisen und Übergaben

Eingefrorene Quellenberichte, Hashmanifeste, Spezifikationen und Startaufträge
bleiben als Nachweise ihres damaligen Stands unverändert. Frühere Formulierungen
wie „Speicherort unbekannt“ werden für belegte `/automount/`-Pfade durch diese
neuere Nutzerangabe ergänzt; sie sind kein Auftrag zu erneuter Suche.
Die bisher offenen elf Nachweisbündel werden durch diese Dokumentationsänderung
nicht automatisch geschlossen.

Aktueller technischer Einstieg bleibt die
[TF-Vorbereitung](CLAUDE_NEXT_STEP_TF_PREPARATION_2026_11_02.md). Bei der weiteren
Abhängigkeitsarbeit die Wiederaufbaupflicht gezielt berücksichtigen. Diese
Festlegung startet noch keine Neuberechnungen oder Trainings.

Die Regel ist in `CLAUDE.md`, `AGENTS.md`,
[Arbeitsplan](LOCAL_DISS_RECONSTRUCTION_PLAN.md),
[Projektstatus](PROJECT_STATUS.md) und
[Claude-Übergabe](CLAUDE_HANDOFF_2026_11_02.md) verlinkt, damit sie beim nächsten
Sitzungsstart erneut gelesen wird. Maßgeblich bleibt zuerst Phase 1, danach
Phase 2; Originale nur lesen, fallbezogene Gates erhalten, keine automatischen
Merges. Plots jung links, alt rechts.
