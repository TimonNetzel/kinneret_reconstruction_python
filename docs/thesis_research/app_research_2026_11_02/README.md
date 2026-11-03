# Forschungsstand direkt in der App

> Fortsetzung: Der unten genannte nächste Sieben-Arten-App-Schritt ist
> inzwischen umgesetzt. [Aktuelles Ergebnis und Einstieg](../app_species_2026_11_02/README.md).

Stand 02.11.2026, Fortsetzung auf `dissertation-reconstruction` in
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).
Abgeschlossenes App-Arbeitspaket. Der Nutzer wünscht nach jedem abgeschlossenen,
geprüften Schritt eine sinnvolle GitHub-Sicherung; die Fortsetzung kann hier
oder mit Claude erfolgen.

Die zusätzliche Ansicht **Forschungsstand** stellt den gespeicherten Projektstand
unabhängig vom gewählten Lauf und Standort dar. Fünf Bereiche trennen die
vorhandene moderne Analog-Baseline, experimentelle Artenmodelle, fehlende
historische numerische TFs, den vorläufigen Papervergleich und die offene
Monticchio-Chronologie. Berichte sind direkt mit festgehaltenem Git-Commit
verlinkt. Die Übersicht funktioniert offline; die Quellenlinks benötigen Netz.

Die Quellen- und Modellbefunde sind keine zusätzlichen Vorhersageeingaben.
Es wurde weder ein Modell trainiert noch ein Sampler oder eine neue
Fossilrekonstruktion ausgeführt. Baseline, Kern, Modellparameter und bisherige
Darstellungen behalten ihr Verhalten. Eine Statusansicht ohne Diagramm bietet
keinen irreführenden Abbildungsexport an.

## Datenbindung und Builds

`forschungsstand.py` bindet die publizierte `summary.json` mit SHA256 und
liest nur gespeicherte Paketbelege. Geänderte Snapshot-Bytes brechen den Build
mit einem klaren Fehler ab. Fehlt der Snapshot, zeigt die App einen sichtbaren
Fehlstand. Fehlende Artenmodelle oder Baseline-Metriken werden auch bei
vorhandenem Snapshot nicht als verfügbare Paketbestandteile ausgegeben.
Persönliche lokale Pfade und Originalbestände werden dafür nicht benötigt.

Die Ordner-App erhält `daten/forschung.js`, einschließlich Offline-Cache.
Die Ein-Datei-App bettet dieselben Daten ein. Die Oberfläche verwendet das
vorhandene Ansichtenregister, die bestehende Gestaltung und Deutsch/Englisch.
Der historische Forschungs-Snapshot bleibt unverändert; sein damaliger
`app_next_step` ist mit diesem Arbeitspaket abgeschlossen.

## Prüfung

- 11 App-/Daten-/Buildtests bestanden; zwei automatische Browseraufrufe wegen
  fehlendem Playwright im normalen Node-Suchpfad übersprungen.
- Die Browserprüfungen wurden zusätzlich ausdrücklich mit dem gebündelten
  Playwright und dem bereits installierten Microsoft Edge ausgeführt, ohne
  Browserinstallation und ohne sichtbares Benutzerfenster.
- Beide Offlineformate je auf Mobil/Desktop in Hell/Dunkel: acht erfolgreiche
  vollständige Durchläufe aller bisherigen Ansichten, Diagramme, Exporte,
  Sprachwechsel und neuer Forschungsstandsansicht; beide Gestenprüfungen bestanden.
- Vier abschließende Durchläufe prüfen die präzise Balanced-Accuracy-Aussage,
  fehlende und teilweise vorhandene Nachweise, Sprache und horizontale Überläufe.
  Keine JavaScript-Seitenfehler. Endgültige Mobil- und Desktopbilder visuell geprüft.
- Das bestehende portable Nachweispaket mit allen 50 Exportdateien bleibt gültig.

Die vorherige GitHub-CI scheiterte an der fehlenden Testabhängigkeit `rdata`.
Der eigene Sicherungscommit `954746c` ergänzt dafür die geprüfte Version 1.1.0
im Test-Extra. Dessen [GitHub-Testlauf](https://github.com/TimonNetzel/kinneret_reconstruction_python/actions/runs/41391320977)
ist erfolgreich. Die App-Laufzeitabhängigkeiten bleiben unverändert.

Nachweise: [Prüfzusammenfassung](validation.json),
[Mobilansicht](mobile.png), [Desktopansicht](desktop.png).
Vollständige lokale Browserprotokolle liegen unter `.local/app_research_v1/`.

## Wiederholen und nächster Einstieg

```bash
python tools/verify_research_handoff.py
python -m pytest tests/test_forschungsstand.py tests/test_app.py -q
```

Für einen vollständigen Browserlauf nach dem normalen App-Build und mit
installiertem Playwright:

```bash
node tools/app_pruefen.js results/app/index.html results/kinneret-app.html --kanal msedge
```

`--kanal` ist optional; ohne Angabe gilt weiterhin Playwright-Chromium.
Die Quellenlinks zeigen auf die erhaltenen Forschungsberichte; sie rechnen nichts.

Als nächster abgegrenzter App-Schritt bietet sich die Darstellung der bereits
gespeicherten Sieben-Arten-Metriken an, mit Balanced Accuracy und Brier gemeinsam,
festgehaltenen räumlichen Tests und ausdrücklicher experimenteller Kennzeichnung.
Dafür keine neuen Fits, Kalibrierungen oder fossilen Wahrscheinlichkeiten ableiten.
Originale numerische TFs, globale Klimaeingaben, Pollentyp-Zuordnung und
Monticchio-/Chronologie-Gates bleiben offen. Plots jung links, alt rechts.
