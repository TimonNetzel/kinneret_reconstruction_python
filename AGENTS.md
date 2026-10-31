# Arbeitsregeln im Netzel-Projekt

## Zuständigkeit und Schutzgrenzen

- Claude betreut CLI/App; Codex Kalibrierung, Quellen und separate Prüfung.
  Analog-Baseline und Claudes Integrationsdateien unverändert lassen.
- Fertige Analog-Baseline, experimentelle Artenmodelle und fossile
  Multiproxy-Rekonstruktion unterscheiden. Keine automatische Variantenübernahme,
  Pollentyp-Zuordnung, Pilotfreigabe, stille Quellenkorrektur oder Autorennachricht.
- Merges nur mit ausdrücklichem Auftrag für die betreffenden PRs. Ein früherer
  Mergeauftrag ist keine dauerhafte Freigabe und keine wissenschaftliche Abnahme.
- Keine neuen Modellentscheidungen anhand äußerer Testergebnisse: insbesondere
  keine Auswahl von Gates, Blockgrößen, Architektur, Varianten oder Arten danach.
  Bekannte Regionen nicht als vollständig ungesehene Bestätigung darstellen.
  Bestätigende Bewertung benötigt vorher festgelegte neue unabhängige Evidenz.
- Verbesserungen und Verschlechterungen aller vorgesehenen Varianten berichten;
  Nichtschätzbarkeit, Ausfälle, Ausschlüsse und ungetestete Fälle vollständig
  ausweisen. Fehlende Bänder niemals als Nullunsicherheit behandeln. Bedingte
  Sensitivitätsbänder sind keine nominell validierten Konfidenzintervalle.
- Wissenschaftliche Berichte, Quellenregister, Parameter und Resultate erhalten.
  Änderungen daran erfordern ein eigenes fachlich begrenztes Arbeitspaket;
  Referenzhashes nicht zur Anpassung an abweichende Ergebnisse ersetzen.

## Standardablauf je Arbeitspaket

1. **Stand prüfen:** lokalen Status einschließlich ungesicherter/unversionierter
   Dateien, Worktrees, verfügbare Branches und aktuelle relevante GitHub-PRs prüfen.
   Remote-Heads und Mergezustand getrennt belegen; fehlenden API-Zugriff nicht als
   Statusnachweis werten. Ungesicherte Arbeit erhalten, nichts blind zurücksetzen.
   Basiscommit bewusst wählen und nennen. Bei gewünschter Isolation eigenes
   Arbeitsverzeichnis/Worktree und eigenen Branch verwenden; in bereits isolierten
   Cloud-Tasks ist ohne Nutzerauftrag kein zusätzlicher Worktree nötig.
2. **Gezielt lesen:** [HANDOFF.md](docs/thesis_research/HANDOFF.md) für Einstieg
   und Leseplan, [PROJECT_STATUS.md](docs/thesis_research/PROJECT_STATUS.md) für
   belegten Stand. Danach nur die für das konkrete Paket notwendigen Berichte,
   Designs und Quellen. Archive nur bei konkreter historischer Frage nachladen.
3. **Begrenzt bearbeiten:** Ziel, erlaubte Änderungen, Eingänge und Abschlusskriterium
   festhalten. Keine stillen fachlichen Erweiterungen. Vor neuen wissenschaftlichen
   Fits Design, Datenrollen und Auswertungsregeln festlegen und sichern.
4. **Angemessen prüfen:** bei Dokumentation Links, Aussagen, Referenzen, Hashabgleich
   und Diffumfang prüfen; bei Code die betroffenen Tests. Große Läufe nur bei
   fachlicher Notwendigkeit und entsprechendem Auftrag. Bestanden, fehlgeschlagen,
   nicht schätzbar, übersprungen und nicht ausgeführt klar unterscheiden.
5. **Sichern und übergeben:** Commit und eigenen PR erstellen, ohne Merge.
   PROJECT_STATUS mit belegtem Ergebnis, Prüfungen, offen/blockiert/abgeschlossen
   und PR-Verweis aktualisieren; HANDOFF mit genauem nächsten Auftrag und Leseplan.
   Große Artefakte nur mit nachgewiesenem dauerhaften Speicherort als archiviert
   bezeichnen; Hash und Reproduktionsbefehl allein sichern keine Dateibytes.

Kurz auf Deutsch zu Beginn und nach Arbeitspaketen berichten: **fertig / aktuell /
als Nächstes**. Gezielte Ausschnitte/Suchen, kompakte Werkzeugausgaben und keine
vollständigen Datentabellen im Chat. Keine unnötigen Wiederholungstests oder
Unteragenten; Delegation nur bei ausdrücklichem Auftrag.
