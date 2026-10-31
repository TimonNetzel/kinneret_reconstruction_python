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

## Abschluss, Folgeauftrag und Nutzer-OK

Auf Nutzerwunsch vom 31.10.2026 wird jedes begrenzte Arbeitspaket so übergeben,
dass zur Fortsetzung ein kurzes „OK“ genügt:

- **Ergebnis im Repository sichern:** PROJECT_STATUS nennt erledigte Teile,
  Prüfungen, offene/blockierte Teile sowie Commit/PR. Ein gesichertes Teilergebnis
  darf nicht als fachlich abgeschlossene Gesamtaufgabe erscheinen.
- **Einen konkreten Folgeauftrag vorbereiten:** HANDOFF führt genau ein nächstes
  empfohlenes Paket mit Status „vorbereitet – wartet auf Nutzer-OK“, Ziel,
  erlaubtem Umfang, gesperrten Bereichen, nötigen Eingängen/Leseplan und
  Abschlusskriterien. Weitere offene Aufgaben separat als blockiert oder später
  ausweisen. Den Folgeauftrag vor dem OK bereits konkret und prüfbar beschreiben.
- **OK eindeutig anwenden:** Ein „OK“, „weiter“ oder sinngleiches Signal im
  passenden Gesprächskontext autorisiert dieses vorbereitete Paket. Nach kurzer
  Standprüfung ausführen, ohne denselben Auftrag erneut bestätigen zu lassen.
  Bereits autorisierte Arbeit vollständig erledigen; für das nächste neue Paket
  wieder einen klaren Folgeauftrag vorbereiten. Ein OK erteilt keine pauschale
  Merge-, Autorenkontakt- oder wissenschaftliche Freigabe.
- **Chatwechsel gezielt empfehlen:** Einen kopierfertigen Übergabeprompt ausgeben,
  wenn ein gesicherter Paketabschluss, ein deutlicher Themenwechsel oder viel
  angesammelter Kontext einen neuen Chat sinnvoll macht. Prompt auch in HANDOFF
  sichern: Branch, letzter verlässlich gesicherter Commit/PR, genaue Aufgabe,
  Leseplan, Schutzregeln und bekannte Blocker. Bei noch ausstehendem OK den
  vorgeschlagenen Startauftrag ausdrücklich als solchen kennzeichnen.
- **Kurz berichten:** Ergebnis, Sicherung und nächsten Auftrag nennen; bei
  sinnvollem Chatwechsel den Prompt direkt mitliefern. Keinen neuen Chat selbst
  anlegen und keine Erinnerungs-/Hintergrundautomatisierung daraus ableiten.

Kurz auf Deutsch zu Beginn und nach Arbeitspaketen berichten: **fertig / aktuell /
als Nächstes**. Gezielte Ausschnitte/Suchen, kompakte Werkzeugausgaben und keine
vollständigen Datentabellen im Chat. Keine unnötigen Wiederholungstests oder
Unteragenten; Delegation nur bei ausdrücklichem Auftrag.
