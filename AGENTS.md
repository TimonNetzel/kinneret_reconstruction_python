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

## Eigenständige Internetrecherche als Standard

Auf Nutzerwunsch vom 01.11.2026 gilt für dieses Repository:

- Bei fachlichen Aufgaben fehlende öffentlich zugängliche Quellen,
  Herkunfts-, Versions-, Methoden- oder Erzeugungsnachweise standardmäßig
  selbst im Internet suchen und die relevanten Dokumentstellen lesen.
  Notwendige Recherche innerhalb des beauftragten Pakets ohne zusätzliches
  Recherche-OK durchführen. Vom Nutzer weder einen Dokumenttitel noch
  einen Pfad/Ref als Voraussetzung verlangen, wenn Codex die passenden
  öffentlichen Belege selbst ermitteln kann.
- Zuerst den gezielten Repository-Einstieg lesen, dann möglichst Original-
  und Herausgeberquellen verwenden. URL/DOI, Dokumentstand/Version, genaue
  Fundstelle, Aussage, Archiv-/Objektbezug und Beleggrenze dokumentieren.
  Suchtreffer und allgemeine Produkttexte sind keine archivgenauen Nachweise;
  nicht gelesene oder unzugängliche Inhalte nicht als geprüft ausgeben.
- Suche auf die konkrete Frage und Abnahme begrenzen. Bei fehlender Evidenz
  die tatsächlich geprüften Stellen und verbleibende Lücke nennen; keine
  Endlossuche oder Wiederholung derselben Angaben als neue Evidenz.
  Nur gezielt beim Nutzer nachfragen, wenn ein privates/nicht zugängliches
  Objekt, eine nicht ermittelbare Identität oder eine echte fachliche
  Entscheidung nötig ist. Einen technischen Zugriffsfehler konkret benennen.
- Historische paketbezogene Recherche-/Datenprüfverbote gelten für das damalige
  Paket, nicht pauschal für neue Aufträge. Auf Nutzerwunsch vom 01.11.2026
  gehört die nötige, eng begrenzte **schreibgeschützte Prüfung vorhandener oder
  öffentlich zugänglicher Originaldaten und Metadaten** zu einer beauftragten
  Quellen-/Provenienzprüfung. Dazu passende Originalquellen selbst beschaffen
  und lesen; kein zusätzliches OK für bloße Metadatenzeilenprüfung verlangen.
  Objekte, Felder/Proben und Zweck vorab begrenzen, Herkunft/Version/Hash und
  Abfragen nachvollziehbar sichern; Originale unverändert erhalten.
- Aktuelle ausdrückliche Nutzereinschränkungen bleiben maßgeblich. Alte
  Berichte/Verträge nicht rückwirkend umschreiben; HANDOFF/Status/Folgeprompts
  müssen den aktuellen Auftrag wiedergeben und überholte Paketverbote als
  Historie behandeln. Einen autorisierten Evidenzschritt nicht durch ein
  zusätzliches reines Stop-/Planungspaket ersetzen. Kein Nutzerdatei-Blocker
  für selbst ermittelbare öffentliche Quellen. Der Standard startet kein
  weiteres Arbeitspaket automatisch.
- Die Regel steuert das Arbeitsverhalten; sie schaltet keine Webwerkzeuge,
  Netzfreigaben oder App-Berechtigungen frei. Daten-/Modellwahl, Reparaturen,
  Implementierung, wissenschaftliche Rechnungen, Freigaben, Nachrichten und
  Merges benötigen weiterhin den jeweiligen Auftrag. Eine Leseprüfung ist
  keine Trainings-/Gatefreigabe und rechtfertigt keine Datenergänzung.

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

## Effiziente Fortsetzung im bestehenden Chat

Auf Nutzerwunsch vom 01.11.2026 gilt:

- Im bestehenden Chat weiterarbeiten. Ein Paketabschluss allein ist kein Grund
  für einen neuen Chat; keinen Chat automatisch erstellen.
- HANDOFF, neuester PROJECT_STATUS-Abschnitt und der zuletzt belegte Commit/PR
  bilden den knappen Einstieg. Bereits im Chat gelesene unveränderte Inhalte
  anhand bekannter Refs weiterverwenden; gezielt nur Änderungen oder für den
  konkreten Schritt noch fehlende Stellen nachlesen. Nach Kontextverlust diese
  Einstiegspunkte lesen, keine vollständige Historie erneut aufbauen.
- Stand-/CI-Prüfungen zum Beginn eines tatsächlichen Arbeitspakets, vor
  Remote-Schreibaktionen oder bei neuen Änderungshinweisen durchführen. Eine
  reine Nachfrage oder unveränderte Blockade löst keine erneute Gesamtprüfung
  aus. Historische Ergebnisse nicht als frisch geprüft ausgeben.
- Bei einem dokumentierten Zugriffsblocker erst nach belegter relevanter
  Zugriffsänderung oder neu verifiziertem zulässigem Originalzugang fortsetzen.
  Keine identischen Abrufe, Warteschleifen, wiederholten Diagnose-/Planungspakete
  oder Hintergrundprüfungen ohne neuen Anlass. Fehlende Eingänge und bereits
  geltende Nutzer-OKs getrennt festhalten; ein OK beseitigt keinen technischen
  Blocker. Autorisierte nötige Quellen-/Metadatenprüfung bleibt möglich.
- Kurz nur neue Befunde, nötige Entscheidungen, Sicherung und den einen nächsten
  Schritt berichten. Den vollständigen Folgeprompt einmal in HANDOFF pflegen;
  im selben Chat bei unverändertem Auftrag darauf verweisen, statt ihn bei jeder
  Antwort zu wiederholen. Kopiertext nur bei geändertem Folgeauftrag, auf Wunsch
  oder bei einem tatsächlich nötigen Chatwechsel ausgeben.
- Aufwand durch kleine Diffs, gezielte Ausschnitte und passende einmalige
  Prüfungen begrenzen. Tokenfreiheit nicht versprechen; keine unveränderten
  Dateien oder bloßen Statuswiederholungen eigens committen.

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
- **Folgeprompt sichern und gezielt ausgeben:** Sobald ein
  Dokumentationspaket abgeschlossen und mit Commit in einem eigenen PR remote
  gesichert ist, den kopierfertigen Startprompt für genau den einen vorbereiteten
  Folgeauftrag in HANDOFF sichern. Im bestehenden Chat bei unverändertem
  Folgeauftrag den Abschnitt verlinken; Kopiertext gemäß der obigen
  Fortsetzungsregel nur bei Bedarf ausgeben. Der Prompt nennt:
  Branch, letzter verlässlich gesicherter Commit/PR, genaue Aufgabe, Leseplan,
  Schutzgrenzen, Abnahme und Stopbedingungen. Fehlende Eingänge und
  Blocker ausdrücklich angeben; ein fehlender Eingang bleibt eine Startbedingung.
  Den tatsächlichen PR-/CI-Stand getrennt berichten; laufende oder fehlgeschlagene
  CI nicht als erfolgreich ausgeben. Die Prompt-Ausgabe startet weder den
  Folgeauftrag noch einen neuen Chat und erteilt keine zusätzliche Freigabe.
- **Chatwechsel nur bei Bedarf:** Den bestehenden Chat bevorzugen; einen Wechsel
  nur auf Nutzerwunsch oder bei einer konkreten technischen Notwendigkeit
  empfehlen und den Grund benennen. Dann einen kopierfertigen Übergabeprompt
  ausgeben. Prompt auch in HANDOFF
  sichern: Branch, letzter verlässlich gesicherter Commit/PR, genaue Aufgabe,
  Leseplan, Schutzregeln und bekannte Blocker. Bei noch ausstehendem OK den
  vorgeschlagenen Startauftrag ausdrücklich als solchen kennzeichnen.
- **Kurz berichten:** Ergebnis, Sicherung und nächsten Auftrag nennen; nach
  Dokumentationsabschluss bei unverändertem Folgeauftrag auf HANDOFF verweisen,
  sonst den geänderten Folgeprompt direkt mitliefern. Keinen neuen Chat selbst
  anlegen und keine Erinnerungs-/Hintergrundautomatisierung daraus ableiten.

Kurz auf Deutsch zu Beginn und nach Arbeitspaketen berichten: **fertig / aktuell /
als Nächstes**. Gezielte Ausschnitte/Suchen, kompakte Werkzeugausgaben und keine
vollständigen Datentabellen im Chat. Keine unnötigen Wiederholungstests oder
Unteragenten; Delegation nur bei ausdrücklichem Auftrag.
