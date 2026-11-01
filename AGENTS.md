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

## Einstieg und Fortsetzung

- Im bestehenden Chat arbeiten; keinen neuen Chat oder Hintergrundmonitor anlegen.
- Einstieg: [HANDOFF](docs/thesis_research/HANDOFF.md), nur der neueste Abschnitt
  von [PROJECT_STATUS](docs/thesis_research/PROJECT_STATUS.md), danach ausschließlich
  auftragsrelevante verlinkte Stellen. Bereits bekannte unveränderte Inhalte
  anhand ihrer Refs wiederverwenden; nach Kontextverlust keine Gesamthistorie laden.
- Bei Quellenarbeit den [Quellenindex](docs/thesis_research/SOURCE_INDEX.json)
  durchsuchen und die referenzierten Evidenzstellen nutzen. Für Beschaffung,
  neue Quellen, Zugriffsfehler oder unklare Originalverfügbarkeit zusätzlich den
  [Rechercheablauf](docs/thesis_research/RESEARCH_WORKFLOW.md) lesen.
- Beim tatsächlichen Paketstart lokalen Status einschließlich ungesicherter/
  unversionierter Arbeit, Worktrees, relevante Remote-Heads, PR-/Merge-/CI-Stand
  prüfen; erneut vor Remote-Schreibaktionen oder bei Änderungshinweisen.
  Fehlender API-Zugriff belegt keinen Remote-Stand. Basis bewusst nennen,
  ungesicherte Arbeit erhalten, nichts blind zurücksetzen. Bei gewünschter
  Isolation eigener Branch/Worktree; bereits isolierte Cloud-Tasks benötigen
  ohne Nutzerauftrag keinen weiteren Worktree.
- Reine Nachfragen und unveränderte Blockaden lösen keine Gesamtprüfung,
  erneute Quellenlektüre, Abrufe, Diagnosepakete oder Statuscommits aus.
  Werkzeugausgaben auf benötigte Felder/Ausschnitte begrenzen; keine vollständigen
  PR-Diffs, Datentabellen oder langen Historien als Routineeinstieg.

## Recherche und Evidenz

- Fehlende öffentliche Original-/Herausgeberquellen und nötige schreibgeschützte
  Originaldaten-/Metadatenprüfungen im beauftragten Umfang selbst beschaffen und
  lesen, ohne zusätzliches Recherche-OK. Keine Nutzerdatei, Titel oder Ref
  voraussetzen, wenn der öffentliche Beleg selbst ermittelbar ist.
- Zuerst vorhandene passende Evidenz wiederverwenden; Original erneut nur für
  neue Fragen, Versionsänderungen, Widersprüche oder unzureichende Fundstellen.
  URL/DOI, Urheber, Version/Stand, genaue Fundstelle, Originalaussage,
  Archiv-/Objekt-/Probenbezug und Beleggrenze sichern. Bei Datenprüfung Objekt,
  Felder/Proben und Zweck vorab begrenzen, Herkunft/Hash/Abfrage dokumentieren;
  Originale unverändert erhalten. Suchtreffer sind keine Originalmethodenbelege.
- Beleg, Zugriff, Originalverfügbarkeit und Freigabe getrennt halten. Ungelesene
  Inhalte nicht als geprüft oder als fachlichen Nichtbeleg darstellen. Gleiche
  Dateinamen/ähnliche Titel beweisen keine Versionsgleichheit. Hash und Befehl
  ersetzen keine dauerhaft gespeicherten Bytes.
- Dokumentierte Zugangsschranken nur bei belegter relevanter Zugriffsänderung oder
  neu verifiziertem zulässigem Originalzugang erneut prüfen. Keine identischen
  Abrufe bei unveränderter Schranke, Warteschleifen, Umgehungen oder erfundenen
  Dateipfade. Neue Recherche
  begrenzen; Fehler und verbleibende Lücke konkret nennen. Nur abhängige Aussagen
  blockieren; unabhängige Schritte innerhalb desselben Auftrags fortsetzen.
- Notwendige Nutzeraktion und optionale Hilfe konkret unterscheiden. Nur gezielt
  bei nicht zugänglichem/privatem Objekt, unklarer Identität oder echter fachlicher
  Entscheidung nachfragen; keinen Such-/Upload-/Kontaktzwang erzeugen. Wenn kein
  verifizierter Nutzerschritt den Blocker behebt, ausdrücklich sagen.
- Aktuelle Nutzeranweisungen gehen historischen Paketverboten vor. Alte Berichte/
  Verträge nicht rückwirkend umschreiben; überholte Einschränkungen als Historie
  behandeln. Recherche erzeugt keine Werkzeuge, Netz-/App-Rechte oder Autorisierung
  für Daten-/Modellwahl, Reparatur, Implementierung, Rechnungen, Kontakte oder
  Freigaben. Eine Leseprüfung ist keine Datenergänzung oder Trainings-/Gatefreigabe.

## Auftrag, Abnahme und Übergabe

- Ziel, Umfang, nötige Eingänge und Stopkriterium festhalten; keine stille
  Erweiterung. Vor wissenschaftlichen Fits Design, Datenrollen und Regeln sichern.
  Autorisierte Evidenzarbeit nicht durch ein weiteres reines Planungspaket ersetzen.
- Ein „OK“, „weiter“ oder sinngleiches Signal autorisiert das konkret vorbereitete
  Paket. Für denselben Auftrag nicht erneut fragen; gültiges OK und technischen
  Blocker getrennt fortschreiben. Kein Auftrag startet ein weiteres Paket automatisch.
- Passend prüfen: Dokumentation auf Aussagen, Links, Fundstellen, Hashbezüge und
  Diffumfang; Code mit betroffenen Tests. Große Läufe nur bei fachlicher Notwendigkeit
  und Auftrag. Bestanden/fehlgeschlagen/nicht schätzbar/übersprungen/nicht ausgeführt
  unterscheiden; historische und laufende CI nicht als frisch erfolgreich darstellen.
- Materiell neue Ergebnisse gemeinsam committen und in eigenem PR ohne Merge sichern.
  Statushistorie erhalten und neuen Abschnitt mit Ergebnis, Prüfung, Grenzen und
  PR-Verweis ergänzen. Keine separaten PRs je Abruffehler, keine Sicherungsketten
  oder unveränderten Dateien eigens committen. Branch/PR und geprüfter Inhaltscommit
  genügen als Sicherung; eine Datei enthält nicht ihren eigenen endgültigen Commit.
- HANDOFF führt genau einen konkreten Anschluss mit Ziel, Grenzen, Eingang,
  Leseplan, Abnahme/Stop und passendem Autorisierungsstatus. Andere Aufgaben als
  blockiert/später kennzeichnen. Neue Pakete vorbereitet auf Nutzer-OK stellen;
  bereits autorisierte bleiben gültig. Kopierprompt dort einmal pflegen und im
  selben Chat verlinken; nur bei Änderung, Nutzerwunsch oder nötigem Chatwechsel
  erneut ausgeben. Chatwechsel nur auf Wunsch/technischen Bedarf, Grund nennen.
- Kurz auf Deutsch zu Beginn/Abschluss berichten: fertig, aktuell, als Nächstes;
  neue Befunde, nötige/optionale Nutzeraktionen, Sicherung und CI-Grenzen nennen.
  Keine Tokenersparnis ohne Messung versprechen. Keine unnötigen Wiederholungstests
  oder Unteragenten; Delegation nur bei ausdrücklichem Auftrag.
