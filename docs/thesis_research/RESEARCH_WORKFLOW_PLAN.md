# Plan: Quellenzugriff und Fortsetzung im Netzel-Projekt

Stand: 01.11.2026. **Recherche und Planung abgeschlossen; Umsetzung vorbereitet.**
Auftrag: Internetrecherche zu ChatGPT/Codex, Zugriffsblockern und Kontextverlust;
konkreten Umsetzungsplan entwickeln. Keine wissenschaftliche Weiterarbeit.
Basis: [PR #72](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/72),
Head `cb7ce16007d440252c82c8913c7e6b89d3511bd3`, offen/ungemergt,
vier erfolgreiche Head-Checks bei Einstieg. Sieben bisherige Worktrees erhalten.
Eigener Planungsbranch `codex/netzel-research-workflow-plan`. Kein Merge.

## Befund aus unserem Arbeitsstand

Bereits vorhanden: gezielte Lesepläne, anhaltendes Nutzer-OK, Verbot identischer
Abrufe bei unverändertem Blocker, Originalzitate mit Seiten/Hash und kurze
Chatfortsetzung. Diese Regeln reichen allein nicht, wenn Befunde schwer
auffindbar sind oder Übergaben selbst viel Wiederholung enthalten.

Gemessene Dateigrößen der Basis: AGENTS 176 Zeilen / 12.146 Bytes,
HANDOFF 120 / 7.120, PROJECT_STATUS 1.473 / 97.038,
Provenienzreview 589 / 59.220. Das sind Dateigrößen, keine Tokenmessung.
Die gesamte Historie ist als Routineeinstieg ungeeignet. Weitere Kosten entstehen
durch vollständige PR-Diffs, große Werkzeugausgaben, wiederholte Standprüfungen
und einzelne Sicherungs-/Übergabeschritte ohne neuen fachlichen Inhalt.

Die PhD-Befunde sind in [Evidenz](./furlanetto_thesis_review_2026_11_01/evidence.json)
und [Review](./A4_E_PROVENANCE_EVIDENCE_REVIEW.md#furlanetto-dissertation-gezielte-trs-18-originalmethodenlesung-01112026)
gesichert. Die Originaldatei ist nicht im Git archiviert. Ein Hash ersetzt ihre
Bytes nicht; ein Workspace-/Anhangspfad beweist keine Verfügbarkeit in einer
neuen Umgebung. **Für diese Planung wurde die PhD nicht erneut eingelesen.**

## Was die offizielle Dokumentation belegt

Alle nachstehenden Quellen wurden am 01.11.2026 über Websuche und die relevanten
Seitenabschnitte geprüft. Produktfunktionen hängen von Konto, Oberfläche und
Workspace ab; sie werden hier nicht als bereits eingerichtet ausgegeben.

| Quelle | Gelesener Abschnitt / belegte Aussage |
| --- | --- |
| [OpenAI: Cloud browser](https://help.openai.com/en/articles/20001280-using-cloud-browser-in-chatgpt) | „When a website blocks the task“: Websites können automatisierte Zugriffe sperren, obwohl der persönliche Browser funktioniert. Offizielle Alternativquelle oder eigener Browser sind mögliche nächste Wege. |
| [OpenAI: Codex Cloud internet access](https://learn.chatgpt.com/docs/cloud/internet-access) | „Configuring agent internet access“ / „Domain allowlist“: Umgebungsfreigaben steuern erlaubte Ziele. Diese als Legacy gekennzeichnete Seite ist kein Nachweis unserer aktuellen Laufzeitkonfiguration; diese muss im Executor/Laufzeitstatus geprüft werden. |
| [OpenAI: AGENTS.md](https://learn.chatgpt.com/docs/agent-configuration/agents-md) | „How Codex discovers guidance“: Repositoryanweisungen werden beim Start geladen und nach Verzeichnisebene zusammengeführt. |
| [OpenAI: Rethinking skills and prompts](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra) | Einleitung / „Better skills“: angewachsene, lange oder widersprüchliche Anweisungen und unnötige Skills können Kontext belasten. Modellbezogene Empfehlung, keine Garantie für unseren Aufwand. |
| [OpenAI: Iterating Development Workflows](https://developers.openai.com/cookbook/examples/codex/iterating-development-workflows-with-codex) | Einleitung / „Context files“: zusätzliche Dateinamen sind optionale Konventionen; materielle Entscheidungen, Grenzen und Blocker mit Herkunft für spätere Arbeit sichern. Keine Pflicht, die dortige gesamte Ordnerstruktur zu übernehmen. |
| [OpenAI: Projects in ChatGPT](https://help.openai.com/en/articles/10169521-projects-in-chatgpt) | „Add context“ / „Memory in projects“: Projektdateien, Anweisungen und Chats können Kontext über Chats hinweg bereitstellen; Memory hängt von Einstellungen ab. Das ist kein zugesicherter Dateimount im Codex-Executor. |
| [OpenAI: Compaction](https://developers.openai.com/api/docs/guides/compaction) | „Overview“: Kontextverdichtung unterstützt lange Interaktionen. Beschriebene API-Einstellungen sind keine hier gesetzte ChatGPT-/Codex-App-Einstellung und ersetzen keine nachprüfbare Repo-Evidenz. |
| [OpenAI: Deep research](https://help.openai.com/en/articles/10500283-deep-research-in-chatgpt) | „Select a data source“ / „Connected apps“: Web, Uploads und geeignete verbundene Quellen können genutzt werden; Apps heben Anbieter-/Workspace-Berechtigungen nicht auf. |

## Zielbild und Zuständigkeiten

Unsere Ableitung aus Recherche und Repo-Befund: ein kurzer aktueller Einstieg,
ein auffindbarer Quellennachweis und ein begrenzter Zugriffsablauf. Kein neuer
Agenten-/API-Dienst und keine Vektordatenbank als Voraussetzung.

| Datei / Ort | Vorgesehene alleinige Aufgabe |
| --- | --- |
| `AGENTS.md` | Dauerhafte Arbeitsregeln und zwingende Schutzgrenzen; Routineeinstieg und situationsabhängige Pflichtlektüre definieren. Dopplungen kürzen, keine fachlichen Grenzen entfernen. Zielwert höchstens etwa 800 Wörter. |
| `docs/thesis_research/HANDOFF.md` | Nur aktueller Auftrag, gültiges OK, Branch/Referenz, Ergebnis, Blocker und genau eine nächste Aktion. Zielwert etwa 300–450 Wörter; Historie verlinken. |
| `docs/thesis_research/RESEARCH_WORKFLOW.md` (neu) | Ausführlicher Ablauf für Beschaffung, Auswertung, Wiederverwendung und Blocker; nur bei einschlägiger Recherche lesen. |
| `docs/thesis_research/SOURCE_INDEX.json` (neu) | Kleines zentrales Findeverzeichnis zu bestehenden Evidenzen; erster Umfang nur Furlanetto-Artikel, Supplement und Dissertation. |
| Bestehende Reviews / `evidence.json` | Eigentlicher Beleg mit Zitat, Seiten, Version und Aussagegrenze. Nicht in neue Wissensdateien duplizieren oder nachträglich umschreiben. |
| `PROJECT_STATUS.md` | Erhaltene Ergebnis-/Entscheidungshistorie; nur neuesten Abschnitt gezielt lesen. Kein zweites aktuelles Auftragsregister. |
| Originalspeicher | PDF/Originalbytes getrennt und unverändert; geprüften Speicherort sowie Verfügbarkeit erfassen. Kein automatisches Hochladen von Volltexten ins öffentliche Git. |

Der Index erhält mindestens Quellen-ID, DOI/Identität, Fassung, Objekt-Hash,
Evidenzpfad/Fundstellen, bearbeitete Fragen sowie verbleibende Lücken.
Original-Locator, Verfügbarkeitsstatus und letzte Verifikation getrennt führen:
`verified_available`, `temporary_or_session_only`, `missing`, `not_checked`.
Pro Zugriffsweg: URL, Werkzeug/Route, Zeitpunkt, konkrete Antwort/Blocker und
Bedingung für einen neuen Versuch. Geschlossene Quellen sind keine Nichtbelege.
Ein großer Literaturbestand wird nicht pauschal migriert.

## Vorgesehener Ablauf je fachlicher Frage

1. **Frage und erforderlichen Nachweis festlegen.** Welche Aussage braucht welche
   Originalfassung? Ergänzende Belege als ergänzend markieren. Ein Versionsabgleich
   ist nur für die Aussagen blockierend, die tatsächlich davon abhängen; keine
   neue wissenschaftliche Gateentscheidung oder Beweisanforderung erfinden.
2. **Zuerst wiederverwenden.** Kurz-HANDOFF und passende Indexeinträge lesen;
   dann ausschließlich referenzierte Evidenzstellen. Gleiche Quelle/Fassung und
   bereits beantwortete Frage: vorhandenen Befund nutzen, keine PDF-Volllektüre.
   Erneut ins Original nur bei neuer Frage, Widerspruch, Versionswechsel oder
   unzureichender gespeicherter Stelle. Ein Hashvergleich muss nicht den Volltext
   in den Modellkontext laden. Die PhD wird für die erste Indexbefüllung nicht
   neu geöffnet; vorhandene Evidenz genügt.
3. **Falls Beschaffung nötig: Route gezielt wählen.** Vorhandenes Original /
   geeignete verbundene Quelle, anschließend offizieller Herausgeber bzw.
   institutioneller Datensatz und tatsächlich verlinkte öffentliche Dateien.
   Nur nachweislich zugeordnete Fassungen verwenden; Dissertation/Accepted
   Manuscript/Verlagsfassung ausdrücklich unterscheiden.
4. **Begrenzt eskalieren.** Vorgeschlagenes Standardbudget für einen neuen
   Quellenblocker: ein Erstabruf je begründeter Route, höchstens zwei gezielte
   Suchrunden nach zulässigen Originalalternativen. Budget ist unser Vorschlag,
   kein OpenAI-Limit; ein engerer Nutzerauftrag geht vor. Bei fehlendem Tool,
   Login oder Website-Challenge keine Umgehung und keine erfundenen Pfade.
5. **Früh und konkret melden.** Benötigtes Objekt und genaue Lücke nennen.
   Optional eine bereits zugängliche Datei/einen bekannten Link anbieten;
   keine allgemeine Recherchepflicht an den Nutzer abgeben. Technischer Eingang
   und bereits gültiges OK bleiben getrennt. Innerhalb des erteilten Umfangs
   unabhängige Teilfragen weiterbearbeiten; keine neue wissenschaftliche Aufgabe
   eigenmächtig beginnen. Nur die abhängige Aussage bleibt blockiert.
6. **Einmal sichern.** Materiell neue Befunde/Blocker im passenden Evidenzobjekt,
   Index und kurzen Übergabestand festhalten. Keine erneute Gesamtstandprüfung,
   kein neuer PR und kein Sicherungspaket nur für unveränderte Statusnachfragen.

### Fehler bestimmen, bevor eine Maßnahme vorgeschlagen wird

| Beobachtung | Behandlung / sinnvoller neuer Eingang |
| --- | --- |
| Werkzeug nicht verfügbar | Verfügbare unterstützte Route prüfen; Funktions-/App-Verfügbarkeit nicht annehmen. |
| Executor-/Sandbox-/Proxyfreigabe fehlt | Unterstützte Berechtigung/Konfiguration verwenden; danach wirksame Beobachtung prüfen. Keine direkte Ausweichroute zur Umgehung. |
| Website liefert 403/Challenge trotz erlaubtem Zugriff | Websiteblocker protokollieren; kein erneutes Hostfreigabeversprechen. Zulässiges Original anders beschaffen oder freiwillige Nutzerdatei nutzen. |
| Anmeldung/individuelles Zugriffsrecht erforderlich | Nur angebotenen sicheren Anmeldeweg/zugelassene verbundene Quelle nutzen; keine Zugangsdaten im Chat anfordern. |
| Vorübergehender Timeout/5xx | Höchstens ein begründeter erneuter Versuch; danach speichern und stoppen. Keine Hintergrundschleife. |
| Datei lesbar, konkrete Information fehlt | Fachliche Evidenzlücke; ein anderes Modell/Chat oder ein erneuter Download belegt die fehlende Information nicht. |

Ein bekannter Blocker wird chatübergreifend wiederverwendet. Neuer Versuch nur
bei relevanter Änderung, etwa neuem Original-Link, anderer tatsächlich
verfügbarer berechtigter Route, wirksamer Policy-/Loginänderung oder neuen
Dateibytes. Ein bloßer Chatwechsel ist keine solche Änderung.

## ChatGPT und Codex praktisch nutzen

Im selben Chat weiterarbeiten, solange es zweckmäßig ist. Bei nötigem Wechsel
reichen Repo/Branch und „Lies HANDOFF und die dort verlinkte Evidenz“;
aktuellen Remote-Head bei tatsächlichem Arbeitsbeginn prüfen. Eine saubere
Fortsetzung darf nicht vom Erinnern aller alten Chats abhängen.

Optional für zusätzliche Recherche in ChatGPT ein Projekt mit kurzem Stand
und zugänglichen Quellen verwenden, sofern in der verwendeten Oberfläche
verfügbar. ChatGPT-Projekt, Memory und Codex-Repo sind unterschiedliche
Kontextwege; keine automatische Synchronisation oder Dateiübertragung annehmen.
Deep Research nur für echte Synthesefragen; es ist kein zugesicherter 403-Fix.
Keine App-Installation, Memory-/Netzänderung oder API-Umstellung nötig, um mit
der Repo-Verbesserung zu beginnen. Neue Infrastruktur erst bei belegtem Bedarf.

## Ein empfohlenes Umsetzungspaket

**Vorbereitet – wartet auf Nutzer-OK für diese neue Prozessumsetzung.**
Das bereits erteilte Furlanetto-OK bleibt separat gültig.

1. AGENTS/HANDOFF entdoppeln; wissenschaftliche Schutzgrenzen vollständig
   erhalten und Verantwortlichkeiten gemäß Zielbild zuweisen. Frühere Übergaben
   über ihre gesicherten Refs erreichbar lassen.
2. `RESEARCH_WORKFLOW.md` und `SOURCE_INDEX.json` anlegen. Nur vorhandene
   Furlanetto-Evidenzen registrieren; keine erneute Originallesung oder Suche,
   kein vollständiges Quellenarchiv und kein pauschaler Datenbankimport.
3. Drei Fälle dokumentarisch durchspielen: neue Fortsetzung findet TRS-18-
   Zitat ohne PDF; bestehende Sage-Challenge erzeugt keinen Abruf; fehlender
   Originalspeicher wird als ungeklärt gemeldet und nicht als archiviert behandelt.
4. Links/JSON/Referenzen/Diff und erhaltene wissenschaftliche Grenzen prüfen;
   ein gemeinsamer Umsetzungsbranch/PR, kein Merge. Kein eigener PR pro
   Abruffehler. HEAD/PR-Sicherung ohne Kette weiterer reiner Sicherungscommits
   über einen unabhängigen Hinweis wie Branch/PR und geprüften Inhaltscommit
   organisieren; eine Datei kann ihren eigenen endgültigen Commit nicht enthalten.

**Erfolgskriterien:** HANDOFF im Zielumfang; eine aktive nächste Aktion;
keine widersprüchlichen OKs; Index verweist auf Originalzitate/Versionsgrenzen;
keine unveränderten Zugriffsversuche und keine Volllektüre alter PDFs;
bestehende wissenschaftliche Berichte/Evidenz/Modelle unverändert.
Beim nächsten ohnehin beauftragten Arbeitsschritt Anzahl erneuter Abrufe,
gelesene Kontextbytes/Dateien und Wiederholungsprüfungen notieren, soweit
verfügbar. Das sind Aufwandindikatoren; Token-/Zeitersparnis nicht ohne Messung
beziffern. Kein neuer Chat, Unteragent, manueller wissenschaftlicher Test oder
Modelllauf für die Abnahme nötig.

## Grenzen dieses Planungsschritts

Nur dieser Plan und die dazugehörige Übergabe/Statuspflege werden gesichert.
AGENTS, Quellenindex, Workflowregeln und Infrastruktur bleiben bis zur
Umsetzung unverändert. Furlanetto-Ergebnisse, R04/R05 und alle realen Gates
bleiben fachlich unverändert. Kein PhD-Neueinlesen, keine neuen Originalabrufe,
Werteanalyse, Implementierung des Forschungscodes, Kontakte oder Merges.
