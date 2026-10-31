# Neuer Arbeitsplan: modulare taxonbasierte Multiproxy-Rekonstruktion

31.10.2026. Vom Nutzer bestätigte Forschungsrichtung und beauftragte neue Planung:
Dissertation konsequent mit modernen Pollenmodellen, insbesondere Menezes,
verbinden; Plan hochladen und eine tokensparende Chatübergabe erstellen.
Basis `df01632d2256a5a7f6c4d3576f9e77c8ea5d7cb9` aus
[PR #49](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/49).
Worktree `/workspace/netzel-modular-plan`, Branch
`codex/netzel-modular-reconstruction-plan`;
[PR #50](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/50)
gegen den #49-Branch, offen/ungemergt. Inhaltscommit
`2c7adc8b748046b599783f7d40d7fe7b5a80b4a4`; danach Sicherungs-/Übergabeverweise.
Eigene automatische CI bei Erstellung noch nicht abschließend geprüft.

## 1. Ziel und festgehaltene Arbeitsrichtung

**Erster greifbarer Abschluss:** ein geprüfter diagnostischer Klimaverlauf für
einen begründet ausgewählten Bohrkern mit transparenten Typzuordnungen,
Chronologie, Kalibrierung, Support und Unsicherheitsgrenzen. Eine allgemeine
Multiproxy-App ist ein späterer Ausbau. Nicht alle historischen 279 Funktionen
müssen dafür wiederhergestellt werden.

Grundlage ist ein modern kalibriertes Pollen-Beobachtungsmodell. Seine
Kalibrierunsicherheit wird in einen zeitlich gekoppelten Rekonstruktionsblock
mit Chronologien, Klimaprior/Ankern und unabhängigen weiteren Proxys übertragen.
Die moderne Kalibrierung wird zunächst durch einen Cut vor fossiler/Proxy-
Rückkopplung geschützt. Dieselben Kalibrierziehungen gelten gemeinsam für alle
betroffenen Proben. Der nachgelagerte Block wird je Ziehung separat normiert:

\[
p_{cut}(\psi,C,\tau,\theta\mid E,P,Y,D)
=p(\psi\mid E)\,p(C,\tau,\theta\mid P,Y,D,\psi).
\]

Die Forschungsrichtung ist bestätigt; konkrete Verteilungen, Klimaziele,
Datensätze, Typen, Kern und Hyperparameter sind noch nicht empirisch freigegeben.
Chronologie-Rückkopplung innerhalb des zweiten Blocks ist eine gesondert zu
begründende Entscheidung. Eine zusätzliche Chronologie-Sperre wäre ein eigener
Cut. Diesen modularen Ansatz nicht mit einem vollständigen gemeinsamen
Bayes-Posterior oder einem normierten Mittel der Likelihoods gleichsetzen.

Menezes liefert Vorarbeiten, keinen fertigen Einsatzvertrag für unsere Daten.
Countsmodell nur bei belegten Counts/N und Feldstatus; Prozent-only erfordert
ein eigenes Beobachtungs-/Präzisionsmodell, keine Pseudo-Counts mit N=100.
TANN/PANN sind nicht GDD5/MTCO/AET-PET. Die sieben kartierten Artenfunktionen
sind keine Pollenanteilsfunktionen; zunächst unveränderte Vergleichsreferenzen,
optionale kalibrierte Merkmale erst als eigene spätere Fragestellung.

## 2. Arbeitspakete mit prüfbarem Ende

| Paket / Status | Ergebnis und Abschlusskriterium |
| --- | --- |
| **A1 – Methodenabgrenzung; bereits autorisiert, nächster Chat** | NOVELTY_MATRIX.md: genau Parnell 2015, Bowen 2020 und Menezes v1 2026 gegen unseren Ansatz. Für sechs Prüffelder Originalgleichung/Abschnitt, Gemeinsamkeit, Differenz und offenen Beleg sichern; Cut-Normierung besonders prüfen. Nach drei Arbeiten stoppen. Höchstens einen prüfbaren Beitragskandidaten benennen oder festhalten, dass noch keiner abgegrenzt ist. |
| **A2 – Modell- und Datenvertrag; geplant** | Einen zusammenhängenden MODEL_CONTRACT.md statt weiterer lose verbundener Konzeptpapiere erstellen: Aussageziel, Klimavariablen/Einheiten/Perioden, Pollenbeobachtung/Nullen/Abdeckung, Kalibrierung, bedingte Normierung, zeitlicher Prior, Chronologie-Rückkopplung, Proxy-Abhängigkeiten, Support und Identifizierbarkeit. Fehlende Daten und Stopbedingungen konkret festhalten; noch keine Datenwahl oder Rechnung aus der Planung ableiten. |
| **A3 – Synthetischer Methodenprototyp; geplant** | Erst gesperrtes Design, dann separat beauftragte Implementierung/Ausführung. Bekannte künstliche Wahrheit: Normierung/Cut-Marginalen, gemeinsame Kalibrierunsicherheit, fehlende Beobachtungen/Nullstatus, zeitliche und taxonübergreifende Abhängigkeiten, Altersverschiebung, Proxykonflikt und nicht identifizierbare Fälle prüfen. Prior-only und fehlerhafte Eingänge explizit; getrennte Nachrechnung, alle Resultate berichten. Keine realen Trainings-/Fossildaten. |
| **A4 – Moderne Kalibrierung und Basisverfahren; geplant, Datenvertrag fehlt** | Begründeten modernen Eingang mit Typ-, Klima-, Zeit-/Orts- und Messrollen sperren. Vor Training Abhängigkeiten, räumliche Aufteilung/Support und unabhängige Prüfung festlegen. Menezes-nahe Basis mit passenden Klimazielen reproduzieren oder Abweichungen präzise ausweisen; moderne Pollenantwort und ihre Unsicherheit prüfen. Kein Vorgriff auf ungesicherte EMPD2-Counts oder bestehende äußere Scores. |
| **A5 – Erster diagnostischer Kern; geplant, reale Gates offen** | Nach T0–T3: Kern, Typabdeckung, Counts-/Prozentvertrag, Chronologie und gegebenenfalls externe Proxys vorab sperren. Pollenbasis, plus Zeit/Chronologie, plus externe Proxys unter vorab festgelegten vergleichbaren Bedingungen berichten. Konvergenz, Support, Konflikte, Nichtschätzbarkeit und Unsicherheit auditieren. Ausfälle und Verschlechterungen erhalten. Monticchio nur nach seinem P1-Nachweis. |
| **A6 – Integration/Erweiterung; später** | Erst geprüften Eingangs-/Ausgangsvertrag und eigenen begrenzten Auftrag an Claude für CLI/App; weitere Kerne/Taxa/Proxys separat. Diagnostischer Kernlauf ist keine pauschale Fossil-/App-Freigabe. |

Die Zeilen A2–A6 sind der Gesamtweg, keine gleichzeitig aktiven Folgeaufträge.
Jedes folgende Paket bekommt erst einen konkreten begrenzten Umfang und die
erforderliche Autorisierung. Der jetzige Auftrag erstellt ausschließlich den Plan;
A1 ist bereits separat genehmigt und für die Fortsetzung im neuen Chat bestimmt.

## 3. Was einen Mehrwert belegen müsste

Beitragskandidat ist die konkret gelöste und geprüfte zeitliche/Chronologie-/
Multiproxy-Erweiterung bei geschützter moderner Kalibrierung. Diese Richtung
hat Vorläufer und wird von Menezes selbst teilweise als Erweiterung genannt.
Die allgemeine Gleichung oder eine neue Priorfamilie beweist keine Neuheit.

Spätere Bewertung vorab festlegen: unabhängige Vorhersage bei ausgelassenen
Daten/Proxys, geeignete prädiktive Scores, Kalibrierung und Abdeckung der
Unsicherheiten bei bekannter Wahrheit, Datierungsverträglichkeit, Support und
Konflikterkennung. Schmalere Bänder/glattere Kurven allein sind kein Gewinn.
Zur Anpassung verwendete Proxys nicht als unabhängige Bestätigung ausgeben;
AP/NAP aus denselben Pollendaten nicht doppelt als Evidenz zählen.
Bekannte äußere Testergebnisse wählen keine neue Variante oder Freigabeschwelle.

## 4. Tokensparender Arbeitsmodus

- **Drei operative Einstiege:** AGENTS.md für Regeln, HANDOFF.md für genau den
  nächsten Auftrag, dieses Dokument für Richtung/Gates. PROJECT_STATUS nur den
  obersten aktuellen Abschnitt lesen, nicht die gesamte Historie.
- A1 benötigt daraus nur §1, §2 Zeile A1 und §5. Die drei Primärquellen gezielt
  nach Methodenabschnitten/Gleichungen öffnen; keine kompletten PDFs in den Chat.
  Quellenstelle/Version und Befund sofort in die Matrix aufnehmen, Wiederlesen
  nur bei konkreter offener Frage. Zugangslücken begrenzt dokumentieren.
- Alte Memo-/Designpakete nur bei präziser Zuordnungsfrage. Keine Rekonstruktion
  der Chatgeschichte, keine Wiederholung bereits bestandener wissenschaftlicher
  Läufe, keine Neu-Inventur der Daten/Taxa oder breite Archiv-/Schneeballsuche.
- **Ein aktives Ergebnisdokument je Paket** plus kurze Status-/Übergabeänderung;
  neue Evidenz ergänzt die Entscheidung, statt ein weiteres allgemeines
  Konzeptpapier zu erzeugen. Unveränderte Resultate/Quellenregister erhalten.
- Git-/PR-/CI-/Worktree-/Änderungsprüfung zu Beginn einmal kompakt bündeln;
  vor Abschluss gezielter Diff-/Link-/Belegcheck. CI nur für relevante Heads,
  keine pauschalen Tests bei Dokumentation. Neue Risiken/Änderungen rechtfertigen
  weitere Prüfungen, unveränderte erfolgreiche Checks nicht wiederholen.
- Kurze deutsche Updates nur bei Start, materiellem Befund und Abschluss;
  während längerer Arbeit spätestens nach etwa 60 Sekunden. Werkzeugausgaben
  filtern, keine Rohdaten/Volltexte/kompletten API-Antworten drucken.
  Abschluss enthält Ergebnis, Commit/PR, verbleibende Grenze und genau einen
  nächsten Auftrag. Keine Unteragenten oder neuen Chats automatisch anlegen.

## 5. Einziger unmittelbarer Auftrag A1: Umfang und Quellen

**Bereits per Nutzer-OK autorisiert; kein erneutes OK nötig.**
Ergebnis: neue NOVELTY_MATRIX.md, plus Status/Übergabe. Keine Ausführung A2–A6.

1. [Parnell et al. 2015](https://doi.org/10.1111/rssc.12065): Polleninversion,
   Kalibrierung, zeitlicher Klimaprozess und Behandlung von Zeitunsicherheit.
2. [Bowen et al. 2020](https://cp.copernicus.org/articles/16/65/2020/): gemeinsame
   Proxy-Inversion und Datenabhängigkeiten; Behandlung und tatsächliche Nutzung
   von Altersunsicherheit im Beispiel unterscheiden.
3. [Menezes et al. 2026, v1](https://arxiv.org/html/2609.08866v1): Schwerpunkt
   §1.3, §2.1–2.3, einschlägige Validierung und §5. Counts/Prozente/Nullen,
   Cut-Ziel und Normierungsrollen von Gl. (2)/(5), zeitlicher Ausbau und
   zusätzliche Proxys prüfen. Keine Fehlerbehauptung aus einer verkürzten Formel.

Sechs Matrixfelder: Beobachtung/Nullen, Kalibrierbrücke, Klimaprozess,
Chronologie/Rückkopplung, Proxy-Abhängigkeit, Unsicherheitsvalidierung.
Je Zelle nur Quelle/Locator und wesentlichen Befund; offene Punkte gesondert.
Bei nicht erreichbarem Original Zugangslücke nennen, keine Ersatzquelle suchen.
Nach diesen drei Arbeiten abschließen, auch ohne abgegrenzten Neuheitsbeitrag.
Alte [Erstprüfung](./method_literature_check_2026_10_31/LITERATURE_CHECK.md) und
[Integrationsmemo](./menezes_integration_2026_10_31/INTEGRATION_MEMO.md) nur
bei einer konkreten offenen Zuordnungs-/Normierungsfrage nachladen.

## 6. Schutz, Stand und Sicherung

Keine Implementierung, Datenbeschaffung/-auswahl, Matrix/Kohorte/Bereinigung,
Projektcode-/Test-/Fit-/Score-/Simulations-/Kernläufe im Plan- oder A1-Paket.
W96/Artbrücken, EMPD2-Archiv-/Messstatus und reale T0/T1/T3/T4/Monticchio-P1
bleiben offen. Fehlende Chronologie-/Kalibrierbelege nicht durch Literaturmodelle
ersetzen. Beendete EMPD2-Archivsuche nicht wieder öffnen; neue Datenwege wären
später gesondert begrenzt zu entscheiden. Keine Autorennachrichten oder Merges.
Bestehende Designs/Modelle/Resultate/Register/Verträge/Importer/Flags/Counts/
Gruppen, Baseline und CLI/App unverändert. Eigener Worktree/Branch/PR.

Plan-Einstieg: #49 offen/ungemergt/mergefähig, Head wie Basis, PR-CI
41206168515 erfolgreich; main `7f1826c`, elf vorherige Worktrees sauber.
Bestehende Integrationsbelege verwendet, keine neue Recherche. Nur dieser neue
Plan plus Status/Übergabe; Referenzen, Links und geschützten Diff geprüft.
725 geschützte bestehende Git-Blobs bytegleich zur Basis, erlaubter
Drei-Dateien-Diff und git diff --check bestanden; keine Tests/Rechenläufe.
Der Plan ist der operative Einstieg für neue Arbeit; historische Pläne bleiben
als Belege erhalten, ihre älteren Empfehlungen lösen keine weiteren Aufträge aus.
