# Netzel-Projekt: nächster Einstieg

## Gesicherte Ergänzung: Dissertation und Menezes verbinden

Am 31.10.2026 vor dem gewünschten Chatwechsel vom Nutzer erfragt:
konsequente Verbindung der modernen Methoden mit der Dissertation und echter
Mehrwert, Schwerpunkt Menezes. Basis
`877f572e40ea028bbd12e4127fbb16ac5f4acd9b` aus
[PR #48](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/48).
Worktree `/workspace/netzel-menezes-integration`, Branch
`codex/netzel-menezes-integration`;
[PR #49](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/49)
gegen den #48-Branch, offen/ungemergt. Inhaltscommit
`08d8ec612bac8f967ce0b9d6be73f96378cee3ac`; danach Sicherungs-/Übergabeverweise.
Eigene automatische CI bei Erstellung noch nicht abschließend geprüft.
[INTEGRATION_MEMO](./menezes_integration_2026_10_31/INTEGRATION_MEMO.md).

**Integrationsvorschlag schriftlich konkretisiert; keine reale Methode gewählt.**
Menezes v1 §1.3/§2.1 und §5 gezielt nachgelesen: modern kalibrierte Pollen-
Countsantworten mit cut-posterior, derzeit keine zeitliche Abhängigkeit oder
Altersunsicherheit in der Inversion. Erweiterungen dazu nennen die Autoren
selbst. Vorschlag: geschütztes modernes Kalibriermodul, nachgelagerter gemeinsamer
Klima-/Chronologie-/Multiproxyblock als Weiterentwicklung der Dissertation.
Dieser Block muss je Kalibrierparameterziehung bedingt normiert werden;
Kalibrierunsicherheit gemeinsam über alle Proben erhalten. Damit wird #47s
vollständiger gemeinsamer Posterior als Vorschlag präzisiert, nicht umgeschrieben.

Eigener Beitragskandidat: nachvollziehbare zeitliche/Chronologie-/Multiproxy-
Erweiterung mit unabhängiger Prüfung und Konfliktdiagnose. Nicht als neu belegt;
schmalere Bänder/glattere Kurven reichen nicht. Klimaziele (Menezes GDD5/MTCO/
AET-PET versus unsere TANN/PANN) und Datenrollen vor Verbindung vereinheitlichen.
Countsmodell benötigt echte Counts/N; keine Pseudo-Counts aus Prozenten.
Arealfunktionen allenfalls als kalibrierte Merkmale oder separater Vergleich,
keine doppelte Polleninformation. W96, Archiv-/Messstatus und reale Gates offen.

## Genau ein nächster Auftrag – bereits autorisiert, für neuen Chat vorbereitet

**Begrenzte methodische Abgrenzungsmatrix zu drei nahen Primärarbeiten erstellen.**
Der Nutzer hat diesem Auftrag bereits mit „Ok“ vor der Chatwechsel-Frage
zugestimmt. Kein zusätzliches OK verlangen. Diese Ergänzungsfrage führt die
vollständige Matrix nicht vorzeitig aus; die Fortsetzung erfolgt im neuen Chat.

- Neue NOVELTY_MATRIX.md plus Status/Übergabe. Nur Methodenteile von
  Parnell 2015 (10.1111/rssc.12065), Bowen 2020 (10.5194/cp-16-65-2020) und
  Menezes v1 2026 (arXiv:2609.08866v1) gegen #47 und dieses Integrationsmemo lesen.
- Felder: Counts/Prozente/Nullen, Kalibrierbrücke, Klimaprozess, Chronologie,
  Proxy-Abhängigkeit und Unsicherheitsvalidierung. Schwerpunkt Menezes:
  cut versus gemeinsamer Posterior, Normalisierungsrollen von Gl. (2)/(5),
  zeitliche Erweiterung und zusätzliche Proxy-Beobachtungsmodelle.
- Je Feld Gleichung/Abschnitt, Gemeinsamkeit, konkrete Differenz und offene
  Beleglage nennen. Bei fehlendem Volltext Zugangslücke; nach genau drei
  Arbeiten stoppen, keine Schneeballsuche oder umfassende Neuheitsbehauptung.
- Abschluss: höchstens einen überprüfbaren Beitragskandidaten abgrenzen oder
  feststellen, dass noch keiner belegt ist. Keine neue reale Methode wählen.
- Leseplan: AGENTS/HANDOFF/PROJECT_STATUS, INTEGRATION_MEMO, #48-LITERATURE_CHECK,
  #47-PERCENTAGE_INDICATOR_DESIGN und genau die drei Originalarbeiten.
- Keine Implementierung, Datenbeschaffung/-auswahl, Bereinigung/Matrix/Kohorte,
  Code-/Test-/Fit-/Score-/Simulations-/Kernläufe, Kontakte, Merges oder Unteragenten.
  Alte Designs/Modelle/Resultate/Register/Verträge/Importer/Flags/Counts/Gruppen,
  Baseline und CLI/App unverändert; W96 und reale Gates offen. Eigener
  Worktree/Branch/PR, Quellen-/Link-/Diffprüfung, genau ein Folgeauftrag.

## Stand und Grenzen

Zuerst Git-/PR-/Merge-/CI-Stand, Remote-Heads, Branches, Worktrees und lokale/
unversionierte Änderungen prüfen, Arbeit erhalten, Basis nennen; eigener
Worktree/Branch. main enthält die ungemergte Kette nicht automatisch.
Einstieg #48 offen/ungemergt/mergefähig, CI 41205275351 erfolgreich,
main `7f1826c`, zehn vorherige Worktrees sauber. Literaturmodelle ersetzen
keine reale Eingangsprüfung. W96, Monticchio-P1 und physischer LGM90D→MON16-
Anschluss, EMPD2-Exportkette/Messstatus bleiben offen; Archivsuche nicht wieder öffnen.

## Startprompt für den bereits beauftragten Methodenvergleich

> Setze das Netzel-Projekt auf codex/netzel-menezes-integration fort,
> gesicherter Head/PR aus PROJECT_STATUS. Prüfe Git-/PR-/Merge-/CI-Stand,
> Branches, Worktrees und lokale Änderungen; erhalte Arbeit. Führe den bereits
> autorisierten Auftrag NOVELTY_MATRIX zu Parnell 2015, Bowen 2020 und
> Menezes v1 2026 aus, Schwerpunkt Verbindung von Menezes mit meiner Dissertation
> gemäß INTEGRATION_MEMO. Prüfe besonders Kalibrier-Cut/Normierung, gemeinsame
> Klima-/Chronologie-Inferenz und zusätzliche Proxy-Modelle. Nach drei Arbeiten
> stoppen; höchstens einen prüfbaren eigenen Beitragskandidaten abgrenzen.
> Keine ungeprüfte Neuheitsbehauptung, reale Modellwahl oder Datenbeschaffung/
> Auswahl/Bereinigung/Matrix/Kohorte/Code-/Test-/Fit-/Score-/Simulations-/Kernläufe.
> Alte Designs/Modelle/Resultate/Register/Verträge/Importer/Flags/Counts/Gruppen,
> Baseline und CLI/App unverändert; W96 und reale Gates offen. Eigener
> Worktree/Branch/PR, keine Unteragenten/Kontakte/Merges. Kurze deutsche Updates,
> knappe Abschlussantwort; Ergebnis und genau einen Folgeauftrag sichern.
