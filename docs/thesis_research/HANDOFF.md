# Netzel-Projekt: nächster Einstieg

## Gesicherter Abschluss: Prüfung der Prozent-Indikatoridee

Am 31.10.2026 vom Nutzer beauftragt: die neue Idee mit Dissertation und
Projektstand abgleichen, bewerten und verbessern. Basis
`e6a1dbe3611b636edfe3f29d9e8ee5b785eb8527` aus
[PR #45](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/45).
Worktree `/workspace/netzel-percentage-review`, Branch
`codex/netzel-percentage-indicator-review`; eigener PR nach Inhaltscommit.
[CONCEPT_REVIEW](./percentage_indicator_review_2026_10_31/CONCEPT_REVIEW.md).

**Konzeptprüfung abgeschlossen; keine neue Hauptmethode oder reale Freigabe.**
Die Architektur knüpft an Dissertation §2.3/2.4 und den frühen Multiproxy-Plan
an. Prozentwerte können relative Indikator-Auswahlgewichte sein; sie sind
nicht die Wahrscheinlichkeiten kartierten Artenvorkommens der vorhandenen
Transferfunktionen. Nur Taxa mit Funktion zu verwenden verlangt belegte
Pollentyp–Taxon-Brücken und Erhaltung der ursprünglichen Spektralabdeckung.
Prozent-only hebt den fehlenden Zählumfang nicht auf und braucht eine eigene
Evidenzstärke-/Unsicherheitsdefinition.

Empfohlener nächster Entwurf: gemeinsamer Klimazustand mit explizitem
prozentgewichtetem Indikatorscore; als verallgemeinerte Bayes-Auswertung
kennzeichnen, keine abgeleitete Pollen-Likelihood behaupten. Taxonmischung und
gewichtetes Klimamittel haben andere Aussageziele. Ganze Chronologiekurven
weiterreichen; geglättete Altersstufen sind keine neuen unabhängigen Proben.
Zusätzliche Proxys über eigene Beobachtungsmodelle mit erklärtem Varianzanteil,
Autokorrelation/Chronologie und eindeutigen Datenrollen anbinden. Keine
Nachkorrektur auf gewünschtes R², priornahe Stichprobenauswahl oder künstliche
Bandverengung durch Mittelung von Posterior-Kopien.

Die neue Nutzerfrage konkretisiert einen möglichen Weg B aus der
[Entscheidungsvorlage](./empd2_counts_decision_2026_10_31/COUNTS_DECISION_BRIEF.md).
Sie beauftragt hier nur Prüfung/Verbesserung der Idee, keine Implementierung
oder Verwendung echter Daten. Die vorherige bloße A/B-Protokollierung ist
als nächster Einstieg durch den folgenden konkreten Designvorschlag ersetzt;
keine A/B-Wahl rückwirkend als erteilt behandeln.

## Genau ein Folgeauftrag – vorbereitet, wartet auf Nutzer-OK

**Ein schriftliches Vorabdesign für den prozentgewichteten Indikatorscore
mit gemeinsamem Klimazustand erstellen.**

- **Ausgabe:** neue `PERCENTAGE_INDICATOR_DESIGN.md` plus Status/Übergabe.
  Abstrakte Formulierung, keine reale Proben-/Taxon-/Kern-/Modellauswahl.
- **Inhalt:** Aussageziel, Score/Prior/Normierung, Rolle der Evidenzstärke,
  Abdeckung, Fehlwerte/Nullen, taxonübergreifende Abhängigkeiten, gemeinsamer
  Support, Chronologieintegration und späterer Proxy-Anschluss. Unterschiede
  zu Mischung/Klimamittel und echten Counts-/Kompositionsmodellen festhalten.
- **Prüfplan nur vorbereiten:** Widerspruch, gleiche Prozente bei verschiedenem
  N, geringe/fehlende Abdeckung, Prior-only, mehrdeutiges Klima und gemeinsame
  Altersverschiebungen. Keine Simulationen oder Implementierung ausführen.
- **Leseplan:** AGENTS, diese Übergabe, PROJECT_STATUS und CONCEPT_REVIEW;
  `taxon_transfer_2026_10_30/DESIGN.md` und
  `counts_observation_design_2026_10_31/DESIGN_COUNTS.md` nur zur Abgrenzung.
  Dissertation nur bei einer konkreten ungeklärten Formelstelle nachladen.
- **Abschluss:** alle mathematischen Rollen, Annahmen, freien Entscheidungen,
  Nichtidentifizierbarkeit und Stopbedingungen prüfbar sichern. Keine empirische
  Kalibrierung oder nominal gültigen Unsicherheitsbänder behaupten.
- **Schutz:** alte Designs/Modelle/Resultate/Register/Verträge/Importer/Flags/
  Counts/Gruppen, Baseline und CLI/App unverändert. Keine Quellen-/Datenbeschaffung,
  Datenauswahl, Bereinigung, Code-/Test-/Fit-/Score-/Simulations-/Kernläufe,
  Autorennachrichten, Merges oder Unteragenten. Eigener Worktree/Branch/PR,
  Dokument-/Link-/Hash-/Diffprüfung; genau einen Folgeauftrag sichern.

Ein OK beauftragt genau diesen schriftlichen Entwurf und Prüfplan, keine
wissenschaftliche Ausführung oder reale Freigabe. Die Konzeptprüfung ist erledigt.

## Zuerst Stand prüfen

Git-/PR-/Merge-/CI-Stand, Remote-Heads, Branches, Worktrees und lokale/
unversionierte Änderungen prüfen; Arbeit erhalten. Aktuellen ungemergten
Quellenstand verwenden, Basis nennen, eigenen Worktree/Branch anlegen.
main enthält die offene Kette nicht automatisch. Nur gezielter Leseplan.

## Separat offen, kein zusätzlicher Auftrag

EMPD2-Counts-Pfad und genaue Archiv-Erzeugungskette bleiben blockiert; die
[begrenzte Archivsuche](./empd2_archive_lineage_2026_10_31/ARCHIVE_LINEAGE.md)
endet nach zwei gezielten Quellen. Kein neuer Nenner-/Nullbeleg aus Prozenten.
W96-Bestimmungs-/Artbrücken und Monticchio-P1 einschließlich physischem
LGM90D→MON16-Anschluss/Unsicherheit bleiben offen. Klimaperioden/Variablen
harmonisieren; moderne Kartenvalidierung ist kein fossiler Eignungsnachweis.
Reale T0/T1/T3/T4 und allgemeine Multiproxy-App weiterhin nicht abgeschlossen.

## Startprompt für eine spätere Fortsetzung

> Setze das Netzel-Projekt auf codex/netzel-percentage-indicator-review fort;
> nutze den gesicherten Head/PR aus PROJECT_STATUS. Prüfe zuerst Git-/PR-/Merge-/
> CI-Stand, Branches, Worktrees und lokale Änderungen; erhalte ungesicherte Arbeit.
> Erstelle ausschließlich das in HANDOFF vorbereitete abstrakte Vorabdesign
> PERCENTAGE_INDICATOR_DESIGN.md samt kleinem späteren synthetischen Prüfplan.
> Keine neue Recherche/Datenquelle, reale Proben-/Taxon-/Kern-/Modellwahl,
> Matrix/Kohorte/Bereinigung/Codeausführung/Fits/Scores/Simulationen/Kernläufe.
> Alte Designs/Modelle/Resultate/Register/Verträge/Importer/Flags/Counts/Gruppen,
> Baseline und CLI/App unverändert. W96 und reale Gates offen. Eigener
> Worktree/Branch/PR, keine Unteragenten/Kontakte/Merges; kurze deutsche Updates,
> Ergebnis und genau einen nächsten begrenzten Folgeauftrag sichern.

Ein OK im passenden Kontext oder Absenden dieses Prompts beauftragt genau
dieses Designpaket. Kein Chatwechsel erforderlich.
