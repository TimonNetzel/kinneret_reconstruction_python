# Netzel-Projekt: nächster Einstieg

## Gesicherter Abschluss: Literatur-Erstprüfung des Gleichungsentwurfs

Am 31.10.2026 vom Nutzer gefragt, ob der Ansatz bereits verwendet wurde.
Basis `6e28a8541f58a263cbd455d46b5ef8df101bae43` aus
[PR #47](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/47).
Worktree `/workspace/netzel-literature-check`, Branch
`codex/netzel-method-literature-check`; eigener PR nach Inhaltscommit.
[LITERATURE_CHECK](./method_literature_check_2026_10_31/LITERATURE_CHECK.md).

**Erstprüfung abgeschlossen: Grundarchitektur und wichtige Bausteine haben
klare Vorläufer; exakte Kombination und eigener Neuheitsbeitrag bleiben offen.**
Primärquellen: Parnell et al. 2015 (Bclim, Pollen/Chronologie), Werner & Tingley
2015 (Aktualisierung unsicherer Altersmodelle), Bowen et al. 2020 (Joint Proxy
Inversion, marine Anwendung ohne mitgerechnete Altersunsicherheit), Menezes
et al. 2026 arXiv:2609.08866v1 (zero-/N-inflated multinomial logistic-normal,
BART, cut-posterior, Monticchio). Letzteres ist eine sehr nahe Preprintarbeit,
kein hier bestätigter begutachteter Zeitschriftenartikel. Counts nicht mit
Prozent-only oder gemeinsamer Multiproxy-Chronologie gleichsetzen.
Nur im Bericht benannte Abstract-/Textstellen gelesen, keine Vollprüfung.

Der neue Nutzerauftrag begründet diese Literaturprüfung; der vorher
vorbereitete Kalibrier-/Identifizierbarkeitsvertrag ist nicht ausgeführt und
wird als nächster Einstieg durch die folgende Abgrenzung ersetzt. #47 bleibt
unveränderter Entwurf. Keine Neuheit/Realvalidierung/Freigabe behauptet.

## Genau ein Folgeauftrag – vorbereitet, wartet auf Nutzer-OK

**Begrenzte methodische Abgrenzungsmatrix zu drei nahen Primärarbeiten erstellen.**

- Neue NOVELTY_MATRIX.md plus Status/Übergabe. Nur Methodenteile von
  Parnell 2015, Bowen 2020 und Menezes v1 2026 gegen den #47-Entwurf lesen.
- Prüffelder: Beobachtung/Counts/Prozente/Nullen, Kalibrierbrücke, Klimaprozess,
  Chronologie-Rückkopplung, Proxy-Abhängigkeit und Unsicherheitsvalidierung.
- Je Feld Gleichung/Abschnitt, Gemeinsamkeit, konkrete Differenz und offene
  Beleglage festhalten. Fehlender Volltext wird Zugangslücke. Nach genau diesen
  drei Arbeiten stoppen, keine Schneeballsuche/umfassende Neuheitsbehauptung.
- Abschluss: höchstens einen überprüfbaren eigenen Beitragskandidaten
  abgrenzen oder feststellen, dass noch keiner belegt ist. Vorhandene Methoden
  als Vorarbeiten anerkennen; keine neue reale Methode oder Variante wählen.
- Leseplan: AGENTS/HANDOFF/PROJECT_STATUS, LITERATURE_CHECK, #47-Entwurf
  joint_percentage_equation_2026_10_31/PERCENTAGE_INDICATOR_DESIGN.md und
  genau die drei im Bericht verlinkten Primärarbeiten.
- Keine Implementierung, Datenbeschaffung/-auswahl, Matrix/Kohorte/Bereinigung,
  Fits/Scores/Tests/Simulationen/Kernläufe, Kontakte, Merges oder Unteragenten.
  Alte Designs/Modelle/Resultate/Register/Verträge/Importer/Flags/Counts/Gruppen,
  Baseline und CLI/App unverändert; W96 und reale Gates offen. Eigener
  Worktree/Branch/PR, Quellen-/Link-/Diffprüfung und genau ein Folgeauftrag.

Ein OK beauftragt nur diese begrenzte Methodenlektüre und Matrix, keine
wissenschaftliche Ausführung. Die Erstprüfung ist erledigt.

## Zuerst Stand prüfen; separat offene Grenzen

Git-/PR-/Merge-/CI-Stand, Remote-Heads, Branches, Worktrees und lokale/
unversionierte Änderungen prüfen, Arbeit erhalten, Basis nennen; eigener
Worktree/Branch. main enthält die ungemergte Kette nicht automatisch.
Einstieg #47 offen/ungemergt/mergefähig, CI 41201264089 erfolgreich,
main `7f1826c`, neun vorherige Worktrees sauber.

Literaturmodelle sind keine Freigabe unserer Typbrücken, Archiv-Messstatus,
Kalibrierung oder Chronologie. W96, Monticchio-P1/physischer LGM90D→MON16-
Anschluss und EMPD2-Exportkette bleiben offen; keine neue Kern-/Modellwahl.
Begrenzte EMPD2-Archivsuche wird nicht wieder geöffnet.

## Startprompt

> Setze das Netzel-Projekt auf codex/netzel-method-literature-check fort,
> gesicherter Head/PR aus PROJECT_STATUS. Prüfe Git-/PR-/Merge-/CI-Stand,
> Branches, Worktrees und lokale Änderungen; erhalte Arbeit. Bearbeite nur die
> vorbereitete NOVELTY_MATRIX zu Parnell 2015, Bowen 2020 und Menezes v1 2026
> gegen den Entwurf aus PR #47. Nach drei Arbeiten stoppen. Keine umfassende
> Neuheitsbehauptung oder reale Modellentscheidung, keine Datenbeschaffung/
> Auswahl/Bereinigung/Matrix/Kohorte/Codeausführung/Fits/Scores/Tests/
> Simulationen/Kernläufe. Alte Designs/Modelle/Resultate/Register/Verträge/
> Importer/Flags/Counts/Gruppen, Baseline und CLI/App unverändert. W96 und
> reale Gates offen. Eigener Worktree/Branch/PR, keine Unteragenten/Kontakte/
> Merges; kurze deutsche Updates, Ergebnis und genau einen Folgeauftrag sichern.

Ein OK oder Absenden dieses Prompts beauftragt genau dieses Paket.
