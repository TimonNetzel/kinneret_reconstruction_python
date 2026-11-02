# Aktueller Projekteinstieg für Claude

Aktuelle Priorität auf Nutzerauftrag vom 02.11.2026:
**Zuerst die lokalen Rekonstruktionen der Dissertation nachbauen,
anschließend Verbesserungen.** Verbindlicher Ablauf:
`docs/thesis_research/LOCAL_DISS_RECONSTRUCTION_PLAN.md`.

Lies zuerst `AGENTS.md`, den neuesten Abschnitt von
`docs/thesis_research/PROJECT_STATUS.md`, den neuen Arbeitsplan und
`docs/thesis_research/CLAUDE_HANDOFF_2026_11_02.md`.
Nächster Schritt: quellengebundene Fallmatrix der lokalen Dissertationsfälle,
dann den ersten anhand der Originaleingaben belegbaren Referenzfall festlegen.
Neue Kalibrierung, methodische Verbesserungen und zusätzliche
Reliabilitätsansicht gehören in Phase 2 nach dem gesicherten Nachbau.

Die Fortsetzung liegt auf `dissertation-reconstruction`; vor Beginn Git-/PR-
Stand prüfen. Nicht voraussetzen, dass `main` PR #77 schon enthält.
Deutsch und kurze Fortschrittsmeldungen; keine Subagenten. Nur den vom Nutzer
benannten vorhandenen Checkout verwenden, keine neuen Klone oder Worktrees.

Historische Rechenschritte und spätere Verbesserungen getrennt halten.
App/CLI, Analog-Baseline, Paperreferenz und sieben experimentelle Artenmodelle
bewahren. Die sieben modernen Artenmodelle ersetzen keine historischen TFs;
279 historische numerische Funktionen gelten weiterhin nicht als wiederhergestellt.
Offene Originaleingaben, Quellkonflikte und fachliche Gates sichtbar halten.
Monticchio-/Chronologie-Gates bleiben fallbezogen erhalten. Diese Planänderung
startet keine Trainings-/Samplerläufe oder neuen Fossilrekonstruktionen.
Plots immer jung links, alt rechts. Technische Sicherung ist keine Pilotfreigabe.

Forschungsstands-/Quellenansicht und Sieben-Arten-Vergleich einschließlich
Foldansicht sind umgesetzt und geprüft:
`docs/thesis_research/app_folds_2026_11_02/README.md`.
Abgeschlossene, geprüfte Pakete jeweils auf dem bestehenden Branch in PR #77
committen und pushen; keine automatischen Merges.
