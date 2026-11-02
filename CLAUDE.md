# Aktueller Projekteinstieg für Claude

Aktuelle Priorität auf Nutzerauftrag vom 02.11.2026:
**Zuerst die lokalen Rekonstruktionen der Dissertation nachbauen,
anschließend Verbesserungen.** Verbindlicher Ablauf:
`docs/thesis_research/LOCAL_DISS_RECONSTRUCTION_PLAN.md`.

Lies zuerst `AGENTS.md`, den neuesten Abschnitt von
`docs/thesis_research/PROJECT_STATUS.md`, den neuen Arbeitsplan und
`docs/thesis_research/CLAUDE_HANDOFF_2026_11_02.md`.
Die quellengebundene Fallmatrix ist abgeschlossen:
`docs/thesis_research/local_case_matrix_2026_11_02/README.md`.
Kinnerets älterer Proben-/Taxonstand ist jetzt teilweise exakt gebunden:
`docs/thesis_research/kinneret_binding_2026_11_02/README.md`.
Die finale v1.2-Basisdatei und numerischen TFs fehlen weiter; der Nutzer kennt
keinen weiteren Speicherort. Noch kein Fall vollständig reproduzierbar.
Kinnerets Referenz-/Ankerpfad ist dokumentiert, numerische Eingaben und
Laufbindung bleiben offen:
`docs/thesis_research/kinneret_references_2026_11_02/README.md`.
Die historischen Nachbearbeitungsfassungen sind getrennt gebunden:
`docs/thesis_research/kinneret_postprocessing_2026_11_02/README.md`.
Originalposterioren und finale Auswahl-/Kombinationsfassung fehlen weiter.
Der maschinenprüfbare Phase-1-Belegvertrag ist abgeschlossen:
`docs/thesis_research/kinneret_phase1_spec_2026_11_02/README.md`.
Lesender Prüfer: `python tools/check_kinneret_phase1_spec.py` (erwartet Exit 2).
26 Belegdateien, 19 Befunde und 50 Snapshotdateien intakt; elf Nachweisbündel offen.
Nächster fachlicher Einstieg erfordert finale Originalbasis/Chronologieliste,
numerische TFs und historische Laufzuordnung. Keine Quelle automatisch auswählen;
keine Wiederholung der bereits beantworteten Speicherortfrage.
Keine erneute pauschale Suche oder neuen Läufe; ältere Eingaben nicht ersetzen.
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
