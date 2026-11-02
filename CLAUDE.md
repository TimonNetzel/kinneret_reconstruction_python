# Aktueller Projekteinstieg für Claude

> **Aktueller Einstieg für den Wechsel zu Claude, 02.11.2026:**
> **[Hier beginnen: vollständige Übergabe und climate_modelling](docs/thesis_research/CLAUDE_RESUME_CLIMATE_MODELLING_2026_11_02.md).**
> Der Nutzer stellt in der nächsten Sitzung einen neuen Ordner
> `climate_modelling` bereit. Noch nicht geprüft, genauer Pfad noch nicht gebunden.
> Zuerst gegen `main.R`, `main_v1.2.R` und die
> [priorisierte Fehlenden-/Kopienliste](docs/thesis_research/MISSING_PROJECT_FILES_2026_11_02.md)
> abgleichen. Dann erst verbleibende Lücken/weiteren technischen Auftrag festlegen.
> `/automount/` lag auf gelöschter Festplatte; lokale Kopien können trotzdem
> existieren. Erst Kopien prüfen, nur verbleibende nötige Dateien neu erstellen.
> [Quellenregel](docs/thesis_research/SOURCE_RECOVERY_PLAN.md). Keine neuen Läufe.
> Bisher 200 historische Teiltests gesichert; acht neue Parserprüfungen bestanden.
> Elf Nachweisbündel offen, keine finale Variante. Frühere Einstiege sind Verlauf.

> **Neuester Teilfortschritt: Altersraster/Zeilenauswahl, 02.11.2026:**
> [Belege und synthetische Prüfung](docs/thesis_research/literal_age_selection_2026_11_02/README.md).
> 39 neue Tests, mit Vorpaketen 124 grün. Exakte Rastermitgliedschaft ist kein
> Intervalltest; vorläufige Holozänmaske wird im erhaltenen Kontrollfluss auf
> alle Zeilen zurückgesetzt. Taxonwechselzweig bleibt durch `& FALSE` aus.
> `age`/`site.age` und freie Maskenvariablen nicht still synchronisieren.
> Keine finale Laufbindung; elf Nachweisbündel bleiben offen.
> Nächster unabhängiger Einstieg: kumulativer Dichte-/EMD-Ausschnitt in
> `src/mcmc_functions_v1.6.R:366–370,485–490` gegen skalare Referenzargumente
> der Fallstudien prüfen. Keine neue Referenz/Kalibrierung oder Vollrunner.
> Ältere Einstiege unten sind Verlauf. Keine neuen Läufe oder Speicherortfrage.

> **Neuester Teilfortschritt, 02.11.2026:**
> [Historische Alters-/Tiefenübertragung](docs/thesis_research/literal_age_depth_2026_11_02/README.md)
> aus `check_ages.R` und `mcmc_prepare.R` begrenzt nachgebildet: 43 neue
> synthetische Tests, mit Vorpaketen 85 grün. Listen-/Matrixform, rho(D|A),
> Normierungszeitpunkt und einmalige Altersverschiebung bleiben getrennt.
> Keine finale Variante oder R-Laufzeitgleichheit; elf Nachweisbündel offen.
> Nächster unabhängiger Einstieg: `age`/`site.age`/`reconst_ages` im gebundenen
> `src/mcmc_functions_v1.6.R` auf Zeilen-/Zeitfensterauswahl prüfen, bevor Module
> verbunden werden. Keine neuen Läufe, Ersatzdaten oder Speicherortfrage.
> Die älteren nächsten Schritte unten sind Verlauf. Phase 1 bleibt vorrangig.

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
Für den vollständigen Fall fehlen finale Originalbasis/Chronologieliste,
numerische TFs und historische Laufzuordnung. Keine Quelle automatisch auswählen;
keine Wiederholung der bereits beantworteten Speicherortfrage.
Unabhängiger technischer Teilnachbau der erhaltenen Taxongewichtung/PDF-Summe:
`docs/thesis_research/literal_final_pdf_2026_11_02/README.md`.
22 neue synthetische Tests bestanden; keine finale Laufbindung hergestellt.
Als nächsten begrenzten Teil historische Alters-/Tiefenübertragung aus dem
bereits gebundenen `src/check_ages.R` isolieren und synthetisch prüfen.
Keine neue Chronologie berechnen oder Matrix-/Listenformen gleichsetzen.
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
