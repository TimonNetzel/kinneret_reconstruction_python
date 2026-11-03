# Fortsetzung nach climate_modelling-Abgleich

Stand 02.11.2026. Gilt für ChatGPT/Codex und Claude gleichermaßen.
Deutsch, kurze Fortschrittsmeldungen, keine Subagenten; nur vorhandener vom
Nutzer benannter Checkout, keine Klone/Worktrees. Bestehender Branch
`dissertation-reconstruction`, [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
offen, kein automatischer Merge. Tatsächlichen Git-/PR-Stand vor Beginn prüfen.
Den Sicherungscommit dieses Pakets liefert
`git log -1 -- docs/thesis_research/climate_modelling_audit_2026_11_02/README.md`.

## Jetzt gesicherte Quellenlage

Zuerst [Abgleich und Grenzen](climate_modelling_audit_2026_11_02/README.md) und
neuesten [Projektstatus](PROJECT_STATUS.md) lesen. climate_modelling ist jetzt
privat absolut gebunden und tatsächlich zusätzlich geprüft: 1.828 Dateien,
161 R-Quellen, ein ZIP-Verzeichnis, 48 gezielte Abfragen. Die frühere Aussage
„Ordner noch ungeprüft“ gilt nur für den eingefrorenen Stand `705cf7b`.

- `main.R:128`-Verbreitungsbasis gefunden, bytegleich zur Kinneret-Kopie,
  720 × 360 × 275. Keine Wiederherstellung der 279 numerischen TFs.
- F14/F15: CSV-Kandidaten mit 455/6.849 Zeilen, SISAL-V2-Identität ungeklärt.
  Ohne Versionsbeleg weder einsetzen noch neu erzeugen.
- F26 v4.0 nicht namentlich gefunden; zwei v1.4-Fassungen unterscheiden sich.
  Unversionierte Dubletten sind gehasht. Große Sammelstände nicht voll laden.
- Übrige 37 priorisierte Namen und acht Originalposteriorvarianten ohne neue
  Namenskopie. Finale Basis/Listenchronologie, numerische TFs, globale CRU-/Anker-
  und Isotopenworkspaces bleiben offen; sftlf ist keine mask2-Eismaske.
- 23 Parser-/Lesertests bestanden. Elf Nachweisbündel weiter offen; keine finale
  Variante oder vollständige historische Fallreproduktion. Keine Quellenkorrektur.

Die alte [Fehlendenliste](MISSING_PROJECT_FILES_2026_11_02.md) bleibt eingefrorener
Snapshot. Bei neuen Dateifragen immer den neuen Crosswalk ergänzend lesen.
Der alte Scanner besitzt zwei fest konfigurierte Wurzeln; für climate_modelling
nur den neuen Collector verwenden, nicht die alten Ergebnisse überschreiben.
Keine erneute pauschale Inventur oder Frage nach dem unbekannten Erzeuger der
finalen Basisdatei. Automount-Datenträger gelöscht; erst lokale Kopien prüfen,
nur verbleibende notwendige Lücken für kontrollierten Wiederaufbau planen.

## Nächster begrenzter technischer Auftrag in Phase 1

Der neue Bestand schließt keine Laufvoraussetzung. Deshalb als unabhängig
prüfbaren nächsten Schritt den bereits quellengebundenen
[TF-Vorbereitungsauftrag](CLAUDE_NEXT_STEP_TF_PREPARATION_2026_11_02.md) ausführen:
Zeilennormierung und anschließendes Nullsetzen **explizit übergebener**
Klimazellmasken aus `original:src/mcmc_prepare.R:1581–1586,1613–1618` begrenzt
nachbilden und synthetisch gegen unabhängige rationale Sollwerte prüfen.
Dessen eingefrorenen Text, Quellauszüge und Hashes unverändert lassen.

Konkrete Abnahme: neues getrenntes Modul, aussagekräftige synthetische Prüfung,
Erhalt der Reihenfolge Normierung → Maskierung, keine automatische erneute
Normierung nach Maskierung. TANN/PANN, Taxon-/Klimazellenachsen und explizite
Masken sauber binden; nicht aus diesen Tests auf historische TF-Werte schließen.
Keine unbelegte `inside.range`-Randkonvention oder Reparatur von Nullsummen/
Nichtendlichkeit einführen; geltende Domäne und Grenzen aus dem eingefrorenen
Auftrag übernehmen. Bestehende vier Module und Belegpakete unverändert bewahren.
Keine TF neu schätzen, keine Zufalls-/Medianpfade oder Vollrunner ausführen.

Neue Belege für CSV-Versionen oder die verbliebenen Lücken nur bei konkretem
Anlass gezielt erschließen. Etwaige Wiederaufbauplanung muss Erzeuger,
Eingaben, Einheiten, Version und Abnahme benennen; eine Neuerschaffung ist kein
Originalbeleg. Fachliche Variantenwahl bleibt offen und braucht belastbare
Quellen oder eine konkrete Nutzerentscheidung.

Vor Abschluss relevante Prüfungen sowie die beiden Phase-1-Belegprüfer
(erwartet Exit 2) ausführen, neue Ergebnisse dokumentieren und geprüftes Paket
auf demselben Branch committen/pushen. Baseline, App/CLI, Originale und
Monticchio-/Chronologie-Gates erhalten; Phase 2 bleibt nachgeordnet.
Keine Trainings/Sampler/Altersmodelle/Fossilläufe; Plots jung links, alt rechts.
