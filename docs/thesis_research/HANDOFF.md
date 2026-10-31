# Netzel: Übergabe nach A3-Design

## Gesichertes Ergebnis

Nutzer-OK hat das in PR #53 vorbereitete Designpaket autorisiert.
[A3_SYNTHETIC_DESIGN](./A3_SYNTHETIC_DESIGN.md) sperrt vor jeder Ausführung
künstliche Zustände/Wahrheit, Messprozesse, Cut-Zielverteilung, Fälle S00–S12
mit Unterfällen, Fehlerliste I01–I09, getrennte Nachrechnung und exakte
rationale Abnahmekriterien. **A3-Design dokumentarisch abgeschlossen;
A3 insgesamt begonnen, Implementierung/Ausführung nicht erfolgt.** A4–A6
nicht begonnen. Keine realen OFFEN-Felder des unveränderten
[MODEL_CONTRACT](./MODEL_CONTRACT.md) ausgefüllt; A1/A2 dokumentarisch abgeschlossen.

Basis [PR #53](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/53)
bei `ea3157068e53a765d77ded5214ddaf29a732e5b1`, offen/ungemergt und mergebar;
CI [41220662031](https://github.com/TimonNetzel/kinneret_reconstruction_python/actions/runs/41220662031)
erfolgreich. Main `7f1826c`, #52-Head `1bf23cd`; sechs vorgefundene
Worktrees sauber, keine Stashes, kein Reset/Merge. Eigener Worktree
`/workspace/netzel-a3-synthetic-design`, Branch `codex/netzel-a3-synthetic-design`;
eigener PR gegen den #53-Branch; Sicherungsverweise im neuesten Status.
Nur ein neues Designdokument plus kurze Übergabe/Status, keine Modellrechnung.

## Genau ein Folgeauftrag: endliche A3-Referenzprüfung – vorbereitet, wartet auf Nutzer-OK

**Ziel:** ausschließlich das gesperrte synthetische Design implementieren und
seine vollständig festgelegte endliche Prüfung einmal ausführen, mit getrennter
Nachrechnung und vollständigem Audit. Kein Ausbau zur realen Rekonstruktion.
Das jetzige OK autorisiert nur das erledigte Design; die Umsetzung/Ausführung
ist ein gesonderter, noch nicht beauftragter Schritt der A3-Planzeile.

**Eingänge/Leseplan:** Standprüfung von Git/PR/Merge/CI/Branches/Worktrees und
lokalen Änderungen, ungesicherte Arbeit erhalten; eigener Worktree/Branch/PR
ohne Merge. AGENTS, diese HANDOFF, nur neuester Status, das gesperrte
A3_SYNTHETIC_DESIGN und MODEL_CONTRACT lesen; Plan nur §1/§2-Zeile A3/§4.
NOVELTY_MATRIX nur bei konkreter Belegfrage. Keine Quellen-/Archivsuche,
Originalprüfung der Dissertation oder realen Dateien als Dateneingang.
Den Inhaltscommit des Designs vor der Umsetzung ausdrücklich als Sperrstand
benennen; Definitionen, Fälle und Abnahme nicht anhand von Resultaten ändern.

**Erlaubter Umfang:** neue isolierte Python-Dateien ausschließlich unter
`scripts/netzel_a3_synthetic/`, ein neues zugehöriges Prüfmodul
`tests/test_netzel_a3_synthetic.py`, ein neuer Bericht
`docs/thesis_research/A3_SYNTHETIC_AUDIT.md` samt kleiner vollständiger rationaler
Ergebnistabellen unter einem neuen gleichnamigen Artefaktverzeichnis; kurze
HANDOFF-/Statuspflege. Nur Standardbibliothek für die Rechnung, keine
bestehenden Importer, Projektmodelle oder App einbinden. Prüfkern und Referenz
nach den Unabhängigkeitsregeln des Designs trennen. Nur diese neue endliche
Prüfung manuell starten; automatische Repository-CI gesondert berichten.
Keine Zufallsdatenerzeugung, Optimierung, Fits, Scores, Simulationen oder
Kernläufe. Ganze Tabellenbytes dauerhaft im PR sichern; Hash allein genügt nicht.

**Schutzgrenzen:** bestehende Designs einschließlich A3_SYNTHETIC_DESIGN,
MODEL_CONTRACT, Modelle, Resultate, Register, Verträge, Importer, Flags,
Counts/Gruppen, Baseline und CLI/App unverändert; keine reale Auswahl,
Beschaffung, Bereinigung, Datenmatrix/Kohorte, Typbrücke oder Gatefreigabe.
A4–A6 außerhalb des Auftrags. Keine Unteragenten, Autorennachrichten oder Merges.

**Stopbedingungen:** fehlt eine ausführbare Definition, widersprechen sich
Design/Vertrag, ist eine echte Designänderung nötig oder reichen die neuen
isolierten Pfade nicht aus, den betroffenen Teil blockiert ausweisen und
nicht still erweitern. Bei falschem Ergebnis Definitionen/Erwartungen erhalten;
Implementierungsfehler innerhalb des gesperrten Designs beheben und Änderungen
nachvollziehbar berichten. Ungültiges q/Z, unbekannter Messstatus,
Doppelevidenz oder reales Eingangsmaterial stoppen die betroffene Berechnung.
Keine Auswahl günstiger Fälle; kein bestandener Gesamtanspruch bei offenen,
fehlgeschlagenen, nicht schätzbaren oder nicht ausgeführten Pflichtfällen.

**Prüfbares Ende:** alle S-/I-Unterfälle und Gegenverfahren mit festem Eingang,
Erwartung, tatsächlichem Befund und Status vollständig berichten; rationale
Zustandsmassen/Marginalen beider Wege exakt vergleichen, Fehlerablehnungen und
Erkennung der absichtlich defekten Verfahren belegen. Alle Tabellen und
Reproduktionsbefehle im neuen Audit mit dauerhaften Pfaden sichern.
Quellen-/Link-/Diff-/Blobprüfung, gezielte neue Prüfung, Commit und eigenen PR
ohne Merge. Teilresultat auch bei Blockade sichern, ohne A3 dann als insgesamt
bestanden zu bezeichnen. Selbst eine bestandene endliche Prüfung bestätigt
nur die künstlichen Fälle, keine reale Gültigkeit, Deckung oder Neuheit.

**Weiter offen:** Parnell-Originalzugang blockiert, Menezes-(2)/(5) offen,
kein eigener Neuheitsbeitrag abgegrenzt; W96, EMPD2-Archiv-/Messstatus und reale
T0/T1/T3/T4/Monticchio-P1 offen. Diese Grenzen sperren reale Nutzung und werden
auch durch synthetischen Erfolg nicht gelöst.
