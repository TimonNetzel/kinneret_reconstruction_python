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
[PR #54](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/54)
gegen den #53-Branch, offen/ungemergt und mergebar. Design-Sperrcommit
`a4c03c6752708dead6ecd98f016758538ddb71b2`, Design-Blob
`21879814c77f0341457f9f4790dd40e24b555d55`; danach nur Sicherungsverweise.
Automatische CI 41234277995 zum Inhaltscommit bei Abfrage queued; aktuellen
Remote-Head und CI-Stand über PR #54 prüfen.
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

## Kopierfertiger Startprompt für den nächsten Chat

Vorbereitet, noch nicht ausgeführt: Erst das Absenden dieses Prompts beauftragt
die endliche A3-Referenzprüfung. Verlässlich geprüfter Übergabestand
`43d2b030c8c4de0264bef9e42773f184a37916af`; danach nur diese Prompt-/Statuspflege.

```text
Setze das Netzel-Projekt nach PR #54 fort.
Branch: codex/netzel-a3-synthetic-design
Geprüfter Übergabestand: 43d2b030c8c4de0264bef9e42773f184a37916af
Design-Sperrcommit: a4c03c6752708dead6ecd98f016758538ddb71b2
Design-Blob: 21879814c77f0341457f9f4790dd40e24b555d55
PR: https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/54
Danach gegebenenfalls nur Übergabepflege; aktuellen Remote-Head prüfen.

Prüfe zuerst Git-/PR-/Merge-/CI-Stand, Branches, Worktrees und lokale Änderungen;
erhalte ungesicherte Arbeit. Eigener Worktree, Branch und PR, kein Merge.
Lies AGENTS.md, HANDOFF.md, ausschließlich den neuesten Abschnitt von
PROJECT_STATUS.md, A3_SYNTHETIC_DESIGN.md und MODEL_CONTRACT.md; Plan nur
§1, §2-Zeile A3 und §4. NOVELTY_MATRIX nur bei konkreter Belegfrage nachladen.

Hiermit beauftrage ich genau den vorbereiteten Folgeauftrag: das gesperrte
endliche synthetische A3-Design isoliert implementieren, vollständig ausführen
und getrennt nachrechnen. Kein erneutes OK für dieses Paket verlangen.
Nur neue Dateien unter scripts/netzel_a3_synthetic/, das neue Prüfmodul
tests/test_netzel_a3_synthetic.py und den neuen A3_SYNTHETIC_AUDIT.md samt
kleinen vollständigen rationalen Ergebnistabellen sowie kurze HANDOFF-/Statuspflege.
Nur die neue endliche Prüfung manuell starten; keine Zufallsdatenerzeugung,
Fits, Scores, Optimierung, Simulationen oder Kernläufe. Alle S-/I-Unterfälle,
Gegenverfahren, Ausfälle und Grenzen vollständig berichten; Tabellenbytes
und Reproduktionsbefehle dauerhaft sichern. Design/Abnahme nicht an Resultate
anpassen; bei nötiger Designänderung den betroffenen Teil blockiert ausweisen.

Bestehende Designs einschließlich A3_SYNTHETIC_DESIGN, MODEL_CONTRACT,
Modelle, Resultate, Register, Verträge, Importer, Flags, Counts/Gruppen,
Baseline und CLI/App unverändert. Keine reale Daten-/Modellwahl, Beschaffung,
Bereinigung, Datenmatrix/Kohorte, Literatur-/Archivsuche oder erneute
Originalprüfung der Dissertation. A4–A6 nicht ausführen. A1/A2 dokumentarisch
abgeschlossen; Parnell-Zugang, Menezes-(2)/(5), W96, EMPD2-Archiv-/Messstatus
und reale Gates bleiben Grenzen; kein eigener Neuheitsbeitrag abgegrenzt.
Keine Unteragenten, Autorennachrichten oder Merges. Kurze deutsche Updates.
Quellen-/Link-/Diff-/Blobprüfung und gezielte neue Prüfung; Ergebnis und
genau einen begrenzten Folgeauftrag per Commit und eigenem PR sichern.
```
