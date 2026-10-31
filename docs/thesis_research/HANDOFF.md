# Netzel: Übergabe nach dokumentarischer E-Belegzuordnung

## Gesichertes Ergebnis

Das Nutzer-OK nach [PR #56](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/56)
ist umgesetzt: [A4_E_EVIDENCE_GAPS](./A4_E_EVIDENCE_GAPS.md) ordnet 20 moderne
E-Anforderungen vorhandenen EMPD2-Archiv-/Messstatusdokumenten zu, mit exakten
Fundstellen, Reichweite, Status und fehlenden Nachweisen. **Belegzuordnung
dokumentarisch abgeschlossen; reale Counts-Bereitschaft weiter blockiert.**
EMPD2 nicht als E gewählt, kein realer Vertrag ausgefüllt, keine Gatefreigabe.

Technische Schlüssel, Archividentität und begrenzte Prozentdarstellung sind
dokumentiert. `S_flag` ist laut früherer Prüfung für genau drei Zeugen mit
gelieferten Prozenten kompatibel; `N_original`, allgemeine Sparse-Nullregel,
angewandte Export-/Hierarchiekette, Typ-/Messabdeckung und weitere reale
Eignungsnachweise bleiben offen. 36/40 Gruppen und erhaltenes Volumen 7,4
nicht angeglichen. Keine unvereinbaren Dokumentaussagen gleichen Geltungsbereichs
festgestellt; begrenzte Release-Abweichungen vollständig erhalten.

Gelesen: vier festgelegte Ausgangsdokumente; DENOMINATOR_RULES und
COUNTS_DECISION_BRIEF nur zur konkreten Nenner-/Releasezuordnung. Keine
zusätzliche Evidenz-JSON oder Originaldaten geöffnet; frühere Zahlen/Original-
prüfungen nicht wiederholt. A1/A2 dokumentarisch abgeschlossen, A3 ausschließlich
endlich-synthetisch bestanden; A4-Umsetzung/reale Kalibrierung und A5/A6 nicht
begonnen. Alle bestehenden Fachdateien unverändert; keine manuellen Tests,
Rechnungen, Fits, Scores oder Läufe.

Basis `e69d95cd2833716987065833e5284d4430316b2f`, PR #56 offen/ungemergt und
mergebar; automatische
[CI 41237720645](https://github.com/TimonNetzel/kinneret_reconstruction_python/actions/runs/41237720645)
erfolgreich. Sieben vorgefundene Worktrees sauber, keine Stashes/Resets/Merges;
main unverändert `7f1826c`. Eigener Worktree `/workspace/netzel-a4-e-evidence-gaps`,
Branch `codex/netzel-a4-e-evidence-gaps`, eigener gestapelter PR gegen #56.
Inhaltscommit `b6b75f8b2296b197c189327b2dac08e6949a5d61` über
[PR #57](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/57)
remote gesichert, offen/ungemergt und mergebar; automatische
[CI 41238645929](https://github.com/TimonNetzel/kinneret_reconstruction_python/actions/runs/41238645929)
an diesem Commit bei Abfrage queued. Danach nur Sicherungsverweise;
aktuellen PR-Head/CI prüfen. Nur neues Ergebnisdokument plus kurze
HANDOFF-/Statuspflege; alte Statushistorie erhalten.

## Genau ein Folgeauftrag: Klimaprovenienz und Zeitbezug dokumentarisch präzisieren – vorbereitet, wartet auf Nutzer-OK

**Ziel:** die offenen E12/E13/E15-Angaben aus A4_E_EVIDENCE_GAPS innerhalb
vorhandener Repository-Dokumentation präzisieren: Welche Aussage zu Klimaquelle,
Version, Einheiten, räumlicher Zuordnung, Normalperiode und Probenzeit stammt
aus welcher dokumentierten Prüfung, und welche konkrete Verbindung fehlt?
Keinen Klimaeingang oder neuen Bezugszeitraum wählen. Der reale Counts-Einstieg
bleibt unverändert gesperrt; keine historische A/B-Option wird damit entschieden.

**Eingänge/Leseplan:** Standprüfung von Git/PR/Merge/CI/Branches/Worktrees und
lokalen Änderungen, ungesicherte Arbeit erhalten; eigener Worktree/Branch/PR,
kein Merge. AGENTS, diese HANDOFF, nur neuester PROJECT_STATUS,
A4_E_EVIDENCE_GAPS E12/E13/E15 und MODEL_CONTRACT zu Klima-/Zeitrollen;
Plan nur §1, §2-Zeile A4 und §4. Danach gezielt:

- [tools/thesis_import/README](../../tools/thesis_import/README.md): nur vorhandene
  Klima-/Einheiten-/Referenzperioden- und Quellenangaben, keine Importausführung;
- [source_audit_2026_10_29/README](./source_audit_2026_10_29/README.md): nur
  einschlägige EMPD2-Klimaprovenienz-/Zeitabschnitte;
- [INPUT_CONTRACT](./empd2_counts_readiness_2026_10_31/INPUT_CONTRACT.json):
  `climate` und `sample_context`; [MODERN_COUNTS_READINESS](./empd2_counts_readiness_2026_10_31/MODERN_COUNTS_READINESS.md)
  nur „Klima, Standort und Zeit“;
- [DENOMINATOR_RULES](./empd2_denominator_rules_2026_10_31/DENOMINATOR_RULES.md)
  nur die dokumentierte frühere Originallesung in „Zwei eng begrenzte Quellenobjekte“.

Keine dort verlinkten externen Originale, Archiv-/Klimadateien oder Datenbanken
öffnen; keine breit angelegte Berichtssuche. Weitere vorhandene Dokumentstelle
nur bei konkretem, in diesen Abschnitten genanntem Belegverweis. A3/NOVELTY_MATRIX
nicht erneut lesen, außer eine konkrete Vertrags-/Belegfrage erfordert es.

**Erlaubter Umfang:** genau ein neues Dokument
`docs/thesis_research/A4_E_CLIMATE_TIME_GAPS.md` und kurze HANDOFF-/Statuspflege.
Eine kompakte Nachweiskette für die genannten Klimarollen mit Datei/Abschnitt,
früherem Prüf-/Quellenstatus, Geltungsbereich, vereinbaren/abweichenden Aussagen
und fehlendem Nachweis erstellen. Originalbefund, damaliger Repositoryvermerk
und ungesicherte Schlussfolgerung unterscheiden. Keine Versions-/Periodenlabels
ergänzen, Feldrollen vertauschen oder Vertragseinträge überschreiben.

**Sperren/Stop:** alle bestehenden Designs, MODEL_CONTRACT, Modelle, Resultate,
Register, Verträge, Importer, Flags, Counts/Gruppen, Baseline und CLI/App
unverändert. Keine Quellenbeschaffung, Recherche, Dissertationprüfung, reale
Datenprüfung, Klima-Join, Auswahl, Bereinigung, Datenmatrix/Kohorte, Typbrücke,
Implementierung, Rechnungen, Tests, Fits, Scores, Optimierung, Simulationen
oder Kernläufe. A4 nicht umsetzen, A5/A6 nicht beginnen. Bei nötiger externer
Evidenz oder echter Datenprüfung den betroffenen Nachweis offen/blockiert
halten. Keine Gatefreigabe, Unteragenten, Autorennachrichten oder Merges.

**Prüfbares Ende:** begrenzte Klimaprovenienz-/Zeit-Nachweiskette mit genauen
Belegen und verbleibenden Lücken; keine reale Einsatzreife daraus ableiten.
Quellen-/Link-/Diff-/Blobprüfung, Statushistorie erhalten, Commit/eigener PR
ohne Merge. Genau einen begrenzten Folgeauftrag vorbereiten. Dieses neue Paket
wartet auf Nutzer-OK; die beauftragte E-Belegzuordnung ist vollständig erledigt.

**Weiter offen/blockiert:** Parnell-Originalzugang, Menezes-(2)/(5), W96,
EMPD2-Archiv-/Messstatus und reale T0/T1/T3/T4/Monticchio-P1; kein eigener
Neuheitsbeitrag abgegrenzt. Keine reale Gültigkeit/Deckung aus A3 abgeleitet.

## Kopierfertiger Startprompt für den nächsten Chat

Vorbereiteter Folgeauftrag; das Absenden dieses Prompts beauftragt nur die
beschriebene dokumentarische Präzisierung. Verlässlich gesicherter Inhaltscommit
`b6b75f8b2296b197c189327b2dac08e6949a5d61`, PR #57; danach nur
Sicherungsverweise. Aktuellen Remote-Head prüfen.

```text
Setze Netzel nach PR #57 fort.
Branch: codex/netzel-a4-e-evidence-gaps
Inhaltscommit: b6b75f8b2296b197c189327b2dac08e6949a5d61
PR: https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/57
Danach nur Sicherungsverweise; aktuellen Remote-Head prüfen.
Git-/PR-/Merge-/CI-/Branch-/Worktree-/Änderungsstand zuerst prüfen, ungesicherte
Arbeit erhalten. Eigener Worktree,
Branch und PR, kein Merge.

Hiermit beauftrage ich genau den vorbereiteten Folgeauftrag aus HANDOFF:
vorhandene Klimaprovenienz-/Zeitbezugsangaben zu E12/E13/E15 ausschließlich
dokumentarisch in A4_E_CLIMATE_TIME_GAPS.md präzisieren. Genaue Nachweiskette,
Reichweite und fehlende Verbindung dokumentieren, keinen Klimaeingang wählen.
Kein erneutes OK für dieses begrenzte Paket verlangen. Lies AGENTS, HANDOFF,
nur neuesten Status, A4_E_EVIDENCE_GAPS E12/E13/E15 und MODEL_CONTRACT zu
Klima-/Zeitrollen; Plan nur §1, §2-Zeile A4 und §4. Danach ausschließlich die
in HANDOFF festgelegten vorhandenen Abschnitte und konkrete Belegverweise.

Nur neues A4_E_CLIMATE_TIME_GAPS.md plus kurze HANDOFF-/Statuspflege. Alle
bestehenden Fachdateien unverändert. Keine externen Quellen oder realen
Datenprüfungen, Auswahl, Klima-Joins, Beschaffung, Bereinigung, Matrix/Kohorte,
Typbrücke, Recherche, Dissertationprüfung, Implementierung, Rechnungen, Tests,
Fits, Scores, Optimierung, Simulationen oder Kernläufe. Fehlende Versions-/
Perioden-/Zeitbelege und Widersprüche erhalten, keine Labels ergänzen.
A4 nicht umsetzen, A5/A6 nicht beginnen; keine Gatefreigabe. Dokumentarisch
prüfen, Commit/eigenen PR sichern, genau einen begrenzten Folgeauftrag vorbereiten.

Reale Counts-Bereitschaft bleibt blockiert; keine EMPD2-Auswahl oder historische
A/B-Entscheidung. A1/A2 dokumentarisch abgeschlossen, A3 nur endlich-synthetisch
bestanden. Parnell, Menezes-(2)/(5), W96, EMPD2-Archiv-/Messstatus und reale
Gates bleiben Grenzen; kein eigener Neuheitsbeitrag abgegrenzt. Keine
Unteragenten, Autorennachrichten oder Merges. Kurze deutsche Updates.
```
