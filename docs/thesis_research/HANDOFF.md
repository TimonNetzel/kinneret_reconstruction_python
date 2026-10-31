# Netzel: Übergabe nach dokumentarischer A4-Auftragsabgrenzung

## Gesichertes Ergebnis

Das Nutzer-OK nach [PR #55](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/55)
ist umgesetzt: [A4_TASK_BOUNDARY](./A4_TASK_BOUNDARY.md) grenzt die unveränderte
A4-Planzeile dokumentarisch ab. Eingangsrollen, belegte Struktur versus offene
reale Nachweise, Schutz-/Stopgrenzen und späteres fachliches Ende festgehalten.
**Nur Auftragsabgrenzung dokumentarisch abgeschlossen; A4-Umsetzung/reale
Kalibrierung nicht begonnen.** Kein realer Datenvertrag ausgefüllt, keine
Eingangs-/Modellwahl oder Gatefreigabe. A1/A2 dokumentarisch abgeschlossen;
A3 ausschließlich im gesperrten endlichen synthetischen Umfang bestanden,
Audit/Tabellen und sämtliche bestehenden Fachdateien unverändert. A5/A6 nicht
begonnen. Keine manuell gestarteten Tests, Rechnungen, Fits, Scores oder Läufe.

Basis `b1e78244bd8484004bc8f46de8ecbc09dadf5963`, PR #55 offen/ungemergt und
mergebar; automatische
[CI 41236453941](https://github.com/TimonNetzel/kinneret_reconstruction_python/actions/runs/41236453941)
erfolgreich. Main unverändert `7f1826c`; sechs vorgefundene Worktrees sauber,
keine Stashes/Resets/Merges. Eigener Worktree `/workspace/netzel-a4-task-boundary`,
Branch `codex/netzel-a4-task-boundary`, eigener gestapelter PR gegen #55.
Inhaltscommit `e6653da9c88a09803c9ddb68bcb5e6826359b18a` über
[PR #56](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/56)
remote gesichert, offen/ungemergt und mergebar; automatische
[CI 41237516679](https://github.com/TimonNetzel/kinneret_reconstruction_python/actions/runs/41237516679)
an diesem Commit bei Abfrage queued. Danach nur Sicherungsverweise;
aktuellen PR-Head/CI prüfen. Nur neues A4_TASK_BOUNDARY plus kurze
HANDOFF-/Statuspflege; alte Statushistorie erhalten.

## Genau ein Folgeauftrag: vorhandene EMPD2-Belege zu modernen E-Lücken zuordnen – vorbereitet, wartet auf Nutzer-OK

**Ziel:** ausschließlich vorhandene dokumentarische EMPD2-Archiv-/Messstatusbelege
auf die offenen E-Anforderungen der A4-Auftragsgrenze abbilden. Der Prüfgegenstand
ist eine vorhandene Dokumentation; EMPD2 wird damit nicht als moderner Eingang
gewählt oder freigegeben. Keine neue Quelle beschaffen oder reale Datei als
Datenmaterial öffnen. Das jetzige OK autorisierte nur die erledigte Abgrenzung.

**Eingänge/Leseplan:** zuerst Git-/PR-/Merge-/CI-/Branch-/Worktree-/Änderungsstand,
ungesicherte Arbeit erhalten; eigener Worktree/Branch/PR ohne Merge. AGENTS,
diese HANDOFF, nur neuester PROJECT_STATUS, A4_TASK_BOUNDARY und MODEL_CONTRACT;
Plan nur §1, §2-Zeile A4 und §4. Danach gezielt vorhandene Abschnitte in:

- [MODERN_COUNTS_READINESS](./empd2_counts_readiness_2026_10_31/MODERN_COUNTS_READINESS.md)
  und [INPUT_CONTRACT](./empd2_counts_readiness_2026_10_31/INPUT_CONTRACT.json);
- [ARCHIVE_LINEAGE](./empd2_archive_lineage_2026_10_31/ARCHIVE_LINEAGE.md)
  und [DECISION_GAPS](./empd2_archive_lineage_2026_10_31/DECISION_GAPS.md).

Nur bei konkreter dort belegter Zuordnungsfrage die einschlägige Stelle in
[COUNTS_DECISION_BRIEF](./empd2_counts_decision_2026_10_31/COUNTS_DECISION_BRIEF.md)
oder [DENOMINATOR_RULES](./empd2_denominator_rules_2026_10_31/DENOMINATOR_RULES.md)
nachladen; zusätzliche bestehende Evidenz-JSON nur wenn ein bestimmter Beleg
sonst nicht prüfbar ist. Keine Neu-Inventur von Daten/Taxa, Tabellen-/Archivdateien,
Literatur-/Archivsuche oder Dissertationprüfung. NOVELTY_MATRIX und A3-Details
nur bei konkreter Beleg-/Abnahmefrage, kein erneuter A3-Lauf. Diese älteren
EMPD2-Dokumente wurden im jetzigen Abgrenzungspaket nicht erneut fachlich geprüft.

**Erlaubter Umfang:** genau ein neues Dokument
`docs/thesis_research/A4_E_EVIDENCE_GAPS.md` und kurze HANDOFF-/Statuspflege.
Eine kompakte Belegmatrix mit E-Anforderung, vorhandener Aussage samt exaktem
Datei-/Abschnittsbezug, Reichweite, Status **belegt / offen / widersprüchlich /
nicht geprüft** und konkret fehlendem Nachweis erstellen. Messbeleg, Nenner,
Typabdeckung, Null-/Missingness-/Prozentstatus, Klima-/Zeit-/Ortsbezug und
Herkunft getrennt ausweisen, soweit die festgelegten Eingänge dazu Aussagen
tragen. Keine Vollständigkeit außerhalb dieses Dokumentumfangs behaupten.
Abgeleitete E-Anforderungen den bestehenden Vertragsstellen zuordnen; keinen
Vertrag ändern und kein OFFEN-Feld durch eine Annahme füllen.

**Sperren/Stop:** alle bestehenden Designs, MODEL_CONTRACT, Modelle, Resultate,
Register, Verträge, Importer, Flags, Counts/Gruppen, Baseline und CLI/App
unverändert. Keine reale Auswahl, Beschaffung, Bereinigung, Datenmatrix/Kohorte,
Typbrücke, Implementierung, neue Rechnung, Tests, Fits, Scores, Optimierung,
Zufallssimulationen oder Kernläufe. A4 nicht umsetzen, A5/A6 nicht beginnen;
keine Gatefreigabe, Unteragenten, Autorennachrichten oder Merges. Benötigt eine
Belegfrage neue externe Evidenz oder echte Datenprüfung, die betroffene Zeile
als offen/blockiert ausweisen. Widersprüche erhalten, nicht still auflösen.

**Prüfbares Ende:** nachvollziehbare Belegmatrix innerhalb der festgelegten
Dokumente; verfügbare Aussage und fehlender Nachweis klar unterschieden,
Widersprüche und unprüfbare Felder vollständig berichtet. Quellen-/Link-/Diff-/
Blobprüfung, alte Statushistorie erhalten; Commit/eigener PR ohne Merge.
Genau einen anschließenden begrenzten Auftrag vorbereiten. Dieses neue Paket
wartet auf Nutzer-OK; die jetzt autorisierte Abgrenzung ist vollständig erledigt.

**Weiter offen/blockiert:** Parnell-Originalzugang, Menezes-(2)/(5), W96,
EMPD2-Archiv-/Messstatus, reale T0/T1/T3/T4/Monticchio-P1; kein eigener
Neuheitsbeitrag abgegrenzt. Keine reale Gültigkeit/Deckung aus A3 abgeleitet.

## Kopierfertiger Startprompt für den nächsten Chat

Vorbereiteter Folgeauftrag; das Absenden dieses Prompts autorisiert nur die
hier begrenzte dokumentarische Belegzuordnung. Verlässlich gesicherter
Inhaltscommit `e6653da9c88a09803c9ddb68bcb5e6826359b18a`, PR #56;
danach nur Sicherungsverweise. Aktuellen Remote-Head prüfen.

```text
Setze Netzel nach PR #56 fort.
Branch: codex/netzel-a4-task-boundary
Inhaltscommit: e6653da9c88a09803c9ddb68bcb5e6826359b18a
PR: https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/56
Danach nur Sicherungsverweise; aktuellen Remote-Head prüfen.
Prüfe Git-/PR-/Merge-/CI-/Branch-/Worktree-/Änderungsstand zuerst, erhalte
ungesicherte Arbeit. Eigener Worktree,
Branch und PR; kein Merge.

Hiermit beauftrage ich genau den vorbereiteten Folgeauftrag aus HANDOFF:
vorhandene EMPD2-Archiv-/Messstatusdokumentation auf offene moderne E-Anforderungen
abbilden, ausschließlich dokumentarisch in A4_E_EVIDENCE_GAPS.md, mit exakten
Belegen, Reichweite, Status und fehlenden Nachweisen. EMPD2 nicht als Eingang
wählen oder freigeben. Kein erneutes OK für dieses begrenzte Paket verlangen.
Lies AGENTS, HANDOFF, nur neuesten Status, A4_TASK_BOUNDARY und MODEL_CONTRACT;
Plan nur §1, §2-Zeile A4 und §4; die in HANDOFF festgelegten bestehenden EMPD2-
Dokumente gezielt lesen, weitere Belege nur bei dort konkreter Zuordnungsfrage.

Nur neues A4_E_EVIDENCE_GAPS.md und kurze HANDOFF-/Statuspflege. Alle bestehenden
Fachdateien unverändert. Keine neuen Quellen, realen Datenprüfungen, Auswahl,
Beschaffung, Bereinigung, Matrix/Kohorte, Typbrücke, Recherche, Dissertation-
prüfung, Implementierung, Rechnungen, Tests, Fits, Scores, Optimierung,
Simulationen oder Kernläufe. A4 nicht umsetzen, A5/A6 nicht beginnen; keine
Gatefreigabe. Fehlende Belege/Widersprüche erhalten; bei nötiger Erweiterung
betroffenen Teil offen/blockiert ausweisen. Nur dokumentarisch prüfen, Commit
und eigenen PR sichern, genau einen begrenzten Folgeauftrag vorbereiten.

A1/A2 dokumentarisch abgeschlossen; A3 nur endlich-synthetisch bestanden.
Parnell, Menezes-(2)/(5), W96, EMPD2-Archiv-/Messstatus und reale Gates bleiben
Grenzen; kein eigener Neuheitsbeitrag abgegrenzt. Keine Unteragenten,
Autorennachrichten oder Merges. Kurze deutsche Updates.
```
