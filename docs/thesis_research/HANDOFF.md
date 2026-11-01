# Netzel: automatische Folgeprompts und blockierte Herkunftsbelegbewertung

## Aktueller Arbeitsablauf: Folgeprompt bei jedem gesicherten Abschluss

Am 01.11.2026 auf Nutzerfrage die Regel aus
[PR #60](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/60)
übernommen und in [AGENTS.md](../../AGENTS.md) auf jeden gesicherten
Teilaufgabenabschluss erweitert. #60 ist offen/ungemergt; der #61-Branch
enthielt die Regel deshalb nicht. Prompt künftig automatisch aus dem aktuellen
HANDOFF erzeugen, vollständig im Chat ausgeben und hier speichern. Keine neue
Nachfrage nötig; neuer Chat muss den Branch mit dieser Regel verwenden.

Basis `7630c480a71849a45a4e58646be2e2ffa7d5e431`,
[PR #61](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/61)
offen/ungemergt und mergebar, automatische
[Head-CI 41284948221](https://github.com/TimonNetzel/kinneret_reconstruction_python/actions/runs/41284948221)
erfolgreich. Separater Worktree `/workspace/netzel-followup-prompt-continuation`,
Branch `codex/netzel-followup-prompt-continuation`, eigener PR gegen #61 ohne Merge.
Nur AGENTS und kurze HANDOFF-/Statuspflege; kein fachliches Paket ausgeführt.
Der genau eine fachliche Folgeauftrag unten bleibt am fehlenden Belegobjekt
blockiert. Sicherungsverweis nach Erstellung ergänzen.

## Gesichertes Ergebnis

[A4_E_PROVENANCE_EVIDENCE_REVIEW](./A4_E_PROVENANCE_EVIDENCE_REVIEW.md)
dokumentiert den fehlenden Eingang: Der Nutzerauftrag enthält nur den
Platzhalter für Pfad/Ref und Archivbezug. **Stand-/Eingangsprüfung abgeschlossen;
inhaltliche Bewertung blockiert.** R01–R03 sind mangels Eingang unbelegt und
nicht inhaltlich geprüft; keine Bewertung eines tatsächlich vorgelegten Dokuments.
Das Nutzer-OK liegt vor, ersetzt aber kein Belegobjekt. Reale Klima-/Zeitverbindung
und Counts-Bereitschaft bleiben blockiert; keine Gatefreigabe.

Basis `02e004cea401f19146bee6b23cc97d4a982051d1`,
[PR #59](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/59)
offen/ungemergt und mergebar; Head entspricht dem bestätigten Sicherungscommit.
Automatische [Head-CI 41240408738](https://github.com/TimonNetzel/kinneret_reconstruction_python/actions/runs/41240408738)
erfolgreich; separate Commitstatus-Liste leer. #58 ebenfalls offen/ungemergt
und mergebar, Head `0ac78de8ec007bb8e92c5ea571cefc7ba9e414a0`.
#59-Head nicht in main; main unverändert `7f1826cffddc8893ba6354cd3cca3fa43954b9be`.
Fünf vorgefundene Worktrees sauber und erhalten, keine ungesicherten Änderungen
oder Stashes vorgefunden; kein Reset/Stash/Merge. Eigener Worktree
`/workspace/netzel-a4-e-provenance-evidence-review`, Branch
`codex/netzel-a4-e-provenance-evidence-review`; eigener gestapelter PR gegen #59.
Inhaltscommit `921b5ccb5b1bdb0cb7b1fe20c3e12d7ea254ea7f` über
[PR #61](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/61)
remote gesichert, offen/ungemergt und mergebar. Automatische
[CI 41284909253](https://github.com/TimonNetzel/kinneret_reconstruction_python/actions/runs/41284909253)
an diesem Inhaltscommit bei Abfrage eingereiht; keine manuellen Tests ausgeführt.
Danach nur Sicherungsverweise; aktuellen PR-Head/CI prüfen.

Gelesen: AGENTS, HANDOFF, nur oberster aktueller PROJECT_STATUS-Abschnitt,
R01–R03 samt Herkunfts-/Stopregeln, MODEL_CONTRACT Rollen/Stop sowie Plan §1,
§2 nur A4 und §4. Ein Tabellenfilter zeigte zusätzlich R04–R06 an;
daraus keine Bewertung oder neue fachliche Prüfung abgeleitet.
Kein Herkunftsdokument und keine Altquelle als Ersatz geöffnet.
Nur neues Ergebnisdokument und kurze HANDOFF-/Statuspflege; bestehende
Fachdateien und bisherige Statushistorie erhalten. Dokumentarische Abnahme
betrifft den fehlenden Eingang, nicht die weiter blockierte Inhaltsbewertung.
Quellenbezüge, elf lokale Links/Anker und Drei-Dateien-Diff geprüft; 753
geschützte Git-Blobs/Dateimodi einschließlich Symlink unverändert, alte
Statushistorie bytegleich erhalten. Automatische Repository-CI separat ausweisen.

## Genau ein Folgeauftrag: Herkunftsbelegbewertung R01–R03 nach eindeutigem Eingang fortsetzen – vorbereitet, am Eingang blockiert

**Ziel:** ausschließlich den begrenzten Nachweisbeitrag eines einzelnen,
bereits bereitgestellten Herkunfts-/Erzeugungs-Textdokuments bewerten.
Der Prüfauftrag ist bereits autorisiert; kein erneutes OK für dieses Paket
nötig. Bis zur eindeutigen Bezeichnung des Belegobjekts nicht fortsetzen.

**Nötiger Eingang:** genau ein vom Nutzer bezeichneter lokaler oder versionierter
Repository-Text mit Pfad/Ref und beanspruchter Archividentität. Kein bloßer
Platzhalter. Keine eigene Dokument-/Archiv-/Versions-/Perioden-/Probenwahl,
Ersatzsuche, Beschaffung oder Beleganforderung; keine Datenbank, Rohdaten,
Raster oder Archive öffnen. Ein weiteres OK allein schließt die Eingangslücke nicht.

**Leseplan:** Git-/PR-/Merge-/CI-/Branch-/Worktree-/Änderungsstand prüfen,
ungesicherte Arbeit erhalten; eigener Worktree/Branch/PR, kein Merge.
AGENTS, diese HANDOFF, nur oberster aktueller PROJECT_STATUS-Abschnitt,
A4_E_PROVENANCE_EVIDENCE_REVIEW und A4_E_REOPENING_REQUIREMENTS nur R01–R03
samt Herkunfts-/Stopregeln; MODEL_CONTRACT Rollen/Stop; Plan §1, §2 nur A4,
§4. Danach nur relevante Passagen des genau bezeichneten Belegobjekts.
A4_E_CLIMATE_TIME_GAPS C04–C06 nur bei konkreter Rückfrage zum Altstand.
Keine Original-/Archiv-/Literatur-/Berichtssuche oder erneute Altbelegbewertung.

**Umfang:** ausschließlich A4_E_PROVENANCE_EVIDENCE_REVIEW.md mit dem tatsächlichen
Eingang fortschreiben, dazu kurze HANDOFF-/Statuspflege. Kein zweites
Ergebnisdokument oder allgemeines Kriterienpapier. Für R01–R03 jeweils genaue
Fundstelle, Originalaussage/Übernahme, Herkunft, Eingangs-/Archiv-/Feldidentität,
Reichweite und Status belegt/teilweise belegt/unbelegt ausweisen. Fehlendes und
Widersprüche offen halten; ein Dokument muss nicht alle Anforderungen erfüllen.
Keinen realen Vertrag ausfüllen, keine Version/Periode/Probe auswählen.

**Sperren/Stop:** R04–R06 und alle übrigen Gates bleiben außerhalb des Pakets offen.
Alle bestehenden Fachdateien einschließlich Kriterien/Vertrag/Designs/Modelle/
Resultate/Register/Importer/Flags/Counts/Gruppen/Baseline/CLI/App unverändert.
Keine Recherche/Beschaffung, realen Daten-/Dateiprüfungen, Daten-/Modellwahl,
Klima-Joins, Bereinigung, Matrix/Kohorte, Typbrücke, Implementierung, Tests,
Rechnungen, Fits, Scores, Optimierung, Simulationen oder Kernläufe.
Bei weiterem Objektbedarf oder realer Entscheidung nur die Lücke dokumentieren.
A4 nicht umsetzen, A5/A6 nicht beginnen; keine Gatefreigabe, Unteragenten,
Autorennachrichten, versandte Beleganforderung oder Merges.

**Abnahme:** nachvollziehbarer Beitrag genau dieses Dokuments bei unveränderten
R01–R03; auch vollständiger Nichtbeleg ist zulässig. Keine Gesamtfreigabe oder
Probenzeiteignung. Quellen-/Link-/Diff-/Blobprüfung, Statushistorie erhalten,
Commit/eigener PR ohne Merge und genau einen begrenzten Folgeauftrag vorbereiten.

**Weiter offen/blockiert:** reale Klima-/Zeitverbindung, Counts-Bereitschaft,
Parnell-Zugang, Menezes-(2)/(5), W96, EMPD2-Archiv-/Messstatus und reale
T0/T1/T3/T4/Monticchio-P1; kein eigener Neuheitsbeitrag abgegrenzt.
Keine reale Gültigkeit/Deckung aus A3 abgeleitet.

## Kopierfertiger Startprompt für die Fortsetzung

```text
Setze Netzel auf codex/netzel-followup-prompt-continuation fort.
Gesicherter Commit/PR: siehe aktueller Sicherungsverweis oben.
Aktuellen Remote-Head/CI prüfen. AGENTS.md enthält die automatische Promptregel.
Fertig: Stand-/Eingangsprüfung und verbindliche Regel für Folgeprompts.
Vorgänger-Head-CI #61 erfolgreich; aktuelle eigene CI gesondert prüfen.
Inhaltliche R01–R03-Bewertung weiterhin am fehlenden Belegobjekt blockiert.
Genau ein Folgeauftrag: die bereits autorisierte Herkunftsbelegbewertung
R01–R03 nach Eingang fortsetzen. Kein erneutes OK für dieses Paket nötig.
Belegobjekt (Pfad/Ref und Archivbezug): noch eindeutig vom Nutzer zu benennen.
Ohne eindeutiges Belegobjekt blockiert bleiben; nur Stand/fehlenden Eingang
melden, keine wiederholten Dokumentationspakete ohne neue Evidenz erzeugen.
Keine Ersatzrecherche/Beschaffung oder Beleganforderung.
Git-/PR-/Merge-/CI-/Branch-/Worktree-/Änderungsstand prüfen, Arbeit erhalten.
Eigener Worktree/Branch/PR ohne Merge. Leseplan, Umfang, Abnahme und Stop aus
HANDOFF einhalten. Nur A4_E_PROVENANCE_EVIDENCE_REVIEW.md fortschreiben plus
kurze HANDOFF-/Statuspflege; bestehende Fachdateien unverändert.
Keine Recherche, Beschaffung, realen Daten-/Dateiprüfungen, Auswahl, Klima-Joins,
Implementierung, Tests oder Rechnungen. R04–R06/übrige Gates offen lassen.
A4 nicht umsetzen, A5/A6 nicht beginnen. Keine Gatefreigabe, Unteragenten,
Autorennachrichten oder versandte Beleganforderung. Dokumentarisch prüfen,
Abnahme: R01–R03 mit Fundstelle, Originalaussage/Übernahme, Identitätsbezug,
Reichweite und Status belegt/teilweise belegt/unbelegt nachvollziehbar bewerten;
Quellen-/Link-/Diff-/Blobprüfung und Statushistorie erhalten.
Commit/eigenen PR sichern und genau einen begrenzten Folgeauftrag vorbereiten.
Nach jedem gesicherten Teilaufgabenabschluss den Folgeprompt automatisch aus
dem aktuellen HANDOFF erzeugen, dort speichern und hier vollständig als
kopierfertigen Textblock ausgeben; diese Anweisung in den Folgeprompt übernehmen.
Kurze deutsche Updates.
```
