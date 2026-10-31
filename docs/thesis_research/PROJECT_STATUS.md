# Netzel-Projekt: belegter aktueller Stand

Standprüfung 31.10.2026. Ziel bleibt eine überprüfbare taxonbasierte
Multiproxy-Klimarekonstruktion; kein fertiger fossiler Gesamtdurchlauf.
Arbeitsregeln stehen ausschließlich in [AGENTS.md](../../AGENTS.md),
der nächste Auftrag und Leseplan in [HANDOFF.md](./HANDOFF.md).

Gemeinsamer nächster Meilenstein, am 31.10.2026 vom Nutzer festgehalten:
**erster diagnostischer lokaler Kernlauf mit verbessertem Taxonverfahren**.
Die [Roadmap](./TAXON_PILOT_ROADMAP.md) verbindet die vorhandenen Bausteine mit
Bestimmungs-/Abdeckungsprüfung, vorab festgelegter Rekonstruktionsregel,
synthetischer Prüfung und gesperrten realen Eingängen. Kernwahl und konkrete
Modellvariante bleiben offen; Monticchio-P1 unverändert verbindlich. Keine
Voraussetzung, zuerst alle 279 historischen Funktionen oder die App fertigzustellen.
Im Roadmap-Paket #36 nur den Weg dokumentiert, keine neuen Modelle oder Läufe.
Sicherung als [PR #36](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/36)
auf `codex/netzel-first-taxon-pilot-roadmap` ab `b9c7932` (#35),
gegen den #35-Branch; ohne Merge erstellt.

## Aktueller Paketabschluss: EMPD2-Archivherkunft, 31.10.2026

Nach Nutzerauftrag **begrenzt abgeschlossen; exakte Erzeugungskette offen**:
[ARCHIVE_LINEAGE](./empd2_archive_lineage_2026_10_31/ARCHIVE_LINEAGE.md),
[LINEAGE_EVIDENCE](./empd2_archive_lineage_2026_10_31/LINEAGE_EVIDENCE.json),
[DECISION_GAPS](./empd2_archive_lineage_2026_10_31/DECISION_GAPS.md).
Offizielle Archivmetadaten ordnen die drei Originalformate demselben DOI zu.
Nur der dort ausdrücklich verlinkte PostgreSQL-Dump als zweites Originalobjekt
begrenzt geprüft: HTTP-Header und Bytebereich 0–32767, kein vollständiger Dump.
PostgreSQL/pg_dump 11.2 und gleitkommafähiges count-Feld belegt; keine angewandte
Code-/Exportversion zur gesperrten SQLite nachgewiesen. Suche nach zwei Quellen
beendet, keine Ersatzsuche. Fehlender Messstatus/Nullregel, N_original und
Hierarchieanwendung bleiben offen; S_flag und n_ausgewählt nicht gleichsetzen.

Basis `90d0350f20d8adc5d1feab09722cf02f0ebb8b69` aus PR #43; eigener Branch
`codex/netzel-empd2-archive-lineage`, Worktree `/workspace/netzel-empd2-lineage`.
[PR #44](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/44),
gestapelt gegen den #43-Branch, offen/ungemergt. Inhaltscommit
`35152172f8f048a755bd5897d1687c888e9917d2`; danach Sicherungs-/Übergabeverweise.
717 geschützte bestehende Git-Blobs bytegleich, JSON/Links/Hashes/Diff geprüft.
Bei PR-Erstellung technisch mergefähig; automatische PR-CI am Inhaltscommit
noch laufend (Run 41197720179), keine grüne Gesamtfreigabe behauptet.
Einstieg #31–#43 offen/ungemergt, main `7f1826c`, #43-PR-CI erfolgreich.
Fünf hier vorhandene vorherige Worktrees sauber erhalten. Vorhandene SQLite
nur gehasht, nicht geöffnet oder neu geladen; 21.327.872 Bytes/Sperrhash bestätigt.
Keine neuen Zeugen gelesen, bekannte #43-Befunde unverändert übernommen.
Keine neue Matrix/Kohorte, Bereinigung, Projekt-/Quellcodeausführung, Fits,
Scores, Simulationen, Tests oder Kernläufe; Designs/Modelle/Resultate/Register/
Importer/Flags/Counts/Gruppen, Baseline und CLI/App unverändert. Keine
Unteragenten, Autorennachrichten oder Merges. W96 und reale Gates weiter offen.

**Genau ein Folgeauftrag vorbereitet, wartet auf Nutzer-OK:** dokumentarische
Entscheidungsvorlage zum blockierten realen EMPD2-Counts-Einstieg anhand
vorhandener Evidenz; keine neue Suche, Zieländerung, Implementierung oder
Freigabe. [Umfang und Leseplan](./HANDOFF.md).

## Vorheriger Paketabschluss: EMPD2-Nenner-/Sparse-Quellen, 31.10.2026

Nach Nutzer-OK hier im Chat **als begrenztes Quellenpaket mit Teilbefunden erledigt**:
[DENOMINATOR_RULES](./empd2_denominator_rules_2026_10_31/DENOMINATOR_RULES.md),
[RULE_EVIDENCE](./empd2_denominator_rules_2026_10_31/RULE_EVIDENCE.json),
[Quellenvergleich](./empd2_denominator_rules_2026_10_31/source_comparison.json).
Originalmethodik gelesen, offizielles EMPD2/EMPD-data v2.0 am Commit
`737b76eb269b071422edded6ac332654ff83c103` geprüft, zwölf Dateien per Git-Blobhash
verifiziert, nicht ausgeführt. Drei bekannte Zeugen: alle 108 Messwert-Tupel
zwischen Release und Sperr-SQLite identisch; 98 bereitgestellte Prozentwerte
innerhalb gedruckter Präzision kompatibel mit S_flag=311/731/840, zehn leere
Prozentfelder passen zu percent_values=False. Keine Prozent→Counts-Rückrechnung.

**Nicht geklärt:** ursprünglicher gemessener N_original, allgemeine archiveigene
Sparse-Nullregel, rekursive higher_groupid-Aggregation und exakter Erzeugungs-/
Exportpfad der gesperrten SQLite-Datei. Neuanlagecode rundet, bestehende Proben
werden übersprungen; erhaltenes 7,4 und 36/40 Gruppen begrenzen den Transfer.
Weder komplette Code→Archiv-Kette noch „Code nie benutzt“ behauptet.
Klimanormalperiode unverifiziert; keine Nullfüllung/Rundung, Trainings-/Pilotfreigabe.
W96/Artbrücken, reales T1/T0/T3/T4 und Monticchio-P1 unverändert offen.

Basis PR #42 bei `e1e8ebe32f17487c4c91d2044db845112a8365bb`, Branch
`codex/netzel-empd2-denominator-rules`, Worktree `/workspace/netzel-empd2-rules`.
Neun bisherige Worktrees sauber erhalten; #31–#42 offen, main `7f1826c`,
#42-CI grün. [Standprüfung](./empd2_denominator_rules_2026_10_31/git_snapshot.json).
Keine Matrix/Kohorte, Fits, Scores, Simulationen oder Tests; Originaldatenbank-
hash unverändert. Alle bestehenden Designs/Modelle/Resultate/Register/Importer/
Flags/Fossilcounts/Gruppen, Baseline und CLI/App unverändert. Keine Unteragenten,
Merges oder Kontakte. [Prüfung](./empd2_denominator_rules_2026_10_31/verification.json).
Sicherung als [PR #43](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/43)
gegen `codex/netzel-empd2-counts-readiness` (#42), offen, kein Merge.
Letzter gesicherter **Inhaltscommit**:
`ed979ba2dee39f2fc1b503f9b09353812f9fbcc4`; danach nur PR-/Übergabeverweise.
709 geschützte bestehende Git-Blobs bytegleich, neun vorherige Worktrees sauber.
Eigene CI bei Erstellung noch laufend, technisch MERGEABLE/UNSTABLE;
[PR-Snapshot](./empd2_denominator_rules_2026_10_31/pr_snapshot.json).
Keine behauptete fachliche oder grüne Gesamtfreigabe.

**Damals vorbereitet, inzwischen nach Nutzerauftrag begrenzt abgeschlossen:** enges Herkunfts-/Exportaudit genau der
Sperr-SQLite anhand offizieller PANGAEA-Metadaten und nur bei Verweis eines
zweiten Originalobjekts. Nach zwei gezielten Quellen ohne Kette einen
Entscheidungspunkt dokumentieren, keine weitere Ersatzsuche.
[Restbedarf](./empd2_denominator_rules_2026_10_31/SOURCE_GAPS.md),
[genauer Folgeauftrag](./HANDOFF.md).

## Vorheriger Paketabschluss: EMPD2-Counts-Readiness-Audit, 31.10.2026

Nach Nutzer-OK hier im Chat **lesend abgeschlossen**:
[Ergebnis](./empd2_counts_readiness_2026_10_31/MODERN_COUNTS_READINESS.md),
[INPUT_CONTRACT](./empd2_counts_readiness_2026_10_31/INPUT_CONTRACT.json),
[Originalbelege](./empd2_counts_readiness_2026_10_31/evidence.json).
Cached EMPD2.sqlite3 vor/nach read-only-Zugriff identischer Sperrhash/21.327.872
Bytes; kein neuer Download. Schema/Metadaten und genau drei Zeugen mit 108
verschiedenen gespeicherten Messzeilen gelesen, kein kompletter Matrixexport.

**Technische Eingänge vorhanden, Counts-Trainingsbereitschaft noch nicht belegt.**
Kein eigenes Proben-N im Schema, Sparse-Nichttreffer unaufgelöst; Gruppenflags
sind kein Nachweis der tatsächlich angewandten Nenner-/Hierarchieregeln.
Champvillair_a18 enthält 7,4 als Volumen cm³ außerhalb used_in_sum=1, bei
arithmetischer flag-1-Summe 731. Historische 244 fractional_counts-Proben sind
nicht automatisch 244 Proben mit gebrochenen Pollenzählungen; historische
1.059/244-Werte nicht neu berechnet, Flags/Importer/Referenzen unverändert.
Gesamtsumme aller gespeicherten Werte kann physikalisch heterogene Größen
addieren und ist kein N. Text-NULL/X nicht repariert, Klima-Periode unverified,
Typ-/Fossilbrücken offen. Keine neue Kohorte/Matrix, Fits/Tests/Simulationen,
Scores oder Kernläufe. Nur Quellen-/Dokumentations-/Link-/Hash-/Diffprüfung.
[Prüfprotokoll](./empd2_counts_readiness_2026_10_31/verification.json).

Basis PR #41 bei `62ffeaf08ee7c7538f8d9c4cc99f78e7c0bdb394`;
Branch `codex/netzel-empd2-counts-readiness`, Worktree
`/workspace/netzel-empd2-readiness`. Acht bisherige Worktrees sauber erhalten;
#31–#41 offen, main `7f1826c`, #41-CI grün.
[Standprüfung](./empd2_counts_readiness_2026_10_31/git_snapshot.json).
Bestehende Modelle/Designs/Resultate/Register/Fossilcounts/Gruppen, Baseline
und CLI/App bytegleich; keine Unteragenten, Merges oder Autorennachrichten.
Reales T1, W96/Artbrücken, T0/T3/T4 und Monticchio-P1 unverändert offen.
Sicherung als [PR #42](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/42)
gegen `codex/netzel-synthetic-counts-t2` (#41), offen, kein Merge.
Letzter gesicherter **Inhaltscommit**:
`1627a11d22f6fe8827dd7eed509acb6df2720b08`; danach nur PR-/Übergabeverweise.
701 geschützte bestehende Git-Blobs bytegleich und acht vorherige Worktrees sauber;
Originaldatenbankhash unverändert. Eigene CI bei Erstellung noch laufend,
technisch MERGEABLE/UNSTABLE; [PR-Snapshot](./empd2_counts_readiness_2026_10_31/pr_snapshot.json).
Kein behaupteter grüner Gesamtstand und keine wissenschaftliche Freigabe.

**Damals vorbereitet, inzwischen nach Nutzer-OK geprüft:** ursprüngliche EMPD2-Nenner- und Sparse-
Verarbeitungsregeln anhand der registrierten Methodenpublikation und nur bei
konkretem offiziellen Verweis eines zweiten Originalobjekts belegen.
[Enger Quellenbedarf](./empd2_counts_readiness_2026_10_31/SOURCE_GAPS.md),
[genauer Auftrag und Leseplan](./HANDOFF.md). Keine Matrix-/Modellläufe daraus.

## Vorheriger Paketabschluss: synthetisches Counts-T2, 31.10.2026

Nach Nutzer-OK hier im Chat **begrenzt abgeschlossen**:
[52 anonyme Counts-Auswertungen](./counts_synthetic_t2_2026_10_31/README.md),
25 künstliche Datensätze, zwei zusätzliche unmögliche Fälle nicht schätzbar.
51 neue gezielte Tests bestanden; alle 52×441 Massen separat nachgerechnet
(max. Differenz 1,263×10⁻¹⁵), zwölf PMF-Normierungen per Enumeration einschließlich
je 495 Countvektoren bei N=8 geprüft, fünf Produktionsdateien im Zweitlauf
bytegleich. Code/Test/Prüfer vor erstem Ergebnislauf bei `5b7e67a` gesichert,
ein Test-Syntaxfehler vor Produktion in `220ef55` repariert.
[Prüfprotokoll](./counts_synthetic_t2_2026_10_31/verification.json),
[Tests](./counts_synthetic_t2_2026_10_31/tests.json),
[Wiederholung](./counts_synthetic_t2_2026_10_31/repeat_verification.json).
Keine wissenschaftlichen Parameter/Seeds/Kriterien nach Ergebnis angepasst.

Base-Mittelfehler N=8→40 uniform 0,36575→0,19421, informativ 0,38615→0,19704.
Informativer Prior im Mittel schlechter, kleinere HPD-Sets; eine bekannte Wahrheit
verfehlt HPD. Im einzelnen shared-Datensatz liefert die missspezifizierte
je-Korn-Inferenz kleinere Mittelfehler als das Daten erzeugende gemeinsame Modell;
alle Varianten berichtet, keine Modellwahl daraus. Typzusammenlegung bleibt
strukturell ±u-mehrdeutig; fehlende Beobachtung/N=0 prior-only, unmögliche Fälle
explizit ohne Posterior. Kein SBC-/Deckungsnachweis und keine echte Artfreigabe.

Basis PR #40 bei `19a75426320d0143c09cd00c756bc31f97cef0d0`;
eigener Branch `codex/netzel-synthetic-counts-t2`, Worktree
`/workspace/netzel-counts-t2`. Sieben vorhandene Worktrees sauber erhalten;
#31–#40 offen, main `7f1826c`, #40-CI grün.
[Standprüfung](./counts_synthetic_t2_2026_10_31/git_snapshot.json).
Keine ganzen Repository-/Browser-/Sampler-Tests, echten Modelle, realen Fits
oder Kernläufe. Modelle, alte Designs/Resultate/Register/Fossilcounts/Gruppen,
Baseline und CLI/App unverändert; keine Unteragenten, Kontakte oder Merges.
Reales T1, W96/Artbrücken, T0/T3/T4/Monticchio-P1 bleiben offen.
Sicherung als [PR #41](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/41)
gegen `codex/netzel-counts-observation-design` (#40), offen, kein Merge.
Letzter gesicherter **Inhaltscommit**:
`fa61c113904429f6b89b9ce6beca3d07fdeea749`; danach nur PR-/Übergabeverweise.
682 geschützte vorhandene Git-Blobs bytegleich, sieben vorherige Worktrees
weiterhin sauber; [Erhaltungsprüfung](./counts_synthetic_t2_2026_10_31/scope_verification.json).
Eigene CI bei PR-Erstellung noch laufend, technisch MERGEABLE/UNSTABLE;
kein behaupteter grüner Gesamtstand. [PR-Snapshot](./counts_synthetic_t2_2026_10_31/pr_snapshot.json).

**Damals vorbereitet, inzwischen nach Nutzer-OK ausgeführt:** lesendes Audit eines
modernen Counts-Kalibrierungswegs an EMPD2, ohne Fits oder neue Auswertungsmatrix.
Konkrete Eingänge, Dateien und Grenzen in [HANDOFF](./HANDOFF.md).

## Vorheriger Paketabschluss: T1-Zählkomposition/Typgemische, 31.10.2026

Nach Nutzer-OK hier im Chat **als begrenztes Designpaket abgeschlossen**:
[Ergebnis](./counts_observation_design_2026_10_31/README.md),
[DESIGN_COUNTS](./counts_observation_design_2026_10_31/DESIGN_COUNTS.md) und
[INPUT_GAPS](./counts_observation_design_2026_10_31/INPUT_GAPS.md).
Vegetation→Produktion/Transport/Erhaltung→Typzuordnung→Counts mathematisch
getrennt, Multinomial-Likelihood/latente Mischung/Normierung und Nennervertrag
festgelegt. Algebraische Gegenbeispiele belegen nicht identifizierbare
Flussskala, Produktions-/Klimakompensation und untrennbare Typgemische.
Proper Priors ersetzen diese fehlende Dateninformation nicht.

**Damals vorbereitet, inzwischen nach Nutzer-OK ausgeführt:** kleiner anonymer Zählprototyp mit
52 festgelegten Posterioren und Struktur-/Negativfällen, nach
[SYNTHETIC_PLAN](./counts_observation_design_2026_10_31/SYNTHETIC_PLAN.md) und
[JSON](./counts_observation_design_2026_10_31/synthetic_plan.json).
Der gemeinsame-H-Vergleich hält Einzelkorn-Marginalen gleich.
**Hier nicht implementiert oder ausgeführt:** keine Simulationen, Fits,
Posterioren, Tests oder realen Kernläufe. Nur Dokumentationsprüfung:
Gleichungen/Gegenbeispiele, Nenner/Rest, Inventar, lokale Links,
Eingangshashes, Erhaltungsprüfung und begrenzter Diff.
[Prüfprotokoll](./counts_observation_design_2026_10_31/verification.json).

Reales T1, T0/Artbrücken, W96, T3/T4 und Monticchio-P1 bleiben offen.
Bestehende Designs, Modelle, Resultate/Quellregister, Fossilcounts/Gruppierungen,
Baseline und CLI/App unverändert. Eigener Worktree `/workspace/netzel-counts-design`,
Branch `codex/netzel-counts-observation-design` ab
`35185534bf7f171c6f39537ebbc0449b28f5421a` aus offenem PR #39.
Alle sechs vorhandenen Worktrees bei Einstieg sauber und erhalten;
main weiterhin `7f1826c`, #31–#39 offen. Keine Unteragenten, Merges oder
Autorennachrichten. [Standprüfung](./counts_observation_design_2026_10_31/git_snapshot.json).
Sicherung als [PR #40](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/40)
gegen `codex/netzel-t1-synthetic-t2` (#39), offen und ohne Merge.
Letzter gesicherter **Inhaltscommit**:
`62e0ee6272323246f6d29eff5f390a12136f116b`; danach nur PR-/Übergabeverweise.
674 geschützte bestehende Git-Blobs unverändert gegen die Basis; ausschließlich
Status/Übergabe als bestehende Dokumente fortgeschrieben.
Bei eigener PR-Erstellung CI noch laufend, technisch MERGEABLE/UNSTABLE;
PR #39 inzwischen vier erfolgreiche CI-Checks, CLEAN/MERGEABLE, weiterhin offen.
[PR-/CI-Snapshot](./counts_observation_design_2026_10_31/pr_snapshot.json).
Das ist kein neuer lokaler Testlauf und keine wissenschaftliche Freigabe.

## Vorheriger Paketabschluss: T1-Kandidat / synthetisches T2, 31.10.2026

**Begrenzt abgeschlossen:** [Design und synthetischer Prototyp](./t1_synthetic_t2_2026_10_31/README.md)
auf `codex/netzel-t1-synthetic-t2` ab PR #38, `d003fb9`.
Design vor Implementierung/Lauf separat als `2b41c37` committed und gepusht.
108 Posteriore/38 künstliche Datensätze, beide Priors und alle vorgesehenen
Varianten/Grenzfälle berichtet; 29 neue Tests bestanden, separater Direktprüfer
aller 108×1.681 Posteriormassen bestanden (max. Differenz 1,75×10⁻¹⁵),
38 Datensätze nacherzeugt, fünf Produktionsdateien im Zweitlauf bytegleich.
Alle Dateibytes werden in Git gesichert; keine neue Bytearchivlücke.

Passende Modelle enthalten 58/60 innere Wahrheiten im diskreten 95%-HPD-Set,
bewusst missspezifizierte 24/30; kein Deckungs-/SBC-Nachweis und keine Modellwahl.
Widerspruch kann trotz engem Posterior bestehen; fehlende Dimension, Prior-only-
und ±u-Mehrdeutigkeit bestätigt. Ein Randfall verfehlt HPD unter informativem Prior.
Beobachtungsparameter/Antwortkurven sind vorgegeben, keine echten Arten/Pollen.
**T1 für reale Pollenzählungen bleibt offen**, ebenso T0/T3/T4, W96 und
Monticchio-P1. Modelle, bestehende Resultate/Register, Counts, Baseline und
CLI/App unverändert; keine Unteragenten, Autorennachrichten oder Merges.

Operative Prüfabweichung: Zusätzlich gestarteter Gesamtstandardtest abgebrochen,
56 bestanden/zwei Browserfehler/zwei Langtests abgewählt; restliche Fälle nicht
als geprüft ausweisen. Fehlendes Chromium-Binary ist belegt. Vorhandene kurze
Regressionsketten mit realen Testeingängen liefen teilweise; keine neue
wissenschaftliche Kernkurve ausgewertet/gesichert.
[Prüfstand und genaue Abgrenzung](./t1_synthetic_t2_2026_10_31/TESTING.md).

Aktuelle Standprüfung: #31–#38 offen, #38-Head exakt `d003fb9`, main `7f1826c`.
Fünf bestehende Worktrees beim Einstieg sauber und erhalten; eigener Worktree
`/workspace/netzel-t1-synthetic-t2`. PR #38 wechselte während seiner CI von
UNSTABLE nach CLEAN, technisch MERGEABLE; keine Mergefreigabe.
[Remote-/API-/Branch-Snapshot](./t1_synthetic_t2_2026_10_31/git_snapshot.json).
Die folgenden älteren Paketabschnitte dokumentieren historische Prüfstände.

**Damals vorbereitet, inzwischen durch Nutzer-OK ausgeführt:** Anschlussdesign
für Pollen-Zählkomposition und Typgemische, siehe aktuellen Paketabschluss oben.
Der neue Folgeauftrag steht in [HANDOFF](./HANDOFF.md).
Sicherung als [PR #39](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/39)
gegen `codex/netzel-next-task-workflow` (#38), offen und ohne Merge.
Letzter gesicherter **Inhaltscommit** mit Code, Ergebnisbytes und Übergabe:
`953d1e995429a209b336cb83e89c41fda66a3f62`; danach nur PR-/Übergabeverweise.
654 bestehende Blobs einschließlich eines Symlinks gegen `d003fb9` unverändert
(ausgenommen die zwei erlaubten Einstiegsdokumente), alle fünf ursprünglichen
Worktrees sauber. [Erhaltungsprüfung](./t1_synthetic_t2_2026_10_31/preservation.json).
Bei PR-Erstellung war #39 MERGEABLE/UNSTABLE bei laufender CI; kein behauptetes
grünes Gesamtergebnis. [PR-Snapshot](./t1_synthetic_t2_2026_10_31/pr_snapshot.json).

## Git- und PR-Stand

**Erneut geprüft für das Bestimmungspaket am 31.10.2026:** #31–#36 offen,
#36-Head `f7e5290081dedb775a5cae7e210db88b71d9b1cb`, Ziel #35-Branch,
GitHub-Mergezustand CLEAN (technisch, keine Mergefreigabe), nicht gemergt.
Remote-main unverändert `7f1826cffddc8893ba6354cd3cca3fa43954b9be`;
enthält #36 und damit die ungemergte #31–#36-Kette nicht. Alle fünf vorhandenen
Worktrees ohne lokale/unversionierte Änderungen; vollständig erhalten.
Verfügbare lokale/Remote-Branches, Remote-Heads und separate API-Abfrage im
[Snapshot](./pollen_methods_2026_10_31/git_snapshot.json).
Eigener Worktree `/workspace/netzel-pollen-methods`, Branch
`codex/netzel-pollen-methods` ab genau `f7e5290`; PR-Ziel #36-Branch.
Keine Merges oder Autorennachrichten.

| Stand | Nachweis am 31.10.2026 |
| --- | --- |
| `main` / Integration #30 | Remote-Head `7f1826cffddc8893ba6354cd3cca3fa43954b9be`; #18–#29 erhalten |
| [PR #31](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/31) | GitHub-Seite: OPEN; Head `9458011198c69368764b8698c966c0ac23cc381c`, Branch `codex/netzel-spatial-probability-calibration`, Ziel `main` |
| [PR #32](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/32) | GitHub-Seite: OPEN; Head `9ee2f84e41a04fafe5af203e809deeaaf329dba3`, Branch `codex/netzel-calibration-block-uncertainty`, Ziel #31-Branch |
| Vollständiger Forschungsstand | #32-Head enthält #31; gesperrter #32-Ergebniscommit `6d35bbfe06cf54572dba2cb439495a8556eddb4d`, danach nur Sicherungsdokumentation in `9ee2f84` |
| Lokaler Ausgangspunkt | Checkout `work` bei `7f1826c`, sauber; keine ungesicherten Änderungen gefunden. Remote-Branches abgerufen; separater Worktree/Branch `codex/netzel-workflow-handoff` ab `9ee2f84` |

Abfrage über GitHub-PR-Seiten (eingebetteter Zustand OPEN) und `git ls-remote`
für Branches/PR-Heads. REST und GraphQL waren bei der ersten Prüfung über den Proxy mit HTTP 403 gesperrt;
das ist kein Beleg fehlender PRs. `main` enthält #31/#32 nach lokaler
Abstammungsprüfung nicht. Kein Merge ausgeführt oder beauftragt.

## Forschungsstand und Grenzen

| Bereich | Belegter Stand / offene Grenze |
| --- | --- |
| Analog-Baseline #18, CLI/App #19 | Fertige Baseline, Referenz `f9acc16`; Integration über #30. Claude betreut CLI/App; beides unverändert. |
| Quellen #20–#27 / Monticchio-P1 | 28 numerische Kandidaten, 41 native Nichttreffer, 47 Herkunftsfragen. Physischer LGM90D→MON16-Anschluss samt Korrelationsunsicherheit fehlt. 35 Plantago-Bilanzen und neun Scrophulariaceae-/Digitalis-Zahlengleichheiten sind kein Umcodierungsbeleg. [Quellenbedarf](./variable_provenance_2026_10_30/SOURCE_GAP.md) |
| Artenfunktionen #28 | Sieben experimentelle Artenmodelle; feste Referenz `8f02794434f561f5b30228360247935f3c025cc8`. Historische 279 Funktionen nicht wiederhergestellt. [Bericht](./taxon_transfer_2026_10_30/README.md) |
| SMOTE/Netze #29 | Balanced Accuracy 5°/10°: 71,64/69,20 % ohne gegen 90,41/89,74 % mit SMOTE (14/14 besser); Brier 0,03975/0,04438 gegen 0,07747/0,08245 (14/14 schlechter). Keine universelle Überlegenheit; keine fossile Wahrscheinlichkeitsfreigabe. [Bericht](./smote_nnet_2026_10_30/README.md) |
| Kalibrierung #31 | Vorabdesign `afa0c8a`; drei räumliche Mitglieder je Fold, innen/außen 100-km-Puffer. SMOTE-Brier roh 0,07129/0,07699 → kalibriert 0,03995/0,04408 (14/14 besser); rohes Ensemble ohne SMOTE 0,03939/0,04321 in 13/14 Fällen besser. Ohne SMOTE Kalibrierung nur 3/14 besser, Mittel leicht schlechter. Vier Varianten vollständig berichtet, keine automatische Übernahme. Ensemblewerte nicht mit #29-Einzelfits gleichsetzen. [Bericht](./calibration_2026_10_30/README.md) |
| Bedingte Sensitivität #32 | Vorabdesign `5a4a96f`; 200 gepaarte Block-Replikate, Netze/Normierung/SMOTE fest, nur innere Original-Kalibrierungszellen neu gewichtet. Band ab 190/200 schätzbaren Replikaten: 22/28 Vollgebiets-Ensembles, 104/140 äußere Folds. Abies beide Designs und Fagus 10° ohne Vollgebietsband; weitere äußere Lücken bei Picea/Taxus 10°. Fehlende Bänder sind keine Nullunsicherheit. [Bericht](./calibration_uncertainty_2026_10_30/README.md) |

#32 ist bedingte Bootstrap-Sensitivität, kein nominell validiertes
95%-Konfidenzintervall. Netz-, Karten-, Label-/Quellenunsicherheit,
Kreuztaxon-Kovarianz und gemeinsame Beobachtungsmodelle bleiben offen.
Belegte Pollentyp–Art-/Gattungszuordnung, weitere Taxa/Areale und neue unabhängige
Bestätigung fehlen. Keine fossile Klimakurve oder P2-Freigabe. Äußere Scores
werden nicht zur weiteren Modellwahl verwendet; bekannte Regionen sind keine
vollständig ungesehene Bestätigung.

## Prüf- und Speicherstand

Historische Prüfungen, **in diesem Dokumentationspaket nicht wiederholt**:
#31: 504 Fits / 1.512 Starts, 502 Kalibratorlösungen plus zwei Identitäten;
79 Repositorytests und fünf gezielte Tests dokumentiert. Elf Enddateien bytegleich
nach dokumentierter Checkpoint-Wiederherstellung gegen vorhandenen Manifest-Hash;
ein Lauf mit unvollständiger Gzip-Spur verworfen. #32: 100.800 Mitglied×Replikat-
Fälle (95.876 geschätzt, 3.724 unzureichende Klassenunterstützung, 1.200 wegen
gesperrtem Grundensemble), 1.589.440 Zellzeilen und 168 Zusammenfassungen geprüft;
85 Repositorytests plus sechs gezielte Tests dokumentiert. Sieben Enddateien zweier
frischer Läufe bytegleich, keine Wiederherstellung. Qhull-Eckpunktkorrektur nur im
neuen Evaluator; #31-Quellen/Parameter unverändert. Je zwei Browser-/Langtests
nicht ausgeführt. Dies ist kein zweiter menschlicher Review.
Details und genaue Nachweise bleiben in den Berichten und im
[historischen Archiv](./CHAT_WORK_ARCHIVE_2026_10_30.md).

Code, Modelle und kompakte Ergebnisse sind in Git gesichert; für vier große
Ausgaben fehlt ein belegtes dauerhaftes Bytearchiv. [Artefaktverzeichnis](./ARTIFACTS.md)
enthält Hash, Größe, Erzeugungsweg, Umgebung und ausdrücklich die Speicherlücke.
Keine großen Läufe in diesem Paket.

Dokumentationspaket #33: Einstiegsdokumentation vereinfacht, vollständige frühere
Texte im [Einstiegsarchiv](./ENTRYPOINT_ARCHIVE_2026_10_31.md) erhalten;
Links, Referenzen, Artefaktnachweise und reiner Dokumentationsumfang geprüft.
PR-Sicherung: [PR #33](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/33),
Branch `codex/netzel-workflow-handoff` gegen #32-Branch. Nach Veröffentlichung
der Netzwerkeinstellung funktioniert die GitHub-API; #33 wurde am 31.10.2026
erstellt, ohne Merge.

## Quellenpaket Pollentyp–Taxon, 31.10.2026

[Sieben-Arten-Inventur](./pollen_taxon_evidence_2026_10_31/README.md) abgeschlossen:
7/7 Namensbezüge mit 21 Fundstellen und elf gesperrten Repository-Eingängen.
Sechs Fossilkommentare bleiben auf Gattungsebene; bei Taxus steht der Artname
T. baccata im Kommentar. Dieser Namensbeleg ist keine hier bestätigte
morphologische Artbestimmung. Keine der sieben Artmodell-Zuordnungen neu
freigegeben, keine Counts/Gruppen/Modelle oder CLI/App geändert.

Prüfung: sieben übertragene Datensätze, 21 Locator, elf Hash-/Basiscommit-
Vergleiche und vier historische Originalquellenregistrierungen bestanden;
Originaldateien nicht erneut gelesen oder gehasht. Monticchio/EMPD2-Downloads
scheitern am Proxy (CONNECT HTTP 403 für doi.pangaea.de / store.pangaea.de).
Die 2025-Originale wurden nicht angefordert. Keine neuen morphologischen
Primärbelege behauptet; abgeschlossene Inventur von offener Originalprüfung
und Artbrücke getrennt. Monticchio-P1 und 28/41/47 unverändert.

Sicherung als [PR #34](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/34)
auf `codex/netzel-pollen-taxon-evidence`, Basis `6f35df1` aus offenem #33;
PR-Ziel ist der #33-Branch, kein Merge. Der damalige Netzwerkentwurf wurde
inzwischen vom Nutzer veröffentlicht; die Originalprüfung folgt unten.

## Originaldefinitionen und EMPD2-Namensketten, 31.10.2026

[Originalprüfung](./original_pollen_names_2026_10_31/README.md) abgeschlossen:
EMPD2.sqlite3 und Monticchio PANGAEA.739452 frisch bezogen, beide bestehenden
Hashes und Bytezahlen identisch. Sieben Fossildefinitionen und sieben direkte
Artnamenslinks gesichert. Monticchio-Zeile 9 bezeichnet Parameterkommentare
explizit als akzeptierte EPD-Namen; auch Taxus baccata ist damit kein für sich
hinreichender Beleg einer ursprünglichen Artbestimmung.

Alle 5.230 p_vars-Zeilen durchsucht; 128 lexikalische Treffer über sieben
Abfragen, mit Referenzzielen 135 Abfrage×Zeilen / 120 unterschiedliche Zeilen.
Alle ausgewählten acc_var_-Ketten schließen. Vollständigkeit gilt nur für die
deklarierte Namenssuche, nicht alle botanischen Synonyme oder Aggregatpools.
Neu originalbelegt u. a. Betula pendula 1731→128 Betula pubescens-type und
Taxus 198→111 gegenüber Taxus 3433→3433. Datenbankrelationen sind keine hier
bestätigten botanischen Synonyme. Vier Artnamenslinks enden auf dem bisherigen
Katalogkandidaten, drei auf anderen IDs; keine neue Artmodell-Zuordnung genehmigt.

Separater direkter Quellenprüfer: zwei Originalhashes, drei Repository-
Hash-/Basisvergleiche, sieben Parameterdefinitionen, alle 135 Kettenzeilen und
sieben Artnamenslinks bestanden. Kein unabhängiger menschlicher/palynologischer
Review behauptet. Originalzählungen nicht ausgewertet, keine Modell-/Langläufe.
Bestehende Modelle, Quellregister, Baseline und CLI/App unverändert.

Gesichert als [PR #35](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/35),
Branch `codex/netzel-original-pollen-names` ab `98ef1bf` (#34), gegen den
#34-Branch; offen und ohne Merge erstellt. Als Nächstes Methoden von
Watts, Allen & Huntley (1996) und tatsächlich verwendete Bestimmungsreferenzen
prüfen; je Art diagnostische Typgrenzen und verbleibende Lücken dokumentieren.
[Präziser nächster Auftrag](./original_pollen_names_2026_10_31/SOURCE_GAP.md).
2025-Matrix/S1 nicht frisch geprüft; Monticchio-P1, 28/41/47 und vier große
Bytearchivlücken unverändert.

## Bestimmungsumfänge und Literaturgrenzen, 31.10.2026

Gesichert als [PR #37](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/37),
Branch `codex/netzel-pollen-methods`, Basis `f7e5290` und Ziel
`codex/netzel-first-taxon-pilot-roadmap` (#36), ohne Merge.

[Literaturpaket](./pollen_methods_2026_10_31/README.md): **7/7 Arten** mit
morphologischen Referenzbefunden, Typgrenzen, Gegen-/Fehlbelegen und exakten
Seiten/Abschnitten/Objekten dokumentiert. 14 gelesene Referenzquellen/-seiten,
ein zusätzlicher W96-Eintrag nur mit Metadaten/Abstract. Keine mikroskopische
oder unabhängige menschliche/palynologische Prüfung behauptet.

**W96-Methodenkette blockiert:** Watts, Allen & Huntley (1996), DOI
10.1016/0277-3791(95)00093-3, Methodenteil und tatsächlich verwendete
Bestimmungsreferenzen nicht zugänglich/gelesen. Verlags-/Durham-Weböffnung
HTTP 403, während mehrere andere Referenztexte über das Webwerkzeug lesbar
waren. Keine generelle Methodik von W96 oder dortige Schlüsselnutzung aus
späterer Literatur konstruiert. Bei **allen sieben Fossilparametern weiterhin
kein datensatzbezogener Bestimmungsbeleg im zugänglichen Material**.
Die vollständige historische Methodenkette und T0 sind nicht abgeschlossen.

Belegt sind unter anderem ein direkt beschriebener nicht artspezifischer
Abies-Merkmalsumfang, indirekt referierter Betula-pubescens-Typ mit pendula,
Corylus/Myrica-Überlappung im britischen Typkatalog, begrenzte Artunterscheidung
in fremdem Fagus-Fossilmaterial und in modernen Taxus-Vergleichen sowie
publizierte Tilia-Unterscheidungsmerkmale. Eine originale morphologische
Picea-abies-type/obovata-Umschreibung fehlt weiterhin. Grenzen moderner Studien,
Präparationen und fremder Regionen nicht auf LGM90D übertragen; Datenbankrelationen
aus #35 bleiben ohne botanische Synonymiebehauptung oder automatische Artfreigabe.

Dokumentationsprüfung: sieben Evidenzzeilen, Quellenreferenzen, lokale Links,
Hash-/Basisabgleich der übernommenen Eingänge und begrenzter Diffumfang geprüft;
[Prüfprotokoll](./pollen_methods_2026_10_31/verification.json).
Keine Originalzählungen ausgewertet, Modelle gerechnet oder äußeren Scores
zur Entscheidung verwendet. Bestehende Modelle/Forschungsresultate/-register,
Fossilcounts, Analog-Baseline und CLI/App bytegleich; keine Unteragenten.

**Nächster enger Quellenauftrag:** W96-Methodenteil/Referenzverzeichnis und
daraus tatsächlich angewandte Schlüssel samt Kategorienkette nachliefern;
[konkreter Restbedarf](./pollen_methods_2026_10_31/SOURCE_GAP.md).
Die Referenzevidenz ist gesichert, der Datensatzbezug bleibt offen. T1 und
synthetisches T2 können als getrenntes vorab gesichertes Designpaket folgen;
T3/T4, Monticchio-P1, 28/41/47 und große Bytearchivlücken unverändert.

## Ablaufvereinbarung und vorbereiteter Folgeauftrag, 31.10.2026

Auf Nutzerwunsch in AGENTS.md festgehalten: Nach jedem begrenzten Paket Ergebnis,
Prüfungen und Restlücken im Repository sichern, genau einen ausführbaren
Folgeauftrag in HANDOFF vorbereiten und nach kurzem Nutzer-OK fortsetzen.
Bei sinnvollem Chatwechsel einen kopierfertigen Prompt im Chat ausgeben und in
HANDOFF erhalten. Keine automatische Chat-Erstellung oder Hintergrundaufgabe.

**Erledigt:** diese Ablaufdokumentation und konkrete Übergabe vorbereitet.
**Vorbereitet, wartet auf OK:** T1-Design samt begrenztem synthetischem
T2-Forschungsprototyp; Umfang, Leseplan, Schutzregeln und Abschlusskriterien stehen
in [HANDOFF](./HANDOFF.md). Noch kein Methodenentwurf, Prototyp oder neuer Lauf
in diesem Dokumentationspaket. W96-Methodenkette bleibt separat blockiert,
T0 und reale Eingangsfreigaben bleiben offen. Bestehende Forschungsdateien und
das Literaturpaket #37 unverändert. Eigener Branch
`codex/netzel-next-task-workflow` ab `fc52ce7` (#37); kein Merge.
Sicherung als [PR #38](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/38)
gegen den #37-Branch; initialer Inhaltscommit `8244754`.

Prüfung dieses Ablaufpakets: lokale Links und reiner Drei-Dateien-Diffumfang
geprüft; keine Code-/Modelltests nötig oder ausgeführt.
