# Sicherungsstand der Chat-Arbeit zu #31 und #32

Stand 30.10.2026, auf den Nutzerauftrag „Speichere alles, was du hier gemacht hast, im Repo“.
Diese Übersicht sichert Arbeitsstand, Entscheidungen, Ergebnisverweise und bekannte
Speicherlücken. Sie dokumentiert eine Repositoryprüfung; die wissenschaftlichen
Läufe und Audits wurden dafür nicht erneut ausgeführt.

## Vollständiger Forschungsstand und Einstieg

Der Branch `codex/netzel-calibration-block-uncertainty` enthält die Arbeiten aus
#31 und #32 gemeinsam. Beide PRs sind offen und nicht gemergt.
`main` bleibt auf `7f1826cffddc8893ba6354cd3cca3fa43954b9be`.
Der aktuelle Speicherauftrag enthält keinen neuen Mergeauftrag.

| Arbeit | Gesperrter Ergebniscommit | Repository-Einstieg |
| --- | --- | --- |
| Räumliche Wahrscheinlichkeitskalibrierung, [#31](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/31) | `9458011198c69368764b8698c966c0ac23cc381c` | [Bericht](./calibration_2026_10_30/README.md), [Reproduktion](./calibration_2026_10_30/REPRODUCE.md), [Übergabe](./CHAT_RESTART_CALIBRATION.md) |
| Bedingte Kalibrierungssensitivität, [#32](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/32) | `6d35bbfe06cf54572dba2cb439495a8556eddb4d` | [Bericht](./calibration_uncertainty_2026_10_30/README.md), [Reproduktion](./calibration_uncertainty_2026_10_30/REPRODUCE.md), [Übergabe](./CHAT_RESTART_UNCERTAINTY.md) |

Gemeinsamer Einstieg: [HANDOFF.md](./HANDOFF.md) und
[PROJECT_STATUS.md](./PROJECT_STATUS.md). Die bisherigen Arbeiten #18–#29 sind
über #30 auf main erhalten; [Integrationsnachweis](./consolidation_2026_10_30/README.md).

## Was als Datei im Repository liegt

Alle 44 unterschiedlichen geänderten Pfade aus #31/#32 sind im geprüften
#32-Ergebnisbaum vorhanden: Code, Tests, Vorabdesigns, Konfigurationen,
Modellgewichte einschließlich äußerer Fits und Starts, Vollgebietsmodelle,
vollständiges Bootstrap-Parameter-/Statusregister, kompakte Ergebnistabellen,
Quellenverweise, Laufmanifeste, Prüf- und Wiederholungsberichte sowie Übergaben.

[CHAT_WORK_ARCHIVE_2026_10_30.json](./CHAT_WORK_ARCHIVE_2026_10_30.json)
listet jeden Pfad mit Git-Blob-ID und Bytezahl am obigen Ergebniscommit.
Die wissenschaftlichen #31-Dateien sind im #32-Baum blobgleich erhalten;
nur die gemeinsamen Status-/Übergabedokumente wurden dort fortgeschrieben.
Git-Blob-IDs stammen aus dem vollständigen GitHub-Dateibaum. Ausgabe-SHA-256
stammen aus den bereits veröffentlichten Laufmanifesten; sie wurden in diesem
Speicherschritt nicht durch einen erneuten Download sämtlicher Ausgaben geprüft.

## Bekannte Grenze: vier große Ausgaben nur reproduzierbar gesichert

Die folgenden Ausgaben sind **nicht als Datei im Git-Baum enthalten**.
Ihre SHA-256-Hashes und vollständigen Reproduktionsbefehle sind gespeichert.
In der aktuellen Arbeitsumgebung liegen die ursprünglichen Laufdateien nicht
vor; sie konnten bei diesem Speicherschritt nicht nachträglich hochgeladen werden.
Ein Hash ersetzt keine Sicherung der Dateibytes.

| Arbeit | Nicht als Git-Datei gespeichert | Gespeicherter Nachweis / Wiederherstellungsweg |
| --- | --- | --- |
| #31 | `predictions.csv`, `smote_trace.csv.gz` | [Manifest](./calibration_2026_10_30/run_manifest.json), [Reproduktion](./calibration_2026_10_30/REPRODUCE.md) |
| #32 | `block_draws.csv.gz`, `intervals.csv.gz` | [Manifest](./calibration_uncertainty_2026_10_30/run_manifest.json), [Reproduktion](./calibration_uncertainty_2026_10_30/REPRODUCE.md) |

Das JSON-Inventar unterscheidet deshalb `git_blob` und
`hash_and_reproduction_only`. Der Stand ist keine vollständige Bytekopie
aller erzeugten oder verworfenen temporären Laufdateien. Für ein solches
Vollarchiv bleiben diese vier abgeleiteten Ausgaben nachzuliefern oder anhand
der gesperrten Originalquellen frisch zu reproduzieren und gegen die Hashes zu prüfen.
Fremde Roharchive und Originalpublikationen werden wie bisher über
Quellenregister, Prüfsummen und Bezugswege dokumentiert.

## Ergebnisse und vorab festgelegte Entscheidungen

- #31: drei innere räumliche Netz-/Kalibrator-Paare je äußerem Fold,
  innen und außen 100-km-Puffer. Transformationen und SMOTE nur im inneren
  Training, Kalibrierung auf zurückgehaltenen Originalzellen. Design vor
  jedem Fit in `afa0c8a8845a5f40daee4289cc429fdf4a6ecf6a` veröffentlicht.
  Kalibrierung verbessert den SMOTE-Brier in 14/14 Art×Block-Fällen;
  kalibriertes SMOTE bleibt in 13/14 Fällen schlechter als das rohe
  Netzensemble ohne SMOTE. Keine automatische Variantenübernahme.
- #32: 200 gepaarte räumliche Block-Replikate bei festen Netzen,
  Transformationen und SMOTE. Nur innere Kalibrierungszellen neu gewichtet.
  Vorabdesign `5a4a96f1d7c01b64a230fb8ad6a7fbaf84857697`.
  Band erst ab 190/200 schätzbaren Ensemble-Replikaten; 189 bleibt gesperrt.
  Verfügbar für 22/28 Vollgebiets-Ensembles und 104/140 äußere Folds.
  Abies in beiden Blockdesigns und Fagus bei 10° ohne Vollgebietsband.
  Bedingte Sensitivitätsbänder, keine nominell validierten
  95%-Konfidenzintervalle oder vollständige Rekonstruktionsunsicherheit.
- Bereits untersuchte Gebiete gelten nicht als vollständig ungesehene
  Bestätigung. Äußere #29/#31-Ergebnisse dürfen keine weitere Optimierung steuern.

## Prüfungen und technische Befunde bleiben nachvollziehbar

#31: separater Audit von 198.680 Labels, 397.360 äußeren Testzeilen,
504 Netzfits und 2.279.093 synthetischen Elternpaaren. 79 Repositorytests
und fünf gezielte Forschungstests dokumentiert. Elf Enddateien bytegleich
nach ausdrücklich dokumentierter Checkpoint-Wiederherstellung aus dem
vollständigen Modell desselben Ersatzlaufs gegen dessen vorhandenen
Manifest-Hash. Ein Lauf mit unvollständiger Gzip-Spur wurde verworfen.
[Ausführungsbefunde](./calibration_2026_10_30/execution_notes.json) und
[Wiederherstellungsnachweis](./calibration_2026_10_30/checkpoint_recovery.json).

#32: separater Audit aller 100.800 Mitglied×Replikat-Fälle,
2.174.335 nichtnulligen Blockmultiplizitäten, 1.589.440 Zellzeilen und
168 Zusammenfassungen. 85 Repositorytests und sechs gezielte Tests dokumentiert.
Zwei frische Endläufe in sieben Dateien bytegleich, ohne Wiederherstellung.
Frühere unvollständige komprimierte Protokolle wurden verworfen.
Der Audit fand zusätzlich eine Qhull/Delaunay-Fehlklassifikation exakter
Klimahülleneckpunkte; nur der neue Evaluator wurde korrigiert und durch einen
Regressionstest gesichert. Keine Toleranzerweiterung oder Parameteränderung.
[Ausführungsbefunde](./calibration_uncertainty_2026_10_30/execution_notes.json).

Jeweils zwei Browserprüfungen und zwei Langtests wurden nicht ausgeführt.
Die GitHub-Prüfungen der beiden gesperrten Ergebniscommits sind bei dieser
Sicherung erneut als erfolgreich abgerufen und im JSON-Inventar verlinkt.
Das ist kein neu ausgeführter wissenschaftlicher Audit und kein zweiter
menschlicher Review.

## Offene Arbeit und Zuständigkeit

Belegte Pollentyp–Art-/Gattungszuordnung, weitere Taxa/Areale,
Netz-/Quellen-/Kartenunsicherheit, Kreuztaxon-Kovarianz und gemeinsame
Beobachtungsmodelle bleiben offen. Monticchio-P1 bleibt wegen fehlender
physischer LGM90D→MON16-Zuordnung einschließlich Korrelationsunsicherheit
blockiert; 28 Kandidaten, 41 native Nichttreffer und 47 Herkunftsfragen unverändert.

Claude betreut CLI/App; Codex Kalibrierung, Datenrecherche und separate
Prüfung. Die Analog-Baseline bleibt erhalten. Keine stillen
Quellenkorrekturen, Pilotfreigaben, Autorennachrichten oder Merges.
