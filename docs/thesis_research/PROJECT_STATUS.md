# Netzel-Projekt: belegter aktueller Stand

Standprüfung 31.10.2026. Ziel bleibt eine überprüfbare taxonbasierte
Multiproxy-Klimarekonstruktion; kein fertiger fossiler Gesamtdurchlauf.
Arbeitsregeln stehen ausschließlich in [AGENTS.md](../../AGENTS.md),
der nächste Auftrag und Leseplan in [HANDOFF.md](./HANDOFF.md).

## Git- und PR-Stand

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

Aktuelles Arbeitspaket: Einstiegsdokumentation vereinfacht, vollständige frühere
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
PR-Ziel ist der #33-Branch, kein Merge. Netzwerkentwurf für doi.pangaea.de und
store.pangaea.de additiv gespeichert; noch nicht angewendet oder veröffentlicht.
Als Nächstes Originaldefinitionen und vollständige EMPD2-Namensketten frisch
hashprüfen, danach gezielte morphologische Belege, gemäß
[SOURCE_GAP.md](./pollen_taxon_evidence_2026_10_31/SOURCE_GAP.md).
