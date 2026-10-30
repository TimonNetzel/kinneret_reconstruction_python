# Fortsetzung nach der bedingten räumlichen Kalibrierungssensitivität

**Speicherprüfung 30.10.2026:** [Sicherungsnachweis der Chat-Arbeit](./CHAT_WORK_ARCHIVE_2026_10_30.md)
mit Dateiinventar und festen Ergebniscommits für #31/#32. Code, Modelle,
Parameter und Berichte liegen im Repository; vier große Laufausgaben sind bisher
nur durch Hashes und Reproduktionsbefehle gesichert. Das vollständige Bytearchiv
dieser vier Dateien bleibt offen. PRs weiterhin ungemergt.

30.10.2026. [PR #32](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/32),
Branch `codex/netzel-calibration-block-uncertainty`, abhängig von offenem #31,
gesperrte Basis `9458011198c69368764b8698c966c0ac23cc381c`. PR-Ziel ist dessen
Branch `codex/netzel-spatial-probability-calibration`. Vor Fortsetzung aktuellen
PR-/Merge-Stand prüfen. #31/#32 nicht ohne erneuten ausdrücklichen Auftrag mergen.
Solange offen, neuesten Forschungsstand auf #32 lesen; main enthält ihn noch nicht.

Zuerst [HANDOFF](./HANDOFF.md), [PROJECT_STATUS](./PROJECT_STATUS.md),
[neuen Bericht](./calibration_uncertainty_2026_10_30/README.md),
[Vorabdesign](./calibration_uncertainty_2026_10_30/DESIGN.md),
[Quellen-/Aussagegrenzen](./calibration_uncertainty_2026_10_30/SOURCES.md) und
[Reproduktion](./calibration_uncertainty_2026_10_30/REPRODUCE.md) lesen.

## Fertig

Design, Konfiguration und Quellen vor jeder Bootstrap-Kalibrierung in `5a4a96f`
veröffentlicht und danach unverändert. 200 gepaarte räumliche Block-Replikate,
gleiche Multiplizitäten für beide Familien und drei Mitglieder. Nur innere
Original-Kalibrierungszellen neu gewichtet; Netze, Normierung und SMOTE fest.
Äußere Labels gehen weder in Bänder noch in Verfügbarkeitsauswahl ein.

Mindestens 190/200 schätzbare Ensemble-Replikate für punktweise Quantile
0,025/0,5/0,975. Fehlende Bänder werden nicht durch Nullbreite ersetzt.
**22/28 Vollgebiets-Ensembles und 104/140 äußere Ensemble-Folds verfügbar.**
Abies in beiden Designs und Fagus bei 10° ohne Vollgebiets-Band. Weitere äußere
Lücken bei Picea 10° und Taxus 10°. Alle Fälle einschließlich Ausfällen gespeichert.

Der unabhängige numerische Audit bestätigt 198.680 Original-Labels,
2.174.335 nichtnullige Blockmultiplizitäten, alle 100.800 Mitglied×Replikat-
Fälle (95.876 geschätzt, 3.724 Klassenunterstützung unzureichend, 1.200 Plätze
eines schon grundkalibrierungsbedingt gesperrten Ensembles), 1.589.440 Zellzeilen
und 168 Zusammenfassungen. Kein numerischer Fit-Ausfall. Alle Quantile,
Nichtschätzbarkeiten, Rollen und KKT-Reste separat geprüft. Neue Produzenten-/
Fitterimporte im Prüfer ausgeschlossen; gemeinsame alte Audithilfen und
GIS-/Numerikbibliotheken offengelegt, kein zweiter menschlicher Reviewer.

Zwei frische vollständige Endläufe in sieben Dateien bytegleich, alle Gzip-
Streams gültig, keine Wiederherstellung der Endausgaben. Frühere unvollständige
Laufprotokolle verworfen. Ein konservativer Qhull/Delaunay-Randfehler an exakten
Hülleneckpunkten ist ausschließlich im neuen Evaluator behoben und getestet;
keine Toleranzerweiterung oder geänderte Hüllen. #31-Quelldateien und alle
Kalibrierungsparameter bleiben unverändert. Ausführungsbefunde dokumentiert.
85 Repositorytests und sechs gezielte Tests bestanden; zwei Browser-/zwei
Langtests nicht ausgeführt. Große Zell-/Ziehungsdateien anhand gespeicherter
Hashes frisch reproduzieren; keine alten temporären Dateien voraussetzen.

## Aussagegrenzen und als Nächstes

**Bedingte Bootstrap-Sensitivitätsbänder bei festen Netzen und schätzbaren
Replikaten, keine nominell validierten 95%-Konfidenzintervalle.** Netz-, Karten-,
Label-/Quellenunsicherheit und Kreuztaxon-Kovarianz nicht variiert. Keine
vollständige Kalibrierungs- oder Rekonstruktionsunsicherheit. Bekannte Regionen
aus #28–#31 sind keine vollständig ungesehene Bestätigung. Äußere Ergebnisse
nicht zur Optimierung von Gates, Blockgrößen, Architektur oder Artenauswahl nutzen.

Als Nächstes belegte Pollentyp–Art-/Gattungszuordnung und weitere Taxa/Areale
bearbeiten; volle Unsicherheit und gemeinsame Beobachtungsmodelle weiter offen.
Keine automatische Verbindung der sieben botanischen Arten mit Gattungspollen.
Keine fossile Klimakurve, Pilotfreigabe oder Quellenkorrektur. Claude betreut
CLI/App; Codex Kalibrierung, Datenrecherche und separate Prüfung. Analog-Baseline
und Claudes Integration unverändert lassen.

Monticchio-P1 bleibt blockiert: physische LGM90D→MON16-Zuordnung einschließlich
Korrelationsunsicherheit fehlt. 28 Kandidaten / 41 native Nichttreffer /
47 Herkunftsfragen unverändert. Keine Autorennachricht oder Merge.
Kurze deutsche Updates: fertig / aktuell / als Nächstes.
