# Fortsetzung nach der räumlich verschachtelten Kalibrierung

Stand 30.10.2026. [PR #31](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/31),
Branch `codex/netzel-spatial-probability-calibration`, Basis main nach #30 auf
`7f1826cffddc8893ba6354cd3cca3fa43954b9be`. Vor Fortsetzung aktuellen PR-/Merge-
Stand prüfen. #18–#29 sind über #30 erhalten; #31 nicht ohne neuen Auftrag mergen.
Solange #31 offen ist, dessen Forschungsbranch als Ergebnisstand verwenden;
nicht voraussetzen, dass main die neuen Kalibrierungsdateien bereits enthält.

Zuerst [HANDOFF](./HANDOFF.md), [PROJECT_STATUS](./PROJECT_STATUS.md),
[Ergebnisbericht](./calibration_2026_10_30/README.md),
[Vorabdesign](./calibration_2026_10_30/DESIGN.md) und
[Reproduktion](./calibration_2026_10_30/REPRODUCE.md) lesen.

## Fertig

Vorabdesign und feste Konfiguration vor jedem neuen Fit in `afa0c8a` veröffentlicht.
Je äußerem Fold drei räumliche Netz-/Kalibrator-Paare; Transformationen und SMOTE
nur im ursprünglichen inneren Netztraining, Kalibrierung nur auf zurückgehaltenen
inneren Originalzellen. Äußerer Test plus 100-km-Puffer ausgeschlossen; innerer
Puffer nochmals 100 km. Keine nachträgliche Architektur-, Kalibrator- oder
Schwellenwahl. Roh/kalibriert sind Mittel derselben drei Netzfits.

SMOTE-Brier in allen 14 Art×Block-Fällen verbessert. Artenmittel 5°/10°:
0,07129/0,07699 roh → 0,03995/0,04408 kalibriert. Das rohe Ensemble ohne SMOTE
bleibt im Brier in 13/14 Fällen besser, Mittel 0,03939/0,04321. Kalibrierung ohne
SMOTE verbessert nur 3/14 Fälle und verschlechtert den Mittelwert leicht.
SMOTE-BA bei 0,5 fällt durch Kalibrierung von 90,55/89,35 auf 67,75/65,39 %.
Kein automatischer Sieger oder Variantenwechsel in der App.

504 Netzfits / 1.512 Starts und 504 Mitgliedskalibratoren gespeichert und
separat numerisch geprüft: 502 geschätzt, zwei vorab deklarierte Identitäten
wegen nur zwei positiven Kalibrierungszellen (Abies, 10°, outer 3 / inner 2).
198.680 Labels, 397.360 äußere Prognosezeilen, 2.279.093 synthetische Elternpaare;
kleinster innerer/äußerer Abstand 100,093 km. Kein Import des neuen Produzenten
oder Kalibrators im Audit; gemeinsame alte Auditwerkzeuge und Bibliotheken benannt.

Zwei vollständige Trainingsausführungen haben elf bytegleiche Enddateien,
nach ausdrücklich dokumentierter technischer Checkpoint-Wiederherstellung
aus der gleichlaufenden vollständigen Modellkopie gegen den vorher vorhandenen
Manifest-Hash. Ein anderer Lauf mit unvollständiger Gzip-Elternspur wurde verworfen.
Keine stillen Reparaturen, Designänderungen oder nachjustierten Gewichte.
79 Repositorytests und fünf neue Forschungstests bestanden; zwei Browser-/zwei
Langtests nicht ausgeführt. Vollständige Hinweise in den Reproduktionsdateien.

28 Vollgebiets-Ensembles / 84 Netz-/Kalibrator-Paare und alle äußeren Parameter
gespeichert. Rohdaten und große Laufdateien anhand gesperrter Quellen frisch
reproduzieren; keine alten temporären Dateien oder ScienceDirect-ZIP nötig.

## Grenzen und als Nächstes

Äußere #29- und #31-Ergebnisse nicht zur weiteren Optimierung verwenden.
Die Kartenregionen sind bereits untersucht; die verschachtelte Bewertung ist
keine vollständig ungesehene Bestätigung, Feldvalidierung oder fossile Prüfung.
Keine ursprünglichen 279 Funktionen und keine bitgleiche historische R-Reproduktion.
Weitere Taxa, belegte Pollentyp–Art-/Gattungszuordnung, Kalibrierungs-/Quellen-
unsicherheit und gemeinsame Beobachtungsmodelle bleiben offen. Die drei Mitglieder
sind keine Unsicherheitsintervalle. Kein fossiler Klimadurchlauf oder Pilotfreigabe.

Monticchio-P1 bleibt blockiert: physischer LGM90D→MON16-Crosswalk einschließlich
Korrelationsunsicherheit fehlt. 28 Kandidaten / 41 native Nichttreffer /
47 Herkunftsfragen unverändert. Keine Quellenkorrekturen oder Autorennachrichten.
Claude betreut CLI/App; Codex Kalibrierung, Daten und separate Prüfung.
Analog-Baseline nicht neu entwickeln. Kurze deutsche Updates: fertig / aktuell /
als Nächstes. Keine neuen PRs ohne erneuten Nutzerauftrag mergen.
