# Netzel-Projekt: kurzer Arbeitsstand

Stand 30.10.2026. Vor Fortsetzung aktuellen PR-/Merge-Stand prüfen.
Ziel: überprüfbare App für taxonbasierte Multiproxy-Klimarekonstruktionen
europäischer Fundstellen, mit belegter Kalibrierung und Unsicherheiten.

| Bereich | Stand |
| --- | --- |
| Moderne Analog-Baseline #18 | Fertig; #18 auf main gemergt. Unveränderte Referenz f9acc16. |
| Erste sieben Artenfunktionen #28 | Fertiger experimenteller Trainingsstand aus Meusel/BfN + WorldClim 2.1; feste Referenz. |
| SMOTE + einfaches Netz #29 | **Berechnet und unabhängig geprüft:** sieben Arten, identische äußere räumliche Tests; deutlich bessere Balanced Accuracy, zugleich schlechtere rohe SMOTE-Wahrscheinlichkeiten im Brier. |
| Historische 279 Funktionen | Nicht wiederhergestellt; keine bitgleiche historische R-/caret-Reproduktion. |
| Räumliche Wahrscheinlichkeitskalibrierung #31 | **Berechnet und separat numerisch geprüft:** innere räumliche Kalibrierungsfolds, unabhängige äußere Bewertung je Fit; SMOTE-Brier verbessert, gegenüber rohem Netzensemble ohne SMOTE in 13/14 Fällen weiter schlechter. |
| Bedingte Kalibrierungssensitivität #32 | **Berechnet und separat geprüft:** 200 räumliche Block-Replikate bei festen Netzen; Band für 22/28 Vollgebiets-Ensembles, 104/140 äußere Ensemble-Folds. Keine nominell validierte Deckung. |
| Taxonverfahren insgesamt | Weitere Taxa, belegte Fossiltyp-Zuordnung, externe Bestätigung und Kalibrierungs-/Quellenunsicherheit offen. |
| Monticchio-P1 | Weiter offen: physischer LGM90D→MON16-Crosswalk samt Korrelationsunsicherheit fehlt. |
| Proben und Quellen aus #26/#27 | Unverändert: 28 numerische Kandidaten, 41 native Nichttreffer, 47 offene Herkunftsfragen. |
| Fossilrekonstruktion / Multiproxy-App | Kein fertiger Gesamtdurchlauf, keine neue fossile Klimakurve und keine P2-Freigabe. Claude betreut CLI/App; Integration #19 in den gemeinsamen Stand übernommen. |

**Aktueller Abschluss #32:** [Bedingte Kalibrierungssensitivität](./calibration_uncertainty_2026_10_30/README.md),
[PR #32](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/32),
Branch `codex/netzel-calibration-block-uncertainty`, abhängig von offenem #31 am
festen Head `9458011`; PR-Ziel ist dessen Forschungsbranch. Kein Mergeauftrag.
Vorabdesign `5a4a96f`, unverändert. 200 gepaarte geographische Block-Replikate,
nur innere Original-Kalibrierungszellen neu gewichtet, Netze/Normierung/SMOTE fest.
Äußere Labels nicht für Bänder oder Verfügbarkeitsauswahl verwendet.

22/28 Vollgebiets-Ensembles und 104/140 äußere Ensemble-Folds mit bedingtem
Sensitivitätsband; Abies in beiden Designs und Fagus 10° ohne Vollgebiets-Band.
Gate 190/200 schätzbare Replikate unverändert; fehlende Bänder sind keine Nullbreite.
95.876 Kalibrierungen / insgesamt 100.800 Mitglied×Replikat-Fälle, 1.589.440
Zellzeilen, 2.174.335 Blockmultiplizitäten und alle 168 Tabellenzeilen separat
geprüft. Zwei frische Endläufe: sieben bytegleiche Dateien, ohne Wiederherstellung.
Vorige I/O-Befunde und exakte Hülleneckpunkt-Korrektur ausdrücklich dokumentiert.
85 Repositorytests und sechs gezielte Tests bestanden; Browser-/Langtests ausgenommen.

**Als Nächstes:** belegte Pollentyp-Zuordnung und weitere Arten/Areale.
Vollständige Kalibrierungs-/Quellen-/Netzunsicherheit und Kreuztaxon-Kovarianz
bleiben offen. Keine nominell validierten 95%-Konfidenzintervalle, unabhängige
neue Region oder Fossilfreigabe. Monticchio-P1 und 28/41/47 unverändert.
Aktuelle Fortsetzung: [CHAT_RESTART_UNCERTAINTY.md](./CHAT_RESTART_UNCERTAINTY.md).
#31/#32 bleiben offen; Merges nur nach erneutem ausdrücklichem Auftrag.

## Historischer Abschluss #31 vor der Sensitivitätsanalyse

**Abschluss #31:** [Kalibrierungsbericht](./calibration_2026_10_30/README.md),
[PR #31](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/31),
Branch `codex/netzel-spatial-probability-calibration`, Basis aktuelles main nach
#30 (`7f1826c`). Design vor jedem Fit veröffentlicht in `afa0c8a` und unverändert.
504 Netzfits / 1.512 Starts; 502 geschätzte und zwei vorab deklarierte Identitäts-
Kalibratoren. 198.680 Labels, 397.360 Testzeilen und 2.279.093 synthetische
Elternpaare separat geprüft. Kleinster innerer/äußerer Abstand 100,093 km.

Artenmittel Brier 5°/10°: SMOTE roh 0,07129/0,07699 → kalibriert
0,03995/0,04408; Verbesserung in allen 14 Fällen. Rohes Ensemble ohne SMOTE
0,03939/0,04321, besser als kalibriertes SMOTE in 13/14 Fällen. Ohne SMOTE
verbessert Kalibrierung nur 3/14 Fälle und verschlechtert die Mittelwerte leicht.
Vier Varianten vollständig berichtet; keine automatische Variantenübernahme.
Verglichen werden dieselben drei Mitglieder je roh/kalibriert, keine direkten
Einzelfit-Ergebnisse aus #29. Bereits untersuchte Regionen sind keine vollständig
ungesehene Bestätigung; äußere #29/#31-Werte nicht zur Optimierung verwenden.

Elf Enddateien zweier vollständiger Trainingsausführungen bytegleich nach
explizit dokumentierter Wiederherstellung des letzten Checkpoint-Eintrags aus
der vollständigen Modellkopie desselben Laufs gegen den vorher gespeicherten
Manifest-Hash. Ein Lauf mit unvollständiger Gzip-Spur wurde verworfen;
[Ausführungshinweise](./calibration_2026_10_30/execution_notes.json).
79 Repositorytests und fünf neue Forschungstests bestanden; zwei Browser-/zwei
Langtests nicht ausgeführt. Reproduktion ohne alte Zwischendateien.

**Als Nächstes:** belegte Pollentyp-Zuordnung, weitere Taxa und Kalibrierungs-/
Quellenunsicherheit bearbeiten. Kein fossiler Gesamtdurchlauf oder Pilotfreigabe.
Claude behält CLI/App. Monticchio-P1 und 28/41/47 bleiben unverändert offen.
#31 bleibt offen und wird nur auf erneuten ausdrücklichen Auftrag gemergt.
Aktuelle Fortsetzung: [CHAT_RESTART_CALIBRATION.md](./CHAT_RESTART_CALIBRATION.md).

## Historischer Stand nach #29, vor Kalibrierung

**Fertig:** [SMOTE-/Netzvergleich](./smote_nnet_2026_10_30/README.md),
[PR #29](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/29),
ursprünglicher Ergebnisbranch `codex/netzel-smote-nnet-comparison`.
Die Ergebnisse aus #18–#29 werden auf ausdrücklichen Nutzerauftrag gemeinsam
auf `main` gesichert ([Integrations-PR #30](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/30)); [gemeinsame Übergabe](./HANDOFF.md) und
[Integrationsnachweis](./consolidation_2026_10_30/README.md). Vorabdesign veröffentlicht
in `d47129c6b8be8bd5e73701f6137ffb196dab9d91`, bevor neuronale Fits begannen.

Artenmittel der Balanced Accuracy bei Schwelle 0,5: ohne SMOTE 71,64/69,20 %,
mit SMOTE 90,41/89,74 % (5°/10°-Blöcke, 100-km-Puffer). Gewinn in allen 14
Art×Block-Fällen. Flächengewichteter Brier dagegen im Artenmittel 0,03975/0,04438
ohne SMOTE und 0,07747/0,08245 mit SMOTE: Verschlechterung in allen 14 Fällen.
Die vorab vorgesehene Prior-Odds-Diagnose hilft, garantiert aber keine Kalibrierung
und schlägt das Netz ohne SMOTE im Brier nur in zwei von 14 Fällen.

14 vollständige neue Artennetze und alle Foldparameter gespeichert.
198.680 Art-Zell-Labels, 397.360 Testzeilen mit 1.589.440 Wahrscheinlichkeiten,
1.274.548 synthetische Elternpaare und sämtliche 462 Optimierungsstarts separat
geprüft. Zwei vollständige Läufe in allen zehn Ergebnisdateien bytegleich.
Sechs neue Kerntests bestanden. Gemeinsame GIS-/Numerikbibliotheken benannt.

**Aktuell:** Die Klassifikationsverbesserung aus der Dissertation wurde auf den
neuen sieben Arten nachgebildet. Experimentelle Arealindikatoren sind noch
keine validierten fossilen Auftretenswahrscheinlichkeiten. Keine automatische
Variantenübernahme oder Pollentyp-Zuordnung. Die #28-Referenz wird ausdrücklich
am gesperrten Commit `8f02794434f561f5b30228360247935f3c025cc8` bezogen; die ursprünglichen getrennten
PR-Branches enthielten nicht automatisch alle Vorarbeiten. Der gemeinsame
Integrationsstand enthält jetzt alle zwölf PR-Historien.

**Als Nächstes:** neue trainingsinterne Wahrscheinlichkeitskalibrierung mit inneren
räumlichen Teilungen und unabhängiger äußerer Bewertung vorab festlegen; weitere
Arten/Areale und eine belegte Typ–Art-/Gattungs-Zuordnung bearbeiten. Äußere
Testergebnisse dieses PR nicht zur weiteren Optimierung verwenden. Moderne
Kalibrierung kann parallel zur weiterhin offenen Monticchio-Quellenfrage laufen.

Die frühere Merge-Sperre wurde für die Zusammenführung #18–#29 durch den
aktuellen ausdrücklichen Nutzerauftrag aufgehoben. Das ändert keine fachlichen
Freigaben: keine stillen Quellenkorrekturen, Pilotfreigaben oder Autorennachrichten.
Wissenschaftliche Berichte, Quellenregister, Modellparameter und Ergebniswerte
werden unverändert erhalten. Der gemeinsame Einstieg ist [HANDOFF.md](./HANDOFF.md).
