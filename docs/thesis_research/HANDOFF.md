# Netzel-Projekt: aktueller Einstieg

Stand 30.10.2026. Repository: `TimonNetzel/kinneret_reconstruction_python`.
Ziel bleibt eine wissenschaftlich überprüfbare Europa-App für etwa 0–150 ka.
Die bisher geprüften Pilotdaten decken dieses Zielintervall noch nicht ab.

## Zuerst lesen

1. [Kompakter Claude-Auftrag](./baseline2025_2026_10_29/CLAUDE_TASK.md).
2. [Kleine Ergebnistabelle](./baseline2025_2026_10_29/metrics_summary.csv).
3. Nur bei Detailfragen: [Baseline-Bericht und Reproduktion](./baseline2025_2026_10_29/README.md).

Die moderne Baseline ist bereits berechnet und unabhängig geprüft. Claude
übernimmt nun den dünnen CLI-/App-Zugang und die Darstellung dieser Ergebnisse.
Quellenrecherche und Modellvergleich müssen für diesen Auftrag nicht neu
entwickelt werden. OpenAI betreut Datenprüfung, Baseline und Ergebnisprüfung.

## Repository- und Datenstand

- PR #14 integrierte den Importer; PR #15, #16 und #17 sind nach `main` gemergt.
- Die neue Baseline liegt auf `codex/netzel-modern-baseline`, ausgehend von
  `main`-Commit `e002f2b`. Vor Weiterarbeit den aktuellen PR-/Merge-Stand prüfen.
- Publizierte 2025-Kohorte: 7.634 moderne Proben; davon 7.078 mit mindestens
  100 gemeinsamen Zählungen ohne Pinus; 37 ausdrücklich gemeinsame Gruppen.
  Nenner der Teilzusammensetzung einschließlich Pinus. Die alte SQLite-Kohorte
  wird nicht hinzugefügt. [Datenvertrag](./modern2025_2026_10_29/README.md).
- Monticchio: 69 chronologisch zugeordnete Proben bis rund 20,2 ka BP.
  Altersgrenzen und Tiefenoffsets bleiben erhalten; vollständige Altersensembles
  und eine lange, durchgehend geprüfte 150-ka-Abdeckung fehlen.

## Berechnetes Ergebnis

Vorab festgelegtes 5-Nachbarn-Pollenmodell, fünf räumliche Folds je 5°-/10°-Gitter,
100-km-Puffer; Design und Code vor dem ersten Lauf in Commit `e594aa9` festgehalten.
Globaler moderner Tann-RMSE: 3,765/3,926 °C; Pann-RMSE: 365,2/373,1 mm/Jahr
(5°/10°). Gegenüber dem jeweiligen Trainingsmittel ist der Temperaturgewinn
deutlich, der Niederschlagsgewinn kleiner. Die eurasische Gesamtgüte ist keine
ausschließlich europäische Güte. Die regionale Diagnose ist eine nachgelagerte
Teilansicht derselben Vorhersagen, kein weiterer unabhängiger Test.

Zehn Fold-Aufteilungen, 56.624 Vorhersagezeilen und alle aggregierten Fehler sind
geprüft; 90 Analogvorhersagen zusätzlich unabhängig nachgerechnet. Zwei volle
Läufe waren bytegleich. Noch keine fossilen Klimavorhersagen oder Fehlerbänder.

## Regeln für die Fortsetzung

- Bestehenden Importer, seine Referenz und die ursprüngliche Kinneret-Kalibrierung
  nicht durch die Vergleichsvariante ersetzen. Eigene Branches/PRs verwenden.
- Jahresmitteltemperatur Tann und Kinneret-Wintertemperatur TDJF unterscheiden.
- Die verlorenen 279 Transferfunktionen sind nicht wiederhergestellt.
  Moderne Testfehler sind keine Unsicherheitsintervalle fossiler Rekonstruktionen.
- Keine Zählungen aus Prozenten erfinden; Labormarker und überlappende Untertypen
  nicht als unabhängige Pflanzensignale zählen. Quellen-/Policy-Hashes erhalten.
- Für spätere Multiproxy-Arbeit gemeinsame Chronologien und abhängige Reihen
  berücksichtigen. Ein feines Rechengitter schafft keine feinere Datenauflösung.
- Kurz im Chat berichten; ausführliche Ergebnisse und Quellen in GitHub halten.

## Aktueller Forschungsanschluss

Die Folgearbeiten sind auf eigenen Branches dokumentiert: PR #20 prüft die
Analogabdeckung, #21 die ältere Chronologie/Probenkorrespondenz, #22 den
Pilotplan, #23 die eingegangenen MON16-Supplemente und #24 die historischen
Marker. Der Claude-Integrations-PR #19 wurde unabhängig geprüft; keine
Integrationsdateien gleichzeitig bearbeiten. Vor Änderungen PR-Köpfe prüfen.

Der [neue EPD-Archivaudit](./epd_sample_identity_2026_10_30/README.md) bestätigt
alle 69 Datenbank-Probenkennungen und ihre exakten Mittelpunkttiefen. Acht
Pilotabweichungen um 5 mm sind Exportpräzision. Physische Kernabschnitte fehlen
im Archiv; die dort als Standard markierte ältere 14C-Chronologie darf die
beibehaltenen Cut-Alter nicht ersetzen. P1 bleibt offen, keine neuen MON16-Alter
oder fossilen Klimaintervalle. Der [konkret fehlende Quellenbeleg](./epd_sample_identity_2026_10_30/MISSING_SOURCE_REQUEST.md)
ist beschrieben. Die moderne Baseline bleibt unverändert.

## Hintergrund bei Bedarf

[Ursprünglicher Europa-Plan](./Claude_Plan_Europa_0_150ka.md),
[Dissertation und Datenverzeichnis](./Netzel_Analyse_Datenverzeichnis.md),
[Quellenregister](./source_audit_2026_10_29/sources.json).
Die ursprüngliche lange Übergabe ist in der Git-Historie erhalten.
