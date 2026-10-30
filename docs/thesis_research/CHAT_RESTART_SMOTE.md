> Aktualisierung 30.10.2026: Die Zusammenführung von #18–#29 ist ausdrücklich
> beauftragt. Gemeinsamer Einstieg: [HANDOFF.md](./HANDOFF.md); nach Merge `main`
> verwenden. Nachfolgend bleibt der historische Paketstand dokumentiert.

# Fortsetzung nach dem SMOTE-/Netzvergleich

Stand 30.10.2026. Vor jeder Fortsetzung PR-/Merge-Stand prüfen.
Nutzerauftrag nach #28: den in seiner Dissertation erfolgreichen Ansatz
SMOTE + einfaches neuronales Netz nachbilden. Ergebnis-PR:
[PR #29](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/29),
Branch `codex/netzel-smote-nnet-comparison`, Basis weiterhin offener #18 auf
`f9acc16c58a47a51b1063690a69a2db01381c76a`.

Zuerst [PROJECT_STATUS.md](./PROJECT_STATUS.md),
[Ergebnisbericht](./smote_nnet_2026_10_30/README.md),
[vorab veröffentlichtes Design](./smote_nnet_2026_10_30/DESIGN.md) und
[Reproduktion](./smote_nnet_2026_10_30/REPRODUCE.md) lesen.
Designcommit **d47129c6b8be8bd5e73701f6137ffb196dab9d91** lag vor dem ersten
neuronalen Fit vor; keine anschließende Hyperparameter-/Schwellenwahl.

Der neue Branch enthält **nicht automatisch die getrennten PR-Vorarbeiten**.
#28 ist ausdrücklich am Commit `8f02794434f561f5b30228360247935f3c025cc8`
gesperrt; `sources.json` hält die sieben benötigten Dateien fest und
`audit_sources.json` den achten unabhängigen Rohdatenprüfer. Vorbereitungsskript
holt alle benötigten Quellen neu und prüft die Hashes; keine alten temporären
Dateien und kein ScienceDirect-ZIP für diesen modernen Vergleich nötig.

Neue Modelle: sieben botanische Arten, je ein 2–6–1-Netz ohne und mit SMOTE.
Gleiche Meusel-/WorldClim-Daten, Masken, Area-Gewichte und gepufferte äußere
5°-/10°-Tests wie #28. Normierung original-trainingsintern, SMOTE nur aus
Minderheits-Trainingszellen. Alle Elternpaare werden aufgezeichnet und geprüft.
Feste Referenz #28 und diagnostische Prior-Odds-Anpassung ebenfalls ausgewertet.
Die Ausgaben sind Karten-Arealindikatoren, keine bereits validierten fossilen
Auftretenswahrscheinlichkeiten. BA-Gewinn und Wahrscheinlichkeitsgüte getrennt
beurteilen; keine automatische Variantenübernahme.

Keine bitgleiche historische R-/caret-Reproduktion und keine ursprünglichen
279 Taxonfunktionen. Sechs Neuronen sind hier vorab festgelegt, in der Quelle
für ein Biomnetz beschrieben, nicht als verifizierte Einstellung aller
historischen Arten ausgegeben. Keine automatische Brücke von Arten zu
Gattungs-Pollentypen. Drei Starts pro Fit dienen der Trainingsoptimierung,
liefern keine vollständige Kalibrierungsunsicherheit.

Als Nächstes: auf Grundlage der vollständigen Vergleichstabelle eine neue,
trainingsinterne Wahrscheinlichkeitskalibrierung und Unsicherheitsprüfung
vorab festlegen, oder weitere Areale/Zielarten und die belegte Typ–Art-Zuordnung
bearbeiten. Äußere Testwerte aus #29 nicht zur weiteren Optimierung verwenden;
neue Kalibrierung braucht innere räumliche Teilungen und unabhängige äußere
Bewertung. Bestehende CLI/App bleibt bei Claude, Codex betreut Daten und Prüfung.

P1 weiter offen: physischer LGM90D→MON16-Crosswalk und Korrelationsunsicherheit
fehlen. 28 numerische Kandidaten / 41 native Nichttreffer und die 47 offenen
Provenienzfragen aus #26/#27 unverändert. Keine fossile Freigabe, stillen
Quellenkorrekturen, Autorennachrichten oder Merges.

Frühere Übergaben ausdrücklich über gesperrte Branchstände lesen:

- [#28: Artenfunktionen](https://github.com/TimonNetzel/kinneret_reconstruction_python/blob/8f02794434f561f5b30228360247935f3c025cc8/docs/thesis_research/CHAT_RESTART_TAXON.md)
- [#27: Variablenherkunft](https://github.com/TimonNetzel/kinneret_reconstruction_python/blob/5fd3bad46d5108e8986c054171fba66a05c26589/docs/thesis_research/CHAT_RESTART_VARIABLES.md)
- [#26: Neotoma-Quelle](https://github.com/TimonNetzel/kinneret_reconstruction_python/blob/4c9f1150e766a7f855dc9e28fd121638a710ff9a/docs/thesis_research/CHAT_RESTART_PR26.md)
- [#25: ältere Übergabe](https://github.com/TimonNetzel/kinneret_reconstruction_python/tree/00c9f037642470c7d1a9a44e488f062109292c2d/docs/thesis_research)

Kurze deutsche Updates: fertig / aktuell / als Nächstes. #18 nicht neu entwickeln.
