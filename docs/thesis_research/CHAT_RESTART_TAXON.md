# Fortsetzung nach dem ersten Taxon-Training

Stand 30.10.2026. Nutzer hat nach PR #27 die eigenständige Erstellung neuer
Taxon-Transferfunktionen beauftragt, parallel zur offenen Monticchio-Chronologie.
Arbeitsbasis war der offene #18 auf f9acc16; eigener Branch
`codex/netzel-taxon-response-functions`. Vor Weiterarbeit PR-/Merge-Stand prüfen.

Zuerst [PROJECT_STATUS.md](./PROJECT_STATUS.md) und
[Trainingsbericht](./taxon_transfer_2026_10_30/README.md) lesen. Design, Quellen
und Konfiguration wurden vor dem ersten Fit in `64120202f605406124326d0c06d42c29ac4aeb0b`
veröffentlicht. Sieben neue botanische **Artenfunktionen**, keine ursprünglichen
279 Funktionen, keine Biome und keine Gattungs-Pollentypen als Arten umgedeutet.

Modelle: Abies alba, Betula pendula, Corylus avellana, Fagus sylvatica,
Picea abies, Taxus baccata, Tilia cordata. Meusel/BfN-Areale vom gesperrten
Commit 5ab8e638; WorldClim 2.1 (1970–2000), BIO1/BIO12. Originale in sources.json
mit URL, Bytezahl und Hash. Keine alten temporären Dateien erforderlich;
Original-ScienceDirect-ZIP nicht benötigt.

Zwei ganze Läufe in acht Ausgaben bytegleich, separater Prüfer ohne
Produzentenimport bestätigt 198.680 Art-Zell-Labels, 397.360 Testvorhersagen,
70 räumliche Fold-Fits und sieben vollständige Fits. Gemeinsame GIS-/Numerik-
Bibliotheken benannt. Drei Kerntests in normaler Testsuite, zwei GIS-Tests.
Alle sieben Arten schlagen die Trainingsprävalenz in beiden Designs;
Abies alba nur schwach. Keine fossilen Klimafehler daraus ableiten.

Nächste Arbeit: weitere Areale und Zielarten, eine belegte Typ–Art-/Gattungs-
Zuordnung und Kalibrierungs-/Kartenunsicherheit. Neue Modellvarianten und
Sensitivitäten vor ihrer Auswertung festlegen. Vorwärtsevaluator sperrt
Extrapolation aus der Klimahülle. Keine automatische Anwendung auf Fossilproben.

P1 bleibt offen: physischer MON16-Anschluss und Korrelationsunsicherheit fehlen.
28 Kandidaten / 41 native Nichttreffer und alle 47 Provenienzfragen aus #26/#27
bleiben unverändert. Frühere getrennte Übergabe:
[PR #27](https://github.com/TimonNetzel/kinneret_reconstruction_python/blob/5fd3bad46d5108e8986c054171fba66a05c26589/docs/thesis_research/CHAT_RESTART_VARIABLES.md),
dort weiter zu #25/#26. Claude betreut CLI/App, Codex Kalibrierung, Quellen und
unabhängige Prüfung. Keine Baseline-Neuentwicklung, stillen Quellenkorrekturen,
Pilotfreigaben, Autorennachrichten oder Merges. Kurze deutsche Updates.
