# Netzel-Projekt: aktueller Einstieg

Stand 30.10.2026. Aktueller Auftrag: SMOTE + einfaches neuronales Netz nach der
Dissertation für die sieben neuen Artenfunktionen nachbilden und unabhängig
gegen #28 sowie dasselbe Netz ohne SMOTE prüfen.

Zuerst [CHAT_RESTART_SMOTE.md](./CHAT_RESTART_SMOTE.md),
[PROJECT_STATUS.md](./PROJECT_STATUS.md) und den
[Ergebnisbericht](./smote_nnet_2026_10_30/README.md) lesen.
Ergebnis und Reproduktion liegen auf eigenem Branch in
[PR #29](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/29).

Die moderne Analog-Baseline #18 ist fertig. Solange offen, ihren Branch als
Basis verwenden; nach Merge `main`. #29 verwendet die Daten und Referenz aus
dem getrennten #28 ausdrücklich mit gesperrten Dateien. Frühere PR-Branches
enthalten nicht automatisch alle Vorarbeiten. Keine alten temporären Dateien
nötig; kein Original-ScienceDirect-ZIP für diesen modernen Vergleich.

Noch keine vollständige taxonbasierte Multiproxy-Fossilrekonstruktion, keine
Wiederherstellung der ursprünglichen 279 Modelle. Sieben botanische Arten
sind keine automatische Zuordnung zu Gattungs-Pollentypen. Klassifikationsgüte
und kalibrierte Wahrscheinlichkeiten getrennt prüfen.

P1 bleibt offen: physischer LGM90D→MON16-Anschluss samt Korrelationsunsicherheit.
28 numerische Kandidaten / 41 native Nichttreffer / 47 offene Herkunftsfragen
bleiben bestehen. Keine stillen Quellenkorrekturen, Pilotfreigaben,
Autorennachrichten oder Merges. Claude betreut CLI/App; Codex Daten und Prüfung.

Die historische Übergabe wird nicht rückwirkend umgeschrieben:
[HANDOFF aus #25](https://github.com/TimonNetzel/kinneret_reconstruction_python/blob/00c9f037642470c7d1a9a44e488f062109292c2d/docs/thesis_research/HANDOFF.md),
[CHAT_RESTART aus #25](https://github.com/TimonNetzel/kinneret_reconstruction_python/blob/00c9f037642470c7d1a9a44e488f062109292c2d/docs/thesis_research/CHAT_RESTART.md).
Weitere unveränderliche PR-Fundstellen stehen in der aktuellen Übergabe.
