# Netzel-Projekt: kurzer Arbeitsstand

Stand 30.10.2026. Vor Fortsetzung aktuellen PR-/Merge-Stand prüfen.

**Ziel:** App für taxonbasierte Multiproxy-Klimarekonstruktionen europäischer
Fundstellen mit belegter Kalibrierung und Unsicherheiten.

| Bereich | Stand |
| --- | --- |
| Moderne Analog-Baseline #18 | Fertig, unverändert; weiterhin offene Arbeitsbasis. |
| Neue Taxon-Transferfunktionen | **Begonnen und erster Trainingsstand fertig:** sieben botanische Artenfunktionen aus Meusel-Arealen und WorldClim 2.1 angepasst und räumlich geprüft. |
| Taxonverfahren insgesamt | Noch offen: weitere Arten, Fossiltyp-Zuordnung, Kalibrierungsunsicherheit und gemeinsame Beobachtungsmodelle. Ursprüngliche 279 Funktionen nicht reproduziert. |
| Monticchio-P1 | Weiter offen: physischer LGM90D→MON16-Crosswalk und Korrelationsunsicherheit fehlen. |
| Proben-/Quellenstand aus #26/#27 | Unverändert 28 numerische Kandidaten und 41 native Nichttreffer; 47 Herkunftsfragen ohne ursprüngliche Einreichung/Umcodierungsbelege offen. |
| Fossilrekonstruktion / Multiproxy-App | Kein abgeschlossener Gesamtdurchlauf und keine neue fossile Klimakurve. P2 nicht freigegeben. Claude betreut CLI/App. |

**Fertig bei Codex:** [sieben neue Artenfunktionen](./taxon_transfer_2026_10_30/README.md)
mit Koeffizienten, Vorwärtsevaluator, 5°-/10°-Prüfung und Trainingsprävalenz als
Vergleich. Vorabdesign im veröffentlichten Commit `64120202f605406124326d0c06d42c29ac4aeb0b`.
Zwei vollständige Läufe bytegleich, 397.360 Testvorhersagen und 198.680
Art-Zell-Labels separat bestätigt. Fünf neue Tests bestanden.
Eigenständiger Branch: `codex/netzel-taxon-response-functions`,
[PR #28](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/28).
Geprüfter Ergebniscommit: `78ab48aab1a5a81614def6683fb2a0fa78eb4845`.

**Aktuell:** Experimentelle Kartenfunktionen für sieben **Arten**, noch keine
automatische Brücke zu Gattungs-Pollentypen. Abies alba bringt gegenüber
konstanter Prävalenz nur wenig Gewinn. Karten-CV liefert keine fossilen
Klimafehler oder vollständige Kalibrierungsunsicherheit.

**Als Nächstes:** Zielarten und zugehörige Areale erweitern, Typ–Art-/Gattungs-
Zuordnung dokumentieren und Unsicherheits-/Sensitivitätsprüfung vorab festlegen.
Monticchio-Chronologie bleibt ein getrennter offener Quellenstrang; sie blockiert
das moderne Training nicht. Fossile Anwendung erst nach fachlicher Abnahme.

PRs #18–#27 beim Beginn dieser Arbeit offen und ungemergt; #19 unverändert
auf `c8977a5`. Frühere Forschungsstände bleiben auf getrennten Branches, siehe
[Übergabe](./CHAT_RESTART_TAXON.md). Keine Änderungen an Baseline, vorhandenen
Altern, Quellenwerten oder Claudes Dateien. Keine Autorennachrichten oder Merges.
