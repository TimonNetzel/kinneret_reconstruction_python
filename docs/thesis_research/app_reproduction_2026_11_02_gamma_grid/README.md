# Dissertationsnachbau in der App, Stand 3: Gamma-Prior-Raster

Stand 02.11.2026. Bestehender Branch `dissertation-reconstruction`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
Ausgangscommit `94f81cc` (Checkout sauber, identisch mit dem PR-Kopf, PR offen,
nicht gemergt, laut GitHub `MERGEABLE`). Ausschließlich der vom Nutzer benannte
vorhandene Checkout, ohne Klon, Worktree oder Subagenten.
Auftrag: [Claude-Übergabe nach dem Taxon-Pruning](../CLAUDE_NEXT_STEP_AFTER_TAXON_PRUNING_2026_11_02.md).

## Ergebnis

Neuer datierter Nachbaustand [reproduction_state.json](reproduction_state.json)
(SHA-256 `43edfcb67a64826ce72611c93be33490463e59566934393aeff952bd217fbf4f`),
Quellenlinks auf Commit `95bd0ef` mit dem neuen
[Gamma-Prior-Raster-Paket](../literal_gamma_prior_grid_2026_11_02/README.md).
Als **neuer letzter Eintrag** in `nachbaustand.STATES`. Die früheren Stände bleiben
**bytegleich eingefroren** und werden weiter per Hash geprüft:
[Stand 1](../app_reproduction_2026_11_02/README.md) (`c358b2c3…`, Link auf `47a926c`) und
[Stand 2](../app_reproduction_2026_11_02_pruning/README.md) (`3656316d…`). Der
`STATES`-Eintrag von Stand 2 erhielt wie vorgesehen nur seinen Linkcommit `3d013b2`
(gepusht, enthält die unveränderte Zustandsdatei); Pfad, Pin und Modulliste sind unverändert.

Die Ansicht **Forschungsstand → Dissertationsnachbau** zeigt jetzt:

- unveränderten Status „Teilmodule geprüft; vollständiger Originalfall offen“;
- **zwei** Zeilen „Früherer Stand eingefroren“ (5 Teilmodule/251 Tests und 6/297);
- **sieben** Teilmodule und **352** Teil-/Belegtests. Neu: „Gamma-Prior über ein
  Formraster“ (55 Tests, Quellen `mcmc_prepare.R`, `mcmc_prep_v1.2.R`) – **ohne**
  Etikett „Bedingter Schritt“, weil die Rasterblöcke im gebundenen Hauptlauf unbedingt
  laufen; das Taxon-Pruning bleibt als einziger bedingter Schritt markiert;
- unverändert die elf offenen Originaleingaben aus dem Belegvertrag;
- im weiterhin eingeklappten **synthetischen** Beispiel eine Gamma-Tabelle mit vier
  Kunstfällen als nummerierten Spalten (passt auf 390 px): erfundene Mittel/Streuung,
  `(Mittel/Streuung)²`, nächster Rasterpunkt (naiv, abweichende Werte farbig), Form laut
  Quelle, R-Index und Rate. Darunter die Erklärung (Double-Rasterwert
  `25,000000000000004`, unterer statt nächster Punkt, Sprung auf `0,1` jenseits des
  Rasters) und ein Kettensatz: Verdopplung der Streuung, Rate aus Zeile 1281
  `0,012449899597988732`, durch den Rezentzweig (1733) ersetzt durch `0,0124`.
  Die Kunstkette heißt jetzt „ohne Taxon-Pruning und Gamma-Raster“.

Keine Erfolgsanzeige aus Testzahlen; Kunstbeispiele fließen in keinen Lauf, keinen
Laufkatalog und keine Rekonstruktion ein.

## Datenbindung

[nachbaustand.py](../../../src/kinneret_recon/nachbaustand.py) liest den neuesten Stand,
prüft beide früheren Stände gegen Pin, Freigabeflags und Modulliste und übernimmt das
Gamma-Beispiel aus vier gepinnten Dateien des Pakets. Index, Form, Rate und `rate1`
müssen **bitgleich** mit der unabhängigen Referenz sein, der R-Index mit der
Handerwartung; die Kette muss vollständig mit der Referenz übereinstimmen.

| Fall | Verhalten |
| --- | --- |
| Neuester Stand, ein früherer Stand oder ein gebundener Beleg fehlt | `available: false` mit Liste fehlender Pfade |
| Bytes einer gebundenen Datei (auch beider eingefrorener Stände) geändert | Build bricht ab (`hash mismatch`) |
| Freigabeflag, gewählte Variante, R-Ausführung, abweichende Referenz-/Handwerte | Build bricht ab |

## Tatsächlich ausgeführte Prüfungen

[validation.json](validation.json) enthält Befehle, Zahlen und lokale Protokoll-Hashes.

- **App-/Daten-/Buildtests:** `tests/test_nachbaustand.py`, `test_forschungsstand.py`,
  `test_app.py`, `test_artenvergleich.py` → **47 bestanden, 2 übersprungen**
  (Python-Browserwrapper ohne Playwright im Standard-Node-Pfad; unten explizit ersetzt).
  7 neue Testfälle (Gamma-Beispiel, Bytewechsel an Stand 2 und fünf Gamma-Dateien).
- **Historische Teil-/Belegtests:** neun gezielte Dateien → **352 bestanden**
  (297 bisherige + 55 neue). Beide Phase-1-Prüfer weiter Exit 2.
- **Browser, offline, Edge über das Playwright 1.62.1 der Codex-Laufzeit**
  (`NODE_PATH`, `--kanal msedge`): Ordner- und Ein-Datei-App × Handy/PC ×
  Hell/Dunkel, Gesten, DE/EN, Aufklappzustand beim Sprachwechsel,
  Fehlstandskontrollen, 15 Links (13 auf `95bd0ef`, je einer auf die eingefrorenen
  Stände), sechs Tabellen, drei markierte naive Abweichungen, Seiten- und
  Tabellenüberlauf → alle 10 Fälle **ok** im ersten Lauf.
- **Zwei Standorte** (Kopie mit Standort „zweit“ wie in `tests/test_app.py`,
  `--zwei-standorte --kanal msedge`) → **ok**.
- Visuell geprüft: [Mobil](mobile.png) (Gamma-Tabelle) und [Desktop dunkel](desktop.png).
- **Nicht ausgeführt:** Gesamtsuite `pytest tests` (Sampler-Unittests),
  `literal_tf_preparation_2026_11_02/verify_packet.py` (kein Regressionsprüfer),
  R, Trainings, Sampler, Altersmodelle, Fossilrekonstruktionen.

```bash
PYTHONPATH=src python -m pytest tests/test_nachbaustand.py tests/test_forschungsstand.py tests/test_app.py tests/test_artenvergleich.py -q
kinneret-recon app
kinneret-recon app --eine-datei
NODE_PATH=<Codex-Laufzeit>/dependencies/node/node_modules node tools/app_pruefen.js results/app/index.html results/kinneret-app.html --kanal msedge
```

## Grenzen

Dritte App-Anbindung des geprüften Nachbaus, **keine historische Fallreproduktion**.
Alle elf Nachweisbündel bleiben offen, keine Variante gewählt. Baseline, CLI, sieben
experimentelle Artenmodelle, Paperreferenz, Forschungs-Snapshot und fallbezogene
Monticchio-/Chronologie-Gates unverändert. Keine neuen Altersplots; jung links,
alt rechts bleibt.

Nächste Übergabe: [Claude-Auftrag nach dem Gamma-Prior-Raster](../CLAUDE_NEXT_STEP_AFTER_GAMMA_GRID_2026_11_02.md).
