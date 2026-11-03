# Dissertationsnachbau in der App, Stand 4: Fehlende TF und `num_taxa`-Priors

Stand 02.11.2026. Bestehender Branch `dissertation-reconstruction`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
Ausgangscommit `7d3d0f6` (Checkout sauber, identisch mit dem PR-Kopf, PR offen,
nicht gemergt, laut GitHub `MERGEABLE`). Ausschließlich der vom Nutzer benannte
vorhandene Checkout, ohne Klon, Worktree oder Subagenten.
Auftrag: [Claude-Übergabe nach dem Gamma-Raster](../CLAUDE_NEXT_STEP_AFTER_GAMMA_GRID_2026_11_02.md).

## Ergebnis

Neuer datierter Nachbaustand [reproduction_state.json](reproduction_state.json)
(SHA-256 `7b7bf1f449a1f9b4c71f44cff6b3b37912d1fe50263eae80a85b3ba8b1137195`),
Quellenlinks auf Commit `9bb8ed9` mit dem neuen
[Paket fehlende TF und `num_taxa`-Priors](../literal_tf_missing_priors_2026_11_02/README.md).
Als **neuer letzter Eintrag** in `nachbaustand.STATES`. Die früheren Stände bleiben
**bytegleich eingefroren** und werden weiter per Hash geprüft:
[Stand 1](../app_reproduction_2026_11_02/README.md) (`c358b2c3…`),
[Stand 2](../app_reproduction_2026_11_02_pruning/README.md) (`3656316d…`) und
[Stand 3](../app_reproduction_2026_11_02_gamma_grid/README.md) (`43edfcb6…`). Der
`STATES`-Eintrag von Stand 3 erhielt wie vorgesehen nur seinen Linkcommit `7d3d0f6`
(gepusht, enthält die unveränderte Zustandsdatei); Pfad, Pin und Modulliste sind unverändert.

Die Ansicht **Forschungsstand → Dissertationsnachbau** zeigt jetzt:

- unveränderten Status „Teilmodule geprüft; vollständiger Originalfall offen“;
- **drei** Zeilen „Früherer Stand eingefroren“ (5/251, 6/297 und 7 Teilmodule/352 Tests);
- **acht** Teilmodule und **437** Teil-/Belegtests. Neu: „Entfernung fehlender TF und
  Taxonzahl-Priors“ (85 Tests, Quellen `mcmc_prepare.R`, `main_v1.2.R`) – **ohne**
  Etikett „Bedingter Schritt“, weil 1664–1665 und 1681–1685 im gebundenen Hauptlauf
  unbedingt laufen (die Entfernung selbst ist datenabhängig geschützt); das
  Taxon-Pruning bleibt als einziger bedingter Schritt markiert;
- unverändert die elf offenen Originaleingaben aus dem Belegvertrag;
- im weiterhin eingeklappten **synthetischen** Beispiel eine neue Tabelle mit vier
  Kunstfällen als nummerierten Spalten (passt auf 390 px): Taxa vorher, fehlende TF
  (R-Positionen), NA nur im x2-Mittel, bleibende Kunsttaxa als Buchstaben, `num_taxa`,
  `round(n/4)` laut Quelle (fett), kaufmännisch aufgerundet (abweichende Werte farbig),
  `phi = n/2` und das unveränderte `phi1` aus Zeile 1440 (farbig, weil veraltet).
  Darunter die Erklärung (Reihenfolge, nur x1-Maske, Rundung zur geraden Ziffer,
  Abweichung für n = 2, 10, 18, 26, 34) und der Satz zur leeren Maske (keine
  Indizierung, `num_taxa` 5 aus Zeile 1435; ohne Schutz leerte `-integer(0)` alles).
  Die Kunstkette heißt jetzt „ohne Taxon-Pruning, Gamma-Raster und TF-Entfernung“.

Keine Erfolgsanzeige aus Testzahlen; Kunstbeispiele fließen in keinen Lauf, keinen
Laufkatalog und keine Rekonstruktion ein. Die NA-Maske ist im Beispiel vorgegeben
und wird nie aus `weightedMedian` abgeleitet.

## Datenbindung

[nachbaustand.py](../../../src/kinneret_recon/nachbaustand.py) liest den neuesten Stand,
prüft alle drei früheren Stände gegen Pin, Freigabeflags und Modulliste und übernimmt
das neue Beispiel aus vier gepinnten Dateien des Pakets. Positionen, Namen, `num_taxa`,
Quellzeile, `round(n/4)`, `phi` und `phi1`/`phi2` müssen **exakt** mit der unabhängigen
Referenz übereinstimmen, `round(n/4)` zusätzlich mit der Handerwartung.

| Fall | Verhalten |
| --- | --- |
| Neuester Stand, ein früherer Stand oder ein gebundener Beleg fehlt | `available: false` mit Liste fehlender Pfade |
| Bytes einer gebundenen Datei (auch aller drei eingefrorenen Stände) geändert | Build bricht ab (`hash mismatch`) |
| Freigabeflag, gewählte Variante, R-Ausführung, abweichende Referenz-/Handwerte | Build bricht ab |

## Tatsächlich ausgeführte Prüfungen

[validation.json](validation.json) enthält Befehle, Zahlen und lokale Protokoll-Hashes.

- **App-/Daten-/Buildtests:** `tests/test_nachbaustand.py`, `test_forschungsstand.py`,
  `test_app.py`, `test_artenvergleich.py` → **54 bestanden, 2 übersprungen**
  (Python-Browserwrapper ohne Playwright im Standard-Node-Pfad; unten explizit ersetzt).
  7 neue Testfälle (TF-Beispiel, Bytewechsel an Stand 3 und fünf Paketdateien).
- **Historische Teil-/Belegtests:** zehn gezielte Dateien → **437 bestanden**
  (352 bisherige + 85 neue). Beide Phase-1-Prüfer weiter Exit 2.
- **Browser, offline, Edge über das Playwright 1.62.1 der Codex-Laufzeit**
  (`NODE_PATH`, `--kanal msedge`): Ordner- und Ein-Datei-App × Handy/PC ×
  Hell/Dunkel, Gesten, DE/EN, Aufklappzustand beim Sprachwechsel,
  Fehlstandskontrollen, 18 Links (15 auf `9bb8ed9`, je einer auf die drei eingefrorenen
  Stände), sieben Tabellen, sechs markierte Abweichungszellen der neuen Tabelle,
  Seiten- und Tabellenüberlauf. **Erster Lauf:** auf Handybreite war die neue Tabelle
  breiter als ihr Rahmen (Handyfälle; alle PC-Fälle ok). Zeilenköpfe gekürzt, Zeilennummern und
  Gegenlesart in den Begleittext verlegt → zweiter Lauf alle 10 Fälle **ok**.
- **Zwei Standorte** (Kopie über `tests/test_app.py::_zweiter_standort`,
  `--zwei-standorte --kanal msedge`) → **ok**.
- Visuell geprüft: [Mobil hell](mobile.png) und [Desktop dunkel](desktop.png).
- **Nicht ausgeführt:** Gesamtsuite `pytest tests` (Sampler-Unittests),
  `literal_tf_preparation_2026_11_02/verify_packet.py` und die zeitpunktgebundenen
  `tools/verify_*` (keine Regressionsprüfer), R, Trainings, Sampler, Altersmodelle,
  Fossilrekonstruktionen.

```bash
PYTHONPATH=src python -m pytest tests/test_nachbaustand.py tests/test_forschungsstand.py tests/test_app.py tests/test_artenvergleich.py -q
kinneret-recon app
kinneret-recon app --eine-datei
NODE_PATH=<Codex-Laufzeit>/dependencies/node/node_modules node tools/app_pruefen.js results/app/index.html results/kinneret-app.html --kanal msedge
node tools/app_pruefen.js --zwei-standorte <Kopie>/index.html --kanal msedge
```

## Grenzen

Vierte App-Anbindung des geprüften Nachbaus, **keine historische Fallreproduktion**.
Alle elf Nachweisbündel bleiben offen, keine Variante gewählt. Baseline, CLI, sieben
experimentelle Artenmodelle, Paperreferenz, Forschungs-Snapshot und fallbezogene
Monticchio-/Chronologie-Gates unverändert. Keine neuen Altersplots; jung links,
alt rechts bleibt.

Nächste Übergabe: [Claude-Auftrag nach der TF-Entfernung](../CLAUDE_NEXT_STEP_AFTER_TF_MISSING_2026_11_02.md).
