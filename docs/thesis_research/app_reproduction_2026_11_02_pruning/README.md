# Dissertationsnachbau in der App, Stand 2: Taxon-Pruning

Stand 02.11.2026. Bestehender Branch `dissertation-reconstruction`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
Ausgangscommit `47a926c` (Checkout sauber, identisch mit GitHub, PR offen und
nicht gemergt). Ausschließlich der vom Nutzer benannte vorhandene Checkout,
ohne Klon, Worktree oder Subagenten.
Auftrag: [Claude-Übergabe nach der App-Anbindung](../CLAUDE_NEXT_STEP_AFTER_APP_REPRODUCTION_2026_11_02.md).

## Ergebnis

Neuer datierter Nachbaustand [reproduction_state.json](reproduction_state.json)
(SHA-256 `3656316de4211500184f1a1c772bf07c87e3bd4b2d210ed6316e70c5a8ca0784`),
Quellenlinks auf Commit `f3b71bb` mit dem neuen
[Taxon-Pruning-Paket](../literal_taxon_pruning_2026_11_02/README.md).
Der [Stand vom 02.11.2026](../app_reproduction_2026_11_02/README.md) bleibt
**bytegleich eingefroren** (`c358b2c3…`) und wird weiter per Hash geprüft.

Die Ansicht **Forschungsstand → Dissertationsnachbau** zeigt jetzt:

- unveränderten Status „Teilmodule geprüft; vollständiger Originalfall offen“;
- eine Zeile „Früherer Stand eingefroren: 2026-11-02, 5 Teilmodule, 251 Tests,
  SHA-256 c358b2c3382f…“ mit Link auf die Datei im Commit `47a926c`;
- **sechs** Teilmodule (bisherige fünf + Taxon-Pruning, 46 Tests) und **297**
  Teil-/Belegtests; das Pruning trägt das Etikett **„Bedingter Schritt“** und
  sagt, dass es im gebundenen Hauptlauf abgeschaltet ist
  (`null.ratio.check = FALSE`) und `weightedMedian`, Splines und Zufallsschritte
  offen bleiben;
- unverändert die elf offenen Originaleingaben aus dem Belegvertrag;
- im weiterhin eingeklappten **synthetischen** Beispiel zusätzlich eine
  Pruning-Tabelle: 20 Kunstproben, 6 Kunsttaxa, Nullen `0/2/5/10/10/3`,
  Nullanteile `0, 1/10, 1/4, 1/2, 1/2, 3/20` (exakt aus der Bruchreferenz),
  entfernt werden beide `1/2`-Taxa. Ein Satz erklärt den abwärts zählenden
  R-Ausdruck `16:T` und dass naive Lesarten hier nichts entfernen würden.
  Die Kunstkette heißt jetzt ehrlich „über die ersten fünf Teilmodule (ohne
  Taxon-Pruning)“.

Keine Erfolgsanzeige aus Testzahlen; Kunstbeispiele fließen in keinen Lauf,
keinen Laufkatalog und keine Rekonstruktion ein.

## Datenbindung

[nachbaustand.py](../../../src/kinneret_recon/nachbaustand.py) führt jetzt eine
Liste `STATES` (älteste zuerst, je eigener Ordner und Pin). Angezeigt wird der
neueste Stand; ältere Stände werden geladen, gegen Pin, Freigabeflags und
Modulliste geprüft und nur als eingefrorene Zeile gezeigt. Das Pruning-Beispiel
wird aus vier gepinnten Dateien des Pakets übernommen; Indizes, Zweig und
Steuerwerte müssen exakt mit der Bruchreferenz übereinstimmen, die Nullanteile
zusätzlich mit `Nullen / Zeilen` der Fixture.

| Fall | Verhalten |
| --- | --- |
| Neuester Stand, früherer Stand oder ein gebundener Beleg fehlt | `available: false` mit Liste fehlender Pfade |
| Bytes einer gebundenen Datei (auch des eingefrorenen Stands) geändert | Build bricht ab (`hash mismatch`) |
| Freigabeflag, gewählte Variante, R-Ausführung, abweichende Pruning-Indizes | Build bricht ab |

## Tatsächlich ausgeführte Prüfungen

[validation.json](validation.json) enthält Befehle, Zahlen und lokale Protokoll-Hashes.

- **App-/Daten-/Buildtests:** `tests/test_nachbaustand.py`, `test_forschungsstand.py`,
  `test_app.py`, `test_artenvergleich.py` → **40 bestanden, 2 übersprungen**
  (Python-Browserwrapper ohne Playwright im Standard-Node-Pfad; explizit ersetzt).
  6 neue Testfälle (eingefrorener Stand, Pruning-Beispiel, vier Bytewechsel).
- **Historische Teil-/Belegtests:** acht gezielte Dateien → **297 bestanden**
  (251 bisherige + 46 neue). Beide Phase-1-Prüfer weiter Exit 2.
- **Browser, offline, Edge über das Playwright der Codex-Laufzeit**
  (`NODE_PATH`, `--kanal msedge`): Ordner- und Ein-Datei-App × Handy/PC ×
  Hell/Dunkel, Gesten, DE/EN, Aufklappzustand beim Sprachwechsel,
  Fehlstandskontrollen, Seiten- und Tabellenüberlauf → alle 10 Fälle **ok**.
  Erster Lauf fand eine zu breite Pruning-Tabelle auf dem Handy; nach
  nummerierten Spaltenköpfen alle Fälle erneut ausgeführt und bestanden.
- **Zwei Standorte** (Kopie mit Standort „zweit“ wie in `tests/test_app.py`,
  `--zwei-standorte --kanal msedge`) → **ok**.
- Visuell geprüft: [Mobil](mobile.png) (Pruning-Tabelle) und [Desktop dunkel](desktop.png).
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

Zweite App-Anbindung des geprüften Nachbaus, **keine historische
Fallreproduktion**. Alle elf Nachweisbündel bleiben offen, keine Variante
gewählt. Baseline, CLI, sieben experimentelle Artenmodelle, Paperreferenz,
Forschungs-Snapshot und fallbezogene Monticchio-/Chronologie-Gates unverändert.
Keine neuen Altersplots; jung links, alt rechts bleibt.

Nächste Übergabe: [Claude-Auftrag nach dem Taxon-Pruning](../CLAUDE_NEXT_STEP_AFTER_TAXON_PRUNING_2026_11_02.md).
