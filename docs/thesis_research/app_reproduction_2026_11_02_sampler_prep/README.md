# Dissertationsnachbau in der App, Stand 5: Sampler-Vorbereitung

Stand 02.11.2026. Bestehender Branch `dissertation-reconstruction`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).

- Ausgangscommit `0a3dcdb`: Checkout sauber, identisch mit dem PR-Kopf, PR offen, nicht gemergt,
  laut GitHub `MERGEABLE`.
- Ausschließlich der vom Nutzer benannte vorhandene Checkout, ohne Klon, Worktree oder Subagenten.
- Auftrag: [Claude-Übergabe nach den fehlenden TF](../CLAUDE_NEXT_STEP_AFTER_TF_MISSING_2026_11_02.md).

## Ergebnis

Neuer datierter Nachbaustand [reproduction_state.json](reproduction_state.json)
(SHA-256 `61c6f23142df9c40fad9d271803eb6dc0a2dce4082ee77c5058895d0f5110ccb`), als **neuer letzter
Eintrag** in `nachbaustand.STATES`. Quellenlinks zeigen auf Commit `00ab021` mit dem neuen
[Paket Sampler-Vorbereitung](../literal_sampler_preparation_2026_11_02/README.md).

Die früheren Stände bleiben **bytegleich eingefroren** und werden weiter per Hash geprüft:
[Stand 1](../app_reproduction_2026_11_02/README.md) (`c358b2c3…`),
[Stand 2](../app_reproduction_2026_11_02_pruning/README.md) (`3656316d…`),
[Stand 3](../app_reproduction_2026_11_02_gamma_grid/README.md) (`43edfcb6…`) und
[Stand 4](../app_reproduction_2026_11_02_tf_missing/README.md) (`7b7bf1f4…`).
Der `STATES`-Eintrag von Stand 4 erhielt nur seinen Linkcommit `0a3dcdb`. Er ist mit
`git log -1 --format=%H -- …/app_reproduction_2026_11_02_tf_missing/reproduction_state.json` bestätigt,
gepusht und enthält die unveränderte Zustandsdatei. Pfad, Pin und Modulliste sind unverändert.

Die Ansicht **Forschungsstand → Dissertationsnachbau** zeigt jetzt:

- unveränderten Status „Teilmodule geprüft; vollständiger Originalfall offen“;
- **vier** Zeilen „Früherer Stand eingefroren“ (5/251, 6/297, 7/352 und 8 Teilmodule/437 Tests);
- **neun** Teilmodule und **555** Teil-/Belegtests. Neu ist „Sampler-Vorbereitung aus Taxonzahl und
  Taxonmatrix“ (118 Tests, Quellen `mcmc_conduction.R`, `main_v1.2.R`).
  - Das Modul trägt **kein** Etikett „Bedingter Schritt“: Die Zeilen 53–79 laufen im gebundenen
    Hauptlauf unbedingt, und `gibbs <- F` ist dort fest gesetzt.
  - Das Taxon-Pruning bleibt der einzige bedingte Schritt.
- unverändert die elf offenen Originaleingaben aus dem Belegvertrag;
- im weiterhin eingeklappten **synthetischen** Beispiel zwei neue Tabellen, beide passend auf 390 px:
  - **Matrixbeispiel** mit Kunsttaxa als Spalten: Maximum, drei normierte Zeilen als exakte Brüche
    (Double als Tooltip) und `colMeans` der Rohmatrix. `*` markiert die drei Mittel, deren letzte
    Stelle vom Summierer abhängt.
  - Darunter die zeilenweise Abflachung aus Zeile 75 und das Gegenbeispiel: Eine spaltenweise
    Abflachung gäbe in 8 von 12 Positionen das Gewicht eines fremden Taxons.
  - **Sonderwerte** mit drei Kunstfällen als Spalten: Matrixgröße, Maxima und Anzahl NA, NaN und
    `-Inf`. Darunter die Erklärung; eine Spalte nur aus NA wird abgelehnt.
  - Ein Satz zu `gibbs <- F`: `how_many_taxa = 5` statt `floor(5/2) = 2` im toten Gibbs-Zweig,
    Jeffreys-Prior 0,5 nur dort gelesen.
- Die Kunstkette heißt jetzt „ohne Taxon-Pruning, Gamma-Raster, TF-Entfernung und
  Sampler-Vorbereitung“.

Keine Erfolgsanzeige aus Testzahlen. Kunstbeispiele fließen in keinen Lauf, keinen Laufkatalog und
keine Rekonstruktion ein. Es wurde kein Sampler gestartet.

## Datenbindung

[nachbaustand.py](../../../src/kinneret_recon/nachbaustand.py) liest den neuesten Stand und prüft alle
vier früheren Stände gegen Pin, Freigabeflags und Modulliste. Das neue Beispiel stammt aus vier
gepinnten Dateien des Pakets. Für sechs Fälle müssen alle gespeicherten Werte **exakt** mit der
unabhängigen Referenz übereinstimmen, zusätzlich mit der Handerwartung:

- Anzahlen, Quellzeilen und Parameterstufe,
- Doubles und R-Sonderwerte,
- exakte Brüche der Werte und Mittel,
- Akkumulator-Kennzeichen.

Gibbs- und Gewichtsfall dürfen sich nur in den Zeilen 53–59 unterscheiden.

| Fall | Verhalten |
| --- | --- |
| Neuester Stand, ein früherer Stand oder ein gebundener Beleg fehlt | `available: false` mit Liste fehlender Pfade |
| Bytes einer gebundenen Datei (auch aller vier eingefrorenen Stände) geändert | Build bricht ab (`hash mismatch`) |
| Freigabeflag, gewählte Variante, R-Ausführung, abweichende Referenz-/Handwerte | Build bricht ab |

## Tatsächlich ausgeführte Prüfungen

[validation.json](validation.json) enthält Befehle, Zahlen und lokale Protokoll-Hashes.

- **App-/Daten-/Buildtests** (`tests/test_nachbaustand.py`, `test_forschungsstand.py`, `test_app.py`,
  `test_artenvergleich.py`): **61 bestanden, 2 übersprungen**.
  - Übersprungen sind die Python-Browserwrapper ohne Playwright im Standard-Node-Pfad; sie sind unten
    explizit ersetzt.
  - 7 neue Testfälle: Sampler-Beispiel, Bytewechsel an Stand 4 und an fünf Paketdateien.
- **Historische Teil-/Belegtests:** elf gezielte Dateien → **555 bestanden** (437 bisherige + 118 neue),
  nach den App-Änderungen erneut 555. Beide Phase-1-Prüfer geben weiter Exit 2.
- **Browser, offline, Edge** über das Playwright 1.62.1 der Codex-Laufzeit (`NODE_PATH`,
  `--kanal msedge`):
  - Ordner- und Ein-Datei-App × Handy (390 px)/PC × Hell/Dunkel, Gesten, DE/EN;
  - Aufklappzustand beim Sprachwechsel und Fehlstandskontrollen;
  - 21 Links (17 auf `00ab021`, je einer auf die vier eingefrorenen Stände) und neun Tabellen;
  - drei `*`-Mittel und drei markierte Sonderwertzellen;
  - Seiten- und Tabellenüberlauf.
- **Läufe:** Der erste Lauf nach den App-Änderungen war in allen 10 Fällen **ok**. Gemessene
  Tabellenbreiten bei 390 px: 206 und 184 px in 332 px Rahmen. Danach zeigte die Sichtprüfung,
  dass „Fall 1“ doppelt belegt war; der Begleittext sagt jetzt „Zuerst die Rohmatrix“.
  Der Endlauf nach neuem Build war wieder in allen 10 Fällen **ok**.
- **Zwei Standorte:** Kopie über `tests/test_app.py::_zweiter_standort`, dann
  `--zwei-standorte --kanal msedge` → **ok**.
- **Visuell geprüft:** [Mobil hell](mobile.png) und [Desktop dunkel](desktop.png). Für den Ausschnitt
  wurde die feste Kopfleiste statisch gesetzt.
- **Nicht ausgeführt:**
  - Gesamtsuite `pytest tests` (enthält Sampler-Unittests);
  - `literal_tf_preparation_2026_11_02/verify_packet.py` und die zeitpunktgebundenen `tools/verify_*`
    (keine Regressionsprüfer);
  - R, Trainings, Sampler, Altersmodelle, Fossilrekonstruktionen.

```bash
PYTHONPATH=src python -m pytest tests/test_nachbaustand.py tests/test_forschungsstand.py tests/test_app.py tests/test_artenvergleich.py -q
kinneret-recon app
kinneret-recon app --eine-datei
NODE_PATH=<Codex-Laufzeit>/dependencies/node/node_modules node tools/app_pruefen.js results/app/index.html results/kinneret-app.html --kanal msedge
node tools/app_pruefen.js --zwei-standorte <Kopie>/index.html --kanal msedge
```

## Grenzen

Fünfte App-Anbindung des geprüften Nachbaus, **keine historische Fallreproduktion**.

- Alle elf Nachweisbündel bleiben offen; keine Variante gewählt.
- Baseline, CLI, sieben experimentelle Artenmodelle, Paperreferenz, Forschungs-Snapshot und
  fallbezogene Monticchio-/Chronologie-Gates sind unverändert.
- Keine neuen Altersplots; jung links, alt rechts bleibt.

Nächste Übergabe: [Claude-Auftrag nach der Sampler-Vorbereitung](../CLAUDE_NEXT_STEP_AFTER_SAMPLER_PREP_2026_11_02.md).
