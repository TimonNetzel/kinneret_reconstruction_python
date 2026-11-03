# Dissertationsnachbau in der App, Stand 6: Kern der Gewichtsstufe

Stand 03.11.2026. Bestehender Branch `dissertation-reconstruction`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).

- Ausgangscommit `235f797`: Checkout sauber, identisch mit dem PR-Kopf, PR offen, nicht gemergt,
  laut GitHub `MERGEABLE`.
- Ausschließlich der vom Nutzer benannte vorhandene Checkout, ohne Klon, Worktree oder Subagenten.
- Auftrag: [Claude-Übergabe nach der Sampler-Vorbereitung](../CLAUDE_NEXT_STEP_AFTER_SAMPLER_PREP_2026_11_02.md).

## Ergebnis

Neuer datierter Nachbaustand [reproduction_state.json](reproduction_state.json)
(SHA-256 `7eb9ac5e4eb616e17db8e0ddb5e9926c9abb600094a8e88219d151d92b2dc95d`), als **neuer letzter
Eintrag** in `nachbaustand.STATES`. Quellenlinks zeigen auf Commit `4266c28` mit dem neuen
[Paket Kern der Gewichtsstufe](../literal_weights_stage_2026_11_03/README.md).

Die früheren Stände bleiben **bytegleich eingefroren** und werden weiter per Hash geprüft:
[Stand 1](../app_reproduction_2026_11_02/README.md) (`c358b2c3…`),
[Stand 2](../app_reproduction_2026_11_02_pruning/README.md) (`3656316d…`),
[Stand 3](../app_reproduction_2026_11_02_gamma_grid/README.md) (`43edfcb6…`),
[Stand 4](../app_reproduction_2026_11_02_tf_missing/README.md) (`7b7bf1f4…`) und
[Stand 5](../app_reproduction_2026_11_02_sampler_prep/README.md) (`61c6f231…`).
Der `STATES`-Eintrag von Stand 5 erhielt nur seinen Linkcommit `235f797`. Er ist mit
`git log -1 --format=%H -- …/app_reproduction_2026_11_02_sampler_prep/reproduction_state.json` bestätigt,
gepusht und enthält die unveränderte Zustandsdatei. Pfad, Pin und Modulliste sind unverändert.

Die Ansicht **Forschungsstand → Dissertationsnachbau** zeigt jetzt:

- unveränderten Status „Teilmodule geprüft; vollständiger Originalfall offen“;
- **fünf** Zeilen „Früherer Stand eingefroren“ (5/251, 6/297, 7/352, 8/437 und 9 Teilmodule/555 Tests);
- **zehn** Teilmodule und **689** Teil-/Belegtests. Neu ist „Kern der Gewichtsstufe mit vorgegebenen
  Gewichten“ (134 Tests, Quellen `mcmc_functions_v1.6.R`, `mcmc_conduction.R`).
  - Das Modul trägt **kein** Etikett „Bedingter Schritt“: Die Zeilen 1259 und 1276–1280 laufen in jeder
    Sampler-Iteration unbedingt. Die Gewichte sind ausdrücklich vorgegeben, weil `rdirichlet` zufällig ist
    und sein Paket ungebunden bleibt.
  - Das Taxon-Pruning bleibt der einzige bedingte Schritt.
- unverändert die elf offenen Originaleingaben aus dem Belegvertrag;
- im weiterhin eingeklappten **synthetischen** Beispiel zwei neue Tabellen, beide passend auf 390 px:
  - **Gewichtsbeispiel** mit Kunsttaxa als Spalten. Es setzt die normierte Matrix der
    Sampler-Vorbereitung darüber fort, mit den Gewichten 1/8, 3/8, 1/4, 1/4. Zeilen 1–2 stehen als Dezimalzahl,
    ihre Summen tragen `*` (akkumulatorabhängig). Zeile 3 ist exakt: 1/29, 12/29, 8/29, 8/29, Summe 29/32.
  - Darunter: Recycling und Gegenbeispiel (8 von 12 Positionen bekämen bei spaltenweisen Werten ein fremdes
    Gewicht), `z = 1:4`, Leser `reconstruction` (Zeile 1817).
  - **Zeilensummen mit Sonderwerten** mit den Kunstfällen **G1–G3** als Spalten: Matrixgröße, Zeilen mit
    Summe 0 und `-Inf`, Ergebniszellen NA, NaN, `±Inf` und `-0`. Die Fallbezeichnungen G1–G3 wiederholen keine
    frühere Bezeichnung im Abschnitt; die älteren Tabellen behalten ihre eingefrorenen Nummern.
  - Ein Satz zu den Rücksetzgewichten `rep(1/3, 3)`: Summe `1 − 2⁻⁵⁴` rundet zur geraden 1, gewichtet
    0,5714285714285715 eine Einheit der letzten Stelle über dem ungewichteten Ergebnis. Dazu: Gewichte mit
    Summe 2 ändern nichts.
  - Ein Satz zu `rdirichlet`/`ddirichlet`: nicht in den Quellen definiert, Anbieter nicht belegt, nur
    dokumentiert.
- Die Kunstkette heißt jetzt „ohne Taxon-Pruning, Gamma-Raster, TF-Entfernung, Sampler-Vorbereitung und
  Gewichtsstufe“.

Keine Erfolgsanzeige aus Testzahlen. Kunstbeispiele fließen in keinen Lauf, keinen Laufkatalog und
keine Rekonstruktion ein. Es wurde kein Sampler gestartet und keine Dirichlet-Ziehung ausgeführt.

## Datenbindung

[nachbaustand.py](../../../src/kinneret_recon/nachbaustand.py) liest den neuesten Stand und prüft alle
fünf früheren Stände gegen Pin, Freigabeflags und Modulliste. Das neue Beispiel stammt aus vier
gepinnten Dateien des Pakets. Für sieben Fälle müssen alle gespeicherten Werte **exakt** mit der
unabhängigen Referenz übereinstimmen, zusätzlich mit der Handerwartung:

- Produkte, Zeilensummen (exakt und gerundet), Zählungen und Akkumulator-Kennzeichen,
- Ergebniszellen einschließlich R-Sonderwerten und Vorzeichen der Null,
- `z` und die Zeilen mit Summe 0.

Außerdem müssen der Zweierpotenz-Fall dem Grundfall gleichen und die Rücksetzsumme `1 − 2⁻⁵⁴` sein.
Eine Datei, die ein Dirichlet-Paket nennt, wird abgelehnt.

| Fall | Verhalten |
| --- | --- |
| Neuester Stand, ein früherer Stand oder ein gebundener Beleg fehlt | `available: false` mit Liste fehlender Pfade |
| Bytes einer gebundenen Datei (auch aller fünf eingefrorenen Stände) geändert | Build bricht ab (`hash mismatch`) |
| Freigabeflag, gewählte Variante, Dirichlet-Paket, R-Ausführung, abweichende Referenz-/Handwerte | Build bricht ab |

## Tatsächlich ausgeführte Prüfungen

[validation.json](validation.json) enthält Befehle, Zahlen und lokale Protokoll-Hashes.

- **App-/Daten-/Buildtests** (`tests/test_nachbaustand.py`, `test_forschungsstand.py`, `test_app.py`,
  `test_artenvergleich.py`): **69 bestanden, 2 übersprungen**.
  - Übersprungen sind die Python-Browserwrapper ohne Playwright im Standard-Node-Pfad; sie sind unten
    explizit ersetzt.
  - 8 neue Testfälle: Gewichtsbeispiel, Ablehnung einer Paketangabe, Bytewechsel an Stand 5 und an fünf
    Paketdateien.
- **Historische Teil-/Belegtests:** zwölf gezielte Dateien → **689 bestanden** (555 bisherige + 134 neue),
  nach den App-Änderungen erneut 689. Beide Phase-1-Prüfer geben weiter Exit 2.
- **Browser, offline, Edge** über das Playwright der Codex-Laufzeit (`NODE_PATH`, `--kanal msedge`):
  - Ordner- und Ein-Datei-App × Handy (390 px)/PC × Hell/Dunkel, Gesten, DE/EN;
  - Aufklappzustand beim Sprachwechsel und Fehlstandskontrollen (jetzt zehn Module);
  - 24 Links (19 auf `4266c28`, je einer auf die fünf eingefrorenen Stände) und elf Tabellen;
  - zwei `*`-Zeilensummen, eine exakte Zeilensumme und zwei markierte Sonderwertzellen;
  - Seiten- und Tabellenüberlauf.
- **Läufe:** Der erste Lauf nach den App-Änderungen war in allen 10 Fällen **ok**. Gemessene
  Tabellenbreiten bei 390 px: 300 und 164 px in 332 px Rahmen; Zeilenköpfe `nowrap`. Nach der
  Sichtprüfung wurde nur ein Satz präzisiert (ungewichtetes Ergebnis als Double zu 4/7). Der Endlauf nach
  neuem Build war wieder in allen 10 Fällen **ok**.
- **Zwei Standorte:** Kopie über `tests/test_app.py::_zweiter_standort`, dann
  `--zwei-standorte --kanal msedge` → **ok**.
- **Visuell geprüft:** [Mobil hell](mobile.png) und [Desktop dunkel](desktop.png). Für den Ausschnitt
  wurde die feste Kopfleiste statisch gesetzt.
- **Nicht ausgeführt:**
  - Gesamtsuite `pytest tests` (enthält Sampler-Unittests);
  - `literal_tf_preparation_2026_11_02/verify_packet.py` und die zeitpunktgebundenen `tools/verify_*`
    (keine Regressionsprüfer);
  - R, `rdirichlet`/`ddirichlet`, Trainings, Sampler, Altersmodelle, Fossilrekonstruktionen.

```bash
PYTHONPATH=src python -m pytest tests/test_nachbaustand.py tests/test_forschungsstand.py tests/test_app.py tests/test_artenvergleich.py -q
kinneret-recon app
kinneret-recon app --eine-datei
NODE_PATH=<Codex-Laufzeit>/dependencies/node/node_modules node tools/app_pruefen.js results/app/index.html results/kinneret-app.html --kanal msedge
node tools/app_pruefen.js --zwei-standorte <Kopie>/index.html --kanal msedge
```

## Grenzen

Sechste App-Anbindung des geprüften Nachbaus, **keine historische Fallreproduktion**.

- Alle elf Nachweisbündel bleiben offen; keine Variante und kein Dirichlet-Paket gewählt.
- Baseline, CLI, sieben experimentelle Artenmodelle, Paperreferenz, Forschungs-Snapshot und
  fallbezogene Monticchio-/Chronologie-Gates sind unverändert.
- Keine neuen Altersplots; jung links, alt rechts bleibt.

Nächste Übergabe: [Claude-Auftrag nach der Gewichtsstufe](../CLAUDE_NEXT_STEP_AFTER_WEIGHTS_STAGE_2026_11_03.md).
