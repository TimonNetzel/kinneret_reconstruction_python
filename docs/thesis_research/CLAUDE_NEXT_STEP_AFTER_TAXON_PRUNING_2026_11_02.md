# Claude-Übergabe: nach Taxon-Pruning und App-Stand 2

Stand 02.11.2026. Gilt ebenso für ChatGPT/Codex. Vorrangiges Nutzerziel bleibt:
**die lokalen taxonbasierten Rekonstruktionen der Dissertation historisch
nachvollziehbar nachbauen und in der App umsetzen; Verbesserungen danach.**
Deutsch, kurze Fortschrittsmeldungen, keine Subagenten. Nur den vom Nutzer
benannten vorhandenen Checkout verwenden; keine Klone/Worktrees. Bestehender
Branch `dissertation-reconstruction`, [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
kein automatischer Merge. Zuerst Git-/PR-Stand und lokale Änderungen prüfen.
Ausgangspunkt dieses Pakets `47a926c`; Rechenschritt in `f3b71bb`, den
App-Sicherungscommit liefert `git log -1 -- docs/thesis_research/app_reproduction_2026_11_02_pruning`.
GitHub meldete PR #77 vor diesem Paket als `CONFLICTING` gegenüber `main`
(dort kam #78 mit `CLAUDE.md` hinzu); Konfliktlösung oder Merge nur auf
ausdrücklichen Nutzerauftrag.

## Was jetzt fertig ist

1. Lies `CLAUDE.md`, `AGENTS.md`, den neuesten [Projektstatus](PROJECT_STATUS.md),
   den [Arbeitsplan](LOCAL_DISS_RECONSTRUCTION_PLAN.md), das
   [Taxon-Pruning-Paket](literal_taxon_pruning_2026_11_02/README.md) und den
   [App-Stand 2](app_reproduction_2026_11_02_pruning/README.md).
2. **Determinismusprüfung** ([determinism_audit.json](literal_taxon_pruning_2026_11_02/determinism_audit.json)):
   Von den Kandidaten um die TF-Vorbereitung ist nur das Taxon-Pruning
   (`mcmc_prepare.R:51–80`) ohne Zufall **und** ohne ungebundene Paketsemantik.
   `weightedMedian` (1631/1651) ist aktiv, aber `matrixStats` ist ohne Fassung
   geladen (`main_v1.2.R:27`) → offen im Bündel Laufzeit/RNG. Splines 1627/1647
   wirken nur in auskommentierten Zeilen; Erwartungswert 1623–1624 wird
   überschrieben; `sample`/`quantile` 1591–1603, 1399 und 1896–1914 sind
   zufallsabhängig (nur 1896 mit Seed, RNG-Art/R-Fassung ungebunden).
3. **Modul** `tools/historical_taxon_pruning.py`: R wörtlich (abwärts zählendes
   `16:ncol` mit NA-Positionen, Gleichstandserweiterung über `%in%`,
   `metro_hastings` nur auf TRUE, `next()` als `skip_site`); Reliktzweig als
   expliziter Boolwert; Vektorverlängerung und Dimensionsverlust abgelehnt.
   Im gebundenen Hauptlauf `main_v1.2.R:659` ist der Schritt abgeschaltet
   (`null.ratio.check <- FALSE`) → als **bedingter Schritt** geführt.
   46 neue Tests gegen unabhängige Bruchreferenz, mit den bisherigen **297 grün**.
4. **App:** neuer datierter Nachbaustand
   `app_reproduction_2026_11_02_pruning/reproduction_state.json` (Pin
   `3656316d…`); Leser `nachbaustand.py` hat jetzt eine Standliste `STATES`,
   zeigt den neuesten und prüft/listet den eingefrorenen Stand 1 (`c358b2c3…`).
   Sechs Module, Etikett „Bedingter Schritt“, Pruning-Kunsttabelle im
   eingeklappten synthetischen Beispiel. Beide Offlineformate in Edge geprüft
   (Handy/PC, Hell/Dunkel, DE/EN, Gesten, zwei Standorte):
   [validation.json](app_reproduction_2026_11_02_pruning/validation.json).

**Weiter offen:** kein vollständiger Originalfall. Alle elf Nachweisbündel offen
(finale Basis/Listenchronologie, numerische historische TFs, Isotopen-Workspaces,
CRU/kgc, Referenz-/Ankervariante, Originalposterioren, finale Nachbearbeitung,
Laufzeit/RNG einschließlich `matrixStats`-Fassung, numerische Originalausgaben).
Keine Variante gewählt.

## Nächster fachlicher Schritt (Phase 1)

Speicherortfrage nicht wiederholen, keine pauschale Inventur. Nächsten
begrenzten deterministischen Schritt aus bereits gebundenen Quellen isolieren.
Empfohlener Kandidat nach Quelltextlage (vorher selbst belegen):

- **Gamma-Prior-Parameter über ein Raster** in `mcmc_prepare.R:114–121`
  (Standardwerte, unbedingt ausgeführt) und das Gegenstück `1270–1281`
  (nach der bedingten Varianzverdopplung 1264–1267): `seq(0.1, 50000, by = 0.1)`,
  `rate1 = shape/mean`, `rate2 = sqrt(shape)/sd`, `which.min(|rate2 − rate1|)`.
  Nur base R. Zu klären und zu testen: R-`seq`-Erzeugung (`from + (0:n)*by`),
  erstes Minimum bei Gleichstand, Abstand zur geschlossenen Lösung
  `shape = (mean/sd)²` nur als Kontrolle, nicht als Ersatz. Eingaben
  `prior.recent` stammen aus `prior_climate.R:128–130` (Fits) → nur Kunstwerte,
  keine historische Belegung.
- Alternativ die reine Entfernungsmechanik `which.tf.missing` (1664–1679) mit
  **explizit übergebener** NA-Maske; ihre Ursache (`weightedMedian`) bleibt offen.
- `weightedMedian` erst nach belegter `matrixStats`-Fassung; keine
  Interpolationswahl. Zufallsschritte weiter nur dokumentieren.

Bisherige Fixtures, Module, Tests und Pakete bytegleich lassen.

## Nächster App-Schritt

Nach einem neuen geprüften Teilmodul: neuen Ordner
`app_reproduction_<Datum>_<Kurzname>/` mit eigener `reproduction_state.json`,
Eintrag **ans Ende** von `nachbaustand.STATES` (ältere Einträge nie ändern,
dem bisher neuesten ein `commit` mit gepushtem Commit geben, in dem seine Datei
liegt). Modul-/Testzahlen aus dem neuen Paket; `tools/app_pruefen.js`
(Modulanzahl, Testzahl, Links/Commit, Tabellenzahl, Fehlstandskontrolle)
anpassen. Prüfen:

```bash
PYTHONPATH=src python -m pytest tests/test_nachbaustand.py tests/test_forschungsstand.py tests/test_app.py tests/test_artenvergleich.py -q
python -m pytest tests/test_historical_taxon_pruning.py tests/test_historical_tf_preparation.py tests/test_historical_phase1_chain.py tests/test_historical_cdf_discrepancy.py tests/test_historical_age_selection.py tests/test_historical_age_depth.py tests/test_historical_final_pdf.py tests/test_kinneret_phase1_spec.py -q
NODE_PATH=<Codex-Laufzeit>/dependencies/node/node_modules node tools/app_pruefen.js results/app/index.html results/kinneret-app.html --kanal msedge
```

Zwei Standorte: Kopie über `tests/test_app.py::_zweiter_standort` anlegen, dann
`node tools/app_pruefen.js --zwei-standorte <Kopie>/index.html --kanal msedge`.
Die Gesamtsuite `pytest tests` nicht ausführen (Sampler-Unittests);
`literal_tf_preparation_2026_11_02/verify_packet.py` nicht als Regressionsprüfer.

## Arbeitsgrenzen

Originale nur lesen; `diss` und private Quellpfade nie ändern. Keine stillen
Ersatzdaten, Quellenkorrekturen oder Variantenentscheidungen (kein
`inside.range`, Median oder Seed ohne Beleg). Keine Trainings, Sampler,
Altersmodelle, R-Läufe oder Fossilrekonstruktionen. Baseline, CLI, sieben
experimentelle Artenmodelle, Paperreferenz und fallbezogene Monticchio-/
Chronologie-Gates erhalten. Plots jung links, alt rechts. Diese und frühere
Übergaben eingefroren lassen. Am Ende neues Belegpaket, aktualisierter
Projektstatus/Arbeitsplan und **eine neue konkrete Claude-Übergabe mit Link im
Abschluss**; geprüftes Paket auf dem bestehenden Branch committen/pushen,
PR #77 fortschreiben, nicht mergen.
