# Claude-Übergabe: nach der R-Gegenprobe der sechs Module und App-Stand 9

Stand 08.11.2026. Gilt ebenso für ChatGPT/Codex. Vorrangiges Nutzerziel bleibt:
**die lokalen taxonbasierten Rekonstruktionen der Dissertation historisch nachvollziehbar nachbauen und in der App
umsetzen; Verbesserungen danach.**

- Deutsch, kurze Fortschrittsmeldungen, keine Subagenten.
- Nur den vom Nutzer benannten vorhandenen Checkout verwenden; keine Klone oder Worktrees.
- Bestehender Branch `dissertation-reconstruction`,
  [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77), kein automatischer Merge.
- Zuerst Git-/PR-Stand prüfen. Ausgangspunkt dieses Pakets war `416d547`. Belegpaket in `8faa035`, Quellenlinks auf `0f044c0`;
  den App-Sicherungscommit liefert `git log -1 -- docs/thesis_research/app_reproduction_2026_11_08_r_crosscheck_six`.
- Commits nur mit `git -c user.name=TimonNetzel -c user.email=134919762+TimonNetzel@users.noreply.github.com`.

## Geltende Nutzerentscheidungen

- **CRAN-Quellen lesen ist erlaubt.**
- **R-Läufe für Gegenproben** auf Kunstfixtures oder belegten Teilschritten sind erlaubt. R 3.5.0 unter `.local/R/R-3.5.0`,
  LaplacesDemon 16.1.6 unter `.local/R/library_ld`, matrixStats 0.56.0 unter `.local/R/library_ms` (unversioniert).
- Weiterhin **keine** Sampler, Trainings, Altersmodelle oder Fossilrekonstruktionen.
- Kinneret ist Slot 3 des Zusatzorte-Zweigs.

## Was jetzt fertig ist

1. **Sechs ältere Module gegen R 3.5.0** ([Paket](r_crosscheck_six_modules_2026_11_08/README.md)): Originalzeilen wörtlich.
   - Alle 7.064 Fixture-Werte bitgleich; keine Abweichung bei Sonderwerten.
   - Mit 150 Stressfällen: 34.918 von 41.966 Werten bitgleich (ohne Altersraster), höchstens 4 ulp, nur Summen und
     Matrixprodukte.
   - Ursache belegt: R-Nachrechnung mit `long double`-Summen und Referenz-BLAS trifft alle 24.822 betroffenen Werte bitgenau.
   - Module unverändert. Sie sind nicht bitgleich zu R; das bleibt markiert.
   - `rowCumsums`: im Fenster nur matrixStats; `na.rm=T` wird von `...` verschluckt.
   - 41 neue Tests, 975 gezielte Tests grün.
2. **App-Stand 9** ([README](app_reproduction_2026_11_08_r_crosscheck_six/README.md), Pin `50fe8b0a…`): Tabelle je Modul; Stände 1–8
   eingefroren, Stand 8 erhielt nur Linkcommit `416d547`. Prüfstand: 33 Links, 15 Tabellen, 12 Module, 8 frühere Stände.

**Noch vom Nutzer zu entscheiden:** `tools/r_sort_gpl.py` ist eine Übersetzung von R-Quellen und bleibt GPL-2-oder-später; das
Repository ist sonst MIT. Nicht still ändern.

## Nächster Schritt (Phase 1)

Speicherortfrage nicht wiederholen.

**Empfohlen:** weitere deterministische Ränder mit expliziten Eingaben. Kandidaten:

- `reconstruction_climate` (`mcmc_functions_v1.6.R:1188`) und `mcmc_prepare.R:1412–1413`; beide nutzen `weightedMedian`.
- Korrelationsteil ab `mcmc_functions_v1.6.R:1939` (`predictant_current`, `wt_power`).

Alternativen, jeweils nur auf Nutzerwunsch:

- Eine R-nahe Rechenschicht (`long double`-Summen, Referenz-BLAS-Schleife) für bitgleiche Module. Das ändert die eingefrorenen Pakete
  nicht, wäre aber eine neue Variante der Module und braucht eine ausdrückliche Entscheidung.
- Die fünf `exportPattern`-Pakete hinter matrixStats (`verification`, `CircStats`, `smotefamily`, `fields`, `snow`) an ihren Tarballs auf
  Base-Masken prüfen.

Danach ein neuer App-Stand am Ende von `nachbaustand.STATES`. Stand 9 bekommt dann seinen Linkcommit:
`git log -1 --format=%H -- docs/thesis_research/app_reproduction_2026_11_08_r_crosscheck_six/reproduction_state.json`.
Neue Fallbezeichnungen dürfen keine im Abschnitt verwendete wiederholen: 1–4, G1–G3, D1–D4 und M1–M9 sind belegt.

## Prüfen

```bash
python -m pytest tests/test_r_crosscheck_six_modules.py tests/test_historical_weighted_median.py tests/test_historical_dirichlet_ratio.py tests/test_r_runtime_crosscheck.py tests/test_historical_weights_stage.py tests/test_historical_sampler_preparation.py tests/test_historical_tf_missing_priors.py tests/test_historical_gamma_prior_grid.py tests/test_historical_taxon_pruning.py tests/test_historical_tf_preparation.py tests/test_historical_phase1_chain.py tests/test_historical_cdf_discrepancy.py tests/test_historical_age_selection.py tests/test_historical_age_depth.py tests/test_historical_final_pdf.py tests/test_kinneret_phase1_spec.py -q
PYTHONPATH=src python -m pytest tests/test_nachbaustand.py tests/test_forschungsstand.py tests/test_app.py tests/test_artenvergleich.py -q
python docs/thesis_research/r_crosscheck_six_modules_2026_11_08/run_six_crosscheck.py .local/R/R-3.5.0/bin/x64/Rscript.exe .local/R/library_ms
NODE_PATH=C:/Users/tnetz/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules node tools/app_pruefen.js results/app/index.html results/kinneret-app.html --kanal msedge
```

- Builds und zwei Standorte wie in der [Übergabe nach der Gewichtsstufe](CLAUDE_NEXT_STEP_AFTER_WEIGHTS_STAGE_2026_11_03.md).
- Hinweise: Python-Heredocs mit Anführungszeichen scheitern; `< /dev/null` gegen hängende Eingaben; CRAN drosselt Download-Bündel;
  Rscript braucht Windows-Pfade (`cygpath -m`).
- Nicht ausführen: Gesamtsuite `pytest tests`, `verify_packet.py`, `tools/verify_*` als Regressionsprüfer.

## Arbeitsgrenzen

- Originale nur lesen; `diss` nie ändern; nur absolute Pfade.
- Keine stillen Ersatzdaten, Quellenkorrekturen oder Variantenentscheidungen. Paketbindungen nur mit Beleg und Zeitfenster.
- R nur für Gegenproben; Abweichungen messen und markieren, nicht verstecken.
- Baseline, Gates, Artenmodelle und Paperreferenz erhalten; Plots jung links, alt rechts.
- Testzahlen sind keine Fallreproduktion. Diese und frühere Übergaben eingefroren lassen.

Am Ende: Belegpaket, Status und Arbeitsplan, **neue konkrete Claude-Übergabe mit Link im Abschluss**, Commit und Push auf PR #77,
PR-Beschreibung fortschreiben, nicht mergen.
