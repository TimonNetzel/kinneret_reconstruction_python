# Claude-Übergabe: nach `reconstruction_climate` und App-Stand 10

Stand 08.11.2026. Gilt ebenso für ChatGPT/Codex. Vorrangiges Nutzerziel bleibt:
**die lokalen taxonbasierten Rekonstruktionen der Dissertation historisch nachvollziehbar nachbauen und in der App
umsetzen; Verbesserungen danach.**

- Deutsch, kurze Fortschrittsmeldungen, keine Subagenten.
- Nur den vom Nutzer benannten vorhandenen Checkout verwenden; keine Klone oder Worktrees.
- Bestehender Branch `dissertation-reconstruction`,
  [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77), kein automatischer Merge.
- Zuerst Git-/PR-Stand prüfen. Ausgangspunkt dieses Pakets war `4373353`. Der Rechenschritt liegt in `77ccb85`;
  den App-Sicherungscommit liefert `git log -1 -- docs/thesis_research/app_reproduction_2026_11_08_reconstruction_climate`.
- Commits nur mit `git -c user.name=TimonNetzel -c user.email=134919762+TimonNetzel@users.noreply.github.com`.

## Geltende Nutzerentscheidungen

- **CRAN-Quellen lesen ist erlaubt.**
- **R-Läufe für Gegenproben** auf Kunstfixtures oder belegten Teilschritten sind erlaubt. R 3.5.0 unter `.local/R/R-3.5.0`,
  LaplacesDemon 16.1.6 unter `.local/R/library_ld`, matrixStats 0.56.0 unter `.local/R/library_ms` (unversioniert).
- Weiterhin **keine** Sampler, Trainings, Altersmodelle oder Fossilrekonstruktionen.
- Kinneret ist Slot 3 des Zusatzorte-Zweigs.

## Was jetzt fertig ist

1. **`reconstruction_climate`** `mcmc_functions_v1.6.R:1188–1199`
   ([Paket](literal_reconstruction_climate_2026_11_08/README.md), `tools/historical_reconstruction_climate.py`).
   - **Der gebundene Lauf ruft sie nicht auf.** Einziger Aufruf: `mcmc_sampling_v1.4.R:79`, die Datei wird nirgends geladen (540 R-Dateien durchsucht).
   - Aktiv ist stattdessen `mcmc_functions_v1.6.R:1850–1851` (`predictant_tann[1]`, aus Ziehungen).
   - Auch `mcmc_prepare.R:1408–1413` wird nicht erreicht: Zeile 1380 verlangt `!add.sites`, gebunden ist `add.sites <- T`.
   - R 3.5.0 + matrixStats 0.56.0: 5.510 von 5.510 Werten bitgleich; Lazy Evaluation mit `stop()`-Argumenten belegt.
   - 62 neue Tests, 1.037 gezielte Tests grün.
2. **App-Stand 10** ([README](app_reproduction_2026_11_08_reconstruction_climate/README.md), Pin `329e6a8b…`): 13 Teilmodule, das neue mit Etikett
   „Bedingter Schritt“, Tabelle K1–K8. Stände 1–9 eingefroren. Prüfstand: 36 Links, 16 Tabellen, 9 frühere Stände.

**Noch vom Nutzer zu entscheiden:** `tools/r_sort_gpl.py` bleibt GPL-2-oder-später im sonst MIT-lizenzierten Repository; eine R-nahe, bitgleiche
Rechenschicht für die sechs NumPy-Module wäre eine neue Modulvariante. Beides nicht still ändern.

## Nächster Schritt (Phase 1)

Speicherortfrage nicht wiederholen. **Vor jedem neuen Baustein zuerst die Aufrufer prüfen**, wie in `bind_source.py` des letzten Pakets.

**Empfohlen, weil tatsächlich aktiv:** das lokale `cor` in `mcmc_functions_v1.6.R:2355` (`function(x, y, na.rm=FALSE)`, es überdeckt `stats::cor`) mit expliziten
Vektoren nachbauen, einschließlich der Leser im aktiven `reconstruction`: 1964–1970 und 1999.

Danach, jeweils mit expliziten Eingaben und ohne Ziehung:

- `reconst`/`predictant` aus vorgegebenem `tf_sample` (`mcmc_functions_v1.6.R:1817–1820`);
- `wt_power` (Zeile 1003) und ihre Leser 2023/2031.

Danach ein neuer App-Stand am Ende von `nachbaustand.STATES`. Stand 10 bekommt dann seinen Linkcommit:
`git log -1 --format=%H -- docs/thesis_research/app_reproduction_2026_11_08_reconstruction_climate/reproduction_state.json`.
In `tools/app_pruefen.js` auch die Fehlstandskontrolle anpassen (jetzt `[0, 13]`).
Neue Fallbezeichnungen dürfen keine im Abschnitt verwendete wiederholen: 1–4, G1–G3, D1–D4, M1–M9 und K1–K8 sind belegt.

## Prüfen

```bash
python -m pytest tests/test_historical_reconstruction_climate.py tests/test_r_crosscheck_six_modules.py tests/test_historical_weighted_median.py tests/test_historical_dirichlet_ratio.py tests/test_r_runtime_crosscheck.py tests/test_historical_weights_stage.py tests/test_historical_sampler_preparation.py tests/test_historical_tf_missing_priors.py tests/test_historical_gamma_prior_grid.py tests/test_historical_taxon_pruning.py tests/test_historical_tf_preparation.py tests/test_historical_phase1_chain.py tests/test_historical_cdf_discrepancy.py tests/test_historical_age_selection.py tests/test_historical_age_depth.py tests/test_historical_final_pdf.py tests/test_kinneret_phase1_spec.py -q
PYTHONPATH=src python -m pytest tests/test_nachbaustand.py tests/test_forschungsstand.py tests/test_app.py tests/test_artenvergleich.py -q
python docs/thesis_research/literal_reconstruction_climate_2026_11_08/run_r_crosscheck.py .local/R/R-3.5.0/bin/x64/Rscript.exe .local/R/library_ms
NODE_PATH=C:/Users/tnetz/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules node tools/app_pruefen.js results/app/index.html results/kinneret-app.html --kanal msedge
```

- Builds und zwei Standorte wie in der [Übergabe nach der Gewichtsstufe](CLAUDE_NEXT_STEP_AFTER_WEIGHTS_STAGE_2026_11_03.md).
- Hinweise: Python-Heredocs mit Anführungszeichen scheitern; `< /dev/null` gegen hängende Eingaben; CRAN drosselt Download-Bündel;
  Rscript braucht Windows-Pfade (`cygpath -m`). Lange Browserläufe getrennt von anderen Schritten starten.
- Nicht ausführen: Gesamtsuite `pytest tests`, `verify_packet.py`, `tools/verify_*` als Regressionsprüfer.

## Arbeitsgrenzen

- Originale nur lesen; `diss` nie ändern; nur absolute Pfade.
- Keine stillen Ersatzdaten, Quellenkorrekturen oder Variantenentscheidungen. Paketbindungen nur mit Beleg und Zeitfenster.
- R nur für Gegenproben; Abweichungen messen und markieren, nicht verstecken.
- Baseline, Gates, Artenmodelle und Paperreferenz erhalten; Plots jung links, alt rechts.
- Testzahlen sind keine Fallreproduktion. Diese und frühere Übergaben eingefroren lassen.

Am Ende: Belegpaket, Status und Arbeitsplan, **neue konkrete Claude-Übergabe mit Link im Abschluss**, Commit und Push auf PR #77,
PR-Beschreibung fortschreiben, nicht mergen.
