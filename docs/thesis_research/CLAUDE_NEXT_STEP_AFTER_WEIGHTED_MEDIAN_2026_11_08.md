# Claude-Übergabe: nach dem gewichteten Median und App-Stand 8

Stand 08.11.2026. Gilt ebenso für ChatGPT/Codex. Vorrangiges Nutzerziel bleibt:
**die lokalen taxonbasierten Rekonstruktionen der Dissertation historisch nachvollziehbar nachbauen und in der App
umsetzen; Verbesserungen danach.**

- Deutsch, kurze Fortschrittsmeldungen, keine Subagenten.
- Nur den vom Nutzer benannten vorhandenen Checkout verwenden; keine Klone oder Worktrees.
- Bestehender Branch `dissertation-reconstruction`,
  [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77), kein automatischer Merge.
- Zuerst Git-/PR-Stand prüfen. Ausgangspunkt dieses Pakets war `cfd0ad8`. Der Rechenschritt liegt in `a21c8f1`;
  den App-Sicherungscommit liefert `git log -1 -- docs/thesis_research/app_reproduction_2026_11_08_weighted_median`.
- Commits nur mit `git -c user.name=TimonNetzel -c user.email=134919762+TimonNetzel@users.noreply.github.com`.

## Geltende Nutzerentscheidungen

- **CRAN-Quellen lesen ist erlaubt.**
- **R-Läufe für Gegenproben** auf Kunstfixtures oder belegten Teilschritten sind erlaubt. R 3.5.0 liegt unter
  `.local/R/R-3.5.0`, LaplacesDemon 16.1.6 unter `.local/R/library_ld`, matrixStats 0.56.0 unter `.local/R/library_ms`
  (alles unversioniert).
- Weiterhin **keine** Sampler, Trainings, Altersmodelle oder Fossilrekonstruktionen.
- Kinneret ist Slot 3 des Zusatzorte-Zweigs ([Laufzeitbelege](r_runtime_crosscheck_2026_11_03/README.md)).

## Was jetzt fertig ist

1. **`weightedMedian` und NA-Maske** `mcmc_prepare.R:1628–1664`
   ([Paket](literal_weighted_median_2026_11_08/README.md), `tools/historical_weighted_median.py`, `tools/r_sort_gpl.py`).
   - Der normalisierte C-Kern ist in allen 13 Fassungen des Fensters ein einziger Text.
   - **NA genau dann**, wenn irgendein Gewicht NA/NaN ist oder keines positiv ist (bei endlichem `x`). Mit NA in `x`:
     nur bei Gewicht `> 0`, und bei `+Inf` zählt die Reihenfolge. Volle Regel im Paket.
   - Quellnahe Eigenheiten: `-Inf` zählt im `+Inf`-Zweig, ungerades n im Gleichgewichtszweig ist nicht der Median,
     Ties über das instabile `R_qsort_I`. Überlauf der Gesamtmasse wird abgelehnt (C liest außerhalb der Felder).
   - R 3.5.0 + matrixStats 0.56.0 (Binärpaket vorher per Hash festgehalten): **439 von 439 Werten bitgleich**, NA und
     NaN unterschieden, wörtliche Zeilen 1628–1640, 1648–1660, 1664 auf Kunstmatrix ergeben `which.tf.missing = 2 3 5`.
   - 182 neue Tests, 934 gezielte Tests grün.
2. **`tann <- FALSE` belegt** ([tann_binding.json](literal_weighted_median_2026_11_08/tann_binding.json)): Bedingung
   1248, Zuweisung 1249; einziger Leser im aktiven `reconstruction` ist `mcmc_functions_v1.6.R:1939`
   (`predictant_pann` wird `predictant_current`). Leser nur gesammelt, nicht gedeutet.
3. **App-Stand 8** ([README](app_reproduction_2026_11_08_weighted_median/README.md), Pin `4d25a3f6…`).
   - Zwölf Teilmodule, R-Zeile um matrixStats ergänzt, Tabellen M1–M9 und NA-Maske.
   - Stände 1–7 eingefroren; Stand 7 erhielt nur Linkcommit `cfd0ad8`.
   - Prüfstand: 31 Links, 14 Tabellen.

**Lizenzhinweis, noch vom Nutzer zu entscheiden:** `tools/r_sort_gpl.py` ist eine Übersetzung von R-Quellen und bleibt
GPL-2-oder-später; das Repository ist sonst MIT. Nicht still ändern.

## Nächster Schritt (Phase 1)

Speicherortfrage nicht wiederholen.

**Empfohlen:** Die sechs Module ohne R-Gegenprobe gegen ihre Originalzeilen prüfen (Alters-/Tiefenübertragung,
Altersauswahl, End-PDF, CDF-Abweichung, TF-Vorbereitung, Taxon-Pruning). `r_runtime_crosscheck_2026_11_03/crosscheck.R`
und [run_r_crosscheck.py](literal_weighted_median_2026_11_08/run_r_crosscheck.py) lassen sich als Muster nehmen.
Eingaben als C99-Hex; Abweichungen in ulp messen und markieren.

Alternativen:

- `weightedMedian` steht auch in `mcmc_prepare.R:1412–1413` und `mcmc_functions_v1.6.R:1191` (`tann.recent`); mit dem
  neuen Modul lässt sich `reconstruction_climate` als nächster deterministischer Rand prüfen, sofern die Eingaben
  explizit bleiben.
- Korrelationsteil `mcmc_functions_v1.6.R:1939ff.` weiter belegen (`predictant_current`, `wt_power`), ohne Sampler.

Danach ein neuer App-Stand am Ende von `nachbaustand.STATES`. Stand 8 bekommt dann seinen Linkcommit:
`git log -1 --format=%H -- docs/thesis_research/app_reproduction_2026_11_08_weighted_median/reproduction_state.json`.

`tools/app_pruefen.js` (jetzt 31 Links, 14 Tabellen, 12 Module, 7 frühere Stände) und `tests/test_nachbaustand.py`
anpassen. Neue Fallbezeichnungen dürfen keine im Abschnitt verwendete wiederholen: 1–4, G1–G3, D1–D4 und M1–M9 sind
belegt.

## Prüfen

```bash
python -m pytest tests/test_historical_weighted_median.py tests/test_historical_dirichlet_ratio.py tests/test_r_runtime_crosscheck.py tests/test_historical_weights_stage.py tests/test_historical_sampler_preparation.py tests/test_historical_tf_missing_priors.py tests/test_historical_gamma_prior_grid.py tests/test_historical_taxon_pruning.py tests/test_historical_tf_preparation.py tests/test_historical_phase1_chain.py tests/test_historical_cdf_discrepancy.py tests/test_historical_age_selection.py tests/test_historical_age_depth.py tests/test_historical_final_pdf.py tests/test_kinneret_phase1_spec.py -q
PYTHONPATH=src python -m pytest tests/test_nachbaustand.py tests/test_forschungsstand.py tests/test_app.py tests/test_artenvergleich.py -q
python docs/thesis_research/literal_weighted_median_2026_11_08/run_r_crosscheck.py .local/R/R-3.5.0/bin/x64/Rscript.exe .local/R/library_ms
NODE_PATH=C:/Users/tnetz/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules node tools/app_pruefen.js results/app/index.html results/kinneret-app.html --kanal msedge
```

- Builds und zwei Standorte wie in der [Übergabe nach der Gewichtsstufe](CLAUDE_NEXT_STEP_AFTER_WEIGHTS_STAGE_2026_11_03.md).
- Shell-Hinweise:
  - Python-Heredocs mit Anführungszeichen scheitern; Hilfsskripte als Datei anlegen.
  - `< /dev/null` gegen hängende Standardeingaben.
  - CRAN drosselt Bündel von Downloads.
  - Rscript braucht Windows-Pfade (`cygpath -m`), keine `/c/…`-Pfade.
- Nicht ausführen: Gesamtsuite `pytest tests`, `verify_packet.py`, `tools/verify_*` als Regressionsprüfer.

## Arbeitsgrenzen

- Originale nur lesen; `diss` nie ändern; nur absolute Pfade.
- Keine stillen Ersatzdaten, Quellenkorrekturen oder Variantenentscheidungen. Paketbindungen nur mit Beleg und
  Zeitfenster.
- R nur für Gegenproben; Abweichungen messen und markieren, nicht verstecken.
- Baseline, Gates, Artenmodelle und Paperreferenz erhalten; Plots jung links, alt rechts.
- Testzahlen sind keine Fallreproduktion. Diese und frühere Übergaben eingefroren lassen.

Am Ende: Belegpaket, Status und Arbeitsplan, **neue konkrete Claude-Übergabe mit Link im Abschluss**, Commit und Push
auf PR #77, PR-Beschreibung fortschreiben, nicht mergen.
