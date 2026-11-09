# Claude-Übergabe: nach dem lokalen `cor` und App-Stand 11

Stand 09.11.2026. Gilt ebenso für ChatGPT/Codex. Vorrangiges Nutzerziel bleibt:
**die lokalen taxonbasierten Rekonstruktionen der Dissertation historisch nachvollziehbar nachbauen und in der App
umsetzen; Verbesserungen danach.**

- Deutsch, kurze Fortschrittsmeldungen, keine Subagenten.
- Nur den vom Nutzer benannten vorhandenen Checkout verwenden; keine Klone oder Worktrees.
- Bestehender Branch `dissertation-reconstruction`,
  [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77), kein automatischer Merge.
- Zuerst Git-/PR-Stand prüfen. Ausgangspunkt dieses Pakets war `437f6f4`. Der Rechenschritt liegt in `6caf224` und `5af02eb`; den
  App-Sicherungscommit liefert `git log -1 --format=%H -- docs/thesis_research/app_reproduction_2026_11_09_local_cor/reproduction_state.json`.
- Commits nur mit `git -c user.name=TimonNetzel -c user.email=134919762+TimonNetzel@users.noreply.github.com`.

## Geltende Nutzerentscheidungen

- **CRAN-Quellen lesen ist erlaubt.**
- **R-Läufe für Gegenproben** auf Kunstfixtures oder belegten Teilschritten sind erlaubt. R 3.5.0 unter `.local/R/R-3.5.0`,
  LaplacesDemon 16.1.6 unter `.local/R/library_ld`, matrixStats 0.56.0 unter `.local/R/library_ms` (unversioniert).
- Weiterhin **keine** Sampler, Trainings, Altersmodelle oder Fossilrekonstruktionen.
- Kinneret ist Slot 3 des Zusatzorte-Zweigs (`add.sites <- T`).
- **`diss` nur lesen, nur absolute Pfade, dorthin kein `cd`.**
- **Noch vom Nutzer zu entscheiden:** `tools/r_sort_gpl.py` bleibt GPL-2-oder-später im sonst MIT-lizenzierten Repository; eine R-nahe, bitgleiche
  Rechenschicht für die sechs NumPy-Module wäre eine neue Modulvariante. Beides nicht still ändern.

## Was jetzt fertig ist

1. **Lokales `cor` und Zeitraum-Leser** `mcmc_functions_v1.6.R:2355–2366`, `1939–2018`
   ([Paket](literal_local_cor_2026_11_09/README.md), `tools/historical_local_cor.py`).
   - Eine einzige Definition in 540 R-Dateien, nur Basisfunktionen, kein Paket zu binden.
   - **Kinneret:** `multiple` ist genau `sources$core`. Es läuft 1964–1970, sofern `var(AP-Referenz) != 0` (ohne finale Eingaben offen).
     Sonst 1993. **Zeile 1999 wird nicht erreicht** (`sources[,4]` bleibt 1); das stand in der vorigen Übergabe anders.
     `hi_time` ist FALSE, also 0,95 und keine Zeilen 1982/2009. Zeile 1964 liest pann, Zeile 1966 tann.
   - `sum` läuft in `long double` (64 Bit je Schritt, einmal auf Double). Eine einfache Double-Summe ändert 199 von 300 Stressfällen.
     Python-`sum()` ist ab 3.12 kompensiert und taugt nicht als einfache Summe.
   - NA/NaN: Nur `x` wird bei `na.rm` geprüft. Das Token folgt dem ersten fehlenden Wert in `x`, sonst in `y`: ein Messbefund aus R 3.5.0, ohne Wirkung nach `is.na`.
   - R 3.5.0 mit wörtlichen Originalzeilen: 447 von 447 Werten bitgleich, 0 ulp. 135 neue Tests, 1.172 gezielte Tests grün.
2. **App-Stand 11** ([README](app_reproduction_2026_11_09_local_cor/README.md)): 14 Teilmodule, Tabelle C1–C8. Stände 1–10 eingefroren.

## Nächster Schritt (Phase 1)

Speicherortfrage nicht wiederholen. **Vor jedem neuen Baustein zuerst die Aufrufer und Zweige belegen**, wie in `bind_source.py` des letzten Pakets.

**Empfohlen:** `reconst`/`predictant` aus vorgegebenem `tf_sample` (`mcmc_functions_v1.6.R:1817–1820`): explizite Matrix, `%*%`, Spalten trennen. Der Weg ist für jeden Lauf aktiv.
Ein R-Vergleich auf Matrixprodukten ist mit `dgemm` nötig; Zeilen und Spalten sauber trennen. `reconst_ages`, `tf_sample` und `taxa.appear.core` bleiben Eingaben, keine Ziehung.

Danach `wt_power` (Zeile 1003) mit den Lesern 2023/2031. Erreichbarkeit dort zuerst klären: `spectral` hängt an `sources$speleothem`, also an `d18o.age` und der 150-km-Prüfung;
Zeile 2031 und 2032 sind für Kinneret nicht erreichbar (`hi_spec` FALSE, `mcmc_prepare.R:881`). Paketbindung (`biwavelet`, FFT) nur mit Beleg und Zeitfenster.

Danach ein neuer App-Stand am Ende von `nachbaustand.STATES`. Stand 11 bekommt dann seinen Linkcommit:
`git log -1 --format=%H -- docs/thesis_research/app_reproduction_2026_11_09_local_cor/reproduction_state.json`.
In `tools/app_pruefen.js` die Fehlstandskontrolle anpassen (jetzt `[0, 14]`), außerdem 39 Links, 17 Tabellen, 10 frühere Stände.
Neue Fallbezeichnungen dürfen keine verwendete wiederholen: 1–4, G1–G3, D1–D4, M1–M9, K1–K8 und C1–C8 sind belegt.

## Prüfen

```bash
python -m pytest tests/test_historical_local_cor.py tests/test_historical_reconstruction_climate.py tests/test_r_crosscheck_six_modules.py tests/test_historical_weighted_median.py tests/test_historical_dirichlet_ratio.py tests/test_r_runtime_crosscheck.py tests/test_historical_weights_stage.py tests/test_historical_sampler_preparation.py tests/test_historical_tf_missing_priors.py tests/test_historical_gamma_prior_grid.py tests/test_historical_taxon_pruning.py tests/test_historical_tf_preparation.py tests/test_historical_phase1_chain.py tests/test_historical_cdf_discrepancy.py tests/test_historical_age_selection.py tests/test_historical_age_depth.py tests/test_historical_final_pdf.py tests/test_kinneret_phase1_spec.py -q
PYTHONPATH=src python -m pytest tests/test_nachbaustand.py tests/test_forschungsstand.py tests/test_app.py tests/test_artenvergleich.py -q
python docs/thesis_research/literal_local_cor_2026_11_09/run_r_crosscheck.py .local/R/R-3.5.0/bin/x64/Rscript.exe
PYTHONPATH=src python -m kinneret_recon.cli app
PYTHONPATH=src python -m kinneret_recon.cli app --eine-datei
NODE_PATH=C:/Users/tnetz/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules node tools/app_pruefen.js results/app/index.html results/kinneret-app.html --kanal msedge
```

- Zwei Standorte wie in der [Übergabe nach der Gewichtsstufe](CLAUDE_NEXT_STEP_AFTER_WEIGHTS_STAGE_2026_11_03.md).
- Hinweise: Python-Heredocs scheitern, Hilfsskripte als Datei anlegen; `< /dev/null` gegen hängende Eingaben; Rscript braucht Windows-Pfade bzw. absolute Pfade;
  lange Browserläufe getrennt starten. Bei den Tests sind 119 App-Tests grün, 2 Browserwrapper übersprungen und explizit ersetzt.
- Nicht ausführen: Gesamtsuite `pytest tests`, `verify_packet.py`, `tools/verify_*` als Regressionsprüfer.

## Arbeitsgrenzen

- Keine stillen Ersatzdaten, Quellenkorrekturen oder Variantenentscheidungen. Paketbindungen nur mit Beleg und Zeitfenster.
- R nur für Gegenproben; Abweichungen messen und markieren, nicht verstecken. Kunstbeispiele nie als historische Rechnung darstellen.
- Baseline, Gates, Artenmodelle und Paperreferenz erhalten; Plots jung links, alt rechts.
- Testzahlen sind keine Fallreproduktion. Diese und frühere Übergaben eingefroren lassen.

Am Ende: Belegpaket, Status und Arbeitsplan, **neue konkrete Claude-Übergabe mit Link im Abschluss**, Commit und Push auf PR #77,
PR-Beschreibung fortschreiben, nicht mergen.
