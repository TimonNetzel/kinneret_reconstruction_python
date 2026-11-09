# Claude-Übergabe: nach `reconst`/`predictant` und App-Stand 12

Stand 09.11.2026. Gilt ebenso für ChatGPT/Codex. Vorrangiges Nutzerziel bleibt:
**die lokalen taxonbasierten Rekonstruktionen der Dissertation historisch nachvollziehbar nachbauen und in der App
umsetzen; Verbesserungen danach.**

- Deutsch, kurze Fortschrittsmeldungen, keine Subagenten.
- Nur den vom Nutzer benannten vorhandenen Checkout verwenden; keine Klone oder Worktrees.
- Bestehender Branch `dissertation-reconstruction`,
  [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77), kein automatischer Merge.
- Zuerst Git-/PR-Stand prüfen. Ausgangspunkt dieses Pakets war `38775e1`. Der Rechenschritt liegt in `7fb319f`, `3b6f2b3` und `599c6e4`; den
  App-Sicherungscommit liefert `git log -1 --format=%H -- docs/thesis_research/app_reproduction_2026_11_09_reconst_predictant/reproduction_state.json`.
- Commits nur mit `git -c user.name=TimonNetzel -c user.email=134919762+TimonNetzel@users.noreply.github.com`.
- Der Checkout-Pfad ist lang. Skripte vom Repo-Wurzelverzeichnis aus mit relativen Pfaden starten; `../../..` scheitert an der Windows-Pfadlänge.

## Geltende Nutzerentscheidungen

- **CRAN-Quellen lesen ist erlaubt.** Das R-3.5.0-Quellarchiv liegt unter `.local/reconst_predictant_2026_11_09/r_source` (unversioniert).
- **R-Läufe für Gegenproben** auf Kunstfixtures oder belegten Teilschritten sind erlaubt. R 3.5.0 unter `.local/R/R-3.5.0`,
  LaplacesDemon 16.1.6 unter `.local/R/library_ld`, matrixStats 0.56.0 unter `.local/R/library_ms` (unversioniert).
- Weiterhin **keine** Sampler, Trainings, Altersmodelle oder Fossilrekonstruktionen.
- Kinneret ist Slot 3 des Zusatzorte-Zweigs (`add.sites <- T`).
- **`diss` nur lesen, nur absolute Pfade, dorthin kein `cd`.**
- **Noch vom Nutzer zu entscheiden:** `tools/r_sort_gpl.py` bleibt GPL-2-oder-später im sonst MIT-lizenzierten Repository; eine R-nahe, bitgleiche
  Rechenschicht für die sechs NumPy-Module wäre eine neue Modulvariante. Beides nicht still ändern. Für `wt_power` kommt dieselbe Frage dazu (siehe unten).

## Was jetzt fertig ist

1. **`reconst` und die Prädiktanten** `mcmc_functions_v1.6.R:1817–1820`, Leser `1825`, `1829–1857`, `1939–1943`
   ([Paket](literal_reconst_predictant_2026_11_09/README.md), `tools/historical_reconst_predictant.py`).
   - **`%*%` im 3-Schleifen-Pfad rechnet in `double`.** Der Auftrag und die vorige Übergabe sprachen von `long double`; das stimmt nur für `internal_matprod` (`array.c:686`), nicht für `simple_matprod` (`array.c:702`). Quelle, `options.Rd` und Messung stimmen überein. Eine NaN-Zeile ändert die endlichen Zeilen also nicht.
   - **`reconst_ages` ist immer „alle Zeilen“.** Zeile 1500 endet in `& FALSE`, darum überschreibt 1522 die Holozänmaske aus 1466. Eine einzelne Zeile entsteht im lebenden Pfad nur bei `length(age) == 1`.
   - **Das Argument `tf_sample` wird nie gelesen;** der Rumpf liest `posterior_temp$tf_sample`.
   - Leser für Kinneret: 1825, 1942 und, je nach `sources$core`, 1964/1966 oder 1993. 1831–1837 (`taxa_change`), 1867–1902 (`past_climate_module`), 1923–1929 (`stad_interstad`) werden nicht erreicht. Zeile 1850 läuft nur bei `recent`, also bei einem Alter ≤ 11000: **offen ohne finale Alter.**
   - NA/NaN: Das Produkt nimmt den ersten fehlenden Operanden von `x * y`, die Summe behält einen fehlenden Akkumulator. Messbefund auf 1.296 Kombinationen, keine Herleitung.
   - R 3.5.0 mit wörtlichen Originalzeilen: 3.009 von 3.009 Werten bitgleich, 0 ulp. 230 neue Tests, 1.402 gezielte Tests grün.
2. **App-Stand 12** ([README](app_reproduction_2026_11_09_reconst_predictant/README.md)): 15 Teilmodule, Tabelle R1–R8. Stände 1–11 eingefroren. Geprüft und festgeschrieben in [validation.json](app_reproduction_2026_11_09_reconst_predictant/validation.json): App-Tests 134 bestanden und 2 übersprungen, historisch 1.402, Edge in zehn Fällen und zwei Standorte mit Exit 0.

## Nächster Schritt (Phase 1): erst Entscheidung, dann `wt_power`

`wt_power` (Zeile 1003) ist **nicht** gebaut. Vorab belegt, noch nicht als Paket:

- **Erreichbarkeit:** `spectral` wird in `mcmc_prepare.R:1203` aus `which_sources == 3` gesetzt. `sources$speleothem` entscheidet sich in 762–814 (Abstand ≤ 150 km, Altersabdeckung > 0,5) und braucht `d18o.age` und die finalen Koordinaten: für Kinneret offen. Zeile 2031/2032 werden nicht erreicht (`hi_spec` FALSE, `mcmc_prepare.R:881`). Die aktive Stelle ist also höchstens 2023.
- **Abhängigkeiten:** `wt_power` liest die freie Globale `wb` (nicht im Argument), gebaut in `mcmc_prepare.R:932–935` über `rcpp_wt_bases_morlet`; `n.obs`, `zeros`, `invflen` in 938–941. `site.age` wird bei `sources$speleothem == 1` in 825–826 **überschrieben**. Zeile 1007 (`test <- …`) hat keinen Leser.
- **Nutzerentscheidung nötig:** `fft`/`mvfft` sind Basis-R. Eine bitgleiche Nachbildung heißt, R-`fft.c` (GPL) zu übersetzen. Das ist dieselbe offene Lizenzfrage wie `tools/r_sort_gpl.py`. Alternative: NumPy-FFT mit gemessener ulp-Abweichung als ausdrücklich markierte Nicht-Bitgleichheit. Das nicht still entscheiden.
- Paketmaskierung von `fft` durch angehängte Pakete (`signal`, `pracma`, `WaveletComp`, `biwavelet`) wurde noch nicht geprüft; nur mit Beleg und Zeitfenster binden.

Falls der Nutzer das nicht entscheidet oder `wt_power` zurückstellt: Empfohlen ist `rtruncnorm_cpp` (1812–1813) nur als Quellen- und Bindungsprüfung, ohne Ziehung, oder der Eingabevertrag des Kinneret-Falls (Slot 3) als fachlicher Meilenstein.

Danach ein neuer App-Stand am Ende von `nachbaustand.STATES`. Stand 12 bekommt dann seinen Linkcommit (Befehl oben, entsprechend für den neuen Ordner).
In `tools/app_pruefen.js` die Fehlstandskontrolle anpassen (jetzt `[0, 15]`), außerdem 42 Links, 18 Tabellen, 11 frühere Stände.
Neue Fallbezeichnungen dürfen keine verwendete wiederholen: 1–4, G1–G3, D1–D4, M1–M9, K1–K8, C1–C8 und R1–R8 sind belegt.

## Prüfen

```bash
python -m pytest tests/test_historical_reconst_predictant.py tests/test_historical_local_cor.py tests/test_historical_reconstruction_climate.py tests/test_r_crosscheck_six_modules.py tests/test_historical_weighted_median.py tests/test_historical_dirichlet_ratio.py tests/test_r_runtime_crosscheck.py tests/test_historical_weights_stage.py tests/test_historical_sampler_preparation.py tests/test_historical_tf_missing_priors.py tests/test_historical_gamma_prior_grid.py tests/test_historical_taxon_pruning.py tests/test_historical_tf_preparation.py tests/test_historical_phase1_chain.py tests/test_historical_cdf_discrepancy.py tests/test_historical_age_selection.py tests/test_historical_age_depth.py tests/test_historical_final_pdf.py tests/test_kinneret_phase1_spec.py -q
PYTHONPATH=src python -m pytest tests/test_nachbaustand.py tests/test_forschungsstand.py tests/test_app.py tests/test_artenvergleich.py -q
python docs/thesis_research/literal_reconst_predictant_2026_11_09/run_r_crosscheck.py .local/R/R-3.5.0/bin/x64/Rscript.exe
PYTHONPATH=src python -m kinneret_recon.cli app
PYTHONPATH=src python -m kinneret_recon.cli app --eine-datei
NODE_PATH=C:/Users/tnetz/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules node tools/app_pruefen.js results/app/index.html results/kinneret-app.html --kanal msedge
```

- Zwei Standorte wie in der [Übergabe nach der Gewichtsstufe](CLAUDE_NEXT_STEP_AFTER_WEIGHTS_STAGE_2026_11_03.md). Exit-Code **und** Log von `app_pruefen.js` prüfen: Ein Abbruch kann ohne `FEHLER`-Zeile enden.
- Bei einem Abbruch zuerst das Windows-Systemprotokoll auf Kernel-Power 506/507 prüfen. Das Klick-Timeout von Stand 12 kam von einem Standby durch Zuklappen des Deckels ([Befund](app_reproduction_2026_11_09_reconst_predictant/README.md)).
- Die lokale Fetch-Konfiguration holt nur `main`, `origin/dissertation-reconstruction` existiert hier also nicht. Den PR-Kopf mit `gh pr view 77` oder `git ls-remote` prüfen.
- Hinweise: Python-Heredocs scheitern, Hilfsskripte als Datei anlegen; `< /dev/null` gegen hängende Eingaben; Rscript braucht Windows-Pfade bzw. absolute Pfade;
  lange Browserläufe getrennt starten. App-Tests: 134 grün, 2 Browserwrapper übersprungen und explizit ersetzt.
- Nicht ausführen: Gesamtsuite `pytest tests`, `verify_packet.py`, `tools/verify_*` als Regressionsprüfer.

## Arbeitsgrenzen

- Keine stillen Ersatzdaten, Quellenkorrekturen oder Variantenentscheidungen. Paketbindungen nur mit Beleg und Zeitfenster.
- R nur für Gegenproben; Abweichungen messen und markieren, nicht verstecken. Kunstbeispiele nie als historische Rechnung darstellen.
- Kein GPL-Text aus R in dieses Repo übernehmen, solange die Lizenzfrage offen ist; das Paket speichert nur Zeilennummern und Hashes.
- Baseline, Gates, Artenmodelle und Paperreferenz erhalten; Plots jung links, alt rechts.
- Testzahlen sind keine Fallreproduktion. Diese und frühere Übergaben eingefroren lassen.

Am Ende: Belegpaket, Status und Arbeitsplan, **neue konkrete Claude-Übergabe mit Link im Abschluss**, Commit und Push auf PR #77,
PR-Beschreibung fortschreiben, nicht mergen.
