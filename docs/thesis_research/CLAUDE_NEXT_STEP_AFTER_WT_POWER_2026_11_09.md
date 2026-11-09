# Claude-Übergabe: nach `wt_power` und App-Stand 13

Stand 09.11.2026. Gilt ebenso für ChatGPT/Codex. Vorrangiges Nutzerziel bleibt:
**die lokalen taxonbasierten Rekonstruktionen der Dissertation historisch nachvollziehbar nachbauen und in der App
umsetzen; Verbesserungen danach.**

- Deutsch, kurze Fortschrittsmeldungen, keine Subagenten.
- Nur den vom Nutzer benannten vorhandenen Checkout verwenden; keine Klone oder Worktrees.
- Bestehender Branch `dissertation-reconstruction`,
  [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77), kein automatischer Merge.
- Zuerst Git-/PR-Stand prüfen. Der Rechenschritt liegt in `ddbce9e`; den
  App-Sicherungscommit liefert `git log -1 --format=%H -- docs/thesis_research/app_reproduction_2026_11_09_wt_power/reproduction_state.json`.
- Commits nur mit `git -c user.name=TimonNetzel -c user.email=134919762+TimonNetzel@users.noreply.github.com`.
- Der Checkout-Pfad ist lang. Skripte vom Repo-Wurzelverzeichnis aus mit relativen Pfaden starten; `../../..` scheitert an der Windows-Pfadlänge.

## Geltende Nutzerentscheidungen

- **`wt_power`: NumPy-FFT mit gemessener und ausdrücklich markierter ulp-Abweichung** (09.11.2026), keine Übersetzung von R-`fft.c`. Nicht erneut fragen. `wt_power` nie als bitgleich zu R darstellen.
- **CRAN-Quellen lesen ist erlaubt.** Das R-3.5.0-Quellarchiv liegt unter `.local/reconst_predictant_2026_11_09/r_source` (unversioniert).
- **R-Läufe für Gegenproben** auf Kunstfixtures oder belegten Teilschritten sind erlaubt. R 3.5.0 unter `.local/R/R-3.5.0`,
  LaplacesDemon 16.1.6 unter `.local/R/library_ld`, matrixStats 0.56.0 unter `.local/R/library_ms` (unversioniert).
- Weiterhin **keine** Sampler, Trainings, Altersmodelle oder Fossilrekonstruktionen.
- Kinneret ist Slot 3 des Zusatzorte-Zweigs (`add.sites <- T`).
- **`diss` nur lesen, nur absolute Pfade, dorthin kein `cd`.**
- **Noch vom Nutzer zu entscheiden:** `tools/r_sort_gpl.py` bleibt GPL-2-oder-später im sonst MIT-lizenzierten Repository; eine R-nahe, bitgleiche
  Rechenschicht für die sechs NumPy-Module wäre eine neue Modulvariante. Beides nicht still ändern.

## Was jetzt fertig ist

1. **`wt_power` und sein Leser** `mcmc_functions_v1.6.R:1003–1012`, `2023–2026`
   ([Paket](literal_wt_power_2026_11_09/README.md), `tools/historical_wt_power.py`), mit expliziten Reihen und explizitem `wb`.
   - R 3.5.0, 120 Stressfälle: 22.256 von 22.256 zentrierten Werten bitgleich; 5.938 von 39.876 Leistungen bitgleich, höchstens 606 ulp, relativ zum Spaltenmaximum höchstens 2,5e-15; `post_spec` höchstens 12 ulp; keine Token-Abweichung.
   - 212 neue Tests, 1.614 gezielte Tests grün in 20 Dateien.
2. **App-Stand 13** ([README](app_reproduction_2026_11_09_wt_power/README.md)): 16 Teilmodule, `wt_power` bedingt, Tabelle W1–W8. Stände 1–12 eingefroren. Geprüft und festgeschrieben in [validation.json](app_reproduction_2026_11_09_wt_power/validation.json): App-Tests 151 bestanden und 2 übersprungen, historisch 1.614, Edge in zehn Fällen und zwei Standorte mit Exit 0, keine der 19 Tabellen bei 390 px breiter als ihr Rahmen.

## Offen bei `wt_power`: Erreichbarkeit für Kinneret

- `wt_power` läuft bei `sources$speleothem == 1` auch in `mcmc_prepare.R:947` und `966` (dort 100-mal in `system.time`), nicht nur in `mcmc_functions_v1.6.R:2023`. Die Ergebnisse haben dort keinen Leser.
- `mcmc_prepare.R:1145` setzt für Kinneret `sources$speleothem <- 0`, steht aber hinter `if(case_studies)`. `main_v1.2.R:672` setzt `case_studies <- F`, im gebundenen Lauf greift die Zeile also nicht.
- **Die Erreichbarkeit für Kinneret bleibt offen.** `sources$speleothem` hängt an `d18o.age` und den finalen Koordinaten (`mcmc_prepare.R:759–814`). Nicht aus der Zeile 1145 auf „läuft nicht“ schließen.
- W8 zeigt `NaN` (Modul) gegen `NA` (R): ein Mischfall (`kind` `mixed`), für den das Modul nur „fehlend“ beansprucht. Die Tabelle erklärt das nicht eigens. Bei einem nächsten App-Stand könnte ein Satz dazu; Stand 13 bleibt eingefroren.

## Nächster Schritt (Phase 1): Vorschlag

Es ist kein Rechenschritt festgelegt. **Empfohlen: `rtruncnorm_cpp` nur als Quellen- und Bindungsprüfung, ohne Ziehung.** Das ist klein, braucht keine fachliche Entscheidung und schließt die Herkunft von `tf_sample`, das `reconst` bisher als Vorgabe bekommt.

Vorab gelesen, noch nicht als Paket:

- Die Funktion steht in der Quelle selbst, `mcmc_functions_v1.6.R:1556–1570`, als `sourceCpp`-Block, und läuft in einer Schleife über `r_truncnorm(mean(i), sd(i), a(i), b(i))`.
- `r_truncnorm` kommt aus `truncnorm.h` mit `Rcpp::depends(RcppDist)` (Zeilen 1558/1560). Die Fassung von RcppDist ist **nicht gebunden**.
- Aufrufe: Zeile 1812 mit `tf_quarts_tann[,1]`, `[,3]`, Zeile 1813 mit `tf_quarts_pann[,1]`, `[,3]`.

Umfang: CRAN-Fassungen von RcppDist im Fenster 2018-04-23…2023-07-07 lesen, Textgleichheit von `r_truncnorm` und seinem Zufallsverbrauch belegen, Zeitfenster und Lücken nennen. **Keine Ziehung, kein Lauf.** Kein GPL-Text ins Repo, nur Zeilennummern und Hashes.

Danach oder stattdessen der fachliche Meilenstein: **Eingabevertrag des Kinneret-Falls (Slot 3)**. Dort laufen die offenen Bedingungen zusammen (`sources$speleothem`, `sources$core`, `recent`). Weil die finalen Eingaben fehlen, kann das nur festhalten, was gebunden ist und was fehlt.

Wird ein neuer App-Stand nötig: Eintrag am Ende von `nachbaustand.STATES`. Stand 13 bekommt dann seinen Linkcommit (Befehl oben).
In `tools/app_pruefen.js` die Fehlstandskontrolle anpassen (jetzt `[0, 16]`), außerdem 45 Links, 19 Tabellen, 12 frühere Stände.
Neue Fallbezeichnungen dürfen keine verwendete wiederholen: 1–4, G1–G3, D1–D4, M1–M9, K1–K8, C1–C8, R1–R8 und W1–W8 sind belegt.

## Prüfen

```bash
python -m pytest tests/test_historical_wt_power.py tests/test_historical_reconst_predictant.py tests/test_historical_local_cor.py tests/test_historical_reconstruction_climate.py tests/test_r_crosscheck_six_modules.py tests/test_historical_weighted_median.py tests/test_historical_dirichlet_ratio.py tests/test_r_runtime_crosscheck.py tests/test_historical_weights_stage.py tests/test_historical_sampler_preparation.py tests/test_historical_tf_missing_priors.py tests/test_historical_gamma_prior_grid.py tests/test_historical_taxon_pruning.py tests/test_historical_tf_preparation.py tests/test_historical_phase1_chain.py tests/test_historical_cdf_discrepancy.py tests/test_historical_age_selection.py tests/test_historical_age_depth.py tests/test_historical_final_pdf.py tests/test_kinneret_phase1_spec.py -q
PYTHONPATH=src python -m pytest tests/test_nachbaustand.py tests/test_forschungsstand.py tests/test_app.py tests/test_artenvergleich.py -q
python docs/thesis_research/literal_wt_power_2026_11_09/run_r_crosscheck.py .local/R/R-3.5.0/bin/x64/Rscript.exe
PYTHONPATH=src python -m kinneret_recon.cli app
PYTHONPATH=src python -m kinneret_recon.cli app --eine-datei
NODE_PATH=C:/Users/tnetz/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules node tools/app_pruefen.js results/app/index.html results/kinneret-app.html --kanal msedge
```

- Zwei Standorte: `.local/wt_power_2026_11_09/zweiter_standort.py .`, dann `app_pruefen.js --zwei-standorte .local/wt_power_2026_11_09/app2/index.html --kanal msedge`. Tabellenbreiten und Fotos: `.local/wt_power_2026_11_09/breiten_und_fotos.js` (für einen neuen Stand die Tabellenkennung anpassen). Beide Skripte sind unversioniert.
- Exit-Code **und** Log von `app_pruefen.js` prüfen: Ein Abbruch kann ohne `FEHLER`-Zeile enden. Lange Browserläufe getrennt von anderen Schritten starten; der Lauf dauert knapp 3 Minuten.
- Der Deckel muss während der Browserläufe offen bleiben. Bei einem Abbruch zuerst das Windows-Systemprotokoll auf Kernel-Power 506/507 prüfen ([Befund von Stand 12](app_reproduction_2026_11_09_reconst_predictant/README.md)).
- Die lokale Fetch-Konfiguration holt nur `main`, `origin/dissertation-reconstruction` existiert hier also nicht. Den PR-Kopf mit `gh pr view 77` oder `git ls-remote` prüfen.
- Hinweise: Python-Heredocs scheitern, Hilfsskripte als Datei anlegen; `< /dev/null` gegen hängende Eingaben; Rscript braucht Windows- bzw. absolute Pfade.
- Nicht ausführen: Gesamtsuite `pytest tests`, `verify_packet.py`, `tools/verify_*` als Regressionsprüfer.

## Arbeitsgrenzen

- Keine stillen Ersatzdaten, Quellenkorrekturen oder Variantenentscheidungen. Paketbindungen nur mit Beleg und Zeitfenster.
- R nur für Gegenproben; Abweichungen messen und markieren, nicht verstecken. Kunstbeispiele nie als historische Rechnung darstellen.
- Kein GPL-Text aus R in dieses Repo; das Paket speichert nur Zeilennummern und Hashes.
- Baseline, Gates, Artenmodelle und Paperreferenz erhalten; Plots jung links, alt rechts.
- Testzahlen sind keine Fallreproduktion. Elf Nachweisbündel bleiben offen. Diese und frühere Übergaben, eingefrorene Nachweise und die App-Stände 1–13 unverändert lassen.

Am Ende: Belegpaket, Status und Arbeitsplan, **neue konkrete Claude-Übergabe mit Link im Abschluss**, Commit und Push auf PR #77,
PR-Beschreibung fortschreiben, nicht mergen.
