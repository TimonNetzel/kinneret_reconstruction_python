# Claude-Übergabe: nach Laufzeitbelegen und R-3.5.0-Gegenprobe

Stand 03.11.2026. Gilt ebenso für ChatGPT/Codex. Vorrangiges Nutzerziel bleibt:
**die lokalen taxonbasierten Rekonstruktionen der Dissertation historisch
nachvollziehbar nachbauen und in der App umsetzen; Verbesserungen danach.**

- Deutsch, kurze Fortschrittsmeldungen, keine Subagenten.
- Nur den vom Nutzer benannten vorhandenen Checkout verwenden; keine Klone oder Worktrees.
- Bestehender Branch `dissertation-reconstruction`,
  [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77), kein automatischer Merge.
- Zuerst Git-/PR-Stand und lokale Änderungen prüfen. Ausgangspunkt dieses Pakets war `3104f3c` (App-Stand 6);
  den Sicherungscommit liefert `git log -1 -- docs/thesis_research/r_runtime_crosscheck_2026_11_03`.
- Commits nur mit `git -c user.name=TimonNetzel -c user.email=134919762+TimonNetzel@users.noreply.github.com`.

## Neue Nutzerentscheidungen (Chat 03.11.2026)

- Der Nutzer erinnert sich: `main_v1.2.R` rechnete alle lokalen Rekonstruktionen **taxonbasiert, ohne Biome**. Das
  umfasste alle durchführbaren EPD-Orte und die selbst erprobten Orte im Nahen Osten (erste Versuche in
  `my_reconst`, dann `my_…`).
- **CRAN-Quellen lesen ist erlaubt.**
- **R-Läufe für Gegenproben sind erlaubt.** Weiterhin keine Sampler, Trainings, Altersmodelle oder
  Fossilrekonstruktionen.
- Paketfassungen weiß der Nutzer nicht mehr; selbst suchen.

## Was jetzt fertig ist

Belegpaket: [r_runtime_crosscheck_2026_11_03](r_runtime_crosscheck_2026_11_03/README.md).

1. **Kinneret = Slot 3** des Zusatzorte-Zweigs ([site_binding.json](r_runtime_crosscheck_2026_11_03/site_binding.json)).
   - `add.sites <- T` (`main_v1.2.R:73`), `stations = 1:length(core.data)` ohne Slot 4, `station_id == 469`
     übersprungen.
   - Slotquellen: 1 Prespa, 2 En Gedi, 3 Kinneret, 4 Birkat Ram (alt), 5 Iznik, 6 Ohrid, 7 Dead Sea,
     8 Birkat Ram (Komposit), 9 Lake Van.
   - Kinneret-Schalter `mcmc_prepare.R` 790, 881, 1145, 1209 und **1248 `tann <- FALSE`** (oberste Ebene). Die
     Bedeutung von `tann` im Lauf ist noch nicht gebunden; nicht deuten, sondern belegen.
2. **R-Fassung:** Alle Laufeingaben in `local_reconstr_all_epd/in` stammen aus **R 3.5.0**
   ([rdata_writers.json](r_runtime_crosscheck_2026_11_03/rdata_writers.json)). In `diss` gibt es keine
   Versionsdatei; die Laufzeitfassung bleibt offen.
3. **Paketanbieter** im Fenster 2018-04-23…2023-07-07
   ([provider_analysis.json](r_runtime_crosscheck_2026_11_03/provider_analysis.json)):
   - `rdirichlet`/`ddirichlet` → **LaplacesDemon**, vor extraDistr, gtools und SBMSplitMerge. Der Text ist in
     16.1.0–16.1.6 gleich ([dirichlet_versions.json](r_runtime_crosscheck_2026_11_03/dirichlet_versions.json)).
     `ddirichlet` ist deterministisch; Paketbefund: `x / rowSums(x)` ohne Zuweisung.
   - `weightedMedian` → **matrixStats**, sechs Quelltextstände
     ([Diffs](r_runtime_crosscheck_2026_11_03/weighted_median_diffs.txt)). Gemeinsame Hilfsdateien sind noch nicht
     verglichen.
   - Bindung nur unter der Fensterannahme, nicht als Laufzeitfassung.
4. **R 3.5.0 lokal** unter `.local/R/R-3.5.0` (unversioniert; Installer in `.local/evidence_r_env_2026_11_03/downloads`,
   MD5 und Signatur geprüft). [run_crosscheck.py](r_runtime_crosscheck_2026_11_03/run_crosscheck.py) und
   [crosscheck.R](r_runtime_crosscheck_2026_11_03/crosscheck.R) werten Originalzeilen wörtlich aus und vergleichen
   bitgenau. Geprüft sind Sampler-Vorbereitung, Gewichtsstufe, Gamma-Raster (Obergrenze 50000) und
   Taxonzahl-Priors: **389 Prüfungen, alle gleich**, einschließlich der 14 markierten Werte.
5. 13 neue Belegtests, **702** gezielte Tests grün, beide Phase-1-Prüfer Exit 2. Keine App-Änderung.

## Nächster Schritt (Phase 1)

Speicherortfrage nicht wiederholen.

**Empfohlen: der `ddirichlet`-Quotient** `mcmc_functions_v1.6.R:1269–1274` mit **expliziten** `old_weights` und
`new_weights`:

- `weights_ratio = ddirichlet(new, rep(0.5, num_taxa)) / ddirichlet(old, rep(0.5, num_taxa))` (globales
  `num_taxa`);
- `proposal_ratio = ddirichlet(old, new) / ddirichlet(new, old)`, ohne `dirichlet_spread`;
- Produkt; `is.na` → 0 (fängt NA und NaN).

Den LaplacesDemon-Text wörtlich nachbilden, einschließlich der wirkungslosen Normierungszeile. Dazu gehören:

- Fehlerpfade (`x < 0`, `alpha <= 0`);
- `log(0)` bei Nullgewichten;
- `lgamma` mit nicht ganzzahligen Argumenten. Bitgleichheit mit R ist hier nicht selbstverständlich, weil
  `lgamma`, `log` und `exp` der C-Bibliothek nicht korrekt gerundet sein müssen. Abweichungen messen und markieren,
  nicht verstecken.

Die R-Gegenprobe läuft mit LaplacesDemon 16.1.x aus dem CRAN-Archiv. Es ist reines R, also aus dem Tarball in die
`.local`-Bibliothek installierbar; Fassung und Hash festhalten. `rdirichlet` bleibt ein Zufallsschritt (`rgamma`),
kein Ersatzseed.

Alternativen:

- `weightedMedian`: zuerst die gemeinsamen matrixStats-Hilfsdateien (Makros, Sortierung) im Fenster vergleichen.
  Dann die NA-Maske von `which.tf.missing` (`mcmc_prepare.R:1631/1651`) nachbilden und in R 3.5.0 gegen
  matrixStats prüfen (Windows-Binärpaket für R 3.5 aus dem CRAN-Archiv).
- Die übrigen sechs Module (Alters-/Tiefenübertragung, Altersauswahl, End-PDF, CDF-Abweichung, TF-Vorbereitung,
  Taxon-Pruning) gegen ihre Originalzeilen in R 3.5.0 prüfen. `crosscheck.R` lässt sich dafür erweitern.

Danach ein neuer App-Stand am Ende von `nachbaustand.STATES`, der die R-Gegenprobe je Modul zeigt. Stand 6 erhält
dann seinen Linkcommit: `git log -1 --format=%H -- docs/thesis_research/app_reproduction_2026_11_03_weights_stage/reproduction_state.json`.

## Prüfen

```bash
python -m pytest tests/test_r_runtime_crosscheck.py tests/test_historical_weights_stage.py tests/test_historical_sampler_preparation.py tests/test_historical_tf_missing_priors.py tests/test_historical_gamma_prior_grid.py tests/test_historical_taxon_pruning.py tests/test_historical_tf_preparation.py tests/test_historical_phase1_chain.py tests/test_historical_cdf_discrepancy.py tests/test_historical_age_selection.py tests/test_historical_age_depth.py tests/test_historical_final_pdf.py tests/test_kinneret_phase1_spec.py -q
python docs/thesis_research/r_runtime_crosscheck_2026_11_03/run_crosscheck.py .local/R/R-3.5.0/bin/x64/Rscript.exe
```

App-Prüfungen (bei App-Änderungen) wie in der
[vorigen Übergabe](CLAUDE_NEXT_STEP_AFTER_WEIGHTS_STAGE_2026_11_03.md).

Hinweise zur Shell:

- Python-Heredocs mit Anführungszeichen scheitern in der Bash dieser Umgebung; Hilfsskripte als Datei anlegen.
- Hängt ein Befehl an der Standardeingabe, hilft `< /dev/null`.
- CRAN drosselt Bündel von Downloads; die Skripte wiederholen deshalb.

## Arbeitsgrenzen

- Originale nur lesen; `diss` nie ändern; nur absolute Pfade.
- Keine stillen Ersatzdaten, Quellenkorrekturen oder Variantenentscheidungen.
- Paketbindung nur mit Beleg und mit genanntem Fenster.
- R nur für Gegenproben auf Kunstfixtures oder belegten Teilschritten. Keine Sampler, Trainings, Altersmodelle oder
  Fossilrekonstruktionen.
- Gesamtsuite `pytest tests`, `verify_packet.py` und `tools/verify_*` nicht als Regressionsprüfer verwenden.
- Baseline, Gates, sieben Artenmodelle und Paperreferenz erhalten. Plots jung links, alt rechts.
- Testzahlen sind keine Fallreproduktion. Diese und frühere Übergaben eingefroren lassen.

Am Ende: Belegpaket, Status und Arbeitsplan, eine **neue konkrete Claude-Übergabe mit Link im Abschluss**, Commit und
Push auf PR #77, PR-Beschreibung fortschreiben, nicht mergen.
