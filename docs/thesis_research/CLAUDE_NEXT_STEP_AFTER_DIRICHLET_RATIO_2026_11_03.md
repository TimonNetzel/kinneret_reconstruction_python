# Claude-Übergabe: nach dem Dirichlet-Quotienten und App-Stand 7

Stand 03.11.2026. Gilt ebenso für ChatGPT/Codex. Vorrangiges Nutzerziel bleibt:
**die lokalen taxonbasierten Rekonstruktionen der Dissertation historisch nachvollziehbar nachbauen und in der App
umsetzen; Verbesserungen danach.**

- Deutsch, kurze Fortschrittsmeldungen, keine Subagenten.
- Nur den vom Nutzer benannten vorhandenen Checkout verwenden; keine Klone oder Worktrees.
- Bestehender Branch `dissertation-reconstruction`,
  [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77), kein automatischer Merge.
- Zuerst Git-/PR-Stand prüfen. Ausgangspunkt dieses Pakets war `c0cdf81`. Der Rechenschritt liegt in `766a1ff`;
  den App-Sicherungscommit liefert `git log -1 -- docs/thesis_research/app_reproduction_2026_11_03_dirichlet_ratio`.
- Commits nur mit `git -c user.name=TimonNetzel -c user.email=134919762+TimonNetzel@users.noreply.github.com`.

## Geltende Nutzerentscheidungen

- **CRAN-Quellen lesen ist erlaubt.**
- **R-Läufe für Gegenproben** auf Kunstfixtures oder belegten Teilschritten sind erlaubt. R 3.5.0 liegt unter
  `.local/R/R-3.5.0`, LaplacesDemon 16.1.6 unter `.local/R/library_ld` (beides unversioniert).
- Weiterhin **keine** Sampler, Trainings, Altersmodelle oder Fossilrekonstruktionen.
- Kinneret ist Slot 3 des Zusatzorte-Zweigs ([Laufzeitbelege](r_runtime_crosscheck_2026_11_03/README.md)).

## Was jetzt fertig ist

1. **Dirichlet-Quotient** `mcmc_functions_v1.6.R:1269–1274`
   ([Paket](literal_dirichlet_ratio_2026_11_03/README.md), `tools/historical_dirichlet_ratio.py`).
   - Gewichte explizit.
   - `ddirichlet` wörtlich nach LaplacesDemon; die Normierungszeile weist nichts zu.
   - `is.na` → 0, `Inf` bleibt; Vorschlag ohne `dirichlet_spread`.
   - `lgamma`/`log`/`exp` einmal korrekt gerundet (80-stellige `Decimal`-Zwischenrechnung); nicht `math.lgamma`
     verwenden (bis 3 ulp daneben).
   - Unabhängige Spouge-Referenz: bitgleich.
   - R 3.5.0 + LaplacesDemon 16.1.6: 59 von 62 Werten bitgleich, höchstens 4 ulp. Ursache ist R-`lgamma(0.4)`,
     1 ulp neben dem korrekt gerundeten Wert; an den Zwischenschritten belegt.
   - 50 neue Tests, 752 gezielte Tests grün.
2. **App-Stand 7** ([README](app_reproduction_2026_11_03_dirichlet_ratio/README.md), Pin `cf1b77f2…`).
   - Elf Teilmodule.
   - Neue Zeile „R-Gegenprobe“ aus `runtime_check`; die Testsumme schließt deren 13 Belegtests ein.
   - Tabelle D1–D4 im Beispiel.
   - Stände 1–6 eingefroren; Stand 6 erhielt nur Linkcommit `3104f3c`.
   - Edge in beiden Formaten und an zwei Standorten im ersten Lauf ok.
   - Prüfstand: 28 Links, 12 Tabellen.

## Nächster Schritt (Phase 1)

Speicherortfrage nicht wiederholen.

**Empfohlen: `weightedMedian`** für die NA-Maske von `which.tf.missing`:

- Aufruf: `weightedMedian(x1range.bck, tf.x1.current[j,])` bzw. mit `x2range.bck` (`mcmc_prepare.R:1631/1651`),
  alle Standardwerte. Der Anbieter ist matrixStats, nur im Fenster gebunden.
- Vorgehen:
  1. Zuerst die gemeinsamen matrixStats-Hilfsdateien (Makros, Sortierung, `R_INDEX_GET`) über die 13 Fassungen im
     Fenster vergleichen. Die eigenen `weightedMedian`-Dateien zeigen für diesen Aufruf keine Rechenänderung
     ([Diffs](r_runtime_crosscheck_2026_11_03/weighted_median_diffs.txt)).
  2. Dann den C-Rechenweg für numerisches `x`, `w` mit NA, `idxs = NULL`, `na.rm = FALSE`,
     `interpolate = TRUE`, `ties = NULL` nachbilden.
  3. In R 3.5.0 gegen matrixStats prüfen. Das Windows-Binärpaket für R 3.5 aus `bin/windows/contrib/3.5` im
     CRAN-Archiv vorher per Hash und Fassung festhalten.
  4. Klären, ob eine NA-Gewichtszeile wirklich NA liefert; davon hängt die Maske `which.tf.missing` ab.

Alternativen:

- Die sechs Module ohne R-Gegenprobe gegen ihre Originalzeilen prüfen: Alters-/Tiefenübertragung, Altersauswahl,
  End-PDF, CDF-Abweichung, TF-Vorbereitung, Taxon-Pruning. `r_runtime_crosscheck_2026_11_03/crosscheck.R` lässt
  sich erweitern.
- Belegen, was `tann <- FALSE` (`mcmc_prepare.R:1248`, oberste Ebene) im Kinneret-Lauf bewirkt. `tann` wird
  `reconstruction` in `mcmc_conduction.R:146` übergeben; die Leser sind belegt zu sammeln, nicht zu deuten.

Danach ein neuer App-Stand am Ende von `nachbaustand.STATES`. Stand 7 bekommt dann seinen Linkcommit:
`git log -1 --format=%H -- docs/thesis_research/app_reproduction_2026_11_03_dirichlet_ratio/reproduction_state.json`.

`tools/app_pruefen.js` (jetzt 28 Links, 12 Tabellen, 11 Module, 6 frühere Stände) und `tests/test_nachbaustand.py`
anpassen. Neue Fallbezeichnungen dürfen keine im Abschnitt verwendete wiederholen: 1–4, G1–G3 und D1–D4 sind belegt.

## Prüfen

```bash
python -m pytest tests/test_historical_dirichlet_ratio.py tests/test_r_runtime_crosscheck.py tests/test_historical_weights_stage.py tests/test_historical_sampler_preparation.py tests/test_historical_tf_missing_priors.py tests/test_historical_gamma_prior_grid.py tests/test_historical_taxon_pruning.py tests/test_historical_tf_preparation.py tests/test_historical_phase1_chain.py tests/test_historical_cdf_discrepancy.py tests/test_historical_age_selection.py tests/test_historical_age_depth.py tests/test_historical_final_pdf.py tests/test_kinneret_phase1_spec.py -q
PYTHONPATH=src python -m pytest tests/test_nachbaustand.py tests/test_forschungsstand.py tests/test_app.py tests/test_artenvergleich.py -q
python docs/thesis_research/literal_dirichlet_ratio_2026_11_03/run_r_crosscheck.py .local/R/R-3.5.0/bin/x64/Rscript.exe .local/R/library_ld
NODE_PATH=C:/Users/tnetz/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules node tools/app_pruefen.js results/app/index.html results/kinneret-app.html --kanal msedge
```

- Builds und zwei Standorte wie in der [Übergabe nach der Gewichtsstufe](CLAUDE_NEXT_STEP_AFTER_WEIGHTS_STAGE_2026_11_03.md).
- Shell-Hinweise:
  - Python-Heredocs mit Anführungszeichen scheitern; Hilfsskripte als Datei anlegen.
  - `< /dev/null` gegen hängende Standardeingaben.
  - CRAN drosselt Bündel von Downloads.
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
