# Claude-Übergabe: nach dem Kern der Gewichtsstufe und App-Stand 6

Stand 03.11.2026. Gilt ebenso für ChatGPT/Codex. Vorrangiges Nutzerziel bleibt:
**die lokalen taxonbasierten Rekonstruktionen der Dissertation historisch
nachvollziehbar nachbauen und in der App umsetzen; Verbesserungen danach.**

- Deutsch, kurze Fortschrittsmeldungen, keine Subagenten.
- Nur den vom Nutzer benannten vorhandenen Checkout verwenden; keine Klone oder Worktrees.
- Bestehender Branch `dissertation-reconstruction`,
  [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77), kein automatischer Merge.
- Zuerst Git-/PR-Stand und lokale Änderungen prüfen.
- Ausgangspunkt dieses Pakets war `235f797` (PR-Kopf, App-Stand 5, GitHub meldete `MERGEABLE`).
  Der Rechenschritt liegt in `4266c28`; den App-Sicherungscommit liefert
  `git log -1 -- docs/thesis_research/app_reproduction_2026_11_03_weights_stage`.
- Im Checkout ist keine Git-Identität gesetzt. Für Commits nur
  `git -c user.name=TimonNetzel -c user.email=134919762+TimonNetzel@users.noreply.github.com`.

## Was jetzt fertig ist

1. Lies `CLAUDE.md`, `AGENTS.md`, den neuesten [Projektstatus](PROJECT_STATUS.md), den
   [Arbeitsplan](LOCAL_DISS_RECONSTRUCTION_PLAN.md), das
   [Gewichtsstufen-Paket](literal_weights_stage_2026_11_03/README.md) und den
   [App-Stand 6](app_reproduction_2026_11_03_weights_stage/README.md).
2. **Determinismusprüfung** ([determinism_audit.json](literal_weights_stage_2026_11_03/determinism_audit.json)):
   `mcmc_functions_v1.6.R:1259` und `1276–1280` sind base R ohne Zufall und laufen in jeder
   Sampler-Iteration. [bind_source.py](literal_weights_stage_2026_11_03/bind_source.py) bindet dieselben acht
   Originale per Hash an das Vorpaket und prüft:
   - Klammertiefen (die Zeilen stehen auf Tiefe 1 des Funktionsrumpfs, nach der `while`-Schleife);
   - Definition und Aufruf von `parameter_stage_weights` (einmal definiert in 1242, einziger Aufruf
     `mcmc_conduction.R:142`, gewählt durch `gibbs <- F`);
   - alle Zuweisungen und Vorkommen von `rdirichlet`, `ddirichlet`, `new_weights`, `dirichlet_spread`,
     `$taxa.appear.core.temp`, `$z`, `$taxa_weights`;
   - dass `matrix`, `rowSums`, `ncol`, `nrow`, `min`, `rep`, `is.na` nicht umdefiniert werden;
   - im Rohtext einschließlich Rcpp-Zeichenketten, dass `rdirichlet`/`ddirichlet` nirgends definiert werden;
   - alle `library()`-Aufrufe in Laufreihenfolge.

   Befunde ohne Korrektur:
   - Der Anbieter von `rdirichlet`/`ddirichlet` ist **ungebunden**. Gebunden ist nur die Reihenfolge von
     55 `library()`-Aufrufen: 52 in `main_v1.2.R:4–62`, dazu `npreg`, `truncnorm` (`mcmc_prepare.R:1885–1886`)
     und `colorRamps` (`mcmc_trimming_posterior_trafo.R:416`) in der Schleife. Keine Paketfassung, keine
     Namensraumdatei, lokal keine R-Installation. `SBMSplitMerge::rdirichlet` (1216) steht im toten Gibbs-Zweig
     und belegt nichts für 1248.
   - Frühere Pakete nennen „50 angehängte Pakete“; programmatisch sind es 52 vor der Schleife. Die früheren
     Angaben bleiben unverändert.
   - Die Stufe liest `num_taxa` global (1254, 1269), ihr Argument `taxa.appear.core` nie.
   - Bei Ablehnung stellt `mcmc_conduction.R:329/330` Gewichte und `z` zurück, nicht aber
     `taxa.appear.core.temp`; das wird vor dem nächsten Lesen (1817) in 1277 überschrieben.
   - Überzählige `}` in `main_v1.2.R:740/788/829` bleiben.
3. **Modul** `tools/historical_weights_stage.py` mit **expliziten** Gewichten:
   - Recycling der `ncol` Gewichte über die zeilenweisen Werte, `matrix(byrow = T)`;
   - `rowSums(na.rm = T)`: NA und NaN übersprungen, Zeile ohne Werte 0, Zeile mit `-Inf` gibt `-Inf`;
     korrekt gerundete exakte Summe mit Kennzeichen „akkumulatorunabhängig“ (hinreichende
     Zweierpotenzschranke, Referenz zählt alle Teilsummen auf);
   - Division je Zeile mit IEEE-Sonderfällen (NaN, `±Inf`, vorzeichenbehaftete Null), `z = 1:ncol`;
   - abgelehnt: `+Inf`/negative Null in den Werten, falsche Länge, unter zwei Zeilen/Spalten, Gewichte, die
     nicht genau ein endlicher positiver Double je Spalte sind, Über-/Unterlauf.

   134 neue Tests gegen eine unabhängige `Fraction`-Referenz und Handerwartungen, **exakt**; vier Fälle sind
   direkt aus dem Modul der Sampler-Vorbereitung verkettet, dazu eine Gegenprobe mit `rw_coefficients`.
   Mit den bisherigen zusammen **689 grün**.
4. **App:** neuer datierter Nachbaustand
   `app_reproduction_2026_11_03_weights_stage/reproduction_state.json` (Pin `7eb9ac5e…`) als
   **letzter** Eintrag in `nachbaustand.STATES`.
   - Stände 1–5 sind bytegleich; Stand 5 erhielt nur seinen Linkcommit `235f797`.
   - Zehn Module; das neue Modul trägt kein Etikett „Bedingter Schritt“.
   - Zwei neue Kunsttabellen (Gewichtsbeispiel als Fortsetzung der Stand-5-Matrix; Zeilensummen mit
     Sonderwerten, Fälle **G1–G3**) und Sätze zu Rücksetzgewichten, Zweierpotenzen und `rdirichlet` im
     eingeklappten synthetischen Beispiel.
   - Beide Offlineformate in Edge geprüft (Handy 390 px/PC, Hell/Dunkel, DE/EN, Gesten, zwei
     Standorte): [validation.json](app_reproduction_2026_11_03_weights_stage/validation.json).
     Erster und Endlauf ok; dazwischen wurde nur ein Satz präzisiert.

**Weiter offen:**

- Kein vollständiger Originalfall; alle elf Nachweisbündel offen.
- Die echte Taxonmatrix hängt an Alters-Tiefen-Modell, Pollendaten und der NA-Maske aus
  `weightedMedian` (`matrixStats`-Fassung offen); die echten Gewichte an `rdirichlet`.
- Sampler-Schleife und Zufallsschritte sind nur dokumentiert.
- Die älteren Beispieltabellen (Gamma, fehlende TF, Sonderwerte der Sampler-Vorbereitung) tragen weiter die
  eingefrorenen Fallnummern 1–4; die neuen Tabellen nutzen G1–G3. Eine Vereinheitlichung wäre eine spätere
  App-Verbesserung, kein Nachbauschritt.

## Nächster fachlicher Schritt (Phase 1)

Speicherortfrage nicht wiederholen, keine pauschale Inventur. Den nächsten begrenzten
deterministischen Schritt aus bereits gebundenen Quellen isolieren und vorher selbst belegen.

**Empfohlen: die deterministische Sampler-Initialisierung** `mcmc_conduction.R:109–135` **ohne** 118
(`prior.recent`, hängt an `fitdistr`) und **ohne** 133 (`rmvnorm_rcpp`, Zufall). Zu belegen:

- Startgewichte `rep(1/num_taxa, num_taxa)` (113; Leser der Gewichtsstufe 1243) und
  `z = 1:posterior$how_many_taxa` (114);
- `vector("list", length = sample_length)` (112) und die NA-Arrays (109, 115–117, 120) mit ihren
  Dimensionsquellen (`n_age`, `num_age`, `nrow(tf.x1.current)`, `length(age)`): Herkunft und Gleichheit der
  Namen belegen, nicht annehmen;
- `x_curr <- rep(NA, 2*num_taxa*num_age)` (130) und die Verschränkung `tann_fill_ids = seq(1, 2*num_taxa, by = 2)`,
  `pann_fill_ids = seq(2, 2*num_taxa, by = 2)` (131–132) samt Lesern;
- `checking_intervals <- seq(burn_in, sample_length, by = 1000)` (135) mit `sample_length`/`burn_in` aus
  `main_v1.2.R:629–633` (`gibbs <- F`: 25000/5000; der Gibbs-Zweig 12500/2500 nur melden) und ihren Lesern.

Alternativen:

- `tf_x12_mean_current` (`mcmc_prepare.R:1692–1700`), reine Spaltenbindung; Leser belegt
  (`mcmc_conduction.R:146`, `mcmc_functions_v1.6.R:445/470/858`), Eingaben hängen an `weightedMedian`.
- Die Buchhaltung der Akzeptanz (`mcmc_conduction.R:302–355`) mit **expliziter** Gleichverteilungszahl und
  expliziten Quotienten; `runif`, `dnorm`, `dgamma`, `dbeta` bleiben dabei Eingaben bzw. nur dokumentiert.

`weightedMedian` erst nach belegter `matrixStats`-Fassung, keine Interpolationswahl. `rdirichlet`/`ddirichlet`
ohne Paketbeleg nicht emulieren. Zufallsschritte bleiben nur dokumentiert, kein Ersatzseed: `set.seed(2022)`/`rgamma`
1297–1303, `rbeta` 1709–1738, `sample` 1591, `rmvnorm_rcpp` 133, `rdirichlet` 1248, `runif` 304.

Bisherige Fixtures, Module, Tests und Pakete bytegleich lassen.

## Nächster App-Schritt

Nach einem neuen geprüften Teilmodul:

- Neuen Ordner `app_reproduction_<Datum>_<Kurzname>/` mit eigener `reproduction_state.json` anlegen.
- Eintrag **ans Ende** von `nachbaustand.STATES`; ältere Einträge nicht ändern.
- Dem bisher neuesten Stand 6 einen `commit` geben, der seine Datei enthält und gepusht ist:
  `git log -1 --format=%H -- docs/thesis_research/app_reproduction_2026_11_03_weights_stage/reproduction_state.json`.
- Modul- und Testzahlen aus dem neuen Paket übernehmen.
- `tools/app_pruefen.js` anpassen: Modulanzahl, Testzahl, eingefrorene Zeilen und Links (jetzt 24),
  Linkcommit, Tabellenzahl (jetzt 11), Fehlstandskontrolle.
- `tests/test_nachbaustand.py` anpassen.
- Neue Tabellen auf 390 px prüfen. Zeilenköpfe sind `nowrap`; lange Erklärungen gehören in den
  Begleittext. Neue Fallbezeichnungen dürfen keine im Abschnitt schon verwendete wiederholen.

Prüfen:

```bash
PYTHONPATH=src python -m pytest tests/test_nachbaustand.py tests/test_forschungsstand.py tests/test_app.py tests/test_artenvergleich.py -q
python -m pytest tests/test_historical_weights_stage.py tests/test_historical_sampler_preparation.py tests/test_historical_tf_missing_priors.py tests/test_historical_gamma_prior_grid.py tests/test_historical_taxon_pruning.py tests/test_historical_tf_preparation.py tests/test_historical_phase1_chain.py tests/test_historical_cdf_discrepancy.py tests/test_historical_age_selection.py tests/test_historical_age_depth.py tests/test_historical_final_pdf.py tests/test_kinneret_phase1_spec.py -q
NODE_PATH=C:/Users/tnetz/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules node tools/app_pruefen.js results/app/index.html results/kinneret-app.html --kanal msedge
```

Vorher beide Builds erzeugen: `kinneret-recon app` und `kinneret-recon app --eine-datei`. Ohne
Installation: `PYTHONPATH=src python -c "import sys; from kinneret_recon.cli import main; sys.argv=['kinneret-recon','app']; main()"`.

Zwei Standorte:

1. Kopie über `tests/test_app.py::_zweiter_standort` anlegen, z. B.
   `PYTHONPATH="src;tests" python -c "from pathlib import Path; import test_app; test_app._zweiter_standort(Path('results/app'), Path('.local/<paket>/app2'))"`.
2. Dann `node tools/app_pruefen.js --zwei-standorte <Kopie>/index.html --kanal msedge` mit demselben
   `NODE_PATH`.

Hinweis zur Shell: Längere Python-Heredocs mit Anführungszeichen scheiterten in der Bash dieser Umgebung;
Hilfsskripte als Datei im Scratchpad ablegen und mit absolutem Pfad aufrufen.

Nicht ausführen:

- die Gesamtsuite `pytest tests` (enthält Sampler-Unittests);
- `literal_tf_preparation_2026_11_02/verify_packet.py` und die zeitpunktgebundenen `tools/verify_*`
  als Regressionsprüfer.

## Arbeitsgrenzen

- Originale nur lesen; `diss` und private Quellpfade nie ändern. Den Shell-Arbeitsordner nicht in
  `diss` belassen, nur absolute Pfade.
- Keine stillen Ersatzdaten, Quellenkorrekturen oder Variantenentscheidungen: kein `inside.range`,
  Median, Seed, Rasterobergrenze oder Dirichlet-Paket ohne Beleg.
- Keine Trainings, Sampler, Altersmodelle, R-Läufe oder Fossilrekonstruktionen.
- Erhalten bleiben: Baseline, CLI, sieben experimentelle Artenmodelle, Paperreferenz und
  fallbezogene Monticchio-/Chronologie-Gates.
- Plots jung links, alt rechts.
- Testzahlen sind keine Fallreproduktion; Kunstbeispiele nie als historische Rechnung darstellen.
- Diese und frühere Übergaben eingefroren lassen.

Am Ende:

- neues Belegpaket;
- aktualisierter Projektstatus und Arbeitsplan;
- **eine neue konkrete Claude-Übergabe mit Link im Abschluss**;
- das geprüfte Paket auf dem bestehenden Branch committen und pushen, PR #77 fortschreiben, nicht mergen.
