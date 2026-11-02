# Claude-Übergabe: nach der Sampler-Vorbereitung und App-Stand 5

Stand 02.11.2026. Gilt ebenso für ChatGPT/Codex. Vorrangiges Nutzerziel bleibt:
**die lokalen taxonbasierten Rekonstruktionen der Dissertation historisch
nachvollziehbar nachbauen und in der App umsetzen; Verbesserungen danach.**

- Deutsch, kurze Fortschrittsmeldungen, keine Subagenten.
- Nur den vom Nutzer benannten vorhandenen Checkout verwenden; keine Klone oder Worktrees.
- Bestehender Branch `dissertation-reconstruction`,
  [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77), kein automatischer Merge.
- Zuerst Git-/PR-Stand und lokale Änderungen prüfen.
- Ausgangspunkt dieses Pakets war `0a3dcdb` (PR-Kopf, App-Stand 4, GitHub meldete `MERGEABLE`).
  Der Rechenschritt liegt in `00ab021`; den App-Sicherungscommit liefert
  `git log -1 -- docs/thesis_research/app_reproduction_2026_11_02_sampler_prep`.
- Im Checkout ist keine Git-Identität gesetzt. Für Commits nur
  `git -c user.name=TimonNetzel -c user.email=134919762+TimonNetzel@users.noreply.github.com`.

## Was jetzt fertig ist

1. Lies `CLAUDE.md`, `AGENTS.md`, den neuesten [Projektstatus](PROJECT_STATUS.md), den
   [Arbeitsplan](LOCAL_DISS_RECONSTRUCTION_PLAN.md), das
   [Sampler-Vorbereitungspaket](literal_sampler_preparation_2026_11_02/README.md) und den
   [App-Stand 5](app_reproduction_2026_11_02_sampler_prep/README.md).
2. **Determinismusprüfung** ([determinism_audit.json](literal_sampler_preparation_2026_11_02/determinism_audit.json)):
   `mcmc_conduction.R:53–79` ist base R ohne Zufall und im gebundenen Hauptlauf aktiv.
   [bind_source.py](literal_sampler_preparation_2026_11_02/bind_source.py) bindet acht Originale per Hash:
   die fünf Laufdateien sowie `check_ages.R`, `prior_climate.R` und `loess_functions.R`. Es prüft:
   - Klammertiefen;
   - alle Zuweisungen von `gibbs`, `F`, `T`, `num_taxa`, `taxa.appear.core` und `taxa_information`;
   - dass keine Basisfunktion umdefiniert wird;
   - alle Leser.

   `gibbs <- F` (`main_v1.2.R:624`) ist die einzige Zuweisung.
   Quellbefunde ohne Korrektur:
   - `main_v1.2.R` hat überzählige `}` in 740, 788 und 829 (nach der gebundenen Schleife 651–684).
     Ein Ganzdatei-`source()` ist deshalb unmöglich; das bleibt eine Frage der Laufbindung.
   - `c` wird in `mcmc_prepare.R:1703` zur Schleifenvariable.
   - Annahme wie bisher: Die 50 angehängten Pakete überdecken die Basisfunktionen nicht
     (Namensräume ungebunden).
3. **Modul** `tools/historical_sampler_preparation.py`:
   - `gibbs` als Boolwert: `FALSE` gibt `num_taxa` (Zeile 57); `TRUE` gibt `floor(n/2)` (Zeile 54),
     nur gemeldet.
   - `c(t(.))` flacht zeilenweise ab; Zeile 75 überschreibt 64.
   - `ncol`, `nrow` und `colMeans(na.rm = T)` gelten der Rohmatrix. `colMeans` liefert den exakten
     Bruch und den korrekt gerundeten Double und markiert akkumulatorabhängige Mittel; ohne Leser.
   - Spaltendivision durch `max(na.rm = T)`: NA bleibt NA, `0/0` gibt NaN, `x/0` für `x < 0` gibt `-Inf`.
   - `jeffreys_prior = rep(0.5, n)`.
   - Abgelehnt: Ganz-NA-Spalte (`-Inf` mit Warnung), eine Zeile oder Spalte (`drop`),
     `num_taxa ≠ ncol`, `Inf`, negative Null, nichtnumerische Eingaben.

   118 neue Tests gegen eine unabhängige `Fraction`-Referenz und Handerwartungen, **exakt**;
   mit den bisherigen zusammen **555 grün**.
4. **App:** neuer datierter Nachbaustand
   `app_reproduction_2026_11_02_sampler_prep/reproduction_state.json` (Pin `61c6f231…`) als
   **letzter** Eintrag in `nachbaustand.STATES`.
   - Stände 1–4 sind bytegleich; Stand 4 erhielt nur seinen Linkcommit `0a3dcdb`.
   - Neun Module; das neue Modul trägt kein Etikett „Bedingter Schritt“.
   - Zwei neue Kunsttabellen (Matrixbeispiel mit `*`-Mitteln, Sonderwerte) und ein Gibbs-Satz im
     eingeklappten synthetischen Beispiel.
   - Beide Offlineformate in Edge geprüft (Handy 390 px/PC, Hell/Dunkel, DE/EN, Gesten, zwei
     Standorte): [validation.json](app_reproduction_2026_11_02_sampler_prep/validation.json).
     Erster und Endlauf ok; dazwischen war nur die doppelte Fallbezeichnung im Text zu korrigieren.

**Weiter offen:**

- Kein vollständiger Originalfall; alle elf Nachweisbündel offen.
- Die echte Taxonmatrix hängt an Alters-Tiefen-Modell, Pollendaten und der NA-Maske aus
  `weightedMedian` (`matrixStats`-Fassung offen).
- Sampler-Schleife und Zufallsschritte sind nur dokumentiert.

## Nächster fachlicher Schritt (Phase 1)

Speicherortfrage nicht wiederholen, keine pauschale Inventur. Den nächsten begrenzten
deterministischen Schritt aus bereits gebundenen Quellen isolieren und vorher selbst belegen.

**Empfohlen: der deterministische Kern der Gewichtsstufe** `parameter_stage_weights`
(`mcmc_functions_v1.6.R:1276–1280`) mit **expliziten** Gewichten. Er ist der einzige aktive Leser
der jetzt vorbereiteten `taxa_information`. Zu belegen:

- `taxa_information$values * new_weights`: Recycling der Länge `ncol` über den zeilenweisen Vektor.
  Ein Gegenbeispiel mit spaltenweiser Lesart liegt im Paket.
- `matrix(..., ncol, nrow, byrow = T)`.
- `rowSums(na.rm = T)`: überspringt NA **und** NaN; Akkumulator wie bei `colMeans` markieren.
- Division der Matrix durch den Zeilensummenvektor (Recycling spaltenweise, also je Zeile).
- Zeilensumme 0 sowie Zellen mit NaN aus Nullspalten.
- `z <- 1:ncol(...)`.

Nur dokumentieren, nicht emulieren:

- `rdirichlet` (1248) und `ddirichlet` (1269–1270): Zufall bzw. Paketsemantik. Mehrere angehängte
  Pakete kommen als Anbieter in Frage; das ist zu belegen, nicht zu raten.
- Die Rücksetzung nach 1000 Versuchen (1253–1257).
- `if(is.na(parameter_ratio))` (1273).

Alternativen, falls die Gewichtsstufe nicht sauber abgrenzbar ist:

- Deterministische Sampler-Initialisierung `mcmc_conduction.R:109–135` ohne 118 (`prior.recent`,
  `fitdistr`) und ohne 133 (`rmvnorm_rcpp`): Startgewichte `rep(1/n, n)`, `z = 1:how_many_taxa`,
  `tann_fill_ids`/`pann_fill_ids` (Verschränkung), `checking_intervals` aus
  `sample_length`/`burn_in` (`main_v1.2.R:629–633`, `gibbs <- F`).
- `tf_x12_mean_current` (`mcmc_prepare.R:1692–1700`), reine Spaltenbindung. Die Leser sind belegt
  (`mcmc_conduction.R:146`, `mcmc_functions_v1.6.R:445/470/858`); die Eingaben hängen an
  `weightedMedian`.

`weightedMedian` erst nach belegter `matrixStats`-Fassung, keine Interpolationswahl.
Zufallsschritte bleiben nur dokumentiert, kein Ersatzseed: `set.seed(2022)`/`rgamma` 1297–1303,
`rbeta` 1709–1738, `sample` 1591, `rmvnorm_rcpp` 133, `rdirichlet`.

Bisherige Fixtures, Module, Tests und Pakete bytegleich lassen.

## Nächster App-Schritt

Nach einem neuen geprüften Teilmodul:

- Neuen Ordner `app_reproduction_<Datum>_<Kurzname>/` mit eigener `reproduction_state.json` anlegen.
- Eintrag **ans Ende** von `nachbaustand.STATES`; ältere Einträge nicht ändern.
- Dem bisher neuesten Stand 5 einen `commit` geben, der seine Datei enthält und gepusht ist:
  `git log -1 --format=%H -- docs/thesis_research/app_reproduction_2026_11_02_sampler_prep/reproduction_state.json`.
- Modul- und Testzahlen aus dem neuen Paket übernehmen.
- `tools/app_pruefen.js` anpassen: Modulanzahl, Testzahl, eingefrorene Zeilen und Links (jetzt 21),
  Linkcommit, Tabellenzahl (jetzt 9), Fehlstandskontrolle.
- `tests/test_nachbaustand.py` anpassen.
- Neue Tabellen auf 390 px prüfen. Zeilenköpfe sind `nowrap`; lange Erklärungen gehören in den
  Begleittext. Eindeutige Fallbezeichnungen über alle Tabellen eines Abschnitts.

Prüfen:

```bash
PYTHONPATH=src python -m pytest tests/test_nachbaustand.py tests/test_forschungsstand.py tests/test_app.py tests/test_artenvergleich.py -q
python -m pytest tests/test_historical_sampler_preparation.py tests/test_historical_tf_missing_priors.py tests/test_historical_gamma_prior_grid.py tests/test_historical_taxon_pruning.py tests/test_historical_tf_preparation.py tests/test_historical_phase1_chain.py tests/test_historical_cdf_discrepancy.py tests/test_historical_age_selection.py tests/test_historical_age_depth.py tests/test_historical_final_pdf.py tests/test_kinneret_phase1_spec.py -q
NODE_PATH=C:/Users/tnetz/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules node tools/app_pruefen.js results/app/index.html results/kinneret-app.html --kanal msedge
```

Vorher beide Builds erzeugen: `kinneret-recon app` und `kinneret-recon app --eine-datei`. Ohne
Installation: `PYTHONPATH=src python -c "import sys; from kinneret_recon.cli import main; sys.argv=['kinneret-recon','app']; main()"`.

Zwei Standorte:

1. Kopie über `tests/test_app.py::_zweiter_standort` anlegen, z. B.
   `PYTHONPATH="src;tests" python -c "from pathlib import Path; import test_app; test_app._zweiter_standort(Path('results/app'), Path('.local/<paket>/app2'))"`.
2. Dann `node tools/app_pruefen.js --zwei-standorte <Kopie>/index.html --kanal msedge` mit demselben
   `NODE_PATH`.

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
