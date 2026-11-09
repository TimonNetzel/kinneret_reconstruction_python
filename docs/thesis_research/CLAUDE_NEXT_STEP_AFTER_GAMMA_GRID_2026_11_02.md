# Claude-Übergabe: nach Gamma-Prior-Raster und App-Stand 3

Stand 02.11.2026. Gilt ebenso für ChatGPT/Codex. Vorrangiges Nutzerziel bleibt:
**die lokalen taxonbasierten Rekonstruktionen der Dissertation historisch
nachvollziehbar nachbauen und in der App umsetzen; Verbesserungen danach.**
Deutsch, kurze Fortschrittsmeldungen, keine Subagenten. Nur den vom Nutzer
benannten vorhandenen Checkout verwenden; keine Klone/Worktrees. Bestehender
Branch `dissertation-reconstruction`, [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
kein automatischer Merge. Zuerst Git-/PR-Stand und lokale Änderungen prüfen.
Ausgangspunkt dieses Pakets `94f81cc` (PR-Kopf, `main` eingemergt, GitHub meldete
`MERGEABLE`); Rechenschritt in `95bd0ef`, den App-Sicherungscommit liefert
`git log -1 -- docs/thesis_research/app_reproduction_2026_11_02_gamma_grid`.
Im Checkout ist keine Git-Identität gesetzt: für Commits nur
`git -c user.name=TimonNetzel -c user.email=134919762+TimonNetzel@users.noreply.github.com`.

## Was jetzt fertig ist

1. Lies `CLAUDE.md`, `AGENTS.md`, den neuesten [Projektstatus](PROJECT_STATUS.md),
   den [Arbeitsplan](LOCAL_DISS_RECONSTRUCTION_PLAN.md), das
   [Gamma-Raster-Paket](literal_gamma_prior_grid_2026_11_02/README.md) und den
   [App-Stand 3](app_reproduction_2026_11_02_gamma_grid/README.md).
2. **Determinismusprüfung** ([determinism_audit.json](literal_gamma_prior_grid_2026_11_02/determinism_audit.json)):
   Das Gamma-Prior-Raster (`mcmc_prepare.R:114–128`, `526–536`, `1264–1281`, `1733`)
   nutzt nur base R und läuft im gebundenen Hauptlauf **unbedingt**
   (Klammertiefe 0; 526–531 ist `else` von wörtlich `stad_interstad <- FALSE`, 468).
   Alle vier aktiven Blöcke sind bis auf Variablennamen gleich (programmatisch geprüft).
   `which.tf.missing` (1664–1679) ist mit expliziter Maske deterministisch, aber nicht
   gewählt, weil die Maske aus `weightedMedian` stammt. `set.seed(2022)` + `rnorm`/`rgamma`
   (1297–1303) und `rbeta` (1734) sind zufallsabhängig → nur dokumentiert.
3. **Modul** `tools/historical_gamma_prior_grid.py`: `seq.default` wörtlich
   (`+1e-10`-Fuzz, `from + (0:n)*by` als zwei Rundungen, `pmin`), erstes `which.min`,
   `rate` aus dem Wurzelzweig. Obergrenze nur 50000 (`mcmc_prepare.R`) oder 5000
   (Kopie `mcmc_prep_v1.2.R:1103`, nicht vom Hauptlauf geladen). `recent` und
   `min(age) >= 2000` sind Boolwerte; `age` wird nicht ausgewertet. Kette meldet
   Periodenkopien (127–128), `diff_rate = shape/10` (536, unverdoppelt), Verdopplung
   1264–1267, zweites Raster und die durch 1733 ersetzte `rate = shape/mean`.
   Für 50000 sind Fuzz und `pmin` wirkungslos (versionsrobust), für 5000 nicht.
   55 neue Tests gegen unabhängige Float-/Bruchreferenz und Handerwartungen,
   **bitgleich**; mit den bisherigen **352 grün**.
4. **App:** neuer datierter Nachbaustand
   `app_reproduction_2026_11_02_gamma_grid/reproduction_state.json` (Pin `43edfcb6…`)
   als **letzter** Eintrag in `nachbaustand.STATES`; Stand 1 (`c358b2c3…`) und
   Stand 2 (`3656316d…`) bytegleich, Stand 2 erhielt nur seinen Linkcommit `3d013b2`.
   Sieben Module (Gamma-Raster ohne „Bedingter Schritt“), Gamma-Kunsttabelle mit vier
   nummerierten Fällen im eingeklappten synthetischen Beispiel. Beide Offlineformate
   in Edge geprüft (Handy/PC, Hell/Dunkel, DE/EN, Gesten, zwei Standorte):
   [validation.json](app_reproduction_2026_11_02_gamma_grid/validation.json).

**Weiter offen:** kein vollständiger Originalfall. Alle elf Nachweisbündel offen.
`prior.recent` selbst (`fitdistr` in `prior_climate.R:128–130`, Untergrenzen
`sd.tmp ≥ 2`, `sd.pann ≥ 150`) ist ungebunden; die App zeigt nur Kunstwerte.
`weightedMedian` (matrixStats-Fassung) und Zufallsschritte bleiben offen.

## Nächster fachlicher Schritt (Phase 1)

Speicherortfrage nicht wiederholen, keine pauschale Inventur. Nächsten begrenzten
deterministischen Schritt aus bereits gebundenen Quellen isolieren; vorher selbst belegen:

- **Empfohlen: `which.tf.missing` und die `num_taxa`-Priors** (`mcmc_prepare.R:1664–1685`)
  mit **explizit übergebener** NA-Maske (keine Ableitung aus `weightedMedian`):
  negative Indizierung `[-idx]`/`[,-idx]` mit `drop` (eine Restspalte → Vektor:
  ablehnen statt emulieren), Zweig `taxa_change` als Boolwert,
  `num_taxa <- nrow(tf.x1.current)`, dann `taxa.weights = 1/num_taxa`,
  `phi = num_taxa/2`, `nr.taxa.mean = num_taxa`, `nr.taxa.sd = round(num_taxa/4)`.
  Zu belegen und zu testen: R-`round` bei genau ,5 (z. B. `num_taxa = 2, 6, 10`),
  Verhalten bei leerer Maske (`length(...) != 0`-Schutz), Reihenfolge der Spalten.
- Schwächer: Schwellen 131–137 (Mittel ± Streuung) – `tann_thresh`/`pann_thresh`
  stammen aus geseedeten Quantilen (1299/1303), Periodenschwellen teils aus Quantilen.
- `weightedMedian` erst nach belegter `matrixStats`-Fassung; keine Interpolationswahl.
  Zufallsschritte weiter nur dokumentieren, kein Ersatzseed.

Bisherige Fixtures, Module, Tests und Pakete bytegleich lassen.

## Nächster App-Schritt

Nach einem neuen geprüften Teilmodul: neuen Ordner
`app_reproduction_<Datum>_<Kurzname>/` mit eigener `reproduction_state.json`,
Eintrag **ans Ende** von `nachbaustand.STATES` (ältere Einträge nicht ändern; dem
bisher neuesten Stand 3 einen `commit` geben, der seine Datei enthält und gepusht ist –
`git log -1 --format=%H -- docs/thesis_research/app_reproduction_2026_11_02_gamma_grid/reproduction_state.json`).
Modul-/Testzahlen aus dem neuen Paket; `tools/app_pruefen.js` (Modulanzahl, Testzahl,
eingefrorene Zeilen/Links, Linkcommit, Tabellenzahl, Fehlstandskontrolle) und
`tests/test_nachbaustand.py` anpassen. Prüfen:

```bash
PYTHONPATH=src python -m pytest tests/test_nachbaustand.py tests/test_forschungsstand.py tests/test_app.py tests/test_artenvergleich.py -q
python -m pytest tests/test_historical_gamma_prior_grid.py tests/test_historical_taxon_pruning.py tests/test_historical_tf_preparation.py tests/test_historical_phase1_chain.py tests/test_historical_cdf_discrepancy.py tests/test_historical_age_selection.py tests/test_historical_age_depth.py tests/test_historical_final_pdf.py tests/test_kinneret_phase1_spec.py -q
NODE_PATH=C:/Users/tnetz/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules node tools/app_pruefen.js results/app/index.html results/kinneret-app.html --kanal msedge
```

Builds vorher mit `kinneret-recon app` und `kinneret-recon app --eine-datei`
(ohne Installation: `PYTHONPATH=src python -c "import sys; from kinneret_recon.cli import main; sys.argv=['kinneret-recon','app']; main()"`).
Zwei Standorte: Kopie über `tests/test_app.py::_zweiter_standort` anlegen, dann
`node tools/app_pruefen.js --zwei-standorte <Kopie>/index.html --kanal msedge`.
Die Gesamtsuite `pytest tests` nicht ausführen (Sampler-Unittests);
`literal_tf_preparation_2026_11_02/verify_packet.py` nicht als Regressionsprüfer.

## Arbeitsgrenzen

Originale nur lesen; `diss` und private Quellpfade nie ändern (Shell-Arbeitsordner
nicht in `diss` belassen, nur absolute Pfade). Keine stillen Ersatzdaten,
Quellenkorrekturen oder Variantenentscheidungen (kein `inside.range`, Median, Seed
oder Rasterobergrenze ohne Beleg). Keine Trainings, Sampler, Altersmodelle, R-Läufe
oder Fossilrekonstruktionen. Baseline, CLI, sieben experimentelle Artenmodelle,
Paperreferenz und fallbezogene Monticchio-/Chronologie-Gates erhalten. Plots jung
links, alt rechts. Diese und frühere Übergaben eingefroren lassen. Am Ende neues
Belegpaket, aktualisierter Projektstatus/Arbeitsplan und **eine neue konkrete
Claude-Übergabe mit Link im Abschluss**; geprüftes Paket auf dem bestehenden Branch
committen/pushen, PR #77 fortschreiben, nicht mergen.
