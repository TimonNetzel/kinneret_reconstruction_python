# Claude-Übergabe: nach fehlenden TF, `num_taxa`-Priors und App-Stand 4

Stand 02.11.2026. Gilt ebenso für ChatGPT/Codex. Vorrangiges Nutzerziel bleibt:
**die lokalen taxonbasierten Rekonstruktionen der Dissertation historisch
nachvollziehbar nachbauen und in der App umsetzen; Verbesserungen danach.**
Deutsch, kurze Fortschrittsmeldungen, keine Subagenten. Nur den vom Nutzer
benannten vorhandenen Checkout verwenden; keine Klone/Worktrees. Bestehender
Branch `dissertation-reconstruction`, [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
kein automatischer Merge. Zuerst Git-/PR-Stand und lokale Änderungen prüfen.
Ausgangspunkt dieses Pakets `7d3d0f6` (PR-Kopf, App-Stand 3, GitHub meldete
`MERGEABLE`); Rechenschritt in `9bb8ed9`, den App-Sicherungscommit liefert
`git log -1 -- docs/thesis_research/app_reproduction_2026_11_02_tf_missing`.
Im Checkout ist keine Git-Identität gesetzt: für Commits nur
`git -c user.name=TimonNetzel -c user.email=134919762+TimonNetzel@users.noreply.github.com`.

## Was jetzt fertig ist

1. Lies `CLAUDE.md`, `AGENTS.md`, den neuesten [Projektstatus](PROJECT_STATUS.md),
   den [Arbeitsplan](LOCAL_DISS_RECONSTRUCTION_PLAN.md), das
   [TF-Entfernungspaket](literal_tf_missing_priors_2026_11_02/README.md) und den
   [App-Stand 4](app_reproduction_2026_11_02_tf_missing/README.md).
2. **Determinismusprüfung** ([determinism_audit.json](literal_tf_missing_priors_2026_11_02/determinism_audit.json)):
   `which.tf.missing` samt Priors (`mcmc_prepare.R:1664–1685`, dazu 1435–1442) ist base R
   ohne Zufall und im gebundenen Hauptlauf aktiv. [bind_source.py](literal_tf_missing_priors_2026_11_02/bind_source.py)
   prüft Hashes, Klammertiefen (Kommentare/Zeichenketten entfernt, auch mehrzeiliges Rcpp),
   alle Zuweisungsstellen (`num_taxa` 1435/1678, `taxa_change` 1462/1501, `all.avail` bis 1676
   nicht verkleinert) und dass `prior$phi*`, `taxa.weights`, `nr.taxa.*` in den fünf
   Laufdateien keinen aktiven Leser haben. Schwellen 131–137 nicht gewählt (Periodenzweige
   mit `fitdistr`/`quantile`, Leser `inside.range` ungebunden).
3. **Modul** `tools/historical_tf_missing_priors.py`: NA-Maske **explizit** (nur x1-Mittel),
   Längenschutz (leere Maske → keine Indizierung, `num_taxa` aus 1435), Reihenfolge der
   Resttaxa in allen Objekten, `taxa_change` als Boolwert (nur `_change`-Mittel gekürzt,
   `_change`-TFs nicht; im Lauf tot), `num_taxa` aus `nrow` (1678), Priors 1681–1685 mit
   `round(n/4)` zur geraden Ziffer (dokumentierte R-Regel, nicht ausgeführt). Gemeldet:
   veraltete `phi1`/`phi2` aus 1439–1440, `posterior$nr.taxa.mean` durch 1706 verworfen.
   Abgelehnt: ein oder kein Resttaxon, `drop` durch Ausdehnung 1, ungleiche Taxonzahlen.
   85 neue Tests gegen unabhängige Listen-/Ganzzahl-/Bruchreferenz und Handerwartungen,
   **exakt**; mit den bisherigen **437 grün**.
4. **App:** neuer datierter Nachbaustand
   `app_reproduction_2026_11_02_tf_missing/reproduction_state.json` (Pin `7b7bf1f4…`)
   als **letzter** Eintrag in `nachbaustand.STATES`; Stände 1 (`c358b2c3…`), 2 (`3656316d…`)
   und 3 (`43edfcb6…`) bytegleich, Stand 3 erhielt nur seinen Linkcommit `7d3d0f6`.
   Acht Module (neues Modul ohne „Bedingter Schritt“), neue Kunsttabelle mit vier
   nummerierten Fällen und Satz zur leeren Maske im eingeklappten synthetischen Beispiel.
   Beide Offlineformate in Edge geprüft (Handy/PC, Hell/Dunkel, DE/EN, Gesten, zwei
   Standorte): [validation.json](app_reproduction_2026_11_02_tf_missing/validation.json).
   Erstlauf: neue Tabelle auf Handybreite zu breit → Zeilenköpfe gekürzt, dann alles ok.

**Weiter offen:** kein vollständiger Originalfall. Alle elf Nachweisbündel offen.
Die echte NA-Maske hängt an `weightedMedian` (matrixStats-Fassung) und den historischen
TFs; `prior.recent` (`fitdistr`) ungebunden; Zufallsschritte nur dokumentiert.

## Nächster fachlicher Schritt (Phase 1)

Speicherortfrage nicht wiederholen, keine pauschale Inventur. Nächsten begrenzten
deterministischen Schritt aus bereits gebundenen Quellen isolieren; vorher selbst belegen:

- **Empfohlen: Sampler-Vorbereitung aus `num_taxa` und `taxa.appear.core`**
  (`mcmc_conduction.R:53–79`), direkt im Anschluss an dieses Paket. Nur die Vorbereitung,
  **kein Samplerlauf**. Zu belegen: `gibbs <- F` (`main_v1.2.R:624`, Klammertiefe, keine
  weitere Zuweisung) → `how_many_taxa = num_taxa` statt `floor(num_taxa/2)`;
  `taxa_information$values = c(t(taxa.appear.core))` (zeilenweise Abflachung),
  `ncol`/`nrow`, `colMeans(na.rm = T)` (Verhalten bei NA, Ganz-NA-Spalte → `NaN`),
  spaltenweise Division durch `max(..., na.rm = T)` (Ganz-NA-Spalte → `-Inf` mit Warnung:
  ablehnen statt emulieren; Nullspalte → `NaN`), Überschreiben von `values` in 75,
  `jeffreys_prior = rep(0.5, num_taxa)`. Leser von `taxa_information` (ggf. eingebettetes
  Rcpp in `mcmc_functions_v1.6.R`) vorher prüfen und nur dokumentieren.
- `combs` (`mcmc_prepare.R:1702–1703`, `choose`) hat keinen Leser → nur dokumentieren.
  `tf_x12_mean_current` (1692–1700) ist reine Spaltenbindung; Leser vorher prüfen.
- `weightedMedian` erst nach belegter `matrixStats`-Fassung; keine Interpolationswahl.
  Zufallsschritte (`set.seed(2022)`/`rgamma` 1297–1303, `rbeta` 1709–1738, `sample` 1591,
  `rmvnorm_rcpp` in `mcmc_conduction.R:133`) weiter nur dokumentieren, kein Ersatzseed.

Bisherige Fixtures, Module, Tests und Pakete bytegleich lassen.

## Nächster App-Schritt

Nach einem neuen geprüften Teilmodul: neuen Ordner
`app_reproduction_<Datum>_<Kurzname>/` mit eigener `reproduction_state.json`,
Eintrag **ans Ende** von `nachbaustand.STATES` (ältere Einträge nicht ändern; dem
bisher neuesten Stand 4 einen `commit` geben, der seine Datei enthält und gepusht ist –
`git log -1 --format=%H -- docs/thesis_research/app_reproduction_2026_11_02_tf_missing/reproduction_state.json`).
Modul-/Testzahlen aus dem neuen Paket; `tools/app_pruefen.js` (Modulanzahl, Testzahl,
eingefrorene Zeilen/Links, Linkcommit, Tabellenzahl, Fehlstandskontrolle) und
`tests/test_nachbaustand.py` anpassen. Neue Tabellen auf 390 px prüfen (Zeilenköpfe sind
`nowrap`; lange Erklärungen in den Begleittext). Prüfen:

```bash
PYTHONPATH=src python -m pytest tests/test_nachbaustand.py tests/test_forschungsstand.py tests/test_app.py tests/test_artenvergleich.py -q
python -m pytest tests/test_historical_tf_missing_priors.py tests/test_historical_gamma_prior_grid.py tests/test_historical_taxon_pruning.py tests/test_historical_tf_preparation.py tests/test_historical_phase1_chain.py tests/test_historical_cdf_discrepancy.py tests/test_historical_age_selection.py tests/test_historical_age_depth.py tests/test_historical_final_pdf.py tests/test_kinneret_phase1_spec.py -q
NODE_PATH=C:/Users/tnetz/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules node tools/app_pruefen.js results/app/index.html results/kinneret-app.html --kanal msedge
```

Builds vorher mit `kinneret-recon app` und `kinneret-recon app --eine-datei`
(ohne Installation: `PYTHONPATH=src python -c "import sys; from kinneret_recon.cli import main; sys.argv=['kinneret-recon','app']; main()"`).
Zwei Standorte: Kopie über `tests/test_app.py::_zweiter_standort` anlegen
(z. B. `PYTHONPATH="src;tests" python -c "from pathlib import Path; import test_app; test_app._zweiter_standort(Path('results/app'), Path('.local/<paket>/app2'))"`),
dann `node tools/app_pruefen.js --zwei-standorte <Kopie>/index.html --kanal msedge`
(mit demselben `NODE_PATH`). Die Gesamtsuite `pytest tests` nicht ausführen
(Sampler-Unittests); `literal_tf_preparation_2026_11_02/verify_packet.py` und die
zeitpunktgebundenen `tools/verify_*` nicht als Regressionsprüfer verwenden.

## Arbeitsgrenzen

Originale nur lesen; `diss` und private Quellpfade nie ändern (Shell-Arbeitsordner
nicht in `diss` belassen, nur absolute Pfade). Keine stillen Ersatzdaten,
Quellenkorrekturen oder Variantenentscheidungen (kein `inside.range`, Median, Seed
oder Rasterobergrenze ohne Beleg). Keine Trainings, Sampler, Altersmodelle, R-Läufe
oder Fossilrekonstruktionen. Baseline, CLI, sieben experimentelle Artenmodelle,
Paperreferenz und fallbezogene Monticchio-/Chronologie-Gates erhalten. Plots jung
links, alt rechts. Testzahlen sind keine Fallreproduktion; Kunstbeispiele nie als
historische Rechnung darstellen. Diese und frühere Übergaben eingefroren lassen. Am Ende
neues Belegpaket, aktualisierter Projektstatus/Arbeitsplan und **eine neue konkrete
Claude-Übergabe mit Link im Abschluss**; geprüftes Paket auf dem bestehenden Branch
committen/pushen, PR #77 fortschreiben, nicht mergen.
