# Claude-Übergabe: nach dem Eingabevertrag des Kinneret-Falls und App-Stand 15

Stand 09.11.2026. Gilt ebenso für ChatGPT/Codex. Vorrangiges Nutzerziel bleibt:
**die lokalen taxonbasierten Rekonstruktionen der Dissertation historisch nachvollziehbar nachbauen und in der App
umsetzen; Verbesserungen danach.**

- Deutsch, kurze Fortschrittsmeldungen, keine Subagenten.
- Nur den vom Nutzer benannten vorhandenen Checkout verwenden; keine Klone oder Worktrees.
- Bestehender Branch `dissertation-reconstruction`,
  [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77), kein automatischer Merge.
- Zuerst Git-/PR-Stand prüfen. Das Belegpaket liegt in `daaa2dc`; den App-Sicherungscommit liefert
  `git log -1 --format=%H -- docs/thesis_research/app_reproduction_2026_11_09_input_contract/reproduction_state.json`.
- Commits nur mit `git -c user.name=TimonNetzel -c user.email=134919762+TimonNetzel@users.noreply.github.com`.
- Der Checkout-Pfad ist lang. Skripte vom Repo-Wurzelverzeichnis aus mit relativen Pfaden starten; `../../..` scheitert an der Windows-Pfadlänge.
- **`diss` nur lesen, nur absolute Pfade, dorthin kein `cd`**, auch nicht in Hilfsaufrufen.

## Geltende Nutzerentscheidungen

- **`wt_power`: NumPy-FFT** und **`d_truncnorm`: eigene Normalverteilung**, jeweils mit gemessener und markierter Abweichung (09.11.2026). Nicht erneut fragen, beide nie als bitgleich zu R darstellen. Sie gelten **nur für diese zwei Bausteine**.
- CRAN-Quellen lesen und R-Läufe für Gegenproben auf Kunstfixtures oder belegten Teilschritten sind erlaubt. R 3.5.0 unter `.local/R/R-3.5.0`. Kein Compiler und kein Rcpp vorhanden.
- Weiterhin **keine** Sampler, Ziehungen, Trainings, Altersmodelle oder Fossilrekonstruktionen.
- **Noch vom Nutzer zu entscheiden:** `tools/r_sort_gpl.py` (GPL im sonst MIT-lizenzierten Repository), eine R-nahe bitgleiche Rechenschicht für die sechs NumPy-Module und `dnorm`/`dgamma`/`dbeta` in `mcmc_conduction.R:155–186`.

## Was jetzt fertig ist

1. **Eingabevertrag des Kinneret-Falls** ([Paket](kinneret_input_contract_2026_11_09/README.md), [JSON](kinneret_input_contract_2026_11_09/input_contract.json), `tests/test_kinneret_input_contract.py`): 48 Eingaben (5 gebunden, 10 teilweise gebunden, 6 Quellkonflikte, 27 fehlend) und 18 Bedingungen, je mit Datei, Zeile und Stand. Über 200 Zeilenbezüge sind gegen die gehashten Originale geprüft. Kein Bündel geschlossen.
2. **Neue Befunde daraus:**
   - Standortname: Der ältere Container nennt `Kinneret`, die Schalter prüfen `Lake Kinneret`. Der Wert bestimmt den Levante-Zweig `mcmc_prepare.R:1209`.
   - `inside.giesecke` ist mit den älteren Koordinaten FALSE, das Gitter ist vorhanden. `sources$core` wird im Zweig 636 gesetzt, der Varianztest 662 bleibt offen.
   - `climatezones` steckt nicht in `kgc_distribution_ww.rdata`; `d18o.age` hat in `local_reconstr_all_epd` keinen Erzeuger.
   - `mcmc_prepare.R:2057` ist ein aktives `stop()`, und der Speläothemzweig ersetzt `site.age` (825–826).
   - Zufallszustand: Letzter Reset `set.seed(2023)` in 1898, Startwerte davor, kein `RNGkind`. Offen.
3. **App-Stand 15** ([README](app_reproduction_2026_11_09_input_contract/README.md)): neue Karte mit Zählungen, Befunden, Schaltertabelle, Bedingungen und eingeklappt allen 48 Eingaben. 17 Teilmodule, 2.143 Tests, 23 Tabellen, 51 Links, 14 frühere Stände. Stände 1–14 eingefroren.

## Offen

- `wt_power` für Kinneret bleibt offen, weil `d18o.age` fehlt.
- Finaler Basiscontainer, Listenchronologie, numerische Transferfunktionen, globale CRU-Dateien, Isotopen-Workspaces: fehlen.
- Kein Originalfall; alle elf Nachweisbündel offen.

## Nächster Schritt (Phase 1): Vorschlag

Der Eingabevertrag zeigt, dass der Originalfall an fehlenden Eingaben hängt, nicht an weiterem Nachbau. Die verbleibenden lösbaren Punkte ohne neue Daten sind Paketbindungen ohne Ziehung nach dem Muster der [rtruncnorm-Bindung](literal_rtruncnorm_binding_2026_11_09/README.md):

1. **Empfohlen: `inside.range` aus `spatstat.utils`** (`main_v1.2.R:50`, `mcmc_prepare.R:716`): Die Randwertregel bei 6500 a BP ist seit dem Referenzvertrag offen und bestimmt die AP-Maske. Deterministisch und klein.
2. `rcpp_wt_bases_morlet` und `wt` aus `biwavelet` (`mcmc_prepare.R:901`, `934`): Anbieter und Export prüfen.
3. Das R-Paket `truncnorm` (`mcmc_prepare.R:1886`, `1957–1958`).

Größere Fortschritte verlangen neue Originaldaten oder eine ausdrückliche fachliche Entscheidung für einen anders gekennzeichneten methodischen Nachbau; beide sind nicht getroffen. Die Speicherortfrage nicht wiederholen. Die Entscheidungen aus „Noch vom Nutzer zu entscheiden“ nicht still treffen.

Bei einem neuen App-Stand: Eintrag am Ende von `nachbaustand.STATES`, Stand 15 bekommt seinen Linkcommit. In `tools/app_pruefen.js` die Fehlstandskontrolle (`[0, 17]`), 51 Links, 23 Tabellen, 14 frühere Stände anpassen.
Neue Fallbezeichnungen dürfen keine verwendete wiederholen: 1–4, G1–G3, D1–D4, M1–M9, K1–K8, C1–C8, R1–R8, W1–W8 und Q1–Q8 sind belegt.

## Prüfen

Wie in der [Übergabe nach wt_power](CLAUDE_NEXT_STEP_AFTER_WT_POWER_2026_11_09.md), die historische Liste zusätzlich mit `tests/test_rtruncnorm_binding.py`, `tests/test_historical_ratios_tfs.py` und `tests/test_kinneret_input_contract.py` (jetzt 2.143 in 23 Dateien; App-Tests 188 bestanden, 2 übersprungen). Hilfsskripte für zwei Standorte und Tabellenbreiten liegen unversioniert unter `.local/input_contract_2026_11_09`; `breiten_und_fotos.js` öffnet auch die eingeklappte Eingabeliste. Lange Browserläufe getrennt starten, Deckel offen lassen, bei einem Abbruch zuerst Kernel-Power 506/507 prüfen.

Am Ende: Belegpaket, Status und Arbeitsplan, **neue konkrete Claude-Übergabe mit Link im Abschluss**, Commit und Push auf PR #77, PR-Beschreibung fortschreiben, nicht mergen.
