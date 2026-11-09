# Claude-Übergabe: nach `ratios_tfs_no_ages` und App-Stand 14

Stand 09.11.2026. Gilt ebenso für ChatGPT/Codex. Vorrangiges Nutzerziel bleibt:
**die lokalen taxonbasierten Rekonstruktionen der Dissertation historisch nachvollziehbar nachbauen und in der App
umsetzen; Verbesserungen danach.**

- Deutsch, kurze Fortschrittsmeldungen, keine Subagenten.
- Nur den vom Nutzer benannten vorhandenen Checkout verwenden; keine Klone oder Worktrees.
- Bestehender Branch `dissertation-reconstruction`,
  [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77), kein automatischer Merge.
- Zuerst Git-/PR-Stand prüfen. Der Rechenschritt liegt in `9f6165e`; den App-Sicherungscommit liefert
  `git log -1 --format=%H -- docs/thesis_research/app_reproduction_2026_11_09_ratios_tfs/reproduction_state.json`.
- Commits nur mit `git -c user.name=TimonNetzel -c user.email=134919762+TimonNetzel@users.noreply.github.com`.
- Der Checkout-Pfad ist lang. Skripte vom Repo-Wurzelverzeichnis aus mit relativen Pfaden starten; `../../..` scheitert an der Windows-Pfadlänge.
- **`diss` nur lesen, nur absolute Pfade, dorthin kein `cd`**, auch nicht in Hilfsaufrufen.

## Geltende Nutzerentscheidungen

- **`wt_power`: NumPy-FFT** und **`d_truncnorm`: eigene Normalverteilung**, jeweils mit gemessener und ausdrücklich markierter ulp-Abweichung (09.11.2026). Nicht erneut fragen, beide nie als bitgleich zu R darstellen.
- Diese Entscheidungen gelten **nur für diese zwei Bausteine**. Weitere R-Funktionen (`dgamma`, `dbeta`, `dnorm` an anderer Stelle) nicht still mit erfassen.
- CRAN-Quellen lesen und R-Läufe für Gegenproben auf Kunstfixtures sind erlaubt. R 3.5.0 unter `.local/R/R-3.5.0`. Kein Compiler und kein Rcpp vorhanden.
- Weiterhin **keine** Sampler, Trainings, Altersmodelle oder Fossilrekonstruktionen. Kinneret ist Slot 3 des Zusatzorte-Zweigs (`add.sites <- T`).
- **Noch vom Nutzer zu entscheiden:** `tools/r_sort_gpl.py` (GPL im sonst MIT-lizenzierten Repository) und eine R-nahe, bitgleiche Rechenschicht für die sechs NumPy-Module.

## Was jetzt fertig ist

1. **`ratios_tfs_no_ages` und sein Leser** `mcmc_functions_v1.6.R:1485–1525`, `mcmc_conduction.R:228–230` ([Paket](literal_ratios_tfs_2026_11_09/README.md), `tools/historical_ratios_tfs.py`).
   - Es gilt die Definition 1485, nicht 1437. Der Aufruf läuft ohne Bedingung in jeder Iteration, Zeile 302 liest das Ergebnis.
   - R 3.5.0, 120 Stressfälle: Gitterquotienten und Produkt mit R-Faktoren bitgleich; Dichtequotienten 2.533 von 3.406 bitgleich, höchstens 15 ulp; `transfer_ratio` höchstens 35 ulp.
   - Die C++-Zeilen liefen nicht (kein Compiler). R führt die R-Fassung derselben Funktion wörtlich aus (`1529–1552`).
   - 212 neue Tests, 1.858 gezielte Tests grün in 22 Dateien.
2. **App-Stand 14** ([README](app_reproduction_2026_11_09_ratios_tfs/README.md)): 17 Teilmodule, Tabelle Q1–Q8. Stände 1–13 eingefroren. Geprüft in [validation.json](app_reproduction_2026_11_09_ratios_tfs/validation.json): App-Tests 172 bestanden und 2 übersprungen, historisch 1.858, Edge in zehn Fällen und zwei Standorte mit Exit 0, keine der 20 Tabellen bei 390 px breiter als ihr Rahmen.
3. **Zwei falsche Beschriftungen behoben** (doppelte Schlüssel in `i18n.js`), siehe README von Stand 14. Neue Sprachschlüssel eindeutig benennen; `test_every_language_key_is_defined_once` prüft das.

## Offen

- Das R-Paket `truncnorm` (`mcmc_prepare.R:1886`, `1957–1958`) ist nicht gebunden; der Zufallszustand beim Samplerstart ist nicht belegt.
- Installierte Fassungen von RcppDist und Rcpp im historischen Lauf sind unbekannt.
- Ob `wt_power` für Kinneret läuft, bleibt offen.
- Kein Originalfall; elf Nachweisbündel offen.

## Nächster Schritt (Phase 1): Vorschlag

**Empfohlen: der Eingabevertrag des Kinneret-Falls (Slot 3) als fachlicher Meilenstein.** Die deterministischen Ränder des Samplers sind weitgehend nachgebaut, und mehrere offene Bedingungen hängen an denselben fehlenden Eingaben (`sources$speleothem`, `sources$core`, `recent`, die Transferfunktionen). Der Vertrag hält fest, was gebunden ist und was fehlt; er braucht keine Ziehung und keine Rechnung.

Alternative, kleiner: das R-Paket `truncnorm` nach dem Muster der [rtruncnorm-Bindung](literal_rtruncnorm_binding_2026_11_09/README.md) binden, ohne Ziehung.

Weitere Rechenschritte wie die Quotienten in `mcmc_conduction.R:155–186` brauchen `dnorm`, `dgamma` und `dbeta` aus R und damit **eine eigene Nutzerentscheidung** wie bei `d_truncnorm`.

Bei einem neuen App-Stand: Eintrag am Ende von `nachbaustand.STATES`, Stand 14 bekommt seinen Linkcommit. In `tools/app_pruefen.js` die Fehlstandskontrolle (`[0, 17]`), 49 Links, 20 Tabellen, 13 frühere Stände anpassen.
Neue Fallbezeichnungen dürfen keine verwendete wiederholen: 1–4, G1–G3, D1–D4, M1–M9, K1–K8, C1–C8, R1–R8, W1–W8 und Q1–Q8 sind belegt.

## Prüfen

Wie in der [Übergabe nach wt_power](CLAUDE_NEXT_STEP_AFTER_WT_POWER_2026_11_09.md), die historische Liste zusätzlich mit `tests/test_rtruncnorm_binding.py` und `tests/test_historical_ratios_tfs.py` (jetzt 1.858 in 22 Dateien; App-Tests 172 bestanden, 2 übersprungen). Hilfsskripte für zwei Standorte und Tabellenbreiten liegen unversioniert unter `.local/ratios_tfs_2026_11_09`. Lange Browserläufe getrennt starten, Deckel offen lassen, bei einem Abbruch zuerst Kernel-Power 506/507 prüfen.

Am Ende: Belegpaket, Status und Arbeitsplan, **neue konkrete Claude-Übergabe mit Link im Abschluss**, Commit und Push auf PR #77, PR-Beschreibung fortschreiben, nicht mergen.
