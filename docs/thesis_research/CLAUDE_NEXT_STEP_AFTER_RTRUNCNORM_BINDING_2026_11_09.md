# Claude-Übergabe: nach der Quellen- und Bindungsprüfung von `rtruncnorm_cpp`

Stand 09.11.2026. Gilt ebenso für ChatGPT/Codex. Vorrangiges Nutzerziel bleibt:
**die lokalen taxonbasierten Rekonstruktionen der Dissertation historisch nachvollziehbar nachbauen und in der App
umsetzen; Verbesserungen danach.**

- Deutsch, kurze Fortschrittsmeldungen, keine Subagenten.
- Nur den vom Nutzer benannten vorhandenen Checkout verwenden; keine Klone oder Worktrees.
- Bestehender Branch `dissertation-reconstruction`,
  [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77), kein automatischer Merge.
- Zuerst Git-/PR-Stand prüfen. Commits nur mit `git -c user.name=TimonNetzel -c user.email=134919762+TimonNetzel@users.noreply.github.com`.
- Der Checkout-Pfad ist lang. Skripte vom Repo-Wurzelverzeichnis aus mit relativen Pfaden starten; `../../..` scheitert an der Windows-Pfadlänge.
- **`diss` nur lesen, nur absolute Pfade, dorthin kein `cd`**, auch nicht in Hilfsaufrufen.

Alle weiteren Nutzerentscheidungen, Prüfbefehle und Arbeitsgrenzen gelten wie in der [Übergabe nach wt_power](CLAUDE_NEXT_STEP_AFTER_WT_POWER_2026_11_09.md). Dort stehen auch die Sollzahlen für einen neuen App-Stand; App-Stand 13 ist unverändert der aktuelle.

## Was jetzt fertig ist

[Belegpaket](literal_rtruncnorm_binding_2026_11_09/README.md), keine Ziehung, keine Kompilierung, kein R-Lauf, kein neuer App-Stand.

- `rtruncnorm_cpp` hat eine Definition (`mcmc_functions_v1.6.R:1563`) und zwei Aufrufer (1812, 1813).
- `r_truncnorm` kommt aus RcppDist. Im Fenster 2018-04-23…2023-07-07 gab es auf CRAN genau **0.1.1** (2018-10-28); davor keine Fassung. Die installierte Fassung des historischen Laufs ist unbekannt.
- Rechenweg: Inversion mit einem `R::runif(0, 1)` je Element. Aus den Quellen gelesen, nicht gemessen: `num_taxa` Zufallszahlen je Aufruf, `2 · num_taxa` je Iteration.
- 32 neue Tests, **1.646** gezielte Tests grün in 21 Dateien.

## Offen aus diesem Schritt

- **Zweiter Anbieter:** `mcmc_prepare.R:1886`, `1957–1958` zieht mit dem R-Paket `truncnorm`, nicht gebunden.
- **Zustand beim Samplerstart nicht belegt.** Die Liste der `set.seed`-Stellen im Paket ist keine vollständige Verfolgung.
- Die numerischen Eingaben (`tf.x1.current` usw.) fehlen weiter. Elf Nachweisbündel offen.

## Nächster Schritt (Phase 1): Vorschlag

**Empfohlen: `ratios_tfs_no_ages` (`mcmc_functions_v1.6.R:1485–1525`, Aufruf `mcmc_conduction.R:228`) mit expliziten Vektoren und Matrizen.** Der Schritt ist deterministisch, wird in jeder Iteration erreicht und schließt an die Bindung an.

- Es gilt die Definition 1485, nicht 1437: Nur ihre Argumentnamen passen zum Aufruf.
- `tann_tf_ratio` und `pann_tf_ratio` (1507–1513) brauchen nur `which_min(abs(...))` und eine Division.
- `tann_dists_tf_ratio` und `pann_dists_tf_ratio` (1516–1522) brauchen `d_truncnorm` (RcppDist 0.1.1, Header 142–152) und damit `R::pnorm` und `R::dnorm`.

**Nutzerentscheidung nötig**, bevor der `d_truncnorm`-Teil gebaut wird: `pnorm` und `dnorm` bitgleich zu R hieße, GPL-Code aus R (`pnorm.c`, `dnorm.c`) zu übersetzen. Das ist dieselbe offene Lizenzfrage wie bei `tools/r_sort_gpl.py`. Die FFT-Entscheidung vom 09.11.2026 galt nur für `wt_power` und überträgt sich nicht von selbst. Alternative: eigene Rechnung mit gemessener und ausdrücklich markierter ulp-Abweichung gegen R 3.5.0.

Ohne diese Entscheidung bleiben möglich: nur der Teil 1507–1513, oder der fachliche Meilenstein **Eingabevertrag des Kinneret-Falls (Slot 3)**. Bei einem neuen App-Stand gelten die Sollzahlen aus der Übergabe nach wt_power.

## Prüfen

Wie in der [Übergabe nach wt_power](CLAUDE_NEXT_STEP_AFTER_WT_POWER_2026_11_09.md), die historische Liste zusätzlich mit `tests/test_rtruncnorm_binding.py` (jetzt 1.646). Der Cache der CRAN-Fassungen liegt unversioniert unter `.local/rtruncnorm_2026_11_09/cran/RcppDist`.

Am Ende: Belegpaket, Status und Arbeitsplan, **neue konkrete Claude-Übergabe mit Link im Abschluss**, Commit und Push auf PR #77, PR-Beschreibung fortschreiben, nicht mergen.
