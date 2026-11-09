# Netzel-Projekt: kurzer Arbeitsstand

## `ratios_tfs_no_ages` nachgebildet (eigene Normalverteilung, nicht bitgleich zu R), App-Stand 14, 09.11.2026

**Abgeschlossen** (Nutzerauftrag): [Modul und Belege](literal_ratios_tfs_2026_11_09/README.md) (`9f6165e`), [App-Stand 14](app_reproduction_2026_11_09_ratios_tfs/README.md).

- `mcmc_functions_v1.6.R:1485–1525` und Leser `mcmc_conduction.R:228–230` mit expliziten Vektoren und Matrizen. Keine Ziehung, kein Sampler, kein Originalfall. Der Aufruf läuft ohne Bedingung in jeder Iteration.
- **Nutzerentscheidung vom 09.11.2026:** `d_truncnorm` mit eigener Normalverteilung und gemessener, markierter Abweichung, keine Übersetzung von R-`dnorm.c` und `pnorm.c`. Der Baustein ist **nicht** bitgleich zu R.
- R 3.5.0, 120 Stressfälle: Gitterquotienten (3.406) und Produkt mit R-Faktoren (120) bitgleich; Dichtequotienten 2.533 von 3.406 bitgleich, höchstens 15 ulp; `transfer_ratio` höchstens 35 ulp. Die C++-Zeilen liefen nicht (kein Compiler); R führt die R-Fassung derselben Funktion wörtlich aus.
- 212 neue Tests, **1.858** gezielte Tests grün in 22 Dateien; App-Tests 172 grün (2 Wrapper explizit ersetzt); Edge in beiden Formaten (zehn Fälle) und zwei Standorte mit Exit 0 und ohne `FEHLER`-Zeile; keine der 20 Tabellen bei 390 px breiter als ihr Rahmen; kein Standby während der Läufe. Phase-1-Prüfer Exit 2.

**App:** [Stand 14](app_reproduction_2026_11_09_ratios_tfs/README.md) (Pin `a45d172c…`): 17 Teilmodule, Tabelle Q1–Q8. Stände 1–13 eingefroren. Zwei falsche Beschriftungen aus doppelten Sprachschlüsseln behoben (G1–G3 der Gewichtstabelle, „NA nur x2“), keine Zahl betroffen.

**Offen:** R-Paket `truncnorm` nicht gebunden; Zufallszustand beim Samplerstart nicht belegt; installierte Fassungen von RcppDist und Rcpp unbekannt. Kein Originalfall; elf Nachweisbündel offen, keine Variante gewählt. PR #77 offen, nicht gemergt.

**Beim Nutzer:** Lizenz von `tools/r_sort_gpl.py`, R-nahe bitgleiche Rechenschicht für die sechs NumPy-Module.

**Als Nächstes:** [Claude-Übergabe](CLAUDE_NEXT_STEP_AFTER_RATIOS_TFS_2026_11_09.md); Vorschlag: Eingabevertrag des Kinneret-Falls (Slot 3).

## `rtruncnorm_cpp`: Quellen- und Bindungsprüfung ohne Ziehung, 09.11.2026

**Abgeschlossen** (Nutzerauftrag „Mache weiter“): [Belegpaket](literal_rtruncnorm_binding_2026_11_09/README.md). Kein Rechenmodul, keine Ziehung, keine Kompilierung, kein neuer App-Stand; App-Stand 13 bleibt aktuell.

- Eine Definition (`mcmc_functions_v1.6.R:1563`), zwei Aufrufer (1812, 1813). `r_truncnorm` kommt aus RcppDist. Im Fenster 2018-04-23…2023-07-07 gab es auf CRAN genau 0.1.1 (2018-10-28), davor keine Fassung. Die installierte Fassung des historischen Laufs ist unbekannt.
- Inversion mit einem `R::runif(0, 1)` je Element, aus den Quellen gelesen und nicht gemessen: `2 · num_taxa` Zufallszahlen je Iteration.
- 32 neue Tests, **1.646** gezielte Tests grün in 21 Dateien.

**Offen:** Zweiter Anbieter R-Paket `truncnorm` (`mcmc_prepare.R:1886`, `1957–1958`) nicht gebunden; Zufallszustand beim Samplerstart nicht belegt. Kein Originalfall; elf Nachweisbündel offen. PR #77 offen, nicht gemergt.

**Blockiert (Nutzerentscheidung):** `d_truncnorm` braucht `R::pnorm` und `R::dnorm`; bitgleich hieße GPL-Code aus R zu übersetzen.

**Als Nächstes:** [Claude-Übergabe](CLAUDE_NEXT_STEP_AFTER_RTRUNCNORM_BINDING_2026_11_09.md); Vorschlag: `ratios_tfs_no_ages` (1485–1525) mit expliziten Eingaben.

## `wt_power` nachgebildet (NumPy-FFT, nicht bitgleich zu R), App-Stand 13, 09.11.2026

**Abgeschlossen** (Nutzerauftrag): [Modul und Belege](literal_wt_power_2026_11_09/README.md) (`ddbce9e`), [App-Stand 13](app_reproduction_2026_11_09_wt_power/README.md).

- `mcmc_functions_v1.6.R:1003–1012` und Leser `2023–2026` mit expliziten Reihen und explizitem `wb`. Keine Ziehung, kein Sampler, kein Originalfall.
- **Nutzerentscheidung vom 09.11.2026:** NumPy-FFT mit gemessener und ausdrücklich markierter ulp-Abweichung, keine Übersetzung von R-`fft.c`. `wt_power` ist **nicht** bitgleich zu R.
- R 3.5.0, 120 Stressfälle: 22.256 von 22.256 zentrierten Werten bitgleich; 5.938 von 39.876 Leistungen bitgleich, höchstens 606 ulp, relativ zum Spaltenmaximum höchstens 2,5e-15; `post_spec` höchstens 12 ulp; keine Token-Abweichung.
- 212 neue Tests, **1.614** gezielte Tests grün in 20 Dateien; App-Tests 151 grün (2 Wrapper explizit ersetzt); Edge in beiden Formaten (zehn Fälle) und zwei Standorte mit Exit 0 und ohne `FEHLER`-Zeile; keine der 19 Tabellen bei 390 px breiter als ihr Rahmen; kein Standby während der Läufe. Phase-1-Prüfer Exit 2.

**App:** [Stand 13](app_reproduction_2026_11_09_wt_power/README.md) (Pin `d84fe644…`): 16 Teilmodule, `wt_power` bedingt, Tabelle W1–W8. Stände 1–12 eingefroren. Validierung abgeschlossen ([validation.json](app_reproduction_2026_11_09_wt_power/validation.json)), in [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77) gesichert.

**Offen:** Ob `wt_power` für Kinneret läuft: Es läuft bei `sources$speleothem == 1` auch in `mcmc_prepare.R:947/966`, und Zeile 1145 steht hinter `case_studies`, das F ist. Kein Originalfall; elf Nachweisbündel offen, keine Variante gewählt. PR #77 offen, nicht gemergt.

**Beim Nutzer:** Lizenz von `tools/r_sort_gpl.py`, R-nahe bitgleiche Rechenschicht für die sechs NumPy-Module.

**Als Nächstes:** [Claude-Übergabe](CLAUDE_NEXT_STEP_AFTER_WT_POWER_2026_11_09.md); Vorschlag: `rtruncnorm_cpp` nur als Quellen- und Bindungsprüfung ohne Ziehung, danach der Eingabevertrag des Kinneret-Falls (Slot 3).

## `reconst` und die Prädiktanten nachgebildet, App-Stand 12, 09.11.2026

**Abgeschlossen** (Nutzerauftrag): [Modul und Belege](literal_reconst_predictant_2026_11_09/README.md).

- `mcmc_functions_v1.6.R:1817–1820` aus vorgegebenem `tf_sample` samt den direkten Lesern `1825`, `1829–1857`, `1939–1943`. Keine Ziehung; `rtruncnorm_cpp` (1812–1813) bleibt draußen.
- **Abweichungen von der Übergabe, belegt:** `%*%` rechnet im 3-Schleifen-Pfad von R 3.5.0 in `double`, nicht in `long double` (`array.c:702`, nur `internal_matprod` nutzt `long double`).
  `reconst_ages` ist in jedem Lauf „alle Zeilen“, weil Zeile 1522 die Holozänmaske aus 1466 überschreibt. Das Argument `tf_sample` wird nie gelesen.
- **Kinneret:** Zeile 1818, 1825 und 1942 laufen; 1831–1837, 1867–1902 und 1923–1929 nicht. Zeile 1850 hängt an `recent` (Alter ≤ 11000), ohne finale Alter offen.
- R 3.5.0 mit wörtlichen Originalzeilen: 3.009 von 3.009 Werten bitgleich, 0 ulp. 230 neue Tests, **1.402** gezielte Tests grün; App-Tests 134 grün
  (2 Wrapper explizit ersetzt); Edge in beiden Formaten und zwei Standorte mit Exit 0 und ohne `FEHLER`-Zeile, siehe [App-Stand 12](app_reproduction_2026_11_09_reconst_predictant/README.md). Phase-1-Prüfer Exit 2.

**App:** [Stand 12](app_reproduction_2026_11_09_reconst_predictant/README.md) (Pin `ce676ea4…`): 15 Teilmodule, Tabelle R1–R8. Stände 1–11 eingefroren. Validierung abgeschlossen, in [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77) gesichert. Das einmalige Klick-Timeout eines Browserlaufs fiel mit einem Standby durch Zuklappen des Deckels zusammen und trat ohne Standby nicht wieder auf; App und Prüfskript blieben deswegen unverändert.

**Begonnen, nicht gebaut:** `wt_power`. Erreichbarkeit und Abhängigkeiten sind in der Übergabe festgehalten.

**Blockiert (Nutzerentscheidung):** Bitgleiches `fft`/`mvfft` hieße, GPL-Code aus R zu übersetzen; das hängt an derselben offenen Lizenzfrage wie `tools/r_sort_gpl.py`.

**Offen:** Kein Originalfall; elf Nachweisbündel offen, keine Variante gewählt. `tf_sample` stammt im Lauf aus Ziehungen. PR #77 offen, nicht gemergt.

**Als Nächstes:** [Claude-Übergabe](CLAUDE_NEXT_STEP_AFTER_RECONST_2026_11_09.md).

## Lokales `cor` nachgebildet, App-Stand 11, 09.11.2026

**Abgeschlossen** (Nutzerauftrag): [Modul und Belege](literal_local_cor_2026_11_09/README.md).

- `mcmc_functions_v1.6.R:2355–2366` und die Zeitraum-Leser `1939–2018` mit expliziten Vektoren; eine Definition in 540 R-Dateien, nur Basisfunktionen.
- **Kinneret:** `multiple` ist genau `sources$core`. Es läuft 1964–1970, sofern die Varianz der AP-Referenz nicht 0 ist (ohne finale Eingaben offen), sonst 1993.
  **Zeile 1999 wird nicht erreicht** (`sources[,4]` bleibt 1); die frühere Übergabe nannte sie als Leser. `hi_time` ist FALSE.
- `sum` in `long double`; eine einfache Double-Summe ändert 199 von 300 Stressfällen. NA/NaN-Token als Messbefund; nach `is.na` ohne Wirkung.
- R 3.5.0 mit wörtlichen Originalzeilen: 447 von 447 Werten bitgleich, 0 ulp. 135 neue Tests, **1.172** gezielte Tests grün; App-Tests 119 grün
  (2 Wrapper explizit ersetzt); Edge in beiden Formaten und zwei Standorte ok. Phase-1-Prüfer Exit 2.

**App:** [Stand 11](app_reproduction_2026_11_09_local_cor/README.md) (Pin `cbfaf04f…`): 14 Teilmodule, Tabelle C1–C8. Stände 1–10 eingefroren.

**Offen:** Kein Originalfall; elf Nachweisbündel offen, keine Variante gewählt. Die Vektoren stammen im Lauf aus Ziehungen. Lizenzfrage zu `tools/r_sort_gpl.py`
und die R-nahe Rechenschicht liegen beim Nutzer. PR #77 offen, nicht gemergt.

**Als Nächstes:** [Claude-Übergabe](CLAUDE_NEXT_STEP_AFTER_LOCAL_COR_2026_11_09.md): `reconst`/`predictant` aus vorgegebenem `tf_sample`, danach `wt_power`.

## `reconstruction_climate` nachgebildet, im gebundenen Lauf ohne Aufrufer, App-Stand 10, 08.11.2026

**Abgeschlossen** (Nutzerauftrag „Ja, nachbauen“): [Modul und Belege](literal_reconstruction_climate_2026_11_08/README.md).

- `mcmc_functions_v1.6.R:1188–1199` mit expliziten Globalen; erste Zeile mal TF, dann `weightedMedian`, oder ohne `recent` nur der Prior.
- **Der gebundene Lauf ruft die Funktion nicht auf.** Einziger Aufruf: `mcmc_sampling_v1.4.R:79`, eine nirgends geladene Datei (540 R-Dateien durchsucht).
  Aktiv ist `mcmc_functions_v1.6.R:1850–1851`, aus Ziehungen. Auch `mcmc_prepare.R:1408–1413` wird wegen `recent & !add.sites` nicht erreicht.
- R 3.5.0 + matrixStats 0.56.0: 5.510 von 5.510 Werten bitgleich. 62 neue Tests, **1.037** gezielte Tests grün; App-Tests 108 grün
  (2 Wrapper explizit ersetzt); Edge in beiden Formaten und zwei Standorte ok. Phase-1-Prüfer Exit 2.

**App:** [Stand 10](app_reproduction_2026_11_08_reconstruction_climate/README.md) (Pin `329e6a8b…`): 13 Teilmodule, das neue als bedingt markiert. Stände 1–9 eingefroren.

**Offen:** Kein Originalfall; elf Nachweisbündel offen, keine Variante gewählt. Der Baustein ändert nichts am Nachbau des Laufs. Lizenzfrage zu `tools/r_sort_gpl.py`
liegt beim Nutzer. PR #77 offen, nicht gemergt.

**Als Nächstes:** [Claude-Übergabe](CLAUDE_NEXT_STEP_AFTER_RECONSTRUCTION_CLIMATE_2026_11_08.md): das lokale, tatsächlich aktive `cor` (`mcmc_functions_v1.6.R:2355`).

## Sechs ältere Module gegen R 3.5.0 geprüft, App-Stand 9, 08.11.2026

**Abgeschlossen** (Nutzerauftrag „Ja, weiter prüfen“): [Gegenprobe und Belege](r_crosscheck_six_modules_2026_11_08/README.md).

- Originalzeilen der Alters-/Tiefenübertragung, Altersauswahl, End-PDF, CDF-Abweichung, TF-Vorbereitung und des Taxon-Prunings
  wörtlich in R 3.5.0, auf den früheren Fixtures und auf 150 deterministischen Stressfällen mit ungenauen Doubles.
- **Alle 7.064 Fixture-Werte bitgleich**, keine Abweichung bei Sonderwerten.
- Zusammen mit den Stresszahlen 34.918 von 41.966 Werten bitgleich (ohne Altersraster), höchstens 4 ulp, nur Summen und
  Matrixprodukte. Masken, Raster, Indizes und Pruning-Entscheidungen sind überall bitgleich.
- **Ursache belegt:** R summiert in `long double` und nutzt Referenz-BLAS, die Module Double-Summen über NumPy. So nachgerechnet
  stimmen alle 24.822 betroffenen Werte bitgenau mit R. Die Module bleiben unverändert und damit nicht bitgleich; markiert.
- `rowCumsums` stammt im Fenster nur aus matrixStats. Quellbefund: `na.rm=T` wird dort von `...` verschluckt.
- 41 neue Tests, **975** gezielte Tests grün; App-Tests 98 grün (2 Wrapper explizit ersetzt). Phase-1-Prüfer Exit 2.
- Edge in beiden Formaten und zwei Standorte im ersten Lauf ok.

**App:** [Stand 9](app_reproduction_2026_11_08_r_crosscheck_six/README.md) (Pin `50fe8b0a…`): Tabelle je Modul. Stände 1–8 eingefroren.

**Offen:** Kein Originalfall; elf Nachweisbündel offen, keine Variante gewählt. Lizenzfrage zu `tools/r_sort_gpl.py` liegt beim Nutzer.
PR #77 offen, nicht gemergt.

**Als Nächstes:** [Claude-Übergabe](CLAUDE_NEXT_STEP_AFTER_SIX_MODULE_CROSSCHECK_2026_11_08.md).

## Gewichteter Median nachgebildet, App-Stand 8, 08.11.2026

**Abgeschlossen** (Auftrag: [Übergabe nach dem Dirichlet-Quotienten](CLAUDE_NEXT_STEP_AFTER_DIRICHLET_RATIO_2026_11_03.md)):
[Modul und Belege](literal_weighted_median_2026_11_08/README.md).

- `mcmc_prepare.R:1628–1664` auf oberster Ebene; `taxa_change` ist `FALSE`. 1623/1624 werden überschrieben, die
  `spline`-Ergebnisse 1627/1647 haben keine Leser.
- matrixStats im Fenster: Der normalisierte C-Kern ist in allen 13 Fassungen ein Text; die R-Sortierung war im Fenster
  für diesen Aufruf unverändert.
- **Wann NA:** irgendein Gewicht NA/NaN oder kein Gewicht positiv; ein NA in `x` nur bei Gewicht `> 0`, bei `+Inf` zählt die
  Reihenfolge. Damit fällt eine TF-Zeile bei endlichem Raster genau dann in `which.tf.missing`, wenn sie ein NA/NaN enthält
  oder keinen positiven Wert hat. Nur der x1-Median wird in 1664 geprüft.
- Quellnahe Eigenheiten erhalten: `-Inf` zählt im `+Inf`-Zweig, ungerades n im Gleichgewichtszweig ist nicht der Median,
  Ties über das instabile `R_qsort_I`. Überlauf der Gesamtmasse wird abgelehnt (C liest außerhalb der Felder).

**Prüfung:**

- 29 Kunstfälle bitgleich mit einer unabhängigen Referenz (NA über geschlossene Regel, Werte in exakten Brüchen); die
  NA-Regel stimmt für 400 Stressfälle.
- **R 3.5.0 mit matrixStats 0.56.0** (Binärpaket vorher per Hash festgehalten): 439 von 439 Werten bitgleich, NA und NaN
  unterschieden; Originalzeilen wörtlich auf Kunstmatrix, `which.tf.missing = 2 3 5`.
- 182 neue Tests, **934** gezielte Tests grün; App-Tests 91 grün (2 Wrapper explizit ersetzt). Phase-1-Prüfer Exit 2.
- Edge in beiden Formaten (390 px/PC, Hell/Dunkel, DE/EN, Gesten) und zwei Standorte ok, nach einer
  Korrektur der Tabellenzellen.

**Zusatz:** [`tann <- FALSE`](literal_weighted_median_2026_11_08/tann_binding.json) belegt: Bedingung 1248, Zuweisung 1249;
einziger Leser im aktiven `reconstruction` ist `mcmc_functions_v1.6.R:1939`. Nur gesammelt, nicht gedeutet.

**App:** [Stand 8](app_reproduction_2026_11_08_weighted_median/README.md) (Pin `4d25a3f6…`): zwölf Teilmodule, R-Zeile um
matrixStats ergänzt, Tabellen M1–M9 und NA-Maske. Stände 1–7 eingefroren.

**Offen:** Kein Originalfall; elf Nachweisbündel offen, keine Variante gewählt. Paketanbieter nur im Fenster gebunden.
Die sechs älteren Module sind weiterhin nicht gegen R geprüft. Lizenzfrage zu `tools/r_sort_gpl.py` (GPL-Übersetzung im
sonst MIT-lizenzierten Repository) liegt beim Nutzer. PR #77 offen, nicht gemergt.

**Als Nächstes:** [Claude-Übergabe](CLAUDE_NEXT_STEP_AFTER_WEIGHTED_MEDIAN_2026_11_08.md): sechs Module gegen R.

## Dirichlet-Quotient nachgebildet, App-Stand 7, 03.11.2026

**Abgeschlossen** (Nutzerentscheidung „Ja, ddirichlet“): [Modul und Belege](literal_dirichlet_ratio_2026_11_03/README.md).

- `mcmc_functions_v1.6.R:1269–1274` laufen nach jeder Ziehung unbedingt (Tiefe 1). `num_taxa` ist global, der
  Vorschlag enthält `dirichlet_spread` nicht. Leser: `mcmc_conduction.R` 97/151/302.
- `ddirichlet` wird wörtlich nach LaplacesDemon nachgebildet (im CRAN-Fenster gebunden):
  - die Normierungszeile weist nichts zu, ungenormte Gewichte bleiben ungenormt;
  - Abbruch bei `x < 0` oder `alpha <= 0`;
  - `is.na` → 0 erfasst NaN aus `0 · Inf`, `Inf` bleibt.
- `lgamma`, `log` und `exp` werden einmal korrekt gerundet; CPythons `math.lgamma` liegt bis zu 3 ulp daneben.

**Prüfung:**

- 8 Quotienten- und 3 Dichtefälle sind bitgleich mit einer unabhängigen Spouge/Decimal-Referenz; die geschlossenen
  Formen werden auf 0–1 ulp getroffen.
- **R 3.5.0 mit LaplacesDemon 16.1.6** wertet 1269–1274 wörtlich aus: 59 von 62 Werten bitgleich, höchstens 4 ulp.
  Ursache: R rundet `lgamma(0.4)` 1 ulp neben dem korrekten Wert. Das ist als Befund markiert, nicht emuliert.
- 50 neue Tests, **752** gezielte Tests grün; App-Tests 81 grün (2 Wrapper explizit ersetzt).
- Edge in beiden Formaten (390 px/PC, Hell/Dunkel, DE/EN, Gesten, zwei Standorte) im ersten Lauf ok.
- Phase-1-Prüfer Exit 2.

[App-Stand 7](app_reproduction_2026_11_03_dirichlet_ratio/README.md): Pin `cf1b77f2…`, elf Teilmodule, Zeile zur
R-Gegenprobe, Tabelle D1–D4. Stände 1–6 bytegleich, Stand 6 nur mit Linkcommit `3104f3c`.

**Offen:**

- Echte Gewichte (`rdirichlet`), Akzeptanz, Sampler.
- Sechs Module ohne R-Gegenprobe.
- `weightedMedian`.
- Kein Originalfall; elf Nachweisbündel offen.

Rechenschritt `766a1ff`; App-Sicherungscommit siehe
`git log -1 -- docs/thesis_research/app_reproduction_2026_11_03_dirichlet_ratio`.
[Nächste Claude-Übergabe](CLAUDE_NEXT_STEP_AFTER_DIRICHLET_RATIO_2026_11_03.md).
Gesichert auf `dissertation-reconstruction`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77), nicht gemergt.
Der folgende Abschnitt ist Verlauf.

## Laufzeitbelege und R-3.5.0-Gegenprobe, 03.11.2026

**Auftrag (Chat 03.11.2026):** Kinneret-Index bestimmen, R- und Paketfassungen selbst suchen, CRAN-Quellen lesen,
die Module mit R gegen den Originalcode prüfen. R-Läufe für Gegenproben sind damit **ausdrücklich erlaubt**; Sampler,
Trainings, Altersmodelle und Fossilrekonstruktionen weiterhin nicht.

**Abgeschlossen:** [Belegpaket](r_runtime_crosscheck_2026_11_03/README.md).

- **Kinneret = Slot 3** des Zusatzorte-Zweigs. `add.sites <- T` ist die einzige Einstellung (`main_v1.2.R:73`);
  `stations = 1:length(core.data)` ohne Slot 4. Kinneret-Schalter in `mcmc_prepare.R` 790, 881, 1145, 1209 und
  **1248 `tann <- FALSE`** (oberste Ebene). Was `tann` im Lauf bewirkt, ist noch offen.
- **R-Fassung:** Alle 13 Laufeingaben und 31 von 32 Kinneret-Containern stammen aus **R 3.5.0** (2019/2020). In
  `diss` gibt es keine `sessionInfo`, `renv` oder Paketliste; die Laufzeitfassung bleibt ungebunden.
- **Paketanbieter im Fenster 2018-04-23…2023-07-07:** Untersucht wurden 82 Pakete (55 angehängte samt `Depends`),
  11 davon mit `exportPattern` am Tarball.
  - `rdirichlet`/`ddirichlet` liefert vorne im Suchpfad **LaplacesDemon**. Der Funktionstext ist in allen vier
    Fassungen gleich; `ddirichlet` ist deterministisch, normiert `x` aber wegen fehlender Zuweisung nicht.
  - `weightedMedian` liefert nur **matrixStats**: 13 Fassungen mit gleicher Signatur und sechs Quelltextständen,
    ohne erkennbare Änderung des Rechenwegs in den eigenen Dateien. Die Hilfsdateien sind noch nicht verglichen.
- **R 3.5.0** aus dem CRAN-Archiv (MD5 und Signatur geprüft), nur für den Nutzer unter `.local/R` installiert.
  Die wörtlichen Originalzeilen von Sampler-Vorbereitung, Gewichtsstufe, Gamma-Raster und Taxonzahl-Priors liefern
  auf allen Kunstfixtures **389 bitgleiche Prüfungen**. Auch alle 14 akkumulatorabhängig markierten Werte stimmen
  auf dieser Plattform.

**Prüfung:**

- 13 neue Belegtests (ohne R/Netz), **702** gezielte Tests grün.
- Beide Phase-1-Prüfer Exit 2.
- Gesamtsuite, `verify_packet.py` und `tools/verify_*` nicht ausgeführt.

**Offen:**

- Sechs Module noch ohne R-Gegenprobe.
- `weightedMedian`-Hilfsdateien noch nicht verglichen.
- Exakte Paketfassungen des Laufs ungebunden.
- Kein Originalfall; elf Nachweisbündel offen. Keine App-Änderung in diesem Paket.

[Nächste Claude-Übergabe](CLAUDE_NEXT_STEP_AFTER_R_CROSSCHECK_2026_11_03.md).
Gesichert auf `dissertation-reconstruction`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77), nicht gemergt.
Der folgende Abschnitt ist Verlauf.

## Kern der Gewichtsstufe nachgebildet, App-Stand 6, 03.11.2026

**Abgeschlossen:** [Determinismusprüfung und Modul](literal_weights_stage_2026_11_03/README.md).
Der deterministische Kern von `parameter_stage_weights` (`mcmc_functions_v1.6.R:1259, 1276–1280`) nutzt nur
base R ohne Zufall. Die Funktion ist einmal definiert (1242), durch `gibbs <- F` gewählt und wird in jeder
Iteration aufgerufen (`mcmc_conduction.R:142`); die Zeilen liegen auf Tiefe 1 des Funktionsrumpfs.
**Kein Samplerlauf, keine Dirichlet-Ziehung:** Die Gewichte sind explizite Eingabe. Belegt und umgesetzt:

- `values * new_weights` recycelt die `ncol` Gewichte über die zeilenweisen Werte aus Stand 5; jede
  Position trifft ihr eigenes Taxon. `matrix(byrow = T)` füllt die Zeilen zurück.
- `rowSums(na.rm = T)` überspringt NA und NaN; eine Zeile ohne Werte gibt 0, eine mit `-Inf` gibt `-Inf`.
  Zeilen, deren Summe nicht akkumulator- und reihenfolgeunabhängig ist, werden markiert (hinreichende
  Zweierpotenzschranke; die Referenz zählt alle Teilsummen auf).
- Die Division recycelt den Summenvektor je Zeile: `0/0` NaN, `x/0` `±Inf`, `-Inf/-Inf` NaN, `x/-Inf`
  vorzeichenbehaftete Null. `z = 1:ncol`; `taxa_weights` übernimmt die Gewichte ohne Renormierung.

Nur dokumentiert:

- `rdirichlet` (1248), Rücksetzung nach 1000 Versuchen (1253–1257), `ddirichlet`-Quotient und `is.na`
  (1269–1274), Akzeptanz mit `runif` (`mcmc_conduction.R:304`).
- `rdirichlet`/`ddirichlet` sind in keiner der acht Dateien definiert (auch nicht in Rcpp-Zeichenketten).
  Gebunden ist nur die Reihenfolge von 55 `library()`-Aufrufen (52 vor der Schleife, frühere Pakete nannten
  50; deren Angaben bleiben unverändert). Der Anbieter bleibt ungebunden, lokal gibt es keine R-Installation.
- Leser: `reconstruction` (1817) liest `taxa.appear.core.temp`; `z` wählt in `mcmc_conduction.R:146` die
  TF-Zeilen.

[App-Stand 6](app_reproduction_2026_11_03_weights_stage/README.md): neuer letzter `STATES`-Eintrag mit Pin
`7eb9ac5e…`, zehn Teilmodule; Stände 1–5 bytegleich, Stand 5 nur mit Linkcommit `235f797`.

**Prüfung:**

- 134 neue Tests gegen eine unabhängige `Fraction`-Referenz und Handerwartungen (exakt, einschließlich
  Vorzeichen der Null); vier Fälle direkt aus dem Stand-5-Modul verkettet, Gegenprobe mit `rw_coefficients`.
  Mit den bisherigen **689** historischen Teil-/Belegtests grün.
- 69 App-/Daten-/Buildtests (2 Wrapper übersprungen, explizit ersetzt).
- Edge offline in beiden Formaten × Handy (390 px)/PC × Hell/Dunkel × DE/EN, inkl. Gesten und zwei
  Standorten: bestanden im ersten und im Endlauf.
- Beide Phase-1-Prüfer Exit 2.
- Gesamtsuite, `verify_packet.py` und `tools/verify_*` nicht ausgeführt.

**Blockiert/offen:** Kein Originalfall reproduziert; alle elf Nachweisbündel offen, keine Variante und kein
Dirichlet-Paket gewählt. Offen bleiben die echte Taxonmatrix (Alters-Tiefen-Modell, Pollendaten, NA-Maske
aus `weightedMedian`), die echten Gewichte (`rdirichlet`), Sampler-Schleife und alle Zufallsschritte.
Keine Trainings, Sampler, Altersmodelle, R-Läufe oder Fossilrekonstruktionen. Rechenschritt `4266c28`;
App-Sicherungscommit siehe `git log -1 -- docs/thesis_research/app_reproduction_2026_11_03_weights_stage`.
[Nächste Claude-Übergabe](CLAUDE_NEXT_STEP_AFTER_WEIGHTS_STAGE_2026_11_03.md).
Gesichert auf `dissertation-reconstruction`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77), nicht gemergt.
Der folgende Abschnitt ist Verlauf.

## Sampler-Vorbereitung nachgebildet, App-Stand 5, 02.11.2026

**Abgeschlossen:** [Determinismusprüfung und Modul](literal_sampler_preparation_2026_11_02/README.md).
Die Sampler-Vorbereitung `mcmc_conduction.R:53–79` nutzt nur base R ohne Zufall und läuft im
gebundenen Hauptlauf (`main_v1.2.R:676`; 53, 63–79 auf Klammertiefe 0). **Kein Samplerlauf.**
Belegt und umgesetzt:

- `gibbs <- F` (`main_v1.2.R:624`, Tiefe 0) ist die einzige Zuweisung in acht gebundenen Dateien;
  `F`/`T` werden nie neu belegt. Daraus folgen `how_many_taxa = num_taxa` und
  `parameter_stage_weights`. Der Gibbs-Zweig `floor(num_taxa/2)` wird nur gemeldet.
- `c(t(.))` flacht zeilenweise ab; Zeile 75 überschreibt die Rohwerte aus 64; `ncol`/`nrow` und
  `colMeans(na.rm = T)` gelten der Rohmatrix.
- Spaltendivision durch `max(na.rm = T)`: NA bleibt NA, Maximum 0 gibt NaN bzw. `-Inf`.
  Eine Ganz-NA-Spalte (`-Inf` mit Warnung) wird abgelehnt.
- `jeffreys_prior = rep(0.5, n)`.

Leser, nur dokumentiert:

- `values`/`ncol`/`nrow` liest `parameter_stage_weights:1276`. Das Gewichtsrecycling stimmt nur
  bei zeilenweiser Abflachung.
- `taxa_means` hat keinen Leser; bei `colMeans` ist der Akkumulator markiert.
- `jeffreys_prior` liest nur der tote Gibbs-Zweig.
- `combs` hat keinen Leser; `tf_x12_mean_current` ist dokumentiert.

Quellbefunde ohne Korrektur: `main_v1.2.R` hat drei überzählige `}` (740/788/829, nach der
gebundenen Schleife), daher ist kein Ganzdatei-`source()` möglich. `c` in 1703 ist eine
Schleifenvariable.
[App-Stand 5](app_reproduction_2026_11_02_sampler_prep/README.md): neuer letzter `STATES`-Eintrag
mit Pin `61c6f231…`, neun Teilmodule; Stände 1–4 bytegleich, Stand 4 nur mit Linkcommit `0a3dcdb`.

**Prüfung:**

- 118 neue Tests gegen eine unabhängige `Fraction`-Referenz und Handerwartungen (exakt); mit den
  bisherigen **555** historischen Teil-/Belegtests grün.
- 61 App-/Daten-/Buildtests (2 Wrapper übersprungen, explizit ersetzt).
- Edge offline in beiden Formaten × Handy (390 px)/PC × Hell/Dunkel × DE/EN, inkl. Gesten und zwei
  Standorten: bestanden im ersten und im Endlauf.
- Beide Phase-1-Prüfer Exit 2.
- Gesamtsuite, `verify_packet.py` und `tools/verify_*` nicht ausgeführt.

**Blockiert/offen:** Kein Originalfall reproduziert; alle elf Nachweisbündel offen, keine Variante
gewählt. Offen bleiben die echte Taxonmatrix (Alters-Tiefen-Modell, Pollendaten, NA-Maske aus
`weightedMedian`), die Sampler-Schleife und alle Zufallsschritte (`rmvnorm_rcpp`, `rdirichlet`,
`rbeta`, `set.seed(2022)`). Keine Trainings, Sampler, Altersmodelle, R-Läufe oder
Fossilrekonstruktionen. Rechenschritt `00ab021`; App-Sicherungscommit siehe
`git log -1 -- docs/thesis_research/app_reproduction_2026_11_02_sampler_prep`.
[Nächste Claude-Übergabe](CLAUDE_NEXT_STEP_AFTER_SAMPLER_PREP_2026_11_02.md).
Gesichert auf `dissertation-reconstruction`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77), nicht gemergt.
Der folgende Abschnitt ist Verlauf.

## Fehlende TF und `num_taxa`-Priors nachgebildet, App-Stand 4, 02.11.2026

**Abgeschlossen:** [Determinismusprüfung und Modul](literal_tf_missing_priors_2026_11_02/README.md).
`which.tf.missing` samt `num_taxa`-Priors (`mcmc_prepare.R:1664–1685`) nutzt nur base R ohne
Zufall und läuft im gebundenen Hauptlauf (1664–1665/1681–1685 Klammertiefe 0; Entfernung
hinter dem datenabhängigen Längenschutz). Die NA-Maske ist **explizite Eingabe**, weil
`weightedMedian` offen bleibt; nur das x1-Mittel zählt. Belegt und umgesetzt: leere Maske
→ keine Indizierung, `num_taxa` bleibt `ncol(all.avail)` aus **1435**; sonst `nrow` aus 1678;
Reihenfolge bleibt; `taxa_change` Boolwert (im Lauf tot, `& FALSE` in 1500);
`round(num_taxa/4)` nach dokumentierter R-Regel zur geraden Ziffer (Half-up wiche für
n = 2, 10, 18 … ab; `prior$nr.taxa.sd` hat keinen aktiven Leser). Befunde: `phi1`/`phi2`
(1439–1440) bleiben nach der Entfernung veraltet; `posterior <- list()` (1706) verwirft
`posterior$nr.taxa.mean`. Ein Resttaxon (Dimensionsverlust), null Resttaxa und `drop`
durch Ausdehnung 1 werden abgelehnt. Schwellen 131–137 nicht gewählt (`fitdistr`,
`quantile`, `inside.range`). [App-Stand 4](app_reproduction_2026_11_02_tf_missing/README.md):
neuer letzter `STATES`-Eintrag mit Pin `7b7bf1f4…`, acht Teilmodule; Stände 1–3 bytegleich,
Stand 3 nur mit Linkcommit `7d3d0f6`.

**Prüfung:** 85 neue Tests gegen unabhängige Listen-/Bruch-/Ganzzahlreferenz und
Handerwartungen (exakt), mit bisherigen **437** historischen Teil-/Belegtests grün;
54 App-/Daten-/Buildtests (2 Wrapper übersprungen, explizit ersetzt); Edge offline in
beiden Formaten × Handy/PC × Hell/Dunkel × DE/EN inkl. Gesten und zwei Standorten bestanden
(Erstlauf: Tabellenüberlauf auf Handybreite, behoben); beide Phase-1-Prüfer Exit 2.
Gesamtsuite, `verify_packet.py` und `tools/verify_*` nicht ausgeführt.

**Blockiert/offen:** Kein Originalfall reproduziert; alle elf Nachweisbündel offen, keine
Variante gewählt. Echte NA-Maske (`weightedMedian`, historische TFs), `prior.recent`-Anpassung
und alle Zufallsschritte offen. Keine Trainings, Sampler, Altersmodelle, R-Läufe oder
Fossilrekonstruktionen. Rechenschritt `9bb8ed9`; App-Sicherungscommit siehe
`git log -1 -- docs/thesis_research/app_reproduction_2026_11_02_tf_missing`.
[Nächste Claude-Übergabe](CLAUDE_NEXT_STEP_AFTER_TF_MISSING_2026_11_02.md).
Gesichert auf `dissertation-reconstruction`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77), nicht gemergt.
Der folgende Abschnitt ist Verlauf.

## Gamma-Prior-Raster nachgebildet, App-Stand 3, 02.11.2026

**Abgeschlossen:** [Determinismusprüfung und Gamma-Prior-Raster](literal_gamma_prior_grid_2026_11_02/README.md).
Von Gamma-Raster, `which.tf.missing`, `weightedMedian` und den geseedeten
Ziehungen ist das Raster (`mcmc_prepare.R:114–128`, `526–536`, `1264–1281`, `1733`)
der nächste Schritt ohne Zufall und ohne ungebundene Paketsemantik; es läuft im
gebundenen Hauptlauf **unbedingt** (Klammertiefe 0, `else` von `stad_interstad <- FALSE`).
`which.tf.missing` ist mit expliziter Maske ebenfalls deterministisch, hängt aber
an der offenen `weightedMedian`-Ausgabe → dokumentiert, nicht gewählt. Modul hält
`seq.default` (Fuzz, zwei Rundungen, `pmin`), erstes `which.min` und `rate` aus dem
Wurzelzweig wörtlich; Obergrenze explizit (50000 in `mcmc_prepare.R`, **5000** in der
Kopie `mcmc_prep_v1.2.R:1103`); `recent` und `min(age) ≥ 2000` als Boolwerte.
Befunde: geschlossene Form und „nächster Rasterpunkt“ weichen in Kunstfällen ab,
jenseits des Rasters springt `which.min` auf 0,1; der Rezentzweig (1733) ersetzt
`rate` durch `shape/mean`. [App-Stand 3](app_reproduction_2026_11_02_gamma_grid/README.md):
neuer letzter `STATES`-Eintrag mit Pin `43edfcb6…`, sieben Teilmodule, Stände 1 und 2
bytegleich eingefroren.

**Prüfung:** 55 neue Tests gegen unabhängige Float-/Bruchreferenz und
Handerwartungen (bitgleich), mit bisherigen **352** historischen Teil-/Belegtests
grün; 47 App-/Daten-/Buildtests (2 Wrapper übersprungen, explizit ersetzt); Edge
offline in beiden Formaten × Handy/PC × Hell/Dunkel × DE/EN inkl. Gesten und zwei
Standorten bestanden; beide Phase-1-Prüfer Exit 2. Gesamtsuite und
`verify_packet.py` nicht ausgeführt.

**Blockiert/offen:** Kein Originalfall reproduziert; alle elf Nachweisbündel
offen, keine Variante gewählt. `prior.recent`-Anpassung (`fitdistr`), `weightedMedian`
(`matrixStats`-Fassung) und alle Zufallsschritte offen. Keine Trainings, Sampler,
Altersmodelle, R-Läufe oder Fossilrekonstruktionen.
[Nächste Claude-Übergabe](CLAUDE_NEXT_STEP_AFTER_GAMMA_GRID_2026_11_02.md).
Gesichert auf `dissertation-reconstruction`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77), nicht gemergt.
Der folgende Abschnitt ist Verlauf.

## Taxon-Pruning nachgebildet, App-Stand 2, 02.11.2026

**Abgeschlossen:** [Determinismusprüfung und Taxon-Pruning](literal_taxon_pruning_2026_11_02/README.md).
Von `weightedMedian`, Taxon-Pruning, Splines und Zufallsschritten ist nur das
Pruning (`mcmc_prepare.R:51–80`) ohne Zufall und ohne ungebundene Paketsemantik;
`weightedMedian` hängt an der unbekannten `matrixStats`-Fassung, Splines wirken
nur in auskommentierten Zeilen. Modul mit R-Eigenheiten (abwärts zählendes
`16:ncol`, Gleichstände über `%in%`, `next()` als Signal). Im gebundenen
Hauptlauf abgeschaltet (`main_v1.2.R:659`) → bedingter Schritt, keine
Laufaktivierung. [App-Stand 2](app_reproduction_2026_11_02_pruning/README.md):
neuer datierter Nachbaustand mit eigenem Pin, sechs Teilmodule, Stand vom
02.11.2026 bytegleich eingefroren und weiter geprüft.

**Prüfung:** 46 neue Tests gegen unabhängige Bruchreferenz, mit bisherigen
**297** historischen Teil-/Belegtests grün; 40 App-/Daten-/Buildtests
(2 Wrapper übersprungen, explizit ersetzt); Edge offline in beiden Formaten ×
Handy/PC × Hell/Dunkel × DE/EN inkl. Gesten und zwei Standorten bestanden;
beide Phase-1-Prüfer Exit 2. Gesamtsuite und `verify_packet.py` nicht ausgeführt.

**Blockiert/offen:** Kein Originalfall reproduziert; alle elf Nachweisbündel
offen, keine Variante gewählt. `weightedMedian` wartet auf die belegte
`matrixStats`-Fassung. Keine Trainings, Sampler, Altersmodelle, R-Läufe oder
Fossilrekonstruktionen. PR #77 laut GitHub mit `main` in Konflikt (#78);
Lösung nur auf Nutzerauftrag. [Nächste Claude-Übergabe](CLAUDE_NEXT_STEP_AFTER_TAXON_PRUNING_2026_11_02.md).
Gesichert auf `dissertation-reconstruction`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77), nicht gemergt.
Der folgende Abschnitt ist Verlauf.

## Dissertationsnachbau in der App sichtbar, 02.11.2026

**Abgeschlossen:** [App-Anbindung des geprüften Nachbaus](app_reproduction_2026_11_02/README.md).
Die Ansicht „Forschungsstand“ zeigt in Ordner- und Ein-Datei-App einen eigenen
Abschnitt „Dissertationsnachbau“: Status „Teilmodule geprüft; vollständiger
Originalfall offen“, fünf Teilmodule mit Belegen (251 Teil-/Belegtests), die
elf offenen Originaleingaben aus dem unveränderten Belegvertrag und ein
eingeklapptes, ausdrücklich synthetisches TF-/Kettenbeispiel (Summen 1 und 2/3).
Neuer datierter Nachbaustand per SHA-256 gebunden; alter Forschungs-Snapshot,
Pins, Belege und frühere Übergaben unverändert.

**Prüfung:** 34 App-/Daten-/Buildtests (2 Wrapper übersprungen, explizit
ersetzt), Browserregression offline mit Edge in beiden Formaten × Handy/PC ×
Hell/Dunkel × DE/EN inkl. Gesten und zwei Standorten bestanden; 251 historische
Tests grün; beide Phase-1-Prüfer weiter Exit 2. Paketgebundener
`verify_packet.py` des TF-Pakets endet erwartungsgemäß an „src unverändert“.

**Blockiert/offen:** Kein Originalfall reproduziert; alle elf Nachweisbündel
offen, keine Variante gewählt. Keine Trainings, Sampler, Altersmodelle,
R-Läufe oder Fossilrekonstruktionen. Gates und Baseline erhalten.
[Nächste Claude-Übergabe](CLAUDE_NEXT_STEP_AFTER_APP_REPRODUCTION_2026_11_02.md):
nächsten deterministischen Quellschritt isolieren, danach neuer datierter
Nachbaustand in der App. Gesichert auf `dissertation-reconstruction`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77), nicht gemergt.
Der Hinweis „App/CLI/Baseline in diesem Paket unverändert“ im folgenden
Abschnitt betrifft nur das TF-Paket und ist Verlauf.

## Historische TF-Vorbereitung als App-Baustein geprüft, 02.11.2026

**Abgeschlossen:** [TF-Zeilennormierung und anschließende Zellmaske](literal_tf_preparation_2026_11_02/README.md)
aus `mcmc_prepare.R:1581–1586,1613–1618` als isoliertes Modul nachgebildet.
Endliche nichtnegative Eingaben, positive Ausgangsmassen und explizite
Klimazellmasken; keine zweite Normierung. TANN/PANN bleiben getrennt.
Quellhash und vier portable Originalspannen vor/nach Lesen bestätigt.

**Prüfung:** 51 neue Fälle und alle bisherigen 200 historischen Teil-/Belegtests,
insgesamt 251 bestanden. Neue Kunstverbindung mit allen fünf unveränderten/
neuen Teilmodulen gegen unabhängige Bruchrechnung: maximal 2,22e-16 Abweichung.
Zusätzliche Renormierung nach Maskierung verändert diese Kunst-End-PDF maximal
um 0,1000195; keine historische Fehlerabschätzung. Alte Fixture und Belege
unverändert. Beide Phase-1-Prüfer weiter Exit 2 mit elf offenen Voraussetzungen.

**Ziel und nächste Übergabe:** Nutzer bestätigt den vorrangigen Auftrag,
die Dissertation nachzubauen und in die App zu implementieren. Pro geprüftem
Paket eine konkrete Claude-Übergabedatei mit Abschlusslink. [Nächster Auftrag](CLAUDE_NEXT_STEP_APP_REPRODUCTION_2026_11_02.md):
den geprüften Nachbaustand und das klar synthetische TF-Beispiel an die bestehende
Forschungsansicht in beiden Offlineformaten anbinden. Keine historische
Reproduktionsbehauptung aus Kunsttests; ein vollständiger Originalfall bleibt
der fachliche Meilenstein. App/CLI/Baseline in diesem Paket unverändert.

**Grenzen/Sicherung:** Finale Basis/Listenchronologie, numerische TFs und
Laufbindung weiter offen. Keine Variantenwahl, R-Ausführung, Trainings/Sampler,
Altersmodelle oder Fossilläufe. Phase 1 vor Phase 2, Originale und fallbezogene
Gates erhalten. Bestehender Branch/PR #77, kein automatischer Merge. Frühere
„nächster Schritt TF-Vorbereitung“-Angaben unten sind Verlauf.

## climate_modelling tatsächlich zusätzlich abgeglichen, 02.11.2026

**Abgeschlossen:** [Neues Quellen-/Kopienpaket](climate_modelling_audit_2026_11_02/README.md)
prüft den bereitgestellten Ordner zusätzlich zum eingefrorenen Zwei-Wurzeln-
Snapshot: 1.828 Dateien, 395 RData, 343 NetCDF, 161 R-Quellen und ein ZIP-
Inhaltsverzeichnis. Zehn R-Quellen mit mtime nach 2021 priorisiert; Datum ist
kein Erstellungs-/Ausführungsnachweis. Neuer Collector liest den privaten
`climate_modelling_source_root` tatsächlich; keine pauschale PC-/Altinventur.

**Funde und Konflikte:** `main.R:128`-Verbreitungsbasis vorhanden und bytegleich
zur bereits belegten Kinneret-Fassung, 720 × 360 × 275; keine numerischen TFs.
F14/F15 haben passende CSV-Kandidaten (455/6.849 Zeilen), SISAL-V2-Zugehörigkeit
bleibt ungeklärt. Übrige 37 priorisierte Namen und alle acht benannten Posterior-
varianten ohne neue Namenskopie. `main_v1.2.R:858`-Sammelstand v4.0 fehlt namentlich;
gleichnamige v1.4-Dateien unterscheiden sich im Hash. Gefundene `sftlf`-Landfraktion
enthält keine benötigte `mask2`; regionale CRU-Metadaten ersetzen keine globalen
Klima-/Ankerdateien. Keine Quelle ausgewählt, umbenannt oder korrigiert.

**Prüfung:** 23 vorhandene Parser-/Strukturlesertests bestanden; beide Phase-1-
Prüfer erwarteter Exit 2, elf offene Nachweisbündel. Frühere 200 historische
Teiltests hier nicht neu gerechnet. Eingefrorene Nachweise/Startaufträge und
Originale erhalten; App/CLI/Baseline unverändert. Keine Trainings, Sampler,
Altersmodelle, R-Ausführung oder Fossilrekonstruktionen.

**Weiter:** [Neuer konkreter Übergabeauftrag](NEXT_STEP_AFTER_CLIMATE_AUDIT_2026_11_02.md).
Zuerst unabhängig prüfbare deterministische TF-Zeilennormierung und explizite
Zellmaskierung aus dem bereits eingefrorenen Auftrag, kein Vollrunner und
keine fehlenden TFs neu schätzen. Wiederaufbau nur verbleibender notwendiger
Lücken nach Versions-/Datenvertragsprüfung. Branch `dissertation-reconstruction`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77), kein Merge.
Ältere Ankündigungen eines ungeprüften Ordners unten sind Verlauf.

## Datei-Abgleich gesichert; neuer Ordner für Claude angekündigt, 02.11.2026

**Ergebnis:** [Notwendige Dateien, Lücken und lokale Kopien](MISSING_PROJECT_FILES_2026_11_02.md)
mit 39 priorisierten Anforderungen und acht getrennten Posteriorvarianten.
16 direkte Ladelücken des erhaltenen lokalen R-Einstiegs: zwölf RData/NC und
vier TXT/CSV. 369 R-Dateien lesend geprüft; 35 Dissertations- und 149 Zusatz-
`src`-Dateien mit mtime nach 2021 priorisiert. Dateizeiten sind keine gesicherte
Erstellungsgeschichte. Index von 584 Datendateien (257 RData/RDS/NC) und 18 ZIP-
Inhaltsverzeichnissen im bisherigen abgegrenzten Bestand. Keine gesamte PC-Suche.

**Positive Kopien:** Die über Automount referenzierte Ohrid-Verbreitungsdatei
liegt lokal vor. Zwei kleine Verbreitungscontainer gegen frühere Hashes bestätigt.
Automount-Verlust bedeutet daher nicht automatisch fehlender Dateiinhalt.
[Quellenregel entsprechend präzisiert](SOURCE_RECOVERY_PLAN.md).

**Neueste Nutzerinformation, noch ungeprüft:** In der nächsten Sitzung steht
zusätzlich `climate_modelling` zur Verfügung. Nutzer möchte zeitnah zu Claude
wechseln. [Vollständige Übergabe und erster Auftrag](CLAUDE_RESUME_CLIMATE_MODELLING_2026_11_02.md)
sichern den Chat-Arbeitsstand und diesen Hinweis. Claude soll den neuen Ordner
zuerst mit `main.R`/`main_v1.2.R` und der Liste abgleichen. Genauer Pfad und Inhalt
hier noch nicht bekannt; frühere Fehlbefunde gelten nur für bisherigen Suchumfang.

**Prüfung/Grenzen:** Acht neue synthetische Parserprüfungen bestanden. Frühere
200 historische Teiltests in diesem Datei-Abgleich nicht erneut gerechnet.
Originale und eingefrorene Nachweise unverändert; keine R-Ausführung, Trainings,
Sampler, Altersmodelle oder Fossilläufe. Elf Nachweisbündel bleiben offen,
keine finale Kinneret-Variante. Bestehender Branch/PR #77, kein Merge.

## Nutzerangabe zum Originalverlust dauerhaft verankert, 02.11.2026

**Neue gesicherte Kontextinformation:** Alle Dateien mit historischem Pfadpräfix
`/automount/` lagen laut Nutzer auf einer gelöschten Festplatte. Die dortigen
Originale sind verloren; benötigte Dateien müssen neu erstellt werden.
[Verbindlicher Wiederaufbauplan](SOURCE_RECOVERY_PLAN.md) sowie Hinweise in
`CLAUDE.md`, `AGENTS.md`, Arbeitsplan und Claude-Übergabe sichern die Regel für
künftige Sitzungen. Quelle ist die Nutzerangabe, keine neue Dateiinventur.

**Planfolge:** Für konkret an diesen Präfix gebundene Dateien Erzeugercode,
benötigte Eingaben/Parameter, Achsen und Prüfmaßstäbe gezielt erfassen und
Wiederaufbau vorbereiten. Nicht erneut nach dem Originaldatenträger/Speicherort
fragen oder pauschal danach suchen. Andere Fehlpfade nicht ohne Beleg zuordnen.
Neue Dateien als neu erzeugt kennzeichnen; keine automatische Originalidentität,
finale Variantenwahl oder Schließung der elf Nachweisbündel.

**Sicherung/Grenze:** Dokumentationsänderung auf Basis `91436c4` in
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
kein Merge. Eingefrorene Belege, Startaufträge und Numerik unverändert. Keine
Neuberechnungen, Trainings, Sampler, Altersmodelle oder Fossilläufe gestartet.
Nächster technischer Schritt bleibt die begrenzte TF-Vorbereitung.

## Gemeinsame synthetische Rechenkette geprüft, 02.11.2026

**Abgeschlossen:** [Gemeinsame Fixture, Bruchrechnung und Grenzen](literal_phase1_chain_2026_11_02/README.md)
prüfen die vier unveränderten Teilmodule über ihre Schnittstellen. Chronologie
4×5 → Probentiefenkern 4×3 → Pollen 4×2 → RW-Koeffizienten/TF-Produkt →
Roh-CDF-Scores für Original-IDs 1 und 3 → Rohsumme und normierte Kunst-PDF.
Nur Test-/Belegcode; kein historischer Runner und keine App-/CLI-Erweiterung.

**Prüfung:** 33 neue Tests, mit Vorpaketen 200 grün. Unabhängige skalare
`Fraction`-Rechnung ohne NumPy-/Modulimporte bestätigt sämtliche numerischen
Zwischenstufen bis auf höchstens 2.23e-16 (Toleranz 2e-14). Energiegrößen werden
keine Mischungsgewichte; nur verbleibende Original-IDs erhalten Scores.
Gegenbeispiele erkennen verfrühte Normierung, ausgelassene Nullersetzung,
Achsen-/Masken-/Platzhalterkonflikte und semantisch vertauschte TF-Zeilen.
Drei Skripthashes entsprechen vor/nach Lesen den bisherigen Bindungen.

**Bewusste Grenze:** Die TF-Massen 2 und 3 sind künstliche Schnittstelleneingaben;
sie belegen keinen historischen TF-Vorbereitungslauf. Normierung jeder Ziehung
vor der Summe verändert die Kunst-PDF maximal um 0.00282, stilles Normieren
der gelieferten TF-Zeilen um 0.09716. Keine historische R-Laufzeitgleichheit,
keine Originalnumerik, keine Wiederherstellung fehlender TFs oder Freigabe.

**Claude kann direkt fortsetzen:** [Neuer konkreter Startauftrag](CLAUDE_NEXT_STEP_TF_PREPARATION_2026_11_02.md)
mit portablen, hashgebundenen Originalauszügen, Formeln, Beispiel, Prüfkommandos
und Abnahmekriterien. Nächster begrenzter Schritt: TF-Zeilennormierung und
anschließendes Nullsetzen ausdrücklich maskierter Klimazellen aus
`mcmc_prepare.R:1581–1586,1613–1618`. Keine unbelegte `inside.range`-Randkonvention
wählen, keine neue TF schätzen oder umgebende Zufalls-/Medianpfade ausführen.
Der bisherige Startauftrag bleibt als eingefrorener Nachweis unverändert.

**Weiter offen:** Elf Nachweisbündel, finale Basis/Listenchronologie, numerische
historische TFs und Laufzuordnung. Kein vollständiger Fall reproduziert und
keine finale Kinneret-Variante gewählt. Keine Speicherortfrage oder Inventur.

**Sicherung:** Basis `674bf13`, bestehender Branch in
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
kein Merge. App/CLI, Baseline, Originale, Referenzen, bisherige Pakete, Module
und fallbezogene Monticchio-/Chronologie-Gates erhalten. Keine Trainings,
Sampler, Altersmodelle oder Fossilrekonstruktionen. Phase 1 vor Phase 2;
Plots jung links, alt rechts.

## Kumulative Rohdichte-/EMD-Teilprüfung und Claude-Auftrag, 02.11.2026

**Abgeschlossen:** [Quellen, Formeln und Grenzen](literal_cdf_discrepancy_2026_11_02/README.md)
belegen den Nullreferenzzweig und kumulierten Rohdichtevergleich aus
`final_reconstruction`; separates Modul `tools/historical_cdf_discrepancy.py`.
Die erhaltenen Fallstudienaufrufe übergeben zweimal Skalar 1. Der einseitige
Längentest der ersten Referenz ersetzt beide Referenzen durch Nullmatrizen.
Verglichen wird vor späterer Mischungsnormierung mit `sum(abs(C−R))/dims`;
kein Klima-Zellbreitenfaktor und keine zusätzliche Normierung.

**Wichtige Grenze:** Die Platzhalteranlage verwendet `dims` vor der lokalen
Zuweisung. Der Haupteinstieg setzt zwar 300, aber historische Aufrufumgebung
und finale TF-Breite fehlen. Geerbte und spätere Breite werden ausdrücklich
getrennt übergeben; Konflikte nicht still korrigiert. Nullreferenzscore ist
kein belegter Abstand zu einer älteren Rekonstruktions-CDF oder in °C/mm.

**Prüfung:** 43 neue synthetische Tests, mit Vorpaketen 167 grün. Rationale
Scorekontrollen exakt gleich; drei gezielt gelesene Skripte vor/nach Lesen
identisch zu bisherigen Hashbindungen. Keine Originalnumerik oder historische
R-/Paketlaufzeitgleichheit. Nichtendlichkeit, Recycling, Achsenreparatur und
allgemeine R-Längensemantik außerhalb der dokumentierten Domäne bleiben aus.

**Claude kann fortsetzen:** [Konkreter Startauftrag](CLAUDE_NEXT_STEP_2026_11_02.md)
enthält Modulkarte, Prüfkommandos, erwarteten Ausgang und Abnahmekriterien.
Nächster begrenzter Schritt: vorhandene Module über ein gemeinsames künstliches
Beispiel auf Reihenfolge, Achsen und Normierungszeitpunkte prüfen. Nur neues
Testmodul/Belegpaket, kein historischer Vollrunner und keine App-/CLI-Erweiterung.
Exakte Energieduplikate vor aktiven Ziehungen bestimmen, Original-IDs erhalten.

**Weiter offen:** Elf Nachweisbündel, finale Basis/Listenchronologie, numerische
historische TFs und tatsächliche Laufzuordnung. Keine finale Kinneret-Variante
gewählt, kein vollständiger Dissertationsfall reproduziert. Keine Speicherort-
frage, Globalinventur oder Ersatzdatenwahl.

**Sicherung:** Basis `d6e3343`, bestehender Branch in
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
kein Merge. App/CLI, Baseline, Originale, Referenzen, bisherige Belegpakete und
fallbezogene Monticchio-/Chronologie-Gates erhalten. Keine Trainings, Sampler,
Altersmodelle oder Fossilrekonstruktionen. Phase 1 vor Phase 2; Plots jung links,
alt rechts.

## Historische Altersraster und Zeilenauswahl geprüft, 02.11.2026

**Abgeschlossen:** [Quellen, Zustände, Formeln und Grenzen](literal_age_selection_2026_11_02/README.md)
verfolgen `age`, `site.age` und `reconst_ages` durch vier bereits gebundene
Skripte. Separates Modul `tools/historical_age_selection.py`: exakte Perioden-
rastermitgliedschaft, vorläufige/wirksame Maske des erhaltenen Kontrollflusses
und boolesche RW-Zeilenauswahl vor dem Matrixprodukt. Kein Runner.

**Belegt:** `age %in% grid` ist kein Intervalltest; erste Altersdifferenz
bestimmt die Rasterweite. Die erhaltene Vorbereitung setzt zunächst bei
`taxa.merge` eine Holozänmaske, überschreibt sie im wegen `& FALSE` erreichten
Alternativzweig wieder mit allen Alterszeilen. Das gilt nur bei erfolgreichem
Erreichen dieser Zeilen, nicht als historische Laufbindung. Gibbs verwendet
an dieser Stelle keine `reconst_ages`-Maske. Drei Masken-/Schaltgrößen sind
freie Umgebungsvariablen von `final_reconstruction`, keine formalen Parameter.
`site.age` bleibt separat und kann im Speläothemzweig überschrieben werden.

**Prüfung:** 39 neue synthetische Tests, mit den bisherigen Teilpaketen 124
Tests grün. Rechteckige Produkte und rationale Sollwerte stimmen in der
Prüfung exakt überein. Vier gelesene Quellskripte entsprechen vor/nach dem
Lesen ihren bisherigen Hashbindungen. Kein historischer R-Laufzeitvergleich.
Halbe/ganzzahlige Rastereinheiten und strikt passende Boolmasken begrenzen
den Nachbau; keine stillen R-Recycling-/Dimensionsverlustreparaturen.

**Grenzen:** Elf Nachweisbündel unverändert offen. Finale Originalbasis,
Listenchronologie, numerische historische TFs und tatsächliche Laufzuordnung
fehlen weiterhin. Kein finaler Kinneret-Pfad gewählt, keine Ersatzdaten und
kein vollständiger Dissertationsfall reproduziert. Keine Synchronisierung
von `age`/`site.age` oder Aktivierung des deaktivierten Taxonwechselzweigs.

**Nächster Einstieg:** Kumulativen Dichte-/EMD-Ausschnitt im gebundenen
`src/mcmc_functions_v1.6.R:366–370,485–490` gegen Fallstudienaufrufe mit skalaren
Referenzargumenten lesen: Normierungszeitpunkt, `dims`-Faktor, Referenzform und
Einheiten belegen und synthetisch prüfen. Keine neue Referenzkurve oder
Kalibrierung; getrennte Module noch nicht zu einem Vollrunner verbinden.

**Sicherung:** Basis `6241f41`, bestehender Branch in
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
kein Merge. App/CLI, Baseline, Originale, Referenzen, bisherige Belegpakete und
fallbezogene Monticchio-/Chronologie-Gates erhalten. Keine Trainings, Sampler,
Altersmodelle oder Fossilrekonstruktionen. Phase 1 vor Phase 2, Plots jung links,
alt rechts. Keine wiederholte Speicherortfrage oder pauschale Inventur.

## Historischer Alters-/Tiefenteil synthetisch geprüft, 02.11.2026

**Abgeschlossen:** [Quellen, Formeln und Grenzen](literal_age_depth_2026_11_02/README.md)
binden `check_ages.R` an den tatsächlich rechnenden Verbraucher
`mcmc_prepare.R`. Getrenntes Modul `tools/historical_age_depth.py`: explizite
Listenposition, einmalige Altersduplikatverschiebung, Auswahl der Tiefenspalten,
Zeilennormierung, historische Nullersetzungen und Pollen-Matrixprodukt.
Orientierung rho(D|A): Alter × Tiefengitter, dann Alter × Probentiefe;
Produkt mit Probentiefe × Taxon ergibt Alter × Taxon. Keine finale Auswahl.

**Prüfung:** 43 neue synthetische Tests bestanden, zusammen mit Taxongewichtung/
PDF-Summe und Belegvertrag 85 Tests grün. Rechteckige Matrizen und rationale
Sollwerte prüfen Orientierung und Reihenfolge; maximaler Fehler 2.23e-16.
Historisches `10e-10` nach Normierung bleibt 1e-9 ohne Renormierung;
Pollennullen werden separat zu 1e-6. Altersverschiebung um 0,5 kann Duplikate
übriglassen und verändert `site.age` nicht. Drei Quellhashes stimmen vor/nach
Lesen mit der vorherigen Bindung überein. Kein R-Laufzeitvergleich.

**Grenzen:** Mittelwertmatrix ersetzt keine Liste mehrerer Altersraster.
Ungeordnete/doppelte oder nicht passend indizierbare Tiefen werden nicht still
repariert. Standortüberschreibungen, Fallback, Birkat-Ram-LOESS und NA-Füllung
sind ausgeschlossen. Alle elf Nachweisbündel bleiben offen; finale Originalbasis,
Listenchronologie, historische TFs und Laufbindung fehlen weiterhin.
Kein vollständiger Fall reproduziert, keine neue Chronologie oder Variante.

**Nächster Einstieg:** Verbraucher von `age` gegenüber `site.age` und
`reconst_ages` im bereits gebundenen `src/mcmc_functions_v1.6.R` gezielt auf
Zeilen-/Zeitfensterauswahl verfolgen und einen belegten Ausschnitt synthetisch
prüfen, bevor getrennte Module verbunden werden. Keine neue Gesamtinventur,
Speicherortfrage, Ersatzdatenwahl oder Phase 2.

**Sicherung:** Basis `f7d97c1`, bestehender Branch in
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
kein Merge. App/CLI, Baseline, Originale, Referenzen, bisherige Belegpakete und
fallbezogene Monticchio-/Chronologie-Gates erhalten. Keine Trainings, Sampler,
Altersmodelle oder Fossilrekonstruktionen. Plots weiter jung links, alt rechts.

## Historischer deterministischer Teilnachbau, 02.11.2026

**Abgeschlossen:** [Taxongewichtung und PDF-Summe](literal_final_pdf_2026_11_02/README.md)
als getrenntes Modul `tools/historical_final_pdf.py`. Belegte Zeilen der erhaltenen
`final_reconstruction`-Fassung nachgebildet: Spaltenmaximum, RW-/Gibbs-Koeffizienten,
TF-Matrixprodukt, exakte Energieduplikate und anschließende PDF-Zeilensumme.
Kein Runner, keine App-Integration, keine automatische finale Variantenwahl.

**Prüfung:** 22 neue Tests mit handgeleiteten rationalen Sollwerten und
Grenzfällen bestanden; zusammen mit den bisherigen 20 Belegvertragstests
42 Tests grün. Uneinheitliche TF-Massen und rechteckige Proben-/Taxonmatrix
prüfen Normierungszeitpunkt und Reihenfolge. Energie wird nicht zum
Mischungsgewicht. Quellenhash vor/nach Lesen gleich dem Vorpaket. Kein R-
Laufzeitvergleich: nur Formeln im dokumentierten endlichen, nichtnegativen
Eingabebereich geprüft. NA-/Null-/Dimensionsverlustpfade nicht still repariert.

**Grenze:** Die elf offenen Nachweisbündel und finale Laufbindung bleiben
unverändert offen. Dies ist Schritt 4 als isolierter technischer Ausschnitt,
keine abgeschlossene historische Fallreproduktion und keine Ersatzdatenwahl.
**Nächster unabhängiger technischer Einstieg:** Alters-/Tiefenübertragung aus
dem bereits gebundenen `src/check_ages.R` isolieren und mit synthetischen
Matrizen prüfen. Finale Listenchronologie nicht durch Mittelwertmatrix ersetzen;
keine neuen Altersmodelle oder Fossilkurven. Neue Originalbelege für einen
vollständigen Lauf bleiben notwendig, Speicherortfrage nicht wiederholen.

**Sicherung:** Basis `74999b0`, bestehender Branch in
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
kein Merge. App/CLI, Baseline, Originale, Referenzen und Monticchio-/Chronologie-
Gates erhalten. Keine Trainings, Sampler oder Fossilrekonstruktionen; Phase 1
bleibt vorrangig. Für spätere Plots jung links, alt rechts.

## Kinneret: Phase-1-Belegvertrag maschinenprüfbar, 02.11.2026

**Abgeschlossen:** [Spezifikation, Prüfer und Grenzen](kinneret_phase1_spec_2026_11_02/README.md)
führen Eingabe-, Referenz- und Nachbearbeitungsbelege zusammen. Lesender
Prüfer bestätigt 26 eingefrorene Dateien, 19 explizite Befunde und alle 50
Snapshotdateien. Vier Quellgruppen bleiben getrennt, keine finale Variante
gewählt. Elf Nachweisbündel offen; kein Fall vollständig reproduzierbar.

**Geprüft:** 20 gezielte Integritätstests bestanden. Portabler und lokaler
CLI-Aufruf ergeben erwarteten Exitcode 2 (offene Voraussetzungen). Der gezielte
lokale Hashabgleich von 40 bereits dokumentierten Pfaden bestätigt 28 Dateien
unverändert und zwölf Pfade weiterhin fehlend. Keine neue Inventur und keine
Prüfung aller globalen Abhängigkeiten. Neu auftauchende Dateien schließen
keinen Nachweis automatisch. Prüfer ist kein Runner oder App-Freigabesystem.

**Nächster Einstieg/Grenze:** Finalen Originaleingabestand und Laufzuordnung
beschaffen, beginnend mit Basis/Chronologieliste und numerischen TFs, dann an
den eingefrorenen Vertrag binden. Ohne neue Originalbelege kein historischer
Vollnachbau. Der Nutzer kennt keinen weiteren Basisdatei-Speicherort; nicht
erneut danach fragen oder pauschal suchen. Ein Nachbau mit Ersatzdaten wäre
eine gesonderte, noch nicht getroffene fachliche Entscheidung. Keine automatische
Fortsetzung mit Verbesserungen, Kalibrierung oder Phase 2.

**Sicherung:** Basis `00d7d2a`, bestehender Branch in
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
kein Merge. Originalquellen nur gelesen; Vorpakete/Snapshot und App/CLI,
Baseline, Referenzen sowie Monticchio-/Chronologie-Gates erhalten. Keine neuen
Trainings, Sampler oder Fossilrekonstruktionen. Jung links, alt rechts.

## Kinneret: Nachbearbeitungsfassungen gebunden, 02.11.2026

**Abgeschlossenes Quellenpaket:** [Ziehungsauswahl und Dichtekombination](kinneret_postprocessing_2026_11_02/README.md)
trennt Dissertationstext, Fallstudiencode, Vergleichsfassungen und aktuellen
Trimming-Einstieg. Zwei deterministische synthetische Kontrollen belegen den
möglichen Verlust der Originalindizes nach Filterung und den Unterschied
zwischen Dichtemischung und Mittel unabhängiger Zufallsgrößen. Keine Original-
ziehungen verändert und keine Klimakurven berechnet.

**Konkret belegt:** Die Rückabbildung auf `best_combs` ist im Fallstudiencode
kommentiert, in Vergleichsfassungen aktiv. Der Fallstudien-IAT-Refinementloop
ist deaktiviert; der aktuelle `mcmc_trimming.R` schaltet sogar den ganzen
IAT-Zweig ab. Fallstudiencode nutzt PDF-Mischung statt des textlich beschriebenen
Zufallsgrößenmittels; der heutige aktive Einstieg bildet eine KDE über gespeicherte
Klima-Punktwerte. Keine dieser Fassungen ist als finaler Kinneret-Lauf identifiziert.

**Weiter offen:** Alle acht explizit geprüften Kinneret-Posteriorpfade fehlen;
Original-IDs/zehn gewählte Ziehungen und numerische Enddichten sind nicht
wiederhergestellt. Basis-, TF-, Referenz-/CRU- und Laufbindungslücken bleiben
bestehen. Grafik-Bytegleichheit zum Dissertationsarchiv bestätigt nur das
Ausgabeartefakt, keine Auswahl-/Kombinationsfassung. Keine stillen Reparaturen.

**Als Nächstes:** Die gebundenen Eingabe-, Referenz- und Nachbearbeitungsbefunde
zu einer maschinenprüfbaren Phase-1-Reproduktionsspezifikation zusammenführen.
Textverfahren und Quellvarianten getrennt halten; unbekannte Originaleingaben,
Indizes, Achsen, Paket-/RNG- und Ausführungsfassung als offen ausweisen.

**Prüfung/Sicherung:** Quellenhashes vor/nach dem Lesen gleich, synthetische
Kontrollen ohne Zufall, Nachweise wiederholbar; Vorpakete und Forschungssnapshot
erhalten. Basis `edfd3f1`, bestehender Branch in
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
kein Merge. Keine R-Ausführung, IAT-Schätzung, Dichteanpassung, Trainings,
Sampler oder Fossilrekonstruktionen. App/CLI, Baseline, Originale und Monticchio-/
Chronologie-Gates erhalten; Phase 1 vorrangig. Jung links, alt rechts.

## Kinneret: Referenz-/Ankerpfad gebunden, 02.11.2026

**Abgeschlossenes Quellenpaket:** [Referenz-/Ankervertrag](kinneret_references_2026_11_02/README.md)
trennt Dissertationstext und erhaltene Verbraucher. 160 AP-Prozentwerte als
100×AP/(AP+NAP) geprüft; zwölf konkrete Ladepfade fehlen weiterhin. Source-
Auszüge, Hashes, Verzweigungen und ein gezielt gelesener späterer Speläothem-
Container sind dokumentiert; keine neue Referenz- oder Klimakurve berechnet.

**Wichtige Befunde:** Die AP-Altersmaske bleibt bei `hi_time=FALSE` wirksam.
Soreq-Auswahl 105 wird im neueren aktiven Speläothemzweig durch 170 und eine
0–220500-a-Zeitachse mit Lücke überschrieben; die ältere Vorbereitung setzt
für Kinneret 37. Keine dieser Positionen ist ohne den fehlenden Workspace als
konkrete Reihe identifiziert. MEDSTACK wird über Klassen-/Endposition gewählt.
Bei Altern ≤0 verwendet der Ankerpfad eine lokale Zeitreihe, sonst Klassen-
Klimamittel. Zeitindizes 50:115 sind ohne Zeitachse keine belegten Kalenderjahre.
Die heutigen Quellstände sind damit keine laufgebundene finale Konfiguration.

**Weiter offen:** finale Basis und TFs, ursprüngliche Isotopen-/CRU-Workspaces,
Paket-/Zeitachsenbindung, numerische Anker und tatsächliche historische
Ausführungsfassung. Der Nutzer kennt keinen weiteren Basisdatei-Speicherort;
keine wiederholte pauschale Suche oder Ersatzdatenübernahme.
**Als Nächstes:** unabhängig davon die historische Nachbearbeitung in
`mcmc_case_studies.R` gegen Dissertation und aktuellen Trimming-Einstieg binden:
Ziehungsauswahl, Prior-/IAT-/Pareto-Schritte und Dichtekombination.

**Geprüft und Sicherung:** Wiederholbare Metadaten-/Prozentprüfung, Quellen vor/
nach dem Lesen unverändert, Vorpakete und Forschungssnapshot erhalten. Fortsetzung
auf bestehendem Branch in [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
Basis `44092d3`; kein Merge. Keine R-Ausführung, Fits, Wavelets, Altersmodelle,
Sampler oder Fossilrekonstruktionen. App/CLI, Baseline, Referenzdaten und alle
Monticchio-/Chronologie-Gates bleiben erhalten; Phase 1 zuerst. Jung links, alt rechts.

## Kinneret: ältere Proben-/Taxonbasis gebunden, 02.11.2026

**Abgeschlossenes Teilpaket:** [Kinneret-Eingabebindung](kinneret_binding_2026_11_02/README.md)
mit 160-zeiligem Proben-Crosswalk, zwölf exakt gebundenen älteren Taxonspalten
(1920 Werte), identischem numerischem Chronologieblock (283520 Werte) und
drei expliziten aktuellen TF-Namens-/Poolzuweisungen. Eingaben nur gelesen;
keine neuen Altersmodelle, TF-Ziehungen oder Fossilkurven berechnet.

**Grenzen konkretisiert:** Der Einzelcontainer ist ein 35-Taxa-/3-Biomzweig.
Der ältere Sammelstand besitzt zwölf andere Taxonspalten; die jüngeren
positionsbezogenen Umbenennungen würden dort u.a. Pinus als Olea beschriften.
Die erhaltene mittlere Chronologiematrix ersetzt keine vom finalen Code
erwartete Liste mehrerer Altersraster. Gewichtsplots besitzen elf bzw. zehn
Anzeigeeinträge und überschreiben Taxonlabels; daraus folgt kein eindeutiger
finaler TF-Crosswalk. Die vollständige historische TF-Teilmenge bleibt offen.

**Quelllücke:** `core_data_additional_sites_v1.2.rdata` samt finalem Kinneret-
Eintrag oder dessen genauer Erzeuger fehlt weiter. Der Nutzer kennt laut
Antwort vom 02.11.2026 keinen weiteren Speicherort. Keine erneute pauschale
Inventur oder ungefragte Ersatzrekonstruktion. **Als Nächstes unabhängig davon:**
Kinnerets Referenz-/Ankervertrag anhand der bekannten Verbraucher und vorhandenen
Nachweise konkretisieren: AP-Maske, Soreq/MEDSTACK, Csa und konsumierte CRU-
Zeiträume. Die offene Basisdatei dabei nicht als erledigt behandeln.

**Geprüft und Sicherung:** Quellenhashes vor/nach dem Lesen gleich;
Spaltenidentitäten ohne Toleranz, Chronologie als kompletter XDR-Block,
Plotlabels und Archivgleichheit geprüft. Nachweise im Paket. Fortsetzung auf
bestehendem Branch in [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
Basis `756865b`; kein Merge. Baseline, Referenzen, vorherige Fallmatrix,
App/CLI, sieben Arten und Monticchio-/Chronologie-Gates bleiben erhalten.
Keine Trainings oder Sampler; Phase 1 bleibt vorrangig. Jung links, alt rechts.

## Lokale Dissertationsfälle: Fallmatrix abgeschlossen, 02.11.2026

**Abgeschlossen:** [Quellengebundene Fallmatrix](local_case_matrix_2026_11_02/README.md)
für sieben Fälle an sechs Standorten, Totes Meer Holozän/Glazial getrennt.
Lokale Vollfassung und Repository-Ausschnitt mit eigenen Seiten-/Abbildungsbelegen
geprüft. Ohrid ausdrücklich 1165–1365 ka; Van nicht still auf 130 ka begrenzt.
Fossil-/Chronologiekandidaten, Taxa/TFs, Klima, Referenzen, Anker, historische
Code-/Nachbearbeitungsfassungen und Originalausgaben fallweise zugeordnet.

**Ergebnis und Grenzen:** Noch kein Fall vollständig numerisch reproduzierbar.
Historische numerische TFs, geordnete Zuordnung, globale Klima-/Referenzeingaben
und genaue Laufbindung bleiben offen. Gezielte Prüfungen belegen zusätzliche
Chronologie-/Versionslücken bei En Gedi, Birkat Ram, Van, Iznik und Ohrid.
Elf von 14 Original-/Archivgrafikpaaren bytegleich; drei abweichende Stände am
Toten Meer separat erhalten. Grafiken sind keine numerischen Originalposterioren.

**Als Nächstes:** Kinneret als ersten Klärungsfall erschließen: 160 Pollenzeilen,
160 Chronologieeinträge und beide bytegebundene Archivgrafiken liegen vor.
Den historischen `core.data[[3]]`-Basiseintrag an lokale Einzelcontainer und
Erzeuger binden, Proben-/Tiefen-/Pollentyp-/TF-Reihenfolge festhalten und erst
daraus die benötigte numerische TF-Teilmenge ableiten. Keine pauschale Inventur,
Neukalibrierung oder neue Reliabilitätsansicht vorziehen.

**Geprüft und gesichert:** Explizite Quellen vor/nach dem Lesen unverändert;
getrennte PDF-Seiten-/Abbildungsprüfung, strukturierte RData-/CSV-Metadaten,
Archivhashvergleich und Dokumentationslinks geprüft. Nachweise im Matrixpaket.
Ausgangspunkt `865d155`, sauberer Checkout und identischer offener PR #77
vor Arbeitsbeginn bestätigt. Sicherung auf bestehendem Branch in
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
kein Merge. Keine Trainings, Sampler oder Fossilrekonstruktionen. Paper-Biomfall,
sieben moderne Arten, App/CLI, Baseline, Referenzdaten und Monticchio-/Chronologie-
Gates bleiben erhalten. Phase 1 fortsetzen; Plots jung links, alt rechts.

## Priorität geändert: lokale Dissertation zuerst, Verbesserungen danach, 02.11.2026

**Abgeschlossen – Planänderung auf ausdrücklichen Nutzerauftrag:**
[Verbindlicher Arbeitsplan](LOCAL_DISS_RECONSTRUCTION_PLAN.md) mit zwei Phasen.
Phase 1 bindet Originaleingaben und historischen Ablauf der lokalen
Dissertationsrekonstruktionen, reproduziert zunächst einen belegbaren Fall
und erschließt danach die weiteren lokalen Fälle. Bekannte methodische Kritik
bleibt dokumentiert; Verbesserungen werden getrennt gegen den gesicherten
historischen Referenzstand geprüft. Neue Kalibrierung und zusätzliche
Reliabilitätsansicht sind keine Voraussetzungen und keine nächsten App-Aufträge.

**Als Nächstes:** Quellengebundene Fallmatrix der lokalen Dissertationsfälle
anhand vorhandener Berichte und Belege erstellen: Daten, Chronologie, TFs,
Zuordnungen, Klima-/Referenzeingaben, Codefassung, Originalausgaben und Lücken.
Danach ersten Referenzfall nach belegter Eingabeverfügbarkeit festlegen.
Paper-Biomfall und sieben moderne Artenmodelle bleiben erhaltene Vorarbeiten;
sie sind keine abgeschlossene historische Taxonreproduktion.

**Grenzen und Sicherung:** Originaleingaben und numerische historische TFs
bleiben teilweise offen. Monticchio-/Chronologie-Gates bleiben fallbezogen
erhalten. Keine Trainings, Sampler oder Fossilfälle durch diese Planänderung.
Reine Dokumentationsänderung; Einstiege, Prioritäten und lokale Links geprüft.
Ausgangsstand `4b75ead`, Checkout sauber und mit GitHub identisch; Sicherung auf
bestehendem Branch in [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
kein Merge. Jung links, alt rechts. Frühere „Als Nächstes“-Angaben unten
beschreiben den damaligen Stand und werden durch diesen Plan nachgeordnet.


## App: äußere Foldwerte erschlossen, 02.11.2026

**Abgeschlossen:** Im Artenvergleich zwischen Übersicht und Foldwerten wechseln,
Art und Blockgröße wählen, fünf äußere Testfolds mit BA/Brier und Testzellzahlen
lesen. Beide Offlineformate und Sprachen; keine Neuberechnung oder Modellwahl.
280 Foldzeilen und 56 Gesamtzeilen an gespeicherte Prüfsummen gebunden.
Alle tatsächlichen Folds enthalten beide Klassen; hypothetisch fehlende Klassen
bleiben in Negativkontrollen ausdrücklich undefiniert. Folds sind keine
unabhängigen Wiederholungen oder fossilen Unsicherheitsintervalle.

**Geprüft:** 23 App-/Datentests, acht gezielte Offline-Browserfälle mit beiden
Sprachen, sämtliche Fold-/Gesamtwerte und Zellzahlen direkt abgeglichen.
Fehlende Dateien, Klassen und widersprüchliche Daten geprüft. Standortwechsel
und visuelle Mobil-/Desktopkontrolle bestanden. Vollständige Browserregression
mit acht Durchläufen und beiden Gestenprüfungen bestanden. Alle 50 portablen
Forschungsnachweise erhalten.
[Ergebnis, Grenzen und Nachweise](app_folds_2026_11_02/README.md).

**Sicherung:** Bestehender Branch und [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
Ausgangsstand `2f6d0eb`, kein Merge. Keine Trainings, Sampler oder Fossilfälle.
Baseline, Referenzdaten und Monticchio-/Chronologie-Gates bleiben erhalten.
**Als Nächstes:** Schema, Gewichtung und leere Bins der gespeicherten
Reliabilitätstabelle prüfen; rein diagnostische Ansicht erwägen. Keine neue
Kalibrierung; Altersplots jung links, alt rechts.


## App: Sieben-Arten-Vergleich umgesetzt, 02.11.2026

**Abgeschlossen:** Lesende Ansicht „Artenvergleich“ mit Balanced Accuracy und
Brier nebeneinander je Art, 5°-/10°-Blockwahl und beiden Blockdesigns im
ungewichteten Artenmittel. Netze ohne/mit SMOTE standardmäßig sichtbar;
Regression #28 und Prior-Diagnose ergänzbar. Experimenteller Status,
Gewichtung, Testdesign und sechs feste Quellenlinks sichtbar. Beide
Offlineformate, Deutsch/Englisch, keine neuen Fits oder Fossilläufe.

**Geprüft:** 15 App-/Datentests, acht gezielte Offline-Browserfälle mit beiden
Sprachen; 56 Original-Metrikzeilen und acht Artenmittel direkt abgeglichen.
Fehlende/veränderte Metriken sichtbar behandelt, Mobil/Desktop-Bilder geprüft.
Vollständige Browserregression (acht Fälle, zwei Gestenprüfungen) und
Standortwechsel bestanden. Alle 50 bisherigen Forschungsnachweise unverändert. [Ergebnis und Nachweise](app_species_2026_11_02/README.md).

**Sicherung:** Bestehender Branch und [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
kein Merge. Ausgangsstand `2d2fdfd`, Checkout sauber und mit GitHub identisch.
Baseline, Referenzdaten und Monticchio-/Chronologie-Gates bleiben erhalten.
**Als Nächstes:** Schema und undefinierte Werte der gespeicherten Foldscores
prüfen, dann eine rein deskriptive Detailansicht erwägen. Keine neuen Trainings,
Sampler oder Fossilrekonstruktionen; Altersplots jung links, alt rechts.


## App: Forschungsstand und Quellen umgesetzt, 02.11.2026

**Abgeschlossen:** Neue lesende Ansicht „Forschungsstand“ in Ordner- und
Ein-Datei-App, offline und Deutsch/Englisch. Fünf Bereiche trennen vorhandene,
experimentelle und fehlende Ergebnisse; Berichte sind an einen festen Commit
gebunden. Fehlende oder veränderte Nachweise werden sichtbar behandelt.
Keine neuen Modelle, Sampler oder Fossilrekonstruktionen.

**Geprüft:** 11 App-/Daten-/Buildtests bestanden; zwei normale Browserwrapper
übersprungen, durch explizite Browserprüfungen ergänzt. Acht vollständige
Browserdurchläufe plus zwei Gestenprüfungen und vier abschließende
Nachweis-/Sprachprüfungen erfolgreich. Mobil/Desktop, Hell/Dunkel, beide
Offlineformate; Bilder visuell geprüft. Bisherige Ansichten/Exporte erhalten.
Das portable Paket mit 50 Nachweisdateien bleibt unverändert. Die CI-Korrektur
`954746c` ist auf GitHub erfolgreich geprüft.
[App-Ergebnis und Nachweise](app_research_2026_11_02/README.md).

**Gesichert:** Fortsetzung im selben Branch und [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).
Abgeschlossene, geprüfte Arbeitspakete auf Nutzerwunsch jeweils hochladen;
keine automatischen Merges. Basis, Originale und wissenschaftliche Gates bewahren.
**Als Nächstes:** Vorhandene Sieben-Arten-Metriken in der App gemeinsam mit
Balanced Accuracy und Brier verständlich darstellen, ohne neue Trainings-
oder Fossilläufe. Jung links, alt rechts.


## CI-Testabhängigkeit korrigiert, 02.11.2026

Die frische GitHub-CI von [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77)
scheiterte beim Einsammeln der neuen Lesertests an fehlendem `rdata`.
Das Test-Extra enthält jetzt die lokal geprüfte Version `rdata==1.1.0`,
da die Leser deren interne Parserstrukturen verwenden. App-Laufzeitabhängigkeiten
und wissenschaftliche Ergebnisse bleiben unverändert. 67 betroffene Tests
lokal erneut bestanden; der neue GitHub-Lauf prüft beide Python-Versionen.

Die weitere Arbeit erfolgt hier im Chat; abgeschlossene, geprüfte Pakete werden
auf dem bestehenden Branch gesichert. Nächster App-Schritt bleibt die lesende
Forschungsstands-/Quellenansicht.


## GitHub-Sicherung und Claude-/App-Übergabe, 02.11.2026

Auf Nutzerauftrag auf GitHub gesichert: [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
Branch `dissertation-reconstruction`, erster Sicherungscommit `973fddb`.
Der PR bleibt für die Claude-Fortsetzung offen; `main` ist unverändert.
[Aktueller Einstieg](CLAUDE_HANDOFF_2026_11_02.md), automatisch auffindbar
auch über `CLAUDE.md`. 50 ausgewählte Nachweise/Ergebnisdateien, rund 2,8 MB,
stehen im [portablen Paket](research_snapshot_2026_11_02/README.md); persönliche
Pfade nur in Exportkopien ersetzt. Originale, Runtimes und vollständige
Samplerarchive bleiben lokal. 67 betroffene Tests erneut bestanden.

Nächster App-Schritt: vorhandene Architektur prüfen und eine lesende
Forschungsstands-/Quellenansicht mit vorhandenen, experimentellen und fehlenden
Ergebnissen ergänzen. Baseline, App/CLI und sieben Arten bleiben die Grundlage.
Die fehlenden historischen numerischen TFs und fachlichen Gates werden
dadurch nicht aufgehoben. Keine neuen Trainings, Sampler oder Fossilfälle.
Plots jung links, alt rechts. Basis weiterhin PR #30 / `7f1826c`.


## Zweiter TF-Sammelplot und Klimakoordinaten geprüft, 02.11.2026

**Abgeschlossen – Quellenarbeitspaket:** Auch der zweite Sammelplot enthält
278 durch tatsächlichen Titel und grüne Punktzahl eindeutig zugeordnete
Ohrid-Ausgaben, genau ohne Position 151 (`Picea omorika`). `Juglans sp.` bleibt
Position 110; PDF-Seitenordnung ist kein Taxonindex und kein Kinneret-Pfad.

**Neu belegt:** 961.568 Punktpaare unabhängig aus beiden PDF-Inhaltsströmen
verglichen. Sämtliche Y-Koordinaten sind exakt gleich; X weicht höchstens
0,02 PDF-Punkt ab. 23 komplette Punktwolken sind exakt gleich. 269 Achsenfassungen
sind gleich, neun minimal in X verschoben. Konturvektoren sind auf allen Seiten
nicht identisch; das beweist keine 278 verschiedenen Modelle. Keine numerischen
TFs oder exakte Identität der globalen Klimadateien daraus abgeleitet.

**Quellbindung:** Temperatur affine Standardisierung, Niederschlag
Quadratwurzel/Gamma-CDF/Normalquantil; Rücktransformation auf native Plotachsen.
300er-Niederschlagsgitter im Allgemeinen nicht äquidistant. Zeitindizes 61:90
sind ohne Zeitachse keine bewiesenen Kalenderjahre. Aktiver EMD-Schreiber und
Haupteinstieg bleiben verbunden; der exakte BA-only-Erzeuger des zweiten Plots
ist trotz Vergleich von acht Quellen weiterhin nicht identifiziert.

**Geprüft:** 31 gezielte Tests, unabhängiger Dezimalcheck und sechs vollständige
visuelle Kontrollseiten; neuer Abschluss unter `.local/tf_plot_comparison_v1/`.
Originale und bisherige lokale Änderungen erhalten. Keine erneute Modellprüfung,
Trainings, Sampler, Fossilläufe, Commits, Pushs oder Merges. GitHub bestätigt
[PR #30](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/30)
und Basis `7f1826c`. [Ergebnis und Grenzen](HISTORICAL_SECOND_TF_PLOT_RESULT.md).

**Als Nächstes:** Punktwolken gezielt gegen koordinatengebundene erhaltene
regionale Klimamittel und deren Zeitgrundlagen testen; nur neue konkrete
Quellenansätze für die numerischen EMD-Container, globale test.nc/CRU-Dateien
und BA-only-Laufbindung verfolgen. Numerische TFs, globale Zeitachse,
95.006 Rasterabweichungen und früherer Rastererzeuger bleiben offen.
Paper, sieben Arten, App/CLI, Baseline und fachliche Gates bewahren.
Jung links, alt rechts.

## Konsolenspur und 278 grafische TF-Ausgaben verbunden, 02.11.2026

**Abgeschlossen – Quellenabgleich:** Der gespeicherte 279er-Ohrid-Stand erklärt
mit Nordhalbkugel, klassenweisen 70 % und dem synthetischen SMOTE-Ausgabezweig
alle 271 Konsolen-Zeilenzahlen und acht Auslassungen vollständig. Der passende
Fünf-Variablen-Quellzweig berechnet Variablenwichtigkeiten; seine Modellspeicherung
ist auskommentiert. Heutige Titanic-Überschreibung verhindert exakte Laufbindung.

**Neu materiell belegt:** 278 Seiten im erhaltenen EMD-TF-Plotcontainer.
Jeder tatsächliche Titel und jede gespeicherte grüne Punktzahl passt eindeutig
zu einer Position des gespeicherten Ohrid-Stands. Alle Positionen außer 151
(`Picea omorika`, eine Präsenz) sind enthalten, passend zur späteren Auslassungsregel.
Juglans-Ausgabe heißt noch `Juglans sp.`. Keine PDF-Seitenordnung als Taxonindex,
keine Übertragung des Ohrid-Pfads auf Kinneret, kein Ersatz numerischer TF-Arrays.

**Geprüft:** 59 Tests und unabhängige Abschlussprüfung unter
`.local/console_provenance_v1/`. Originale und lokale Vorarbeiten erhalten;
kein Training, Sampler, neuer Fossilfall, Commit, Push oder Merge.
Basis weiterhin [PR #30](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/30) / `7f1826c`.
[Ergebnisse, Grenzen und Nachweise](HISTORICAL_CONSOLE_TF_LINK_RESULT.md).

**Als Nächstes:** zweiten erhaltenen TF-Sammelplot und Varianten gezielt
gegen den belegten Plotstand prüfen; Klimakoordinaten und tatsächliche numerische
`mixed_partial_arts_emd`-Ausgaben weiter verfolgen. Globale Klimaeingaben,
Zeitachse und regionaler Rastererzeuger bleiben offen. Paper, sieben Arten,
App/CLI, Baseline und fachliche Gates bewahren. Jung links, alt rechts.

## Modell-Lesergrenzen abgeschlossen, Quellen fortgesetzt, 01.11.2026

**Abgeschlossen:** Neun Leserfehler und vier Zeitgrenzen gezielt aufgelöst:
13 verschiedene Byteinhalte / 17 Dateialiasen jetzt strukturell gelesen.
65 frühere Einzelbefunde unverändert übernommen; damit im bisherigen Umfang
109 Dateien / 78 Byteinhalte vollständig strukturell lesbar. Containerbezogen
167 caret-Fits: sechs binäre, 161 Mehrklassenfits. Keine Gleichsetzung mit
279 unabhängigen Taxonfunktionen und keine validierten R-Vorhersagen.

**Neu:** Ein gespeicherter GBM-Fit hat eine zusätzliche Taxonklasse gegenüber
den acht anderen Fits desselben Containers; Klassenlisten nicht harmonisiert.
Regionale RData in fünf Zusatzkopien bytegleich. Ein Konsolenprotokoll enthält
Positionen 1–279 und bei 271 Positionen Meldungen für vier Modelle, aber keine
vollständige Taxon-/Klima-/Laufbindung. Der passende heutige Quellkandidat enthält
eine aktive Titanic-Eingabeüberschreibung; daher keine exakte Laufzuordnung.

**Geprüft:** 36 gezielte Tests; Nachweise und unabhängiger Abschluss unter
`.local/provenance_reader_followup_v1/`. Originale und vorherige lokale Änderungen
erhalten. Keine Fits, Sampler, Fossilrekonstruktionen oder Git-Veröffentlichungen.
Basis unverändert nach [PR #30](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/30).
[Ergebnis und Grenzen](HISTORICAL_MODEL_READER_RESULT.md).

**Als Nächstes:** konkrete Eingabe-/Ausgabebindung der Konsolenspur,
frühere Fünf-Klimavariablen-Quellfassung, globale Klima-/TF-Dateien und
vorgelagerten regionalen Rasterexport suchen. Leserreparatur abgeschlossen;
Gesamtprovenienz offen. Paper, Sieben-Arten-Ergebnisse, App/CLI, Baseline und
Monticchio-/Chronologie-Gates bewahren. Jung links, alt rechts.

## Quellen- und Laufprovenienz fortgesetzt, 01.11.2026

**Abgeschlossen – Quellenarbeitspaket:** Die 95.006 regionalen Rasterabweichungen
liegen bereits zwischen regionalem RData und der NetCDF-/eurasischen Fassung;
im regionalen Fenster entspricht die globale Präsenz-Einfügung exakt der
eurasischen Eingabe. Der frühere regionale Export bleibt als Ursprungsschritt
offen. Global sind zusätzlich 74 Randspaltenverluste und zwei durch `== 1`
ausgeschlossene Wert-2-Zellen eingegrenzt; keine Rasterkorrektur.
Ein tatsächlich gespeicherter 279er-Ohrid-Verbreitungsstand stimmt in allen
Ebenen mit dem dokumentierten Löschpfad überein, vor der Juglans-Umbenennung.

109 Modellkandidaten / 78 verschiedene Byteinhalte geprüft: 65 vollständig
strukturell gelesen, 9 Leserfehler, 4 Zeitgrenzen. Gefunden sind sechs
binäre QDA-Teilmodelle und 15 QDA-Paarmodelle, mit Wintertemperatur und einer
gemeinsamen unknown-Klasse; keine Gleichsetzung mit 279 Artenfunktionen.
Erhaltene Klimamittel belegen unterschiedliche Zeitgrundlagen 1901–2015 bzw.
1961–1990. Die globalen test.nc-Eingaben und konkrete Laufprovenienz bleiben offen.

**Geprüft:** 21 gezielte Tests; Abschlussnachweis unter
`.local/provenance_continuation_v1/final_verification.json`. Originale und
vorhandene lokale Änderungen erhalten; keine Fits, Sampler, neuen Fossilfälle
oder Git-Veröffentlichungen. PR #30 / Basis `7f1826c` unverändert.
[Ergebnis, Grenzen und nächster Einstieg](HISTORICAL_PROVENANCE_RESULT.md).
**Als Nächstes:** gezielt verbleibende Modell-Lesergrenzen, regionalen
Rastererzeuger und globale Klima-/TF-Laufquellen klären. Paper, Sieben-Arten-
Ergebnisse, Monticchio-/Chronologie-Gates und App/CLI-Baseline bewahren.
Jung links, alt rechts.

## Lokaler Einstieg 01.11.2026

Neue lokale Bestandsaufnahme und sicherer Arbeitsbereich auf Branch
`dissertation-reconstruction`, Basis `main` bei `7f1826c` nach geprüftem Merge
von [PR #30](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/30).
Schritt 1 der Startdatei ausgeführt, Quellenabgleich begonnen. Originalbestand
unverändert; Katalog und persönliche Pfadkonfiguration lokal ausgeschlossen.
Fortsetzung am selben Tag: portablen Compiler lokal eingerichtet, Kern und
Original separat gebaut. Lokaler Originalvergleich über 20.000 Iterationen
bitgleich; archivierter Vergleich mit anderer C++-Laufzeit weiterhin fehlgeschlagen.
Diagnostischer und vollständiger Originallauf sowie korrigierter Vierkettenlauf
abgeschlossen. Beide Posterior-Sampler vollständig bitgleich wiederholt;
48 CSV-Felder unabhängig nachgerechnet, 362 Klimadiagnosen bestanden.
Alle 43 korrigierten Parameter erfüllen R̂ ≤1,01 und Bulk-ESS ≥400.
Zwei bestehende Tests bleiben als Laufzeit-/Kurzkettenbefunde dokumentiert.
Konkretes Ergebnis und Grenzen: [LOCAL_REFERENCE_RESULT.md](LOCAL_REFERENCE_RESULT.md).
Keine neue fachliche Freigabe; vor weiteren Fossilfällen Bewertung erforderlich.
Darstellungsvorgabe: Alter immer von jung (links) nach alt (rechts).
Aktueller Vergleichsbericht entsprechend aktualisiert; keine neuen Samplerläufe.
Der Paperfall ist keine vollständige Reproduktion der taxonbasierten Dissertation.
Befunde, Umgebungsprüfung und nächster Plan:
[LOCAL_WORKSPACE.md](LOCAL_WORKSPACE.md). Kein neuer PR oder Upload beauftragt.

## Quellenfortsetzung 01.11.2026: Taxonverfahren

**Abgeschlossen:** lokalen 275-Ebenen-Verbreitungsblock anhand des ausdrücklichen
R-Verbrauchers und identischer Fehlwertmasken dem europäischen 0,5°-Klimagitter
zugeordnet; Achsen aufsteigend Länge/Breite. 9.718 verwendbare Zellen je Ebene,
16 Ebenen ohne Präsenz, 233 verschiedene binäre Raster. Namen nicht eindeutig:
272/271 unterschiedliche Einträge in den beiden Spalten; keine Zusammenführung.

**Neu belegt:** `all_trafo.R` schreibt die vom Haupteinstieg erwartete historische
TF-Datei. Dieser Quellstand enthält einen Rücksatz auf die vollständige
Trainingstabelle nach der Hold-out-Aufteilung und eine zusätzliche EMD-Rangwahl.
Das ist ein statischer Befund, kein Beweis für die Ausführung in jedem
Dissertationslauf. Verschiedene Regeln zur Art-/Pollentyp-Zusammenführung
von der späteren gewichteten Taxonkombination getrennt.

**Weiter offen:** globale Eingaben, historische TF-Container, geordnete Taxonliste
und Laufprovenienz. Keine exakten Dateinamenstreffer für die gesuchten fehlenden
RData-Abhängigkeiten in den neun lokalen ZIP-Verzeichnissen. Acht Auditprüfungen
bestanden; 1.783 Originaldateien laut Metadatenkatalog und elf gehashte Quellen
unverändert. Keine Fits oder Sampler wiederholt; keine neuen Fossilfälle.
[Nachweis und Fortsetzung](LOCAL_TAXON_METHOD_AUDIT.md). Ergebnisse aus PR #29,
Paperreferenz, App und bestehende fachliche Gates unverändert. Kein neuer PR.

## Zusatzquelle my_reconst geprüft, 01.11.2026

Nutzerhinweis auf weiteren Originalordner abgearbeitet: 2.687 Dateien inventarisiert.
**Gefunden:** globale Verbreitungscontainer Kinneret (275 Ebenen) und Ohrid
(283 Ebenen), mit eigenen 720×360-Koordinaten. Ohrid enthält acht zusätzliche
Relikttaxa. Die 275 gemeinsamen globalen Ebenen sind exakt gleich. Der explizit
auf Ohrid bezogene historische Lösch-/Umbenennungspfad liefert 279 Positionen;
dieser Pfad ist nicht auf Kinneret übertragbar. Fertige historische TFs weiterhin offen.

**Wichtige Grenze:** Globaler Ausschnitt und bisheriger regionaler 275er-Block
haben 95.006 verschiedene binäre Zellen bei identischen Fehlwertmasken; nur
27 Ebenen sind exakt gleich. Keine Datenkorrektur oder Substitution. Zwei
kleine historische Ohrid-Modelle sind jeweils ein QDA-Mehrklassenfit für sechs
Taxa plus unknown, keine Sammlung von 279 binären Artenfunktionen.

**Geprüft:** sieben Lesertests und zehn Quellenprüfungen bestanden. Große
Raster nur ebenenweise gelesen, Originalbestände unverändert. Kein Training,
Sampler oder neuer Fossilfall. Die konkreten TF-/test.nc-Dateinamen fehlen
weiter im neuen Ordner und seinen neun ZIP-Verzeichnissen.
[Ergebnis und aktuelle Nachweise](MY_RECONST_SOURCE_RESULT.md).
Bestehende Ergebnisse aus PR #29/#30, App und fachliche Gates bleiben erhalten.

## Bisheriger wissenschaftlicher Stand

Stand 30.10.2026. Vor Fortsetzung aktuellen PR-/Merge-Stand prüfen.
Ziel: überprüfbare App für taxonbasierte Multiproxy-Klimarekonstruktionen
europäischer Fundstellen, mit belegter Kalibrierung und Unsicherheiten.

| Bereich | Stand |
| --- | --- |
| Moderne Analog-Baseline #18 | Fertig; #18 auf main gemergt. Unveränderte Referenz f9acc16. |
| Erste sieben Artenfunktionen #28 | Fertiger experimenteller Trainingsstand aus Meusel/BfN + WorldClim 2.1; feste Referenz. |
| SMOTE + einfaches Netz #29 | **Berechnet und unabhängig geprüft:** sieben Arten, identische äußere räumliche Tests; deutlich bessere Balanced Accuracy, zugleich schlechtere rohe SMOTE-Wahrscheinlichkeiten im Brier. |
| Historische 279 Funktionen | Globale Verbreitungsbasis und konkreter Ohrid-Pfad 283→279 gefunden; fertige TFs und bitgleiche historische R-/caret-Reproduktion weiterhin offen. |
| Taxonverfahren insgesamt | Weitere Taxa, Fossiltyp-Zuordnung, unabhängige Wahrscheinlichkeitskalibrierung und Unsicherheiten offen. |
| Monticchio-P1 | Weiter offen: physischer LGM90D→MON16-Crosswalk samt Korrelationsunsicherheit fehlt. |
| Proben und Quellen aus #26/#27 | Unverändert: 28 numerische Kandidaten, 41 native Nichttreffer, 47 offene Herkunftsfragen. |
| Fossilrekonstruktion / Multiproxy-App | Kein fertiger Gesamtdurchlauf, keine neue fossile Klimakurve und keine P2-Freigabe. Claude betreut CLI/App; Integration #19 in den gemeinsamen Stand übernommen. |

**Fertig:** [SMOTE-/Netzvergleich](./smote_nnet_2026_10_30/README.md),
[PR #29](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/29),
ursprünglicher Ergebnisbranch `codex/netzel-smote-nnet-comparison`.
Die Ergebnisse aus #18–#29 werden auf ausdrücklichen Nutzerauftrag gemeinsam
auf `main` gesichert ([Integrations-PR #30](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/30)); [gemeinsame Übergabe](./HANDOFF.md) und
[Integrationsnachweis](./consolidation_2026_10_30/README.md). Vorabdesign veröffentlicht
in `d47129c6b8be8bd5e73701f6137ffb196dab9d91`, bevor neuronale Fits begannen.

Artenmittel der Balanced Accuracy bei Schwelle 0,5: ohne SMOTE 71,64/69,20 %,
mit SMOTE 90,41/89,74 % (5°/10°-Blöcke, 100-km-Puffer). Gewinn in allen 14
Art×Block-Fällen. Flächengewichteter Brier dagegen im Artenmittel 0,03975/0,04438
ohne SMOTE und 0,07747/0,08245 mit SMOTE: Verschlechterung in allen 14 Fällen.
Die vorab vorgesehene Prior-Odds-Diagnose hilft, garantiert aber keine Kalibrierung
und schlägt das Netz ohne SMOTE im Brier nur in zwei von 14 Fällen.

14 vollständige neue Artennetze und alle Foldparameter gespeichert.
198.680 Art-Zell-Labels, 397.360 Testzeilen mit 1.589.440 Wahrscheinlichkeiten,
1.274.548 synthetische Elternpaare und sämtliche 462 Optimierungsstarts separat
geprüft. Zwei vollständige Läufe in allen zehn Ergebnisdateien bytegleich.
Sechs neue Kerntests bestanden. Gemeinsame GIS-/Numerikbibliotheken benannt.

**Aktuell:** Die Klassifikationsverbesserung aus der Dissertation wurde auf den
neuen sieben Arten nachgebildet. Experimentelle Arealindikatoren sind noch
keine validierten fossilen Auftretenswahrscheinlichkeiten. Keine automatische
Variantenübernahme oder Pollentyp-Zuordnung. Die #28-Referenz wird ausdrücklich
am gesperrten Commit `8f02794434f561f5b30228360247935f3c025cc8` bezogen; die ursprünglichen getrennten
PR-Branches enthielten nicht automatisch alle Vorarbeiten. Der gemeinsame
Integrationsstand enthält jetzt alle zwölf PR-Historien.

**Als Nächstes:** neue trainingsinterne Wahrscheinlichkeitskalibrierung mit inneren
räumlichen Teilungen und unabhängiger äußerer Bewertung vorab festlegen; weitere
Arten/Areale und eine belegte Typ–Art-/Gattungs-Zuordnung bearbeiten. Äußere
Testergebnisse dieses PR nicht zur weiteren Optimierung verwenden. Moderne
Kalibrierung kann parallel zur weiterhin offenen Monticchio-Quellenfrage laufen.

Die frühere Merge-Sperre wurde für die Zusammenführung #18–#29 durch den
aktuellen ausdrücklichen Nutzerauftrag aufgehoben. Das ändert keine fachlichen
Freigaben: keine stillen Quellenkorrekturen, Pilotfreigaben oder Autorennachrichten.
Wissenschaftliche Berichte, Quellenregister, Modellparameter und Ergebniswerte
werden unverändert erhalten. Der gemeinsame Einstieg ist [HANDOFF.md](./HANDOFF.md).
