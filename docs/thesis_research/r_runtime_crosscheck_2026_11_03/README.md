# Laufzeitbelege: Kinneret-Slot, R-Fassung, Paketanbieter und R-3.5.0-Gegenprobe

Stand 03.11.2026, Ausgangscommit `3104f3c` (PR-Kopf, App-Stand 6), bestehender Branch
`dissertation-reconstruction`, [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).

**Auftrag im Chat vom 03.11.2026:**

1. Kinnerets Index in `main_v1.2.R` aus den Daten bestimmen.
2. R- und Paketfassungen selbst suchen.
3. Paketquellen aus dem CRAN-Archiv lesen (ausdrücklich freigegeben).
4. Die Module mit R gegen den Originalcode prüfen (R-Läufe ausdrücklich freigegeben).

Die bisherige Regel „keine R-Läufe“ ist damit für **R-Gegenproben auf Kunstfixtures** aufgehoben. Weiter gilt:
keine Sampler, keine Trainings, keine Altersmodelle, keine Fossilrekonstruktionen. Die Originale in `diss` wurden
nur gelesen. Noch kein vollständiger Originalfall.

## 1. Kinneret ist Slot 3 des Zusatzorte-Zweigs

[bind_sites.py](bind_sites.py) → [site_binding.json](site_binding.json). Geprüft an den hashgebundenen Originalen
`main_v1.2.R` und `mcmc_prepare.R`:

- `add.sites <- T` (`main_v1.2.R:73`) ist die einzige Einstellung. Der Zweig lädt das fehlende
  `in/core_data_additional_sites_v1.2.rdata` (107) und überschreibt die Slots aus `my_reconst`.
- Slots: 1 Prespa, 2 En Gedi, **3 Kinneret** (`my_kinneret/.../pollen_counts_epd_taxanomy.csv`, 143–154),
  4 Birkat Ram (alt), 5 Iznik, 6 Ohrid, 7 Dead Sea, 8 Birkat Ram (Komposit), 9 Lake Van.
- `stations <- c(1:length(core.data))[-4]` (615): Slot 4 wird übersprungen, ebenso `station_id == 469` (654).
  Die Gesamtlänge hängt am fehlenden Container und ist mindestens 9.
- Im EPD-Zweig (`add.sites <- F`) laufen die EPD-Einträge in Blöcken zu 100; die gebundene Datei nutzt diesen Zweig
  nicht.
- Kinneret-Schalter in `mcmc_prepare.R` über `sitename == "Lake Kinneret"`:
  - 790: `which.site <- 105`;
  - 881: keine Behandlung des menschlichen Einflusses;
  - 1145: Speläothem- und Meeres-/Eiskernquellen auf 0;
  - 1209: Kinneret vom Levante-Zweig ausgenommen;
  - **1248: `tann <- FALSE`** (oberste Ebene, gilt also in jedem Kinneret-Lauf).

  790, 881 und 1145 liegen in umschließenden Blöcken, deren Bedingungen hier nicht aufgelöst werden. Was `tann`
  im Lauf bewirkt, ist noch nicht gebunden.

Die ältere Bindung hatte Kinneret ebenfalls in Slot 3 gefunden
([kinneret_binding](../kinneret_binding_2026_11_02/README.md)). Gleiche Slotnummern beweisen weiterhin keine
Laufidentität.

## 2. R-Fassung: Eingaben stammen aus R 3.5.0

[collect_rdata_writers.py](collect_rdata_writers.py) liest nur die Serialisierungsköpfe aller 657
RData-/RDS-Container in `diss` → [rdata_writers.json](rdata_writers.json).

- 546 Container wurden von R 3.5.0 geschrieben.
- Alle 13 Eingaben in `local_reconstr_all_epd/in` und 31 der 32 `my_kinneret`-Container stammen aus **R 3.5.0**.
  Ihre Dateizeiten liegen zwischen 31.10.2019 und 17.07.2020. Ein älterer Kinneret-Container stammt aus R 2.11.0.
- `sessionInfo`, `renv.lock`, `.Rhistory` oder Paketversionsdateien gibt es in `diss` nicht. Das gilt auch für
  die R-Skripte, Texte und Protokolle; die Suche nach Versionsangaben fand nur die FAQ einer mitgelieferten
  rbacon-Kopie.
- Damit ist R 3.5.0 für die **Eingaben** belegt. Die R-Fassung des **Laufs** bleibt ungebunden: Die Quelldateien
  wurden bis 07.07.2023 geändert.

## 3. Paketanbieter (CRAN-Archiv und GitHub-Spiegel `cran/*`)

**Fenster:** Fassungen, die zwischen dem Erscheinen von R 3.5.0 (23.04.2018) und der letzten Quelländerung
(07.07.2023) verfügbar waren.

**Vorgehen:**

- [collect_cran_exports.py](collect_cran_exports.py) liest Daten aus den CRAN-Listen und NAMESPACE/DESCRIPTION je
  Fassung aus dem Spiegel. Das umfasst alle 55 angehängten Pakete samt `Depends`, insgesamt 82 Pakete.
- [analyse_providers.py](analyse_providers.py) bildet daraus den Suchpfad in der gebundenen `library()`-Reihenfolge.
- [resolve_unresolved.py](resolve_unresolved.py) prüft Pakete mit `exportPattern` vor dem Anbieter an den
  CRAN-Tarballs selbst. Die Tarballs werden nur im Speicher gelesen, nichts wird entpackt oder ausgeführt.

Ergebnis: [provider_analysis.json](provider_analysis.json).

| Name | Exportierende Pakete im Fenster | Vorne im Suchpfad | Davor exportiert niemand |
| --- | --- | --- | --- |
| `rdirichlet` | LaplacesDemon, extraDistr, gtools, SBMSplitMerge | **LaplacesDemon** (`main_v1.2.R:38`) | ja (11 Pakete mit `exportPattern` am Tarball geprüft) |
| `ddirichlet` | dieselben vier | **LaplacesDemon** | ja |
| `weightedMedian` | nur matrixStats | **matrixStats** (`main_v1.2.R:27`) | ja |

**LaplacesDemon** ([compare_dirichlet.py](compare_dirichlet.py) → [dirichlet_versions.json](dirichlet_versions.json)):

- In allen vier Fassungen des Fensters (16.1.0, 16.1.1, 16.1.4, 16.1.6) sind `rdirichlet` und `ddirichlet`
  **textgleich**.
- `rdirichlet` zieht mit `rgamma` und bleibt damit ein Zufallsschritt.
- `ddirichlet` ist deterministisch (`lgamma`, `log`, `exp`).
- Quellbefund im Paket, nicht korrigiert: `if(any(rowSums(x) != 1)) x / rowSums(x)` weist das Ergebnis nicht zu,
  normiert also nicht.

**matrixStats** ([compare_weighted_median.py](compare_weighted_median.py), [diff_weighted_median.py](diff_weighted_median.py)
→ [weighted_median_versions.json](weighted_median_versions.json), [Diffs](weighted_median_diffs.txt)):

- 13 Fassungen mit gleicher Signatur, aber sechs verschiedenen Quelltextständen.
- In den `weightedMedian`-Dateien ändern sich nur:
  - die Dokumentation;
  - logische `x` (0.58.0);
  - der Umbau der Indexübergabe (0.60.1);
  - eine Kennung für NA-Indizes (1.0.0).
- Für den Originalaufruf `weightedMedian(x, w)` mit numerischem `x` und `idxs = NULL` zeigt sich dort keine Änderung
  des Rechenwegs. Gemeinsame Hilfsdateien (Makros, Sortierung) sind **noch nicht** verglichen.

**Grenze:** Das ist eine Bindung **unter der Fensterannahme**. Die exakten Paketfassungen des Laufs bleiben
ungebunden. Annahme wie bisher: Paketfassungen außerhalb des Fensters und nicht angehängte Namensräume spielen
keine Rolle.

## 4. R 3.5.0: wörtliche Originalzeilen gegen die Module

**Einrichtung:**

- R 3.5.0 für Windows aus dem CRAN-Archiv, MD5 laut CRAN-`md5sum.txt` geprüft, Authenticode gültig.
- Installation nur für den aktuellen Nutzer nach `.local/R/R-3.5.0`, ohne Verknüpfungen und Aufgaben. Gefunden
  wurden keine R-core- oder Deinstallations-Registryeinträge.
- 64 bit, `long double` 16 Byte, keine Zusatzpakete.

**Prüfung:** [run_crosscheck.py](run_crosscheck.py) prüft die Originale per Hash. Dann wertet
[crosscheck.R](crosscheck.R) die Zeilen **wörtlich aus der Originaldatei** in einer frischen Umgebung aus:

- `mcmc_conduction.R:53–79` (Sampler-Vorbereitung, 8 Kunstfälle);
- `mcmc_functions_v1.6.R:1276–1280` (Kern der Gewichtsstufe, 9 Fälle);
- `mcmc_prepare.R:1270–1281` (Gamma-Raster, 6 Fälle mit Obergrenze 50000);
- `mcmc_prepare.R:1681–1685` (Taxonzahl-Priors, n = 2…60).

Ein- und Ausgaben laufen als C99-Hex-Doubles; NA, NaN, ±Inf und das Vorzeichen der Null bleiben unterscheidbar.

Ergebnis ([crosscheck_result.json](crosscheck_result.json)):

- **389 Prüfungen, alle bitgleich** mit den Python-Modulen.
- Auch alle **14 als akkumulatorabhängig markierten** Werte (`colMeans`, `rowSums`) sind auf dieser Plattform
  gleich. Das gilt für R 3.5.0/Windows x86_64 und ist keine Aussage über den historischen Rechner.
- NA bleibt NA, `-0` entsteht wie erwartet, `3·d(1/3)` rundet zur geraden 1.

Die übrigen sechs Module (Alters-/Tiefenübertragung, Altersauswahl, End-PDF, CDF-Abweichung, TF-Vorbereitung,
Taxon-Pruning) sind noch nicht gegen R geprüft.

## Prüfen

```powershell
.venv/Scripts/python.exe docs/thesis_research/r_runtime_crosscheck_2026_11_03/bind_sites.py
.venv/Scripts/python.exe docs/thesis_research/r_runtime_crosscheck_2026_11_03/run_crosscheck.py .local/R/R-3.5.0/bin/x64/Rscript.exe
.venv/Scripts/python.exe -I docs/thesis_research/r_runtime_crosscheck_2026_11_03/collect_cran_exports.py <cache> <cran_exports.json>
.venv/Scripts/python.exe -I docs/thesis_research/r_runtime_crosscheck_2026_11_03/analyse_providers.py <cache> <cran_exports.json>
.venv/Scripts/python.exe -I docs/thesis_research/r_runtime_crosscheck_2026_11_03/resolve_unresolved.py <cache> <cran_exports.json>
.venv/Scripts/python.exe -I docs/thesis_research/r_runtime_crosscheck_2026_11_03/compare_dirichlet.py <cache> <cran_exports.json>
.venv/Scripts/python.exe -I docs/thesis_research/r_runtime_crosscheck_2026_11_03/compare_weighted_median.py <cache> <cran_exports.json>
.venv/Scripts/python.exe -m pytest tests/test_r_runtime_crosscheck.py <die zwölf historischen Testdateien> -q
```

- Zwischenspeicher und Downloads liegen unversioniert unter `.local/evidence_r_env_2026_11_03/`.
- `collect_rdata_writers.py` braucht den `diss`-Pfad.
- Die Tests brauchen weder R noch Netz.

**Ergebnis:**

- 13 neue Belegtests; mit den 689 bisherigen **702 bestanden**.
- Beide Phase-1-Prüfer weiter Exit 2.
- Nicht ausgeführt: die Gesamtsuite, `verify_packet.py` und `tools/verify_*`.

Unterwegs korrigiert:

- Die erste Versionsliste erkannte Paketnamen auch als Endung längerer Namen (`cladoRcpp` → „Rcpp 0.15.1“). Nach
  dem Einfügen einer Wortgrenze wurde die ganze Kette neu gerechnet.
- Downloads werden wegen CRAN-Drosselung wiederholt.

## Grenzen

- Kein Originalfall, keine Originalzahlen. Die R-Gegenprobe nutzt nur Kunstfixtures.
- R-Laufzeitgleichheit gilt nur für R 3.5.0 auf diesem Rechner.
- Paketanbieter sind nur innerhalb des Fensters gebunden, die Fassungen des Laufs nicht.
- Alle elf Nachweisbündel bleiben offen; keine Variante gewählt; keine App-Änderung in diesem Paket.
