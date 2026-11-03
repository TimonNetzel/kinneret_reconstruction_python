# Dirichlet-Quotient der Gewichtsstufe (`mcmc_functions_v1.6.R:1269–1274`)

Stand 03.11.2026, Ausgangscommit `c0cdf81` (PR-Kopf), bestehender Branch `dissertation-reconstruction`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).
Auftrag: Nutzerentscheidung „Ja, ddirichlet“ nach der
[Übergabe nach der R-Gegenprobe](../CLAUDE_NEXT_STEP_AFTER_R_CROSSCHECK_2026_11_03.md).
Elfter begrenzter Rechenschritt: **expliziter Quotient, keine Dirichlet-Ziehung, kein Samplerlauf.** Noch kein
vollständiger Originalfall.

## Beleg zuerst

[bind_source.py](bind_source.py) → [Bindung](source_binding.json), [Auszüge](source_excerpt.json),
[Determinismusprüfung](determinism_audit.json):

- 1269, 1270, 1272, 1273 und 1274 liegen auf Tiefe 1 von `parameter_stage_weights` und laufen nach jeder Ziehung
  unbedingt.
- `num_taxa` ist global. Der Vorschlagsquotient 1270 enthält `dirichlet_spread` nicht; die auskommentierten
  Alternativen 1263–1265 bleiben unberührt.
- Leser von `parameter_ratio`: `mcmc_conduction.R` 97 (Start 1), 151 (Speicherung) und 302
  (Akzeptanzwahrscheinlichkeit, verglichen mit `runif` in 304).
- **Paketsemantik gebunden**, aber nur unter der Fensterannahme 2018-04-23…2023-07-07: `ddirichlet` stammt aus
  LaplacesDemon, das im Suchpfad vorne liegt. Der Text ist in 16.1.0–16.1.6 gleich
  ([Laufzeitbelege](../r_runtime_crosscheck_2026_11_03/README.md)) und wird hier gegen die aufgezeichnete Fassung
  geprüft.

## Implementierter Umfang

Implementierung: [historical_dirichlet_ratio.py](../../../tools/historical_dirichlet_ratio.py),
Tests: [test_historical_dirichlet_ratio.py](../../../tests/test_historical_dirichlet_ratio.py).

**`ddirichlet` wörtlich nach LaplacesDemon:**

- `rbind(x)` und `matrix(alpha, 1, k, byrow=TRUE)`.
- `if(any(rowSums(x) != 1)) x / rowSums(x)` wird ausgewertet, **aber nichts zugewiesen**. Die Gewichte werden also
  nicht normiert. Das Kennzeichen `normalisation_skipped` macht das sichtbar.
- `x < 0` und `alpha <= 0` führen zum Abbruch.
- `dens = (lgamma(rowSums(alpha)) − rowSums(lgamma(alpha))) + rowSums((alpha−1)·log(x))`, dann `exp`.

**Quellzeilen 1269–1274:**

- `weights_ratio` mit `rep(0.5, num_taxa)`;
- `proposal_ratio = ddirichlet(old, new) / ddirichlet(new, old)`;
- das Produkt beider Quotienten;
- `is.na` → 0: NA und NaN werden 0, `Inf` bleibt.

**Genauigkeit:**

- Jede Double-Operation der Quelle ist hier eine IEEE-Operation.
- `lgamma`, `log` und `exp` werden mit 80-stelliger `Decimal`-Zwischenrechnung einmal korrekt gerundet. CPythons
  `math.lgamma` liegt z. B. bei `lgamma(3)` 3 ulp neben `log(2)` und wäre damit untauglich.
- `rowSums` ist die korrekt gerundete exakte Summe mit Akkumulatorkennzeichen.

**Abgelehnt statt emuliert:**

- ungleiche Längen oder Länge ≠ `num_taxa` (in R nicht konforme Matrizen);
- weniger als zwei Gewichte;
- Gewichte, die 0, negativ, nicht endlich, NA oder nicht numerisch sind (die Schleife 1249 verlangt
  `min > 0`, `ddirichlet` bricht bei `alpha <= 0` ab, `log(0)` wird nicht emuliert).

## Unabhängige Referenz, Handwerte, Gegenbeispiele

- [Fixture](synthetic_fixture.json) mit 8 Quotienten- und 3 Dichtefällen ([Erzeuger](make_fixture.py)).
- [Referenz](precise_reference.py): `Decimal` mit `lgamma` nach Spouge (statt Stirling), π nach Gauss-Legendre (statt
  Machin), ohne Modulimport ([Referenzwerte](reference_expectations.json)).
- [Prüfskript](check_dirichlet_ratio.py) → [Ergebnisse](synthetic_result.json),
  [Arithmetik-/Gegenbeleg](arithmetic_check.json).

**Ergebnisse:**

- Das Modul stimmt in **allen** Fällen bitgleich mit der Referenzpipeline überein, einschließlich `Inf`, NaN und der
  Ersetzung in Zeile 1273.
- Handwerte:
  - gleiche Gewichte → alle Quotienten genau 1;
  - 2 Taxa (1/2, 1/2) → (1/4, 3/4): `weights_ratio` = 2/√3 und `parameter_ratio` = 1/√2 auf 0 ulp,
    `proposal_ratio` = √6/4 auf 1 ulp;
  - Dichten 2/π, 9/8 und 6 auf 0 ulp.
- Kunstrand mit subnormalen Gewichten: `weights_ratio = Inf`, `proposal_ratio = 0`, Produkt NaN → **0**.
  `Inf` allein bleibt stehen (`is.na(Inf)` ist FALSE).

**Gegenbeispiele** (keine Kandidatenvarianten):

- Die auskommentierte Fassung mit `dirichlet_spread` im Vorschlag (1265) weicht in 7 von 8 Fällen ab; nur bei gleichen
  Gewichten nicht.
- Normierte Gewichte, als hätte LaplacesDemon zugewiesen, weichen im ungenormten Fall ab.
- Ein symmetrischer Vorschlag (Quotient 1) weicht in 7 von 8 Fällen ab.
- Mit `math.lgamma` wichen 1–4 Dichten je Fall ab.

## R 3.5.0 mit LaplacesDemon 16.1.6

[run_r_crosscheck.py](run_r_crosscheck.py) und [r_crosscheck.R](r_crosscheck.R) werten **1269–1274 wörtlich aus der
Originaldatei** aus (Hash geprüft). LaplacesDemon 16.1.6 kommt aus dem CRAN-Archiv (MIT, Tarball-Hash wie
aufgezeichnet, reines R) und ist in `.local/R/library_ld` installiert; `ddirichlet` stammt aus dem Namensraum
LaplacesDemon. Ergebnis: [r_crosscheck_result.json](r_crosscheck_result.json).

- **62 Werte, 59 bitgleich**, alle Sonderwerte gleich.
- Höchstens **4 ulp** Abweichung, nur im Fall `inexact_weights`.
- Ursache, an den Zwischenschritten belegt: R 3.5.0 liefert `lgamma(0.4)` = `0x1.97e627a8f4a3bp-1`, der korrekt
  gerundete Wert ist `…a3ap-1`. Das bestätigen Modul und unabhängige Spouge-Referenz. Alle `log`-, Produkt- und
  Summenwerte sind dort bitgleich; die Auslöschung in `dens` vergrößert den 1-ulp-Unterschied auf 4 ulp.
- Das ist ein Messbefund zur `lgamma`-Fassung von R (nmath), kein Fehler des Nachbaus. Er bleibt markiert. Für eine
  bitgleiche Nachbildung müsste R's `lgamma` selbst nachgebaut werden; das ist nicht geschehen.

## Prüfen

```powershell
.venv/Scripts/python.exe docs/thesis_research/literal_dirichlet_ratio_2026_11_03/bind_source.py
.venv/Scripts/python.exe docs/thesis_research/literal_dirichlet_ratio_2026_11_03/make_fixture.py
.venv/Scripts/python.exe docs/thesis_research/literal_dirichlet_ratio_2026_11_03/check_dirichlet_ratio.py
.venv/Scripts/python.exe docs/thesis_research/literal_dirichlet_ratio_2026_11_03/run_r_crosscheck.py .local/R/R-3.5.0/bin/x64/Rscript.exe .local/R/library_ld
.venv/Scripts/python.exe -m pytest tests/test_historical_dirichlet_ratio.py tests/test_r_runtime_crosscheck.py <die zwölf bisherigen Testdateien> -q
```

**Ergebnis:**

- **50 neue Tests**; mit den bisherigen 702 zusammen **752 bestanden**.
- Beide Phase-1-Prüfer weiter Exit 2.
- Nicht ausgeführt: die Gesamtsuite, `verify_packet.py` und `tools/verify_*`.
- [Prüfnachweis](verification.json).

## Grenzen

- Kein Originalfall, keine echten Gewichte: diese stammen aus `rdirichlet` (`rgamma`) und bleiben Zufall.
- Der Anbieter ist nur im Fenster gebunden; Bitgleichheit mit R gilt nicht für `lgamma` (gemessen).
- Rücksetzung, Akzeptanz und Samplerlauf sind nicht implementiert. Alle elf Nachweisbündel bleiben offen; keine
  Variante gewählt.
- Frühere Pakete, Module, Fixtures und Tests sind bytegleich.
