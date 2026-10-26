# Prüfbericht: Code zu Netzel et al. (2025), Clim. Past 21, 357–380

Stand: 26.10.2026. Geprüft wurde der Code im Tag `v1.0-paper`.

> **Einschränkung:** Der Volltext des Papers war in der Arbeitsumgebung nicht abrufbar (Netzwerkfreigabe
> für `cp.copernicus.org` fehlt). Geprüft wurde daher der Code **in sich**: mathematische Korrektheit
> des MCMC-Verfahrens, numerische Umsetzung, Reproduzierbarkeit. Ein Abgleich Gleichung für Gleichung
> mit dem Paper (Schritt 2 des Plans) steht noch aus. Alle Aussagen über das, was das Paper
> beabsichtigt, sind deshalb als Annahmen gekennzeichnet.

## 1. Zusammenfassung

| # | Befund | Art | Auswirkung auf die Ergebnisse |
|---|---|---|---|
| 1 | Hastings-Korrektur des Dirichlet-Vorschlags ohne Streuungsparameter κ | **Fehler** (Mathematik) | Rekonstruktion: gering (≤ 0,18 °C, ≤ 12 mm; Unsicherheit bei P +15 %). Taxa-Gewichte: **deutlich** |
| 2 | Verhältnis „Posterior/Prior-CI“ der Transferfunktionen enthält einen Trunkierungseffekt | **Methodik**/Interpretation | Verengung für IT- und SA-Biom entsteht vollständig durch die Trunkierung, nicht durch die Daten |
| 3 | Posterior der TF-Werte stößt an die Trunkierungsgrenzen (v. a. T_Med) | Modellwahl, prüfen | Rekonstruktion robust (≤ 0,06 °C, ≤ 9 mm bei 1/99 % statt 5/95 %) |
| 4 | Startwerte von R², T0, P0 passen nicht zum Startzustand | Fehler (klein) | nur erste Iteration, keine |
| 5 | Gamma-Prior für P0 per Gittersuche statt analytisch | Ungenauigkeit | vernachlässigbar (Sd 135,16 statt 134,98 mm) |
| 6 | Φ⁻¹ im Sampler in einfacher Genauigkeit (`float`) | Ungenauigkeit | vernachlässigbar (~5·10⁻⁷) |
| 7 | Zufallszahlen compilerabhängig, Auswertung teilweise ungeseedet | Reproduzierbarkeit | Ergebnisse nur mit gleichem Compiler reproduzierbar; Prior-CI schwankt ~1 % |
| 8 | Schrittweite nach Rosenthal falsch skaliert (Varianz- statt Sd-Faktor) | Effizienz | keine (nur Mischungsgeschwindigkeit) |
| 9 | Kommentar „# wrong“ beim Klimagitter in `load_data.py` | Kommentar | keine. Die Orientierung ist korrekt (Test `test_grid_orientation`) |
| 10 | Laufzeit, Speicher, Kompatibilität (NumPy 2, Matplotlib ≥ 3.9) | Technik | Original läuft mit aktuellen Bibliotheken nicht mehr |

## 2. Vorgehen

1. **Referenz.** Das Original wurde mit minimalen Kompatibilitäts-Patches ausgeführt
   (`legacy/README.md`) und die Ergebnisse in `reference/v1.0-paper/` gesichert.
2. **Umbau ohne Verhaltensänderung.** Das neue Paket enthält den Original-Sampler unverändert. Es
   reproduziert Kette, Akzeptanzentscheidungen, alle Quartile und die Abbildungen **bitidentisch**
   (Tests `test_legacy_equivalence.py`).
3. **Überarbeiteter Sampler.** Jede Korrektur ist einzeln abschaltbar. Mit allen Korrekturen aus
   (Lauf A) stimmt der Sampler statistisch mit dem Original überein: höchstens 0,06 °C und 3 mm
   Unterschied im Median, also Monte-Carlo-Streuung.
4. **Validierung an bekannten Lösungen.** Ohne Daten muss eine korrekte Kette den Prior
   wiedergeben (Tests `test_sampler_validation.py`).
5. **Vergleichsreihe.** Die Korrekturen wurden schrittweise eingeschaltet (Abschnitt 4), jeweils
   mit 4 Ketten und R̂ ≤ 1,01 und ESS ≥ 400 für alle 43 Größen.

## 3. Befunde im Einzelnen

### 3.1 Hastings-Korrektur der Taxa-Gewichte (Befund 1)

Die Taxa-Gewichte werden mit w′ ~ Dir(κ·w) vorgeschlagen, κ = K² = 1089
(`new_taxa_weights`). Die Hastings-Korrektur muss daher

  q(w | w′) / q(w′ | w) = Dir(w; κ·w′) / Dir(w′; κ·w)

lauten. `taxa_weights_ratios` berechnet aber Dir(w; w′) / Dir(w′; w), also **ohne κ**:

```cpp
double proposal_ratio1 = lgamma(sum_vector(new_weights)) - sum_vector(lgamma_vector(new_weights))
                       + sum_vector(vector_weight_vector(vector_subtract(new_weights,1),log_old_w));
```

Damit erfüllt die Kette die Detailed-Balance-Bedingung für die beabsichtigte Posterior nicht. Ihre
stationäre Verteilung ist eine andere, analytisch nicht angebbare Verteilung.

**Nachweis (Prior-only-Lauf).** Ohne Kalibrierung und Transferfunktion ist die Zielverteilung der
Gewichte genau der Jeffreys-Prior Dir(0,5, …, 0,5):

| | Sd(w) | Median(w) | P(w < 0,001) |
|---|---|---|---|
| Sollwert Dir(0,5) | 0,0410 | 0,0143 | 0,141 |
| korrigierter Sampler | 0,0410 | 0,0143 | 0,140 |
| Verhalten des Originals | 0,0246 | 0,0241 | 0,012 |

Das Original zieht die Gewichte also systematisch zu 1/K hin. Das wirkt wie ein deutlich
informativerer Prior als der angegebene.

**Auswirkung mit Daten** (Lauf B gegen Original):

- Rekonstruktion: Median T höchstens 0,18 °C, Median P höchstens 12 mm verschoben (im Mittel −7 mm).
  Der Interquartilsabstand von P wächst um 15 % (131 → 151 mm), der von T um 5 %. Das mittlere R²
  mit der AP-Kurve sinkt von 0,64 auf 0,55.
- **Taxa-Gewichte:** Mit korrekter Korrektur liegen die Posterior-Gewichte fast aller Taxa beim
  Prior-Median (≈ 0,014–0,015, siehe Abbildung unten). Klar von den Daten bestimmt sind nur
  *Amaranthaceae* (höher, Median 0,019), *Olea europaea* (niedriger, 0,007) und schwächer
  *Quercus ithaburensis type* (0,013). Die im Abstract genannte hohe Gewichtung von *Quercus
  calliprinos type* (Original 0,029) **verschwindet** (0,015 ≈ Prior). Die Aussagen zu
  *Amaranthaceae* und *Olea* bleiben qualitativ bestehen.

![Taxa-Gewichte](abbildungen/vergleich_taxa_gewichte.png)

**Gegenprobe.** Dieselbe korrigierte Zielverteilung wurde mit zwei verschiedenen Vorschlägen
gerechnet: dem Dirichlet-Random-Walk des Originals mit korrekter Korrektur (5 Mio. Iterationen) und
dem neuen log-Gamma-Random-Walk. Beide stimmen überein (T 9,93 gegen 9,96 °C, P 436 gegen 438 mm,
gleiche Gewichte).

### 3.2 Verhältnis „Posterior/Prior-CI“ der Transferfunktionen (Befund 2)

Die TF-Klimawerte werden auf das 5-%- bis 95-%-Quantil der Prior-Randverteilungen trunkiert. Die
Auswertung vergleicht die Posterior aber mit dem 95-%-Intervall der **untrunkierten** Randverteilung,
und die P-Werte stammen dabei vom nichtlinearen Klimagitter. Schon ohne jede Information aus den
Daten (Prior-only-Lauf) ergibt sich deshalb ein Verhältnis < 1:

| 95-%-CI post/prior | T Med | T IT | T SA | P Med | P IT | P SA |
|---|---|---|---|---|---|---|
| Original (Abbildung im Paper) | 0,68 | 0,80 | 0,64 | 0,62 | 0,79 | 0,76 |
| ohne Daten, gleiche Methode | 0,88 | 0,80 | 0,64 | 0,77 | 0,79 | 0,76 |
| **korrekte Referenz** (Prior-Lauf des Modells) | **0,79** | **1,00** | **1,02** | **0,81** | **0,99** | **1,00** |

**Folgerung:** Für das irano-turanische und das saharo-arabische Biom liefern die Daten **keine**
Information über die Klimabereiche. Die gezeigte Verengung ist ein Effekt der Trunkierung. Beim
mediterranen Biom werden die Intervalle durch die Daten um rund 20 % enger. Das stützt die Aussage
im Abstract, dass sich das mediterrane Biom am stärksten ändert, aber mit geringerem Ausmaß.
Das Paket verwendet jetzt standardmäßig die korrekte Referenz (`prior_reference = "model"`).

### 3.3 Trunkierungsgrenzen (Befund 3)

In der Posterior liegen 8,9 % der T_Med-Werte im obersten 5-%-Band des erlaubten Bereichs
(gleichverteilt wären es 5 %), bei T_SA 7 % im untersten Band. Die Grenzen sind eine Modellwahl.
Die Empfindlichkeitsrechnung mit 1 %/99 % (Lauf E) verschiebt die Rekonstruktion gegenüber dem Standardlauf höchstens um
0,06 °C bzw. 9 mm und weitet die Unsicherheit leicht (IQR P +5 %). Die Rekonstruktion ist also
robust. Die Wahl sollte im Paper aber begründet werden.

### 3.4 Kleinere Befunde (4–9)

- **Startwerte (4):** `prior[0..2][0]` werden zufällig gesetzt (0,2 / 4 °C / 200 mm) statt aus dem
  Startzustand berechnet. Der erste MH-Schritt vergleicht daher mit einem nicht existierenden
  Zustand. Nach dem Burn-in ohne Wirkung. Korrigiert: Der Startzustand wird ausgewertet.
- **Gamma-Prior (5):** Form 16,938 / Rate 0,030451 (Gitter) statt 16,982 / 0,030530 (exakt).
- **Φ⁻¹ (6):** `my_erfinvf` rechnet in `float`. Ersetzt durch AS241 in `double`.
- **Reproduzierbarkeit (7):** `std::default_random_engine` und `std::gamma_distribution` sind
  implementierungsabhängig, also auf einem anderen Compiler andere Ketten. Zusätzlich ein eigener
  Park-Miller-Generator je Iteration mit Seeds aus Python. Neu: ein portabler Generator
  (`mt19937_64` mit eigenen Verteilungen), Seeds je Kette aus `numpy.random.SeedSequence`.
- **Rosenthal-Skalierung (8):** Kommentar und Code skalieren die Standardabweichung mit 2,38²/K.
  Rosenthal (2010) skaliert die *Varianz* mit 2,38²/d. Außerdem ist d die Dimension des
  gemeinsamen Vorschlags (39), nicht K = 33. Beeinflusst nur die Effizienz.
- **Klimagitter (9):** Der Kommentar `# wrong` / `# maybe correct` ist unbegründet. Die
  Randverteilungen werden über die richtigen Achsen gebildet.

### 3.5 Technik (10)

- Mit NumPy ≥ 2 und Matplotlib ≥ 3.9/3.10 bricht das Original an sechs Stellen ab. Außerdem
  scheitert die Kompilierung, wenn `python3-config` zu einer anderen Python-Version gehört.
- Laufzeit des Originals: MCMC ~60 s (Spitzenspeicher ~3 GB, weil alle 10⁶ Zustände als
  Python-Listen übergeben werden), Auswertung 576 s.
- Das Paket braucht im Legacy-Modus 37 s für das MCMC und 10 s für die Auswertung, mit
  bitidentischen Ergebnissen. Im Standardlauf mit 4 Ketten × 2 Mio. Iterationen sind es 93 s.

## 4. Vergleichsreihe

| Lauf | Einstellungen | T Median (Mittel über Alter) | P Median | IQR T | IQR P | min. ESS | max. R̂ |
|---|---|---|---|---|---|---|---|
| Original | `legacy/`, 1 Kette | 9,87 °C | 444,9 mm | 2,40 | 131,5 | 1443 | 1,002 |
| A | neuer Sampler, Korrekturen aus | 9,82 | 446,6 | 2,34 | 129,7 | 5741 | 1,001 |
| B | + Hastings-Korrektur | 9,96 | 437,9 | 2,51 | 151,4 | 12 850 | 1,000 |
| B′ | wie B, Dirichlet-Vorschlag (Gegenprobe) | 9,93 | 436,4 | 2,50 | 152,7 | 1058 | 1,007 |
| C | + exakter Gamma-Prior | 9,94 | 438,6 | 2,49 | 150,9 | 12 523 | 1,000 |
| **Standard** | + Quartile aus Stichproben | 9,94 | 438,6 | 2,47 | 149,7 | 12 523 | 1,000 |
| E | Standard, TF-Grenzen 1/99 % | 9,95 | 445,3 | 2,65 | 156,7 | 12 743 | 1,000 |

Konfigurationen: `configs/*.toml`. Reproduzieren mit `kinneret-recon run -c configs/<name>.toml`.

![Vergleich der Rekonstruktionen](abbildungen/vergleich_rekonstruktion.png)

## 5. Bewertung

- **Die Klimarekonstruktion des Papers hält.** Alle Korrekturen zusammen verschieben die Mediane
  um höchstens 0,17 °C und 12 mm. Das liegt weit innerhalb der Unsicherheitsbänder. Die
  Unsicherheit von P ist aber um etwa 15 % unterschätzt.
- **Die Aussagen zu den Taxa-Gewichten müssen überarbeitet werden.** Die Daten informieren nur
  über wenige Taxa (*Amaranthaceae* ↑, *Olea* ↓). Die übrigen Gewichte folgen dem Prior.
- **Die Aussagen zur Verengung der Transferfunktionen müssen überarbeitet werden.** Nur das
  mediterrane Biom wird durch die Daten eingeschränkt (~20 %).

## 6. Offene Punkte

1. Abgleich mit den Gleichungen des Papers (Volltext nötig), insbesondere: Ist das Fehlen von κ
   eventuell beabsichtigt und im Paper so formuliert? Ist die Wahl der Trunkierung dort begründet?
2. Fachliche Bewertung der Taxa-Gewichte: Ist ein Dir(0,5)-Prior gewollt, der viele Gewichte nahe
   null zulässt? Alternative: ein weniger extremer Prior,
   z. B. Dir(1). In den Stichproben des Standardlaufs liegen 14 % aller Gewichte unter 0,001.
3. Erratum oder Corrigendum zum Paper erwägen (Taxa-Gewichte, CI-Verhältnisse).

## Literatur

- Rosenthal, J. S. (2011): Optimal proposal distributions and adaptive MCMC. In: Handbook of MCMC.
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., Bürkner, P.-C. (2021): Rank-normalization,
  folding, and localization: An improved R̂ for assessing convergence of MCMC. Bayesian Analysis 16(2).
- Marsaglia, G., Tsang, W. W. (2000): A simple method for generating gamma variables. ACM TOMS 26(3).
- Wichura, M. J. (1988): Algorithm AS 241: The percentage points of the normal distribution.
