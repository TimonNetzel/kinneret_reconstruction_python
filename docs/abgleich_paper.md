# Abgleich Paper ↔ Code

Netzel, T., Miebach, A., Litt, T., Hense, A. (2025): *New probabilistic methods for quantitative climate
reconstructions applied to palynological data from Lake Kinneret.* Clim. Past 21, 357–380
(`docs/paper/cp-21-357-2025.pdf`, CC BY 4.0).

Geprüfter Code: `legacy/` (Commit `782c6a0`). Er unterscheidet sich vom Tag `v0.1.0` nur durch
`LICENSE.md` und entspricht damit der im Paper zitierten Zenodo-Version „Reconstruction tools in
Python“ (doi:10.5281/zenodo.8214290). Der Original-Lauf reproduziert die Zahlen des Papers
(Abschnitt 3).

Legende: ✅ stimmt überein · ⚠️ Abweichung oder Unschärfe · ❌ Widerspruch/Fehler · ❓ im Paper nicht
festgelegt, Annahme des Codes

## 1. Modell

| Paper | Inhalt | Code (`legacy/src/…`) | Bewertung |
|---|---|---|---|
| Abschn. 3.3.2, Gl. (2) | Alters-Tiefen-Transformation P(Y\|A) = ∫ P(Y\|D) P(D\|A) dD; 181 Altersstufen à 50 Jahre, 0–9 ka | `load_data.py`: `depth_age_trans` (P(D\|A) an den 160 Probentiefen, je Alter auf 1 normiert), Matrixprodukt mit Spektrum bzw. AP | ✅ Das Integral wird als Summe über die beprobten Tiefen gebildet, P(D\|A) dafür neu normiert. Die Altersachse läuft von −50 bis 8950 cal BP. |
| Gl. (3), (A5), (A6) | P(B\|P,P_s,ω): Pollenspektrum, mit den Taxa-Gewichten gewichtet, auf Biome abgebildet | `spectrum_to_biome_assign` (C++) | ✅ Die genaue Formel steht nicht im Paper; die Umsetzung ist plausibel und konsistent. |
| Gl. (3), (A6) | Rekonstruktion P(C\|P,A,Θ) als **Mischverteilung** Σ_B P(C\|B,ψ)·P(B\|P,A,ω) | `reconstruction` (C++): je Stichprobe der **Mittelwert** Σ_b p_b·c_b | ⚠️ Befund 12, siehe Abschnitt 2.1 |
| Abschn. 3.5, Abb. 5b | Transferfunktion P(B\|C,ψ): Feedforward-Netz mit 2 Eingängen, 6 Hidden-Neuronen, 4 Ausgängen (3 Biome + „unbestimmt“) | `my_nnet_prediction` (Sigmoid, Softmax) | ✅ Die Normierung der Eingänge (T: z-Wert; P: √P → Gamma-Verteilungsfunktion → Probit) beschreibt das Paper nicht, sie stammt aus dem ML-Wettbewerb. |
| Gl. (14) | P(C\|ψ) = N_t(C \| μ(ψ), σ(ψ), a(ψ), b(ψ)), „als Vorschlagsverteilungen“ | `tf_sampling`, `tf_samples_ratios`: N_t nur als **Random-Walk-Vorschlag** (μ = aktueller Wert, σ = Sd(ψ)·2,38²/K); in der Zielverteilung steht für C nur P(B=j\|C,ψ) auf [a, b] | ⚠️ Gl. (14) schreibt N_t(μ(ψ), σ(ψ)) als Teil des Modells. Im Code ist der Prior von C implizit **gleichverteilt** auf [a, b]. Siehe Abschnitt 2.2. |
| Gl. (14) | Trunkierung a(ψ), b(ψ) = „Klimabereich der Transferfunktion“ | 5-%- und 95-%-Quantile der Randverteilungen auf dem Klimagitter | ❓ Die Quantile nennt das Paper nicht (Prüfbericht, Befund 3). |
| Gl. (5), Tab. 2 | Proxy-Pool-Modul Beta(PP\|α₁,α₂); Hauptlauf: Mittel 0,5 und Sd 0,2, also **α = (2,62; 2,62)** | `mcmc_preparation.py`: `shape1 = shape2 = 3` (Sd 0,189); ebenso Abb. 4 („Beta(PP\|3,3)“) | ⚠️ Kleiner Widerspruch zwischen Tab. 2 und Abb. 4/Code. Die Sensitivitätsläufe (Mittel 0,25 und 0,75) gibt es im Repo nicht; sie gingen nur durch Ändern des Codes. Jetzt: `configs/h_*`, `configs/i_*`. |
| Gl. (7), (8) | Prior-Klima-Modul: Γ(P_ANN,m) und N(T_DJF,m) für die jüngste Altersstufe (A ≤ 0 cal BP) | `reconstruction`: Werte der Altersstufe −50 cal BP; N(10,74; 2,48²) und Gamma mit Mittel 556 / Sd 135 mm | ✅ Gamma-Parameter per Gittersuche (Befund 5, vernachlässigbar). ❓ Wie Mittel und Sd aus den CRU-Daten bestimmt wurden, sagt das Paper nicht. Der nächstgelegene Gitterpunkt (Abb. 2c) hat T_DJF ≈ 11,4 °C und P_ANN ≈ 440 mm. |
| Gl. (12) | Startwert der Gewichte gleichverteilt (1/K) | `taxa_weights[0] = 1/num_taxa` | ✅ |
| Gl. (13) | Prior der Gewichte Dir(ω \| ½, …, ½) (Jeffreys) | `jeffreys_taxa_prior = 0.5` in der Zielverteilung | ✅ für den Prior selbst |
| Abschn. 3.4, Abb. 4 | rwMH für ω und C, „independent MH“ für Prior-Klima- und Proxy-Pool-Modul | ein gemeinsamer MH-Schritt für (ω, C); die Module gehen als Dichtefaktoren der Zielverteilung ein | ⚠️ Nur Begrifflichkeit, mathematisch gleichwertig |
| – (nicht im Paper) | Vorschlag der Gewichte: Dir(κ·ω), κ = K² | `new_taxa_weights`, `taxa_weights_ratios` | ❌ Die Hastings-Korrektur enthält κ nicht. Damit sampelt die Kette **nicht** die Posterior mit dem Prior aus Gl. (13) (Prüfbericht, Befund 1). Das Paper beschreibt den Vorschlag nicht; es ist ein reiner Implementierungsfehler. |

## 2. Neue bzw. präzisierte Befunde aus dem Abgleich

### 2.1 Mittelwert statt Mischverteilung (Befund 12)

Nach Gl. (3) ist die Rekonstruktion eine Mischung der Klimaverteilungen der Biome, gewichtet mit den
Biomanteilen. Der Code berechnet je MCMC-Stichprobe dagegen den gewichteten **Mittelwert**
Σ_b p_b·c_b und wertet dessen Verteilung aus. Die Unsicherheitsbänder in Abb. 10 erfassen damit nur
die Unsicherheit der Parameter (ω, C), nicht die Streuung innerhalb der Mischung.

Quantifiziert (Medians gemittelt über alle Alter; IQR = Interquartilsabstand):

| | Median T | IQR T | Median P | IQR P |
|---|---|---|---|---|
| Original, Mittelwert (wie im Paper) | 9,87 °C | 2,37 °C | 445 mm | 130 mm |
| Original, Mischverteilung nach Gl. (3) | 9,93 | 4,19 | 431 | 377 |
| Standardlauf, Mittelwert | 9,94 | 2,47 | 439 | 150 |
| Standardlauf, Mischverteilung | 10,00 | 4,04 | 422 | 379 |

Die Mediane ändern sich kaum. Die Unsicherheit ist bei der Mischverteilung aber 1,7-mal (T) bzw.
etwa 3-mal (P) so groß. Welche Lesart beabsichtigt ist, muss der Autor entscheiden:

- Der **Mittelwert** passt zu einer Mischung der Vegetation im Einzugsgebiet, die ein gemeinsames
  Klima anzeigt.
- Die **Mischverteilung** entspricht Gl. (3) wörtlich.

Der Code ist dazu nicht geändert.

### 2.2 Rolle der trunkierten Normalverteilung, Gl. (14)

Gl. (14) setzt P(C\|ψ) = N_t(C \| μ(ψ), σ(ψ), a(ψ), b(ψ)), mit Mittelwert und Sd aus der
Transferfunktion. Im Code ist N_t nur der Vorschlag eines Random Walks (zentriert auf den aktuellen
Wert), und seine Hastings-Korrektur ist korrekt. In die Zielverteilung geht N_t nicht ein. Der Prior
von C ist dort also gleichverteilt auf [a, b], multipliziert mit P(B\|C,ψ). Es gibt zwei Möglichkeiten:

- Der Text („as proposal distributions“) ist gemeint: Dann ist Gl. (14) als Modellgleichung
  missverständlich, der Code aber konsistent.
- Gl. (14) ist als Prior gemeint: Dann fehlt im Code der Faktor N_t(c; μ(ψ), σ(ψ)).

### 2.3 Beta-Prior des R²

Tab. 2 nennt für Mittel 0,5 und Sd 0,2 die Parameter (2,62; 2,62). Abb. 4 und der Code verwenden (3; 3),
also Sd 0,19. Der Unterschied ist klein, sollte aber vereinheitlicht werden.

## 3. Reproduktion der Ergebnisse des Papers mit dem Originalcode

| Paper | Paper-Wert | Original-Lauf (`kinneret-recon run --legacy`) |
|---|---|---|
| Abschn. 4.2: Laufzeit | 1 Mio. Iterationen, 40–60 s | 1 Mio. Iterationen, ~60 s (Paket: 37 s) |
| Abschn. 4.2: Konvergenz | multivariates Gelman-Rubin 1,001 | im Code nicht enthalten, braucht ≥ 2 Ketten. Rangnormiertes Split-R̂ der einen Kette: max. 1,002 |
| Abschn. 4.2: R² bei Prior 0,5 | Median 0,65, 50 %: 0,50–0,80 | Median 0,66, 50 %: 0,51–0,80 ✅ |
| Abschn. 4.2: mediterranes Biom | T_DJF ≈ 10 °C, P_ANN ≈ 700 mm | Median 10,0 °C, 710 mm ✅ |
| Abb. 7d: CI-Verhältnisse T (Med/IT/SA) | ≈ 0,69 / 0,82 / 0,63 | 0,68 / 0,80 / 0,64 ✅ |
| Abb. 7d: CI-Verhältnisse P | ≈ 0,64 / 0,79 / 0,79 | 0,62 / 0,79 / 0,76 ✅ |
| Abb. 6: Gewichte | *Olea* ↓; *Q. calliprinos*, *Amaranthaceae* ↑ | gleich ✅ |
| Abb. 10: Rekonstruktion | T ≈ 10 °C; P 350–550 mm | gleich (`reference/v1.0-paper/plots/reconstruction.pdf`) ✅ |

Burn-in (250 000) und Ausdünnung (75) nennt das Paper nicht.

## 4. Was sich mit korrektem Sampling an den Aussagen des Papers ändert

| Aussage im Paper | Mit korrekter Hastings-Korrektur, Prior Dir(½) wie in Gl. (13) |
|---|---|
| Abstract, Abschn. 4.2: höchste Gewichte bei *Q. calliprinos* und *Amaranthaceae*, niedrigstes bei *Olea* | *Amaranthaceae* ↑ und *Olea* ↓ bleiben. *Q. calliprinos* liegt beim Prior. Die Gewichte streuen so breit wie der Prior. |
| Abschn. 4.2: „mean posterior taxa weights deviate only slightly from the prior uniform distribution“ | Qualitativ bestätigt, aber der Prior Dir(½) ist **nicht** schmal um 1/K. Die schmalen Boxplots in Abb. 6 sind ein Artefakt des Fehlers. |
| Abschn. 4.2 / Abb. 7d: Verengung der Posterior-Varianzen durch Referenzkurve und CRU-Daten | Gilt nur für das mediterrane Biom (~20 %). Für IT und SA entsteht die gezeigte Verengung allein durch die Trunkierung (Prüfbericht, Befund 2). Das passt zum Text („changes … for the two remaining biomes are minor“), nicht aber zu Abb. 7d. |
| Abschn. 4.2: R²-Median 0,65 bei Prior 0,5 | 0,55 (50 %: 0,38–0,73) |
| Abb. 10: Rekonstruktion | Mediane innerhalb von 0,17 °C und 12 mm, Unsicherheit von P etwa 15 % größer |
| Abb. 8–10: Sensitivität gegenüber R² = 0,25 / 0,5 / 0,75 | siehe Abschnitt 5 |

## 5. Sensitivität gegenüber dem R²-Prior (Tab. 2)

- **Mittel 0,25, Beta(0,92; 2,77):** Mit dem Python-Code **nicht reproduzierbar**. Die Kette bleibt
  im Zustand R² = 0 hängen, weil die Beta-Dichte dort unendlich ist (Prüfbericht, Befund 13).
- **Mittel 0,75, Beta(2,77; 0,92):** reproduzierbar. Das R² der Posterior hat den Median 0,90,
  die Rekonstruktion ändert sich lokal um bis zu 71 mm.
- Korrigiert gerechnet folgt das R² der Posterior weitgehend dem Prior (0,27 / 0,55 / 0,85), und die
  Rekonstruktion ist robust. Details und Tabelle: `docs/pruefbericht.md`, Abschnitt 3.7.
- Konfigurationen: `configs/h_legacy_r2_025.toml`, `configs/h_legacy_r2_075.toml` (Original),
  `configs/i_standard_r2_025.toml`, `configs/i_standard_r2_075.toml` (korrigiert).
