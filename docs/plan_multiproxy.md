# Plan: Rekonstruktionsmodell auf Taxa-Basis mit mehreren Proxys

Stand 29.10.2026. Vom Autor gewünscht: das Modell des Papers (Netzel et al., 2025) mit dem der Dissertation (Netzel,
2022) verbinden. Das neue Modell soll **mathematisch sauber** sein, **keine Biominformationen** brauchen und auf
**mehreren Proxys** beruhen. Grundlagen: `docs/abgleich_paper.md`, `docs/abgleich_thesis.md`, `docs/pruefbericht.md`.

## 1. Leitlinien

1. **Jeder Term hat eine klare Rolle:** Prior oder Likelihood, mit benannter Bedeutung. Keine Terme, die
   gleichzeitig Vorschlag und Prior sind (vgl. Befund 14 und die „proposal prior“ der Dissertation).
2. **Ausgewertet wird die volle Posterior**, als Mittelwert oder Mischung wie im Paket. Keine Auswahl von Stichproben
   nahe an den Prior-Mittelwerten, keine Faltung von N Stichproben (`docs/abgleich_thesis.md`, Abschnitt 2).
3. **Jede Stufe wird validiert,** bevor die nächste beginnt: Prior-only-Lauf, simulationsbasierte Kalibrierung (SBC),
   Vergleich mit dem Biom-Modell.
4. **Der Bestand bleibt unverändert:** Das neue Modell ist eine Option (`sampler.modell = "taxa"`, Voreinstellung
   `"biom"`). Legacy-Modus bitgleich, die **Fingerabdrücke aller 27 Läufe unverändert**. `katalog.fingerabdruck`
   bildet die Prüfsumme über alle flachen Einstellungen; neue Felder dürfen deshalb nur eingehen, wenn sie von der
   Voreinstellung abweichen. Prüfen mit `kinneret-recon katalog`.

## 2. Modell

### 2.1 Bezeichnungen

| Größe | Bedeutung |
|---|---|
| $a = 1 \ldots A$ | Altersstufen (50 Jahre), wie bisher |
| $s_{ak}$ | Anteil von Taxon $k$ im Alter $a$, über $\Pr(D \mid A)$ aus den Tiefen (wie bisher), normiert über die Taxa mit Transferfunktion |
| $k = 1 \ldots K'$ | Taxa mit Transferfunktion (Vorauswahl, Abschnitt 2.6) |
| $\omega$ | Taxa-Gewichte auf dem Simplex |
| $c_k = (T_k, P_k)$ | Klimawerte von Taxon $k$ (ersetzen die Biom-Klimawerte $c_b$) |
| $L_k(c) = \Pr(\text{Taxon}_k \text{ vorhanden} \mid c, \psi)$ | Transferfunktion von Taxon $k$ (Dissertation 2.5) |
| $x_a = (T_a, P_a)$ | Rekonstruktion im Alter $a$ |
| $y_{r,a}$ | Referenzkurve $r$ (AP, Soreq, Medstack, …) auf dem Altersraster |

### 2.2 Lokales Rekonstruktionsmodul

- Anteile: $p_{ak}(\omega) = s_{ak}\,\omega_k / \sum_j s_{aj}\,\omega_j$. Das entspricht dem Paper mit der
  Einheitsmatrix als Zuordnung: jedes Taxon ist eine eigene „Gruppe“.
- Prior der Klimawerte: $\pi(c_k) \propto L_k(c_k)\cdot \mathbf{1}[c_k \in B_k]$, mit $B_k$ dem Kasten aus den
  5-%- und 95-%-Quantilen von $L_k$ als Dichte auf dem Klimaraster. Das ist $\Pr(C \mid \text{vorhanden})$ nach Bayes mit
  gleichverteiltem Prior, genau wie im Paper-Code und in der Dissertation (Annahme 5 zu Gl. 2.13). Die Grenzen sind eine
  Modellwahl (Befund 3), daher als Sensitivität mit 1/99 %.
- **Variante A (Standard): gewichtetes Mittel** $x_a = \sum_k p_{ak}\, c_k$, wie im Paper. Option: Mischung nach Gl. (3)
  des Papers in der Auswertung.
- **Variante B (nur als Vergleich): Produkt nach der pdf-Methode** mit dem Klima $x_a$ je Alter als Parameter. Mit
  Transferfunktionen für Vorkommen und Fehlen modelliert B die Prozentwerte nicht; sie müssten als Exponent oder über
  Schwellen eingehen. Das ist kein sauberes Wahrscheinlichkeitsmodell. B kommt deshalb nur als Vergleichslauf in
  Frage, falls sich am Code der Dissertation zeigt, dass dort so gerechnet wurde.
- **Ausblick (nicht in diesem Plan):** Wirklich generativ wäre ein Modell der Pollenzählungen (z. B.
  Dirichlet-Multinomial) mit Transferfunktionen der Häufigkeiten. Dafür fehlen die Trainingsdaten (moderne
  Pollenspektren).

### 2.3 Referenzkurven (Proxy-Pool-Modul)

Jede Referenz $r$ hat eine **Klimagröße** $v_r \in \{T, P\}$, ein **Vorzeichen** $\sigma_r \in \{+1, -1\}$ (z. B. AP ↔
$P$ positiv; $\delta^{18}$O der Soreq-Höhle ↔ $P$ negativ), ein **Altersfenster** $W_r$ (z. B. AP nur vor 6,5 ka wegen
der Landnutzung) und ein eigenes **Altersmodell** (Höhlen: eigenes $\Pr(D \mid A)$, angewendet wie beim Kern).
Referenzen gelten als bedingt unabhängig gegeben das Klima; die Terme werden multipliziert.

Zwei Formen des Terms:

- **R1 – wie im Paper:** $\mathrm{Beta}(R^2_r \mid 3, 3)$ auf die quadrierte Korrelation von $x^{(v_r)}$ und $y_r$ über
  $W_r$, mit Vorzeichen. Der heutige Code setzt $R^2 = 0$, sobald die Korrelation ≤ 0 ist; das muss das Vorzeichen
  $\sigma_r$ berücksichtigen. R1 ist eine Pseudo-Likelihood (ein Prior auf eine abgeleitete Größe); SBC ist damit nicht
  sauber definiert.
- **R2 – Proxy-Modell (empfohlen nach Vergleich):** standardisierte Referenz
  $\tilde y_{r,a} = \sigma_r \sqrt{\rho_r}\; z_a\!\left(x^{(v_r)}\right) + \sqrt{1-\rho_r}\;e_{r,a}$, mit $z_a$ der über
  $W_r$ standardisierten Rekonstruktion, $e_r$ ein AR(1)-Prozess mit Varianz 1 und Parameter $\phi_r$, Prior
  $\rho_r \sim \mathrm{Beta}(3, 3)$ und $\phi_r \sim \mathrm{Unif}(0, 1)$ oder ähnlich.
  - $\rho_r$ ist genau der Anteil der erklärten Varianz. Die Idee des Papers („die Rekonstruktion erklärt etwa 50 % der
    Referenz“) wird damit zu einem echten Modell mit echtem Prior.
  - Wie bei R1 hängt der Term nicht von Niveau und Amplitude der Rekonstruktion ab; diese kommen aus den
    Transferfunktionen und den Ankern.
  - AR(1) berücksichtigt die Autokorrelation (geglättete Kurven) und begrenzt das Gewicht der 181 Altersstufen.
  - Unsicherheit des Altersmodells der Referenz: zuerst nur der Erwartungswert. Später als zusätzliche Varianz je
    Alter aus den Stichproben des Altersmodells.

### 2.4 Prior-Klima-Modul (Anker)

- **Heute:** $\mathcal N(T_{a_0})$ und $\Gamma(P_{a_0})$ auf die jüngste Altersstufe, aus CRU für das Einzugsgebiet
  (`cru-prior`), mit **derselben Klimagröße wie die Transferfunktionen**. Die Transferfunktionen der Dissertation nutzen
  $T_\mathrm{ANN}$; `cru.py` kann bisher nur $T_\mathrm{DJF}$ und muss dafür erweitert werden.
- **Vergangenheit (optional, Dissertation Gl. 2.19):** Anker für ein bestimmtes Alter (z. B. LGM) aus unabhängigen
  Quellen. Sie dürfen nicht aus denselben Daten stammen wie eine verwendete Referenz.
- **Ohne Anker:** möglich (Dissertation: Totes Meer glazial, Ohrid). Dann braucht jede Klimagröße mindestens eine
  Referenz, sonst bestimmt allein die Transferfunktion den Verlauf dieser Größe.

### 2.5 Priors im Überblick

| Parameter | Prior | Bemerkung |
|---|---|---|
| $\omega$ | Dir(α), $\log\alpha \sim \mathcal N(\log 1, 1)$ | hierarchisch wie Standard v3; kleine α ergeben dünn besetzte Gewichte, sie ersetzen die Gibbs-Auswahl der Dissertation |
| $c_k$ | $\propto L_k(c_k)$ auf $B_k$ | Abschnitt 2.2 |
| $\rho_r$, $\phi_r$ | Beta(3, 3), Unif(0, 1) | nur R2 |
| Anker | Normal bzw. Gamma | Daten, nicht Prior im engeren Sinn |

### 2.6 Vorauswahl der Taxa

Fest und vor dem Lauf, keine Auswahl im Sampler. Taxa mit Transferfunktion, Mindestanteil (z. B. im Mittel ≥ 0,5 %),
ohne Kulturbäume (*Olea*, ggf. *Vitis*, *Juglans*) oder mit eigenem Altersfenster. Diagnose je Alter: welcher Anteil der
Pollensumme durch die gewählten Taxa abgedeckt ist.

**Pollentyp ↔ Transferfunktion:** Die Zuordnungstabelle liefert der Autor. Steht ein Pollentyp für mehrere Arten, gilt
z. B. $L_\text{Typ} = 1 - \prod_s (1 - L_s)$ (mindestens eine Art vorhanden). Das ist eine Entscheidung des Autors.

### 2.7 Sampler und Auswertung

- Gewichte: log-Gamma-Random-Walk mit hierarchischem α (vorhanden). $c$: adaptiver Random Walk (vorhanden), bei vielen
  Taxa blockweise je Taxon. $\rho_r$, $\phi_r$: Random Walk auf der Logit-Skala.
- Transferfunktionen als **Raster** $\log L_k$ auf dem Klimagitter, im C++-Kern bilinear interpoliert. Das ist
  unabhängig vom Verfahren (GLM, nnet, QDA, MDA, GBM), schnell und braucht kein R im Paket.
- Auswertung wie bisher: Quantile aus den Stichproben, Mittelwert oder Mischung, Konvergenzdiagnostik (R̂, ESS).

## 3. Daten vom Autor

| Nr. | Daten | Format (Vorschlag) |
|---|---|---|
| D1 | **Transferfunktionen der Taxa** | je Taxon $\Pr(\text{vorhanden} \mid T, P)$ auf einem Raster (z. B. 500 × 500 wie `temp_range`/`pann_range`), dazu Verfahren, Klimagrößen, CRU-Version, Zeitraum, Region, Gütemaß aus dem Hold-out und Normierung. Export in R siehe unten |
| D2 | Zuordnung Pollentyp ↔ Taxon der Transferfunktion, je Standort | CSV |
| D3 | **Alters-Tiefen-Dichten** $\Pr(D \mid A)$ der Kerne (Kinneret liegt vor; Birkat Ram, Totes Meer) und der Speläotheme | wie `age_depth_relationship.npz` |
| D4 | **Isotopendaten:** Soreq (Tiefe bzw. Abstand, Werte, Proxy-Art), Medstack (Alter, Wert, ggf. Unsicherheit) | CSV |
| D5 | Welche Referenz zu welcher Größe, mit Vorzeichen und Altersfenster (Vorschlag aus der Dissertation: AP ↔ $P$ vor 6,5 ka, Soreq ↔ $P$, Medstack ↔ $T$) | Eintrag in `standort.toml` |
| D6 | optional: Code der Dissertation (klärt die Kombinationsregel, Variante B) | Repo oder Archiv |

Ablage: kleine Dateien unter `data/<standort>/`, große im privaten Daten-Repo `kinneret_reconstruction_laeufe`.
Weitergabe großer Dateien über Git (bis 100 MB je Datei), nicht über den Upload im Browser (25 MB).

Export der Transferfunktionen aus R (Skizze, `fits` = Liste der caret-Modelle je Taxon):

```r
grid <- expand.grid(T = seq(-10, 30, length.out = 500), P = seq(0, 1500, length.out = 500))
# gleiche Normierung der Eingänge wie beim Training (inverse Verteilungsfunktion) anwenden
probs <- sapply(fits, function(m) predict(m, newdata = normiere(grid), type = "prob")[, "presence"])
write.csv(cbind(grid, probs), "taxa_tf_raster.csv", row.names = FALSE)
```

## 4. Schritte

Voraussetzung, gemeinsam mit `docs/plan_standorte.md`: dort Schritt 1 (Code auf mehrere Standorte umstellen) und
Schritt 2 (Werkzeug zur Umwandlung der Rohdaten). Die Biom-Läufe für Birkat Ram und das Tote Meer bleiben sinnvoll,
als Vergleich für das neue Modell.

| Schritt | Inhalt | Abnahme |
|---|---|---|
| M0 | Daten D1–D5 einlesen und prüfen: Raster normiert, Klimagrößen einheitlich, Zuordnungstabelle vollständig, Altersmodelle normiert; Format `taxa_tf.npz` festlegen | Prüfbericht der Eingangsdaten; Abbildung der Transferfunktionen |
| M1 | **Taxa-Modell mit Paper-Kalibrierung:** `sampler.modell = "taxa"`, Raster-Transferfunktionen im Kern, Einheitsmatrix als Zuordnung, R1 nur mit AP, CRU-Anker ($T_\mathrm{ANN}$) | Fingerabdrücke unverändert, Tests grün; Prior-only-Lauf trifft die Priors; Kinneret im Vergleich mit Standard v3 (Verlauf von $P$, Niveaus, Abdeckung der Pollensumme) |
| M2 | **Proxy-Modell R2** für die AP-Kurve | SBC mit synthetischen Referenzen (Ränge gleichverteilt); Vergleich R1 ↔ R2 (Rekonstruktion, $\rho$ gegen $R^2$) |
| M3 | **Mehrere Proxys:** Referenzen als Liste (Größe, Vorzeichen, Fenster, Altersmodell); Soreq und Medstack | Leave-one-out je Referenz (wie gut sagt das Modell die ausgelassene Referenz vorher?); Konsistenz der Referenzen |
| M4 | Anker der Vergangenheit; Läufe ohne heutigen Anker (Frage 1 des Autors) | Kerntop-Validierung: rekonstruiertes Klima der jüngsten Stufe ohne Anker gegen CRU |
| M5 | Sensitivität und Katalog: Grenzen $B_k$, Vorauswahl der Taxa, α-Prior, Fenster, R1/R2, Mittelwert/Mischung, ggf. Variante B; als Faktoren im Faktor-Register | Lauf-Katalog, App-Daten |
| M6 | Weitere Standorte (Birkat Ram, Totes Meer) mit dem Taxa-Modell; App: Ansicht „Taxa“ statt „Biome“ | Vergleich mit den Biom-Läufen und der Dissertation (3.3) |

## 5. Verbesserungen gegenüber dem ursprünglichen Vorhaben

1. **Proxy-Modell R2 statt Beta auf $R^2$:** Die Idee bleibt (Anteil erklärter Varianz mit Prior Beta(3, 3)), wird aber
   eine echte Likelihood. Erst damit sind SBC und eine saubere Kombination mehrerer Referenzen möglich.
2. **Vorzeichen und Klimagröße je Referenz:** Der heutige Code verlangt eine positive Korrelation; Isotope sind meist
   negativ mit dem Klima verknüpft.
3. **Altersmodell je Referenz:** Höhlen haben eigene U/Th-Chronologien und kommen über ihr eigenes $\Pr(D \mid A)$ auf
   dasselbe Altersraster.
4. **Transferfunktionen als Raster:** unabhängig vom Verfahren aus dem caret-Wettbewerb, kein R im Paket.
5. **Gewichte statt Gibbs-Auswahl, volle Posterior statt Pareto-Auswahl und Faltung:** vermeidet die unscharfen
   Gl. 2.28/2.29 und die künstlich engen Bänder der Dissertation.
6. **Einheitliche Klimagrößen** für Transferfunktionen und Anker ($T_\mathrm{ANN}$ und $P_\mathrm{ANN}$ bei den
   Transferfunktionen der Dissertation).
7. **Validierung** in jeder Stufe: Prior-only-Lauf, SBC, Leave-one-out je Referenz, Vergleich mit dem Biom-Modell.
8. **Zirkularität beachten:** AP stammt aus demselben Kern wie die Taxa. Mit externen Referenzen (Soreq, Medstack)
   prüfen, ob AP als Referenz nötig ist oder nur mit Fenster bzw. ohne Kulturbäume.

## 6. Offene Entscheidungen (Autor)

1. Klimagrößen: $T_\mathrm{ANN}$/$P_\mathrm{ANN}$ (folgt aus den Transferfunktionen der Dissertation) oder
   $T_\mathrm{DJF}$ (neu trainieren).
2. Referenzen je Standort mit Vorzeichen und Altersfenster (D5); AP als Referenz behalten?
3. Vorauswahl der Taxa und Regel für Pollentypen mit mehreren Arten (Abschnitt 2.6).
4. Standardform der Referenzterme nach M2: R1 oder R2.
5. Reihenfolge: zuerst die Biom-Läufe der neuen Standorte (`docs/plan_standorte.md`) oder zuerst M0–M2 an Kinneret.
   Vorschlag: die gemeinsamen Schritte 1–2 aus `plan_standorte.md`, dann M0–M2 an Kinneret.

## 7. Risiken

- **Güte der Transferfunktionen für Pollentypen:** Typen fassen Arten mit verschiedenen Klimaansprüchen zusammen; breite
  Klimahüllen dämpfen die Rekonstruktion.
- **Dämpfung der Amplitude:** Das gewichtete Mittel zieht zur Mitte, wenn sich die Klimahüllen überlappen (wie bei
  der gewichteten Mittelung, WA). In M1 gegen das Biom-Modell prüfen.
- **$T$ an Kinneret:** Nur mit Referenzen zu $P$ ist der Verlauf von $T$ kaum bestimmt (Prüfbericht 3.8, *r* 0,74–0,92);
  Medstack (M3) behebt das teilweise.
- **Chronologien:** Versätze zwischen den Altersmodellen von Kern und Höhle verschieben die Korrelation; Fenster und
  Alterssicherheit prüfen.
- **Rechenzeit:** $2K'$ Klimawerte statt 6; mit Raster und C++ unkritisch, Mischung bei vielen Taxa beobachten.
