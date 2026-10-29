# Abgleich Dissertation ↔ Paper ↔ Code

Netzel, T. (2022): Dissertation, Universität Bonn. Gelesen wurden die Abschnitte 2.3–2.5, 3.2 und 3.3
(`docs/paper/Phd_thesis_netzel*.PDF`, Stand 29.10.2026). Dieses Dokument fasst sie so zusammen, dass die PDFs für
die weitere Arbeit nicht noch einmal gelesen werden müssen. Gleichungsnummern beziehen sich auf die Dissertation.

**Token-sparend nachlesen**, falls doch nötig: `pdftotext datei.PDF - | tr -s ' ' | grep -v '^\s*$' > datei.txt`,
dann mit `grep -n` nach Stichworten suchen und nur die Treffer mit Kontext lesen; die PDFs nicht direkt einlesen.

## 1. Inhalt der Abschnitte

| Abschnitt | Inhalt | Verhältnis zu Paper und Code |
|---|---|---|
| 2.3 | Alters-Tiefen-Modell mit Bacon (`Bacon.d.Age`), $\Pr(D \mid A)$ diskretisiert, 50-Jahre-Raster, Anwendung als Matrixprodukt (Gl. 2.10, 2.11) | ✅ wie `model.py` (`depth_age_trans`) |
| 2.4 | Rekonstruktionsmodell auf **Taxa-Basis**: Auswahl $P_s$ der Taxa, Transferfunktionen $\Pr(P_s \mid C, \psi)$, $\Pr(C \mid \psi)$ **gleichverteilt** (Annahme 5 zu Gl. 2.13), Rekonstruktion als Mischung über die Auswahlen (Gl. 2.12, 2.15). Proxy-Pool-Modul Beta(3, 3) je Referenzkurve (Gl. 2.17), Prior-Klima-Modul mit Ankern für heute und optional für die Vergangenheit, sonst gleichverteilt (Gl. 2.18–2.20), Gewichte Dir(½), rwMH oder Metropolis-within-Gibbs mit Auswahl $z \sim \mathrm{Cat}(\omega)$, $t = K/2$ Taxa je Schritt, $\omega \mid z \sim \mathrm{Dir}(\tfrac12 + z)$ (Gl. 2.24–2.30) | Paper: Biome statt Taxa, feste Zuordnung, eine Referenz (AP). Gleichverteilter Prior von $C$ bestätigt die Lesart von Gl. (14) des Papers als Vorschlag (Befund 14) |
| 2.5 | Transferfunktionen **je Taxon binär** (vorhanden/fehlt), $T_\mathrm{ANN}$ und $P_\mathrm{ANN}$ aus CRU 4.01 (1961–1990), Verbreitungskarten (Meusel 1974); Wettbewerb mit `caret`: quadratische logistische Regression, `nnet`, QDA, MDA, GBM; SMOTE gegen unausgewogene Klassen; 70/30-Hold-out mit Kreuzvalidierung; Eingänge über die inverse Verteilungsfunktion normiert | Biom-Netz des Papers stammt aus derselben Kette (gleiche Normierung) |
| 3.2.1 | > 600 Standorte (EPD), Höhlen aus SISAL im Umkreis von 150 km (Auswahl per Pareto-Optimum); MG 12 500, rwMH 50 000 Iterationen, je 10 Ketten, multivariates Gelman-Rubin < 1,1; MG und rwMH liefern dieselbe Klimarekonstruktion (CRPSS ≈ 0) | Klima robust, wie im Prüfbericht; Gewichte nicht verglichen |
| 3.2.2 | Nachbearbeitung (Prespa): nur Stichproben in den IQR der Priors, Ausdünnen bis IAT < 1,3, Pareto-Auswahl der **10 Stichproben am nächsten an den Prior-Mittelwerten**; jede Stichprobe ist eine Dichte; Kombination per Mischung (Gl. 3.2), normiertem Produkt oder **Mittel der Zufallsgrößen** (Faltung, 1/N), Letzteres für alle Rekonstruktionen | Paket: volle Posterior, Mittelwert oder Mischung |
| 3.3 | Lokale Rekonstruktionen: Totes Meer (Holozän; Glazial 14,5–147 ka **ohne Klimaanker**), Kinneret, Birkat Ram, Van, Iznik, Ohrid (ohne Anker). Referenzen: AP/NAP, Soreq-Höhle (Isotope, Wavelet-Spektren), Medstack ($\delta^{18}$O, mit $T$ verknüpft); Anker aus Köppen-Geiger-Klassen; Einfluss des Menschen: AP nur vor ca. 6,5 ka als Referenz | Kinneret in der Dissertation: Phase um 7 ka ca. 18 °C ($T_\mathrm{ANN}$) und 550 mm |

## 2. Bewertung

**Übernehmen** (Ziel: Modell ohne Biome, mit mehreren Proxys):

1. Transferfunktionen je Taxon (binär, Vorkommen und Fehlen) statt Biomzuordnung.
2. Mehrere Referenzkurven mit je eigener Klimagröße (AP ↔ $P$, Soreq ↔ $P$, Medstack ↔ $T$) und eigenem Altersfenster.
3. Klimaanker für heute und, wo begründet, für die Vergangenheit (Gl. 2.19).
4. Gleichverteilter Prior der Klimawerte der Transferfunktionen (wie im Code).

**Nicht übernehmen:**

1. **Auswahl der 10 Stichproben am nächsten an den Prior-Mittelwerten.** Das ist keine Posterior-Auswertung: Der Prior
   wird zur Zielgröße ($R^2$ praktisch auf 0,5, Anker auf ihren Mittelwert festgelegt).
2. **Mittel von 10 Stichproben als unabhängige Zufallsgrößen (Faltung).** Die Varianz sinkt mit 1/N, die Bänder werden
   bei N = 10 rund 3,2-mal schmaler als die Mischung. Die Stichproben sind aber Ziehungen derselben Unsicherheit, keine
   unabhängigen Messungen; die Breite hängt von der frei gewählten Zahl N ab.
3. **Gibbs-Auswahl $z$.** Gl. 2.28/2.29 („$\alpha_i \sim \omega_i$“) ist unscharf; so geschrieben hängt $z$ nicht von den
   Daten ab. Die Auswahl übernehmen besser die Gewichte (hierarchischer Prior, kleine α ergeben dünn besetzte Gewichte)
   und eine feste Vorauswahl der Taxa.
4. **Wavelet-Spektren als Zielgröße.** Kein Wahrscheinlichkeitsmodell; allenfalls als Diagnose.

**Weitere Befunde:**

- Der $R^2$-Prior heißt in 3.2.1 „symmetric proposal prior“; Vorschlag und Prior werden wie in Gl. (14) des Papers nicht
  klar getrennt.
- Das $R^2$ der Posterior liegt für die Pollen-Referenz nahe am Prior. Die Dissertation erklärt das damit, dass Taxa und
  AP-Kurve aus demselben Sediment stammen (Zirkularität).
- **Offen:** Wie die Wahrscheinlichkeiten der gewählten Taxa innerhalb einer Stichprobe zu einer Klimadichte kombiniert
  werden, steht in keinem der gelesenen Abschnitte ausdrücklich. Naheliegend ist das Produkt
  $\prod_k \Pr(\text{Taxon}_k \mid C)$ mit gleichverteiltem Prior (pdf-Methode nach Kühl et al., 2002). Klären am Code
  der Dissertation, falls vorhanden. Für den Plan (`docs/plan_multiproxy.md`) ist die Frage nicht blockierend: Dort ist
  die Kombination eine Modellwahl mit zwei Varianten.
- Klimagrößen: Dissertation $T_\mathrm{ANN}$/$P_\mathrm{ANN}$ (CRU 4.01), Paper $T_\mathrm{DJF}$/$P_\mathrm{ANN}$ (CRU 4.07 im Paket).
  Transferfunktionen und Anker müssen dieselbe Größe verwenden.
