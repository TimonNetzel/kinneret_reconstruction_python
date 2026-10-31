# Vorbereiteter, noch nicht ausgeführter Zähltest

Dieser Plan ist Teil des autorisierten **Designpakets**. Implementierung und
Ausführung brauchen das nächste Nutzer-OK. Feste Parameter in
[synthetic_plan.json](./synthetic_plan.json); keine bestehenden Modelle laden.
Keine neue Art-/Typzuordnung, keine Fits, keine biologisch geschätzten Raten.

## Vollständig künstliches Vorwärtsmodell

Dimensionsloses Klima c=(u,v) auf 21×21 Zentren in [−1,1]², Abstand 0,1.
Vier anonyme Quellen plus eine sonstige Quelle U:

```text
a(c) = (2u, -2u, 2v, -2v, 0)
V_j(c,h) = exp(a_j + h*b_j) / sum_l exp(a_l + h*b_l)
b = (1, 1, -1, -1, 0)
r = (1,1,1,1,1), d = (1,1,1,1,1), B = (0,0,0,0,0)
```

Damit sind V normierte Vegetationsanteile im Spielzeugmodell, keine realen
Arealwahrscheinlichkeiten. Basismodell h=0. A=I_5 ordnet vier Quellen auf vier
anonyme Typen und U auf R ab. R ist hier eine **bekannt künstlich** definierte
gezählte Kategorie; keinerlei Aussage über tatsächliche fossile Nenner.
Exponent 2 und alle weiteren Zahlen dienen begrenzter Rechenbarkeit und klaren
algebraischen Invarianten; nicht aus den Ergebnissen von #39 ausgewählt.
Feste Referenzfaktoren erlauben einen bedingt identifizierbaren Grundfall:
`pi_1/pi_2=exp(4u)` und `pi_3/pi_4=exp(4v)`; dies ist kein Identifikationsbeweis
bei unbekannten realen Faktoren.

Zwei Gitterpriors, jeweils über alle 441 Zustände normiert: uniform und
`exp(-((u-0.4)²+(v+0.3)²)/(2*0.55²))`. Unabhängig von den erzeugten Daten.
Keine MCMC-/Parameterfits, exakte endliche Normierung mit Log-sum-exp.
Alle vorgesehenen Varianten berichten; keine Auswahl eines Gewinners.

## Festes Fallinventar

| Fall-ID | Eingänge und Varianten | Vorgesehene Posterioren |
| --- | --- | ---: |
| base | Wahrheit (0,0), (−0,5;0,5), (0,5;−0,5); N=8 und 40; Seeds 201,202,203; h=0; beide Priors | 36 |
| shared | Wahrheit (0,0), N=40, Seed 303. Ein H pro Probe, ±ln(2) mit Gewicht 1/2. Auswertung korrekte Mischung `0.5 Mult(pi_−)+0.5 Mult(pi_+)` und bewusst falsch `Mult(0.5 pi_−+0.5 pi_+)`, jeweils beide Priors | 4 |
| merged | Wahrheit (0,5;0), N=40, Seed 404; h=0; erste zwei Quellen auf denselben Typ zusammengelegt, Quellen 3/4 auf getrennte Typen, U auf R; beide Priors | 2 |
| missing_group | Fester ursprünglicher Countvektor (2,1,0,2,3), N=8. Nur Kategorien 1,3,4 beobachtet; 2 und R fehlend; gemeinsame fehlende Masse 4. Kollabierte Likelihood, beide Priors | 2 |
| conditional_subset | Derselbe ursprüngliche Vektor, nur Kategorien 1–4, ausdrücklich künstlicher bedingter Auswahlprozess; n_O=5 statt N=8, pi innerhalb O neu normiert; beide Priors | 2 |
| all_missing | Alle Kategorien fehlend, N=40 bekannt, ignorierbare Missingness; L=1, beide Priors | 2 |
| zero_total | Vollständig beobachtete fünf Nullen, N=0; L=1, beide Priors | 2 |
| observed_zero | Fester Vektor (0,2,2,2,2), N=8; h=0; beide Priors; Nullcount bleibt gültige Beobachtung | 2 |

**Insgesamt 52 geplante Posterioren**, noch null berechnet. Seeds werden je
Zustand/N wiederverwendet; keine unabhängigen Bestätigungsserien behaupten.
Generator: NumPy default_rng(Seed), im base/merged jeweils ein multinomialer
Ziehschritt; shared zuerst rng.choice aus [−ln2,+ln2], danach rng.multinomial.
Beobachtungen werden einmal je Fall erzeugt und von beiden Priors/Varianten
geteilt. Feste Vektoren und Masken werden ohne Zufallsersatz übernommen.
Eine missspezifizierte Auswertung wird nicht zum Daten erzeugenden Modell erklärt.

Bei shared stimmen die **Einzelkorn-Marginalen** beider Auswertungen exakt überein.
Der Unterschied ist ausschließlich, ob H pro Probe gemeinsam oder je Korn neu
gezogen wird. Das vermeidet den im #39-Vergleich dokumentierten gleichzeitigen
Wechsel von Abhängigkeit und marginalen Antwortkurven. Auch dieser gezielte
Vergleich ist keine Auswahl eines realen Fehler-/Abhängigkeitsmodells.

## Vorab definierte Rechen- und Strukturkriterien

Alle numerischen Identitäten mit absoluter Toleranz 1e−12; Fehler und nicht
schätzbare Fälle protokollieren, keine Parameter/Seeds nach Ergebnis ändern.

1. Für N=0 und N=8 an den drei bekannten Zuständen alle Countvektoren über
   fünf Kategorien enumerieren (N=8: 495 Vektoren). PMF-Summe gleich 1 für base
   und shared. Multinomialkoeffizient vollständig erhalten; keine Evidenzwerte
   aus einer nur proportionalen Likelihood vergleichen.
2. Separate Nachrechnung aller 52×441 Massen ohne Produzentenimport anhand
   direkter multinomialer Produkte und endlicher H-Summe. N≤40 und strikt
   positive Spielzeugwahrscheinlichkeiten erlauben diese begrenzte Nachrechnung;
   unterlaufende Fälle wären als solche auszuweisen, keine Toleranzlockerung.
3. Gemeinsame Mischung gegen Produkt gemittelter Einzelkornwahrscheinlichkeiten
   mit mindestens einem analytisch festgelegten Ereignis prüfen: bei c=0,
   N=2, beide Körner Typ 1. Korrekt ist `0.5*pi_−,1²+0.5*pi_+,1²`, im
   per-Korn-Modell `(0.5*pi_−,1+0.5*pi_+,1)²`; Differenz ist
   `Var_H(pi_H,1)>0`. Hier sind die beiden pi-Werte exakt 1/12 und 1/3;
   die Ereigniswahrscheinlichkeiten sind 17/288 bzw. 25/576, Differenz 1/64.
   Das sind analytische Sollwerte, keine ausgeführten Rechenergebnisse.
   Keine Behauptung, jeder zufällig gezogene Datensatz
   müsse ein engeres oder besseres Posterior erzeugen.
4. merged: Likelihood bei (u,v) und (−u,v) gleich; uniformer Posterior
   symmetrisch, informativer Prior kann die Symmetrie brechen. Identifikation
   des Vorzeichens nicht aus der mittleren Fehlerzahl ableiten.
5. missing_group: kollabierte Formel muss der expliziten Summe über alle
   möglichen Aufteilungen der vier fehlenden Körner auf Typ 2 und R entsprechen.
   O leer ergibt L=1. conditional_subset separat nach derselben künstlichen
   Auswahlregel erzeugte Wahrscheinlichkeitsform prüfen; keine Gleichsetzung
   mit der vollständigen oder kollabierten Likelihood verlangen.
6. all_missing/zero_total: posterior=prior, Log-Evidenz=0. observed_zero:
   gültig und ohne biologische Absenzbehauptung; fehlend und 0 nicht austauschbar.
7. Algebraische Äquivalenzen aus DESIGN_COUNTS zusätzlich numerisch prüfen:
   globale Flussskalierung ×7; c=(0,0), r=1 gegen c'=(0.5,0),
   r'=(exp(−1),exp(1),1,1,1); identische A-Spalten mit einer zulässigen
   Flussverschiebung delta=0.05 zwischen Quellen 1/2. Kein Fit freier Faktoren.
8. Isolierte Eingabeprüfungen: negative/nichtganzzahlige Counts, falsche Summe,
   fehlender/unklarer Nenner ohne zulässigen Prozess, ungültiges A, negative
   Flüsse, Lambda=0, falsche Typdimension, NaN/Inf und c außerhalb jeder
   Supportkante abweisen. Bei pi_k=0 gilt für y_k=0 Faktor 1, für y_k>0
   Likelihood 0. Künstlicher Fall, in dem **alle** Gitterzustände für eine
   positive beobachtete Kategorie pi=0 liefern: beide Priors als nicht schätzbar
   mit Modellkonflikt ausweisen; kein posterior und keine Epsilon-Reparatur.
   Dieser negative Test gehört nicht zu den 52 gültigen Posteriorszenarien.

Für alle 52 Fälle vollständige Massen, Mittel, marginale 95%-Quantilbereiche,
95%-HPD-Set mit allen Ties, Randmasse, Log-Evidenz, Nenner/Maskenstatus und
bei bekannter Wahrheit Mittelabstand/Wahrheitsmasse/HPD-Einschluss speichern.
Methodik der endlichen Sets wie im unveränderten
[binären Design](../t1_synthetic_t2_2026_10_31/DESIGN.md).
Kein verlangter Mindestanteil „erfolgreicher“ Rekonstruktionen; alle
Verschlechterungen und Rand-/Priorabhängigkeiten berichten. Kein SBC oder
nomineller Deckungsnachweis; ein endlicher Übereinstimmungstest ist kein
empirischer Rekonstruktionsfehler für Fossilien.

## Grenzen des späteren Implementierungsauftrags

Nur neuer Forschungsprototyp und Tests, keine Änderung alter Designs/Modelle,
Fossilcounts, taxonomischer Gruppen, Quellregister, Baseline oder CLI/App.
Plan und ausführbaren Quellstand vor Ergebnislauf committen; bekannte Planlücke
vorher dokumentieren, keine wissenschaftliche Nachjustierung nach Ergebnis.
Neue Pfade: tools/taxon_counts/, tests/test_synthetic_counts.py und ein neues
Ergebnispaket docs/thesis_research/counts_synthetic_t2_2026_10_31/;
Status/Übergabe als einzige bestehende Dokumente fortschreiben.
Fünf Produktionsdateien wie beim binären Paket: observations.json, results.json,
posterior.csv.gz, environment.json, manifest.json. Separate Prüf-/Wiederholungs-
protokolle daneben. Bytegleiche Wiederholung in gleicher Umgebung, kompakte
Beobachtungen und alle Massen direkt in Git sichern; eigener PR und Folgeauftrag.
Keine Wiederholung
des gesamten Standardtests, keine realen Fits oder Kernläufe, keine Unteragenten,
Autorennachrichten oder Merges. W96 und reale Eingangsfreigaben unverändert offen.
