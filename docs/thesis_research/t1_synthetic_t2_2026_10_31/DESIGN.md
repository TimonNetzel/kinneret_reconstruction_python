# T1: vorab gesichertes Design für einen begrenzten synthetischen T2-Test

31.10.2026. Autorisierter Folgeauftrag aus PR #38; Basis
`d003fb97a21c967a44161230eed807e745ed5b0b`. Eigener Branch
`codex/netzel-t1-synthetic-t2`. Dieses Dokument und config.json werden **vor
Implementierung und Ergebnislauf separat committed und gepusht**. Keine Fits,
keine Auswahl anhand bekannter äußerer Scores, keine Unteragenten.

## Aussageziel und Datenrollen

T1 liefert hier einen mathematisch vollständigen **synthetischen Kandidaten**,
T2 eine kleine technische/diagnostische Prüfung. Das ist keine Wahl der späteren
fossilen Hauptmethode und kein abgeschlossenes T1 für reale Pollenspektren.
Die vorhandenen sieben Funktionen bedeuten `P(kartiertes natives Areal | Klima)`
unter Gebiet, Kartenregeln und Prävalenz aus
[DESIGN](../taxon_transfer_2026_10_30/DESIGN.md) und
[config](../taxon_transfer_2026_10_30/config.json). Sie beschreiben weder
lokales Vorkommen noch Pollenproduktion, Häufigkeit, Transport oder Erhaltung.
Sie werden hier weder geladen, angepasst, verglichen noch ausgewählt.
Eine Produktbildung ihrer Werte wäre ohne weiteres Beobachtungsmodell keine
Pollenspektrum-Likelihood. W96 und die sieben fossilen Artbrücken bleiben
[offen](../pollen_methods_2026_10_31/SOURCE_GAP.md).

Eingänge sind ausschließlich feste Spielzeugparameter und künstliche binäre
Beobachtungen vier **anonymer Kanäle**. Kein Kanal steht für eine echte Art.
Alle Beobachtungsparameter sind bekannt gesetzt, nicht geschätzt. Alle
Zustände/Seeds/Varianten stehen vorab in config.json. Kein Training, keine
Kalibrierungs- oder Holdoutdaten; der Generator ist mit dem passenden
Inferenzmodell bewusst identisch. Das prüft Rechenlogik unter Annahmen,
keine Übertragbarkeit auf unbekannte Daten. Replikate sind unabhängige
künstliche Beobachtungseinheiten desselben Zustands, keine Pollenkörner,
keine Zeitfolge und keine aus einer Probe ableitbare effektive Stichprobengröße.

## Generatives Modell und Inversion

Zustand `c=(u,v)` auf 41×41 gleichabständigen Punkten in `[-1,1]²`, dimensionslos;
keine Celsius-/mm-Interpretation. Gleiches festes Gitter für alle Fälle.
Vier logistische Funktionen `p_j(c,h)=sigmoid(a_j(c)+h)` mit
`a=(4u,-4u,4v,-4v)`. Das Gegenpaar macht Widersprüche sichtbar und beide Achsen
prinzipiell unterscheidbar. Feste Steilheit 4: deutliche, endliche Reaktion im
Spielzeugfenster, kein aus Daten optimierter Wert.

Je Replikat r: latentes `Z_rj ~ Bernoulli(p_j(c,H_r))`, bedingt unabhängig über j.
`P(Y_rj=1|Z_rj=1)=s=0.9`, `P(Y_rj=1|Z_rj=0)=f=0.1`.
Nach Integration von Z ist `q_j=f+(s-f)p_j` und
`P(Y_rj=y|c,h)=q_j^y (1-q_j)^(1-y)`. s/f sind illustrative Detektionsfehler,
keine geschätzten palynologischen Raten. Beobachtetes 0 ist nur im generativen
Modell eine valide Nichtdetektion. Fehlend wird ausgelassen, niemals als 0 kodiert;
die Missingness ist hier fest gesetzt und nicht informativ.

Zwei vollständig spezifizierte Modelle, ohne Wahl eines Gewinners:

- **independent:** `H_r=0`; Produkt der Bernoulli-Terme über beobachtete j und r.
- **shared:** `P(H_r=-1.5)=P(H_r=+1.5)=1/2`, gemeinsamer H über alle Kanäle
  desselben Replikats, unabhängig über r. Pro r zuerst
  `L_r(c)=0.5 Π_j b(Y_rj;q_j(c,-1.5))+0.5 Π_j b(Y_rj;q_j(c,+1.5))`,
  danach `L=Π_r L_r`. Nicht getrennt je Kanal mitteln: dies verlöre die
  Abhängigkeit. Fehlende Kanäle vor Produkt/Mischung entfernen; leere Zeile hat L=1.

Bayes auf endlichem Zustandsraum: `w(c)=π(c)L(c)/Σ_grid πL`, Logarithmen und
Log-sum-exp zur Stabilität. Kein MCMC und kein Konvergenzdiagnostik-Ersatz nötig.
Keine zusätzlichen Areal-, Analog-, Referenzproxy- oder Chronologieterme,
keine mehrfach benutzte Evidenz. Die Gleichungen folgen direkt aus
Bernoulli-Modell, totaler Wahrscheinlichkeit über Z/H und Bayes; keine neue
Literaturbehauptung oder Nachbildung einer Dissertationsformel.

Zwei vorab gesetzte Prior-Massen (jeweils auf dem Gitter normiert): uniform
und `exp(-((u-0.4)²+(v+0.3)²)/(2*0.55²))`. Letzterer provoziert nachvollziehbare
Priorabhängigkeit. Keine geographische Klimahäufigkeit als zusätzliche Likelihood.
Massen betreffen ein **diskretes** Modell, keine Behauptung kontinuierlicher
Quadraturgenauigkeit. Support ist der feste Spielzeugraum, keine Hülle realer
Trainingsdaten. Vorwärtsabfragen außerhalb oder mit NaN/Inf werden abgewiesen.
Eine am Rand konzentrierte Inversion entdeckt unbekannte Extrapolation nicht
zuverlässig; aus Y allein lässt sich ein außerhalb liegender wahrer Zustand
nicht sicher erkennen. Reale gemeinsame Supportprüfung bleibt offen.

## Vorab festgelegte Fälle und Auswertung

Je Zustand `(0,0),(-0.5,0.5),(0.5,-0.5)` und Seed 101–105 werden 40 Replikate
generiert. Immer beide Priors berichten:
independent-Generator→independent-Inversion (30 Auswertungen),
shared-Generator→shared und bewusst falsch independent (60 Auswertungen).
Die letzteren 30 zeigen Missspezifikation; keine Auswahl nach deren Ergebnis.
Weitere feste Fälle, ebenfalls beide Priors, soweit nicht anders genannt:

| Fall | Vorab festgelegtes Kriterium / Interpretation |
| --- | --- |
| Bekannte Zustände, 90 Auswertungen | Posterior-Mittel/Fehler, beide marginalen 95%-Quantilbereiche, wahre Gittermasse im 95%-HPD-Set, HPD-Größe, Randmasse und Log-Evidenz vollständig berichten. Keine verlangte Erfolgsquote; Fehlschläge erhalten. 15 Datensätze je Generator sind kein Deckungsnachweis. |
| Rand `(−1,1),(1,−1)`, independent, Seed 505 | Beide Priors, 4 Auswertungen; Randmasse (äußerster Gitterring) offen berichten, keine automatische Sperrschwelle daraus lernen. |
| Widerspruch | 40 Zeilen, u-Gegenpaar immer beide 1, v fehlt: Likelihood kann in u konzentrieren. Max. unabhängige Doppelpositivchance über Support ≤0.25; obere Schranke `max(q0*q1)^40` für 40/40 Paare berichten. Konfliktmarkierung bei <0.001, unabhängig vom Posterior-Peak. Kein allgemeiner Modellgütetest. |
| Ein Kanal fehlt | independent-Daten bei (0,0), Seed 101, Kanal 3 fehlt; zwei gültige Posterioren, Beobachtungszahl 120 statt 160. Kein automatischer Ausschluss. |
| Nur u-Gegenpaar | Dieselben Daten, Kanäle 2/3 fehlen: v-Likelihood exakt flach, v-Posterior gleich v-Prior (max. Abweichung ≤1e−12). Nicht identifizierbar in v. |
| Alles fehlt | Zwei Priors × beide Modelle, alle vier Kanäle fehlend: Posterior gleich Prior ≤1e−12, Log-Evidenz 0 ≤1e−12. Keine Klimainformation. |
| Detektion uninformativ | s=f=0.1, beliebige feste independent-Daten: Posterior gleich Prior ≤1e−12. Fehlerfreie numerische Ausgabe ist keine Identifikation. |
| Gefaltetes u | Nur hier `a0=4(u²−0.25), a1=−a0`, v unverändert. independent-Daten bei (0.5,0), Seed 606. Gleiche Likelihood für ±u ≤1e−12; uniformer Posterior symmetrisch ≤1e−12. Informierter Prior kann die Symmetrie brechen. Kein einzelner Mittelwert als eindeutiger Zustand; volle Massen sichern. |
| Ungültige Eingänge / Support | Punkte jenseits jeder Supportkante, NaN/Inf; Y außerhalb {0,1,fehlend}, falsche Form, ungültige s/f und Modell-/Priorbezeichner: explizite Fehler, kein Clipping oder stilles Auslassen. |

95%-HPD-Set: nach Masse absteigend, bis mindestens .95; alle Zellen mit Masse
gleich der Grenzmasse einschließen (keine willkürliche Tie-Brechung).
Marginale Quantile: erstes Gitterzentrum mit kumulativer Masse ≥.025/.975.
Keine nominell validierten Konfidenzintervalle, kein SBC-Auftrag/-Nachweis.
Alle Kriterien sind diagnostisch oder numerisch, kein T0/T3/T4-Freigabegate.

## Prüfung, Speicherung und Grenzen

Neuer isolierter NumPy-Prototyp unter tools/taxon_synthetic; vorhandene Pakete,
Modelle, Register, Ergebnisse, Counts, Baseline und CLI/App bleiben bytegleich.
Gezielte Tests prüfen H/Z-Marginalisierung durch direkte Zustandsenumeration,
fehlerhafte Eingänge, Missingness, beide Priors, Symmetrie und Konfliktlogik.
Separates Prüfskript **ohne Produzentenimport** berechnet alle Posteriormassen
mit direkter Produkt-/Mischungsformel neu, Toleranz 1e−12; gemeinsame Abhängigkeit
NumPy/Python offen benennen. Kein unabhängiger menschlicher Review.
Frische zweite Ausführung muss alle Ergebnisdateien bytegleich erzeugen.
Beobachtungen, kompakte Diagnosen, vollständige diskrete Posteriormassen,
Konfiguration, Laufumgebung und Hashmanifest in Git sichern. Kein nur temporär
behauptetes Artefaktarchiv. Numerische/Implementierungsfehler dürfen repariert
werden, jede Designabweichung dokumentieren; keine nachträgliche Anpassung der
Zustände, Steilheiten, Seeds, Priors oder Kriterien an Resultate.

Enthaltene Unsicherheit: künstliche Bernoulli-Beobachtung, vorgegebener gemeinsamer
H und diskreter Zustand unter festem Prior. Fehlend: Transferfunktionsfehler,
Kreuztaxon-Abhängigkeit jenseits dieses H, Pollenproduktion/-transport/-erhaltung,
Zählkomposition, echte Detektionsraten, Karten-/Label-/Quellenunsicherheit,
Standort-/Zeitdynamik, Klimaauflösung, Chronologie und Art-/Typzuordnung. Keine
Übernahme der #32-Bänder als vollständige Rekonstruktionsunsicherheit.
Reale Modelle später nur nach gesondertem Beobachtungsmodell-/Supportdesign;
synthetischer Erfolg überbrückt W96 und Monticchio-P1 nicht.
