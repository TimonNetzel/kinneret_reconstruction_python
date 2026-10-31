# T1-Anschlussdesign: Pollentyp-Zählungen und unbekannte Mischungen

31.10.2026, autorisierter Dokumentationsauftrag aus HANDOFF bei
`35185534bf7f171c6f39537ebbc0449b28f5421a` (PR #39). Dieses Paket enthält
Gleichungen, Datenvertrag, algebraische Gegenbeispiele und einen **späteren**
synthetischen Prüfplan. Keine Implementierung, neuen Fits oder Ergebnisläufe.
Keine vorhandene Funktion oder fossile Zuordnung wird dadurch freigegeben.

## 1. Bedeutung der Eingänge

Die sieben bestehenden Funktionen `m_j(c)` modellieren einen kartierten nativen
Arealindikator unter ihrem festgelegten räumlichen Gebiet und dessen Prävalenz.
Sie modellieren nicht die lokale Vegetationsmenge und nicht die Wahrscheinlichkeit,
dass ein gezähltes Pollenkorn zu Art j gehört. Geltungsbereich, Kartenlabels und
WorldClim-Bezug bleiben im unveränderten
[Artenfunktionsdesign](../taxon_transfer_2026_10_30/DESIGN.md) und dessen
[Konfiguration](../taxon_transfer_2026_10_30/config.json) definiert.
Weder `m_j` noch ein Produkt von `m_j` wird hier zur Counts-Likelihood erklärt.
Für reale Daten bleibt die Brücke unten ein **unvollständig belegter Vertrag**.

Für eine einzelne Probe s bedeuten:

| Symbol | Bedeutung und Rolle |
| --- | --- |
| c | Klima, später mit eindeutigem Variablen-, Einheiten- und Periodenvertrag; TANN/PANN nicht mit TDJF oder dimensionslosem Spielzeugraum gleichsetzen |
| Z_j | optionaler lokaler Vorkommensindikator einer botanischen Quelle j; kein aus Nullcounts abgeleitetes Label |
| V_j ≥ 0 | latente lokale Vegetationsbeiträge; bei Flächenanteilsinterpretation mit Summe 1, einschließlich sonstiger Quellen; nur bei lokalem Vorkommen positiv |
| G_eta(dV\|c) | gemeinsames Vegetationsmodell; enthält bei Bedarf Z, Standort und Abhängigkeiten, muss normiert und belegt oder ausdrücklich künstlich gesetzt sein |
| r_j ≥ 0 | relativer Produktions-/Eintragsfaktor der lokalen Quelle j im festgelegten Referenzsystem |
| B_j ≥ 0 | zusätzlicher Hintergrund-/Ferntransportfluss derselben botanischen Quelle; erlaubt Pollen ohne lokale Präsenz |
| d_j ∈ [0,1] | vor Klassifikation wirksame Erhaltung/Detektion im gezählten Nenner; keine in diesem Paket geschätzte Rate |
| A_kj ≥ 0 | bedingte Klassifikationswahrscheinlichkeit eines erhaltenen, zum Nenner gehörenden Korns aus botanischer Quelle j in beobachteten Typ k |
| y_k, N | nichtnegative ganzzahlige Typcounts und belegter Zählnenner; N ist hier konditioniert, kein gemessener absoluter Pollenfluss |
| eta | sämtliche Modell-/Brückenparameter einschließlich r, B, d, A und G-Parameter; bekannte Werte, externe Information oder offene Unbekannte getrennt ausweisen |

j umfasst auch nicht modellierte botanische Quellen U. Ein breiter Fossiltyp
ist eine Beobachtungskategorie k, keine einzelne Art. Auch ein nicht modelliertes
Taxon kann auf einen ausgewählten Typ einzahlen. Eine Restkategorie R der Counts
ist wiederum nicht identisch mit der botanischen Sammelquelle U.
Die Zahl J der Quellen ist daher nicht automatisch sieben und nicht gleich K.

## 2. Fluss, Typmischung und Komposition

Für jede Quelle ist der effektive Fluss im Zählnenner

```text
x_j(V,eta) = d_j [r_j V_j + B_j]
lambda_k(V,eta) = sum_j A_kj x_j
Lambda = sum_k lambda_k
pi_k = lambda_k / Lambda
```

A ist spaltenstochastisch: `sum_k A_kj = 1`. Klassifikationsunsicherheit wird
damit über **alle im Nenner gezählten**, disjunkten Kategorien einschließlich R
verteilt. Verluste/Ausschluss außerhalb des Nenners gehören in den belegten
Prozess vor A (hier d), nicht als unbemerkt fehlende Spaltenmasse in A.
Produktions-/Transportfaktoren haben ein gemeinsames Referenzmaß; `rV` und B
müssen dieselben Flusseinheiten haben. Ohne dessen Definition bleibt die reale
Parametrisierung gesperrt. d und r werden aus dieser Komposition nicht getrennt
geschätzt. Eine deterministische Art→Typ-Zuordnung wäre ein Spezialfall von A,
der durch Material-/Methodenbelege zu rechtfertigen ist, nicht durch Namensähnlichkeit.

Ein Zustand ist nur gültig bei endlichen, nichtnegativen Größen, gültigem A,
normiertem G und `Lambda > 0` fast sicher unter G. Keine Normalisierung eines
Nullflusses, kein Ersetzen durch Gleichverteilung, keine stillen Epsilon-Counts.
Unsichere A-Matrizen werden über eine vorab begründete Verteilung integriert;
ein unbekannter Parameter wird nicht durch einen scheinbar bekannten Mittelwert
ersetzt. Bloß alle Typen auf R zu legen kann normalisierte Wahrscheinlichkeiten
erzeugen, belegt aber weder Abdeckung noch Klimainformation.

## 3. Zählregel und Inversion

Minimaler **synthetischer** Kandidat, bedingt auf eine Probenkomposition und N:

```text
Y | N,V,c,eta ~ Multinomial(N, pi(V,eta))
P(y | N,V,eta) = N! / product_k y_k! * product_k pi_k^y_k
L(c,eta) = integral P(y | N,V,eta) G_eta(dV | c)
posterior(c,eta | y,N,E) ∝ prior_C(c) prior_eta(eta | E) L(c,eta)
```

E bezeichnet unabhängige, eindeutig zugeordnete externe Brückeninformation.
Die geschriebene Priorfaktorisierung setzt hier vorab unabhängige Klima-/eta-Priors
gegeben E voraus; bei gemeinsam informierten Größen wäre stattdessen ein proper
gemeinsamer Prior zu spezifizieren. Auf dem gemeinsam gültigen Support normieren.
Alle Priors müssen proper/normierbar sein; bei unbekannter G-/eta-Verteilung
steht **kein ausführbarer realer Posterior** fest. Eine Annahme allein ist kein
empirischer Nachweis. Ein Produkt unabhängiger Artenwahrscheinlichkeiten kann
G nicht ohne begründete lokale Präsenz-/Abundanzannahmen ersetzen. Insbesondere
ist `G(V|c)=Multinomial(..., m(c))` weder hier definiert noch erlaubt.

Die multinomiale Regel setzt bedingt unabhängige Klassifikation einzelner Körner
bei derselben Probenkomposition voraus. Sie ist die kleinste normierte Zählregel
für den späteren Rechentest. Für reales Material können Cluster, Präparation,
Transport und zeitliche Mischung zusätzliche Streuung erzeugen. Eine
Dirichlet-Multinomial-Erweiterung hätte z. B. Konzentration kappa und bedingte
Varianz `N pi_k(1-pi_k) (N+kappa)/(1+kappa)`; kappa benötigt eigene Information.
Sie wird hier **nicht** eingeführt, angepasst oder anhand äußerer Scores gewählt.
Multinomial ist damit eine Testannahme, keine bestätigte reale Fehlerverteilung.
Die Normierung folgt aus dem Multinomialsatz: die Summe der PMF über alle
Countvektoren mit Summe N ist `(sum_k pi_k)^N=1`; eine normierte endliche Mischung
solcher PMFs bleibt normiert. Der endliche Prüfplan kontrolliert diese Identität.

Das gesamte G wird **vor** der Posteriornormierung integriert. Bei einem
probenweit gemeinsamen H mit Gewichten w_h gilt `L = sum_h w_h Mult(y;N,pi_h)`.
`Mult(y;N,sum_h w_h pi_h)` beschreibt einen anderen Prozess mit neuem H je Korn.
Es genügt nicht, Mittelwerte von Vegetation/Fluss in die Normalisierung einzusetzen.
Die Summe über gemeinsame eta/Modelle erfolgt ebenfalls außerhalb der passenden
Likelihoodprodukte; dieselbe Probe liefert keine unabhängigen Typ-Likelihoods.
Schon bei fester pi gilt für k ≠ l `Cov(Y_k,Y_l|N,pi)=-N pi_k pi_l`.
Gemeinsame latente Komposition kann darüber hinaus positive Kovarianz erzeugen.

Hier nur eine Probe pro Inversion. Mehrere Proben sind erst nach Festlegung ihrer
Klimazustände, Chronologie, gemeinsamer Parameter und Abhängigkeiten kombinierbar.
Kein Produkt über Zeitproben oder zusätzliche Analog-/Proxy-/Areal-Likelihood
ohne gesonderten Datenrollenvertrag; externe Information nicht doppelt verwenden.
N wird nicht gleichzeitig als Poisson-Flussbeobachtung benutzt. Ein stoppender
Zählprozess bei N Körnern liefert gerade keine absolute Eintragsintensität.

## 4. Nullcount, fehlend, Rest und unbekannter Nenner

- **Beobachtetes y_k=0:** kein Korn dieses Typs gezählt; keine botanische Absenz.
  Nur gemeinsam mit den anderen Counts/N trägt dies Kompositionsinformation.
  Für strukturelles `pi_k=0,y_k=0` gilt der Faktor 1; `pi_k=0,y_k>0` ergibt L=0.
  Null-Likelihood einzelner Zustände ist zulässig; L=0 für alle gültigen Zustände
  bedeutet Modell-/Datenkonflikt und Abbruch, keine positive Bodenwahrscheinlichkeit.
- **Echter vollständiger N=0-Datensatz:** alle Counts 0; bei gültigem Modell
  ist L=1 und der Posterior gleich Prior. Eine leere/fehlende Probe ist dagegen
  kein belegter N=0-Datensatz. Bei ganz fehlender, ignorierbar fehlender Beobachtung
  wird L=1 gesetzt und als fehlend protokolliert, nicht mit N=0 umcodiert.
- **Rest R:** `y_R=N-sum(y_ausgewählt)` nur, wenn derselbe belegte Nenner gilt,
  die ausgewählten Kategorien disjunkt enthalten sind und der Rest tatsächlich
  die verbleibenden gezählten Kategorien umfasst. Negative oder nicht ganzzahlige
  Differenz, Prozentwerte ohne Counts oder unklarer Nenner sperren den Countspfad.
  Nichtpollen, ausgeschlossene Gruppen und ungezählte Körner nicht still in R füllen.
- **Einzelne Counts fehlen, N bekannt:** marginalisieren, nicht nullsetzen.
  Für beobachtete Kategorien O und `m=N-sum_{k in O} y_k` lautet die exakte Regel
  `N!/(m! product_O y_k!) * (1-sum_O pi_k)^m * product_O pi_k^y_k`.
  Dies ist eine kollabierte Multinomialverteilung mit einem gemeinsamen fehlenden
  Rest, keine Aussage über die einzelnen fehlenden Typen. Bei O leer ist L=1.
  Eine bereits beobachtete Restkategorie nicht zugleich als fehlende Masse zählen.
- **Nur ausgewählte Typen, Gesamt-N unbekannt:** erst bei belegtem Zähl-/Auswahl-
  prozess bedingt auf `n_O=sum_O y_k` die eingeschränkte Komposition
  `pi_k/sum_O pi_k` verwenden. Damit wird sämtliche Evidenz über den Anteil
  außerhalb O aufgegeben. Fehlende Daten allein rechtfertigen diese neue
  Konditionierung nicht. Ohne belegten Prozess kein realer Counts-Lauf.

Missingness ist hier als ignorierbar/fest maskiert angenommen. Hängt sie von
Typ, Erhaltung oder Klima ab, ist ein zusätzliches Beobachtungsmodell nötig.

## 5. Algebraische Nichtidentifizierbarkeit

Diese Gegenbeispiele sind Ableitungen, keine simulierten Resultate.

**Absolute Skala.** Für a>0 gilt `pi(a*x)=pi(x)`. Bedingt auf N haben x und a*x
genau dieselbe Verteilung aller Countvektoren. Weder mehr Körner noch ein engerer
Posterior liefern die absolute Flussskala. Ein Referenzfaktor ist eine notwendige
Parametrisierungskonvention, keine aus den Counts gemessene Skala.

**Produktion gegen Vegetation/Klima.** Bei B=0 und d=1 hängt x nur vom Produkt
r_j V_j ab. Für positive V(c), V(c') setze
`r'_j=r_j V_j(c)/V_j(c')`: dieselben x, A und pi bei verschiedenem Klima.
Ein simplexnormiertes V hebt diese Mehrdeutigkeit nicht auf. Konkretes
Spielzeugbeispiel: `V(c)=softmax(2u,-2u,2v,-2v,0)`. Bei c=(0,0), r=(1,1,1,1,1)
ist pi bei Identitäts-A uniform. Bei c'=(0.5,0) und
`r'=(exp(-1),exp(1),1,1,1)` sind erneut alle fünf effektiven Flüsse gleich,
also dieselbe uniforme pi. Freie Produktionsfaktoren können daher ein anderes
Klima vollständig kompensieren. Proper Priors können einen Posterior bestimmen,
beseitigen aber diese fehlende Identifikation durch die Daten nicht.

**Breiter Pollentyp.** Haben zwei Quellen identische A-Spalten, so ändert
`(x_1,x_2)→(x_1+delta,x_2-delta)` bei zulässigem delta keinen Typfluss.
Keine Aufteilung eines Gattungscounts auf zwei Arten allein aus diesen Counts.
Beim Zusammenlegen der ersten beiden Softmaxquellen ist ihr Typbeitrag
proportional `exp(2u)+exp(-2u)=2 cosh(2u)`. u und −u sind bei sonst gleichen
Größen exakt ununterscheidbar, unabhängig von N. Für unbekannte A kommen
Spaltenpermutationen samt vertauschten Quellen als weitere Mehrdeutigkeit hinzu.
Auch voller Spaltenrang eines **bekannten** A garantiert für sich keine globale
Identifizierbarkeit der Klima-/Nuisanceabbildung.

**Lokaler gegen Hintergrundfluss.** Bei konstantem d bleibt `rV+B` gleich,
wenn ein zulässiger Teil von rV nach B verschoben wird. Pollenpräsenz beweist
damit weder lokale Vegetation noch ein bestimmtes Klima. Relative Komposition
trennt Produktion, Transport, Erhaltung und fehlende botanische Quellen nicht.

Für spätere reale Inferenz braucht es ausreichend unabhängige externe Information
zu genau diesen Freiheitsgraden, festgelegte Referenzkonventionen und eine Prüfung,
ob `c→Verteilung(Y|N,c,E)` im zulässigen Raum überhaupt unterscheidbar ist.
Ein informativer Prior oder eine kleine Optimierungsfehlertoleranz ist dafür
kein Ersatz. Bedingte Identifikation bei fest gesetzten Parametern muss als solche
benannt bleiben; breite/mehrgipflige Posterioren vollständig erhalten.

## 6. Support, Unsicherheit und Datenvertrag

Realer gemeinsamer Klimasupport wäre die vorab definierte Schnittmenge der für
alle tatsächlich benötigten Funktionen gültigen Trainingshüllen und weiterer
Brücken-Geltungsbereiche; Einheiten und Bezugsperioden müssen zuvor harmonisiert
sein. Fehlende Modelle/Zuordnungen nicht durch Score-Auswahl, Nullwahrscheinlich-
keiten oder extrapolierte Einzelarten ersetzen. Ein Modell mit nicht bestätigter
Pollenrolle besitzt durch eine passende Klimahülle noch keinen gültigen Fossilsupport.
Supportprüfungen betreffen auch zulässige A-/d-/r-/B-/G-Parameter und den Zählnenner.

Erforderlicher Probenvertrag vor realer Rechnung:

| Feldgruppe | Muss belegt sein |
| --- | --- |
| Identität | Kern, Probe, Tiefe, Zählobjekt, Version/Hash, belegte Alter-/Chronologierolle, Änderungen/Originalfelder erhalten |
| Counts | ganzzahlige Counts oder ausdrücklich andere Datengattung, N samt Nennerdefinition, ausgewählte/Rest-/ausgeschlossene Kategorien, disjunkte Partition, Status je Feld (beobachtet, strukturell, fehlend) |
| Bestimmung | beobachtete Typdefinition, verwendetes Protokoll/Schlüssel/Materialumfang; A-Support und dessen Beleg/Unsicherheit; keine künstliche Verfeinerung archivierter Counts |
| Vegetation und Eintrag | G-/r-/B-/d-Annahmen oder externe Daten samt räumlicher/zeitlicher Reichweite, Referenzmaß und identifizierenden Beschränkungen |
| Klima | Variablen/Einheiten/Periode, Funktionen/Parameter/Support; Labelrolle der Kartenfunktionen ausdrücklich erhalten |
| Inferenz | proper Klimaprior und Nuisanceprior, Datenrollen ohne Doppelnutzung, geplante Diagnosen, nicht identifizierbare Richtungen, Version und Genehmigungsstatus |

Ein späterer Test mit festen r/A/G hat nur Zähl- und vorgegebene latente
Unsicherheit. Nicht enthalten: reale Transferfunktions-/Karten-/Labelunsicherheit,
Produktions-/Transport-/Erhaltungsfehler, Taxonomie, Chronologie, Standort-/Zeitdynamik,
Klimaauflösung und unbekannte Quellen. #32-Bänder sind bedingte Sensitivität und
werden nicht zu vollständigen Rekonstruktionsintervallen umgedeutet.
Gemeinsame Brückenunsicherheit muss später gemeinsam integriert werden;
unabhängiges Ziehen je Typ würde Kovarianz entfernen.

## 7. Nächster ausführbarer Schritt – hier nur vorbereitet

[SYNTHETIC_PLAN.md](./SYNTHETIC_PLAN.md) und [synthetic_plan.json](./synthetic_plan.json)
legen einen separaten, kleinen Zähltest fest. Nur dort ausdrücklich künstlich
gesetzte Größen sind bekannt; keine echte Art oder Datenbankkategorie beteiligt.
Das hier autorisierte Designpaket führt diesen Test **nicht aus**.
Reale fehlende Eingänge und Freigabesprünge stehen in [INPUT_GAPS.md](./INPUT_GAPS.md).
W96, T0, reales T1/T3/T4 und Monticchio-P1 bleiben offen.
