# A3: gesperrtes Design einer endlichen synthetischen Methodenprüfung

Stand 31.10.2026; Basis [PR #53](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/53),
`ea3157068e53a765d77ded5214ddaf29a732e5b1`.
**Durch Nutzer-OK autorisiert ist ausschließlich dieses Design.** A3 hat damit
als Dokumentation begonnen; Implementierung, Datenerzeugung und Ausführung
sind nicht erfolgt. A3 insgesamt ist nicht abgeschlossen. Keine Ergebnisse,
reale Modell-/Datenwahl oder Gatefreigabe werden behauptet.

## Auftrag, Quellen und lokale Annahmen

Maßgeblich sind [Plan §1](./MODULAR_RECONSTRUCTION_PLAN.md#1-ziel-und-festgehaltene-arbeitsrichtung),
ausschließlich die **A3-Zeile** in [Plan §2](./MODULAR_RECONSTRUCTION_PLAN.md#2-arbeitspakete-mit-prüfbarem-ende)
und [Arbeitsmodus §4](./MODULAR_RECONSTRUCTION_PLAN.md#4-tokensparender-arbeitsmodus).
Der unveränderte [MODEL_CONTRACT](./MODEL_CONTRACT.md) liefert die Regeln,
keine bereits ausgewählte reale Likelihood. Die folgenden endlichen Räume,
Zahlen, Verteilungen und Beobachtungen sind **neu festgelegte lokale
synthetische Prüfannahmen**, keine Literaturbefunde oder Ergänzungen seiner
realen OFFEN-Felder. Die Namen E/P/Y/D/C/psi/tau/theta bezeichnen hier nur
künstliche Entsprechungen der Vertragsrollen. Keine realen Klimaeinheiten,
Taxa, Archive, Trainings- oder Fossildaten verwenden.

Ziel ist ein vollständig aufzählbares Gegenprüfsystem für Informationsfluss,
Normierung, gemeinsame Unsicherheit und erkennbare Grenzen. Es beansprucht
keine realistische Ökologie, stetige Zeitprozessmodellierung, Samplerleistung,
Wahrheitsrückgewinnung aus jeder verrauschten Beobachtung oder nominelle
Intervallabdeckung. Dokumentiert werden Definitionen und erwartete Identitäten;
ihre numerische Nachrechnung gehört erst zum separat zu beauftragenden Paket.

## Fester künstlicher Zustandsraum und Evidenzrollen

Alle unten angegebenen Wahrscheinlichkeiten sind exakt rationale Zahlen.
Grundraum: C=(C1,C2,C3) mit Ct in {-1,+1}; psi in {-1,+1};
theta in {1,2}; tau in {a,b}. Keine Hyperparameter werden geschätzt oder gesucht.
Zeitindizes 1,2,3 sind künstliche dimensionslose Indizes in aufsteigender
Reihenfolge; Probentiefen sind lediglich die geordneten Labels d1,d2.

| Rolle | Gesperrte Definition im Grundfall |
| --- | --- |
| E / psi | Ein eigenes künstliches Bernoulli-Ereignis E=1, Prior psi je 1/2; Pr(E=1 bei psi=-1)=1/3 und bei psi=+1=2/3. q(psi) ist der daraus definierte moderne Posterior, also q(-1)=1/3, q(+1)=2/3. E geht ausschließlich hier ein. |
| C | Pr(C1=+1)=3/4; Pr(Ct=C(t-1))=3/4, andernfalls 1/4, für t=2,3. Das Produkt definiert den normierten Markov-Prior pC. Kein nachträglicher Glätter. |
| tau | Eine einzige Kurve je Archiv: a ordnet (d1,d2) auf (1,2), b auf (2,3). Prior je 1/2. Beide sind geordnet; separate Altersziehungen je Probe sind nicht Teil des Supports. Schreibe kj(tau) für den Index der Probe j. |
| D | Ein separates Bernoulli-Ereignis D=1; Pr(D=1 bei a)=3/4, bei b=1/4. Einmaliger Datierungsfaktor LD(tau); bei abwesendem D gilt LD=1. Kein zusätzlich aus D berechneter Chronologieposterior als Faktor. |
| P | Zwei künstliche Kompositionen mit drei Kategorien A,B,R und belegtem Nenner Nj=2. Grundbeobachtungen n1=(2,0,0), n2=(0,2,0). R ist von Beginn an eine explizit gezählte dritte Kategorie, keine nachträglich ergänzte Restmenge. |
| P-Messprozess | Bei c=C(kj(tau)) sind pA=(3+psi*c)/8, pB=(3-psi*c)/8, pR=2/8. Die vollständige Komposition folgt Multinomial(Nj,p). Dieselbe psi-Variable gilt für beide Proben. |
| theta / Y | theta hat Prior je 1/2 und betrifft nur den nachgelagerten Proxyfehler. Yj in {-1,+1}, Pr(Yj=+1 bei C,tau,theta)=1/2+C(kj(tau))/(4*theta). Grundbeobachtung Y=(+1,-1). Beide Proben teilen theta. |
| Künstliche Wahrheit | psi*=+1, C*=(+1,+1,-1), tau*=b, theta*=1. Die handfestgelegten Grundbeobachtungen liegen im Support dieser Wahrheit. Sie sind keine erzeugte Stichprobe und garantieren kein Posteriormaximum bei der Wahrheit. |

Die drei Kategorien sind keine Pollentyp-/Artbrücke. Ihr gemeinsamer Nenner
und ihre erschöpfende Abdeckung sind im künstlichen Experiment bekannt.
Alle Grundwahrscheinlichkeiten sind positiv; eine beobachtete Nullzahl
bezeichnet keine strukturelle Abwesenheit.

**Begründung der lokalen Faktorisierung:** Im künstlichen generativen Versuch
sind E, D und die Messfehler von P und Y getrennte Ereignisse. Bedingt auf
C,tau,psi,theta sind die Messereignisse der beiden Proben sowie P und Y
unabhängig; innerhalb einer P-Komposition gilt dagegen die multinomiale
Abhängigkeit. Gemeinsame C, tau, psi und theta bleiben erhalten. Diese
Konstruktion rechtfertigt die folgende Faktorisierung ausschließlich hier;
sie ist kein Unabhängigkeitsbeleg für reale Proxys.

## Zielverteilung und vorab bestimmte Vergleichsgrößen

Mit LPj als vollständiger Multinomial-Wahrscheinlichkeit einschließlich
Koeffizient und LYj als Bernoulli-Wahrscheinlichkeit der beobachteten Yj:

\[
F_\psi(C,\tau,\theta)
=p_C(C)\,p_\tau(\tau)\,p_\theta(\theta)\,L_D(\tau)
\prod_{j=1}^{2}L_{Pj}(C,\tau,\psi)L_{Yj}(C,\tau,\theta),
\]
\[
Z(\psi)=\sum_{C,\tau,\theta} F_\psi(C,\tau,\theta),\qquad
Q(\psi,C,\tau,\theta)=q(\psi)F_\psi(C,\tau,\theta)/Z(\psi).
\]

Die Summation umfasst den gesamten zweiten Block. q wird weder neu gewichtet
noch werden einzelne Kalibrierzustände anhand fossiler Evidenz entfernt.
Vollständig fehlende, ausdrücklich ignorierbar konstruierte Messereignisse
tragen Faktor 1 bei; Teilbeobachtungen werden wie unten festgelegt summiert.

Für gültige Fälle später zu vergleichen: jeder Zustand von F und Q, Z je psi,
Gesamtsumme Q, psi-/C-/tau-/theta-Marginalen sowie die jeweils benannten
Ereigniswahrscheinlichkeiten, Kovarianzen oder Posterior-Odds. Keine Auswahl
anhand besserer Fits oder Scores. Gleichheit bedeutet **exakte Gleichheit als
gekürzter rationaler Bruch**; Ungleichheit bedeutet einen von null verschiedenen
exakten Unterschied. Keine frei gewählte Fließkommatoleranz, kein Monte Carlo,
kein Seed, keine Konvergenzentscheidung. Eine spätere Float-/Samplerprüfung
wäre ein eigener Auftrag mit eigenen Kriterien.

**Chronologie-Rückkanal:** Für den künstlichen Grundfall wird P/Y -> tau im
zweiten Block ausdrücklich zugelassen. Der Zweck ist, die Reichweite des
Kalibrier-Cuts sichtbar zu machen. D ist unabhängig konstruiert, beide
Alterskurven sind zulässig, und C ist derselbe Verlauf für beide Messkanäle.
Damit entfällt hier eine reale Datierungs-/Abstimmungsfrage; diese Entscheidung
gilt ausschließlich für die künstlichen Fälle. Es wird kein zusätzlicher
Chronologie-Cut und keine reale Rückkopplungsfreigabe beschlossen.

## Vollständiger Fallplan und Abnahme

Alle Fälle sind vor jeder Ausführung verpflichtend. Zustandsfixierungen oder
Ersetzungen gelten nur für die jeweils bezeichnete Diagnose; alle nicht
ersetzten Grunddefinitionen bleiben bestehen. Fixierte latente Zustände
bedeuten Punktmassen an den genannten Werten im zweiten Block, ohne q neu
zu schätzen. S06a–d und S07 ersetzen jeweils ausschließlich den Messkanal P1;
P2, Y und D bleiben dort wie in S01. Kein Fall darf nach Resultaten
entfernt oder umparametrisiert werden. Bei Widerspruch zuerst berichten und
stoppen, statt Erwartungen nachträglich passend zu machen.

| ID / Vertragsbezug | Künstliche Wahrheit und feste Variation | Erwarteter Befund / exaktes Abnahmekriterium | Getrennte Nachrechnung |
| --- | --- | --- | --- |
| S00 – Prior-only; Rollen, Zeitprozess | Grundraum, sämtliche P/Y/D abwesend; Wahrheit wie oben. | Q=q*pC*pTau*pTheta; C bleibt zeitlich gekoppelt. Kein Datengewinn oder Wahrheitsnachweis behaupten. | Direktes Produkt der einzelnen normierten Priors gegen die allgemeine Normierung vergleichen. |
| S01 – gesamter Cut; Normierungsabschnitt | Vollständiger Grundfall mit festgelegter Wahrheit. | Q nichtnegativ, Summe=1, psi-Marginale=q; alle Zustände und Marginalen stimmen zwischen beiden Rechenwegen überein. Kein obligatorisches Maximum an der Wahrheit. | Vollständige unabhängige Zustandssummation gegen den späteren Prüfkern. |
| S02 – bedingte Normierung | Erst S01, dann denselben Kern je psi mit r(-1)=1, r(+1)=3 skalieren. Zusätzlich isolierter Zeugenfall Fpsi=r(psi)*pC*pTau*pTheta ohne Evidenz. | Korrektes Q unter der Skalierung invariant. Im Zeugenfall bleibt die psi-Marginale q. Eine einmalige globale Normierung von q*F hat dort psi-Masse proportional q*r und muss als defekt erkannt werden. | Rationale Zeugenformel ohne Verwendung des allgemeinen Normierers; Zustandsgleichheit für den skalierten Grundfall. |
| S03 – gemeinsame Kalibrierunsicherheit | C=(+1,+1,+1), tau=a, theta=1 fest; keine P/Y/D. Zwei zukünftige einzelne A-Indikatoren Bj unter derselben psi-Mischung q, pA=(3+psi)/8. Wahrheit psi*=+1. | Cov(B1,B2)=Var_q(pA)>0. Defekt mit unabhängigem psi je Probe liefert Produkt der Einzelmarginalen und Kovarianz 0. Gemeinsame Unsicherheit muss erhalten bleiben. | Zwei-Punkt-Momentformel gegen vollständige gemeinsame prädiktive Verteilung; hier keine bereits beobachteten P erneut konditionieren. |
| S04 – Zeitabhängigkeit | S00, gemeinsame C-Kette unverändert. | Pr(C2=C1)=3/4 und Cov(C1,C2)>0. Produkt der getrennten C-Marginalen ist nicht die gemeinsame Verteilung. Teilen von psi allein ersetzt diesen Prozess nicht. | Direkte Zweischritt-Markov-Tabelle mit Summation über C3 gegen C-Marginale von S00. |
| S05 – Kategorienabhängigkeit | Eine neue vollständige P-Komposition mit N=2 bei C=+1, psi=+1; alle übrigen Zustände fixiert. | Summe nA+nB+nR=N für jeden Ausgang; Cov(nA,nB)=-N*pA*pB<0. Unabhängige Binomialkategorien verletzen die gemeinsame Verteilung/den festen Nenner. | Multinomial-Momentformel gegen Aufzählung sämtlicher Kompositionen von N. |
| S06 – gemessene Null, fehlend, N=0 | Ausgehend von S01 getrennt: (a) gemessene vollständige (0,2,0), N=2; (b) gesamte P-Probe explizit ignorierbar fehlend; (c) nA=1 beobachtet, B/R fehlend, N=2 bekannt und Erfassung ignorierbar; (d) vollständige (0,0,0), N=0. | (a) positive Likelihood, keine strukturelle Null; (b) Faktor 1; (c) Summe über B+R=1, identisch Binomial(N,pA) für nA=1, ohne Werte zu imputieren; (d) Kompositionsfaktor 1 und kein Zählkompositionssignal. Status selbst hier ausdrücklich klimaunabhängig. | Multinomialformel, marginale Binomialformel und direkte Summation, nicht dieselbe Missingness-Hilfsfunktion. |
| S07 – Prozent-only / gerundete Null | Eigenständige Messvariation statt der P1-Countsprobe: latenter Anteil H in {0,1/4,3/4,1} mit Massen ((1-pA)/2,(1-pA)/2,pA/2,pA/2). Bericht R0=0 für H<1/2, sonst R0=1. Künstliche Wahrheit H*=1/4, beobachtet R0=0. | Likelihood ist die Summe beider zulässiger H-Massen; H=1/4 hat positive bedingte Masse trotz gedruckter Null. Kein Count/N wird erzeugt; keine Gleichsetzung mit H=0 oder struktureller Abwesenheit. | Aufzählung der vier H-Zustände gegen die festgelegte Berichtswahrscheinlichkeit. Diese Präzisionsregel gilt nur für diese künstliche Variation. |
| S08 – ganze Alterskurven / Verschiebung | (a) P/Y abwesend, D vorhanden; (b) S01; (c) S01 bei festem tau=a bzw. tau=b. Grundwahrheit b. | (a) tau proportional pTau*LD; (b) tau proportional pTau*LD mal die je tau über C/theta summierte Messinformation, getrennt innerhalb jedes psi, danach q-Mischung. Keine Pflicht zur Konzentration auf b. Alle bedingten Verläufe und Verschiebungsfolgen ausweisen. | Summation getrennt nach ganzer Kurve, nie nach unabhängigen Probenaltern; a/b-bedingte Tabellen gegen gemeinsame Tabelle. |
| S09 – Proxykonflikt | S01 und dessen Variante Y=(-1,+1), P unverändert. Zusätzlich Y ganz weglassen. Die Grundwahrheit bleibt fest; jede Beobachtung hat im Messmodell positiven Support. | Alle drei Q vollständig berichten, positiver Z je psi, psi-Marginale=q. Änderung darf C/tau/theta betreffen; weder kleinere Bänder noch bessere Wahrheitstreue sind zwingend. Ein Konflikt ist keine Erlaubnis, Proxys auszuwählen oder q umzuwichten. | Eigenständige Bernoulli-Auswertung beider Y-Belegungen und Fall ohne Y; zustandsweiser Vergleich. |
| S10 – gemeinsame Proxy-Störgröße | C=(+1,+1,+1), tau=a fix, P/D abwesend; theta bleibt unsicher. Zwei zukünftige Y-Indikatoren teilen theta; Wahrheit theta*=1. | Gemeinsame prädiktive Wahrscheinlichkeit ist Summe_theta pTheta*LY1*LY2; ihre Kovarianz ist Var_theta(1/2+1/(4*theta))>0. Produkt separat marginalisierter Faktoren ist defekt. | Zwei-Punkt-Momentformel gegen explizite gemeinsame Y-Tabelle; keine Wiederverwendung desselben bereits beobachteten Y als neues Replikat. |
| S11 – Nichtidentifizierbarkeit / Support | Nur zwei zweite-Block-Zustände: U_A=(C=(+1,-1,-1),tau=a,theta=1) und U_B=(C=(+1,+1,-1),tau=b,theta=1), Prior je 1/2; D fehlt, Grund-P/Y. Wahrheit U_B, psi*=+1. Sensitivität: gleicher Fall mit Prior U_A=3/4, U_B=1/4. | C(k1),C(k2) sind in beiden Zuständen gleich; Mess-Likelihoods je psi gleich. Posterior-Odds bleiben Prior-Odds. Bekannte Wahrheit wird nicht eindeutig identifiziert; die Priorvariation ist kein zusätzlicher Informationsgewinn. Beide Priors berichten. | Direkter Vergleich der Mess-Likelihoods und Odds ohne allgemeinen Normierer; getrennt für jedes psi. |
| S12 – ungültige Eingänge / Stopbedingungen | Feste Fehlerliste im nächsten Abschnitt; einzelne Mutation pro ansonsten gültigem Fall, ausdrücklich keine Datenfälle. | Jeder Fall endet mit dem bezeichneten Fehlerstatus vor einer gültigen Q-Ausgabe. Kein automatisches Reparieren, Aussortieren oder Renormieren von q. | Erwartete Fehlerklasse unmittelbar aus Feldstatus, Support oder mathematischer Definition prüfen. |

Für S08 ist die Kurvenmarginale ausdrücklich

\[
Q(\tau)=\sum_\psi q(\psi)\frac{p_\tau(\tau)L_D(\tau)
\sum_{C,\theta}p_C(C)p_\theta(\theta)
\prod_j L_{Pj}L_{Yj}}{Z(\psi)}.
\]

Kein gemeinsamer Normierer außerhalb der psi-Summe. Als zusätzlicher
verpflichtender Rückkanalzeuge S08d: C=C*, theta=1 fixieren, P1 und sämtliche
Y fehlen, P2=(0,2,0), N2=2 und D=1 bleiben. Innerhalb der bedingten
psi=+1-Komponente unterscheiden sich die P2-Likelihoods für a und b.
Die tau-Odds müssen gegenüber pTau*LD verändert sein und exakt deren
Likelihoodverhältnis enthalten. Eine ausschließlich durch D bestimmte
Chronologie verletzt diesen Zeugen; q selbst bleibt geschützt. Die
unabhängige Nachrechnung verwendet nur das Verhältnis der zwei expliziten
P2-Wahrscheinlichkeiten und der Prior-/Datierungsgewichte.

S03/S05/S10 bezeichnen **zukünftige prädiktive Messungen** unter den jeweils
angegebenen fixierten Zuständen/Priors. Sie sind keine zusätzlichen Daten im
Cut-Posterior. S07 ist ein eigener Prozent-Messkanal, kein aus Counts abgeleiteter
zweiter unabhängiger Proxy. S11 ersetzt explizit den zweiten-Block-Prior durch
den genannten Zweizustandsprior; es handelt sich um einen absichtlichen
Nichtidentifikationszeugen, nicht um eine nachträgliche Beschränkung von S01.

## Fehlerliste und absichtlich defekte Gegenverfahren

Jede Zeile ist ein eigener S12-Fall; keine anderen Eingänge zugleich ändern.
Fehlerklassen sind Teil der geplanten Ausgabe, noch kein implementiertes API.

| Fehler-ID | Fester ungültiger Eingang | Erwarteter Status |
| --- | --- | --- |
| I01 | q=(-1/3,4/3), q=(1/3,1/3) oder fehlender q(+1)-Wert, jeweils separat | INVALID_CALIBRATION; negative, nicht normierte oder fehlende Kalibriermasse nicht reparieren. |
| I02 | Fpsi identisch null für psi=+1 bei q(+1)>0 | INVALID_NORMALIZER; kein Verwerfen dieser psi-Komponente. |
| I03 | Am Zustand psi=+1, C=C*, tau=b, theta=1 den Kernwert durch -1, +Unendlich bzw. undefiniert ersetzen, je eigene Mutation | INVALID_KERNEL; endlicher Zustandsraum erlaubt keinen unendlichen Normierer. |
| I04 | N unbekannt, N=-1, N=3/2, n=(-1,3,0), n=(1/2,3/2,0), oder vollständige n=(0,0,0) bei N=2, jeweils separat | INVALID_OBSERVATION; kein Summen-N, Runden, Nullfüllen oder Ergänzen von R. |
| I05 | Als Null übergebener Sparse-Nichttreffer, Status unbekannt/nicht berichtet/nicht bestimmt/außerhalb Abdeckung, oder unvollständige Kategorien ohne belegte Abdeckungsregel, jeweils separat | UNRESOLVED_STATUS; keine automatische Ignorierbarkeit oder Nullinterpretation. Nur das ausdrücklich konstruierte Missingness-Modell S06 ist zulässig. |
| I06 | Prozent-only oder gerundete/Nachweisgrenzen-Null ohne eigene Präzisionsregel; Versuch N=100 daraus zu erzeugen, separat | MISSING_PRECISION_MODEL bzw. PSEUDOCOUNT_FORBIDDEN. S07 ersetzt keine reale Präzisionsklärung. |
| I07 | Altersindizes (2,2), (2,1) oder Index außerhalb {1,2,3}, jeweils separat | INVALID_CHRONOLOGY; keine erlaubte gemeinsame Kurve. |
| I08 | Derselbe D-Ursprungsbeleg zusätzlich als unabhängiger Chronologieposterior; derselbe P-Ursprungsbeleg nochmals als AP/NAP-Proxy; identische Y-Beleg-ID zweimal als unabhängige Replikate, separat | DUPLICATE_EVIDENCE. Verschiedene Messereignisse dürfen eine Eltern-ID für gemeinsame latente Größen teilen; die einzigartige Messbeleg-ID ist keine solche Eltern-ID. |
| I09 | Fremde reale Dateireferenz, unbekannte Kategorie, nicht deklarierte Klimaeinheit oder nicht deklarierte Chronologie-Rückkanaloption, separat | OUT_OF_DESIGN; keine Übernahme realer Daten oder stiller Erweiterung. |

Zusätzlich vier **absichtlich defekte Gegenverfahren** nur im späteren
Prüfumfang vorsehen: globale statt psi-bedingte Normierung (S02), unabhängig
neu gezogenes psi pro Probe (S03), Produkt unabhängiger Kategorien (S05),
Produkt separat über theta gemittelter Proxyfaktoren (S10). Ihre Erkennung
bedeutet, dass der zugeordnete exakte Zeuge verletzt wird. Sie dürfen keine
fachlich wählbaren Rekonstruktionsvarianten werden. Ein Gegenverfahren, das
den Zeugen unerwartet besteht, ist ein fehlgeschlagener Prüfnachweis; nicht
als erfolgreiche Fehlererkennung berichten. S04 vergleicht zusätzlich mit
dem Produkt der C-Marginalen als bloßer Negativkontrolle für die Zeitkopplung.

## Geplante getrennte Nachrechnung und vollständige Berichterstattung

Erst nach separatem Auftrag zwei Rechenwege implementieren:

1. Prüfkern mit expliziten Faktoren, festem Zustandsraum und rationaler
   Normierung des gemeinsamen Blocks je psi; daraus gemeinsame Tabellen und
   Marginalen ableiten. Jeder Zustand enthält nur eine psi-Variable und eine
   ganze tau-Kurve. Eingangsprüfung vor der Normierung.
2. Referenz mit eigener vollständiger Zustandsschleife und direkt eingesetzten
   rationalen Definitionen dieses Dokuments, ohne Import/Wiederverwendung der
   Faktor-, Normierungs-, Missingness- oder Marginalisierungsfunktionen des
   Prüfkerns. Für S00/S02–S07/S10/S11 zusätzlich die bezeichneten Produkt-,
   Moment- und Odds-Identitäten separat auswerten. Gemeinsam nutzbar sind nur
   deklarierte Fallkonstanten und rationale Standardarithmetik. Eine gemeinsame
   Programmiersprache schließt gemeinsame Fehler nicht aus; die ausdrücklich
   andere Zerlegung und die algebraischen Zeugen begrenzen dieses Risiko.

Vorab festgelegte Abnahme: sämtliche gültigen Zustandsmassen/Marginalen beider
Wege exakt gleich; alle positiven/Null-/Ungleichheitsbedingungen der Tabelle
bestehen; sämtliche I-Fälle liefern ihre erwartete Ablehnung; jede defekte
Gegenrechnung wird vom zugeordneten Zeugen erkannt. Kein Ersatz durch grobe
Kurvenähnlichkeit, Score, Gesamtzahl grüner Fälle oder bloße Konvergenz.

Ein späterer Auditbericht muss jede ID samt Unterfällen enthalten:
**bestanden, fehlgeschlagen, nicht schätzbar, blockiert oder nicht ausgeführt**,
mit festem Eingang, Erwartung, tatsächlichem Befund, Abweichung und Grenze.
Die vier defekten Verfahren separat als absichtlich defekt kennzeichnen;
ihr erwartetes Scheitern ist kein bestandener realer Modellvergleich.
Keine stillen Ausfälle/Ausschlüsse, keine Ergebnisoptimierung und keine
unbegründete Wahrheits-, Deckungs- oder Neuheitsbehauptung. Alle vorgesehenen
Variationen berichten, auch unveränderte oder ungünstige Befunde.

Keine Zufallsdatenerzeugung, Fits, Scores, Optimierung oder Simulationen sind
für diese endliche Prüfung erforderlich. Tabellen und Auditbytes wären erst
bei autorisierter Ausführung an einem benannten dauerhaften Repositorypfad
zu sichern; bloße Hashes gelten nicht als Archivierung. Hier existieren
keine erzeugten Tabellen oder Ausführungsresultate.

## Sperre, Grenzen und dokumentarischer Abschluss

Dieses Design wird mit dem Inhaltscommit vor jeder Implementierung gesperrt.
Nachträgliche Änderungen an Definitionen, Fallmenge, Wahrheit oder Abnahme
benötigen ein gesondert beauftragtes Designänderungspaket vor erneuter
Ausführung; keine Anpassung anhand äußerer Testergebnisse.

**Stop:** fehlende/mehrdeutige Falldefinition, unzulässiger Support, Widerspruch
zum unveränderten Vertrag oder nicht unabhängig beschreibbare Referenz zuerst
als Blocker ausweisen. Ein unerwarteter späterer Befund ist zu erhalten und
stoppt den Anspruch einer vollständig bestandenen Prüfung. Ungültiges q/Z,
Doppelevidenz, unbekannter Feldstatus und nicht deklarierte Erweiterungen
stoppen die betroffene Berechnung. Nichtidentifizierbarkeit in S11 ist
hingegen ausdrücklich erwarteter Befund, kein zu versteckender Fehler.

Parnell-Originalzugang bleibt blockiert, Menezes-(2)/(5) offen;
kein eigener Neuheitsbeitrag abgegrenzt. W96, EMPD2-Archiv-/Messstatus und reale
T0/T1/T3/T4/Monticchio-P1 bleiben offen. Kein neues Quellenmaterial und keine
erneute Originalprüfung der Dissertation. Reale Klimavariablen, Einheiten,
Perioden, Verteilungen, Parameter, Kern, Typen und Proxywahl bleiben OFFEN.
Die künstliche positive Modellwahrscheinlichkeit ist kein empirischer Support.

Nur dieses neue Designdokument und kurze HANDOFF-/Statuspflege gehören zum
aktuellen Auftrag. Bestehende Designs, Modelle, Resultate, Register, Verträge
inklusive MODEL_CONTRACT, Importer, Flags, Counts/Gruppen, Baseline und CLI/App
unverändert. Keine Datenbeschaffung/-bereinigung/-matrix/Kohorte,
Implementierung, Tests, Fits, Scores, Simulationen oder Kernläufe in diesem
Paket. A4–A6 nicht begonnen; keine Unteragenten, Autorennachrichten oder Merges.
Dokumentationsabschluss durch Quellen-/Link-/Diffprüfung und Blobabgleich;
keine numerische Validierung oder Gesamtfertigstellung von A3 behaupten.
Genau ein begrenzter Folgeauftrag steht in [HANDOFF](./HANDOFF.md).
