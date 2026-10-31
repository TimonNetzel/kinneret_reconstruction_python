# A2: Modell- und Datenvertrag als dokumentarischer Entwurf

Stand 31.10.2026; Basis PR #51,
`665daede79048df7efb6fe60db85c765b45718bb`.
**A2 ist als Dokumentationspaket autorisiert. Dieser Entwurf ist keine reale
Daten-/Modellwahl, kein implementierter Vertrag und keine Lauf- oder Gatefreigabe.**
Maßgeblich sind [Plan §1](./MODULAR_RECONSTRUCTION_PLAN.md#1-ziel-und-festgehaltene-arbeitsrichtung),
die [A2-Zeile in §2](./MODULAR_RECONSTRUCTION_PLAN.md#2-arbeitspakete-mit-prüfbarem-ende)
und [Arbeitsmodus §4](./MODULAR_RECONSTRUCTION_PLAN.md#4-tokensparender-arbeitsmodus).
Die [A1-Matrix](./NOVELTY_MATRIX.md) liefert dokumentierte Beleggrenzen;
sie ersetzt keinen Einsatznachweis für unsere Daten. Im Folgenden bedeutet
**Festlegung** ausschließlich die bestätigte Richtung aus Plan §1;
**OFFEN** bezeichnet ein unbesetztes Feld oder eine noch zu begründende
Entscheidung. Die Stopbedingungen sichern diese Grenze für spätere Pakete.

## Aussageziel und festgelegte Architektur

**Festlegung:** Erstes greifbares Ziel ist ein geprüfter diagnostischer
Klimaverlauf für einen später begründet ausgewählten Bohrkern, mit transparenten
Pollentypzuordnungen, Chronologie, Kalibrierung, Support und Unsicherheitsgrenzen.
Eine allgemeine Multiproxy-App folgt später; die Wiederherstellung aller 279
historischen Funktionen ist keine Voraussetzung. Ein solcher Verlauf wäre eine
bedingte Rekonstruktion unter ausgewiesenen Daten- und Modellannahmen, kein
Nachweis einer eindeutigen Klimageschichte oder nomineller Intervallabdeckung.

**Festlegung:** Ein modern kalibriertes Pollen-Beobachtungsmodell überträgt
seine Unsicherheit in einen zeitlich gekoppelten Rekonstruktionsblock mit
Chronologien, Klimaprior/Ankern und weiteren unabhängigen Proxys. Ein
Kalibrier-Cut schützt die moderne Kalibrierung vor fossiler/Proxy-Rückkopplung.
Dieselben Kalibrierziehungen gelten gemeinsam für alle betroffenen Proben;
der gesamte zweite Block wird je Ziehung separat normiert.

**OFFEN:** Kern, Datensätze, Proben, Pollentypen, weitere Proxys, Zielgrößen,
Verteilungen, Parameterwerte, Hyperparameter und reale Freigaben. Ob weitere
Proxys tatsächlich unabhängig sind, muss belegt werden; das Zielwort
„unabhängig“ begründet keine Likelihood-Faktorisierung. Chronologie-Rückkopplung
innerhalb des zweiten Blocks bleibt eine gesondert zu begründende Entscheidung.
Keine dieser offenen Stellen wird durch eine Standardwahl ausgefüllt.

## Rollen und unbesetzte Eingänge

Die Symbole benennen Rollen, keine bereits ausgewählten Dateien oder Variablen.

| Rolle | Bedeutung im Entwurf | OFFEN / erforderlicher Beleg vor realem Einsatz |
| --- | --- | --- |
| \(E\) | Moderne Kalibrierevidenz für das Pollen-Beobachtungsmodell; einschließlich des zu begründenden Bezugs zwischen moderner Pollenbeobachtung und Klima. | Quelle/Version, Auswahl, Messstatus, Typkonzepte, Klimadefinitionen, Einheiten, Bezugsperioden, räumlicher und zeitlicher Bezug, Unsicherheit und Abhängigkeiten; kein moderner Trainingssatz gewählt. |
| \(P\) | Fossile Pollenbeobachtungen mit ihrem Beobachtungsstatus und Bezug zu Proben/Archivtiefen; keine aus Arealvorkommen abgeleiteten Pollenanteile. | Counts mit belegtem N oder Prozent-only mit eigenem Präzisionsmodell; Typen, Nenner, Abdeckung, Restkategorie, Missingness, Nachweis-/Rundungsregeln und Zuordnung zu \(\tau\). |
| \(Y\) | Weitere Proxybeobachtungen im Rekonstruktionsblock, die über einen eigenen Mess-/Fehlerprozess Klima informieren könnten. | Welche Proxys, Quellen, Einheiten, zeitliche Mittelung, Kalibrierung, Störgrößen und gemeinsame Fehler; keine Proxywahl oder Unabhängigkeitsannahme. |
| \(D\) | Datierungsevidenz für die Chronologie, getrennt von Klimaankern und Proxywerten. | Datierungsobjekte/-methoden, Zeitskala, Referenz, Einheiten, Fehler/Abhängigkeiten und eventuelle gemeinsame Herkunft mit \(E,P,Y\) oder Klimaankern. |
| \(\psi\) | Gesamte für die Übertragung relevante Unsicherheit des geschützten modernen Kalibriermoduls; \(q(\psi)=p(\psi\mid E)\). | Konkrete Parameter/latente Größen, gemeinsame Struktur, Verteilungen, Priors und Kalibrierverfahren. Fossil beeinflusste Parameter dürfen nicht nur durch Umbenennen zu \(\theta\) aus dem Cut herausfallen. |
| \(C\) | Gemeinsamer zeitlicher Klimaverlauf; gegebenenfalls mehrdimensional, ohne hier gewählte Dimension oder Zielvariable. | Klimavariablen, Definitionen, Einheiten, Perioden/Saisonalität, Zeitraster, räumlicher Bezug, Prozess und Klimaanker. |
| \(\tau\) | Gemeinsame Chronologie(n), die Proben-/Archivtiefen auf Klima-Zeitbezüge abbilden; Unsicherheit ganzer Alterskurven. | Zahl der Archive, Tiefen-/Zeitkonvention, zulässige Ordnung/Hiaten, Kurvenfamilie, Datierungsmodell und Rückkopplungsentscheidung. |
| \(\theta\) | Ausschließlich nachgelagerte unbekannte Größen, etwa noch zu spezifizierende Prozess-, Chronologie-, Proxy- oder Störparameter. | Inhalt, Priors, Support und Verflechtung mit \(\psi\); kein Parameterkatalog oder Wert festgelegt. Gemeinsam kalibrierte Größen benötigen eine eindeutige Modulzuordnung. |

**OFFEN bleibt insbesondere jeder konkrete Klimaeingang:** TANN/PANN und
GDD5/MTCO/AET-PET sind keine umbenennbaren Größen. Die Nennung ist eine
Abgrenzung, keine Auswahl. Eine spätere Verbindung benötigt eine belegte
inhaltliche Zuordnung samt Unsicherheit; Einheiten- oder Periodenumrechnung
wird hier weder unterstellt noch festgelegt. Auch ein Klimaanker braucht
Quelle, Größe, Zeit-/Raumbezug, Unsicherheit und eine einzige Evidenzrolle.
Fehlende Werte werden nicht aus früheren Modellbezeichnungen ergänzt.

## Beobachtungsstatus, Nullen und Abdeckung

**Festlegung:** Ein Countsmodell ist nur bei belegten Counts, originalem
Zählnenner N und Feldstatus zulässig. Prozent-only benötigt ein eigenes
Beobachtungs-/Präzisionsmodell. N=100 als Pseudozählung ist ausgeschlossen.
Die sieben kartierten Artenfunktionen bleiben unveränderte Vergleichsreferenzen;
sie sind keine Pollenanteilsfunktionen. Ihre mögliche spätere Verwendung als
kalibrierte Merkmale ist eine eigene, hier nicht bearbeitete Frage.

Für einen späteren Einsatz sind mindestens folgende Zustände zu unterscheiden;
ihre reale Codierung und jeweilige Wahrscheinlichkeit sind
**OFFEN**:

- Ein belegter gemessener Nullcount ist eine Beobachtung im bezeichneten
  Zählsystem. Er belegt allein keine strukturelle Abwesenheit. Ein struktureller
  Nullprozess wäre ein eigener begründungspflichtiger Modellteil.
- Eine gerundete Null, ein Wert unter Nachweisgrenze oder ein Prozent-Endpunkt
  benötigt die passende Präzisions-/Zensierungsregel. Eine gedruckte Null ist
  nicht automatisch ein exakter Nullanteil; ein Null-/N-Inflationsmodell aus
  einer Vorarbeit ist keine festgelegte Verteilung für unsere Daten.
- Fehlend, nicht bestimmt, nicht berichtet, außerhalb der Abdeckung und ein
  Sparse-Nichttreffer bedeuten nicht automatisch Null. Unbekannter Feldstatus
  bleibt unbekannt; kein Nullfüllen, Runden oder Ergänzen einer Restkategorie.
- N bezeichnet den belegten Nenner der betrachteten Zusammensetzung, nicht
  die Summe beliebiger gespeicherter Werte oder automatisch die Summe einer
  ausgewählten Typenteilmenge. Ausschlüsse, Rest und Zusammenlegungen brauchen
  eine gemeinsame Nenner-/Abdeckungsregel. N=0 liefert bei einem auf N
  bedingten Zählmodell keine Zählkompositionsinformation; ob der Status selbst
  Information trägt, wäre gesondert zu modellieren. Ein Nullvektor bei N>0
  ist ohne belegte Rest-/Missingness-Regel keine vollständige Komposition.

Pollentypen und Arten sind nicht austauschbar. Breite Typen, Artenmischungen,
Produktions-/Transport-/Erhaltungseinflüsse und Messabdeckung bleiben als
mögliche Abhängigkeiten bzw. Identifikationsgrenzen sichtbar. Die konkrete
Beobachtungs-Likelihood für Counts oder Prozent-only ist **OFFEN**. Der
EMPD2-Archiv-/Messstatus und W96 werden nicht durch den Vertrag aufgelöst.

## Kalibrier-Cut und gemeinsame bedingte Normierung

**Festlegung aus Plan §1:**

\[
p_{cut}(\psi,C,\tau,\theta\mid E,P,Y,D)
=p(\psi\mid E)\,p(C,\tau,\theta\mid P,Y,D,\psi).
\]

Dies ist eine modulare Zielverteilung, kein vollständiger gemeinsamer
Bayes-Posterior: P, Y und D aktualisieren q nicht. Die folgende Schreibweise
konkretisiert nur ihre Normierung, keine Verteilungswahl. Setze \(U=(C,\tau,\theta)\),
\(q(\psi)=p(\psi\mid E)\), und bezeichne mit \(F_\psi(U;P,Y,D)\) den noch
zu spezifizierenden nichtnegativen gemeinsamen Rekonstruktionskern. Dann

\[
Z(\psi)=\int F_\psi(U;P,Y,D)\,dU,\qquad
p_{cut}(\psi,U\mid E,P,Y,D)
=q(\psi)\frac{F_\psi(U;P,Y,D)}{Z(\psi)}.
\]

Integration steht gegebenenfalls auch für Summation über diskrete Größen.
Voraussetzung sind ein properes q und \(0<Z(\psi)<\infty\) für q-fast alle
\(\psi\). Dann gilt

\[
\int p_{cut}(\psi,U\mid E,P,Y,D)\,dU=q(\psi),\qquad
p_{cut}(U\mid E,P,Y,D)=\int q(\psi)\frac{F_\psi(U;P,Y,D)}{Z(\psi)}\,d\psi.
\]

Der Normierer umfasst **gemeinsam C, tau und theta über alle betroffenen
Proben/Archive und Proxykanäle**, jeweils bei festem psi. Ein einmal normiertes
\(\int q(\psi)F_\psi(U)\,d\psi\) würde im Allgemeinen Komponenten nach
\(q(\psi)Z(\psi)\) gewichten und die geschützte Kalibrierrandverteilung
verändern. Auch das Produkt separat über psi ausgemittelter Probenposterioren
ersetzt den gemeinsamen Block nicht. Probenweise Normierer sind ohne belegte
Faktorisierung des gesamten Blocks unzulässig; eine gemeinsame Chronologie
oder ein Zeitprozess verhindert diese im Allgemeinen.

Dieselbe gemeinsame psi-Ziehung wirkt auf alle betroffenen Proben. Unabhängig
neu gezogene Kalibrierparameter je Probe würden gemeinsame Unsicherheit
verlieren. Schon das Ausmitteln gemeinsamer Kalibrierunsicherheit erzeugt
Abhängigkeiten, ist aber noch kein zeitlicher Klimaprior.

**OFFEN:** F, seine Likelihood-/Priorzerlegung und alle Verteilungen. Eine
spätere Zerlegung muss Abhängigkeiten und Evidenzherkunft rechtfertigen;
ein Produkt unabhängiger Pollen-, Proxy- und Datierungsfaktoren ist hier
nicht vereinbart. Die Kurznotation setzt voraus, dass die relevante moderne
Kalibrierunsicherheit vollständig in psi übertragen wird. Verbleibende
Abhängigkeiten zu E dürfen nicht still verschwinden oder E erneut als
unabhängige Evidenz eingehen; die Modulgrenze wäre zuerst zu klären.

**Beleggrenze aus A1:** Die bedingte Normierung entspricht der dort belegten
Lesart von Menezes v1 (2) und der beschriebenen Normierung je Kalibrierziehung
im SIR-Verfahren. Die wörtliche Lesart von (5) bleibt ein offener Formelklärpunkt.
Kein Software-/Resultatfehler und keine konkrete Samplerwahl werden daraus
abgeleitet. Parnells Originalgleichungen bleiben mangels Zugang ungeprüft.

## Zeitprozess und Chronologie-Rückkopplung

**Festlegung:** Der zweite Block soll einen zeitlich gekoppelten Verlauf
modellieren. Pollen und weitere Proxys müssen über ihren zeitlichen Bezug
denselben C-Verlauf informieren können. Eine Verbindung von Punkten im Plot,
ein nachträglicher Glätter oder gemeinsame psi-Unsicherheit erfüllt allein
keinen spezifizierten zeitlichen Prozess.

**OFFEN:** Prozessfamilie, Abhängigkeitsmaßstab, Glattheit, Nichtstationarität,
Randbedingungen, zeitliche Aggregation, Klimaanker und sämtliche zugehörigen
Parameter/Priors. Keine bestimmte Prozessfamilie aus älteren Designs oder
einer der drei Vorarbeiten wird übernommen. Beobachtungsfehler, Prozessvariation
und Altersunsicherheit benötigen unterscheidbare Rollen.

Die Chronologie koppelt Proben über ganze Alterskurven. Eine spätere
Spezifikation muss Tiefenordnung, gemeinsame Datierungsanker und zulässige
Lücken/Hiaten respektieren; unabhängige Altersfehler je Probe dürfen nicht
als gleichwertiger Ersatz gelten. Konkrete Ordnungsregeln, Zeitrichtung,
Skalen und Auflösung sind **OFFEN**.

Im gemeinsamen zweiten Block können P und Y über C auch tau und theta
mitinformieren. Der Kalibrier-Cut verhindert nur ihren Rückkanal zu psi;
er schützt tau nicht automatisch. **OFFEN und gesondert zu begründen:**
ob und unter welchen Annahmen diese Chronologie-Rückkopplung zugelassen wird.
Eine nur durch D bestimmte Chronologie würde einen zusätzlichen Chronologie-Cut
und eine entsprechend andere modulare Zielverteilung erfordern. Weder dieser
Cut noch eine reale Rückkopplungsfreigabe wird in A2 beschlossen.

Datierungs-Likelihood und ein bereits aus denselben D gewonnener
Chronologieposterior dürfen nicht als unabhängige Faktoren gemeinsam eingehen.
Dasselbe gilt für klimaabgestimmte Alter, Klimaanker und die dazu verwendeten
Proxywerte. Herkunft, gemeinsame Unsicherheit und Informationsfluss sind vor
einer solchen Nutzung offenzulegen; sie bleiben hier **OFFEN**.

## Proxy-Abhängigkeiten und gemeinsame Evidenz

Weitere Proxys benötigen jeweils einen eigenen begründeten Mess-/Fehlerprozess,
Klimabezug und zeitlichen Bezug. Deren konkrete Eingänge und Verteilungen
sind **OFFEN**. Ein alleiniger Beta(R²)-Faktor ersetzt diese Rollen nicht;
kein bestehendes Verfahren wird hier verändert.

AP/NAP, Summen, Verhältnisse oder andere Ableitungen derselben Pollenwerte
sind keine zusätzlichen unabhängigen Replikate. Eine Arealfunktion ist ebenso
wenig eine weitere unabhängige Pollen-Likelihood. Abhängigkeiten können auch
aus gemeinsamem Archiv, Chronologie, Klimaanker, Kalibrierdatensatz,
Transport-/Erhaltungseinflüssen oder anderen Störgrößen entstehen. Gemeinsames
C allein garantiert keine bedingte Unabhängigkeit der verbleibenden Fehler.

Vor realer Verwendung muss für jede Evidenz feststehen, in welchem Modul sie
einmalig eingeht, welche Größen gemeinsam sind und worauf eine eventuelle
bedingte Unabhängigkeit gestützt wird. Gemeinsame latente Störgrößen oder
korrelierte Fehler sind denkbare Rollen, keine hier gewählte Lösung.
Zusätzliche Proxykalibrierungen dürfen nicht ohne Entscheidung über ihre
Unsicherheit und Modulzuordnung in theta versteckt werden.

## Support und Identifizierbarkeit

Support umfasst den mathematisch zulässigen Bereich von C, tau, psi und theta
sowie die durch E belegte Klima-, Typ-, Raum- und Zeitabdeckung. Mathematisch
positive Dichte belegt noch keine empirische Übertragbarkeit. **OFFEN:** alle
konkreten Grenzen, Extrapolationsregeln, Referenzperioden und diagnostischen
Schwellen; keine nachträgliche Auswahl anhand äußerer Testergebnisse.

Fehlende Klimaüberlappung, unbelegte Typbrücken, unbekannte Nenner oder nicht
zuordenbare Zeitbezüge dürfen nicht durch breitere Priors als gelöst gelten.
Randhäufung und Priorabhängigkeit wären spätere Diagnosefragen, keine hier
bereits geprüften Befunde. Fehlende Unsicherheitskomponenten bedeuten keine
Nullunsicherheit.

Identifizierbarkeit ist ebenfalls **OFFEN**, insbesondere:

- Kompositionen bestimmen keine absolute Produktions-/Konzentrationsskala.
  Frei veränderliche Produktion, Transport oder Erhaltung können Klimasignale
  nachbilden; ein breiter Pollentyp identifiziert keine eindeutigen Artenanteile.
- Mehrere Klimagrößen können ähnliche Pollenantworten erzeugen. Ein enger
  Prior oder eine feste Artbrücke kann rechnerische Eindeutigkeit erzwingen,
  ohne unabhängige Information zu liefern.
- C und tau können zeitliche Verschiebungen gegeneinander ausgleichen;
  Prozessglattheit, Chronologieparameter, Proxyfehler und Kalibrierunsicherheit
  können ähnliche Effekte haben. Anker müssen unabhängige Information beitragen.
- Der Cut bewahrt auch schwach identifizierte moderne Kalibrierunsicherheit.
  Fossile Daten dürfen diese nicht heimlich durch Umgewichtung von psi beheben.

Ob unabhängige Daten oder begründete Einschränkungen diese Mehrdeutigkeiten
auflösen können, bleibt unbelegt. Enge Bänder, Konvergenz oder bessere Scores
wären allein kein Identifikations-, Deckungs- oder Neuheitsnachweis. A2 führt
keine entsprechende Prüfung oder Validierung aus.

## Stopbedingungen und Abschlussgrenze

Die folgenden Bedingungen stoppen die jeweilige spätere reale Nutzung oder
deren Interpretation; sie sind keine Behauptung, A2 habe diese Gates geprüft.

| Stopbedingung | Konsequenz vor Fortsetzung |
| --- | --- |
| Klimavariable, Einheit, Bezugsperiode, Raum-/Zeitzuordnung oder Evidenzrolle fehlt bzw. widerspricht einer anderen Eingabe. | Feld OFFEN erhalten; keine Ersetzung, stillschweigende Umdeutung oder reale Rekonstruktion mit diesem Eingang. |
| Counts/N, Null-/Missingness-Status, Nenner, Typdefinition oder Abdeckung unbelegt; Prozent-only ohne Präzisionsmodell. | Kein Countsmodell, keine Pseudo-Counts, Nullfüllung, Rundung oder automatische Typ-/Artbrücke. |
| Ungelöste Archiv-/Messstatus-, W96- oder reale Gategrenze betrifft den Eingang. | Betroffene Nutzung bleibt gesperrt; A2 erteilt keine T0/T1/T3/T4/Monticchio-P1-Freigabe. |
| q nicht proper oder Z für eine Menge positiven q-Maßes null, unendlich oder nicht definiert. | Keine gültige Cut-Zielverteilung behaupten; keine Kalibrierziehungen verwerfen oder neu gewichten, um das Problem zu verdecken. Numerisch unbestimmtes Z ist kein Nachweis von Gültigkeit. |
| Gemeinsame psi-Unsicherheit fehlt, Normierung erfolgt ohne belegte Faktorisierung nur pro Probe oder bedingte Normierer werden aus dem Integral gezogen. | Nicht als Zielverteilung aus Plan §1 ausgeben; Modul-/Normierungsfrage zuerst klären. |
| Chronologie-Rückkanal ungeklärt, Kurvensupport widersprüchlich oder Datierungs-/Klimaevidenz doppelt verwendet. | Keine Chronologieaktualisierung oder reale gemeinsame Inferenz auf dieser Grundlage. |
| Proxy-Abhängigkeit, Kalibrierunsicherheit oder gemeinsame Evidenzherkunft ungeklärt. | Kein unbegründetes Produkt unabhängiger Likelihoods und keine Bandverengung als Informationsgewinn ausgeben. |
| Fehlender empirischer Support oder ungelöste Nichtidentifizierbarkeit. | Keine bestätigte eindeutige Klimakurve oder validierte Unsicherheit behaupten; Grenze ausweisen, keine stille Auswahl/Ausschließung. |
| Nötige Konkretisierung überschreitet Dokumentation, etwa Daten-/Modellwahl, Matrix, Implementierung oder Rechnung. | A2 beenden; nur ein gesondert autorisiertes, begrenztes Folgepaket darf daran anschließen. |

A1 bleibt abgeschlossen mit blockiertem Parnell-Originalzugang und offenem
Menezes-(2)/(5)-Klärpunkt. Kein eigener Neuheitsbeitrag ist abgegrenzt.
Keine neue Literatur-/Archivsuche und keine erneute Originalprüfung der
Dissertation gehören zu A2. Die Aussagen hier sind aus Plan/A1 abgeleitet;
keine neuen Originalbelege werden beansprucht.

**Abschlussumfang:** genau dieser zusammenhängende Entwurf, ergänzt um kurze
Status-/Übergabepflege. Bestehende Designs, Modelle, Resultate, Register,
Verträge, Importer, Flags, Counts/Gruppen, Baseline und CLI/App bleiben
unverändert. Keine reale Auswahl/Beschaffung/Bereinigung, Datenmatrix/Kohorte,
Implementierung, Tests, Fits, Scores, Simulationen oder Kernläufe. Prüfung
ausschließlich durch Quellen-/Link-/Diffprüfung und Blobabgleich. Genau ein
begrenzter Folgeauftrag steht in [HANDOFF](./HANDOFF.md); A3–A6 sind in A2
nicht begonnen und hier nicht freigegeben.
