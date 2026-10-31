# A1: Methodenabgrenzung und Verbindung mit der Dissertation

Stand 31.10.2026; Basis PR #50, `64a652982b0f271c02bfc0c89331537d6ec08eaa`.
**Begrenzter Vergleich abgeschlossen, mit Zugangslücke bei Parnell. Keine
Neuheit nachgewiesen; kein eigener Beitragskandidat belastbar abgegrenzt.**
Die Matrix beurteilt die Forschungsrichtung aus
[Plan §1](./MODULAR_RECONSTRUCTION_PLAN.md#1-ziel-und-festgehaltene-arbeitsrichtung),
keine gewählte reale Methode. Nach den drei vorgegebenen Arbeiten beendet.

## Quellen und Beleggrenze

| Kürzel | Original und Version | Geprüfter Zugang / Reichweite |
| --- | --- | --- |
| P | Parnell et al. 2015, [DOI 10.1111/rssc.12065](https://doi.org/10.1111/rssc.12065) | DOI-Abruf fehlgeschlagen. Auch der bereits im Projekt belegte [Verlagslocator](https://academic.oup.com/jrsssc/article-abstract/64/1/115/7067578) leitete auf eine nicht erreichbare CDN-Seite weiter. Keine Originalgleichung oder Methodensektion hier verifiziert. Keine Ersatzquelle gesucht. |
| B | Bowen et al. 2020, [Verlagsvolltext](https://cp.copernicus.org/articles/16/65/2020/), DOI 10.5194/cp-16-65-2020 | Gezielt §3, §3.1–3.3 und §4.4 gelesen; Gleichungen (1)–(5). |
| M | Menezes et al. 2026, [arXiv:2609.08866v1](https://arxiv.org/html/2609.08866v1), 08.09.2026 | Gezielt §1.3, §2.1–2.3, §3/§4.2, §5 und Anhang A, Algorithmus 4. Nur v1, keine Codeprüfung oder neue Publikations-/Versionssuche. |

Konkrete Nachladefragen waren die Rollen der Normierungen in M (2)/(5)
und der belegte Anschluss an die Dissertation. Dafür wurden ausschließlich
§„Belegter Anschluss“, §„Empfohlene Verbindung“ und §„Was aus der Dissertation
erhalten und verbessert würde“ des vorhandenen
[Integrationsmemos](./menezes_integration_2026_10_31/INTEGRATION_MEMO.md)
herangezogen; aus der Erstprüfung nur der schon bekannte P-Verlagslocator.
Die Dissertation selbst wurde nicht neu im Original geprüft. Aussagen über
ihre bisherige Architektur sind entsprechend Projektstand, kein neuer
Seiten-/Gleichungsnachweis ihrer Originalfassung. M-Zitate über P ersetzen
die fehlende Primärprüfung von P ausdrücklich nicht.

## Sechs Prüffelder

Die Originalspalten enthalten nur belegte Befunde bzw. benannte Zugangslücken.
Die letzten beiden Spalten sind unser Vergleich mit Plan §1 und offene Fragen.

| Prüffeld | P: Originalbeleg | B: Originalbeleg | M: Originalbeleg | Gemeinsamkeit / Unterschied zum Plan | Offen |
| --- | --- | --- | --- | --- | --- |
| Beobachtung / Nullen | Original nicht erreichbar; Count-Likelihood, Nullmodell und Nenner hier ungeprüft. | §3.1, (1)–(3): normalverteilte Proxy-Beobachtungen; kein Pollen-Nullmodell. | §2.2, (3)/(4): ZANIM-LN-BART; Counts, strukturelle Nullen, Überdispersion, Taxonabhängigkeit. | M bietet den Anschluss für ein kalibriertes Pollen-Beobachtungsmodell; B illustriert eigene Beobachtungsmodelle anderer Proxys. Ein Arealvorkommen ist weiterhin kein Pollenanteil. | Echte Counts/N und Messstatus nötig; Prozent-only benötigt ein eigenes Präzisions-/Beobachtungsmodell. N=100 ist keine Lösung. P nicht gleichsetzen. |
| Kalibrierbrücke | Kalibrier-Cut und dessen Normierung im Original nicht geprüft. | §3.3: Proxy- und Kalibrierdaten informieren gemeinsam die Parameter. | §2.1, (2): moderne Kalibrierung geschützt; §2.3, (5), siehe unten. | Plan übernimmt die Cut-Richtung von M; B erlaubt die vom Plan unterbrochene fossile Rückkopplung. Kalibrierunsicherheit muss trotzdem übertragen werden. | M (2)/(5) auseinanderhalten; Übertragbarkeit/Taxonidentität und Klimabegriffe sind keine Folge des Cuts. |
| Klimaprozess | Zeit-/Volatilitätsmodell: Originalgleichung nicht verifiziert. | §3.2, (4)/(5): korrelierter Random Walk. | §1.3/§5: keine zeitliche Kopplung; zeitlicher Ausbau vorgeschlagen. | Ein gemeinsamer Verlauf ist Anschluss an die Dissertation und B; gegenüber dem tatsächlich verwendeten M-Modell eine Erweiterung. Ein glatter Plot allein belegt keinen Zeitprior. | Prozess, Support, Klimaanker und Identifizierbarkeit unbestimmt; keine reale Priorwahl aus dem Vergleich. |
| Chronologie / Rückkopplung | Behandlung unsicherer Alter und Rückkanäle im Original offen. | §3, letzter Absatz: Altersunsicherheit möglich, im Beispiel weggelassen. | §1.3/§5: Altersunsicherheit fehlt; Chronologieausbau vorgeschlagen. | Plan erlaubt gemeinsame Klima-/Chronologie-Inferenz innerhalb des nachgelagerten Blocks. B zeigt dafür in diesem Beispiel keinen ausgeführten Nachweis; M ebenfalls nicht. | Ganze Alterskurven statt unabhängiger Altersfehler; Datierungsanker und klimatische Information nicht doppelt nutzen. Ein Chronologie-Cut wäre eine weitere Entscheidung. |
| Proxy-Abhängigkeit | Zusätzliche Proxy-Likelihoods und Abhängigkeiten ungeprüft. | §3.1, (2)/(3): gemeinsame Umweltgrößen; §3.3: gemeinsame Kalibrierparameter beider Standorte. | §2.2: Abhängigkeit zwischen Pollentaxa; (1)/(2): keine zusätzlichen externen Proxy-Modelle. | B belegt gemeinsame Inversion mehrerer Proxys; M-Taxonabhängigkeit ersetzt keine externe Proxy-Abhängigkeitsprüfung. Plan erweitert den zweiten Block. | AP/NAP aus denselben Counts ist keine unabhängige Replikation. Gemeinsame Archive, Datierungen, Kalibrierdaten und Störgrößen können bedingte Unabhängigkeit verhindern. |
| Unsicherheitsvalidierung | Originalvalidierung nicht geprüft. | §3.3: Konvergenzprüfung, schwache Durchmischung eines Prozesses offengelegt; §4.4: gemeinsame Unsicherheit beeinflusst Standortvergleich. | §3/§4.2: Simulationen und zufälliger moderner Holdout mit Neukalibrierung. | Vorarbeiten liefern Prüfideen, keine Validierung unseres zeitlichen Multiproxyblocks. Konvergenz und Intervallabdeckung sind verschiedene Fragen. | Räumlicher Transfer, Zeit-/Chronologieabdeckung und Proxy-Abhängigkeit hier unvalidiert. Keine Scores, Verfahren oder Garantien auf unser Projekt übertragen. |

## Kalibrier-Cut: bedingte Normierung explizit

Eigene algebraische Prüfung, keine neue Modellwahl: Schreibe für eine Probe
\(q(\psi)=p(\psi\mid E)\), \(L_s(c,\psi)=p(P_s\mid c,\psi)\) und
\(Z_s(\psi)=\int L_s(c,\psi)\pi_s(c)\,dc\).
M §2.1, Gl. (2) enthält genau die bedingten Normierer
\(\mathcal Z(\mathbf y_s\mid\bm\Psi)\); der moderne Normierer
\(\mathcal Z(\mathbf y^m)\) hat eine andere Rolle. Die aus (2) folgende
Randdichte lautet in unserer Kurznotation:

\[
m_{cut,s}(c)=\int q(\psi)
  \frac{L_s(c,\psi)\pi_s(c)}{Z_s(\psi)}\,d\psi.
\]

M §2.3, Gl. (5) schreibt dagegen proportional
\(\pi_s(c)\int L_s(c,\psi)q(\psi)\,d\psi\).
Wörtlich gelesen und einmal über c normiert ergibt das

\[
m_{avg,s}(c)=
\frac{\pi_s(c)\int L_s(c,\psi)q(\psi)\,d\psi}
     {\int q(\psi)Z_s(\psi)\,d\psi}.
\]

Das ist im Allgemeinen eine andere Mischung: Ihre Komponenten tragen Gewichte
proportional zu \(q(\psi)Z_s(\psi)\), statt zu \(q(\psi)\).
Ein \(\psi\)-abhängiger Faktor innerhalb des Integrals kann durch ein äußeres
Proportionalitätszeichen nicht allgemein entfernt werden. Konstantes
\(Z_s\) wäre eine hinreichende Sonderbedingung; eine solche Bedingung ist
hier nicht belegt. Daraus folgt ein **offener Formelklärpunkt zwischen (2)
und der wörtlichen Lesart von (5)**, keine pauschale Fehlerbehauptung über
Verfahren, Software oder Resultate.

Gegenprüfung am beschriebenen Verfahren: M §2.3.1 und Anhang A,
Algorithmus 4, resampeln für jedes r innerhalb der zugehörigen Vorschläge.
Die Gewichte werden damit pro Kalibrierziehung normiert. Das passt zur
bedingt normierten Mischung aus (2). Endliche Monte-Carlo-Approximation und
veröffentlichte Formel sind getrennt zu beurteilen. §2.3.2 benennt zudem
Konvergenzvorbehalte beim naiven Cut von (c)ESS. Keine der Implementierungen
wurde hier geprüft, kein Sampler gewählt.

Für Plan §1 ist die Konsequenz rein algebraisch. Mit
\(U=(C,\tau,\theta)\) und einem noch zu spezifizierenden gemeinsamen
Rekonstruktionskern \(F_\psi(U;P,Y,D)\) gilt:

\[
Z(\psi)=\int F_\psi(U;P,Y,D)\,dU,\qquad
p_{cut}(\psi,U)=q(\psi)\frac{F_\psi(U;P,Y,D)}{Z(\psi)}.
\]

So bleibt \(\int p_{cut}(\psi,U)\,dU=q(\psi)\), sofern
\(0<Z(\psi)<\infty\) für q-fast alle psi. Unmögliche Beobachtungen oder
fehlender Support sind Stopgründe, keine wegzulassenden Ziehungen.
Dieselbe psi-Ziehung muss alle betroffenen Proben gemeinsam beschreiben.
Bereits gemeinsame Kalibrierunsicherheit kann nach Ausmitteln von psi
Probenabhängigkeit erzeugen; sie ist noch kein zeitlicher Klimaprozess.
Mit einem gemeinsamen Verlauf, Chronologien und weiteren Proxys faktorisiert
Z im Allgemeinen nicht mehr in einzelne Probennormierer. Auch das Produkt
einzeln bereits ausgemittelter Rekonstruktionen ist dann kein Ersatz.

## Verbindung mit der Dissertation: Reichweite und Grenze

Der im Integrationsmemo belegte Projektanschluss ist sinnvoll: Taxonbezug,
gemeinsamer Klimaverlauf, Chronologie, Klimaanker und weitere Proxys bleiben
die inhaltliche Architektur; die Pollenbrücke erhält eine moderne
Beobachtungskalibrierung mit übertragener Unsicherheit. Diese Verbindung ist
eine Weiterentwicklungsidee, kein Originalnachweis zur Dissertation und
kein empirischer Methodenentscheid.

Die bisherigen sieben kartierten Artenfunktionen bleiben unveränderte
Vergleichsreferenzen. Ihre Interpretation als Pollenanteilsfunktionen ist
nicht belegt. Eine mögliche spätere Verwendung als kalibrierte Merkmale
würde eigene Evidenz benötigen. TANN/PANN und GDD5/MTCO/AET-PET sind keine
umbenennbaren Größen. Keiner der drei Vergleiche löst diese Anschlussfragen.

Zusätzliche Proxys benötigen einen eigenen Mess-/Fehlerprozess, zeitlichen
Bezug und gegebenenfalls gemeinsame Störgrößen. Ein alleiniger Beta(R²)-Faktor
ersetzt das nicht. Das Memo beschreibt außerdem priornahe Auswahl und
Bandverengung durch Posterior-Kopien als abzulösende Schritte; hier wird
keines dieser bestehenden Verfahren geändert oder nachgerechnet.

Im geplanten zweiten Block können Pollen und zusätzliche Proxys die
Chronologie mitinformieren, während psi geschützt bleibt. Soll ausschließlich
Datierungsevidenz die Chronologie bestimmen, wäre eine andere modulare
Zielverteilung zu begründen. Das ist noch keine in A1 getroffene Entscheidung.
Abhängigkeiten zwischen Datierungsankern, Klimaankern und Proxybeobachtungen
bleiben ausdrücklich offen.

## Abgrenzungsentscheidung und offene Punkte

**Kein eigener Beitragskandidat abgegrenzt.** Die Kombination von modernem
Kalibrier-Cut, Zeitprozess und Chronologie ist anhand dieser Prüfung nicht
als neu belegbar: M §5 kündigt den zeitlichen/chronologischen Ausbau selbst
an; B demonstriert gemeinsame Proxy-Inversion; für P fehlt der Originalzugang.
Die hier ausgeschriebene Normierungsidentität ist eine notwendige Eigenschaft
der geplanten Zielverteilung, für sich kein Neuheitsbeleg. Zusätzliche
Komponenten allein begründen ebenfalls keine Neuheit.

Offen bleiben P-Originalgleichungen und Validierung; die genaue Lesart von
M (5); die Originalzuordnung der Dissertation; Beobachtungs-/Präzisionsmodell,
Klimabegriffe, gemeinsame Chronologie-/Proxyabhängigkeiten und Identifizierbarkeit.
Dies sind dokumentierte Grenzen, keine weiteren aktiven Rechercheaufträge.
Keine Schneeballsuche, Ersatzliteratur oder Autorennachrichten.

## Abschluss und genau ein vorbereiteter Folgeauftrag

A1 ist als begrenztes Dokumentationspaket erledigt, einschließlich der
Zugangslücke. Keine Datenwahl/-beschaffung, Bereinigung, Datenmatrix/Kohorte,
Implementierung, Tests, Fits, Scores, Simulationen oder Kernläufe ausgeführt.
Bestehende Designs, Modelle, Resultate, Register, Verträge, Importer, Flags,
Counts/Gruppen, Baseline und CLI/App bleiben unverändert; W96 und reale Gates
offen. Quellenlocator, algebraische Aussagen, lokale Links und Diffumfang
werden als Dokumentation geprüft.

Genau ein Folgeauftrag ist in [HANDOFF](./HANDOFF.md) konkret vorbereitet:
A2, ausschließlich ein dokumentarischer MODEL_CONTRACT.md mit expliziten
offenen Feldern und Stopbedingungen; **wartet auf Nutzer-OK**. A2–A6 wurden
nicht begonnen. Die fehlenden Primärbelege bleiben im Vertrag als Grenzen
sichtbar und werden nicht durch Annahmen ersetzt.
