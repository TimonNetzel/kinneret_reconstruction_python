# Räumlich verschachtelte Wahrscheinlichkeitskalibrierung: Vorabdesign

Festgelegt am 30.10.2026 **vor jedem neuen Netz- oder Kalibrierungsfit**.
Basis: aktuelles main `7f1826cffddc8893ba6354cd3cca3fa43954b9be` nach #30;
#18–#29 sind gemergt. Eigenständiger Forschungsbranch; kein Mergeauftrag.

## Fragestellung und unveränderliche Eingänge

Kann eine einfache trainingsinterne logistische Kalibrierung die
Wahrscheinlichkeitsgüte der sieben neuen Artennetze mit/ohne SMOTE verbessern?
Es wird weder anhand der äußeren #29-Scores ein Kalibrator ausgewählt noch
eine neue Netzarchitektur, Schwelle, Taxonauswahl oder Regularisierung gesucht.
Die #29-Modelle, Transformationen, drei Trainingsstarts und SMOTE-Regeln bleiben
als Modellrezept unverändert. Quellen, Zellmasken, sieben Arten, Flächengewichte
und gepufferte äußere 5°-/10°-Fünffachteilung bleiben exakt wie #28.
Gesperrte Quellen: `../smote_nnet_2026_10_30/sources.json` und `audit_sources.json`.
Parameter stehen vollständig in [config.json](./config.json).

## Innere Teilungen und Paarung ohne Refit-Verschiebung

Für jeden äußeren Fold zunächst äußere Testzellen und dessen 100-km-Puffer
ausschließen. Nur auf den verbleibenden Original-Trainingszellen drei innere
räumliche Folds bilden: gleiche Blockgröße wie außen, globale Ursprünge
−180°/−90°, `(Breitenblock + 2*Längenblock) modulo 3`. Jeder innere Fold ist
Kalibrierungsmenge; sein Netztraining besteht aus den übrigen inneren Zellen
mit zusätzlichem 100-km-Abstand zu jeder Kalibrierungszelle.

**Je innerem Mitglied**: T-/P-Transformation und Klimahülle ausschließlich auf
seinem ursprünglichen Netztraining schätzen; SMOTE ausschließlich daraus,
mit beiden Originaleltern und Mischanteil protokollieren. Beide Netzvarianten
erhalten dieselben Originalfälle und gepaarten festen Startseeds. Startwahl
nur nach stationärem penalisiertem Trainingsziel, niemals nach Kalibrierung.
Kalibrierungszellen werden weder transformierend angepasst noch resampelt.
Sie behalten ihre ursprünglichen Labels, Häufigkeiten und cos(Breite)-Gewichte.

Jedes Mitglied erhält seinen eigenen Kalibrator aus seinen zurückgehaltenen
Originalzellen. Äußere Vorhersagen sind das arithmetische Mittel der drei
Mitgliedswahrscheinlichkeiten, jeweils roh bzw. nach Mitgliedskalibrierung.
**Kein nachträglicher Volltrainings-Refit** unter einem Kalibrator, der Scores
kleinerer Netze gelernt hat. Auch die rohe Vergleichsvariante ist dasselbe
Dreierensemble. Der direkte Kalibrierungseffekt wird dadurch innerhalb
derselben Netzfits gemessen; kein isolierter Vergleich mit alten #29-Zahlen.
Für spätere Vorwärtsanwendung pro Art und Blockdesign analog drei Mitglieder
auf allen verwendbaren Originalzellen speichern. Dies ist kein zusätzlicher Test.

## Ein einziger vorab festgelegter Kalibrator

`p = sigmoid(a*eta + b)`, eta = direkter Netzlogit vor Ausgangssigmoid.
Flächengewichtete mittlere Bernoulli-NLL plus `0.0001*a²/2` minimieren;
`a >= 0`, unbestrafter Intercept b, Start `(1,0)`, L-BFGS-B, maximal 2000
Iterationen. Kein Wahrscheinlichkeits-Clipping zur Logitbildung; Logits direkt
aus den Netzgewichten berechnen. Dies ist eine fest gewählte monotone
logistische Rekalibrierung, keine Suche zwischen Isotonie, Temperatur- oder
Beta-Verfahren und keine Behauptung einer exakt historischen Platt-Prozedur.
Projected-gradient-Gate `1e-7`, Bound bei a=0 berücksichtigt.

Netztraining benötigt mindestens 30 Originalfälle pro Klasse; sonst abbrechen
und berichten, keine Splitänderung. Für einen Kalibrator mindestens fünf
Originalfälle jeder Klasse; sonst explizit Identität `(1,0)` und Flag
`insufficient_calibration_class` speichern. Numerisches Kalibrierungsversagen
führt zum Abbruch, keiner stillen Identität. Keine äußere Auswahl, kein
automatischer Ersatz und keine Nachjustierung bei Verschlechterung.

## Äußere Bewertung und Berichtsregeln

Vier Varianten auf exakt denselben ursprünglichen äußeren Testzellen:
Originalnetz-Ensemble roh/kalibriert und SMOTE-Ensemble roh/kalibriert.
Alle sieben Arten und beide Blockdesigns berichten; keine Siegervorauswahl.
Primär flächengewichteter Brier und gepaarte Brier-Differenz kalibriert minus
roh. Zusätzlich Log-Loss (nur zur Auswertung auf `[1e-12,1−1e-12]` begrenzt),
Brier-Skill gegen Prävalenz des äußeren Originaltrainings, BA bei 0,5,
Sensitivität/Spezifität, flächengewichtete BA, mittlere Prognose, Prävalenz und
zehn feste Reliabilitätsbins. Foldwerte und gesamte Out-of-fold-Werte speichern.
Leere Testklasse bleibt undefiniert. Extrapolationsfälle bleiben im primären
Score und erhalten Anzahl der drei Trainingshüllen, die sie einschließen.
Vorwärtsevaluator verweigert standardmäßig Punkte außerhalb irgendeiner
Mitgliedshülle; keine Änderung dieser Regel nach Ergebnissen.

Äußere Labels werden nur für Scores benutzt. Sie beeinflussen weder
Transformation, SMOTE, Netzstartauswahl noch Kalibrierung. Die äußere Bewertung
ist vom jeweiligen Fit getrennt, **aber die Kartenregionen wurden schon in
#28/#29 untersucht**. Dies ist eine verschachtelte methodische Folgeauswertung
auf bekannten Quellen/Regionen, keine vollständig ungesehene Bestätigung,
unabhängige Feldvalidierung oder Kalibrierung fossiler Pollenvorkommen.
Die beiden Blockdesigns sind keine unabhängigen Replikationsdatensätze.

## Prüfung und Reproduktion

Separates Prüfprogramm ohne Import des neuen Produzenten/Kalibrators:
Originalquellen und Labels mit gesperrtem #28-Vektorprüfer rekonstruieren;
äußere/innere Zellrollen mit Haversine-BallTree statt Chord-cKDTree prüfen;
beide SMOTE-Eltern, Nachbarn, Mischwerte, Gewichtung, Normierung, sämtliche
Netzstarts/-ziele und Kalibrator-Gradienten nachrechnen; jede Vorhersage,
Ensemblebildung, Score- und Reliabilitätszeile bestätigen. Gemeinsame
NumPy/SciPy/GDAL/GEOS und gesperrte alte Auditwerkzeuge offenlegen.
Dies ist eine separate numerische Prüfung, kein zweiter menschlicher Reviewer.
Kompletter deterministischer Wiederholungslauf mit Ausgabehashvergleich.
Synthetische Tests sollen räumliche Ausschlüsse, Kalibratorgradienten,
Degenerationsgate, Vorwärtsevaluator und Ablehnung von Extrapolation prüfen.

Parameter, kompakte Tabellen, Audit, Hashes und Reproduktionsanleitung als PR;
große Laufdateien/Roharchive reproduzierbar, nicht neu verteilen. Design und
Konfiguration vor Trainingsbeginn veröffentlichen und anschließend unverändert
lassen; technische Abweichungen explizit dokumentieren.

## Nicht gelöster fachlicher Umfang

Nur experimentelle kartierte native Arealindikatoren; kartiertes Außerhalb
ist kein erhobener Abwesenheitsnachweis. Weitere Taxa, belegte Pollentyp-
Art-/Gattungszuordnung, Kalibrierungs-/Quellenunsicherheit und gemeinsame
Beobachtungsmodelle bleiben offen. Ensemblevariation ist kein
Unsicherheitsintervall. Keine fossile Klimakurve oder Pilotfreigabe.
Monticchio-P1 bleibt offen: physischer LGM90D→MON16-Anschluss einschließlich
Korrelationsunsicherheit fehlt. 28 Kandidaten / 41 Nichttreffer / 47 Herkunftsfragen
unverändert. CLI/App bei Claude; Analog-Baseline unverändert. Keine stillen
Quellenkorrekturen, Autorennachrichten oder Merges.
