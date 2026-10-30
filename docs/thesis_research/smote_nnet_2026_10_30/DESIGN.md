# SMOTE und einfaches Netz: Design vor dem ersten Vergleichslauf

Stand 30.10.2026. Vom Nutzer ausdrücklich gewünschte Fortsetzung: die in seiner
Dissertation erfolgreiche Kombination SMOTE + einfaches neuronales Netz
nachbilden und an den sieben neuen Artenfunktionen prüfen.

Basis bleibt der offene #18 auf f9acc16. Die Zusatzarbeit aus #28 wird
**explizit über den gesperrten Commit 8f02794434f561f5b30228360247935f3c025cc8**
als Eingang bezogen; dieser neue Branch enthält sie nicht automatisch.
Keine Änderung der bisherigen Analog-Baseline oder der #28-Referenz.

## Belegter historischer Ansatz und Nachbildungsgrenze

Dissertation 2.5 (gedruckte S. 40–43): binäre Taxonklassifikation, inverse
CDF-Normierung, SMOTE nur auf Training, einfaches Feedforward-Netz aus `nnet`
mit einer verborgenen Schicht. Der veröffentlichte ursprüngliche R-Lader
(064dca10) belegt T-Standardisierung und `sqrt(P) → Gamma-CDF → Probit` für die
damalige Biomfunktion. Der 2025-Artikel benennt Balanced Accuracy als
Vergleichsmaß und sechs versteckte Neuronen für sein Biomnetz.

Diese Quellen belegen **keinen hier verifizierten identischen Trainingslauf
für alle damaligen 279 Taxa**. Sechs Neuronen sind hier eine vorab festgelegte
kleine Netzarchitektur, keine behauptete historische Einstellung jeder Art.
Neue, offen festgelegte Einstellungen: Decay 0,1, drei Startwerte, Parameter
der Gamma-Anpassung, SMOTE-Verhältnis 1:1, Zufallszahlen und räumlicher Test.
Die Daten bleiben die neue Meusel-/WorldClim-Kalibrierung aus #28, nicht die
historischen CRU-4.01-Daten. Die Python-Nachbildung hat dieselbe einfache
Netzklasse und Entropie-/Weight-decay-Zielform wie binäres `nnet`; sie ist keine
bitgleiche R-/caret-Reproduktion. R-Programme aus den Quellen werden nicht ausgeführt.

## Fester Vergleich

1. Unveränderte quadratische Funktionen aus #28 als Referenz.
2. Einfaches Netz ohne SMOTE.
3. Dasselbe Netz mit SMOTE, ausschließlich aus Original-Trainingsfällen.
4. Zusätzlich dieselben SMOTE-Netzgewichte mit einer nur aus Trainingsdaten
   bestimmten Prior-Odds-Anpassung als **Diagnose**, nicht als garantierte
   Kalibrierung. SMOTE verändert auch die bedingte Merkmalsverteilung.

Alle vier Varianten erhalten dieselben Original-Testzellen. Beide Netzvarianten
verwenden dieselbe trainingsintern geschätzte Normierung, Architektur, Decay
und gepaarte Initialisierungen. Dadurch lässt sich der SMOTE-Effekt innerhalb
dieser Netzklasse prüfen. Der Unterschied zur #28-Regression umfasst auch die
andere Modellklasse und Normierung; er wird nicht allein SMOTE zugeschrieben.

[config.json](./config.json) fixiert sämtliche Einstellungen vor der Auswertung.
Keine Hyperparameter- oder Schwellenwahl anhand der äußeren Testdaten.
Bei jedem Fit werden drei Initialisierungen gerechnet und ausschließlich nach
dem penalisierten Trainingsziel unter stationären Kandidaten gewählt.
Keine automatische Ablösung der Regression aufgrund eines einzelnen Scores.

## Daten und Ausschluss von Leckage

Die sieben Arten, Quellenhashes, Kartierungsregeln und Zellmasken aus #28 bleiben
unverändert. Gleiches 0,5°-Arbeitsgitter, 20° W–180° E / 20°–80° N.
Fünf äußere Folds je 5°-/10°-Gitter, 100-km-Puffer; gleicher Test wie #28.
Normierung und Prävalenzen werden für jeden Fold aus seinem **ursprünglichen
Training vor SMOTE** bestimmt. Test und Puffer liefern weder Elternpunkte
noch Normalisierungs-, Auswahl- oder Kalibrierungsparameter.

SMOTE interpoliert zwischen einem Minderheits-Trainingsfall und einem seiner
fünf nächsten Minderheitsnachbarn im normierten Klimaraum bis zum Verhältnis
1:1 der Zellzahlen. Originalfälle bleiben erhalten; Mehrheitsfälle werden nicht
entfernt. Die Gewichtung synthetischer Fälle interpoliert die Gewichte beider
Eltern. Das sind Fallgewichte, keine neu beobachteten Flächen oder erfundenen
geographischen Koordinaten. Beide ursprünglichen Zell-IDs und der Mischanteil
werden für jeden synthetischen Fall nachvollziehbar protokolliert.

## Zwei verschiedene Bedeutungen von „besser“

Für den historischen Klassifikationsvergleich berichten wir **Balanced
Accuracy bei Schwelle 0,5**, Sensitivität und Spezifität; zusätzlich deren
flächengewichtete Fassung. Für die spätere probabilistische Verwendung bleiben
Brier-Score, Brier-Skill, Log-Loss und Reliabilität erforderlich. Eine höhere
Trefferquote ist nicht automatisch eine besser kalibrierte Wahrscheinlichkeit.
Die Ausgaben des auf umgeschichteten Klassen trainierten Netzes werden als
solche bezeichnet. Die Original-Testprävalenz bleibt erhalten.

Kein Feld-, Pollen- oder Fossiltest: bewertet wird weiterhin der native
kartierte Arealindikator unter denselben begrenzten Kartenannahmen wie #28.
Alle sieben Arten und beide äußeren Blockgrößen werden berichtet, auch bei
Verschlechterung. Genaues Ergebnis erst nach dem Lauf eintragen.

## Vorgesehene Prüfung und Lieferung

Gradienten gegen finite Differenzen; SMOTE-Eltern, Nachbarschaften und
Interpolation synthetisch testen. Ein separater Prüfer soll alle echten
Testvorhersagen, beide Eltern jedes synthetischen Falls, Trainingsausschlüsse,
Netzvorwärtsrechnung, Gradienten, Startauswahl, Scores und Quelle-/Ausgabehashes
prüfen. Gemeinsame Bibliotheken und Abhängigkeiten von #28 werden benannt.
Vollständige Wiederholung mit festen Zufallszahlen und Hashvergleich.

Speichern: ausführbare neue Artennetz-Parameter, Foldparameter, kompakte
Ergebnistabellen, Quellen-/Parameterregister, Audit, Reproduktion und Übergabe
auf eigenem Branch als PR. Große Rohdaten und synthetische Laufdateien werden
reproduziert. CLI/App bleibt bei Claude. P1 und MON16-Crosswalk bleiben offen;
keine fossile Freigabe, Quellenkorrektur, Autorennachricht oder Merge.
