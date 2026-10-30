# Bedingte räumliche Kalibrierungsunsicherheit: Vorabdesign

30.10.2026, vor jeder neuen Bootstrap-Kalibrierung festgelegt. Fortsetzung von
PR #31, gesperrter Head `9458011198c69368764b8698c966c0ac23cc381c`.
#31 ist offen; neuer abhängiger Forschungsbranch, keine Merges. Seine Netze,
Normierungen, SMOTE-Erzeugung, Klassenregeln und äußeren Punktprognosen bleiben
unverändert. Kein neuer Modell-/Schwellenvergleich nach äußeren Ergebnissen.

## Ziel und begrenzte Aussage

Beschreiben, wie empfindlich die Wahrscheinlichkeiten gegenüber der räumlichen
Zusammensetzung der **inneren Kalibrierungszellen** sind, bedingt auf die bereits
trainierten Netze und festgelegten Kartenquellen. Dies ist eine räumliche
Block-Bootstrap-Sensitivitätsanalyse der Kalibrierungsstufe, keine vollständige
Parameter-, Karten-, Areal-, Taxonomie-, Fossil- oder Klimarekonstruktionsunsicherheit.
Keine nominell validierten 95%-Konfidenzintervalle, keine Vorhersageintervalle
für binäre Einzelbeobachtungen. Drei Netzmitglieder allein werden weiterhin
nicht als Unsicherheitsintervall bezeichnet.

Die Methodikliteratur motiviert Block- statt Einzelzellenresampling bei
Abhängigkeit. Sie garantiert keine Deckung für unsere maskierten Karten,
festen Blockgrößen, Regularisierung und seltenen Klassen. Lahiri/Zhu zeigen
gerade Grenzen naiver Übertragung räumlicher Bootstrapverfahren bei nichtuniformen
Designs. [Quellen und genaue Lesegrenzen](./SOURCES.md).

## Fester Bootstrap

Für alle sieben Arten und beide vorhandenen Blockdesigns (5° und 10°), alle
fünf äußeren Folds und analog die Vollgebietsmodelle:

1. Äußere Original-Testzellen und 100-km-Puffer wie #31 ausschließen. Aus dem
   verbleibenden ursprünglichen äußeren Trainingspool seine belegten globalen
   geographischen Blöcke bestimmen. IDs: ganzzahlige Breiten-/Längenblockindizes
   ab −90°/−180°, lexikographisch sortiert (Breite, Länge).
2. Je Ensemble **200 feste Replikate**. Je Replikat G Blöcke mit Zurücklegen
   aus seinen G belegten Trainingsblöcken ziehen; PCG64 und deklarierter
   SHA256-Seed. Multiplizitäten für Original/SMOTE und alle drei inneren
   Mitglieder desselben Ensembles identisch. Keine äußeren Test-/Pufferblöcke
   als Beobachtungen hinzufügen. Blöcke können nach Maskierung fragmentiert sein.
3. Für jedes Mitglied nur seine unveränderten inneren Kalibrierungszellen
   verwenden. Gewicht = ursprüngliches cos(Breite)-Gewicht × Blockmultiplizität.
   Nullgewichte entfernen. Netzlogits bleiben fest. Keine Neuschätzung der
   Transformation, kein neues SMOTE und kein Netztraining.
4. Dieselbe monotone logistische Kalibrierung wie #31 fitten: a≥0,
   flächengewichtete mittlere NLL + 0,0001·a²/2, unbestrafter Intercept,
   gleiche numerische KKT-Grenze 1e-7. Mindestens fünf **verschiedene ursprüngliche
   aktive Zellen jeder Klasse**, nicht fünf durch Duplizieren erzeugte Fälle.
   Unterschreitung macht den Mitgliedsfit und das Ensemble-Replikat nicht
   schätzbar. Numerisches Versagen wird separat markiert, nicht durch Identität
   ersetzt; keine zusätzlichen Starts oder angepassten Toleranzen.
5. War ein ursprünglicher #31-Mitgliedskalibrator bereits Identität wegen
   Klassenknappheit, ist das gesamte Ensemble für diese Intervallanalyse
   nicht schätzbar. Keine Nullbreite als angebliche Sicherheit speichern.
6. Pro schätzbarem Replikat je Familie die drei kalibrierten
   Mitgliedswahrscheinlichkeiten mitteln. Auf den unveränderten äußeren
   Klimazellen fortpflanzen; alle Zellen samt Supportzahl 0–3 behalten.
   Die Konstruktion benötigt **keine äußeren Labels**.

Parameterregister und Blockmultiplizitäten vollständig speichern; alle
Nichtschätzbarkeiten mit Grund und Anzahl berichten. Seeds werden aus
`master_seed|taxon:bBLOCK:oOUTER|block_bootstrap` abgeleitet. Replikate werden
in fester Reihenfolge erzeugt, keine von Scores abhängige Auswahl.

## Intervalle, Auswertung und Mindestschätzbarkeit

Nur wenn mindestens **190/200 Replikate je Familie** alle drei Mitglieder
erfolgreich enthalten: punktweise Quantile 0,025 / 0,5 / 0,975 mit linearer
Interpolation. Diese werden als **zentrales 95%-Bootstrap-Sensitivitätsband,
bedingt auf schätzbare Replikate und feste Netze** bezeichnet. Auch bei
erfülltem Gate fehlen Verteilungen nichtschätzbarer Replikate; diese Bedingung
und Ausfallrate immer nennen. Gate oder Blockgrößen nicht nach Ergebnissen ändern.
Sonst leere Intervallwerte und explizites Nichtschätzbarkeitsflag.

Berichten: alle Art×Block×Fold×Familien einschließlich Vollgebietsstand;
gültige Replikate, aktive Klassen und positive Kalibrierungsblöcke pro Mitglied,
Intervallverfügbarkeit, flächengewichtete mittlere Breite und Breitenquantile
auf den äußeren Testzellen. Vollgebiets-Bänder auf demselben verwendbaren
Klimagitter sind Anwendung, kein weiterer Test. Keine Klassifikationsschwellen
optimieren, keine Modelle anhand der Breite auswählen, keine Brier-Sieger
neu bestimmen. Keine binäre Label-Deckungsrate: ein Intervall für p ist kein
Intervall für eine Bernoulli-Realisierung. Keine nominelle Deckungsbehauptung
ohne zusätzliche validierte Simulation/externen Test.

Beide Blockgrößen berichten; keine datenbasierte Blocklängenwahl. Kein gemeinsames
Unsicherheitsmodell über Arten: taxonspezifische Resampling-Seeds liefern keine
validierte Kreuztaxon-Kovarianz für eine Multiproxy-Inversion.

## Separate Prüfung, Reproduktion und Lieferung

Ein separater Prüfer importiert weder neuen Produzenten noch Bootstrap-Fitter.
Gesperrte #28-Vektorleser/#29-Auditarithmetik und separate #31-Haversine-
Teilungen dürfen ausdrücklich wiederverwendet werden. Quell-/Code-/Modellhashes,
alle äußeren/inneren Rollen, Blockauswahl und Multiplizitäten prüfen. Jeden
gültigen Kalibrator über eigene konvexe Zielfunktion/KKT nachrechnen;
Klassen-/Numerikausfälle bestätigen. Jede Intervallvorhersage, jedes Quantil,
Supportflag und jede kompakte Zusammenfassung separat prüfen. Gemeinsame
GIS-/Numerikbibliotheken offenlegen; kein zweiter menschlicher Reviewer.

Zwei vollständige deterministische Läufe mit Hashvergleich. Ausgaben erst
nach geschlossenen Schreibern lesen; atomare Abschlussdateien statt laufend
überschriebener Modellcheckpoints. Quellen/Modelle unverändert erhalten.
Große Zelltabellen und Multiplizitätsprotokolle reproduzierbar; kompakte
Parameter, Zusammenfassungen, Hashes und Audit als eigener PR gegen #31.

Weitere Taxa, belegte Pollentyp-Zuordnung und vollständige Kalibrierungs-/Quellen-
unsicherheit bleiben offen. Bekannte Regionen aus #28–#31 sind keine vollständig
ungesehene Bestätigung. Keine Änderung an CLI/App oder Analog-Baseline.
Monticchio-P1 und physische LGM90D→MON16-Zuordnung samt Korrelationsunsicherheit
sowie 28 Kandidaten / 41 Nichttreffer / 47 Herkunftsfragen unverändert.
Keine Quellenkorrektur, Pilotfreigabe, Autorennachricht oder Merge.
