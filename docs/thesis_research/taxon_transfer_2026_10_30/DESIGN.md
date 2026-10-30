# Erstes Taxon-Training: vor dem Ergebnislauf festgelegtes Design

Stand 30.10.2026. Neuer, vom Nutzer ausdrücklich beauftragter Arbeitsstrang,
parallel zum offenen Monticchio-P1. Basis ist der weiterhin offene PR #18
auf f9acc16; keine Änderungen an seiner fertigen Analog-Baseline oder CLI/App.

Ziel dieses ersten Laufs: **sieben neue botanische Artenfunktionen**
`Pr(kartiertes natives Areal | TANN, PANN)` mit ausführbarem Vorwärtsevaluator,
gespeicherten Koeffizienten und räumlicher Prüfung. Das ist noch kein
vollständiges 279-Taxa-Modell und keine fossile oder Multiproxy-Rekonstruktion.

Die sieben verfügbaren Vertreter wichtiger Baumgattungen werden vor dem
Training festgelegt: Abies alba, Betula pendula, Corylus avellana, Fagus
sylvatica, Picea abies, Taxus baccata, Tilia cordata. Kein Ersatz eines
Gattungspollentyps durch eine einzelne Art, keine automatische Pilotfreigabe.
Originalnamen und beide BfN-Kennungen bleiben erhalten, insbesondere
`Fagus silvatica` in der Quelle sowie 10719/10718 bei Picea abies.

Originale Eingänge: öffentlich digitalisierte Meusel-Areale aus dem von
FloraWeb verlinkten BfN-Repository, Commit 5ab8e638278c86035bf3c9c3224c90de4880327e;
WorldClim 2.1 (1970–2000), BIO1/BIO12, native 10-Bogenminuten-TIFFs.
Das ist eine **neue Kalibrierung**, mit abweichender Klimatologie von der
Dissertation (CRU 4.01, 1961–1990). Fehlende ursprüngliche 279 Raster/Funktionen
werden nicht als wiedergefunden ausgegeben. Quelldateien nicht verändern.

Die mathematischen und räumlichen Festlegungen stehen vollständig in
[config.json](./config.json). Quadratische logistische Regression folgt der
Modellklasse aus Dissertation 2.5, Gl. 2.31/2.32. Neue bewusste Entscheidungen:
trainingsinterne Standardisierung statt globaler Quantilnormalisierung,
feste Ridge-Stabilisierung, Flächengewichtung statt SMOTE/Klassenumschichtung,
räumliche Folds mit Puffer statt zufälliger Trennung. Es gibt keinen
nachträglichen Modellwettbewerb, kein Tuning an Testwerten und keine Auswahl
der besten Blockgröße. Alle sieben Arten und beide Designs werden berichtet.

Eurasisches Arbeitsfenster: 20°–80° N, 20° W–180° E; gültige WorldClim-
Landpixel an Zentren eines 0,5°-Rasters. Das Fenster enthält die europäischen
Areale und einen breiten Klima-/Außenbereich, ist aber keine globale Aussage.
Außerhalb eines kartierten Areals ist das Label 0 ein **Kartenindikator**,
kein flächendeckender botanischer Abwesenheitsnachweis. Die Klimafunktion
beschreibt diesen Kartenindikator unter dem festgelegten Gebiet und dessen
Prävalenz. Sie berücksichtigt weder Pollenproduktion noch Ferntransport,
lokale Standorteignung, Migration oder glaziale CO2-Effekte.

Nur bestätigte native Vorkommen liefern positive Labels. Polygonzentren und
Punktzellen werden getrennt erfasst. Fragliche, erloschene und synanthrope
Angaben werden samt Umfeld maskiert. Zusätzlich werden Grenzstreifen und die
Umgebung isolierter Punkte maskiert (die bestätigte Punktzelle bleibt nutzbar,
soweit sie nicht aus anderen Gründen ausgeschlossen ist). Die Winkelabstände
in config.json sind feste konservative Arbeitsmasken, keine empirisch
geschätzten Fehlerverteilungen oder überall gleich großen Kilometerpuffer.
Keine automatische Geometriereparatur. Fehlen genügend Trainingszellen einer
Klasse oder konvergiert ein Fit nicht, wird der Fall gesperrt statt die Regeln
nachträglich zu lockern.

Hauptprüfung: flächengewichteter Brier-Score je Art gegen die Prävalenz im
selben Trainingsfold; zusätzlich Log-Loss, Reliabilität und Anteil der
Testklimata außerhalb der Trainingshülle. Fünf Folds je 5°/10°-Blockgitter,
100-km-Trainingspuffer. Skalierung und Prävalenz ausschließlich im jeweiligen
Training berechnen. Beim Test bleibt jede gültige Testzelle in den Hauptscores,
auch bei Extrapolation; der spätere Vorwärtsevaluator verweigert Klima außerhalb
der vollständigen Trainingshülle. Karten-CV ist kein unabhängiger Feld- oder
Fossiltest. Gute Scores allein sind keine fachliche Freigabe.

Vorgesehene Prüfung: vollständige bytegleiche Wiederholung, separater Prüfer
ohne Produzentenimport für Rasterlabels, Foldrollen/Puffer, Modellwahrscheinlichkeiten,
Scores und Gradienten; gezielte synthetische Tests für Geometrie, Fehlwerte,
Skalierung, Ableitungen und Extrapolationssperre. Gemeinsame GIS-/Numerik-
Bibliotheken werden als Abhängigkeit benannt. Große Raster, Klimazellen und
Laufdateien werden reproduziert; in den PR kommen Modellparameter, kleine
Ergebnistabellen, Quellenregister, Manifest und Prüfbericht.
