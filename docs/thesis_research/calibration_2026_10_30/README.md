# Räumlich verschachtelte Kalibrierung der sieben Artennetze

Stand 30.10.2026. [PR #31](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/31),
Basis main `7f1826c` nach dem Merge #30. **Ergebnis: Die festgelegte Kalibrierung
reduziert den großen Wahrscheinlichkeitsfehler des SMOTE-Ensembles deutlich,
liefert aber gegenüber dem rohen Ensemble ohne SMOTE keine Überlegenheit.**
Keine automatische Übernahme in die taxonbasierte Multiproxy-App.

[DESIGN.md](./DESIGN.md) und [config.json](./config.json) wurden vor dem ersten
Fit in `afa0c8a8845a5f40daee4289cc429fdf4a6ecf6a` veröffentlicht und danach
unverändert gelassen. Keine Hyperparameter-, Kalibrator-, Schwellen- oder
Taxonauswahl nach äußeren #29- oder neuen Testergebnissen.

## Ergebnis auf den äußeren Original-Testzellen

Je fünf äußere Folds, 5°-/10°-Blöcke, 100-km-Puffer. Pro äußerem Fold drei
innere räumliche Netz-/Kalibrator-Paare mit weiterem 100-km-Puffer. Vier
Varianten auf identischen äußeren Zellen. Roh/kalibriert verwenden jeweils
**dieselben drei Netzfits**; keine Volltrainings-Neuanpassung unter einem
Kalibrator kleinerer Netze. Die Werte sind ungewichtete Mittel über sieben
Arten, je Art Brier/Log-Loss flächengewichtet. BA bei fester Schwelle 0,5.
Angaben jeweils **5° / 10°**, keine Unsicherheitsintervalle.

| Variante | Brier ↓ | Balanced Accuracy ↑ | Log-Loss ↓ |
| --- | ---: | ---: | ---: |
| Ohne SMOTE, roh | 0.03939 / 0.04321 | 70.71 / 68.97 % | 0.12356 / 0.13510 |
| Ohne SMOTE, kalibriert | 0.03953 / 0.04349 | 69.41 / 67.16 % | 0.12426 / 0.13588 |
| SMOTE, roh | 0.07129 / 0.07699 | 90.55 / 89.35 % | 0.21845 / 0.23814 |
| SMOTE, kalibriert | 0.03995 / 0.04408 | 67.75 / 65.39 % | 0.12547 / 0.13685 |

- SMOTE-Kalibrierung verbessert den Brier gegenüber demselben rohen SMOTE-
  Ensemble in **14 von 14** Art×Block-Fällen. Artenmittel 0,07129→0,03995
  bzw. 0,07699→0,04408; die BA fällt von 90,55/89,35 auf 67,75/65,39 %.
- Ohne SMOTE verbessert Kalibrierung den Brier nur in **3 von 14** Fällen;
  im Artenmittel verschlechtert er sich leicht. Alle Verschlechterungen bleiben
  im Bericht. Keine nachträgliche Auswahl der rohen Variante pro Art/Fold.
- Kalibriertes SMOTE schlägt das rohe Ensemble ohne SMOTE im Brier nur in
  **1 von 14** Fällen (Fagus, 10°). Seine Artenmittel bleiben schlechter.
  Das ist kein Nachweis einer allgemeinen Überlegenheit einer Modellklasse.
- Die frühere #29-Netzbewertung bleibt unverändert. Hier werden räumliche
  Dreierensembles statt der damaligen einzelnen äußeren Netzfits verglichen;
  Unterschiede zu #29 können nicht allein der Kalibrierung zugeschrieben werden.
  Der direkte Kalibrierungseffekt steht in [paired_deltas.csv](./paired_deltas.csv).

### Vollständiger Brier-Vergleich

Je Zelle 5° / 10°. Alle übrigen Größen in [metrics_summary.csv](./metrics_summary.csv),
alle Foldwerte einschließlich undefinierter Maße in [scores.csv](./scores.csv).

| Art | Ohne SMOTE roh | Ohne SMOTE kalibriert | SMOTE roh | SMOTE kalibriert |
| --- | ---: | ---: | ---: | ---: |
| Abies alba | 0.00620 / 0.00634 | 0.00628 / 0.00665 | 0.04844 / 0.05489 | 0.00626 / 0.00791 |
| Betula pendula | 0.09415 / 0.10230 | 0.09456 / 0.10298 | 0.11061 / 0.11719 | 0.09557 / 0.10343 |
| Corylus avellana | 0.03874 / 0.04239 | 0.03896 / 0.04301 | 0.06028 / 0.06033 | 0.03958 / 0.04346 |
| Fagus sylvatica | 0.01748 / 0.01943 | 0.01760 / 0.01927 | 0.04919 / 0.05524 | 0.01757 / 0.01940 |
| Picea abies | 0.03417 / 0.03649 | 0.03437 / 0.03716 | 0.06657 / 0.07136 | 0.03510 / 0.03769 |
| Taxus baccata | 0.03861 / 0.04299 | 0.03857 / 0.04234 | 0.09080 / 0.10324 | 0.03904 / 0.04333 |
| Tilia cordata | 0.04640 / 0.05251 | 0.04640 / 0.05303 | 0.07316 / 0.07667 | 0.04656 / 0.05334 |

## Verfahren und Klassenknappheit

Meusel/BfN-Areale und WorldClim 2.1 / 1970–2000, dieselben Masken und
0,5°-Zellen wie #28. Unverändertes #29-Netzrezept: zwei Klimavariablen,
sechs sigmoidale Hidden-Neuronen, Decay 0,1 und drei ausschließlich nach
stationärem Trainingsziel gewählte Starts. T-Standardisierung und
sqrt(P)→Gamma-CDF→Probit nur im jeweiligen ursprünglichen inneren Netztraining;
SMOTE k=5 bis 1:1 ebenfalls ausschließlich daraus. Die synthetischen Fälle
sind keine neu beobachteten Flächen oder Orte.

Jedes Mitglied wird auf einem anderen inneren Originalfold kalibriert:
`sigmoid(a*Netzlogit+b)`, a≥0, flächengewichtete mittlere Bernoulli-NLL plus
0,0001·a²/2. Keine künstliche Klassenbalance der Kalibrierungszellen.
Der äußere Test einschließlich seines Puffers bleibt aus allen diesen
Schritten ausgeschlossen. Testlabels gehen ausschließlich in die Bewertung ein.

Alle 252 inneren Aufteilungen bestehen das Trainingsgate, kleinste ursprüngliche
Klasse 32. **Ein Kalibrierungsfold hat nur zwei positive und 6.460 negative
Zellen**: Abies alba, 10°, äußerer Fold 3, innerer Fold 2. Für beide Netzvarianten
greift deshalb die vorab bestimmte Identität a=1, b=0 statt einer Kalibrierung.
Die beiden Fälle sind markiert und unverändert in den Scores enthalten;
keine Splitänderung, Fallentfernung oder nachträgliche Ersatzmethode.
502 Kalibratoren sind numerisch geschätzt, zwei explizite Identitäten.

## Unabhängige numerische Prüfung und Speicherung

[verification.json](./verification.json) bestätigt den kompletten Audit:

- 198.680 Art-Zell-Labels aus gesperrten Rohquellen separat rekonstruiert;
  397.360 Testzeilen mit 1.589.440 äußeren Wahrscheinlichkeiten nachgerechnet.
- Alle 2.279.093 synthetischen Fälle auf beide ursprünglichen Minderheits-
  Trainingseltern, Nachbarschaft, Interpolation und Fallgewichte geprüft.
- 504 Netzfits / 1.512 Optimierungsstarts, Normierungen, Klimahüllen,
  Trainingsauswahl und sämtliche Kalibratorlösungen geprüft. Konvexes
  Kalibrierungsziel mit KKT-Prüfung; maximaler projizierter Gradient unter 4,08·10⁻⁹.
- Kleinster innerer und äußerer Haversine-Abstand jeweils 100,093 km.
  336 Scorezeilen, 560 Reliabilitätszeilen und 168 gepaarte Deltazeilen bestätigt.
- Der neue Produzent/Kalibrator wird vom Prüfer nicht importiert. Vektorprädikate,
  direkte TIFF-Indizes, Haversine-/Euclidean-BallTree, scipy.stats-Transformation
  und tanh-Netzrechnung bilden separate Rechenwege. Die hashgesperrte alte
  #28-Vektorprüfung und #29-Auditarithmetik sowie NumPy/SciPy/GDAL/GEOS werden
  ausdrücklich gemeinsam verwendet. Kein zweiter menschlicher Reviewer und
  keine unabhängige Bestätigung jeder botanischen Kartengrenze.

[models.json](./models.json) enthält 28 Vollgebiets-Ensembles (sieben Arten ×
zwei Blockdesigns × zwei Netzvarianten) mit insgesamt 84 Netz-/Kalibrator-Paaren.
[fit_models.json](./fit_models.json) enthält zusätzlich sämtliche äußeren Fits
und Starts. Der Vorwärtsevaluator verweigert standardmäßig Extrapolation aus
irgendeiner Mitgliedshülle. Die äußere Bewertung behält dagegen alle Zellen
samt `support_members` (0–3); kein scoreverbessernder Supportfilter.
[reliability.csv](./reliability.csv) enthält die zehn festen Bins.

[REPRODUCE.md](./REPRODUCE.md) liefert die vollständigen Befehle ohne alte
Chat-Zwischendateien. [run_manifest.json](./run_manifest.json) sperrt Eingänge,
Code und sämtliche Ausgaben einschließlich großer reproduzierbarer Vorhersage-
und Elternprotokolle, die nicht neu verteilt werden. Ein erster Fitlauf wurde
wegen einer unvollständigen Gzip-Elternspur verworfen; keine Änderung an
Wahrscheinlichkeiten oder Einstellungen. [execution_notes.json](./execution_notes.json)
dokumentiert dies ausdrücklich. Im Ersatzlauf wurde der letzte Checkpoint-Eintrag
aus dem vollständigen Modell desselben Laufs wiederhergestellt, exakt gegen
den bereits vorhandenen Manifest-Hash geprüft
([checkpoint_recovery.json](./checkpoint_recovery.json)). Danach sind
**alle elf Enddateien zweier vollständiger Trainingsausführungen bytegleich**,
einschließlich der 53,23-MB-Prognosen und 32,29-MB-Elternspur
([repeat.json](./repeat.json)); keine Behauptung zweier störungsfreier I/O-Läufe.
[validation.json](./validation.json) hält
79 erfolgreiche Repositorytests, fünf neue Forschungstests und die nicht
ausgeführten Browser-/Langtests fest.

## Aussagegrenzen und nächster Schritt

Brier bewertet die gesamte Wahrscheinlichkeitsgüte, nicht ausschließlich
Kalibrierung. Reliabilitätsbins sind deskriptiv. Eine NLL-Kalibrierung muss
weder Brier noch BA verbessern. Kein nachträgliches Verfahrenstuning aufgrund
dieser äußeren Ergebnisse und keine automatische Variantenübernahme.

**Die Regionen wurden bereits in #28/#29 untersucht.** Die äußeren Labels
sind vom jeweiligen Fit ausgeschlossen, aber dies ist eine methodische
Folgeauswertung auf bekannten Kartenquellen, keine vollständig ungesehene
Bestätigung. Beide Blockdesigns sind keine unabhängigen Replikationsdatensätze.
Kartiertes Außerhalb ist kein erhobener Abwesenheitsdatensatz; modelliert wird
der deklarierte native Arealindikator, keine validierte fossile Pollenwahrscheinlichkeit.

Weitere Taxa, belegte Pollentyp–Art-/Gattungszuordnung, Kalibrierungs-/Quellen-
unsicherheit und gemeinsame Beobachtungsmodelle bleiben offen. Die drei
räumlichen Mitglieder sind keine Unsicherheitsintervalle. Vor fossiler
Integration sind diese Grenzen zu bearbeiten; keine neue Klimakurve oder
Pilotfreigabe. Claude betreut CLI/App; Analog-Baseline und bestehende
Integrationsdateien bleiben unverändert.

Monticchio-P1 bleibt ungelöst: physische LGM90D→MON16-Zuordnung einschließlich
Korrelationsunsicherheit fehlt. 28 Kandidaten / 41 native Nichttreffer /
47 Herkunftsfragen unverändert. Keine stillen Quellenkorrekturen,
Autorennachrichten oder Merges. Aktuelle Übergabe: [HANDOFF](../HANDOFF.md).
