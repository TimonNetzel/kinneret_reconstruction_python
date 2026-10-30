# Bedingte räumliche Sensitivität der Wahrscheinlichkeitskalibrierung

Stand 30.10.2026. [PR #32](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/32),
abhängig von dem weiterhin offenen [#31](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/31),
gesperrte Basis `9458011198c69368764b8698c966c0ac23cc381c`.

**Ergebnis: Bedingte Sensitivitätsbänder sind für 22 von 28 Vollgebiets-Ensembles
und 104 von 140 äußeren Ensemble-Folds schätzbar.** Für Abies in beiden Designs
und Fagus im 10°-Design werden keine Vollgebiets-Bänder ausgegeben. Es wäre
falsch, die fehlenden Bänder durch Nullbreite zu ersetzen.

Dies quantifiziert nur die Variation der Kalibrierungsstufe bei festen Netzen,
festen Karten und fester Aufteilung. **Keine nominell validierten 95%-
Konfidenzintervalle, keine vollständige Kalibrierungs-/Rekonstruktionsunsicherheit
und keine Freigabe für fossile Inversion.** Weitere Taxa und belegte Pollentyp-
Zuordnung bleiben offen. Die äußeren Regionen sind bereits aus #28–#31 bekannt.

## Vorab festgelegtes Verfahren

[DESIGN.md](./DESIGN.md), [config.json](./config.json) und
[SOURCES.md](./SOURCES.md) wurden vor jeder Bootstrap-Kalibrierung in
`5a4a96f1d7c01b64a230fb8ad6a7fbaf84857697` veröffentlicht und bleiben unverändert.
Die Literatur motiviert Blockresampling, liefert keinen Deckungsnachweis für
unsere maskierten Karten und festen Blockgrößen; genaue Lesegrenzen sind belegt.

Pro Art×Block×äußerem Fold bzw. Vollgebiet 200 Replikate. Ganze beobachtete
geographische Trainingsblöcke werden mit Zurücklegen gezogen. Dieselben
Multiplizitäten gelten für beide Netzvarianten und die drei Mitglieder eines
Ensembles. Nur innere Original-Kalibrierungszellen werden mit cos(Breite) ×
Multiplizität neu gewichtet. Äußerer Test und Puffer bleiben ausgeschlossen;
Netze, Normierung und SMOTE aus #31 werden **nicht** neu trainiert oder angepasst.
Keine äußeren Labels für Bänder, Verfügbarkeitsauswahl oder Breitenbericht.

Dieselbe monotone Kalibrierung a≥0, unbestrafter Intercept und Ridge 0,0001 wie
#31. Mindestens fünf verschiedene aktive Originalzellen jeder Klasse pro
Mitglied. Künstliche Duplikation ersetzt diese Unterstützung nicht. Fehlt
bereits eine ursprüngliche Mitgliedskalibrierung, wird das Ensemble gesperrt.
Ein Band benötigt mindestens 190/200 Replikate mit drei schätzbaren Mitgliedern.
Die Schwelle bleibt auch bei 189/200 unverändert; keine nachträgliche Ersatzregel.

Die Quantile 0,025/0,5/0,975 sind punktweise und **bedingt auf schätzbare
Replikate sowie feste Netze**; keine simultane Flächendeckung. Auch unter dem
190er-Gate fehlen Informationen über ungültige Ziehungen. Die Verfügbarkeitsregel
ist kein Deckungsnachweis. Zwei Blockgrößen sind keine unabhängigen Datensätze
und werden nicht anhand schmalerer Bänder ausgewählt.

## Verfügbarkeit einschließlich aller Ausfälle

Original und SMOTE haben identische Schätzbarkeitszahlen; alle Ausfälle entstehen
hier durch Klassenunterstützung, nicht durch numerisches Optimierungsversagen.
Insgesamt 126/168 Ensemble-Fälle mit Band, 40 wegen zu weniger schätzbarer
Replikate gesperrt und zwei wegen einer bereits fehlenden Grundkalibrierung.
Letztere betreffen den bekannten Abies-10°-outer-3-Fold aus #31. Alle Ausfälle
sind in [summary.csv](./summary.csv) enthalten.

Je Art eine Familie gezählt; die Zahlen gelten identisch für beide Netzvarianten.
Vollgebiet = schätzbare Replikate, äußere Folds = Anzahl mit verfügbarem Band.

| Art | Vollgebiet 5° | Äußere Folds 5° | Vollgebiet 10° | Äußere Folds 10° |
| --- | ---: | ---: | ---: | ---: |
| Abies alba | 161/200 | 0/5 | 91/200 | 0/5 |
| Betula pendula | 200/200 | 5/5 | 200/200 | 5/5 |
| Corylus avellana | 200/200 | 5/5 | 199/200 | 5/5 |
| Fagus sylvatica | 200/200 | 5/5 | 164/200 | 0/5 |
| Picea abies | 200/200 | 5/5 | 195/200 | 3/5 |
| Taxus baccata | 200/200 | 5/5 | 200/200 | 4/5 |
| Tilia cordata | 200/200 | 5/5 | 198/200 | 5/5 |

Abies zeigt die wesentliche Grenze: Obwohl #31 einige Kalibrierungen numerisch
fitten konnte, tragen zu wenige räumliche positive Blöcke die Neugewichtungen.
Ausfallraten von 39/200 bzw. 109/200 im Vollgebiet sind keine ausreichende
Grundlage für das vorab geforderte Band. Fagus 10° hat 164/200 gültige Replikate.
Positive Blockzahlen pro Mitglied stehen vollständig in [inventory.json](./inventory.json).
Das ist ein Daten-/Schätzbarkeitsbefund, kein Anlass zur nachträglichen Änderung
der Blockgröße, des Gates oder zur Entfernung dieser Arten aus dem Bericht.

## Breiten im Vollgebiets-Klimagitter

Die folgenden Bänder beziehen sich auf die **kalibrierten** Ausgänge beider
Netzfamilien. Für rohe, hier unveränderte Netzprognosen wird kein Band berechnet;
das bedeutet nicht, dass deren gesamte Unsicherheit null wäre.

Flächengewichtete mittlere Bandbreite; in Klammern das **ungewichtete**
95%-Quantil der Zellbreiten. Einheit Wahrscheinlichkeit auf Skala 0–1,
keine °C-/mm-Fehler. Alle ursprünglichen Zellen einschließlich Extrapolationsflags
bleiben in diesen Beschreibungen. Breiten werden nicht als Gütescore verglichen.
Ein kleiner Domänenmittelwert kann große lokale Breiten verdecken.

| Art | Blöcke | Ohne SMOTE: Mittel (q95) | SMOTE: Mittel (q95) |
| --- | ---: | ---: | ---: |
| Abies alba | 5° | nicht schätzbar | nicht schätzbar |
| Abies alba | 10° | nicht schätzbar | nicht schätzbar |
| Betula pendula | 5° | 0.0670 (0.1549) | 0.0692 (0.1574) |
| Betula pendula | 10° | 0.1212 (0.2881) | 0.1235 (0.2907) |
| Corylus avellana | 5° | 0.0347 (0.1948) | 0.0350 (0.1900) |
| Corylus avellana | 10° | 0.0640 (0.3412) | 0.0650 (0.3397) |
| Fagus sylvatica | 5° | 0.0201 (0.1549) | 0.0212 (0.1587) |
| Fagus sylvatica | 10° | nicht schätzbar | nicht schätzbar |
| Picea abies | 5° | 0.0318 (0.2193) | 0.0315 (0.2114) |
| Picea abies | 10° | 0.0567 (0.4039) | 0.0588 (0.3897) |
| Taxus baccata | 5° | 0.0378 (0.2351) | 0.0380 (0.2274) |
| Taxus baccata | 10° | 0.0483 (0.2871) | 0.0504 (0.2811) |
| Tilia cordata | 5° | 0.0424 (0.2021) | 0.0433 (0.1988) |
| Tilia cordata | 10° | 0.0781 (0.3581) | 0.0798 (0.3618) |

Beispiel Betula: Das mittlere Band ohne SMOTE ist bei 5° etwa 0,067 breit,
bei 10° etwa 0,121; im oberen Zellbreitenquantil etwa 0,155/0,288. Die räumliche
Skala wirkt erheblich. Daraus wird weder eine optimale Blockgröße noch eine
allgemeine Überlegenheit einer Netzvariante abgeleitet. Die #31-Punktprognosen
und sein Brier-/BA-Ergebnis bleiben unverändert; kein neuer Testsieger.

## Separater Audit und vollständige Reproduktion

[verification.json](./verification.json) bestätigt:

- 198.680 Art-Zell-Labels separat aus gesperrten Rohquellen rekonstruiert;
  2.174.335 nichtnullige Blockmultiplizitäten auf Trainingszugehörigkeit,
  RNG-Seeds und vollständige Ziehungszahlen geprüft.
- Alle 100.800 Mitglied×Replikat-Fälle bestätigt: 95.876 geschätzte
  Kalibrierungen, 3.724 Fälle mit zu wenig verschiedenen aktiven Klassenzellen
  und 1.200 gesperrte Plätze des Ensembles ohne vollständige Grundkalibrierung.
  Kein numerischer Optimierungsausfall. KKT-Reste aller Fits unter 3,00·10⁻⁸
  bei vorab festem Gate 1e-7.
- Alle 1.589.440 Zellzeilen (äußere und Vollgebiets-Anwendung, zwei Familien)
  einschließlich Punktprognosen, drei Quantilen oder expliziter Nichtschätzbarkeit,
  Supportflags und sämtlichen 168 Zusammenfassungen separat geprüft.
- Kleinster innerer/äußerer Haversine-Abstand jeweils 100,093 km. Keine äußeren
  Labels für die Konstruktion; keine nominelle Deckung behauptet.

Der Prüfer importiert weder neuen Produzenten noch Bootstrap-Fitter. Separate
Vektorlabels, TIFF-Indizes, Haversine-BallTree, Blockzählung mit Counter,
tanh-Netzrechnung, konvexe Zielfunktion/KKT und sortierte interpolierte Quantile.
Gemeinsame alte #28/#29/#31-Auditwerkzeuge und NumPy/SciPy/GDAL/GEOS sind
benannt. Dies ist numerische Rechenprüfung, keine unabhängige Bestätigung der
ökologischen Kartenwahrheit und kein zweiter menschlicher Reviewer.

[repeat.json](./repeat.json) bestätigt **zwei vollständige frische Endläufe mit
sieben bytegleichen Enddateien**, einschließlich vollständiger Gzip-Prüfung.
Keine Wiederherstellung der Endausgaben erforderlich. Ein früherer Versuch mit
unvollständigen komprimierten Dateien wurde verworfen. Der Audit fand außerdem
eine numerisch falsche Außenzuordnung eines exakten gespeicherten Klimahüllen-
eckpunkts durch Qhull/Delaunay. Der neue Evaluator behandelt solche exakten
Eckpunkte definitionsgemäß als innen, ohne Toleranzvergrößerung oder Hüllenänderung.
Ein Regressionstest sichert dies. Kalibrierungsparameter und Vollgebietsparameter
sind vor/nach dieser technischen Korrektur bytegleich. Die bisherigen #31-
Quelldateien bleiben unverändert. Details: [execution_notes.json](./execution_notes.json).

[parameters.csv.gz](./parameters.csv.gz) bewahrt sämtliche 100.800 Parameter-
und Statusdatensätze, [full_parameters.json](./full_parameters.json) die unmittelbar
nutzbaren Vollgebiets-Replikate. [run_manifest.json](./run_manifest.json) sperrt
alle Eingänge, Programme und Ausgaben. Die 56,83-MB-Zelltabelle und 6,21-MB-
Blockspur werden anhand dieser Hashes reproduziert, nicht neu verteilt.
[REPRODUCE.md](./REPRODUCE.md) enthält vollständige Befehle ohne alte temporäre
Dateien, Dateischemata und den Vorwärtsevaluator. [validation.json](./validation.json):
85 Repositorytests und sechs gezielte Forschungstests bestanden; zwei Browser-
und zwei Langtests nicht ausgeführt. Schlanke App-/CI-Umgebung ohne neue GIS-Pflicht.

## Bedeutung für die Multiproxy-App und nächste Arbeit

Die Kalibrierungsstufe erhält eine nachvollziehbare bedingte Sensitivitätsprüfung
mit expliziten Informationslücken. Vollständige Kalibrierungsunsicherheit bleibt
offen: Netz-/Normierungs-/SMOTE-Schätzung, Karten- und Labelunsicherheit werden
nicht variiert; Kreuztaxon-Kovarianz ist nicht modelliert. Keine nominelle
Deckungsvalidierung oder binäre Label-Deckungsrate und keine unabhängige neue
Region. Vorwärtsevaluation verweigert standardmäßig Extrapolation außerhalb
irgendeiner Mitgliedshülle sowie ein Band unterhalb des Verfügbarkeitsgates.

Als Nächstes belegte Pollentyp–Art-/Gattungszuordnung und weitere Taxa/Areale
bearbeiten; vollständige Unsicherheit und gemeinsame Beobachtungsmodelle bleiben
notwendig. Keine fossile Rekonstruktion oder automatische Varianten-/Pilotfreigabe.
Claude betreut CLI/App; Analog-Baseline und bestehende Integration unverändert.
Monticchio-P1, physische LGM90D→MON16-Zuordnung samt Korrelationsunsicherheit
und 28 Kandidaten / 41 Nichttreffer / 47 Herkunftsfragen unverändert offen.
Keine Quellenkorrektur, Autorennachricht oder Merge.
