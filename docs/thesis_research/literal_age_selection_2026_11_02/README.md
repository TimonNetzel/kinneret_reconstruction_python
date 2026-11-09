# Historischer Teilnachbau: Altersraster und Zeilenauswahl

Stand 02.11.2026, Basis `6241f41408ed08ecc373d2dad1ace559a4effbb2`,
Branch `dissertation-reconstruction`, [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).

**Abgeschlossen:** Die begrenzte Rastermitgliedschaft und Zeilenauswahl aus
`mcmc_prepare.R` und `mcmc_functions_v1.6.R` ist im separaten Modul
[historical_age_selection.py](../../../tools/historical_age_selection.py)
nachgebildet. 39 neue synthetische Fälle bestehen, zusammen mit den bisherigen
Teilprüfungen **124 Tests grün**. Die rationale Produktkontrolle weicht um
0 von den Sollwerten ab. Das ist ein technischer Phase-1-Ausschnitt, keine
historische R-Laufzeitgleichheit oder vollständige Kinneret-Reproduktion.

## Quellenbindung und genauer Umfang

Vier bereits gebundene Skripte wurden ausschließlich gelesen; SHA-256 vor/nach
dem Lesen stimmt mit der bisherigen Bindung überein. Keine neue Suche,
Originaldatenrechnung oder Ausführung eines R-Skripts.
[source_binding.json](source_binding.json) enthält die Zeilenauszüge, Bytegrößen
und Hashes. Private Quellwurzeln werden aus `.local/my_reconst_paths.json`
gelesen; der Export enthält nur relative Quellbezeichner.

| Quelle | SHA-256 |
| --- | --- |
| `original:src/check_ages.R` | `c7f92e6030d83540c14f29a2e9d0ad07c74104dfe86befebda921c1cffbd2729` |
| `original:src/mcmc_prepare.R` | `b897e84f8f2d72399de729592066341585974d521bf8164b185f20377ed5cc7e` |
| `original:src/mcmc_functions_v1.6.R` | `b887144370e328fa016d1834ceebcd55622816d4b36838d33a26197110ab453c` |
| `original:src/mcmc_case_studies.R` | `37587ff08565bf284349e71a9ddbfd924d47fb786e0ccce5168a949be5b9c22e` |

Die ersten drei Hashbindungen stammen aus der
[Kinneret-Eingabebindung](../kinneret_binding_2026_11_02/summary.json) bzw. für
`mcmc_functions_v1.6.R` aus dem
[PDF-Teilnachbau](../literal_final_pdf_2026_11_02/source_binding.json).
Auch `mcmc_case_studies.R` ist in der früheren Eingabebindung verzeichnet.

| Funktion | Quellzeilen | Nachgebildeter Ausschnitt |
| --- | --- | --- |
| `period_grids` | `mcmc_prepare.R:435–437` | Drei feste Raster mit Schrittweite `diff(age)[1]` |
| `period_membership` | `mcmc_prepare.R:440–442` | Exakte Mitgliedschaft `age %in% grid` |
| `preserved_selection_trace` | `mcmc_prepare.R:1462–1467,1500–1522` | Vorläufige und bei Erreichen der Zeilen wirksame Maske getrennt; deaktivierten Zweig erhalten |
| `rw_selected_age_product` | `mcmc_functions_v1.6.R:461–462,470` | Bereits vorbereitete Koeffizienten anhand boolescher Zeilenmaske auswählen und mit TF multiplizieren |

Kein Runner, keine App-/CLI-Integration, keine Verbindung zu einem neuen
Fossilrekonstruktionslauf. Taxonmatching und TF-Vorbereitung werden nicht aus
Ersatzdaten abgeleitet; `taxa_merge` ist ein ausdrücklich übergebener,
synthetischer Kontrollzustand.

## `age`, `site.age` und Masken sind verschiedene Zustände

1. `check_ages.R:177–185` wählt eine Listeposition und weist zunächst denselben
   Altersvektor `site.age` und `age` zu. Die fehlende finale Listenchronologie
   wird damit weiterhin nicht durch die erhaltene Mittelwertmatrix ersetzt.
2. Zeitfenster in `check_ages.R:223` werden mit `site.age` berechnet. Erst danach
   erhöht 230–232 spätere exakte Duplikate in `age` einmal um 0,5; `site.age`
   bleibt dabei unverändert. Die [vorherige Teilprüfung](../literal_age_depth_2026_11_02/README.md)
   dokumentiert diesen Schritt und die rho(D|A)-Orientierung.
3. `mcmc_prepare.R:435–442` berechnet Raster und Masken ausdrücklich aus `age`.
   `site.age` ist dafür keine Eingabe. Schon die Duplikatverschiebung kann
   deshalb die Rastermitgliedschaft ändern.
4. Der aktive, bedingte Speläothemzweig überschreibt in 825–826 `site.age`
   mit einer 0–220500-Achse in 500er-Schritten und entfernt Positionen 6:29.
   Das überschreibt weder `age` noch die bereits berechneten Periodenmasken.
   Keine automatische Synchronisierung ist belegt. Dieser früher dokumentierte
   Testpfad wird nicht als finaler Kinneret-Lauf gewählt.
5. `final_reconstruction` nimmt `age` als formalen Parameter entgegen
   (366), aber **nicht** `reconst_ages`, `ages_lg` oder `taxa_change`.
   Diese drei Größen werden aus der Funktionsumgebung gelesen. Auch die
   erhaltenen Fallstudienaufrufe (306–317) übergeben sie nicht explizit.
   Gleiche aktuelle Quellhashes beweisen daher keine historische Workspacebindung.

Der Teilnachbau gleicht diese Zustände nicht durch Sortieren, Interpolation,
Neuberechnung, Achsenverkürzung oder stilles Überschreiben an.

## Rasterformeln und exakte Mitgliedschaft

Mit einsbasierten R-Positionen ist die Schrittweite

`h = age[2] − age[1]`.

Die erhaltenen Raster sind:

| Maske | Rasteranfang `b` | Rasterobergrenze `e` |
| --- | ---: | ---: |
| `ages_hol` | −50 | 11450 |
| `ages_lg` | 11500 | 115000 |
| `ages_eem` | 115500 | 125000 |

Im unterstützten Bereich gilt `grid = b + k*h` für ganze `k` von 0 bis
`floor((e-b)/h)`. Eine nicht erreichte Obergrenze wird nicht angehängt.
Die Maske lautet `M[i] = (age[i] ist exakt in grid enthalten)`.
Das ist weder ein Intervalltest `b <= age[i] <= e` noch ein Nächster-Nachbar-
Abgleich. Lücken, nicht passende Auflösungen und halbe Alterswerte können
innerhalb des Intervalls als FALSE markiert sein. Es gibt keine Toleranz.

Beispiel mit `h=50`: `age = [-50,0,0.5,11500,11550.5,115500]`.
Die Holozänmaske ist `[T,T,F,F,F,F]`, die Glazialmaske `[F,F,F,T,F,F]`,
die Eemmaske `[F,F,F,F,F,T]`. Auf einem unverschobenen `site.age`, das an
Position 3 noch 0 enthält, wäre auch diese Position Holozänmitglied.
Gleiche Vektorlänge stellt diese Differenz nicht richtig.

Nur die erste Differenz bestimmt das Raster. Die Quelle überprüft an diesen
Zeilen keine regelmäßige Abstandsfolge und sortiert die Alter nicht. Der
Nachbau ergänzt solche Prüfungen nicht. Ein erster Abstand 150 erreicht die
Holozänobergrenze 11450 beispielsweise nicht; letzter Rasterwert ist 11350.

Die Zahlen tragen die Einheit des eingesetzten Altersvektors. Für den bereits
gebundenen älteren Kinneret-Kandidaten ist sie **cal a BP**. Weder a→ka noch
eine Anpassung der historischen Grenzen erfolgt. Die synthetischen Zahlen
sind keine neuen historischen Alter. Die Booleschen Masken sind dimensionslos.

## Vorläufige Maske und deaktivierter Taxonwechsel

| Expliziter Kontrollzustand | Initiale `reconst_ages` (1463–1467) | Weiterer erhaltener Kontrollfluss | Maske bei Erreichen des Endzustands |
| --- | --- | --- | --- |
| `taxa.merge = FALSE` | Alle TRUE | Mergeblock wird übersprungen | Alle TRUE |
| `taxa.merge = TRUE` | `ages_hol` | Bedingung 1500 enthält `& FALSE`; deshalb `else` mit Reset in 1522 | Alle TRUE |

`taxa_change` wird in 1462 auf FALSE gesetzt. Die einzige TRUE-Zuweisung im
hier gelesenen Vorbereitungsblock liegt hinter dem deaktivierten Zweig 1500.
Der Quelltext dieses Zweigs bleibt als historische Variante erhalten, wird
aber nicht aktiviert oder zur finalen Methode erklärt. Der Reset setzt voraus,
dass die vorherigen Mergeoperationen erfolgreich durchlaufen und die Zeile
überhaupt erreicht wird. Die synthetische Kontrollflussprüfung beweist diesen
Ausführungserfolg nicht.

## Zeilenauswahl, Dimensionen und Grenzen zum Verbraucher

Sei `C` die bereits vorbereitete Koeffizientenmatrix **A × T** mit Alterszeilen
und geordneten Taxonspalten, `F` die numerische TF-Matrix **T × G** und `M` die
boolesche Maske mit genau **A** Elementen. Mit `K=sum(M)` lautet der isolierte
RW-Produktpfad:

`Y = C[M,] %*% F`, Dimension **(K × T) × (T × G) = K × G**.

Die Reihenfolge der TRUE-Zeilen bleibt erhalten. Die Ausgabe wird nicht nach
Alterswerten sortiert. Keine Normierung der TFs, Koeffizienten oder Produkte
wird in diesen Ausschnitt eingefügt. Bereits erfolgte taxonweise Skalierung
und Gewichtung gehören zum getrennten PDF-Teilnachbau. Das Produkt behält bei
dimensionslosen Koeffizienten die Einheit der eingesetzten TF-Zellen; die
synthetischen Zellen besitzen keine physikalische Klimaskala.

Wichtige Unterschiede im erhaltenen Verbraucher:

- Der Gibbs-Pfad 433–447 verwendet alle Zeilen von `pollen_info_age` und wendet
  `reconst_ages` dort nicht an. Eine identische Maskierung beider Zweige wäre
  eine unzulässige Vereinheitlichung.
- Der RW-Pfad 461–476 verwendet die Maske. Falls `taxa_change` TRUE wäre,
  würden zusätzlich `ages_lg`-Produkte angehängt (`rbind`), ohne an diesen
  Zeilen die gesamte Ausgabe nach `age` zu sortieren. Dieser deaktivierte
  Sonderpfad wird nicht nachgebildet oder still aktiviert.
- Speicherfelder (387/392) und nachfolgende Schleifen (487–490) beziehen sich
  weiter auf `length(age)`. Eine beliebige K-zeilige Teilmatrix ist deshalb
  noch kein konsistenter vollständiger Funktionsaufruf. Die synthetische
  vorläufige Maske testet nur das lokale Produkt, keine vollständige Auswertung.
- Der weitere MCMC-Verbraucher 1818 benutzt ebenfalls `reconst_ages`, 1830
  optional `ages_lg`. Der umgebende Code enthält stochastische Ziehungen und
  einen Übergangstest an Position 300; er wurde ausschließlich gelesen.

Unterstützt sind endliche Alter in ganzzahligen oder halben Einheiten innerhalb
±1e9 mit mindestens zwei Werten und positiver erster Differenz. Die Raster-
erzeugung nutzt dafür exakt darstellbare halbe Einheiten. Allgemeine R-`seq`-
Randtoleranzen und andere Dezimalauflösungen werden nicht emuliert.

Das Produkt akzeptiert endliche nichtnegative, nichtleere Matrizen, mindestens
zwei Taxa und zwei ausgewählte Zeilen sowie eine exakt passende Boolmaske.
NA/Inf, Ganzzahlindizes, R-Recycling, leere oder einzelne ausgewählte Zeilen
und Dimensionsverlustpfade sind ausdrücklich ausgeschlossen. Daraus folgt
keine Aussage, dass R jeden solchen Einzelfall ablehnt. Nicht unterstützte
Eingaben lösen `UnsupportedSourceDomain` aus; keine Imputation oder Reparatur.
Eingaben bleiben unverändert.

## Synthetische Nachweise und Wiederholung

[synthetic_fixture.json](synthetic_fixture.json) enthält sechs abstrakte
Alterszeilen, drei Taxa, vier Dichtezellen und rationale Sollwerte.
Die vorläufige Mergemaske erhält die R-Zeilen `[1,2]` und ergibt **2 × 4**;
der im erhaltenen Kontrollfluss wirksame All-TRUE-Zustand ergibt **6 × 4**.
Die TF-Zeilensummen 2, 1 und 4 sind absichtlich unterschiedlich, sodass eine
zusätzliche Normierung auffällt. Zwei der Alterswerte liegen innerhalb eines
Periodenintervalls, aber außerhalb seines exakten Rasters.

[Tests](../../../tests/test_historical_age_selection.py) prüfen Rasterlängen,
Randwerte bei passenden und unpassenden Schrittweiten, halbe Einheiten,
vorläufige/wirksame Masken, verstreute Zeilenauswahl, rationale Produkte,
fehlende Mutation sowie die ausdrücklich abgewiesenen Domänen.
[synthetic_result.json](synthetic_result.json) hält Masken, Produkte und Fehler
fest; beide rationalen Produktkontrollen sind in dieser Prüfung exakt gleich.
Keine Originalwerte wurden für die Numerik eingesetzt. Kein R-Laufzeitvergleich;
`float64`-Produkte können bei anderen Werten anders runden als R.

```powershell
.venv/Scripts/python.exe docs/thesis_research/literal_age_selection_2026_11_02/collect_evidence.py
.venv/Scripts/python.exe -m pytest tests/test_historical_age_selection.py tests/test_historical_age_depth.py tests/test_historical_final_pdf.py tests/test_kinneret_phase1_spec.py -q -o cache_dir=.local/phase1_spec_v1/pytest_cache
```

[collect_evidence.py](collect_evidence.py) liest gezielt vier zuvor gebundene
Skripte und erzeugt Quellauszüge und synthetische Rechennachweise. Für die
portablen Tests werden keine Originalpfade benötigt.
[verification.json](verification.json) dokumentiert die Prüfung und Erhaltung.

## Ergebnis, offene Voraussetzungen und nächster Einstieg

Der [Phase-1-Belegvertrag](../kinneret_phase1_spec_2026_11_02/README.md) bleibt
unverändert: elf offene Nachweisbündel. Finale Originalbasis, Listenchronologie,
numerische historische TFs und tatsächliche Laufzuordnung fehlen weiterhin.
Matrix und Listenraster bleiben getrennt; kein finaler Kinneret-Pfad gewählt,
kein lokaler Fall vollständig reproduziert. Weitere Speicherorte sind dem
Nutzer nicht bekannt; keine erneute Frage, pauschale Inventur oder Ersatzwahl.

Nächster unabhängiger technischer Einstieg: den gebundenen kumulativen
Dichte-/EMD-Ausschnitt in `src/mcmc_functions_v1.6.R:366–370,485–490`
gegen die erhaltenen Fallstudienaufrufe mit skalaren Referenzargumenten prüfen.
Normierungszeitpunkt, `dims`-Faktor, Referenzform und Einheiten getrennt
belegen und synthetisch testen; keine neue Referenzkurve oder Kalibrierung.
Noch keine Verbindung der Module zu einem historischen Vollrunner.

App/CLI, Baseline, Originale, Referenzen und fallbezogene Monticchio-/Chronologie-
Gates bleiben erhalten. Keine Trainings, Sampler, Altersmodelle oder
Fossilrekonstruktionen. Phase 1 vor Phase 2; Plots jung links, alt rechts.
Bestehender Branch/PR #77, kein automatischer Merge.
