# Monticchio: Quellenzuordnung und Datierungsanker

Stand 29.10.2026. Folgeforschung zu [PR #20](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/20).
Da #18 weiterhin offen ist, basiert dieser eigenständige PR direkt auf dessen
Commit `f9acc16c58a47a51b1063690a69a2db01381c76a`. Die Ergebnisse von #20
werden nicht ersetzt. Claudes #19 hat weiterhin den unveränderten geprüften
Head `c8977a5`; eine erneute Codeprüfung ohne Änderungen wurde nicht wiederholt.

**Ergebnis:** Für 66 der 69 Pilotproben gibt es jetzt eindeutige numerische
Entsprechungen zur älteren publizierten Pollenzusammenfassung. Das stärkt die
Quellenzuordnung, liefert aber weder Originalprobenetiketten noch einen
Tiefen-Crosswalk zur neuen MON16-Chronologie. Zwei scheinbare LGM90D-AMS-Anker
sind nach dem Originalartikel aus dem Fen-Kern von 1993 übertragen. Die neue
Tephrochronologie von 2026 ist fachlich relevant und muss vor einer
Neudatierung berücksichtigt werden.

## 1. NOAA und PANGAEA: gleiche Altersdaten, verschiedene Präzision

Frisch geladen wurden die [NOAA/WDS-Originaldateien, Studie 5463](https://www.ncei.noaa.gov/access/paleo-search/study/5463),
DOI [10.25921/69tm-zg59](https://doi.org/10.25921/69tm-zg59),
mit `age.txt`, `pollen.txt` und `readme_allen1999.txt`. Der Readme nennt die
Kerne B, D, J und L; die Pollentabelle selbst nennt D 1990 und J 1994.
Diese Bezeichnungen sind kein zeilenweiser Komposit-/Proben-Crosswalk.

Alle **1.120** Alters-Tiefen-Paare in NOAA entsprechen als Multimenge den Daten
von [PANGAEA 707050](https://doi.org/10.1594/PANGAEA.707050), wenn die
vierstelligen NOAA-Tiefen auf drei Dezimalstellen mit **half-even** gerundet
werden. Die Paar-Multiplizitäten bleiben dabei erhalten. 46 Zeilen wechseln
ihre Position; NOAA enthält 16 Tiefenrücksprünge, PANGAEA ist nach Tiefe sortiert.
Vier mehrfach belegten NOAA-Tiefen stehen sechs in PANGAEA gegenüber;
beide enthalten zwei doppelte Tiefen-/Alterspaare. Die höhere Präzision
erklärt somit einen Teil scheinbarer Doppelbelegungen, nicht sämtliche.
Originale werden weder sortiert überschrieben noch dedupliziert.

**Einheitenfalle:** Der NOAA-Header von `age.txt` nennt Lamination years BP,
obwohl etwa der Wert `0.088` numerisch dem ausdrücklich mit ka beschrifteten
PANGAEA-Alter entspricht. `pollen.txt` enthält dagegen echte Jahreswerte,
z. B. `86.8`. Das Skript prüft beide Zuordnungen ausdrücklich; es wendet keine
pauschale Einheitenumrechnung auf Dateien desselben Archivs an.

Die 496 Zeilen der Pollenzusammenfassung in NOAA und
[PANGAEA 707075](https://doi.org/10.1594/PANGAEA.707075) haben sämtlich
dieselben Alter nach der dokumentierten ka/Jahre-Umrechnung. Für acht
Einzeltaxa sind die Prozentwerte innerhalb ihrer Darstellungspräzision
vereinbar. Die Gesamtsummen unterscheiden sich jedoch in sechs Zeilen:

| Zusammenfassungszeile | Alter, Jahre BP | NOAA-Pollensumme | PANGAEA-Pollensumme |
|---|---:|---:|---:|
| 55 | 15.075,2 | 417,5 | 418 |
| 57 | 15.655,0 | 388,5 | 388 |
| 67 | 19.370,2 | 373,5 | 374 |
| 78 | 21.424,3 | 361,5 | 362 |
| 115 | 24.158,2 | 403,5 | 404 |
| 124 | 26.760,2 | 340,5 | 340 |

Auch diese Unterschiede entsprechen half-even-Rundung. **Die Bedeutung der
halbzähligen Originalangaben ist ungeklärt**; sie werden nicht als neue
Rohzählungen übernommen. Quellenstand, Nachbearbeitung oder Mittelung lassen
sich aus dem Export allein nicht entscheiden. [denominator_changes.csv](./denominator_changes.csv)
erhält beide Tokens. Aus Prozentwerten werden keine Zählungen rekonstruiert.

## 2. 66 numerische Probenkandidaten, drei offene Fälle

Explorativer Abgleich mit Pinus, Betula, Quercus, Alnus, Abies, Fagus,
Artemisia und Chenopodiaceae: Für jede vorhandene ganzzahlige Rohzählung
wird der Prozentwert mit der jeweils ausdrücklich publizierten Gesamtsumme
**vorwärts** berechnet. Übereinstimmung verlangt bei allen acht Feldern
höchstens 0,005 Prozentpunkte Abweichung für PANGAEA bzw. 0,05 für NOAA.
Das sind Darstellungsgrenzen, keine an Alter oder gewünschten Treffern
optimierten Toleranzen. Die Merkmalswahl und Untersuchung sind explorativ,
keine vorregistrierte statistische Validierung.

Es werden weder Alter noch Zeilenposition zur Kandidatensuche verwendet.
Gattungs- und Untertypfelder werden nicht addiert. Gramineae bleibt aus der
Identitätssuche ausgeschlossen: Die breitere Zusammenfassung kann Cerealia
einschließen und entspricht nicht automatisch dem Shared37-Merkmal Poaceae.

**66/69 Proben** haben in beiden Zusammenfassungen denselben eindeutigen
Kandidaten; keiner wird mehrfach verwendet. Die Treffer bleiben in
stratigraphischer Reihenfolge. Die Originalalter unterscheiden sich vom
EPD-Cut-Zentralalter im Betrag im Median um **1,005 Jahre**, höchstens
**7,15 Jahre**. Das spricht für gemeinsame Chronologieinformation und ist
keine externe Bestätigung der Altersgenauigkeit. Die Quelle liefert
Zusammensetzung und Alter, aber keine Originalproben-ID oder Probentiefe.
Der Status bleibt deshalb `unique_pollen_fingerprint_candidate`.

| Offene Rohprobe | Pollentiefe | Cut-Tiefe | Status |
|---|---:|---:|---|
| LGM90D:000055 | 9,03 m | 9,030 m | kein vollständiger Acht-Feld-Treffer |
| LGM90D:000057 | 9,23 m | 9,230 m | kein vollständiger Acht-Feld-Treffer |
| LGM90D:000067 | 11,23 m | 11,235 m | kein vollständiger Acht-Feld-Treffer |

Die jeweils ähnlichste Zusammenfassungszeile liegt bei 55, 57 bzw. 67 und
gehört zu den halbzählig angegebenen Summen. Dennoch verschwinden die
Abweichungen auch mit dem ungerundeten NOAA-Nenner nicht vollständig.
Die Toleranz wird nicht nachträglich gelockert; diese drei Proben werden
nicht als identifiziert ausgegeben und ihre Rohwerte nicht korrigiert.
[sample_correspondence.csv](./sample_correspondence.csv) enthält alle 69
Entscheidungen; [unresolved_cells.csv](./unresolved_cells.csv) die verbleibenden
Feldabweichungen zur rein explorativ ähnlichsten Zeile.

## 3. Zwei AMS-Daten stammen aus dem Fen-Kern

Der Originalartikel von **Huntley, Allen & Watts (1996), Il Quaternario 9(2),
605–616**, ist über den [Zeitschriftenserver](https://amq.aiqua.it/index.php/amq/article/view/1059)
frei zugänglich. Die Portalangabe 2024 ist das heutige Online-Metadatum;
Heft und gedruckte Seiten weisen die Arbeit als 1996 aus. Die gescannte
Seite **609**, Abschnitt **2.2 The Fen Core (1993)**, wurde nach OCR am
Originalseitenbild geprüft.

Dort werden CAMS 12611, **CAMS 12610** und **CAMS 9855** als relevante
AMS-Daten des Fen-Kerns genannt. Die beiden letzten Labornummern und ihre
unkalibrierten Werte (10.460 ±60 bzw. 12.540 ±130 14C-Jahre BP) stimmen
exakt mit zwei Einträgen in [LGM90D 740512](https://doi.org/10.1594/PANGAEA.740512)
überein. Diese tragen dort bereits den Kommentar zur per Pollenkorrelation
bestimmten Tiefe, bei 7,25 bzw. 9,00 m. **Dies sind übertragene Anker,
keine unabhängigen Direktdatierungen des Materials dieser LGM90D-Tiefen.**
Der dritte archivierte Eintrag ETH-12464 bei 13,85 m bleibt hinsichtlich
seiner ursprünglichen Probenzuordnung hier nicht unabhängig bestätigt.

Für eine Neudatierung braucht es die Originaltiefen des Fen-Kerns von 1993, die dokumentierte
Korrelationsbeziehung einschließlich ihrer Unsicherheit und Materialprovenienz.
Einfaches Neukalibrieren mit IntCal20 würde diese Zuordnungsunsicherheit
nicht lösen. Es werden hier keine neu kalibrierten Alter oder pauschalen
Korrekturen publiziert. Der Artikel diskutiert zudem eine damalige Diskrepanz
zwischen Fen- und Laminationstiming; diese ist keine übertragbare konstante
Korrektur für alle 69 Proben.

## 4. Neue, relevante Chronologie von 2026

**Wulf et al. (2026), Quaternary Science Reviews 389, 110116**,
[doi:10.1016/j.quascirev.2026.110116](https://doi.org/10.1016/j.quascirev.2026.110116),
ist online seit 15.06.2026, im Heft vom 01.10.2026. Titel und Datum sind bei
Verlag und [Autoreninstitution](https://researchportal.port.ac.uk/en/publications/lago-grande-di-monticchio-revisited-revised-age-model-of-the-last/)
bestätigt. Der vollständige, von Mitautor Roberto Isaia öffentlich bereitgestellte
[Artikeltext](https://www.researchgate.net/publication/407540920_Lago_Grande_di_Monticchio_revisited_Revised_age_model_of_the_last_30_kyrs_based_on_refined_tephrostratigraphic_constraints_of_Campanian_volcanoes)
wurde in Abschnitten 3.1.2/5.2 und der Beschriftung von Tabelle 1 gelesen.

Das neue MON16-Komposit hat eine eigene, vor Bacon-Modellierung von Tephren
bereinigte Tiefenskala. Die Chronologie nutzt 16 veröffentlichte Marker-Alter:
ein historisches, acht kalibrierte und sieben bereits modellierte Alter;
terrestrisches und marines 14C werden getrennt behandelt. Der Bereich reicht
etwa von 1,35 bis 29,07 ka. Tabelle 1 vergleicht diese Ergebnisse mit einer
älteren Warvenchronologie mit 5-%-Fehlern. Dies rechtfertigt weder die
Übertragung der MON16-Tiefen auf LGM90D noch pauschales Verfünffachen der
EPD-Altersgrenzen. Die dort verwendeten Modell- und Randfehler haben
unterschiedliche Bedeutungen.

Die für eine Übernahme nötigen **Supplementtabellen A-1, A-2, A-3** sind in
diesem Arbeitsschritt **nicht heruntergeladen oder numerisch geprüft**.
Der institutionelle PDF-Download lieferte HTTP 403; Verfügbarkeitshinweise
des Verlags werden nicht als tatsächlicher Datendownload ausgegeben.
Es wurde kein posteriorer Altersensemble-Satz bezogen. Die neue Publikation
ist ein konkreter Ansatzpunkt, noch kein freigegebenes Ersatzmodell.

## Nächste konkrete Arbeit und Reproduktion

1. Supplement A-1/A-2/A-3 einschließlich Modellkonfiguration und möglichst
   gemeinsamer posteriorer Kurven beschaffen und gegen die Publikation prüfen.
2. LGM90D → ältere Kompositprofile → MON16 anhand dokumentierter Marker,
   Kernabschnitte, Tephrenbereinigung und Originalprobentiefen verbinden.
   Die 66 numerischen Kandidaten helfen dabei, ersetzen diese Kette nicht.
3. Die drei halbzählig abweichenden Zusammenfassungszeilen und die Herkunft
   der übertragenen Fen-Kern-Daten klären. Abhängige Anker nicht doppelt zählen.

Alle neun verwendeten Dateien sind in [sources.json](./sources.json) mit URL,
Größe und SHA256 gesperrt. Quelldateien gemäß Katalog in ein gemeinsames
Verzeichnis laden, die Originale unverändert lassen. Keine alten temporären
Dateien sind zur Reproduktion vorausgesetzt:

```bash
python -m unittest discover -s tools/thesis_import -p test_chronology_crosswalk.py -v
python tools/thesis_import/audit_chronology_crosswalk.py \
  --raw-dir /path/to/sources --output-dir /path/to/new_crosswalk
python tools/thesis_import/verify_chronology_crosswalk.py \
  --raw-dir /path/to/sources --run-dir /path/to/new_crosswalk
```

Python-Standardbibliothek genügt. Fünf Tests prüfen Rundung, Multiplizität,
Mehrdeutigkeit, fehlende Zählungen und Erhalt halbzähliger Nenner. Der separate
Auditor berechnet alle Kandidatensuchen mit exakten rationalen Brüchen neu.
Zwei vollständige Läufe sind einschließlich Manifest bytegleich.
Die 1.120-zeilige `age_archive_alignment.csv` wird reproduziert, nicht
eingecheckt. Die Quelldaten behalten ihre Katalogbedingungen; insbesondere
die LGM90D-Ableitungen bleiben CC-BY-NC-3.0. Die Originalartikel und
Originalzählungen werden nicht erneut verteilt.

Baseline, Importer, Shared37-Gruppen, die bestehenden Alterswerte und Claudes
Integrationsdateien bleiben unverändert. Keine Erweiterung der freigegebenen
69-Proben-Kohorte oder der zeitlichen Abdeckung.
