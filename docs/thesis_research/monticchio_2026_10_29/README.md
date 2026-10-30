# Monticchio: Analogabdeckung und Altersunsicherheit

Stand 29.10.2026. Basis ist der offene PR #18, Commit
`f9acc16c58a47a51b1063690a69a2db01381c76a`.
Der [Diagnoseplan](./DESIGN.md) wurde vor dem Ergebnislauf im lokalen Commit
`442e750c3ab93e26128e32d23f411f7811791d5d` festgehalten; dessen unveränderter
Planstand ist nachträglich im GitHub-Commit
`146aa19113e49ffdd92ade8112d9146f9f1cc07b` veröffentlicht. Der lokale
Git-Push hatte keine Schreibanmeldung; die Veröffentlichung erfolgt über
die GitHub-Verbindung. Es wird keine externe Vorregistrierung behauptet.
Die vorhandene moderne Klimabaseline bleibt unverändert.

**Befund:** Alle 69 Fossilproben lassen sich technisch untersuchen, aber ihre
Analogabdeckung ist uneinheitlich. Altersgrenzen und Probenabstände erlauben
keine pauschale jährliche oder hundertjährliche Zeitauflösung. Eine fossile
Klimakurve oder Unsicherheitsfreigabe folgt daraus noch nicht.

## Analogabdeckung

Unveränderte Shared37-Zählungen einschließlich Pinus; moderne Ausgangskohorte
7.078 Proben nach der bestehenden Mindestzählung. Hellinger-Abstände wie in
der Baseline, ohne Klimazielwerte. d1/d5 bezeichnen den ersten/fünften Nachbarn.
Die ersten fünf modernen Proben, Koordinaten, Kontexte und Entfernungen stehen
in [fossil_analogues.csv](./fossil_analogues.csv); pro Fossilprobe die
[Diagnosewerte und Flags](./fossil_support.csv).

Eine wichtige Quellenabhängigkeit ist jetzt sichtbar: Die moderne Matrix enthält
`EPDcoretop_E932` (lake), also den Kernkopf derselben EPD-Entity wie der fossile
Pilot, und `EPDcoretop_E453` (wetland) an identischen Koordinaten.
Die Diagnose schließt beide aus. Damit werden Fossilproben gegen **7.076**
moderne Proben außerhalb derselben Koordinatengruppe verglichen. Das ist eine
Vorsichtsentscheidung für diese Diagnose, keine Änderung des Baseline-Modells.
Weitere mögliche Archivabhängigkeiten sind damit nicht vollständig geklärt.

Referenzverteilung: Jede der 7.078 modernen Proben wird gegen alle anderen
Koordinatengruppen verglichen. Die Quantile sind probengewichtet und werden
linear interpoliert. Eine Sensitivität schließt auf beiden Seiten zusätzlich
alle Referenzproben innerhalb von 100 km aus; für Monticchio bleiben 7.045.

| Diagnose | Referenz d1 P95 / P99 | Fossile d1 über P95 / P99 | Referenz d5 P95 / P99 | Fossile d5 über P95 / P99 |
|---|---:|---:|---:|---:|
| Andere Koordinatengruppe | 0,274526 / 0,351271 | 30 / 3 | 0,336407 / 0,426544 | 19 / 3 |
| Zusätzlich 100 km | 0,315959 / 0,397052 | 7 / 1 | 0,367251 / 0,457623 | 7 / 0 |

Fossiler d1-Median: 0,272166, Spannweite 0,181399–0,418447.
Der größte d1 tritt bei `LGM90D:000020` auf (Zentralalter 3.313,94 BP).
Die drei P99-Überschreitungen im ersten Szenario betreffen 3.313,94,
3.455,88 und 4.018,12 BP. Die sieben d1-P95-Flags im 100-km-Szenario
liegen bei 2.648,45; 2.980,68; 3.313,94; 3.455,88; 4.018,12;
4.050,47 und 13.528,99 BP. Schlechtere Abdeckung betrifft hier also
auch das Holozän, nicht nur die ältesten Proben.

Die geringere Flagzahl mit Puffer bedeutet vor allem eine größere moderne
Referenzschwelle: Der fossile d1 ist für alle 69 Proben in beiden Szenarien
gleich. Sie belegt keine verbesserte fossile Übertragbarkeit. **Die Flags sind
keine validierten Nichtanalog-Grenzen, p-Werte oder Erfolgswahrscheinlichkeiten.**
Der gepoolte Distanzvergleich ist kein weiterer räumlicher Klimatest und nicht
mit den Testfehlern von #18 gleichzusetzen.

Bei elf Proben kommen die fünf nächsten Nachbarproben aus weniger als fünf
Koordinatengruppen (achtmal vier, zweimal drei, einmal zwei). Im Pufferszenario
betrifft dies zwölf. k=5 bedeutet daher nicht fünf unabhängige Standorte.
Die Entfernung der 345 Nachbarn zum Fossilort reicht im ersten Szenario von
1,51 bis 8.828,99 km, Median 1.689,41 km. Eurasische Ähnlichkeit in der
37-Gruppen-Teilzusammensetzung garantiert keine lokale Klimaentsprechung.

## Chronologie: was die Quellen tatsächlich liefern

Entity **932 / LGM90D** hat Modelltyp **1 = linear** und einen behaltenen
Tiefenbereich **0–12 m**. Alle 69 Cut-Zeilen sind identisch auch in EpdAgeAll
enthalten; All enthält keine weiteren Proben innerhalb dieser Tiefengrenzen.
Die unabhängige Tiefensuche bestätigt 61 exakte und acht gerundete Zuordnungen
sowie alle 2.553 fossilen Merkmalszellen. Die Zuordnung bleibt
`candidate_depth_link`; sie ist kein neuer Identitätsnachweis anhand von
Originalprobenetiketten.

| Diagnose der 69 Cut-Proben | Ergebnis |
|---|---:|
| Zentrale Alter | 85,75–20.222,57 Jahre BP |
| Äußerste publizierte Altersgrenzen | 84,35–20.617,90 Jahre BP |
| Intervallbreite min / Median / max | 3 / 418 / 775 Jahre |
| Abstand zentraler Alter min / Median / max | 12,85 / 176,27 / 2.258,95 Jahre |
| Benachbarte Intervallpaare mit positiver Überlappung | 46 von 68 |
| Unsicherheitsklasse 3 / 4 | 17 / 52 Proben |

Die Zentralalter steigen mit der Tiefe streng an. Die acht Tiefenrundungen
entsprechen unter der ausdrücklich nur diagnostischen lokalen Steigung der
publizierten Zentralalter höchstens etwa 13,45 Jahren. Das ist **kein**
zusätzliches Altersintervall und korrigiert die Quellenalter nicht.
Ein feines Ausgaberaster würde weder die großen Lücken noch die Altersunsicherheit
auflösen. Überlappende Randintervalle liefern keine Wahrscheinlichkeit einer
Altersumkehr und keine unabhängigen Altersziehungen.

### Kontrollpunkte und Bedeutung der Unsicherheit

Frisch geladenes [Control_P, PANGAEA 804606](https://doi.org/10.1594/PANGAEA.804606):
1.393 LGM90D-Zeilen, davon 1.392 `ANL` und ein `TOP_A`; innerhalb 0–12 m
liegen 328 `ANL`-Zeilen. Die Felder für Radiokarbonalter sind leer, die
Kalenderalter vorhanden. Empirisch entsprechen alle ANL-Fehler gerundet 1 %
des Alters mit einer Untergrenze von einem Jahr. Das sind keine 1.392
unabhängigen absoluten Datierungen.

Die veröffentlichte Tabelle enthält 29 mehrfach belegte Tiefen, davon 18 mit
unterschiedlichen Altern; elf Zeilen sind vollständig doppelt. Tiefen und Alter
sind nicht fallend, enthalten aber Plateaus. Diese Quelldaten wurden nicht
dedupliziert und kein neues Modell daraus angepasst.

Nach [Giesecke et al., Methodenabschnitte](https://doi.org/10.1007/s00334-012-0390-y)
stehen `ANL` für jährliche Schichtung und die Klassen für Datierungsdichte und
Sequenzverlauf. Bei fehlenden Warvenfehlern wurde 1 % angesetzt. CLAM 1.0.2
erzeugte 1.000 Alters-Tiefen-Beziehungen; publiziert wurden wahrscheinlichste
Probenalter und 95-%-Grenzen. Diese Randgrenzen bewahren die gemeinsame
Abhängigkeit der Kurven nicht. Die Autoren weisen auf unterschätzbare
Unsicherheiten zwischen Kontrollen hin. Für diesen Pilot ist kein gemeinsames
Ensemble aus den bezogenen Tabellen verfügbar.

### Zusätzliche Quellen und Grenzen einer Neudatierung

Die neuen Dateien und Prüfsummen stehen in [sources.json](./sources.json).

- [Watts: LGM90D-Datierungen, 740512](https://doi.org/10.1594/PANGAEA.740512):
  drei AMS-14C-Daten bei 7,25 / 9,00 / 13,85 m; die ersten beiden liegen im
  Cut und haben laut Originalkommentar eine durch Pollenkorrelation bestimmte
  Tiefe. Werte 10,460 / 12,540 / 23,140 ka sind unkalibrierte 14C-Alter
  (Libby-Halbwertszeit, BP=1950). Sie dürfen nicht direkt von Kalenderaltern
  subtrahiert oder als vollständig unabhängige Probenanker behandelt werden.
- [Watts: LGM90D-Tephren, 741058](https://doi.org/10.1594/PANGAEA.741058):
  elf Lagen, davon sechs im Cut. Auch deren Altersfeld ist ausdrücklich
  **14C age BP**, ohne Unsicherheitsfelder. L4–L9 nicht anhand ähnlicher Namen
  automatisch in jüngere TM-Serien umbenennen oder als präzise Kalenderanker nutzen.
- [Allen et al. 1999, 707050](https://doi.org/10.1594/PANGAEA.707050):
  1.120 Alters-Tiefen-Zeilen, 1.118 eindeutige Paare; 622 Paare stimmen exakt
  mit EPD-ANL-Kontrollen überein. Diese numerische Überlappung spricht für
  gemeinsame Chronologieinformation, nicht für eine unabhängige Validierung.
  Die MON-B-D-J-Kompositzuordnung ersetzt keinen belegten Crosswalk zu LGM90D.
- [VARDA](https://varve.gfz-potsdam.de/database/details/94dd95c2-dacf-46fc-aa8f-f8be21c09a8e)
  führt diese unterschiedlichen Kerne/Komposite separat.
  [MON-2014](https://doi.org/10.5194/cp-10-2099-2014) betrifft 76–112 ka,
  [Martin-Puertas et al. 2019](https://doi.org/10.1016/j.quascirev.2019.05.019)
  den letzten Interglazial-Glazial-Übergang. Beides ist kein direktes Ersatzmodell
  für die 0–20-ka-Proben. Die lange Kernfolge am See ist laut
  [GFZ](https://www.gfz.de/en/section/geomorphology/projects/lago-grande-di-monticchio/the-last-interglacial-in-monticchio)
  ein Komposit aus mehreren Kampagnen.

Zusätzlicher Metadatenkonflikt: EPD nennt 1.326 m Höhe, Allen 707050 nennt
656 m. Die Höhenangabe wurde nicht für diese Distanzdiagnose verwendet oder
still korrigiert. Auch die leicht verschiedenen Ortskoordinaten bleiben als
Quellversionen getrennt.

**Nächster Forschungsschritt:** Originale Tiefen-/Proben- und Kompositzuordnung
sichern; datierte Materialien, Pollenkorrelation und Tephraidentitäten prüfen;
erst dann mit ausdrücklich gewählter Kalibrierkurve und korrelierten
Warven-/Ankerfehlern gemeinsame monotone Altersensembles erstellen bzw. beziehen.
Die 176 ausgeschlossenen Proben werden weiterhin weder extrapoliert noch über
EpdAgeAll freigegeben. Dessen Zentralalter reichen hier nur bis ca. 76,2 ka,
nicht bis 150 ka.

## Reproduktion und Prüfung

Quellen aus dem bisherigen Register und `sources.json` frisch laden; die
Shared37-Tabellen nach dem bestehenden Präparer erzeugen. Dessen vollständiges
Manifest ist identisch mit der gespeicherten Referenz aus #17. Danach:

```bash
python tools/thesis_import/diagnose_monticchio.py \
  --input-dir /path/to/shared37 --raw-dir /path/to/old_and_new_sources \
  --output-dir /path/to/new_monticchio_diagnosis
python tools/thesis_import/audit_monticchio.py \
  --run-dir /path/to/new_monticchio_diagnosis
```

Das Ziel darf nicht bestehen. Keine Klimazielwerte werden ausgewertet.
Alle 488.382 Fossil-Modern-Abstände wurden zusätzlich mit Python-Subtraktion
und `math.fsum` nachgerechnet, alle 138 Top-5-Ranglisten beider Szenarien
abgeglichen. Drei moderne Referenzproben wurden ebenso direkt geprüft.
Ein zusätzlicher Audit mit `statistics.quantiles(method='inclusive')` bestätigt
die Quantile, sämtliche Flags und den Intervallmedian. Zwei vollständige Läufe
sind einschließlich Manifest bytegleich. [verification.json](./verification.json)
nennt Umfang und Grenzen; [reference_manifest.json](./reference_manifest.json)
sperrt Quellen, Code und Ausgaben.

Eingecheckt sind die kleinen fossilbezogenen Diagnosen und Zusammenfassungen.
`modern_distance_reference.csv` und `chronology_controls.csv` werden vollständig
reproduziert. Quelldaten und abgeleitete Tabellen behalten ihre angegebenen
CC-BY-4.0 / CC-BY-NC-3.0 / CC-BY-3.0-Bedingungen; sie fallen nicht automatisch
unter die Code-Lizenz. Die Originalzählungen werden hier nicht erneut verteilt.

Claudes Integrations-PR #19 wurde separat geprüft:
[Reviewnachweis](./INTEGRATION_REVIEW.md). Keine seiner Integrationsdateien ist
Bestandteil dieser Änderungen.
