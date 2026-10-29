# Nächster Meilenstein: Kalibrierungsdaten und Taxonkonzepte

Stand: 29.10.2026; Ausgangspunkt `main` auf `853bb0916f8f46ff819321733020d56d3c574be6`. PR #11–#14 sind gemergt. Claudes PR #14 stellt `kinneret-recon thesis-import` bereit. Sein Integrationsbericht dokumentiert den vollständigen Originaldatenlauf mit identischen Quell-, Ausgabe- und Importer-Prüfsummen. Diese Sitzung hat den Diff und Bericht gelesen; die gesamte Projekttestsuite wurde hier nicht erneut ausgeführt.

Der technische Import ist damit abgeschlossen. Der nächste Meilenstein ist eine begründete Kalibrierungsauswahl und eine fachlich geprüfte gemeinsame Taxonomie. Ein erfolgreiches Einlesen ist keine Trainingsfreigabe.

## Tatsächlich ausgeführte Prüfung

Die vier Quelldateien werden erneut gegen das Quellenregister gesperrt. `audit_readiness.py` zählt verschachtelte Kohortenszenarien und sucht Namenskandidaten für alle 135 Monticchio-Datenfelder in der EMPD-SQLite-Datenbank. Es ändert weder Importer noch Referenz. Alle folgenden Zahlen sind eigene Dateiauszählungen; keine neue Modellsimulation und keine nachgewiesene Abdeckung aller europäischen Klimanischen.

| Verschachteltes Prüfszenario | Moderne Proben |
|---|---:|
| Gesamter EMPD-Dateibestand | 8.746 |
| Numerisches TANN/PANN, gültige numerische Koordinaten, keine Prozentdaten | 8.217 |
| Zusätzlich Standortqualität A, B oder C | 4.140 |
| Zusätzlich keine gebrochenen Werte in der gesamten Probenzählung | 3.956 |
| Zusätzlich innerhalb 25–75°N, 15°W–45°E | 2.664 |

Diese Szenarien sind **keine festgelegten Trainingssätze**. Insbesondere folgt aus der Box keine fachliche Europaauswahl, aus Klima-Vollständigkeit keine Eignung von Meeres- oder anderen Probentypen und aus Ganzzahligkeit keine passende Pollensumme. Die Alters- und Probentypauswahl sowie räumliche Doppelungen sind noch zu prüfen. Klimaherkunft bleibt WorldClim2; die genaue Datei-Referenzperiode ist weiter unbestätigt. Die 244 Proben mit gebrochenen Werten werden im Import erhalten; das Ganzzahl-Szenario ist eine offen ausgewiesene mögliche Sensitivitätswahl.

Im vollständigen Klima-/Koordinatenszenario besitzen 4.034 Proben die unbekannte Standortqualität als Text `NULL` und 43 Klasse D. Die pauschale Verwerfung unbekannter Qualität würde einen großen Teil des Archivs entfernen. Vor einer endgültigen Wahl sind räumliche und klimatische Abdeckung mit und ohne diese Gruppen zu vergleichen. Nicht aus einer hohen Gesamtprobenzahl auf ausreichende Glazialanaloga schließen.

## Namen sind noch keine Taxonkonzepte

Für jedes Monticchio-Feld werden der originale Parametername und der EPD-Kommentar aus der PANGAEA-Datei separat erhalten. Vergleichskanäle sind exakte Namen gegen originale oder akzeptierte EMPD-Namen; lediglich Groß-/Kleinschreibung und Whitespace werden normalisiert. Keine unscharfe Suche, keine manuell erfundenen Synonyme, keine Werteaggregation. Ein EMPD-Code wird nur Kandidat, wenn mindestens eine positive Messung im konservativen 3.956-Proben-Szenario vorliegt.

| Maschinelles Suchergebnis | Monticchio-Datenfelder |
|---|---:|
| Genau eine akzeptierte EMPD-ID als Namenskandidat | 93 |
| Mehrere akzeptierte IDs als Kandidaten | 18 |
| Kein Namenskandidat | 3 |
| Als technisch oder unbestimmbar vorzuprüfen | 21 |

Die Summe umfasst 135 Datenfelder, **keine 135 sicher getrennten Pollentaxa**. Auch die 93 Felder mit nur einem Kandidaten bleiben ungeprüft. Bei 24 Feldern werden Ziel-IDs von anderen Quellfeldern ebenfalls getroffen; diese Mengen überlappen die obigen Zeilen. Positive Probenzahlen je Originalcode in der ausführlichen JSON dürfen nicht über Synonyme addiert werden.

Konkrete Prüffälle:

| Quellfeld | Befund im gesperrten Datenbestand | Erforderliche Entscheidung |
|---|---|---|
| `Euc [#]` | EMPD-Kandidaten Eucalyptus, ID 39/TRSH, und Eucalyptus spike (counted), ID 2741/NOPO | Pflanzenbeobachtung und zugesetzten Marker unterscheiden. Für Monticchio ist die konkrete Rolle hier nicht aus dem Parameternamen allein belegt; bis zur Methodenprüfung nicht als Pflanzen-Klimasignal freigeben. |
| `Cypae [#]` | Gleichnamige Kandidaten Cyperaceae mit IDs 33/HERB und 3868/AQUA | Probentyp und Konzept prüfen; Namen allein lösen den Konflikt nicht. |
| `Que [#]`, `Que.r-T [#]`, `Que.i [#]` | Gattungs- und Typfelder gleichzeitig vorhanden | Klären, wann das Gattungsfeld residuale oder zusammengefasste Werte enthält. Über alle 245 Originalzeilen gilt die Gleichheit Gattung = Summe der beiden Typen nur in 27 Zeilen; keine pauschale Summenidentität annehmen. |
| `Can [#]`, `Urt-T [#]` | Über Originalnamen und Archivkommentare mehrere, teilweise gemeinsame Zielkonzepte | Unterschiedliche Auflösung und mögliche gemeinsame Sammeltypen prüfen; keine unabhängigen identischen Zielspalten erzeugen. |
| `Cornacae [#]`, `Crn [#]` | Beide treffen Cornus | Vor einer Zusammenfassung Originalzählpraxis und Überlappung klären. |
| Sporen-/Farnfelder | EMPD-Gruppen BRYO/VACR u. a. sind im Archiv teilweise `used_in_sum=1` | `used_in_sum` ist kein fertiger Filter für eine terrestrische Pollensumme. |

Ohne Namenskandidat im geprüften Szenario: `Adl-T [#]` (Asphodelus/Asphodeline), `Orn-T [#]` (Fritillaria-type) und `Ser-T [#]` (Serratula-type). Dies belegt weder ein globales Fehlen in EMPD noch eine Abwesenheit in der Vergangenheit. Es bedeutet nur: Die beschriebene exakte Namenssuche fand im Szenario keinen positiv beobachteten Kandidaten.

Primärbelege: Pollen- und Parameterdaten [Watts 2010, PANGAEA.739452](https://doi.org/10.1594/PANGAEA.739452), EMPD-SQLite und Gruppen-/Taxondefinitionen [Chevalier et al. 2019, PANGAEA.909130](https://doi.org/10.1594/PANGAEA.909130). Vollständige URLs und SHA256 stehen im Quellenregister. Eine ergänzende öffentliche Suche hat die konkrete Eucalyptus-Rolle in der Monticchio-Methode nicht geklärt; diese bleibt ausdrücklich offen.

## Nächste Arbeitsteilung

1. **OpenAI/Datenrecherche:** Zuerst die obigen Konflikte in Primärdokumentation und Originalzählpraxis prüfen. Daraus eine begründete Entscheidungstabelle mit `include`, `exclude`, `pool` oder `unresolved` je Konzept erstellen; Eltern-/Kindfelder und Marker zuerst. Danach Pollensumme und Ausschlussfolgen pro fossiler Probe ausweisen.
2. **OpenAI/Datenprüfung:** Die Kalibrierungsszenarien nach Probentyp, Raum und Klimabereich vergleichen. Unbekannte Standortqualität als Sensitivität erhalten. Einen räumlich getrennten Validierungsentwurf vor dem Training dokumentieren.
3. **Claude:** Erst nach diesem kleinen Datenvertrag eine einfache Pollen-Klima-Baseline integrieren. Die vorhandenen Importer unverändert weiterverwenden. Noch keine neuen Proxy-Likelihoods und keine aus Prozentwerten erzeugten Zählungen.

Du musst Claude derzeit keinen weiteren Implementierungsauftrag geben. Der technische Übergabeschritt ist erledigt; zunächst liegen die genannten Fachprüfungen bei der Datenrecherche.

## Reproduktion und sparsames Nachlesen

```bash
python tools/thesis_import/audit_readiness.py \
  --raw-dir /path/to/netzel_raw \
  --output-dir /path/to/new_readiness_audit
```

Der Vorgang benötigt keine Netzwerkverbindung. Im Dokumentationsordner `docs/thesis_research/baseline_readiness_2026_10_29/` stehen die kompakte `readiness.json` und die 135-zeilige `taxon_mapping_review.csv`. Die ausführliche `taxon_name_candidates.json` mit jeder Code-Evidenz wird reproduzierbar erzeugt und nicht zusätzlich eingecheckt. Zum Einstieg genügt dieser Bericht; die gesamte JSON muss kein Agent erneut lesen.

Die Datenlizenzen bleiben EMPD CC BY 4.0, Monticchio-Zählungen CC BY-NC 3.0 und EPD-Chronologien CC BY 3.0. Keine Quelle oder Namenszuordnung erhält hier wissenschaftliche Modellfreigabe.
