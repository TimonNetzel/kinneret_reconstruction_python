# EMPD2: lesendes Audit eines modernen Counts-Kalibrierungswegs

31.10.2026. Nutzer-OK hier im bestehenden Chat; Basis PR #41 bei
`62ffeaf08ee7c7538f8d9c4cc99f78e7c0bdb394`. Eigener Worktree
`/workspace/netzel-empd2-readiness`, Branch `codex/netzel-empd2-counts-readiness`.
[Git-/PR-Snapshot](./git_snapshot.json): #31–#41 offen, main `7f1826c`, #41 mit
vier erfolgreichen CI-Checks CLEAN/MERGEABLE. Kein Merge und keine Unteragenten.

**Ergebnis:** Technische Eingänge für einen direkten modernen Typkompositionsweg
sind vorhanden. **Ein realer Trainingssatz oder eine Counts-Likelihood ist noch
nicht freigegeben.** Der konkrete Zählnenner, die Bedeutung fehlender Sparse-
Zeilen, harmonisierte Typkonzepte und Klimaprovenienz sind nicht hinreichend
belegt. Negative Bereitschaft ist hier das fachliche Ergebnis dieses Audits,
nicht ein Fehlschlag des synthetischen Prototyps.

## Quellen und tatsächlich gelesener Umfang

Das vorhandene Original `EMPD2.sqlite3` wurde erneut gehasht: 21.327.872 Bytes,
SHA-256 `5634e85d922ee0741c41a77d081010c9cc1c113966b283bc52e2051a9d3722e2`,
identisch zum gesperrten [#35-Manifest](../original_pollen_names_2026_10_31/data/manifest.json).
Keine neuen Downloads. SQLite mit `mode=ro&immutable=1`, zusätzlich
`PRAGMA query_only=ON`; Hash vor/nach identisch. Originalquelle
[PANGAEA.909130](https://doi.org/10.1594/PANGAEA.909130),
[registrierte Datei](https://store.pangaea.de/Publications/ChevalierM-etal_2019/EMPD2.sqlite3),
CC BY 4.0, Chevalier et al. (2019)/Davis et al. (2020) gemäß bestehendem Register.
Publikationsvolltexte und Bestimmungsmethoden in diesem Paket nicht neu gelesen.

Gelesen: relevante Tabellenstrukturen, alle 40 Gruppendefinitionen, Standort-/
Alterscodebeschreibungen, unterschiedliche `metadata.ispercent`-Werte sowie
**genau drei deterministisch ausgewählte Zeugenproben mit zusammen 108
gespeicherten Messzeilen**. Auswahl lexikalisch erste Nicht-Prozentprobe,
erste Probe mit gebrochenem numerischem Wert, erste mit textuellem Jahresklima.
Keine Auswahl anhand Klima-/Modell-Scores oder erwünschter Eignung.
Kein kompletter Messmatrixexport, keine Kohortenstatistik, keine Fits/Simulationen/
Tests/Kernläufe. [evidence.json](./evidence.json) enthält SQL-Abfragen, Locator,
Metadaten, wenige Messwertbeispiele und gruppenweise Demonstrationssummen.

## Schema: Werte sind nicht automatisch Pollencounts

| Fundstelle im Original | Befund | Bedeutung für den Vertrag |
| --- | --- | --- |
| `p_counts(samplename,var_,count)` | Zusammengesetzter Primärschlüssel; `count NUMERIC(12,6)` | Numerische Messwerte, keine durch den Typ erzwungenen ganzzahligen Kornzahlen. Schlüssel nicht vor Harmonisierung verwerfen. |
| `metadata.ispercent`, `isdigitised` | In dieser Datei liefert `SELECT DISTINCT ispercent,typeof(ispercent)` ausschließlich 0/integer. Drei Zeugen jeweils zusätzlich isdigitised=0. | Nicht-Prozentflag ist kein ausreichender Beleg diskreter Pollenkornzahlen; kein Umrechnen oder Runden. |
| `p_vars` | Originalcode/-name, akzeptierte ID/Name, `groupid`; keine morphologische Typdefinition im Schema | Akzeptierte Namensrelation ist weder botanische Synonymie noch automatische Aggregations-/Bestimmungsfreigabe. |
| `groupid` | `higher_groupid`, `used_in_sum`, `percent_values`; TRSH/HERB/VACR u. a. mit used_in_sum=1, NOPO/CHAR/FUNG mit 0 | Definitionen sind vorhanden, aber noch kein probenbezogener Nachweis des tatsächlich verwendeten Nenners oder der hierarchischen Aggregationsregel. Die beiden Tabellenflags nicht gleichsetzen. |
| `metadata` / `p_counts` | Kein eigenes strukturiertes Feld für den gemessenen probenspezifischen Zählnenner N | N nicht als beliebige Summe aller Werte oder bloße Summe der sieben bisherigen Arten erzeugen. Originalprotokoll/Verarbeitung nötig. |
| `climate` | Schlüssel samplename; t_ann, p_ann, t_djf, Monats-/Saisonfelder | Technischer Join existiert; SQL-Affinität NUMERIC verhindert Text-NULL nicht. TANN und TDJF nicht vertauschen. |

`groupid.percent_values=1` ist ein Gruppenflag, `metadata.ispercent=0` ein
Probenflag; sie existieren gleichzeitig. Ihre Kombination ist nicht als
Prozent→Counts-Umrechnungsanweisung belegt. Auch used_in_sum=1 bei Bryophyten/
Pteridophyten illustriert, dass eine fachliche Nennerdefinition mehr erfordert
als das Wort „Pollen“ oder eine unkommentierte Gruppenmaske.

## Drei konkrete Zeugen: keine Trainingsfreigabe aus Beispielen

Alle folgenden Summen sind **arithmetische Demonstrationen an gespeicherten
Werten, keine genehmigten Pollensummen**. Insbesondere die Gesamtsumme kann
verschiedene physikalische Messgrößen addieren und besitzt dann keine sinnvolle
einheitliche Zähleinheit.

| Probe / Auswahlgrund | Gelesene Zeilen | Alle Werte addiert | Nur Gruppenflag used_in_sum=1 | Konkreter Befund |
| --- | ---: | ---: | ---: | --- |
| `AlbaSanchez_a1`, erste Nicht-Prozentprobe | 20 | 311 | 311 | Alle gelesenen Werte ganzzahlig; TRSH 285/HERB 26. Mossprobe, Koordinaten und Klima numerisch, Standortklasse B; Alter/Unsicherheitsklasse und samplemethod Text-NULL. Gleichheit beider Summen beweist keinen ursprünglichen Zählnenner. |
| `Champvillair_a18`, erste gebrochene Messung | 49 | 13009,4 | 731 | Einziger gebrochener Wert dieser Probe: Originalcode 693, „Volume (cm3)“ → „Volume of sample (cm3)“, 7,4, NOPO, used_in_sum=0/percent_values=0. Summe der flag-1-Gruppen ganzzahlig; Sammlungsmethode „hand picking - multiple samples“, Standortklasse/Alter unbekannt. |
| `Barboni_a1000`, erste textuelle Jahresklimazeile | 39 | 843 | 840 | AQUA-Wert 3 liegt außerhalb flag-1-Summe. Koordinaten, Höhe, Jahres-/Saisonklima sind Text-NULL, Standortklasse X „Do not use!“. Kein Rettungs-Join aus Koordinaten im Freitext. |

Für Champvillair ergeben DWAR 1 + HERB 270 + TRSH 450 + VACR 10 die
Demonstrationssumme 731. Außerhalb liegen unter anderem CHAR, FUNG, RHIZ und
NOPO einschließlich Volumen. Ein Rest `13009,4 − ausgewählte Taxa` wäre daher
keine Zahl gezählter sonstiger Pollenkörner.

Die [Importerreferenz](../../../tools/thesis_import/reference_run.json) berichtet
historisch 1.059 gebrochene Messwerte aus 244 Proben. Diese Zahlen wurden hier
**nicht neu über die Matrix berechnet**. Der Importer markiert bereits bei einem
gebrochenen Wert **irgendeiner** Messgröße die gesamte Probe mit fractional_counts
([import_data.py](../../../tools/thesis_import/import_data.py), Funktion empd_exports,
Messwertschleife und samples). Unser Volumenbeispiel zeigt: Daraus folgt weder
„244 Proben haben gebrochene Pollencounts“ noch deren pauschale Unbrauchbarkeit
unter jeder denkbaren fachlichen Pollensumme. Es folgt auch keine Freigabe dieser
244 Proben: Die geeignete Teilmenge ist ohne Nenner-/Typprüfung unbekannt.
Keine Änderung an Referenzzahlen, Flags, Importer oder alten Berichten.

## Fehlend, Null und Typkonzept

Konkreter Sparse-Zeuge: `(samplename='AlbaSanchez_a1', var_=7)` hat keine
p_counts-Zeile; p_vars[7] bezeichnet Artemisia. Dieser SQL-Nichttreffer belegt
keinen gemessenen Nullcount und keine lokale botanische Absenz. Das vorhandene
[Importprotokoll](../../../tools/thesis_import/README.md) hält Sparse-Missingness
ausdrücklich ungelöst und ergänzt keine Nullen. Ein fehlendes Paar darf bis zur
Belegung des Erfassungsprozesses nicht still zur vollständigen Countmatrix werden.

Original-/akzeptierte Konzepte müssen erhalten bleiben. Beispiel p_vars[18]:
„Caryophyllaceae“ → „Caryophyllaceae (periporate excl. Paronychioideae)“;
p_vars[424]: Abies pinsapo → Abies pinsapo. Das ist keine Zuordnung zur bestehenden
Artfunktion Abies alba. Artenmodell-IDs, EMPD-Originalcodes und akzeptierte
Konzept-IDs sind verschiedene Schlüsselräume; keine Zuordnung anhand gleicher
Ziffern. Weitere Taxonnamen wurden nicht neu breit inventarisiert.
Die [#37-Grenzen](../pollen_methods_2026_10_31/README.md) bleiben verbindlich;
W96-Methodenkette und die fossilen Artbrücken bleiben offen.

## Klima, Standort und Zeit: Rollen statt stiller Reparatur

AlbaSanchez_a1 hat `t_ann=12.33`, `p_ann=833`, `t_djf=5.97`;
Champvillair_a18 `t_ann=0.57`, `p_ann=1867`, `t_djf=-5.97`.
Der [Importvertrag](../../../tools/thesis_import/README.md) kennzeichnet
tann_c/pann_mm_per_year als WorldClim2 mit Referenzperiode **unverified**.
Der frühere [Quellenbericht](../source_audit_2026_10_29/README.md) referiert die
Publikationsangabe 30 Bogensekunden/nächster Gitterpunkt; dies ist in diesem
Paket gelesener Repositorykontext, keine neue Volltextprüfung.
Das SQLite-Schema selbst enthält keinen Klimaversion-/Perioden-/Einheitennachweis.
Kein stilles WorldClim-2.1-/1970–2000- oder CRU-1961–1990-Label übernehmen.

Historisch berichtet der Importer 529 fehlende Klimazuordnungen, 284 ungültige
Koordinaten und 4.136 unbekannte Standortklassen. Nicht frisch gezählt, kein
neuer geeigneter Stichprobenumfang daraus abgeleitet. Standort-Codetabelle:
A <100 m, B <1 km, C <5 km, D <20 km, X „Do not use!“; dies sind gespeicherte
Klassenbeschreibungen, keine hier nachgemessenen Positionsfehler.
Barboni_a1000 belegt Text-NULL und X direkt. Keine Freitext-/Nullwertkorrekturen.

Alle drei Zeugen haben unbekanntes Alter/Altersklasse. Eine moderne Datenbank
macht eine einzelne Probe nicht automatisch zeitgleich zur Klimanormalperiode.
Moss-/Mehrfachsammlung und unbekannte zeitliche Integration sind ein offener
Beobachtungs-/Repräsentativitätsvertrag, keine N unabhängige Klimareplikate.
Für Champvillair enthält metapubli→publications die bibliographische Referenz
Furlanetto et al. (2018), DOI 10.1177/0959683618810395; nur Metadaten gelesen,
keine dortige Methodik behauptet. Bei den anderen beiden Zeugen liefert der
direkte Join keine Publikationszeile; das beweist nicht, dass keine Quelle existiert.

## Möglicher direkter moderner Weg – nur prospektiv

Ein späteres Modell könnte die **beobachtete Typkomposition** `pi(c,Kontext)`
direkt aus geklärten modernen Pollenproben schätzen, bedingt auf einen belegten
Zählnenner und Erfassungsprozess. Es würde Produktion/Transport/Erhaltung im
beobachteten Datenregime gemeinsam abbilden; es identifiziert diese Größen nicht
einzeln und liefert keine lokale Artenpräsenzwahrscheinlichkeit. Das wäre ein
eigenes Kalibrierungsdesign mit passender Fehlermodellierung, keine Inversion
oder automatische Übernahme der sieben Arealwahrscheinlichkeiten.
Die synthetischen #41-Erfolge ersetzen weder Nenner-/Typbelege noch empirische
Bestätigung oder paläoökologische Übertragbarkeit.

Vor einer Kalibrierung wären Typumfang/Partition und Rest, zulässige Messarten,
Sparse-Regel, Probenkontext/Alter, Klimaprovenienz und Qualitätsregeln zu sichern.
Standorte, Mehrfachsammlungen und geteilte Herkunft benötigen getrennte,
vorab festgelegte Trainings-/Bestätigungsrollen ohne räumliche/Quellenlecks.
Unabhängige Bestätigung braucht neue geeignete Evidenz, keine bekannten äußeren
Scores. Keine Folds, Schwellen, Architektur oder Varianten danach auswählen.
Support wäre derjenige der tatsächlich zugelassenen modernen Daten und ihrer
Typ-/Kontext-/Klimageltung; hier weder Kohorte noch Hülle berechnet.

Der [INPUT_CONTRACT](./INPUT_CONTRACT.json) hält verfügbar, ungesichert und
gesperrt maschinenlesbar auseinander. [SOURCE_GAPS](./SOURCE_GAPS.md) benennt
den nächsten engen Originalquellenauftrag. Reale Kalibrierung und Fossilpiloten
bleiben gesperrt; unveränderte Analog-Baseline und CLI/App bleiben Referenzen.

## Prüf- und Sicherungsstand

[Dokumentationsprüfung](./verification.json) bestanden: Quellenhash vor/nach,
Vertrag und drei Zeugen/108 verschiedene Messzeilen, Demonstrationssummen,
historische gegenüber frisch geprüften Aussagen, JSON, lokale Links und Diff.
701 geschützte vorhandene Git-Blobs einschließlich Symlink bytegleich gegen
`62ffeaf`; nur HANDOFF/PROJECT_STATUS als bestehende Dateien fortgeschrieben.
Alle acht bisherigen Worktrees sauber erhalten. Keine Tests/Fits/Simulationen
gestartet und kein neuer Modell-/Messmatrixexport; keine unabhängige menschliche
oder palynologische Prüfung behauptet. Kompakte Belege und Vertrag werden in Git
gesichert; die Quelldatei bleibt als geprüftes lokales Original außerhalb Git.
Der nächste Quellenauftrag in SOURCE_GAPS/HANDOFF wartet auf erneutes Nutzer-OK.
