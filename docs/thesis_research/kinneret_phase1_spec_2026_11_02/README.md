# Kinneret: maschinenprüfbarer Phase-1-Belegvertrag

Stand 02.11.2026, Basis `00d7d2a55759ccd8f3157294d16adaa4c3fae301`,
Branch `dissertation-reconstruction`, [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).

Die bisherigen Eingabe-, Referenz- und Nachbearbeitungsbefunde sind in
[spec.json](spec.json) zusammengeführt. Die Integritätsprüfung besteht:
26 eingefrorene Belegdateien, 19 explizite Befunde und 50 Dateien des erhaltenen
Forschungssnapshots stimmen mit den gespeicherten Prüfsummen überein.
**Kein lokaler Dissertationsfall ist dadurch vollständig reproduziert oder
startbereit.** Kinneret bleibt der gezielte Klärungsfall aus der
[Fallmatrix](../local_case_matrix_2026_11_02/README.md).

## Inhalt und belegte Grenzen

Die Spezifikation trennt vier Quellgruppen: das textlich beschriebene Verfahren,
den erhaltenen Fallstudiencode, den heutigen aktiven Einstieg im Originalquellstand
und verwandte Vergleichsfassungen. `selected_variant` bleibt `null`.
Quelltextfund und Bindung an den tatsächlichen historischen Lauf sind verschiedene
Nachweise. Die Spezifikation ist ein eingefrorener Belegvertrag, keine ausführbare
Rekonstruktionskonfiguration. Die in ihr verankerten JSON-Pointer verweisen direkt
auf die vorhandenen Verträge; deren Quellenhashes und Quellauszüge bleiben erhalten.

| Voraussetzung | Status | Zum Schließen benötigter Originalnachweis |
| --- | --- | --- |
| Finaler Basiseintrag | fehlt | `core_data_additional_sites_v1.2.rdata`, Kinneret-Slot oder exakt gebundener Erzeuger samt Eingaben |
| Finale Chronologie | widersprüchlich | Erwartete Liste von Altersrastern mit Probenbindung, Einheiten und rho(D gegeben A); Mittelwertmatrix reicht nicht |
| Historische TFs | fehlt | Numerische Funktionen, Metadaten, finale Reihenfolge/Taxonzuordnung und Achsen |
| Isotopen-Workspaces | fehlt | Ursprüngliche `d18o.age`-Objekte, Ladefolge, konkrete Reihen und Zeitachsen |
| Klima-/Klassenbindung | fehlt | Vier konsumierte globale CRU-Dateien mit Variablen, Einheiten, Zeitachsen und historischer kgc-Version |
| Referenzvorbereitung | widersprüchlich | Laufbindung der Soreq-Positionen 37/105/170, MEDSTACK-Auswahl und des tatsächlichen Altersfensters |
| Klimaanker | widersprüchlich | Originale lokale/regionale Auswahl, Parameter und Bezugsperiode |
| Originalposterioren/Auswahl | fehlt | Tatsächliche Posteriorpaare, Vorverarbeitung und originale Ziehungs-IDs |
| Finale Nachbearbeitung | widersprüchlich | Laufbindung von IQR/IAT/Pareto, Indexrückabbildung und Dichtekombination |
| Laufzeit/Zufallszustand | fehlt | R-/Paketfassung, RNG/Seeds und Eingabe-/Quellidentität für exakte Wiederholung |
| Numerische Originalausgabe | fehlt | Finale Dichten/Quantile und Auswahlmetadaten; Grafikgleichheit allein reicht nicht |

Die elf Zeilen sind offene **Nachweisbündel**, keine elf unabhängigen fehlenden
Dateien. Basis-/TF-Lücken verhindern bereits den vollständigen historischen
Eingabestand. Fehlende RNG-Zustände betreffen insbesondere exakte Wiederholung;
ein späterer approximativer Vergleich müsste entsprechend gekennzeichnet werden.
Das Zeitfenster 0–9 ka BP ist die ungefähre Textangabe, kein belegtes finales
numerisches Raster. TANN in °C, PANN in mm/a; Darstellungsrichtung jung links,
alt rechts. Keine stillen Quellenreparaturen oder Variantenentscheidungen.

## Lesender Prüfer

Aus dem Repository-Hauptverzeichnis, Python >=3.11, ohne Zusatzpakete:

```powershell
python tools/check_kinneret_phase1_spec.py
python tools/check_kinneret_phase1_spec.py --with-local
```

Der Standardlauf liest nur die eingefrorene Spezifikation und ihre expliziten
Repository-Belege. Die SHA-256 der Spezifikation ist im Prüfer fest verankert.
Ein geändertes Dokument, entfernte Voraussetzungen oder ein eingefügtes
Freigabeflag werden abgewiesen; es gibt keinen lokalen Override. Dieser Pin ist
ein Schutz vor unbeabsichtigtem Auseinanderlaufen, keine kryptographische
Beglaubigung gegen gleichzeitige Änderung von Dokument und Prüfer.

`--with-local` liest zusätzlich die private Pfadkonfiguration
`.local/my_reconst_paths.json`. Nur die in drei bestehenden Quellenverträgen
explizit aufgeführten Original-/Zusatzpfade werden geprüft. Kein Verzeichnisbaum
wird durchsucht, kein R-Container deserialisiert, keine Quelldatei geschrieben.
Die 40 unterschiedlichen Pfade ergeben 28 unveränderte vorhandene Dateien und
zwölf weiterhin fehlende Pfade. Das umfasst **nicht** alle externen Abhängigkeiten
oder die außerhalb dieser Wurzeln konsumierten globalen CRU-Pfade. Die bereits
geprüfte [Abhängigkeitsliste](../kinneret_references_2026_11_02/dependencies.csv)
bleibt maßgeblich. Neu auftauchende Dateien werden als `new_source_requires_audit`
gemeldet; bloßes Vorhandensein schließt keine fachliche Voraussetzung.

| Exitcode | Bedeutung |
| --- | --- |
| 1 | Spezifikation/Belege ungültig, verändert, unlesbar oder lokaler Aufruf nicht prüfbar |
| 2 | Intakter Belegvertrag mit offenen Voraussetzungen; bei lokalen Abweichungen zusätzlich Quellenprüfung nötig |

Exitcode 2 ist hier das erwartete Ergebnis. `historical_reproduction_ready=false`
und `execution_supported=false` bleiben in dieser Fassung fest. Der Prüfer
führt keine Rekonstruktion aus und ist kein vorgeschaltetes Freigabesystem für
App/CLI. Eine spätere geschlossene Konfiguration benötigt neue geprüfte Belege
und eine entsprechend aktualisierte Spezifikation; der aktuelle Prüfer kann
keinen startbereiten Fall ausgeben. Monticchio-P1 bleibt offen und fallbezogen;
die erhaltenen Monticchio-Gates werden nicht zu Kinneret-Voraussetzungen.

Gesicherte Ergebnisse: [portabler Bericht](portable_report.json),
[gezielter lokaler Bericht](local_report.json), [Prüfnachweis](verification.json).
Die Berichte enthalten nur Quellaliase/relative Pfade, keine privaten Wurzelpfade.
Der Prüfer selbst schreibt nichts und gibt JSON nach stdout aus.

## Prüfung und nächster Einstieg

20 gezielte Regressionstests bestanden: Beleg- und Snapshot-Manipulation,
entfernte Voraussetzungen, eingefügte Freigabeflags, ungebundene Variantenwahl,
Pfadgrenzen, mehrdeutiges JSON, typgenaue Befunde, lokale Quellenänderungen und
CLI-Exitcode. Testkopien liegen im System-Tempverzeichnis. Ein erster Testaufruf
scheiterte an fehlendem Elternverzeichnis, der zweite an der Windows-Pfadlänge
unter dem tiefen Checkout; mit kurzem System-Temppfad bestanden alle Tests.
Keine Modell-/Sampler- oder Fossilläufe. Frühere Belegpakete und die 50
Snapshotdateien unverändert; App/CLI, Baseline und Referenzen nicht angefasst.

**Nächster sinnvoller Phase-1-Einstieg:** Zuerst einen tatsächlichen finalen
Originaleingabestand samt Laufzuordnung beschaffen und an diesen Vertrag binden,
beginnend mit Basis/Chronologieliste und numerischen TFs. Ohne diese Nachweise
ist keine vollständige historische Kinneret-Reproduktion begründbar. Der Nutzer
kennt keinen weiteren Speicherort der Basisdatei; diese Frage nicht wiederholen
und keine pauschale Gesamtinventur anschließen. Eine vorhandene Grafik, ein
älterer Biomcontainer oder sieben moderne Artenmodelle schließen die Lücke nicht.

Falls Originalwerte dauerhaft fehlen, ist die Wahl eines ausdrücklich anders
gekennzeichneten methodischen Nachbaus mit Ersatz-/Neudaten eine noch nicht
getroffene fachliche Entscheidung. Sie wird hier weder vorausgesetzt noch
umgesetzt. Keine automatische Fortsetzung mit Kalibrierung, Reliabilitätsansicht
oder Phase 2. Historische Verfahren und spätere Kritik bleiben getrennt.
