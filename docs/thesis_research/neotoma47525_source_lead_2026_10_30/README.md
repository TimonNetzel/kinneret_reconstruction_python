# P1: neuer Quellenpfad über Neotoma 47525

**Nachtrag 30.10.2026:** Ein gespeicherter Workshop-RDS-Bestand ist inzwischen
bezogen und geprüft: [neuer Archivbericht](../neotoma_workshop_2026_10_30/README.md).
Die folgenden Abrufgrenzen beschreiben den vorausgehenden Arbeitsschritt.

Stand 30.10.2026. Fortsetzung nach Lesen von `CHAT_RESTART.md` und `HANDOFF.md`
auf PR #25, Commit `00c9f037642470c7d1a9a44e488f062109292c2d`, sowie der
Berichte und Quellenregister aus #23–#25. Die bisherigen Ergebnisse werden
nicht ersetzt. Dies ist eine begrenzte Quellenprüfung, kein Rohdaten-Audit.

## Ergebnis und belastbare Grenze

Ein neuer, konkret identifizierter Eingang ist der vollständige
**Neotoma-Pollendatensatz 47525**, DOI
[10.21233/KVNC-0856](https://doi.org/10.21233/KVNC-0856).
Er wird in der aktuellen Holocän-Studie ausdrücklich benutzt. Die bisherigen
Neotoma-Abrufe in #25 betrafen dagegen Datensatz **27166**, das isolierte
TM16b-Ereignis. Dessen begrenzter Inhalt entscheidet nicht darüber, welche
physischen Probenmetadaten 47525 enthält.

**47525 wurde hier nicht vollständig heruntergeladen.** Weder seine
Zählungsmatrix noch seine Probenetiketten oder Chronologien wurden geprüft.
Es wird ausdrücklich nicht behauptet, er enthalte die gesuchte Originalliste
oder sei mit EPD-Entity 932 identisch. Der Quellenpfad ist neu belegt; die
physische LGM90D→MON16-Zuordnung einschließlich Unsicherheit bleibt offen.

## Neue Fundstellen

[d’Oliveira et al. (2026), QSR 380, 109873](https://doi.org/10.1016/j.quascirev.2026.109873):
gelesen wurde der vom Erstautor bereitgestellte
[Artikeltext](https://www.researchgate.net/publication/401705206_North-south_Holocene_seasonal_contrast_on_the_Italian_peninsula_Clarification_of_the_latitudinal_transition_zone_based_on_the_multi-proxy_study_of_Lago_Grande_di_Monticchio),
nicht ein lokal gespeichertes Original-PDF.

| Fundstelle | Quellenbefund |
| --- | --- |
| Abschnitt 2.2, S. 4 | Pollen stammt aus dem älteren Komposit MONTICCHIO mit LGM-D, -E/P, -K, -L und MF93; MONTI23-II liefert die Biomarker, keine dort neu analysierten fossilen Pollen. |
| Abschnitt 2.5.1, S. 5 | Neotoma-Datensatz 47525, Standort 3295; Auswahl etwa der letzten 13.000 Jahre / 8,17 m und numerische Ausdünnung auf mindestens 50 Jahre Abstand wegen Kernüberlappungen. |
| Abschnitt 3.2, S. 7; Tabelle B.2 | Das Chronomodel-Modell und die physischen Tephraetiketten betreffen MONTI23-II. |
| Abschnitt 2.8, S. 6 | Vergleich der beiden Klimazeitreihen über zeitliche Interpolation, Filterung und Korrelation. |

**Eigene Schlussfolgerung:** Diese gelesenen Passagen liefern keinen
zeilenweisen Anschluss der 69 EPD-Proben an MON16. Die Ausdünnung ist auch
keine ursprüngliche Proben-/Kompositierungsliste. MONTI23-II, das ältere
Pollenkomposit und MON16 müssen getrennt behandelt werden. Das ist eine
Aussage über den gelesenen Umfang, kein Beweis, dass andernorts keine
Zuordnung existiert.

Ein zweiter, direkt gelesener Beleg ist
[PANGAEA.989985](https://doi.pangaea.de/10.1594/PANGAEA.989985),
d’Oliveira et al. (2026), *Pollen-based summer and winter Holocene
palaeoclimate reconstruction of the Lago Grande di Monticchio (Italy)*.
Der Katalog nennt Allen, Huntley & Watts (2021), Version 1.1, als Pollenquelle.
Die [HTML-Tabelle](https://doi.pangaea.de/10.1594/PANGAEA.989985?format=html)
führt Standort 3295 und Datensatz 47525. Die Parameter umfassen Tiefe in m,
Alter in ka BP und Klimarekonstruktionen. Der Katalog ordnet das Event
MONTI23-II zu. **Dieser Eventeintrag allein belegt keine Transformation der
Pollentiefen auf MONTI23-II oder MON16.** Die Tabelle wird hier weder als
Rohzählungsarchiv noch als neue Chronologie für die 69 Proben übernommen.

## Was als Nächstes konkret zu beschaffen und zu prüfen ist

Vorrangig der vollständige Download von **47525**, einschließlich
Collection-Unit-/Dataset-Metadaten, Analysis Units, Sample IDs, Namen,
Tiefen/Einheiten/Mächtigkeiten und sämtlicher Notizen, Chronologie- und
Publikationsverknüpfungen, soweit vorhanden. Downloadbytes mit Datum,
Größe und SHA256 sichern; ein Metadateneintrag oder eine leere Fehlerantwort
ist kein erfolgreicher Eingang.

Danach zuerst prüfen, ob ursprüngliche Kern-/Segment-/Abschnittsetiketten
oder dokumentierte Kompositoffsets überhaupt enthalten sind. Erst dann
einen gesonderten, reproduzierbaren Identitätsvergleich mit den bereits in
#25 bestätigten Datenbankproben 4–72 vornehmen. Keine Zuordnung allein nach
ähnlichen Altern, Tiefen, Zeilenreihenfolge oder Standortnamen. Ein numerischer
Pollenvergleich kann Kandidaten liefern, aber keine physische Herkunft beweisen.

Falls 47525 die physischen Felder nicht liefert, bleibt die Original-
Proben-/Kompositierungsliste der Datenhalter erforderlich. Die in #25
beschriebenen Felder und die Korrelationsunsicherheit bleiben Abnahmekriterien:
[MISSING_SOURCE_REQUEST.md auf festem Commit](https://github.com/TimonNetzel/kinneret_reconstruction_python/blob/00c9f037642470c7d1a9a44e488f062109292c2d/docs/thesis_research/epd_sample_identity_2026_10_30/MISSING_SOURCE_REQUEST.md).
Kein weiterer unveränderter Durchlauf durch die bereits leeren EPD-Tabellen.

## Zugriff und Prüfung dieses Arbeitsschritts

[sources.json](./sources.json) hält Quellen, Fundstellen, gelesenen Umfang
und erfolglose Abrufe getrennt fest. Die Neotoma-API lieferte über den
Webleser keinen lesbaren Download; ein direkter Abruf von
`/v2.0/data/downloads/47525` endete nach 30 Sekunden ohne Nutzdaten.
Die Landingpage war aufrufbar, lieferte dem Textleser aber keinen Seiteninhalt.
Das sind Zugriffsgrenzen, keine negativen Datenbankbefunde. Auch ein
bytegleicher PDF- oder TSV-Eingang wird nicht behauptet; deshalb keine
erfundenen Quellhashes. Die Webfundstellen sind für eine spätere Prüfung an
Originalbytes ausdrücklich zu bestätigen.

Vor Beginn waren PRs #18–#25 offen und ungemergt. Arbeitsbasis ist deshalb
#18, `f9acc16c58a47a51b1063690a69a2db01381c76a`; Claude-PR #19 stand auf
`c8977a57fecabcef8f54abd7eafd6545f53487f9`. Die neue Arbeit umfasst nur
diesen Forschungsordner. Keine Baseline-Neuentwicklung, keine Änderung an
Importer, Shared37, Alterswerten oder CLI/App. Keine Quellenkorrektur,
kein P2-Start, keine Autorennachricht und kein Merge. Das nicht angehängte
ScienceDirect-ZIP wurde nicht erneut geprüft.

Prüfung: JSON-Struktur, Quellen-/Fundstellenkonsistenz, lokale Markdownlinks
und ausschließlich dokumentarischer Diff gegen #18. Keine wissenschaftlichen
Rohdaten- oder Modelltests beansprucht. Bestehende unveränderte Prüfungen
wurden nicht erneut ausgeführt. Die neue PR-Beschreibung enthält den
verifizierten Commit und den tatsächlich abgeschlossenen Prüfumfang.
