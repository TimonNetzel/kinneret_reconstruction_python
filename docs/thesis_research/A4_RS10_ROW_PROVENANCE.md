# A4: begrenzte RS10-Herkunftsprüfung

01.11.2026. Das Nutzer-„Mache weiter“ autorisiert den vorbereiteten
Herkunftsauftrag aus PR #75. Basis
`10afa451def72a8a89ffe717dc0a00f5da3bc610`, offen/ungemergt,
vier Head-Checks erfolgreich. **Prüfung begrenzt abgeschlossen;
Zeilenherkunft des modernen Kandidaten weiterhin unbelegt.**

## Auftrag und übernommener Stand

Ziel war ein Originalnachweis für die Verbindung
`pollen_climate`-Zeile → Originalprobe/Sammlung/Quelle → Klima-Zeitbezug.
Die [gesicherte Objekt-/Strukturprüfung](./A4_MODERN_DATA_CANDIDATE.md) wird
übernommen: Datenblob `e7a7d8b6fec3711af7a1888ba83871009ad8e981`,
7.832 Zeilen, Y/X ohne Originalprobenkennungen. Kein neues Herunterladen,
Deserialisieren oder Prüfen von Beobachtungswerten.

Geprüft wurden der bezeichnete Datenaufnahme-Commit, der Zugriff auf die
direkt referenzierte Allen-Originalbeschreibung und zwei gezielte Suchrunden.
Eine daraus gefundene Originalquelle wurde nur an der RS10-Datenbeschreibung
gelesen. Keine weitere Quellensuche nach Ausschöpfen dieses Umfangs.
Die [kompakte Evidenz](./a4_rs10_provenance_2026_11_01/evidence.json) enthält
Abfragen, Quellenidentität, Fundstellen, Zitat und Beleggrenzen.

## Neue Belege und Grenzen

| Beleg | Ergebnis im gelesenen Umfang |
| --- | --- |
| R1: [zanicc-Datenaufnahme-Commit](https://github.com/AndrMenezes/zanicc/commit/e51f6738df4a397b986c0cb9b6d089e612e90518), 26.08.2026 | Nachricht „add datasets“, keine Kommentare, drei hinzugefügte Binärdateien. Darunter der bereits gesicherte moderne Blob. Die Commitmetadaten liefern keine Herkunftstabelle oder angewandte Aufbereitung; Binärinhalte und Datenpatches nicht gelesen. Kein Nachweis, dass anderswo keine Dokumentation existiert. |
| R2: Allen, Watts & Huntley (2000), [DOI 10.1016/S1040-6182(00)00067-7](https://www.sciencedirect.com/science/article/abs/pii/S1040618200000677), direkt in der gepinnten Paketdokumentation referenziert | Originalabruf HTTP 403. Methodenteil nicht gelesen; Such-/Abstracttext ist kein Herkunftsnachweis. Keine Aussage über unzugängliche Inhalte und kein unveränderter Wiederholungsabruf. |
| R3: Michael Salter-Townshend (2009), [institutionelle Dissertation](https://www.tara.tcd.ie/tara8/server/api/core/bitstreams/40a72ac0-4fc4-4e81-97a9-167da1cf68f5/content#page=152), §6.1.1, Druck-S.138 / PDF-S.152 | Beschreibt einen älteren RS10-Stand mit 7.742 Standorten und 28 Taxa. Für Fälle mit nur Anteilen und fehlender Zählsumme wird ausdrücklich ein Gesamtumfang von 400 angenommen. Diese Passage liefert keine Zeilenbrücke zum heutigen Kandidaten. |

Originalausschnitt R3: **“assuming a total count of 400”**. Nur der
Zeilenumbruch wurde für das Zitat vereinheitlicht. Identität über Titelblatt
(April 2009), Erklärung (07.05.2009), Autor und institutionellen Locator
gesichert; keine allgemeine Dissertation-/Methodenlektüre.

**Folgerung für unseren Eingang:** Der historische Beleg macht einen
Messherkunftsnachweis nötig, der echte Zählungen und aus Anteilen abgeleitete
Werte unterscheidet. Er beweist weder Pseudo-Counts noch einen Fehler im
aktuellen Menezes-Objekt. Unterschiedliche Stände bleiben getrennt; keine
betroffene Zielzeile wurde identifiziert. Auch eine spätere Zeilensumme 400
wäre allein kein Nachweis einer solchen Umrechnung. Hier wurden keine Summen
gebildet oder Daten klassifiziert.

## Abschluss und konkrete Restlücke

In den zwei Suchrunden wurde **keine verifizierte Zuordnungstabelle oder
angewandte Aufbereitungskette für genau den Zielblob** ermittelt. Ähnliche
RS10-Bezeichnungen, passende Taxazahl, andere Datensätze mit 7.832 Einträgen
und fossile Rekonstruktionsausgaben wurden nicht als Identitätsbelege verwendet.
Dies ist ein begrenzter Nichtbeleg, keine Behauptung weltweiter Nichtexistenz.

Zum Wiederöffnen dieses Datenwegs fehlt ein versionsgebundener Originalbeleg
mit Zielzeile/-reihenfolge, Originalproben-/Sammlungs-/Quellkennung und
Zuordnungsschritt. Er muss insbesondere ursprüngliche Counts/N gegenüber
übernommenen Anteilen bzw. angenommenen Summen kenntlich machen; Zeit- und
Klimabezüge bleiben ebenfalls nachzuweisen. Unbekanntes bleibt unbekannt.
Das konkretisiert die vorhandenen Rollen im
[Modellvertrag](./MODEL_CONTRACT.md#beobachtungsstatus-nullen-und-abdeckung),
ohne neue Schwellen oder eine alternative Likelihood festzulegen.

**Keine reale Kalibrierung freigegeben, kein E gewählt.** Für diesen
Datenweg endet die aktive Suche hier. Wiederaufnahme nur mit neuem,
verifiziertem Herkunfts-/Aufbereitungsbeleg; dieselben Suchrunden oder
Blockerabrufe nicht erneut beginnen. Ein neuer Datenweg oder Autorenkontakt
ist nicht beschlossen. Es gibt keine Pflichtdatei oder verifizierte
Konfigurationsaktion, die der Nutzer nun bereitstellen müsste.

Furlanetto bleibt separat vorgemerkt und zugangsabhängig; seine PhD und
bestehenden Belege wurden nicht erneut gelesen. A1/A2/A3, EMPD2-Prüfungen,
Originaldaten, Modelle, Verträge, Importer, Baseline und CLI/App unverändert.
Keine neuen Binärdaten, Joins, Bereinigung, Auswahl, Nennerrechnung, manuelle
wissenschaftliche Tests, Fits, Scores, Simulationen, Unteragenten, Kontakte
oder Merges. Automatische Repository-CI getrennt berichten.

Der dauerhafte Nachweis umfasst Bericht und kompakte Evidenz, keine
Vollarchivierung der Dissertation oder ursprünglicher Daten. Dokumentarische
Abnahme bestanden: Quellen-/Fundstellenbezug, 25 Index-Pointer, sieben Evidenz-
Blobs, neun lokale Links/Anker, erhaltene vorherige Evidenz samt Hash und
Statushistorie; fünf Dateien im Diff, übrige Worktrees sauber. Sicherung und CI in
HANDOFF/PROJECT_STATUS; keine unabhängige palynologische Prüfung behauptet.
