# Netzel: kurzer Projekteinstieg

## Aktueller Stand, 01.11.2026

Nutzer-„Mache weiter“ hat das vorbereitete RS10-Herkunftspaket autorisiert.
[Prüfung abgeschlossen](./A4_RS10_ROW_PROVENANCE.md): Datenaufnahme-Commit
enthält den gesicherten Blob, aber keine Herkunftstabelle/Aufbereitung.
Zwei Suchrunden beendet; kein zeilenbezogener Originalnachweis ermittelt.
Allen-Originalabruf HTTP 403, Methodeninhalt ungelesen.

Neuer Beleg: Salter-Townshend (2009), §6.1.1, Druck-S.138/PDF-S.152,
beschreibt für einen älteren RS10-Stand teils angenommene Zählsummen N=400.
Keine Übertragung auf heutige Zielzeilen, keine Fehlerbehauptung zu Menezes.
Für das gepinnte `pollen_climate` fehlen weiterhin Originalproben-/Quellbindung,
Messherkunft/ursprüngliches N und genauer Klima-Zeitbezug. Kein E gewählt,
kein Gate geöffnet, kein Modelllauf. Objekt-/Strukturprüfung übernommen.

Basis [PR #75](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/75),
`10afa451def72a8a89ffe717dc0a00f5da3bc610`, offen/ungemergt,
vier erfolgreiche Head-Checks. Branch `codex/netzel-rs10-row-provenance`,
Arbeitsverzeichnis `/workspace/netzel-a4-next-data-gap`; Sicherung unten.

## Genau ein Anschluss – Originalbeleg bewerten, neuer Eingang fehlt

**Ziel/Eingang:** Nur mit neuem verifiziertem Originalbeleg die konkrete
Zeilen-/Messherkunft bewerten: versionsgebundene Zuordnungstabelle oder
nachweislich angewandter Aufbereitungsnachweis für Datenblob
`e7a7d8b6fec3711af7a1888ba83871009ad8e981`. Zielzeile/-reihenfolge,
Originalprobe, Sammlung/Quelle, ursprüngliche Counts/N gegenüber abgeleiteten
Anteilen/angenommenen Summen und Zeit-/Klimabezug müssen zuordenbar sein.
Unbekanntes erhalten. Dieser Anschluss ist vorbereitet; aktueller Suchauftrag
abgeschlossen. Ohne neuen Beleg nicht ausführbar, keine erneute gleiche Suche.

**Leseplan:** AGENTS, dieser HANDOFF, nur neuester PROJECT_STATUS;
Index `menezes_modern_pollen_climate`/`rs10_historical_measurement_provenance`
→ Evidenz und konkrete Restlücke. Alte Objekt-/Struktur- und A1–A3-/EMPD2-
Prüfungen übernehmen. Nur neue Frage/Version am Original lesen.

**Abnahme/Stop:** Nach tatsächlicher Zeilenbindung oder konkretem Nichtbeleg
stoppen. Keine Identität aus RS10-Label, Zeilenzahl, Koordinaten oder N=400;
keine Werteanalyse, Datenwahl, Join, Reparatur, Zählsummenrechnung, neue
Implementierung, manuellen wissenschaftlichen Tests, Fits, Scores, Simulationen,
Gatefreigaben, Kontakte oder Merges. Neue Ergebnisse gemeinsam sichern.

**Nutzeraktion:** Keine Pflichtdatei, Nachricht oder Konfigurationsaktion.
Bekannter Originalbeleg/Link wäre optionaler neuer Eingang; seine Verfügbarkeit
ist nicht belegt. Neuer Datenweg/Autorenkontakt nicht beschlossen.
Furlanetto bleibt separat autorisiert, vorgemerkt und zugangsabhängig;
keine PhD-Neulesung oder unveränderten BOA-/Sage-Abrufe. R01–R03, reales A4,
A5/A6 und Monticchio-P1 behalten ihre bisherigen Grenzen.

## Folgeprompt

```text
Bewerte einen neu verfügbaren zeilenbezogenen Originalnachweis für den
gepinnten pollen_climate-Datenblob gemäß HANDOFF. Bestehende Evidenz übernehmen.
Ohne neuen belegten Eingang keine Wiederholung der abgeschlossenen RS10-Suche,
Objekt-/Strukturprüfung oder anderen Fachprüfungen. Historischen N=400-Befund
nicht auf aktuelle Zielzeilen übertragen. Nur Belegprüfung, keine Datenwahl,
Joins oder Modellläufe; gemeinsam sichern, kein Merge.
```

## Sicherung

[PR #76](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/76),
offen/ungemergt gegen #75, Inhaltscommit
`b44010ca6ddd84703d30ee26f3b12225738fcace`. Vier automatische Checks bei
erster Abfrage wartend/laufend. Danach nur Sicherungsverweise ergänzt; aktuellen
Head/CI bei tatsächlicher Fortsetzung prüfen. Kein Merge.
