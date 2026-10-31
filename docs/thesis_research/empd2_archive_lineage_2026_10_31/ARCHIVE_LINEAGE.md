# EMPD2: Archivzuordnung belegt, Erzeugungskette offen

31.10.2026. Autorisierter Archivauftrag aus [PR #43](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/43),
Basis `90d0350f20d8adc5d1feab09722cf02f0ebb8b69`. Eigener Worktree
`/workspace/netzel-empd2-lineage`, Branch `codex/netzel-empd2-archive-lineage`.

**Begrenzter Abschluss:** Die offizielle Zuordnung des SQLite-Originals zum DOI
und ein PostgreSQL-Dumppfad sind belegt. Die tatsächlich angewandte
Quelle→Version→Verarbeitung→SQLite-Exportkette bleibt unbewiesen. Nach den zwei
gezielten Quellen wird die Suche hier beendet. Keine Counts-/Trainings-/Pilotfreigabe.

## 1. Offizielle Archivmetadaten zuerst

Chevalier, M.; Davis, B. A. S.; Sommer, P. S.; Zanon, M.; Carter, V. A.;
Phelps, L. N.; Mauri, A.; Finsinger, W. (2019): *Eurasian Modern Pollen Database
(former European Modern Pollen Database)*. PANGAEA,
[DOI 10.1594/PANGAEA.909130](https://doi.pangaea.de/10.1594/PANGAEA.909130),
CC BY 4.0. Gelesen: Landingpage und deren ausdrücklich verlinkte
[DataCite-4-Metadaten](https://doi.pangaea.de/10.1594/PANGAEA.909130?format=metadata_datacite4).
Beides sind Darstellungen desselben Archivobjekts, keine zusätzlichen Datenquellen.

Die Landingpage nennt Veröffentlichung am 26.11.2019, DOI-Registrierung am
22.02.2020 und drei Originalformate: Excel, PostgreSQL und SQLite. Ihre
Gleichwertigkeit ist eine Archivangabe, hier kein vollständiger Datenvergleich.
Der darunter angebotene Tabellenexport betrifft Klima; er ist kein Ersatz für
das gesperrte Pollen-SQLite. Die DataCite-Relation `IsNewVersionOf` verweist auf
dieselben drei URLs. Diese Relation dokumentiert keine technische
PostgreSQL→SQLite-Konvertierung. Eine konkrete Code-/Releaseversion oder ein
angewandter Exportbefehl für genau diese SQLite-Datei ist darin nicht ausgewiesen.

Ziel bleibt exakt `EMPD2.sqlite3`, 21.327.872 Bytes, SHA-256
`5634e85d922ee0741c41a77d081010c9cc1c113966b283bc52e2051a9d3722e2`.
Der offizielle SQLite-Link entspricht der bereits in #35 gesicherten Bezugs-URL.
Die vorhandene Cachedatei wurde nur gehasht, nicht neu heruntergeladen oder als
Datenbank geöffnet. Ihr Hash stimmt vor/nach der Arbeit mit der Sperre überein.
Eine HEAD-Anfrage bestätigt die aktuelle Bytezahl, nicht allein die Byteidentität
einer heute heruntergeladenen Datei. HTTP-Last-Modified ist kein Erzeugungsdatum.

## 2. Genau ein offiziell verwiesenes zweites Originalobjekt

Die Landingpage verlinkt unter „Original version“ ausdrücklich den
[PostgreSQL-Dump EMPD2.sql](https://store.pangaea.de/Publications/ChevalierM-etal_2019/EMPD2.sql).
Damit ist die Voraussetzung für dieses zweite Objekt erfüllt. Kein Excel,
weiteres Release, heutiger Branch oder Ersatzarchiv nachgeladen.

Nur HTTP-Header und Bytebereich **0–32767** gelesen: HTTP 206,
`Content-Range: bytes 0-32767/12718488`; SHA-256 des **Bereichs**, nicht des
ganzen Dumps: `7c097392cccb08c86d4a39cc3c031cda9c2e8904e9cd4723140ccf7bc0dee1a8`.
Das sind 32.768 Bytes statt einer vollständigen SQL-Kopie. Ausgewertet wurden
Kopf und Schema bis Zeile 298. Der Präfix enthält danach beginnende Klimazeilen;
daraus wurde kein Probenvergleich oder Klimabefund abgeleitet. Keine
`p_counts`-Messzeilen gelesen, kein SQL eingespielt oder Code ausgeführt.

| Fundstelle im Originalpräfix | Belegt | Grenze |
| --- | --- | --- |
| Zeilen 5–6 | Datenbank und `pg_dump` jeweils PostgreSQL 11.2 | Engine-/Werkzeugversion, keine EMPD-Daten- oder SQLite-Exportversion |
| 109–115, `public.groupid` | Elternfeld und beide booleschen Gruppenflags | Kein Beleg für 40 konkrete Gruppen oder rekursiven Algorithmus |
| 137–159, `public.metadata` | `ispercent`, `isdigitised`, `empd_version` als Schemafelder | Kein globaler Buildbeleg oder ursprüngliches Proben-N |
| 193–197, `public.p_counts` | Schlüsselspalten und `count double precision` | Dezimalwerte darstellbar; keine nachgewiesene Übertragung von 7,4 |
| 206–212, `public.p_vars` | Original-/akzeptierte Namen und Gruppenverweis | Kein Messstatus für ein fehlendes Sample×Typ-Paar |

Header und Schema belegen einen PostgreSQL-Dump. Sie liefern keine angewandte
Anweisung zum Erzeugen des gesperrten SQLite-Objekts. Andere, ungelesene Teile
des Dumps werden weder als geprüft noch als frei von weiterer Logik bezeichnet.

## Quelle → Version → angewandter Schritt → Archivfeld

| Verbindung | Status |
| --- | --- |
| DOI → konkrete SQLite-URL → gesperrte lokale Bytes | Offizielle Zuordnung und früherer Bezugsnachweis; Sperrhash jetzt bestätigt |
| DOI → konkrete PostgreSQL-URL → Dumpkopf 11.2 | Offizieller Verweis und Bereichshash jetzt belegt |
| PostgreSQL-Daten-/Codeversion → tatsächliche Konvertierung → SQLite-Felder | **Nicht belegt**; keine Konvertierung aus Ähnlichkeit oder Gleichwertigkeitsangabe ableiten |
| Gepinntes v2.0-Release → genau dieser SQLite-Build | **Weiterhin offen**; Datum, Gruppenunterschied und vorhandene Proben begrenzen die Übertragung |

Die [vorigen Belege](../empd2_denominator_rules_2026_10_31/RULE_EVIDENCE.json)
bleiben unverändert: 108 übereinstimmende Messwert-Tupel der drei bekannten
Zeugen, erhaltenes Volumen 7,4, 36 Release- gegenüber 40 SQLitegruppen und
ein Neuanlagepfad, der rundet und vorhandene Proben überspringt. Diese Befunde
wurden übernommen, **nicht neu berechnet**. Weder „Code nie benutzt“ noch
„komplette Reproduktion gelungen“ folgt daraus. Keine weiteren Zeugen geprüft.

`N_original` bleibt der nicht belegte ursprüngliche Messumfang.
`S_flag` bleibt ein mit den gelieferten Prozenten kompatibler Nenner bei genau
drei Zeugen, nicht automatisch `N_original`. `n_ausgewählt` wäre erst eine
fachlich definierte disjunkte Teilmenge; hier wurde keine gewählt. Messstatus,
allgemeine Nullunterdrückung und Hierarchieanwendung bleiben getrennt offen.

## Sicherung und Prüfung

[LINEAGE_EVIDENCE.json](./LINEAGE_EVIDENCE.json) enthält Objekte, Locatoren,
Fingerprints, Standprüfung und Prüfgrenzen; [DECISION_GAPS.md](./DECISION_GAPS.md)
den offenen Entscheidungspunkt. Temporäre Quellbytes sind keine behauptete
dauerhafte Archivierung; Belege/Hashes und präzise Wiederbezugsanfragen stehen im Git.

Beim Einstieg #31–#43 offen und ungemergt, main `7f1826c`, #43-Head wie beauftragt,
#43-PR-CI erfolgreich. Fünf hier vorhandene vorherige Worktrees sauber erhalten;
frühere Angaben über neun Worktrees betreffen die vorige Umgebung.
Keine neue Matrix/Kohorte, Datenbereinigung, Ausführung von Projekt-/Quellcode,
Fits, Scores, Simulationen, Kern-/CLI-/App-/Standardtests. Keine Unteragenten,
Autorennachrichten oder Merges. Bestehende Designs/Modelle/Resultate/Register/
Importer/Flags/Counts/Gruppen, Baseline und CLI/App unverändert.
W96, reales T1/T0/T3/T4 und Monticchio-P1 bleiben offen.
