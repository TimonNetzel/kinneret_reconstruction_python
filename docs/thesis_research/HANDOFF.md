# Netzel: konkrete Drei-Zeugen-Zeitprüfung und zulässige Leseprüfungen

## Paketabschluss, 01.11.2026

Nutzerauftrag ausgeführt: [AGENTS](../../AGENTS.md#eigenständige-internetrecherche-als-standard)
erlaubt notwendige, begrenzte schreibgeschützte Original-/Metadatenprüfung
innerhalb einer beauftragten Provenienzfrage ohne zusätzliches OK. Alte
Paketverbote gelten als Historie; der zuvor vorbereitete reine
Stop-/Eingangsplan wurde durch den konkreten Drei-Zeugen-Auftrag ersetzt.

[Review der drei Zeugen](./A4_E_PROVENANCE_EVIDENCE_REVIEW.md#konkrete-zeitprüfung-der-drei-zeugen-01112026)
und [Evidenzprotokoll](./witness_time_check_2026_11_01/evidence.json) sichern:
AlbaSanchez_a1 → „1 béticas“, Champvillair_a18 → „TRS 18“, Barboni_a1000 →
„MAROC124“. Genau deren SQLite-Metadaten/Publikationsrelationen frisch
geprüft und mit drei Zeilen des festen v2.0-Release verglichen. Bei allen
Alter/Altersklasse unbekannt; keine individuelle Zeitzuordnung gewonnen.
Material/Entnahme für AlbaSanchez/Champvillair teilweise dokumentiert,
Integration unbekannt; Barboni ohne Zeit-/Material-/Methodenbeleg.
Champvillair ist direkt mit DOI 10.1177/0959683618810395 verknüpft.

**Konkrete technische Restlücke:** Hauptmethoden zugangsbeschränkt;
FLORE-PDF als geschlossen geführt. Verlinktes Supplement ist DOCX, vom
Webwerkzeug nicht gelesen; Executor-Download scheitert am Proxy-CONNECT
mit HTTP 403. `journals.sagepub.com` fehlt in der geltenden Hostliste.
Gefundene öffentliche Furlanetto-Dissertation ebenfalls nicht abrufbar
(Web HTTP 403). Keine Inhalte dieser Dateien als geprüft behauptet und
kein fachlicher Nichtbeleg daraus. Kein Nutzer-Dokument erforderlich;
Repo-Regeln können die technische Netzwerkgrenze nicht aufheben.
Aktuell verfügbare Cloud-Werkzeuge bieten hier keine Konfigurationsänderung.

Basis #66-Head `a9177cf84b2cc3d0b1e1983c7315880d93e5c5e6` geprüft;
#66 offen/ungemergt und mergebar, Head-CI 41288446572/41288442734 mit vier
erfolgreichen Checks. Sechs vorgefundene Worktrees sauber erhalten;
eigener Worktree `/workspace/netzel-witness-time-check`, Branch
`codex/netzel-witness-time-check`. Inhaltscommit
`42d365fc410feac53e90b1047fd0a5a184c8745b` über
[PR #67](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/67)
remote gesichert, offen/ungemergt und mergebar, gestapelt gegen #66.
Automatische [CI 41289425528](https://github.com/TimonNetzel/kinneret_reconstruction_python/actions/runs/41289425528)
am Inhaltscommit bei Abfrage laufend. Danach nur Sicherungsverweise;
aktuellen PR-Head/CI prüfen. Kein Merge.

SQLite-Original vor/nach SHA-256 identisch; Release-Metadatenblob gegen
Git-Blob-SHA geprüft. Nur drei Zeugen inhaltlich bewertet; Gesamtblob für
gezielte Extraktion temporär bezogen, drei Auszüge/Abfragen im Repo gesichert.
Identitäts-/JSON-/Diffprüfung und zehn lokale Links/Anker bestanden; alter
Reviewtext, vollständige Statushistorie sowie 752 geschützte Objekte/Modi erhalten.
Keine Counts-/Klimawerte, neuen Proben, Reparaturen, Projektimplementierung,
manuellen Projekt-/Modelltests, Rechnungen/Fits/Simulationen oder Freigaben.
R01–R03 nicht neu gesucht, R06/übrige Gates offen; keine A4-Umsetzung/A5/A6,
Unteragenten, Kontakte/Nachrichten oder Merges. Ergebnis fachlich teilweise,
Prüfplan bis zu den benannten Nachweis-/Zugriffsgrenzen ausgeführt.

## Genau ein Folgeauftrag: Furlanetto-Originalmethodik für TRS 18 – vorbereitet, wartet auf Nutzer-OK und lesbaren Zugriff

**Ziel:** die konkrete noch nicht gelesene Methodik für Champvillair_a18 /
TRS 18 aus DOI 10.1177/0959683618810395 samt öffentlichem Supplement lesen.
Sammlung, Ablagerungs-/Integrationsbezug und Unsicherheit proben-/gruppenbezogen
belegen oder den genauen Nichtbeleg der gelesenen Stellen sichern.

**Technische Startbedingung:** erlaubter lesbarer Download des bereits
verlinkten Supplement-DOCX (Executor-Host `journals.sagepub.com`, notwendige
tatsächliche Weiterleitungen separat prüfen) oder zugängliches öffentliches
Autorenoriginal. Ein öffentlich ermitteltes Original beschafft Codex selbst.
Keine Nutzerdatei oder Autorennachricht. Die heutige HTTP-/Werkzeuggrenze
ist real; derselbe abgewiesene Abruf wird ohne geänderten Zugriff nicht
wiederholt. Keine Netzwerkregel umgehen oder deren Änderung vortäuschen.

**Leseplan:** aktuellen PR-Head/CI/Worktrees prüfen, Arbeit erhalten;
AGENTS, diese HANDOFF, neuester Status; Review nur aktueller Drei-Zeugen-
Abschnitt; evidence.json nur Champvillair/W3–W6/Zugriffsstatus;
A4_E_REOPENING_REQUIREMENTS R04/R05 und Herkunftsregeln. Anschließend exakt
zugeordnete Originalmethoden/Supplement, gegebenenfalls die bereits gefundene
öffentliche Autorenfassung. Keine erneute Inventur/Definitionstextsuche.

**Umfang/Abnahme:** Review ergänzen und Quellen-/Abfrageevidenz aktualisieren,
HANDOFF/Status kurz pflegen. Einen nachgewiesenen Gruppengeltungsbezug oder
explizite TRS-18-Bindung verlangen; Zitat/Seite/Version und Aussagegrenze
festhalten. Keine Datenergänzung aus allgemeinem Moostyp oder Publikationsjahr.
Nach diesem Original-/Supplementversuch stoppen, auch bei Nichtbeleg;
Zugriffsfehler von inhaltlichem Nichtbeleg trennen. Quellen/Hashes/Links/Diff
prüfen, Commit/eigener PR ohne Merge, genau einen sachlichen Anschluss
vorbereiten und Folgeprompt automatisch mitliefern.

**Schutz:** vorhandene Originale/alte Berichte/Verträge erhalten. Keine neue
Probe/Kohorte, Pollen-/Klimawertanalyse, Datenreparatur, Projektimplementierung,
manuellen Tests, wissenschaftlichen Rechnungen, Gate-/R06-Freigabe,
A4-Umsetzung/A5/A6, Unteragenten, Nachrichten/Kontakte oder Merges.
Notwendige schreibgeschützte Quellen-/Metadatenprüfung ist nach AGENTS erlaubt.

## Kopierfertiger Folgeprompt

```text
Setze Netzel auf codex/netzel-witness-time-check fort.
Inhaltscommit: 42d365fc410feac53e90b1047fd0a5a184c8745b
PR: https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/67
Danach nur Sicherungsverweise; aktuellen PR-Head/CI prüfen.
Prüfe aktuellen PR-Head/CI; erhalte ungesicherte Arbeit.
Eigener Worktree/Branch/PR, kein Merge.
Führe ausschließlich den in HANDOFF vorbereiteten Furlanetto-Auftrag für
Champvillair_a18 / TRS 18 aus: zugeordnete öffentliche Originalmethodik und
Supplement zu DOI 10.1177/0959683618810395 selbst beschaffen und lesen.
Leseplan, lesbare technische Startbedingung, Umfang, Abnahme und Stop einhalten.
Supplementzugriff scheiterte am Executor-Proxy mit HTTP 403; Web kann DOCX
nicht lesen. Ohne geänderten erlaubten Zugriff keinen identischen Abruf
wiederholen; keine Umgehung und keine Nutzerdatei voraussetzen.
Nur Review/Evidenz und HANDOFF/Status pflegen; alte Originale/Berichte erhalten.
Nötige begrenzte schreibgeschützte Quellen-/Metadatenprüfung nach AGENTS
ist erlaubt, kein zusätzliches Recherche-/Zeilenprüfungs-OK verlangen.
Keine neue Probe/Kohorte, Pollen-/Klimawertanalyse, Datenreparatur,
Projektimplementierung, manuellen Tests oder wissenschaftlichen Rechnungen.
Keine R01–R03-Neusuche, R06-/Gatefreigabe, A4-Umsetzung/A5/A6,
Unteragenten, Nachrichten/Kontakte oder Merges.
TRS-18-/Gruppenbindung, Sammlung, Ablagerung, Integration und Unsicherheit
getrennt belegen; nach den Originalstellen oder konkreter Zugriffslücke stoppen.
Ergebnis prüfen, per Commit/eigenem PR sichern, genau einen sachlichen
Folgeauftrag vorbereiten und Folgeprompt automatisch mitliefern.
Kurz auf Deutsch berichten.
```
