# Quellen- und Belegnachprüfung, ohne Ausführung fremden Codes

Basis `e1e8ebe` aus PR #42. Die Original-HTML-Methodik und das dort referenzierte
offizielle EMPD2/EMPD-data-Release v2.0 sind die zwei Quellenobjekte.
Navigationsseiten dienten nur dazu, den offiziellen Repositorybezug zu prüfen.
Keine weiteren Datenquellen/Methodenpakete für die Regelbehauptungen.

Artikel: https://essd.copernicus.org/articles/12/2423/2020/, §2/2.2, §4/§6,
Literaturverweis Sommer et al. 2020. Abschnitts-/Absatzlocator verwenden,
keine ungelesenen PDF-Seiten angeben. Webwerkzeug konnte den Originaltext lesen;
kein dauerhafter Volltext-Bytebestand oder neuer Artikelhash behauptet.

GitHub-Tagmetadaten für v2.0 zeigen auf den annotierten Tag
`f543a1bf60fefb705e46db400f59ebdac96edaf7`, dann auf Commit
`737b76eb269b071422edded6ac332654ff83c103`. Ausschließlich diesen Commit lesen.
source_comparison.json enthält für zwölf Originaldateien unveränderliche URLs,
Git-Blob-SHA1, SHA-256 und Bytezahl. Sie wurden über die GitHub-Verbindung
bezogen, in einem separaten Cache gesichert und gegen Git-Blobhash verifiziert.
Eine zusätzliche Transport-Leerzeile beim Cache-Schreiben wurde vor der
Hashprüfung entfernt; alle gespeicherten Originaldateien stimmen bytegenau
mit ihren jeweiligen Git-Blobhashes überein. Keine inhaltliche Quellenkorrektur.

Nur folgende Dinge nachprüfen, nicht ausführen:

1. Tabellen-/Importdateien und genannte Zeilen am festen Commit lesen;
   kein PostgreSQL/Docker starten, keine Skripte importieren oder Tests ausführen.
2. Dieselben drei Sample-TSVs lesen; ihre Original-/akzeptierten Namen, Gruppe
   und Messwerte mit den gleichen drei Proben des read-only SQLite-Originals
   vergleichen. Keine weiteren Proben oder Vollmatrix.
3. S_flag aus Originalgruppen-Flags berechnen, bereitgestellte percentage-Felder
   lediglich vergleichen. Der erlaubte Rundungsabstand pro Feld ist die Hälfte
   einer Einheit der letzten gedruckten Dezimalstelle. Keine Prozentwerte
   zu Counts rückrechnen; keine neue Null- oder Nennerregel anwenden.
4. 36 Releasegruppen gegen 40 SQLitegruppen als Metadaten abgrenzen; gleiche
   Flags/Eltern bei gemeinsamen IDs nicht als volle Versionsgleichheit ausgeben.

Die drei TSVs haben 108 Messzeilen insgesamt, 98 Prozentfelder und zehn leere
Prozentfelder. Alle 108 Messwert-Tupel stimmen, die drei S_flag-Werte sind
311/731/840. source_comparison.json hält diese begrenzte Kontrolle fest.
Die fremden Quelltexte bleiben außerhalb des Projekt-Git; kompakte Locatoren,
Fingerprints und Vergleichsbelege werden mit diesem Paket in Git gesichert.
Das ist keine behauptete dauerhafte Sicherung aller Originalquellbytes.

SQLite: vorhandene 21.327.872-Byte-Datei mit Sperrhash aus #35, geöffnet über
`mode=ro&immutable=1`, query_only=ON; vor/nach identischer SHA-256. Die drei
früheren Rohdaten-/Metadatenbelege bleiben im [#42-Paket](../empd2_counts_readiness_2026_10_31/evidence.json).
Historische Referenzhashes, Importer und Flags nicht ändern.

Zugangsgrenzen getrennt dokumentiert: Zenodo-Landingpage nicht gelesen;
großer postgres/EMPD2.sql-Inhalt bei begrenzter Connector-Anfrage leer,
nur Blobmetadaten erhalten. Keine Regeln aus diesen ungelesenen Inhalten.
Die verfügbaren kleinen Quelldateien sind belastbare statische Belege,
aber keine nachgewiesene komplette Erzeugungskette des PANGAEA-Exports.

[Dokumentationsprüfung](./verification.json): Locatoren/Versionen/Hashes,
Nennerrollen, Regelstatus, JSON, Links, Originalhash und geschützter Diff.
Keine Modell-/Score-/Matrixauswertung, Simulation oder Tests ausgeführt,
kein unabhängiger menschlicher/palynologischer Review behauptet.
