# Nächster konkreter Einreichungs- und Korrelationsbeleg

Stand 30.10.2026. Quellenbedarf, **keine versandte Nachricht**.

Die 35 Plantago-Bilanzen und neun Digitalis-Zahlengleichheiten sind an festen
Diagnosepaaren nachgerechnet. Für eine fachlich begründete Umcodierung fehlt
die Herkunftsdokumentation; die 41 nativen Nichttreffer bleiben bestehen.

Für Neotoma **26608 und 47525** wird benötigt:

1. Die ursprüngliche Einreichung (z. B. Tilia-Datei oder gleichwertige
   Originaltabelle), einschließlich Originalvariablennamen, Einheiten,
   Identifikationsunsicherheit und Regeln zur Aggregation. Konkret: Werden die
   vier alten Plantago-Kategorien in 47525 gemeinsam geführt? Wurde das alte
   Scrophulariaceae-Morphofeld (EPD 763 → 918) als Digitalis-type eingereicht
   oder erst später umcodiert? Von wem, auf welcher Grundlage und in welcher
   Version? Gleiche Werte beantworten diese Fragen nicht.
2. Datensatzbezogene Einreichungs-/Umcodierungsnachweise: `ndb.synonymy` mit
   `datasetid`, `taxonid`, `reftaxonid`, `fromcontributor`, `publicationid`,
   `datesynonymized` und Notizen; `ndb.datasetsubmissions` mit Typ, Datum und
   Notizen; gegebenenfalls zugehörige `ndb.datataxonnotes` an den betroffenen
   Originalmessungen. Originalschlüssel `dataid` und `variableid` mitliefern.
   Der gelesene RDS-Export belegt nicht deren Abwesenheit in der Datenbank.
3. Die unveränderten Originalzählungen und Notizen für die Diagnoseeinträge
   **434216 (Sambucus 2 statt 4), 434416 (Abies 1,5 statt 1) und 434441
   (Abies 0,5 statt 0)**. Die Zahlenreihenfolge ist neu/alt; die alten
   Gegenstücke sind 269988, 270034 und 270046. Entstehung halbzähliger Werte
   erklären, keine bestimmte Erklärung als gegeben voraussetzen.

Die offiziellen Definitionen auf Commit
`359c92c3f0d762f64ea0106c6df597594ba0f541` enthalten diese Metadatenpfade;
[sources.json](./sources.json) nennt die vier gelesenen Dateien. Keine Abfrage
gegen eine Live-Datenbank ausgeführt. Ein vorhandener Datensatzbeleg oder ein
entsprechender Ausschnitt aus einem autorisierten Snapshot wäre ein echter
neuer Eingang; dieselben erfolglosen Downloadrouten erneut aufzurufen reicht
nicht aus. Belegbeschaffung durch Kontaktaufnahme ist nicht autorisiert.

Unabhängig davon fehlt weiterhin die ursprüngliche Abschnitts-/Kompositierungsliste
mit lokalen Tiefen und Einheiten sowie der belegte Anschluss an MON16 mit
Korrelationsunsicherheit. Hierzu gelten der
[Quellenbedarf aus #25](https://github.com/TimonNetzel/kinneret_reconstruction_python/blob/00c9f037642470c7d1a9a44e488f062109292c2d/docs/thesis_research/epd_sample_identity_2026_10_30/MISSING_SOURCE_REQUEST.md)
und die nun archivierten Kandidatennamen aus #26. Keine MON16-Alter übertragen,
keinen Pilot auswählen. P1 bleibt offen.
