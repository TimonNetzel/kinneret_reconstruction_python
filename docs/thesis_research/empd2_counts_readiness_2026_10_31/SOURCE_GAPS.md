# Genau nächster Quellenbedarf nach dem EMPD2-Readiness-Audit

Das lesende Schema-/Dreiproben-Audit ist abgeschlossen. Es belegt vorhandene
numerische Pollenwerte und Metadaten, **keine Freigabe eines Counts-Trainingssatzes**.
Die [Belege](./evidence.json) und der [Vertrag](./INPUT_CONTRACT.json) sind gesichert.
Keine neuen Kohorten, Taxonpools oder Modellarchitekturen sind gewählt.

## Nächstes empfohlenes Paket – vorbereitet, wartet auf Nutzer-OK

**EMPD2s ursprüngliche Nenner- und Sparse-Verarbeitungsregeln gezielt belegen.**

Der Engpass ist jetzt konkret: Was bedeuten `groupid.used_in_sum`,
`percent_values` und `higher_groupid` in der **tatsächlich angewandten** Verarbeitung,
und sind nicht gespeicherte `(samplename,var_)`-Paare belegte Nullen oder
unaufgelöste Mess-/Erfassungslücken? Was ist die Grundlage eines originalen N
im Verhältnis zu einer später bewusst ausgewählten bedingten Komposition?

- **Erste Originalquelle:** die bereits im Repository registrierte EMPD2-
  Methodenpublikation Davis et al. (2020), DOI
  [10.5194/essd-12-2423-2020](https://doi.org/10.5194/essd-12-2423-2020),
  mit Datenbeschreibung/Erfassungs-/Harmonisierungs- und Berechnungsregeln.
  Diese DOI ist ein bestehender Quellenhinweis; der Volltext wurde in diesem
  Audit nicht neu gelesen. Nicht aus Abstract oder Titel auf Regeln schließen.
- **Zweites Objekt nur bei konkretem Verweis:** der darin oder im offiziellen
  Datenarchiv bezeichnete ursprüngliche Verarbeitungscode bzw. ein Datenwörterbuch.
  Erst genaue URL/Version/Datei und Anwendung belegen; kein beliebiges späteres
  Paket als damals verwendet annehmen. Kein unbefristetes Suchen nach Ersatzdaten.
- **Abgrenzung:** genau diese Regelkette lesen; Klima-Periodenangabe nur dann
  mit sichern, wenn dieselbe gelesene Originalstelle sie explizit enthält.
  Keine komplette neue Taxoninventur, Matrixauswertung, neue Stichprobenauswahl,
  Fits, Scores oder Kernel-/App-Läufe.
- **Vorhandene Zeugen verwenden:** AlbaSanchez_a1 (20 Ganzzahlwerte),
  Champvillair_a18 (Volumen 7,4 außerhalb flag-1-Gruppen) und Barboni_a1000
  (Text-NULL/X). Höchstens diese drei Proben und die gesperrte Originaldatei
  zur Kontrolle der dokumentierten Regeln lesend verwenden; keine Ergebnisse
  als vollständige Datenbereinigung exportieren.
- **Abschluss:** Seite/Abschnitt bzw. Code-Datei/Funktion/Zeile/Version für jede
  Regel; Quelle → dokumentierte Regel → Anwendbarkeit auf konkrete Felder
  getrennt belegen. `N_original`, arithmetische gespeicherte Summe und
  `n_ausgewählt` explizit unterscheiden. Nichtbeleg, Zugangssperre oder abweichende
  Regeln vollständig erhalten. Nur tatsächlich belegte Null-/Nennerbedeutung
  als geklärt melden; keine automatische Count-Likelihood-Freigabe.
- **Ausgaben:** neues DENOMINATOR_RULES.md, kompaktes RULE_EVIDENCE.json und
  verbleibender SOURCE_GAPS.md; nur neue Dokumente plus Status/Übergabe,
  eigener Worktree/Branch/PR ohne Merge. Bestehende Register/Importer/Flags,
  Modelle/Resultate/Counts/Gruppen, Baseline und CLI/App unverändert.

Ein weiteres OK beauftragt dieses enge Quellenpaket. Fehlender Zugriff würde als
Zugangsblock dokumentiert; er wird nicht durch eine vermutete Regel ersetzt.
Kein Autorenkontakt, keine Unteragenten, kein gesamter Repositorytest.

## Weitere Grenzen – gesondert offen, kein zusätzlicher Folgeauftrag

Die Datensatzbeziehung zur fossilen Methodik von W96 bleibt blockiert
([konkreter Bestimmungsbedarf](../pollen_methods_2026_10_31/SOURCE_GAP.md)).
Reales T1, T0/T3/T4 und Monticchio-P1 sind nicht freigegeben.
Eine direkte moderne Typkompositions-Kalibrierung wäre ein separates Modell-
und Datenrollendesign nach ausreichender Quellenklärung; sie ersetzt nicht
stillschweigend Arealfunktionen oder identifiziert Produktion/Transport einzeln.
Standort-/Zeit-/Klimaprovenienz, unabhängige Bestätigung und Taxonomie bleiben
weitere explizite Voraussetzungen. Kein verlässlicher Termin für einen realen
Kernlauf aus diesem Audit abgeleitet.
