# Benötigtes Datenpaket zum Abschluss von P1

Zweck: Die vorhandenen 69 LGM90D-Proben zeitlich nachvollziehbar prüfen,
bevor ein fossiler Klima-Pilot und seine Unsicherheit gerechnet werden.
Dies ist eine Beschaffungs-/Prüfspezifikation; keine Nachricht wurde versandt.

## Erster Eingang: veröffentlichte Original-Supplemente

Bitte die unveränderten Supplementdateien zu **Wulf et al. (2026),
Lago Grande di Monticchio revisited**, DOI
[10.1016/j.quascirev.2026.110116](https://doi.org/10.1016/j.quascirev.2026.110116),
bereitstellen, insbesondere die Tabellen **A-1, A-2 und A-3** und, soweit
enthalten, die Darstellung der tephrenfreien Stratigraphie **C-2**.
Ein vollständiges originales ZIP/XLSX mit allen Anhängen ist geeignet;
einzelne ausgewählte Bildschirmbilder ersetzen die Tabellen nicht.
Die tatsächliche Dateiendung und Spaltenstruktur sind noch nicht verifiziert.

Mit diesen Dateien kann die Prüfung beginnen. Bitte vorhandene Teile schon
bereitstellen; die folgenden Ergänzungen sind keine Voraussetzung dafür,
die veröffentlichten Tabellen überhaupt einzulesen.

## Für eine belastbare Übertragung zusätzlich zu prüfen

| Gegenstand | Benötigte Information | Warum |
|---|---|---|
| Kern-/Probentiefen | Originalproben-ID, LGM90D-Tiefe mit Einheit und Nullpunkt, gegebenenfalls Kernabschnitt | Die 66 spektralen Kandidaten liefern noch keine Originaletiketten. |
| Kompositzuordnung | Dokumentierte Zuordnung von LGM90D zu älteren Kompositprofilen und zu MON16, einschließlich verwendeter Marker und Unsicherheit | Derselbe See oder ähnliche Alterswerte beweisen keine Tiefenidentität. |
| Tephrenfreie Tiefen | Original- und Modell-Tiefe sowie Lage/Mächtigkeit entfernter primärer und umgelagerter Tephren | Die MON16-Modellierung nutzt eine bereinigte Tiefenskala. |
| Modellkonfiguration | Bacon-Eingabe, Einstellungen, verwendete Alters-/Kalibrierkurven, Versionen und gegebenenfalls Reservoirkorrekturen | Ein reproduzierbarer Lauf braucht mehr als eine gezeichnete Alterskurve. |
| Gemeinsame Altersziehungen | Falls vorhanden: originale Modell-/Posteriorausgaben mit zusammengehörigen Kurven und Tiefen | Randgrenzen allein ergeben keine gemeinsame zeitliche Unsicherheit. |
| Übertragene AMS-Anker | Originalmaterial, Fen-Kern-Tiefe und Korrelationsbeziehung für CAMS 12610 und CAMS 9855; Herkunft von ETH-12464 | Übertragene bzw. nicht bestätigte Direktdatierungen dürfen nicht doppelt als unabhängige Anker eingehen. |
| Drei offene Pollenzusammenfassungen | Originaldaten bzw. Bearbeitungsnachweis für LGM90D:000055, :000057, :000067 und die halbzählig angegebenen Summen | Ein Nennerwechsel beseitigt die beobachteten Abweichungen nicht vollständig. |

CSV-Exporte mit erläuterten Spalten sind hilfreich, Originaldateien bleiben
vorrangig. Keine neuen Alter, Proben-IDs oder Zählungen zur Vervollständigung
der Tabellen schätzen. Wenn einzelne Nachweise nicht existieren oder nicht
zugänglich sind, diese Lücke ausdrücklich benennen; die Pilotfreigabe wird
dann auf dieser Grundlage entschieden.

## Annahmeprüfung durch OpenAI

- Quelle, Dateistand, Größe und Prüfsumme dokumentieren; Originale erhalten.
- Tiefen- und Altersmaße prüfen, einschließlich BP-Bezug und Roh-/Komposit-
  bzw. tephrenfreier Tiefe. Keine Einheiten aus bloßen Zahlenbereichen erraten.
- Direkte Datierungen, übertragene Anker und bereits modellierte Alterswerte
  getrennt kennzeichnen. Bekannte gemeinsame Quellenabhängigkeiten erhalten.
- Ergebnisse zunächst als Prüfbericht mit Zuordnungsstatus jeder Probe;
  keine automatische Übernahme neuer Alterswerte in App oder Baseline.

Hintergrund und bereits ausgeführte Kontrollen:
[PR #20](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/20)
und [PR #21](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/21).
