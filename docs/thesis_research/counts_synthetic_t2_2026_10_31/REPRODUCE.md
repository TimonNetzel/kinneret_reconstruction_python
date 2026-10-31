# Reproduktion ausschließlich des neuen Zählprototyps

Plan: PR #40, Design-Inhalt `62e0ee6`, vollständige Basis `19a7542`.
Quellstand vor erster Ausführung `5b7e67a`; ein Sammlungs-Syntaxfehler im Test
wurde vor erstem Ergebnislauf bei `220ef55` korrigiert. Plan-Dateien unverändert.
Ausgaben mit Python 3.12.14, NumPy 2.5.4; gezielte Tests mit pytest 9.2.0.
Kein GIS, keine Originaldaten und keine bestehenden Modelle nötig.

Ab Repositorywurzel, bewusst in einem frischen Verzeichnis:

```sh
/workspace/venvs/kinneret/bin/python -m pytest -q tests/test_synthetic_counts.py
/workspace/venvs/kinneret/bin/python tools/taxon_counts/run.py --output /tmp/netzel-counts-fresh
/workspace/venvs/kinneret/bin/python tools/taxon_counts/verify.py --output /tmp/netzel-counts-fresh --report /tmp/netzel-counts-fresh-check.json
/workspace/venvs/kinneret/bin/python tools/taxon_counts/check_scope.py
```

Keinen gesamten Repositorytest ausführen; er wäre für diese isolierte Änderung
nicht erforderlich und umfasst fremde Sampler-/App-Fälle. Alte Ergebnisse
nicht neu erzeugen oder ersetzen. Der Prüfer importiert weder model.py/run.py
noch das Anwendungspaket und nutzt direkte Faktoren/Fakultäten statt Log-PMFs.

Fünf Produktionsdateien: observations.json (25 benannte künstliche Datensätze),
results.json (52 gültige Posteriorauswertungen und zwei nicht schätzbare
Negativfälle), posterior.csv.gz (u/v und 52 ID-Spalten, 441 Zeilen),
environment.json und manifest.json. None/null ist fehlend, nicht 0.
Denominator_kind unterscheidet vollständigen Nenner von ausgewähltem Teilnenner.
Ein eventuell gespeichertes generiertes H wird bei der Inferenz nicht als bekannt
benutzt: beide shared-Inferenzvarianten integrieren die festgelegte Verteilung.

Gzip-Zeitstempel 0, 17 signifikante Dezimalstellen für float64-Roundtrip, feste
Reihenfolge. Manifest: drei unveränderte Design-/Plandateien, beide
Produktionsmodule und vier Datenausgaben mit SHA-256/Bytezahl, keine Selbsthashung.
Ausgabeschlüssel outputs/... sind verzeichnisunabhängig. Git-Quellstand und
Dateihashes sichern die Implementierung; environment.json nennt ausdrücklich
Basis- und Designcommit, nicht einen vorgetäuschten Ergebniscodecommit.

repeat_verification.json belegt fünf bytegleiche Dateien zweier frischer Läufe
derselben Umgebung. Andere Versionen können andere Bytes erzeugen; dann separat
numerisch prüfen und vorhandene Hashreferenzen erhalten. verification.json,
tests.json und scope_verification.json dokumentieren Prüfer, gezielte Tests und
geschützte Dateien unabhängig vom Produktionsmanifest. Kein SBC-/Deckungsnachweis
oder unabhängiger menschlicher Review. Die älteren vier großen Bytearchivlücken
werden durch dieses kompakte Git-Paket nicht geschlossen oder neu bewertet.
