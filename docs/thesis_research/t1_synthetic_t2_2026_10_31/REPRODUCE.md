# Reproduktion des synthetischen Pakets

Designsicherung: `2b41c377af45035ec8447bd2b2b435bf55121aac` wurde vor dem
ersten Implementierungs-/Ergebnislauf gepusht. Keine Parameteränderung danach.
Python 3.12.14, NumPy 2.5.4, pytest 9.2.0 in der vorhandenen Umgebung
`/workspace/venvs/kinneret`; genaue Laufversion in outputs/environment.json.
Kein GIS, keine Downloads, keine bestehenden Modelle oder Fossileingänge nötig.

Ab Repositorywurzel, Ausgaben bewusst in einem frischen Verzeichnis:

```sh
/workspace/venvs/kinneret/bin/python -m pytest -q tests/test_synthetic_inversion.py
/workspace/venvs/kinneret/bin/python tools/taxon_synthetic/prototype.py --output /tmp/netzel-t2-new
/workspace/venvs/kinneret/bin/python tools/taxon_synthetic/verify.py --output /tmp/netzel-t2-new --report /tmp/netzel-t2-new-check.json
/workspace/venvs/kinneret/bin/python tools/taxon_synthetic/check_scope.py
```

Tests liegen in tests/, damit die normale CI sie ebenfalls findet. Keine volle
Repository-Prüfung für die Reproduktion erforderlich; sie würde fremde Sampler-
und Browserfälle ausführen. Keine Aufträge an CLI/App, keine realen Kernläufe.

Fünf Produktionsausgaben: observations.json (38 vollständige künstliche Matrizen;
null=fehlend), results.json (108 Auswertungen), posterior.csv.gz (u/v und je eine
Spalte pro Auswertungs-ID; 1.681 Zeilen), environment.json und manifest.json.
Gzip-Zeitstempel 0, 17 signifikante Stellen für float64-Roundtrip, feste Ordnung.
Manifest enthält sieben SHA-256-/Bytebelege: Design, config, Produzent und vier
Datenausgaben. Ausgabe-Schlüssel sind pfadunabhängig als outputs/... benannt.
Manifestwerte beziehen sich auf diese Dateien, nicht auf das Manifest selbst.

Zweiter Lauf im selben Python/NumPy-Stand muss alle fünf Dateibytes reproduzieren.
Andere Bibliotheksversionen können Rundungs-/Gzipbytes ändern; dann getrennt
numerisch prüfen und die Abweichung berichten, vorhandene Referenzhashes erhalten.
outputs/environment.json nennt den **Designcommit**, nicht einen vorgetäuschten
Codecommit; exakter ausführbarer Code und Eingänge werden mit demselben PR
gesichert. Der Produzentenhash steht im Manifest.

verify.py importiert prototype.py nicht: Direktprodukte endlicher Bernoulli-
Wahrscheinlichkeiten, Mischung gemeinsamer H pro Zeile, anschließende Normierung
über das ganze Gitter. 40 Replikate sind hier klein genug für diese direkte
Nachrechnung ohne Unterlauf; das ist kein allgemeiner Ersatz für die stabile
Log-Auswertung des Produzenten. Beide teilen Python/NumPy und das Design.
Latente Z/H werden zusätzlich in den Tests explizit enumeriert.

Der einmalige Bytevergleich ist in repeat_verification.json gesichert.
verification.json, TESTING.md und preservation.json halten Prüfergebnis,
Abbruch/Umgebungsfehler und Erhaltungsprüfung getrennt von Produktionsdateien.
Alle Ergebnisdateien werden mit dem PR gepusht. Temporäre Vergleichsverzeichnisse
sind kein behauptetes Dauerarchiv; keine bestehenden Artefaktregister ändern.
