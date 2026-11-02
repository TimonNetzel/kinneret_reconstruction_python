# Sieben-Arten-Metriken in der Offline-App

Stand 02.11.2026. Arbeitspaket auf `dissertation-reconstruction` in
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).
Ausgangsstand `2d2fdfd`, sauberer Checkout und identischer GitHub-Head;
PR offen und nicht gemergt. Ausschließlich der vom Nutzer benannte bestehende
Checkout wurde verwendet, ohne Klon oder Worktree.

## Ergebnis

Die neue Ansicht **Artenvergleich** zeigt Balanced Accuracy und Brier gemeinsam:
übersichtliche Tabellen je Art, eine Blockwahl 5°/10° und beide Blockdesigns
nebeneinander im Artenmittel. Standardmäßig sind die Netze ohne und mit SMOTE
sichtbar; Regression #28 und SMOTE mit Prior-Diagnose lassen sich ergänzen.
Der Forschungsstand verlinkt direkt auf die Ansicht. Beide Offlineformate,
Deutsch/Englisch und Hell/Dunkel werden unterstützt. Die Auswahl bleibt nach
Sprach-/Themenwechsel und Neuladen erhalten; Lauf und Standort werden nicht
geändert. Die Tabellenansicht bietet keinen Abbildungsexport an.

Die Kennzahlen stehen je Variante nebeneinander, mit Leserichtung ↑/↓ und
Erklärung von Schwelle, Gewichtung und Testdesign. Im ungewichteten Artenmittel:

| Variante | BA 5° | BA 10° | Brier 5° | Brier 10° |
| --- | ---: | ---: | ---: | ---: |
| Netz ohne SMOTE | 71,64 % | 69,20 % | 0,03975 | 0,04438 |
| Netz mit SMOTE | 90,41 % | 89,74 % | 0,07747 | 0,08245 |

SMOTE verbessert die BA in allen 14 Art×Block-Fällen, verschlechtert zugleich
den Brier in allen 14 Fällen. Es gibt keine automatische Gewinnerwahl.
Die Ausgangswerte und bisherigen Modellparameter bleiben unverändert.

## Datenvertrag und Grenzen

`artenvergleich.py` liest ausschließlich die 56 gespeicherten Zeilen aus
[metrics_summary.csv](../smote_nnet_2026_10_30/metrics_summary.csv), mit fester
SHA256 `93ff7c368c17f39e87e7843a423f0f36bd259cd37a4c9bca7803586f54aa320b`.
Sieben Arten × zwei Blöcke × vier Methoden müssen vollständig und eindeutig
vorliegen; beide Maße müssen endlich sein und zwischen 0 und 1 liegen.
Die acht Übersichtszeilen sind einfache ungewichtete Artenmittel,
keine gepoolten Zellfehler oder Unsicherheitsintervalle.

BA verwendet die ungewichtete Sensitivität/Spezifität bei Schwelle 0,5.
Brier ist flächengewichtet; die normale SMOTE-Zeile zeigt rohe Ausgaben.
Fünf äußere räumliche Folds und 100-km-Trainingspuffer bleiben fest.
Beide Blockdesigns sind keine unabhängigen Replikationsdatensätze.
Meusel/BfN-Kartenzellen und WorldClim 2.1 sind keine unabhängige Feldvalidierung;
kartiertes Außerhalb ist kein erhobener Abwesenheitsnachweis.

Die Prior-Odds-Diagnose ist keine garantierte Kalibrierung. Der Vergleich mit
Regression #28 verändert zusätzlich Modellklasse und Normierung. Die App
kennzeichnet die Ergebnisse als experimentelle Arealindikatoren, keine
validierten fossilen Auftretenswahrscheinlichkeiten. Keine weitere Verwendung
äußerer Testergebnisse zur Optimierung. Keine neue Typ–Art-Zuordnung,
Pilotfreigabe oder wissenschaftliche Änderung.

Ordner-App: `daten/artenvergleich.js`, im Service-Worker-Cache enthalten.
Ein-Datei-App: dieselben Daten eingebettet. Fehlende CSV ergibt einen sichtbaren
Fehlstand ohne alte Tabellen oder aktive Auswahl; veränderte CSV-Bytes stoppen
den Build vor der Veröffentlichung. Die Übersicht und Kennzahlen funktionieren
offline. Sechs Quellenlinks (Bericht, vollständige Metriken, Vorabdesign,
Quellenregister, unabhängige Prüfung und Foldwerte) zeigen auf den festen
Quellcommit `2d2fdfd019d7af09fd55fd8e60e429d1c2b2f7a0` und benötigen Netz.

Keine Trainings, Sampler oder Fossilrekonstruktionen ausgeführt. Baseline,
Referenzdaten, Kern und offene Monticchio-/Chronologie-Gates bleiben erhalten.
Keine Altersplots ergänzt; bisherige Orientierung jung links, alt rechts bewahren.

## Prüfung

- 15 App-/Daten-/Buildtests bestanden. Die zwei vorhandenen Python-Browserwrapper
  überspringen mangels Playwright im normalen Node-Suchpfad; die expliziten
  Prüfungen verwenden gebündeltes Playwright und den installierten Edge.
- Acht gezielte Browserfälle: Ordner/Ein-Datei × Mobil/Desktop × Hell/Dunkel,
  jeweils Deutsch und Englisch, offline. Alle 56 Metrikzeilen (112 Werte) und
  acht Artenmittelzeilen je Sprache direkt gegen die gespeicherte CSV geprüft.
- Block-/Variantenwahl, Rückkehr aus der Rekonstruktionsansicht, Neuladen,
  fehlende Metriken und Wiederherstellung geprüft. Keine Seitenfehler,
  Netzanforderungen oder horizontalen Überläufe der Seite bzw. Wertetabellen.
- Vollständige bestehende Browserregression: acht Fälle und beide Gestenprüfungen
  bestanden, einschließlich bisheriger Diagramme, Sprache und Exporte.
  Zusätzliche Zwei-Standorte-Prüfung bestanden; Artenmetriken bleiben identisch.
- Alle 50 portablen Forschungsnachweise unverändert und gültig.
- Mobil/Desktop-Bilder visuell geprüft: [Mobil](mobile.png), [Desktop](desktop.png).

[Prüfzusammenfassung](validation.json). Vollständige lokale Protokolle und
Offline-Vorschauen liegen unter `.local/app_species_v1/`.

## Wiederholen und nächster Einstieg

```bash
python -m pytest tests/test_artenvergleich.py tests/test_forschungsstand.py tests/test_app.py -q
python tools/verify_research_handoff.py
kinneret-recon app
kinneret-recon app --eine-datei
node tools/artenvergleich_pruefen.js results/app/index.html results/kinneret-app.html --kanal msedge
node tools/app_pruefen.js results/app/index.html results/kinneret-app.html --kanal msedge
```

`--kanal` ist optional; ohne Angabe verwendet Playwright Chromium. Es werden
nur vorhandene App-Daten und Scores gelesen. Bei nicht installiertem Paket
für lokale Prüfungen das bestehende `src`-Verzeichnis in `PYTHONPATH` aufnehmen.

Nächster begrenzter Einstieg: die bereits gespeicherten Foldwerte aus
`scores.csv` als zusätzliche, ausdrücklich deskriptive Testansicht erschließen.
Fehlende Testklassen müssen dabei undefiniert bleiben; Folds oder Blockdesigns
nicht als unabhängige Replikationen oder fossile Unsicherheitsintervalle deuten.
Vor Umsetzung zuerst Schema und Fehlwerte prüfen. Neue Kalibrierung,
Modellentscheidung oder fossile Nutzung benötigen ein eigenes fachliches
Design; dieses Arbeitspaket liefert dafür keine Freigabe.
