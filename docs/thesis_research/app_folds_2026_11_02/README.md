# Gespeicherte äußere Foldwerte in der App

> **Neue Priorität, 02.11.2026:** Der unten vorgeschlagene nächste
> Reliabilitätsschritt ist nachgeordnet. Zuerst die lokalen Rekonstruktionen
> der Dissertation nachbauen, danach Verbesserungen:
> [aktueller Arbeitsplan](../LOCAL_DISS_RECONSTRUCTION_PLAN.md).
> Die dokumentierte App-Umsetzung und Prüfung bleiben unverändert gültig.


Stand 02.11.2026. Fortsetzung auf `dissertation-reconstruction` in
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).
Ausgangsstand `2f6d0eb`, Checkout sauber und mit GitHub identisch;
PR offen und nicht gemergt. Ausschließlich der vorhandene, vom Nutzer benannte
Checkout wurde verwendet, ohne Klon, Worktree oder Subagenten.

## Ergebnis

Der **Artenvergleich** bietet jetzt die Umschaltung **Übersicht / Foldwerte**.
Die Detailansicht zeigt für eine gewählte Art und 5°-/10°-Blöcke alle fünf
äußeren Testfolds: Balanced Accuracy und Brier nebeneinander, ursprüngliche
Testzellzahl, Präsenzen und kartierte Außerhalb-Zellen. Netze ohne/mit SMOTE
sind standardmäßig sichtbar; Regression #28 und Prior-Diagnose ergänzbar.
Anzeige Fold 1–5 entspricht den unveränderten Quellkennungen 0–4.

Art, Block, Ansicht und Variantenwahl bleiben nach Sprach-/Themenwechsel und
Neuladen erhalten. Die Darstellung ist unabhängig von Rekonstruktionslauf und
Standort, offline in beiden App-Formaten und auf Deutsch/Englisch nutzbar.
Die vorhandene Gesamtübersicht bleibt erhalten. Ihre Aussage zum BA-Gewinn
in allen 14 Art×Block-Fällen ist jetzt ausdrücklich als Gesamtauswertung
beschriftet; sie ist keine Aussage über jeden einzelnen Fold.

## Quellenbindung und Schema

Der App-Leser bindet zusätzlich die unveränderte
[scores.csv](../smote_nnet_2026_10_30/scores.csv) mit SHA256
`69dc8e6e4da6f057f643889edb9334f94bac4f4f5751ae59a5e7134aa7f8c538`.
Die 336 Quellzeilen bestehen aus 280 Foldzeilen
(sieben Arten × zwei Blöcke × vier Methoden × fünf Folds) und 56 Gesamtzeilen.
Die vollständige Kombination muss eindeutig vorliegen. Gesamtmetriken müssen
exakt mit der bereits gebundenen `metrics_summary.csv` übereinstimmen.
Testzellzahlen und Präsenzen sind zwischen Varianten identisch; die Summe
über fünf Testfolds stimmt mit dem Gesamtbestand überein. BA und Brier werden
unverändert übernommen, nicht neu gerechnet. Keine Foldmittel oder Intervalle
werden hinzugefügt.

**Tatsächlicher Befund:** Alle 280 gespeicherten Foldzeilen enthalten beide
Testklassen und definierte BA-/Brier-Werte. Bei seltenen Arten kann die Zahl
positiver Testzellen dennoch sehr klein sein (Abies alba, 10°, erster Fold:
eine Präsenz unter 5.961 Testzellen). Der Brier ist flächengewichtet;
die Zellzahlen in der Beschriftung sind ungewichtete Anzahlen.

Fehlt die Folddatei, bleibt die Gesamtübersicht verfügbar; die Detailansicht
zeigt einen klaren Fehlstand. Geänderte Bytes brechen den Build ab.
Hypothetische fehlende Testklassen sind ausschließlich Negativkontrollen in
Tests: Eine leere BA bleibt `null` und wird als **Nicht definiert / Undefined**
mit Begründung angezeigt, niemals als 0 oder 50 %. Der Brier bleibt getrennt
angezeigt. Originaldaten werden für diese Kontrollen nicht geändert.

Die neuen Folddaten werden zusammen mit den Gesamtmetriken in
`daten/artenvergleich.js` gespeichert und gecacht bzw. in die Ein-Datei-App
eingebettet. Der vorhandene Fold-Quellenlink zeigt auf den festen Quellcommit
`2d2fdfd`; alle sechs bisherigen Quellenlinks bleiben erhalten. Die Kennzahlen
funktionieren offline, die GitHub-Quellenlinks benötigen Netz.

## Grenzen

Folds teilen sich Trainingsdaten und sind keine unabhängigen Wiederholungen.
Ihre Streuung ist kein Konfidenzintervall und kein fossiles Unsicherheitsband.
Die Gesamtmetriken beruhen auf allen zurückgehaltenen Testzellen, nicht auf
einem einfachen Foldmittel. Keine Auswahl oder Optimierung anhand äußerer
Testergebnisse. SMOTE-Fälle bleiben ausschließlich im Training; die Ansicht
zeigt ursprüngliche Testzellen. Experimentelle Arealindikatoren sind keine
validierten fossilen Auftretenswahrscheinlichkeiten. Die Prior-Odds-Diagnose
ist keine garantierte Kalibrierung.

Keine Trainings, Sampler oder Fossilrekonstruktionen ausgeführt. Baseline,
Kern, Referenzdaten und offene Monticchio-/Chronologie-Gates bleiben erhalten.
Keine Altersplots ergänzt; bestehende Orientierung jung links, alt rechts bewahren.

## Prüfung

- 23 App-/Daten-/Buildtests bestanden; zwei vorhandene Python-Browserwrapper
  wegen fehlendem Playwright im Standard-Node-Suchpfad übersprungen und durch
  explizite Prüfungen mit gebündeltem Playwright und installiertem Edge ergänzt.
- Nach Vereinfachung der Testfixtures alle fünf betroffenen Negativkontrollen
  erneut bestanden: fehlende positive/negative Klasse, Duplikat, widersprüchliche
  Gesamtmetriken und nicht endlicher Brier. Keine Rohquellen geändert.
- Acht gezielte Offline-Browserfälle: beide Formate × Mobil/Desktop × Hell/Dunkel,
  jeweils Deutsch/Englisch. Alle 280 Foldzeilen (560 Metrikwerte), Zellzahlen,
  56 Gesamtzeilen und acht Artenmittel gegen die Original-CSVs geprüft.
- Art-/Block-/Variantenwahl, Neuladen, Sprach-/Themenwechsel, fehlende Gesamt-
  und Folddaten, Wiederherstellung und hypothetisch undefinierte BA bestanden.
  Keine Seitenfehler, Netzanforderungen oder horizontalen Tabellenüberläufe.
- Zwei-Standorte-Test bestanden; Gesamt- und Foldansicht bleiben identisch.
- Vollständige Browserregression: acht Durchläufe und beide Gestenprüfungen
  bestanden, einschließlich bisheriger Diagramme, Sprache und Exporte.
- Alle 50 portablen Forschungsnachweise unverändert und gültig.
- [Mobil](mobile.png) und [Desktop](desktop.png) visuell geprüft.

[Prüfzusammenfassung](validation.json). Lokale Protokolle, Testfixtures und
beide Offline-Vorschauen liegen unter `.local/app_folds_v1/`.

## Wiederholen und nächster Einstieg

```bash
python -m pytest tests/test_artenvergleich.py tests/test_forschungsstand.py tests/test_app.py -q
python tools/verify_research_handoff.py
kinneret-recon app
kinneret-recon app --eine-datei
node tools/artenvergleich_pruefen.js results/app/index.html results/kinneret-app.html --kanal msedge
node tools/app_pruefen.js results/app/index.html results/kinneret-app.html --kanal msedge
```

Ohne `--kanal` verwendet Playwright Chromium. Bei nicht installiertem Paket
für lokale Prüfungen das bestehende `src` in `PYTHONPATH` aufnehmen.

Nächster begrenzter Einstieg: Schema, Gewichtung und leere Bins der bereits
gespeicherten `reliability.csv` prüfen und daraus eine lesende diagnostische
Ansicht planen. Keine neue Kalibrierung oder Klassenprior-Korrektur ableiten.
Äußere Testergebnisse nicht zur Optimierung verwenden; neue Kalibrierung oder
fossile Anwendung benötigen ein eigenes fachliches Design.
