# Dissertationsnachbau in der App: geprüfte Teilmodule, offene Eingaben, Kunstbeispiel

Stand 02.11.2026. Bestehender Branch `dissertation-reconstruction`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
Ausgangscommit `26d65af68b6e7dc142b7af1030fb64223940bd89` (Checkout sauber,
identisch mit GitHub, PR offen und nicht gemergt). Ausschließlich der vom Nutzer
benannte vorhandene Checkout, ohne Klon, Worktree oder Subagenten.
Auftrag: [Claude-Übergabe App-Anbindung](../CLAUDE_NEXT_STEP_APP_REPRODUCTION_2026_11_02.md).

## Ergebnis

Die Ansicht **Forschungsstand** enthält unter dem bisherigen Forschungs-Snapshot
einen eigenen Abschnitt **Dissertationsnachbau / Dissertation reproduction**:

- Status sichtbar als **„Teilmodule geprüft; vollständiger Originalfall offen“**.
  Keine Erfolgsanzeige; der Text sagt ausdrücklich, dass kein Originalfall
  berechnet oder reproduziert ist.
- **Fünf geprüfte Teilmodule** mit Testzahl, R-Quelldatei und Belegelink:
  Alters-/Tiefenübertragung (43), Altersraster/Zeilenauswahl (39),
  Taxongewichtung/PDF-Summe (22), Vergleich kumulierter Rohdichten (43),
  TF-Normierung/Klimazellmaske (51); dazu Verbundprüfung (33) und
  Kinneret-Belegvertrag (20) = **251**.
- **Elf offene Originaleingaben** direkt aus dem unveränderten
  [Phase-1-Belegvertrag](../kinneret_phase1_spec_2026_11_02/README.md)
  (`spec.json`, Reihenfolge und Status übernommen: sieben „fehlt“, vier
  „widersprüchlich“). Verständliche Bezeichnungen statt Variablennamen;
  „Zentral“ markiert finale Basis, finale Chronologie, numerische TFs und
  Laufzeit/Laufidentität. Ein Satz benennt die zentralen Lücken.
- Aufklappbares, standardmäßig eingeklapptes **synthetisches Rechenbeispiel**
  mit deutlicher Warnung (keine Messdaten, keine historischen TFs, keine
  Kalenderalter, keine Rekonstruktion, fließt in keinen Lauf ein):
  TF-Werte vor Normierung `[[2,0,0,0],[0,0,1,2]]`, nach Normierung
  `[[1,0,0,0],[0,0,1/3,2/3]]`, Maske `ja/ja/nein/ja`, nach Maske
  `[[1,0,0,0],[0,0,0,2/3]]` mit Zeilensummen **1** und **2/3**; Hinweis auf die
  fehlende zweite Normierung samt synthetischer Sensitivität (0,3333 / 0,1000);
  normierte PDF der Kunstkette über alle fünf Module (vier Kunstproben, keine
  Alter, Ziehungen 1 und 3 in R-Zählweise); größte Abweichung zur exakten
  Bruchrechnung 2,22 × 10⁻¹⁶ bei Toleranz 2 × 10⁻¹⁴.

Die Werte werden unverändert aus den gespeicherten Ergebnisdateien übernommen,
nicht neu gerechnet; Brüche stammen aus `rational_expectations.json`.
Die Zeilensummen im Leser sind reine Konsistenzkontrollen der gespeicherten Werte.

## Datenbindung

Neue datierte Nachbaustandsdatei [reproduction_state.json](reproduction_state.json)
(SHA-256 `c358b2c3382f568a94d070054ba1dee4b230c617506c9a5015e2cc4506f4e63e`,
im Leser [nachbaustand.py](../../../src/kinneret_recon/nachbaustand.py) fest
verankert). Sie bindet per SHA-256 die sieben Paket-READMEs, `spec.json` sowie
`synthetic_fixture.json`, `synthetic_result.json`, `rational_expectations.json`
und `arithmetic_check.json` des [TF-Pakets](../literal_tf_preparation_2026_11_02/README.md).
Quellenlinks zeigen auf den gepushten Commit `26d65af`.

| Fall | Verhalten |
| --- | --- |
| Nachbaustand oder ein gebundener Beleg fehlt | `available: false` mit Liste fehlender Pfade; App zeigt „kein geprüfter Nachbaustand“ |
| Bytes einer gebundenen Datei geändert | Build bricht ab (`hash mismatch`), nichts wird als geprüft angezeigt |
| Freigabe-/Originaldaten-Flag gesetzt, Variante gewählt, R ausgeführt | Build bricht ab |
| Voraussetzungen im Vertrag ≠ elf bekannte Bündel | Build bricht ab |

Der ältere [Forschungs-Snapshot](../research_snapshot_2026_11_02/README.md),
sein Hashpin, `forschungsstand.py` und alle früheren Übergaben/Belege sind
unverändert. Beide Stände liegen als getrennte Daten vor (`daten/forschung.js`,
`daten/nachbau.js`; in der Ein-Datei-App getrennt eingebettet) und werden
unabhängig voneinander als fehlend angezeigt. Beim App-Build werden keine
Originalcontainer, Modelle, R-Skripte oder Sampler geöffnet, keine `.local`-Pfade
gelesen und keine Hintergrundberechnung gestartet. Der Nachbaustand hängt nicht
von Standort- oder Laufwahl ab.

Geänderte Dateien: `src/kinneret_recon/nachbaustand.py` (neu), `app_bauen.py`,
`app/index.html`, `app/forschung.js`, `app/i18n.js`, `app/style.css`,
`tests/test_nachbaustand.py` (neu), `tools/app_pruefen.js`. Baseline, Kern,
CLI-Befehle, Laufkatalog und Referenzdaten unverändert.

## Prüfung (tatsächlich ausgeführt)

Umgebung: Windows 11, `.venv` Python mit `PYTHONPATH=src`; Node 24 mit dem in
der Codex-Laufzeit gebündelten Playwright (`NODE_PATH`) und installiertem
Microsoft Edge (`--kanal msedge`), offline über `file://`.

- **App/Daten/Build:** `tests/test_nachbaustand.py`, `test_forschungsstand.py`,
  `test_app.py`, `test_artenvergleich.py`: 34 bestanden, 2 übersprungen. Die
  beiden Python-Browserwrapper finden Playwright nicht im Standard-Node-Suchpfad;
  dieselben Prüfungen liefen explizit (unten). 11 davon sind neu: Bindung von
  fünf Modulen/elf Bündeln, unveränderte Beispielwerte und Hashes, fehlender
  Stand/fehlender Beleg sichtbar, sechs Byteänderungs-Kontrollen, Freigabeflag
  trotz passendem Pin abgewiesen, beide Offlineformate und Service Worker.
- **Historische Tests:** die sieben Testmodule des TF-Pakets erneut: **251 bestanden**.
- **Gesamte Testsuite** (`pytest tests`): 424 bestanden, 2 übersprungen,
  4 fehlgeschlagen. `test_formel` (Kodierung des Gradzeichens unter Windows) und
  `test_thesis_import` (SQLite-Dateisperre unter Windows) scheitern identisch am
  Ausgangsstand mit zurückgestellten Änderungen, sind also vorbestehend.
  `test_legacy_equivalence` und ein `test_sampler_validation`-Fall betreffen
  unveränderten Sampler-/C++-Code. Offen benannt: Die Gesamtsuite enthält
  vorhandene Sampler-Unittests auf Kunstzielverteilungen; sie liefen einmal mit,
  erzeugten keine Forschungsergebnisse und wurden bewusst nicht wiederholt.
  Details in [validation.json](validation.json).
- **Browserregression** (`tools/app_pruefen.js`, beide Formate × Handy 390 px /
  PC × Hell/Dunkel, Deutsch und Englisch, plus Touchgesten): alle bestanden.
  Neu geprüft: Status, 5 Module, 11 Voraussetzungen davon 4 widersprüchlich und
  4 zentral, 9 an `26d65af` gebundene Links, Beispiel eingeklappt und nach dem
  Öffnen mit Warnung, vier Tabellen, Summen 1 und 2/3, Zahlen der Kunstkette,
  Aufklappzustand bleibt beim Sprachwechsel, kein Seiten- oder Tabellenüberlauf,
  fehlender Nachbaustand bzw. fehlender Snapshot jeweils unabhängig sichtbar.
  Keine Seitenfehler; Kontexte offline.
- **Zwei Standorte:** bestanden, zusätzlich Forschungs-/Nachbaustand beim
  Standortwechsel textgleich.
- **Phase-1-Prüfer:** `check_kinneret_phase1_spec.py` und `--with-local`
  jeweils erwarteter **Exit 2**. `check_tf_preparation.py` Exit 0 mit
  unveränderten Kennzahlen. `verify_research_handoff.py`: 50 Dateien gültig.
- **Nicht wiederholbar:** `literal_tf_preparation_2026_11_02/verify_packet.py`
  ist ein paketbezogener Erzeugungsprüfer. Er bricht jetzt erwartungsgemäß an
  seiner Bedingung „`src` seit Basis `956951b` unverändert“ ab, weil dieses
  Paket die App ändert. Zuvor schreibt er `preserved_evidence.json` neu; Git
  bestätigt danach bytegleichen Inhalt. Er wurde nicht angepasst und nicht
  erneut ausgeführt.
- Visuell geprüft: [Mobil](mobile.png) und [Desktop dunkel](desktop.png).

Lokale Protokolle, JUnit-Dateien, Vorschauen und Bildschirmfotos:
`.local/app_reproduction_v1/` (nicht versioniert).

```bash
PYTHONPATH=src python -m pytest tests/test_nachbaustand.py tests/test_forschungsstand.py tests/test_app.py tests/test_artenvergleich.py -q
python -m pytest tests/test_historical_tf_preparation.py tests/test_historical_phase1_chain.py tests/test_historical_cdf_discrepancy.py tests/test_historical_age_selection.py tests/test_historical_age_depth.py tests/test_historical_final_pdf.py tests/test_kinneret_phase1_spec.py -q
kinneret-recon app
kinneret-recon app --eine-datei
node tools/app_pruefen.js results/app/index.html results/kinneret-app.html --kanal msedge
python tools/check_kinneret_phase1_spec.py   # erwartet Exit 2
```

## Grenzen

Das ist die erste App-Anbindung des geprüften Nachbaus, **keine historische
Fallreproduktion**. Die 251 Tests prüfen Kunstbeispiele und Belegverträge.
Finale Basis/Listenchronologie, numerische historische TFs, globale Klima-/
Anker-/Referenzstände, Laufzuordnung und Originalergebnisabgleich fehlen weiter;
alle elf Nachweisbündel bleiben offen, keine Variante gewählt. Keine Trainings,
Sampler, Altersmodelle, R-Läufe oder Fossilrekonstruktionen. Baseline, sieben
experimentelle Artenmodelle, Paperreferenz und fallbezogene Monticchio-/
Chronologie-Gates erhalten. Keine neuen Altersplots; jung links, alt rechts bleibt.

Nächste Übergabe: [Claude-Auftrag nach der App-Anbindung](../CLAUDE_NEXT_STEP_AFTER_APP_REPRODUCTION_2026_11_02.md).
