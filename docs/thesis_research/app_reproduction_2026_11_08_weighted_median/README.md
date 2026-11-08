# Dissertationsnachbau in der App, Stand 8: gewichteter Median und NA-Maske

Stand 08.11.2026. Bestehender Branch `dissertation-reconstruction`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77). Ausgangscommit `cfd0ad8`
(PR-Kopf, `MERGEABLE`); Rechenschritt in `a21c8f1`.

## Ergebnis

Neuer letzter `STATES`-Eintrag [reproduction_state.json](reproduction_state.json)
(SHA-256 `4d25a3f69f73f5a75340c5e43d4934e94f3faad90609a4f40d9e98bcb4f8c9c6`). Quellenlinks zeigen auf `a21c8f1`.

Die Stände 1–7 bleiben bytegleich und werden weiter per Hash geprüft. Der Eintrag von
[Stand 7](../app_reproduction_2026_11_03_dirichlet_ratio/README.md) (`cf1b77f2…`) erhielt nur seinen Linkcommit
`cfd0ad8`, bestätigt mit `git log -1 --format=%H` auf seiner Zustandsdatei.

**Forschungsstand → Dissertationsnachbau** zeigt jetzt:

- **sieben** eingefrorene frühere Stände, zuletzt 11 Teilmodule/752 Tests;
- **zwölf** Teilmodule und **934** Teil-/Belegtests. Neu ist „Gewichteter Median und NA-Maske der Transferfunktionen“
  (182 Tests, Quellen `mcmc_prepare.R` und matrixStats 0.53.1–1.0.0), ohne Etikett „Bedingter Schritt“;
- in der Zeile R-Gegenprobe zusätzlich: matrixStats 0.56.0, 439 von 439 Werten bitgleich, höchstens 0 ulp;
- im eingeklappten synthetischen Beispiel:
  - Tabelle **Gewichteter Median** mit den Kunstfällen M1–M9 (Eingaben, Ergebnis, Abstand zu R in ulp). NA-Zellen sind
    markiert. Die Bezeichnungen wiederholen keine frühere im Abschnitt;
  - Tabelle **NA-Maske** mit fünf Kunsttaxa: Zeile x1, beide Mediane, entfernt oder nicht. Taxon D hat NA nur in x2 und
    bleibt;
- der Hinweis zur TF-Entfernung nennt jetzt das Teilmodul zum gewichteten Median; die Kunstkette ist entsprechend benannt.

Keine Erfolgsanzeige aus Testzahlen. Kunstbeispiele fließen in keinen Lauf ein; es gab keine echte Transferfunktion.

## Datenbindung

[nachbaustand.py](../../../src/kinneret_recon/nachbaustand.py): `_median_example` verlangt Gleichheit der gespeicherten
Werte mit der unabhängigen Referenz und dem R-Protokoll, lehnt Freigabeflags ab und prüft die Maske gegen Handwert und R.

## Tatsächlich ausgeführte Prüfungen

[validation.json](validation.json).

- **App-/Daten-/Buildtests:** 91 bestanden, 2 übersprungen (Python-Browserwrapper ohne Playwright, explizit ersetzt).
  10 neue Testfälle: Medianbeispiel, zwei Fehlstände, Bytewechsel an Stand 7 und sechs Belegdateien.
- **Historische Teil-/Belegtests:** 934 bestanden, nach den App-Änderungen erneut 934. Phase-1-Prüfer Exit 2.
- **Edge offline**, beide Formate × Handy 390 px/PC × Hell/Dunkel × DE/EN, Gesten:
  - geprüft: 31 Links, 14 Tabellen, R-Zeile, englische Fassung;
  - erster Lauf scheiterte an einer noch alten Erwartung (Fehlstandskontrolle mit 11 statt 12 Teilmodulen) und einer zu breiten Tabelle (Zeilen mit vier
    Einträgen `NaN`, nicht umbrechbar). Danach: `white-space: normal` für die Medianzellen; zweiter und folgender Lauf
    alle 10 Fälle ok;
  - gemessene Tabellenbreite bei 390 px: 297 und 332 von 332 px, keine Seitenverbreiterung.
- **Zwei Standorte** ok. Sichtprüfung: [Mobil hell](mobile.png), [Desktop dunkel](desktop.png).
- **Nicht ausgeführt:** Gesamtsuite, `verify_packet.py`, `tools/verify_*`; Sampler, Ziehungen, Trainings,
  Altersmodelle, Fossilrekonstruktionen.

## Grenzen

Achte App-Anbindung des geprüften Nachbaus, **keine historische Fallreproduktion**.

- Alle elf Nachweisbündel bleiben offen; keine Variante gewählt.
- Paketanbieter sind nur im Zeitfenster gebunden.
- Baseline, CLI, Artenmodelle, Paperreferenz und Gates sind unverändert. Plots jung links, alt rechts.

Nächste Übergabe: [Claude-Auftrag nach dem gewichteten Median](../CLAUDE_NEXT_STEP_AFTER_WEIGHTED_MEDIAN_2026_11_08.md).
