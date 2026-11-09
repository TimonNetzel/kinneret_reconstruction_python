# Dissertationsnachbau in der App, Stand 11: lokales `cor` und seine Zeitraum-Leser

Stand 09.11.2026. Bestehender Branch `dissertation-reconstruction`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77). Ausgangscommit `437f6f4`
(PR-Kopf, `MERGEABLE`); Rechenschritt in `6caf224` und `5af02eb`.

## Ergebnis

Neuer letzter `STATES`-Eintrag [reproduction_state.json](reproduction_state.json)
(SHA-256 `cbfaf04f6bdbf36eb8fff76dac92e2a1a4d19df15f6d607a56e510dcd71e73af`). Quellenlinks zeigen auf `5af02eb`.

Die Stände 1–10 bleiben bytegleich und werden weiter per Hash geprüft. Der Eintrag von
[Stand 10](../app_reproduction_2026_11_08_reconstruction_climate/README.md) (`329e6a8b…`) erhielt nur seinen Linkcommit `437f6f4`.

**Forschungsstand → Dissertationsnachbau** zeigt jetzt:

- **zehn** eingefrorene frühere Stände, zuletzt 13 Teilmodule/1.037 Tests;
- **vierzehn** Teilmodule und **1.172** Teil-/Belegtests. Neu ist „Lokales cor und seine Zeitraum-Leser“ (135 Tests), nicht bedingt, weil `cor` für Kinneret in jedem Fall aufgerufen wird (Zeile 1964/1966 oder 1993);
- im eingeklappten synthetischen Beispiel die Tabelle **Lokales cor** mit den Kunstfällen C1–C8, dem Hinweis auf die Zweige und die unerreichte Zeile 1999.

`_cor_example` verlangt Gleichheit mit Referenz und R-Protokoll. Es lehnt jeden Anspruch ab, Zeile 1999 werde erreicht oder `multiple` sei für Kinneret nicht `sources$core`.

## Tatsächlich ausgeführte Prüfungen

[validation.json](validation.json).

- **App-/Daten-/Buildtests:** 119 bestanden, 2 übersprungen (Python-Browserwrapper ohne Playwright, explizit ersetzt).
  11 neue Testfälle: Beispiel, zwei abgelehnte Ansprüche, Bytewechsel an Stand 10 und sieben Belegdateien.
- **Historische Teil-/Belegtests:** 1.172 bestanden, nach den App-Änderungen erneut 1.172. Phase-1-Prüfer Exit 2.
- **Edge offline**, beide Formate × Handy 390 px/PC × Hell/Dunkel × DE/EN, Gesten: 39 Links, 17 Tabellen.
  - Der erste Lauf zeigte die neue Tabelle bei 390 px breiter als ihren Rahmen (527 statt 332 px). Behoben durch Umbruch und eine gemeinsame Spalte für x und y; danach alle Fälle ok.
  - Breiten bei 390 px stehen in [validation.json](validation.json); keine der 17 Tabellen ist breiter als ihr Rahmen.
- **Zwei Standorte** ok. Sichtprüfung: [Mobil hell](mobile.png), [Desktop dunkel](desktop.png).
- **Nicht ausgeführt:** Gesamtsuite, `verify_packet.py`, `tools/verify_*`; Sampler, Ziehungen, Trainings, Altersmodelle, Fossilrekonstruktionen.

## Grenzen

Keine historische Fallreproduktion. Die Vektoren stammen im Lauf aus Ziehungen; hier sind sie erfunden. Alle elf Nachweisbündel bleiben offen; keine Variante
gewählt. Baseline, CLI, Artenmodelle, Paperreferenz und Gates sind unverändert. Plots jung links, alt rechts.

Nächste Übergabe: [Claude-Auftrag nach dem lokalen cor](../CLAUDE_NEXT_STEP_AFTER_LOCAL_COR_2026_11_09.md).
