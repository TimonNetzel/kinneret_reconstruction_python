# Dissertationsnachbau in der App, Stand 10: `reconstruction_climate`, im gebundenen Lauf ohne Aufrufer

Stand 08.11.2026. Bestehender Branch `dissertation-reconstruction`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77). Ausgangscommit `4373353`
(PR-Kopf, `MERGEABLE`); Rechenschritt in `77ccb85`.

## Ergebnis

Neuer letzter `STATES`-Eintrag [reproduction_state.json](reproduction_state.json)
(SHA-256 `329e6a8b2ada52e22be1a12272853091ca4a6c29894581167ae02b0898877f6a`). Quellenlinks zeigen auf `77ccb85`.

Die Stände 1–9 bleiben bytegleich und werden weiter per Hash geprüft. Der Eintrag von
[Stand 9](../app_reproduction_2026_11_08_r_crosscheck_six/README.md) (`50fe8b0a…`) erhielt nur seinen Linkcommit `4373353`.

**Forschungsstand → Dissertationsnachbau** zeigt jetzt:

- **neun** eingefrorene frühere Stände, zuletzt 12 Teilmodule/975 Tests;
- **dreizehn** Teilmodule und **1.037** Teil-/Belegtests. Neu ist „Rezentes Klima aus der ersten Probe (reconstruction_climate)“
  (62 Tests) mit dem Etikett „Bedingter Schritt“. Der Text sagt ausdrücklich, dass der gebundene Lauf diese Funktion nicht aufruft;
- im eingeklappten synthetischen Beispiel die Tabelle **reconstruction_climate** mit den Kunstfällen K1–K8 und dem Hinweis auf den
  einzigen Aufruf in einer nirgends geladenen Datei.

`_climate_example` verlangt Gleichheit mit Referenz und R-Protokoll. Es lehnt zusätzlich jeden Anspruch ab, der Weg sei im gebundenen Lauf aktiv.

## Tatsächlich ausgeführte Prüfungen

[validation.json](validation.json).

- **App-/Daten-/Buildtests:** 108 bestanden, 2 übersprungen (Python-Browserwrapper ohne Playwright, explizit ersetzt).
  10 neue Testfälle: Beispiel, Anspruch auf aktiven Lauf, Bytewechsel an Stand 9 und sieben Belegdateien.
- **Historische Teil-/Belegtests:** 1.037 bestanden, nach den App-Änderungen erneut 1.037. Phase-1-Prüfer Exit 2.
- **Edge offline**, beide Formate × Handy 390 px/PC × Hell/Dunkel × DE/EN, Gesten:
  - geprüft: 36 Links, 16 Tabellen;
  - erster Lauf scheiterte nur an der Fehlstandskontrolle (12 statt 13 Teilmodule); danach alle 10 Fälle ok;
  - keine Tabelle breiter als ihr Rahmen laut Prüfskript. Die Zahlenwerte der Breite sind wegen einer Unterbrechung nicht erhalten.
- **Zwei Standorte** ok. Sichtprüfung: [Mobil hell](mobile.png), [Desktop dunkel](desktop.png).
- **Nicht ausgeführt:** Gesamtsuite, `verify_packet.py`, `tools/verify_*`; Sampler, Ziehungen, Trainings,
  Altersmodelle, Fossilrekonstruktionen.

## Grenzen

Keine historische Fallreproduktion. Der neue Baustein ändert nichts am Nachbau des Laufs. Alle elf Nachweisbündel bleiben offen; keine Variante
gewählt. Baseline, CLI, Artenmodelle, Paperreferenz und Gates sind unverändert. Plots jung links, alt rechts.

Nächste Übergabe: [Claude-Auftrag nach reconstruction_climate](../CLAUDE_NEXT_STEP_AFTER_RECONSTRUCTION_CLIMATE_2026_11_08.md).
