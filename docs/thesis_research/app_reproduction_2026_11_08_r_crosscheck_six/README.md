# Dissertationsnachbau in der App, Stand 9: R-Gegenprobe der sechs älteren Module

Stand 08.11.2026. Bestehender Branch `dissertation-reconstruction`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77). Ausgangscommit `416d547`
(PR-Kopf, `MERGEABLE`); Belegpaket in `8faa035`, Quellenlinks auf `0f044c0`.

## Ergebnis

Neuer letzter `STATES`-Eintrag [reproduction_state.json](reproduction_state.json)
(SHA-256 `50fe8b0a714b4dd7326353ab86e543d3b6650c9e6ae7d21115171c64a96791c2`).

Die Stände 1–8 bleiben bytegleich und werden weiter per Hash geprüft. Der Eintrag von
[Stand 8](../app_reproduction_2026_11_08_weighted_median/README.md) (`4d25a3f6…`) erhielt nur seinen Linkcommit `416d547`.

**Forschungsstand → Dissertationsnachbau** zeigt jetzt:

- **acht** eingefrorene frühere Stände, zuletzt 12 Teilmodule/934 Tests;
- weiterhin zwölf Teilmodule, aber **975** Teil-/Belegtests (41 neue Belegtests der Gegenprobe);
- in der Modulkarte den Abschnitt **R-Gegenprobe der sechs älteren Teilmodule** mit Tabelle je Modul (Werte, bitgleich,
  größte Abweichung in ulp). Die Abweichungen werden angezeigt, nicht in ein Urteil umgedeutet.

Kein neues Rechenmodul, keine neue Kunsttabelle im Beispiel, keine neuen Fallbezeichnungen.

## Datenbindung

[nachbaustand.py](../../../src/kinneret_recon/nachbaustand.py): `_six_check` liest Ergebnis, Ursachen- und Anbieterdatei über
ihre Pins und lehnt nur Freigabeansprüche oder Zufallszustände ab. Abweichungen führen nicht zum Abbruch.

## Tatsächlich ausgeführte Prüfungen

[validation.json](validation.json).

- **App-/Daten-/Buildtests:** 98 bestanden, 2 übersprungen (Python-Browserwrapper ohne Playwright, explizit ersetzt).
  7 neue Testfälle: Gegenprobe, Freigabeanspruch, Bytewechsel an Stand 8 und vier Belegdateien.
- **Historische Teil-/Belegtests:** 975 bestanden, nach den App-Änderungen erneut 975. Phase-1-Prüfer Exit 2.
- **Edge offline**, beide Formate × Handy 390 px/PC × Hell/Dunkel × DE/EN, Gesten: alle 10 Fälle im ersten Lauf ok;
  geprüft 33 Links, 15 Tabellen. Gemessene Breite der neuen Tabelle bei 390 px: 332 von 332 px, keine Seitenverbreiterung.
- **Zwei Standorte** ok. Sichtprüfung: [Mobil hell](mobile.png), [Desktop dunkel](desktop.png).
- **Nicht ausgeführt:** Gesamtsuite, `verify_packet.py`, `tools/verify_*`; Sampler, Ziehungen, Trainings,
  Altersmodelle, Fossilrekonstruktionen.

## Grenzen

Keine historische Fallreproduktion. Alle elf Nachweisbündel bleiben offen; keine Variante gewählt. Paketanbieter sind nur im
Zeitfenster gebunden. Baseline, CLI, Artenmodelle, Paperreferenz und Gates sind unverändert. Plots jung links, alt rechts.

Nächste Übergabe: [Claude-Auftrag nach der Gegenprobe](../CLAUDE_NEXT_STEP_AFTER_SIX_MODULE_CROSSCHECK_2026_11_08.md).
