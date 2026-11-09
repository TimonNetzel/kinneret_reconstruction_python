# Dissertationsnachbau in der App, Stand 14: `ratios_tfs_no_ages` (eigene Normalverteilung, nicht bitgleich zu R)

Stand 09.11.2026. Bestehender Branch `dissertation-reconstruction`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77). Ausgangscommit und Rechenschritt `9f6165e`.

## Ergebnis

Neuer letzter `STATES`-Eintrag [reproduction_state.json](reproduction_state.json)
(SHA-256 `a45d172c117805bc5d7e75a25ae22606650552aa0a8a73f04a4e706f76226230`). Quellenlinks zeigen auf `9f6165e`.

Die Stände 1–13 bleiben als Zustandsdateien bytegleich und werden weiter per Hash geprüft. Der Eintrag von
[Stand 13](../app_reproduction_2026_11_09_wt_power/README.md) (`d84fe644…`) erhielt nur seinen Linkcommit `52a6acc`.

**Forschungsstand → Dissertationsnachbau** zeigt jetzt:

- **dreizehn** eingefrorene frühere Stände, zuletzt 16 Teilmodule/1.614 Tests;
- **siebzehn** Teilmodule und **1.858** Teil-/Belegtests. Neu ist „Quotienten der Transferfunktionen“ (212 Tests), nicht bedingt, weil `mcmc_conduction.R:228` ohne Bedingung läuft. Die 32 Tests der [rtruncnorm-Bindung](../literal_rtruncnorm_binding_2026_11_09/README.md) sind mitgezählt und verlinkt;
- im eingeklappten synthetischen Beispiel die Tabelle mit den Kunstfällen Q1–Q8, einen Absatz „Nicht bitgleich zu R“ mit den Messzahlen, einen Absatz zur Erreichbarkeit und den Bindungsnachweis.

Die App stellt den Baustein nicht als bitgleich zu R dar. `_ratios_example` lehnt ab, Bitgleichheit, einen Lauf der C++-Zeilen, eine Ziehung oder eine Abweichung im exakten Teil auszugeben.

| Vergleich mit R 3.5.0, 120 Stressfälle | bitgleich | höchstens |
|---|---:|---:|
| Gitterquotienten, 3.406 Werte | alle | 0 ulp |
| Produkt mit den Faktoren aus R, 120 Werte | alle | 0 ulp |
| Dichtequotienten, 3.406 Werte | 2.533 | 15 ulp, relativ 3,3e-15 |
| `transfer_ratio`, 120 Werte | 39 | 35 ulp, relativ 4,5e-15 |

Herkunft: [Rechenschritt](../literal_ratios_tfs_2026_11_09/README.md). R 3.5.0 führt die R-Fassung derselben Funktion wörtlich aus; die C++-Zeilen selbst konnten hier ohne Compiler nicht laufen.

## Tatsächlich ausgeführte Prüfungen

[validation.json](validation.json).

Abschlusslauf am 09.11.2026 nach Neubau beider Formate. Das Windows-Systemprotokoll zeigt von 19:05 bis zum Ende der Läufe keine Kernel-Power-Ereignisse 506/507, also keinen Standby.
`record_validation.py` bricht ab, sobald ein Exit-Code nicht 0 ist, eine `FEHLER`- oder Timeout-Zeile im Log steht, eine Tabelle breiter als ihr Rahmen ist oder eine Testzahl, die Zahl der Fälle, ein Hash oder die Standby-Abfrage nicht stimmt.

- **App-/Daten-/Buildtests:** 172 bestanden, 2 übersprungen (Python-Browserwrapper ohne Playwright, explizit ersetzt), Exit 0. 21 neue Testfälle.
- **Historische Teil-/Belegtests:** 1.858 bestanden in 22 Dateien, Exit 0. Phase-1-Prüfer Exit 2.
- **Edge offline**, beide Formate × Handy 390 px/PC × Hell/Dunkel × DE/EN, Gesten: zehn Fälle ok, Exit 0, keine `FEHLER`-Zeile; 49 Links, 20 Tabellen, 17 Teilmodule, 13 frühere Stände.
- Breiten bei 390 px stehen in [validation.json](validation.json); keine der 20 Tabellen ist breiter als ihr Rahmen.
- **Zwei Standorte** ok, Exit 0. Sichtprüfung: [Mobil hell](mobile.png), [Desktop dunkel](desktop.png).
- **Nicht ausgeführt:** Kompilierung der C++-Zeilen, Gesamtsuite, `verify_packet.py`, `tools/verify_*`; Sampler, Ziehungen, Trainings, Altersmodelle, Fossilrekonstruktionen.

## Was beim Einbau gefunden wurde

**Zwei falsche Beschriftungen in der laufenden App, behoben.** In `i18n.js` waren zwei Schlüssel je Sprache doppelt definiert, und der spätere Eintrag gewann:

- `nb_ex_w_case`: Die Gewichtstabelle zeigte seit Stand 13 W1–W3 statt G1–G3, weil die neue Tabelle W1–W8 denselben Schlüssel nutzte.
- `nb_ex_m_x2`: Die Tabelle der fehlenden Transferfunktionen zeigte seit Stand 8 „Median x2“ statt „NA nur x2“.

Die Schlüssel heißen jetzt `nb_ex_wt_case` und `nb_ex_m_only_x2`. Die Browserprüfung kontrolliert beide Beschriftungen, und `test_every_language_key_is_defined_once` schließt Doppelungen aus. Betroffen war nur die Anzeige, keine Zahl; die eingefrorenen Zustandsdateien sind unverändert.

## Grenzen

Keine historische Fallreproduktion. Gitter, Dichten und Vorschläge sind erfunden. Alle elf Nachweisbündel bleiben offen; keine Variante gewählt. Baseline, CLI, Artenmodelle, Paperreferenz und Gates sind unverändert. Plots jung links, alt rechts.

Nächste Übergabe: [Claude-Auftrag nach den Transferquotienten](../CLAUDE_NEXT_STEP_AFTER_RATIOS_TFS_2026_11_09.md).
