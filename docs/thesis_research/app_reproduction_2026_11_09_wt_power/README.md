# Dissertationsnachbau in der App, Stand 13: `wt_power` (NumPy-FFT, nicht bitgleich zu R)

Stand 09.11.2026. Bestehender Branch `dissertation-reconstruction`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77). Ausgangscommit `ddbce9e`
(PR-Kopf, `MERGEABLE`); der Rechenschritt liegt ebenfalls in `ddbce9e`.

## Ergebnis

Neuer letzter `STATES`-Eintrag [reproduction_state.json](reproduction_state.json)
(SHA-256 `d84fe644553c48325b53dbf29a7ac7fbe8fcfa040d53e6d8f5358ac45851f945`). Quellenlinks zeigen auf `ddbce9e`.

Die Stände 1–12 bleiben bytegleich und werden weiter per Hash geprüft. Der Eintrag von
[Stand 12](../app_reproduction_2026_11_09_reconst_predictant/README.md) (`ce676ea4…`) erhielt nur seinen Linkcommit `9afab2e`.

**Forschungsstand → Dissertationsnachbau** zeigt jetzt:

- **zwölf** eingefrorene frühere Stände, zuletzt 15 Teilmodule/1.402 Tests;
- **sechzehn** Teilmodule und **1.614** Teil-/Belegtests. Neu ist `wt_power` (212 Tests), als bedingter Schritt markiert, weil für Kinneret offen ist, ob es läuft;
- im eingeklappten synthetischen Beispiel die Tabelle mit den Kunstfällen W1–W8, einen eigenen Absatz „Nicht bitgleich zu R“ mit den Messzahlen und einen Absatz zur Erreichbarkeit.

Die App stellt `wt_power` nicht als bitgleich zu R dar. Die Nutzerentscheidung vom 09.11.2026 (NumPy-FFT mit gemessener und markierter Abweichung, keine Übersetzung von R-`fft.c`) steht im Text der Ansicht.

| Vergleich mit R 3.5.0, 120 Stressfälle | bitgleich | höchstens |
|---|---:|---:|
| zentrierte Reihe, 22.256 Werte | alle | 0 ulp |
| Leistung, 39.876 Werte | 5.938 | 606 ulp, relativ zum Spaltenmaximum 2,5e-15 |
| `post_spec`, sechs Kunstketten | – | 12 ulp |

Keine Token-Abweichung. Herkunft: [Rechenschritt](../literal_wt_power_2026_11_09/README.md), [R-Protokoll](../literal_wt_power_2026_11_09/r_crosscheck_result.json).

## Tatsächlich ausgeführte Prüfungen

[validation.json](validation.json).

Abschlusslauf am 09.11.2026 nach Neubau beider Formate. Das Windows-Systemprotokoll zeigt von 18:25 bis zum Ende der Läufe keine Kernel-Power-Ereignisse 506/507, also keinen Standby.
`record_validation.py` bricht ab, sobald ein Exit-Code nicht 0 ist, eine `FEHLER`- oder Timeout-Zeile im Log steht, eine Tabelle breiter als ihr Rahmen ist oder eine Testzahl, die Zahl der Fälle, ein Hash oder die Standby-Abfrage nicht stimmt.

- **App-/Daten-/Buildtests:** 151 bestanden, 2 übersprungen (Python-Browserwrapper ohne Playwright, explizit ersetzt), Exit 0. 17 neue Testfälle.
- **Historische Teil-/Belegtests:** 1.614 bestanden in 20 Dateien, Exit 0. Phase-1-Prüfer Exit 2.
- **Edge offline**, beide Formate × Handy 390 px/PC × Hell/Dunkel × DE/EN, Gesten: zehn Fälle ok, Exit 0, keine `FEHLER`-Zeile. Die Sollzahlen 45 Links, 19 Tabellen, 16 Teilmodule und 12 frühere Stände sind damit in einem vollständigen Lauf bestätigt.
- Breiten bei 390 px stehen in [validation.json](validation.json); keine der 19 Tabellen ist breiter als ihr Rahmen.
- **Zwei Standorte** ok, Exit 0. Sichtprüfung: [Mobil hell](mobile.png), [Desktop dunkel](desktop.png).
- **Nicht ausgeführt:** Gesamtsuite, `verify_packet.py`, `tools/verify_*`; Sampler, Ziehungen, Trainings, Altersmodelle, Fossilrekonstruktionen.

## Was die Prüfung gefunden hat

- **Kein Fehler in App oder Prüfskript.** Beides blieb gegenüber dem eingebauten Stand unverändert. Der frühere, abgebrochene Browserlauf (bis 5 Fälle) zählt nicht als bestanden; maßgeblich ist der vollständige Lauf oben.
- **Sichtbar in W8:** Die Zeile zeigt `NaN` für das Modul und `NA` für R. W8 mischt `NA` und `NaN` in der Eingabe (`kind` `mixed`). Dafür beansprucht das Modul nur „fehlend“, nicht ein bestimmtes Token ([Rechenschritt](../literal_wt_power_2026_11_09/README.md)). Der Zellenunterschied ist also keine Abweichung, die das Modul verneint. Die Tabelle erklärt den Mischfall allerdings nicht eigens, nur das Datenattribut der Zeile nennt ihn.

## Grenzen

Keine historische Fallreproduktion. Reihen und Bänder sind erfunden, `wb` ist keine biwavelet-Ausgabe. Die Abweichung ist auf Kunstfällen mit `n` bis 512 gemessen und gilt für R 3.5.0 x64 unter Windows und NumPy 2.5.4.
Für Kinneret bleibt offen, ob `wt_power` läuft. Alle elf Nachweisbündel bleiben offen; keine Variante gewählt. Baseline, CLI, Artenmodelle, Paperreferenz und Gates sind unverändert. Plots jung links, alt rechts.

Nächste Übergabe: [Claude-Auftrag nach wt_power](../CLAUDE_NEXT_STEP_AFTER_WT_POWER_2026_11_09.md).
