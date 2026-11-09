# Dissertationsnachbau in der App, Stand 12: `reconst` und die Prädiktanten

Stand 09.11.2026. Bestehender Branch `dissertation-reconstruction`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77). Ausgangscommit `38775e1`
(PR-Kopf, `MERGEABLE`); Rechenschritt in `7fb319f`, `3b6f2b3` und `599c6e4`.

## Ergebnis

Neuer letzter `STATES`-Eintrag [reproduction_state.json](reproduction_state.json)
(SHA-256 `ce676ea4122cf237b02d05ad15f2fb7f9663b41b06292aad4666982bbe24bf0d`). Quellenlinks zeigen auf `599c6e4`.

Die Stände 1–11 bleiben bytegleich und werden weiter per Hash geprüft. Der Eintrag von
[Stand 11](../app_reproduction_2026_11_09_local_cor/README.md) (`cbfaf04f…`) erhielt nur seinen Linkcommit `38775e1`, bestätigt mit `git log -1 --format=%H -- …/reproduction_state.json`.

**Forschungsstand → Dissertationsnachbau** zeigt jetzt:

- **elf** eingefrorene frühere Stände, zuletzt 14 Teilmodule/1.172 Tests;
- **fünfzehn** Teilmodule und **1.402** Teil-/Belegtests. Neu ist „Rekonstruktion und Prädiktanten aus vorgegebenem tf_sample“ (230 Tests), nicht bedingt, weil Zeile 1818 in jeder Iteration läuft;
- im eingeklappten synthetischen Beispiel die Tabelle mit den Kunstfällen R1–R8 (Auswahl samt Fall zum Vektor, tann, pann, Rechenweg, ulp) und ein Absatz zu den erreichten und nicht erreichten Lesern.

`_reconst_example` verlangt Gleichheit mit Referenz und R-Protokoll. Es lehnt ab, die Holozänmaske als wirksam oder den 3-Schleifen-Pfad als `long double` auszugeben, oder eine NA-/NaN-/Stopp-Abweichung zu R zu übergehen.

## Tatsächlich ausgeführte Prüfungen

[validation.json](validation.json).

Abschlusslauf am 09.11.2026 nach Neubau beider Formate, ohne Standby während der Läufe. `record_validation.py` bricht ab, sobald ein Exit-Code nicht 0 ist oder eine `FEHLER`- oder Timeout-Zeile im Log steht.

- **App-/Daten-/Buildtests:** 134 bestanden, 2 übersprungen (Python-Browserwrapper ohne Playwright, explizit ersetzt), Exit 0. 15 neue Testfälle.
- **Historische Teil-/Belegtests:** 1.402 bestanden, Exit 0. Phase-1-Prüfer Exit 2.
- **Edge offline**, beide Formate × Handy 390 px/PC × Hell/Dunkel × DE/EN, Gesten: zehn Fälle ok, Exit 0, keine `FEHLER`-Zeile; 42 Links, 18 Tabellen.
- Breiten bei 390 px stehen in [validation.json](validation.json); keine der 18 Tabellen ist breiter als ihr Rahmen.
- **Zwei Standorte** ok, Exit 0. Sichtprüfung: [Mobil hell](mobile.png), [Desktop dunkel](desktop.png).
- **Nicht ausgeführt:** Gesamtsuite, `verify_packet.py`, `tools/verify_*`; Sampler, Ziehungen, Trainings, Altersmodelle, Fossilrekonstruktionen.

## Was die Browserläufe gefunden haben

- **Falsche Sollzahl in der Prüfung.** Der Lauf meldete „synthetisches Beispiel unvollstaendig“. `tools/app_pruefen.js` erwartete 7 NA/NaN-Zellen in der Tabelle R1–R8, gebaut sind 9 (R5: 2, R6: 2, R7: 3, R8: 2). Falsch war die Sollzahl, nicht die Tabelle. Sie steht jetzt auf 9.
- **Tabellenüberlauf bei 390 px.** Der Lauf meldete „1 Tabellen breiter als ihr Rahmen“; es war die neue Tabelle. Sie bricht jetzt um, hat eine kleinere Schrift und engere Innenabstände (`style.css`).
- **Zahlenumbruch bei 390 px.** Die erste Korrektur ließ Zahlen mitten in der Ziffernfolge umbrechen; das zeigte erst das Handyfoto. Werte stehen jetzt in `.nb-wert` ohne Umbruch, frei umbrechen darf nur die Spalte mit dem Rechenweg.
- **Klick-Timeout.** Ein Lauf endete mit Exit 1, aber ohne `FEHLER`-Zeile: `page.click` auf den Reiter „Taxa-Gewichte“ in `pruefeGesten` für `results/kinneret-app.html` lief in das Limit von 30 s.
  - **Ursache:** Der Rechner ging währenddessen in den Standby. Das Windows-Systemprotokoll (Kernel-Power) zeigt Ereignis 506 „Grund: Lid“ um 16:47:25 und 507 um 17:31:31. Der Lauf startete um 16:43:58; ungestörte Läufe dauern 224–225 s, der Standby kam nach 207 s, also im letzten Prüfschritt. Das Log wurde zuletzt um 17:31:35 geschrieben, 4 s nach dem Aufwachen.
  - **Gegenprobe:** Isoliert liefen die Gesten auf der Einzeldatei 5 von 5 Mal ohne Timeout, die Reiterklicks dauerten höchstens 428 ms. Der vollständige Lauf auf demselben Build endete mit Exit 0.
  - **Grenze:** Der Beleg ist die zeitliche Übereinstimmung und das Ausbleiben ohne Standby; ein Standby wurde nicht absichtlich ausgelöst.
  - **Folge:** An App und Prüfskript wurde deswegen nichts geändert. Ein automatischer Wiederholungsversuch hätte echte Hänger verdecken können. Einzelheiten stehen in [validation.json](validation.json).

## Grenzen

Keine historische Fallreproduktion. `tf_sample` stammt im Lauf aus Ziehungen; die Matrizen hier sind erfunden. Alle elf Nachweisbündel bleiben offen; keine Variante
gewählt. Baseline, CLI, Artenmodelle, Paperreferenz und Gates sind unverändert. Plots jung links, alt rechts.

Nächste Übergabe: [Claude-Auftrag nach reconst/predictant](../CLAUDE_NEXT_STEP_AFTER_RECONST_2026_11_09.md).
