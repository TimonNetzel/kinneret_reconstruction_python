# Dissertationsnachbau in der App, Stand 15: Eingabevertrag des Kinneret-Falls

Stand 09.11.2026. Bestehender Branch `dissertation-reconstruction`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77). Ausgangscommit und Belegpaket `daaa2dc`.

## Warum ein neuer Stand

Die App nannte bisher nur die elf offenen Nachweisbündel. Der [Eingabevertrag](../kinneret_input_contract_2026_11_09/README.md) zeigt zusätzlich, was der erhaltene Code für Kinneret liest, in welcher Datei und Zeile, und wie weit jede Eingabe gebunden ist. Damit sieht man in der App, woran der Originalfall konkret hängt. Das ist der Mehrwert. Es kommt kein Rechenbaustein dazu, und die App rechnet nichts Neues.

## Ergebnis

Neuer letzter `STATES`-Eintrag [reproduction_state.json](reproduction_state.json)
(SHA-256 `0706f10d158408ccb8532ed00997510baaf6fda2e2b9975fe05967182a494333`). Quellenlinks zeigen auf `daaa2dc`.

Die Stände 1–14 bleiben als Zustandsdateien bytegleich und werden weiter per Hash geprüft. Der Eintrag von
[Stand 14](../app_reproduction_2026_11_09_ratios_tfs/README.md) (`a45d172c…`) erhielt nur seinen Linkcommit `ad0882a`.

**Forschungsstand → Dissertationsnachbau** zeigt jetzt:

- **vierzehn** eingefrorene frühere Stände, zuletzt 17 Teilmodule/1.858 Tests;
- weiter **siebzehn** Teilmodule, **2.143** Teil-/Belegtests (285 neue Belegtests des Vertrags);
- eine neue Karte „Eingabevertrag des Kinneret-Falls (Slot 3)“ mit den Zählungen, fünf Befunden, der Tabelle der Quellenschalter (8 Fälle), den 18 Bedingungen und, eingeklappt, allen 48 Eingaben mit Stand und Zeilen.

Die App liest nur die gepinnten Dateien. `_input_contract` bricht den Bau ab, sobald der Vertrag eine Freigabe, einen Lauf, eine Ziehung, ein geschlossenes Bündel oder eine gewählte Variante behauptet, ein Stand unbekannt ist, die Zählung nicht passt oder `inside.giesecke` nicht mehr bedingt ist.

## Tatsächlich ausgeführte Prüfungen

[validation.json](validation.json). `record_validation.py` bricht ab, sobald ein Exit-Code nicht 0 ist, eine `FEHLER`- oder Timeout-Zeile im Log steht, eine Tabelle breiter als ihr Rahmen ist oder eine Testzahl, die Zahl der Fälle, ein Hash oder die Standby-Abfrage nicht stimmt.

- **App-/Daten-/Buildtests:** 188 bestanden, 2 übersprungen (Python-Browserwrapper ohne Playwright, explizit ersetzt), Exit 0. 16 neue Testfälle.
- **Historische Teil-/Belegtests:** 2.143 bestanden in 23 Dateien, Exit 0. Phase-1-Prüfer Exit 2.
- **Edge offline**, beide Formate × Handy 390 px/PC × Hell/Dunkel × DE/EN, Gesten: zehn Fälle ok, Exit 0, keine `FEHLER`-Zeile; 51 Links, 23 Tabellen, 17 Teilmodule, 14 frühere Stände.
- Breiten bei 390 px stehen in [validation.json](validation.json); keine der 23 Tabellen ist breiter als ihr Rahmen.
- **Zwei Standorte** ok, Exit 0. Sichtprüfung: [Mobil hell](mobile.png), [Desktop dunkel](desktop.png).
- **Nicht ausgeführt:** Gesamtsuite, `verify_packet.py`, `tools/verify_*`; R-Skripte, Sampler, Ziehungen, Trainings, Altersmodelle, Fossilrekonstruktionen.

## Grenzen

Keine historische Fallreproduktion. „Gebunden“ heißt am Verbraucherpfad vorhanden oder Literal im Code, nie an den historischen Lauf gebunden. Alle elf Nachweisbündel bleiben offen; keine Variante gewählt. Baseline, CLI, Artenmodelle, Paperreferenz und Gates sind unverändert. Plots jung links, alt rechts.

Nächste Übergabe: [Claude-Auftrag nach dem Eingabevertrag](../CLAUDE_NEXT_STEP_AFTER_INPUT_CONTRACT_2026_11_09.md).
