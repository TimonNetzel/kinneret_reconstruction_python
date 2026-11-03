# Dissertationsnachbau in der App, Stand 7: Dirichlet-Quotient und R-Gegenprobe

Stand 03.11.2026. Bestehender Branch `dissertation-reconstruction`,
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77). Ausgangscommit `c0cdf81`
(PR-Kopf, `MERGEABLE`); Rechenschritt in `766a1ff`. Auftrag: Nutzerentscheidung „Ja, ddirichlet“.

## Ergebnis

Neuer letzter `STATES`-Eintrag [reproduction_state.json](reproduction_state.json)
(SHA-256 `cf1b77f22bddb4deca27cebe9a1a98a1a9541d786fc6078e007b43f8f7d2bfea`). Quellenlinks zeigen auf `766a1ff`.

Die Stände 1–6 bleiben bytegleich und werden weiter per Hash geprüft. Der Eintrag von
[Stand 6](../app_reproduction_2026_11_03_weights_stage/README.md) (`7eb9ac5e…`) erhielt nur seinen Linkcommit
`3104f3c`, bestätigt mit `git log -1 --format=%H` auf seiner Zustandsdatei.

**Forschungsstand → Dissertationsnachbau** zeigt jetzt:

- **sechs** eingefrorene frühere Stände, zuletzt 10 Teilmodule/689 Tests;
- **elf** Teilmodule und **752** Teil-/Belegtests. Neu ist „Dirichlet-Quotient der Gewichtsstufe“ (50 Tests,
  Quellen `mcmc_functions_v1.6.R` und LaplacesDemon 16.1.x), ohne Etikett „Bedingter Schritt“;
- in der Modulkarte eine Zeile **R-Gegenprobe**, deren Werte nur aus den gepinnten Ergebnisdateien stammen:
  - R 3.5.0, vier Teilmodule mit 389 bitgleichen Prüfungen;
  - Dirichlet-Quotient: 59 von 62 Werten bitgleich, höchstens 4 ulp wegen `lgamma(0,4)` in R;
  - Kinneret ist Slot 3;
  - 13 Belegtests;
- im eingeklappten synthetischen Beispiel eine Tabelle **Dirichlet-Quotient** mit den Kunstfällen D1–D4:
  - Zeilen: Taxa, Gewichts-, Vorschlags- und Produktquotient, Abweichung zu R in ulp;
  - D3 zeigt `Inf` und die Ersetzung NaN → 0 (Zelle markiert);
  - D4 zeigt die 4 ulp;
  - die Fallbezeichnungen D1–D4 wiederholen keine frühere im Abschnitt;
- die Kunstkette heißt jetzt „… Sampler-Vorbereitung, Gewichtsstufe und Dirichlet-Quotient“.

Keine Erfolgsanzeige aus Testzahlen. Kunstbeispiele fließen in keinen Lauf ein; es gab keine Ziehung und keinen
Samplerlauf.

## Datenbindung

[nachbaustand.py](../../../src/kinneret_recon/nachbaustand.py):

- `_dirichlet_example` verlangt Gleichheit der gespeicherten Werte mit der unabhängigen Referenz und lehnt
  Freigabe- oder Variantenflags ab.
- `_runtime_check` liest die R-Ergebnisse und die Slotbindung. Es lehnt Freigabeflags, angehängte Pakete und
  Zufall in der Gegenprobe ab und verlangt die Bitgleichheit aller nicht markierten Werte.
- Die Testsumme schließt jetzt die 13 Belegtests der Laufzeitprüfung ein (689 + 50 + 13 = 752).

## Tatsächlich ausgeführte Prüfungen

[validation.json](validation.json):

- **App-/Daten-/Buildtests:** 81 bestanden, 2 übersprungen (Python-Browserwrapper ohne Playwright, explizit
  ersetzt). 12 neue Testfälle:
  - Dirichlet-Beispiel und R-Gegenprobe;
  - zwei Fehlstände;
  - Bytewechsel an Stand 6 und an neun Belegdateien.
- **Historische Teil-/Belegtests:** 752 bestanden, nach den App-Änderungen erneut 752. Beide Phase-1-Prüfer Exit 2.
- **Edge offline**, beide Formate × Handy 390 px/PC × Hell/Dunkel × DE/EN, Gesten:
  - geprüft: 28 Links (22 auf `766a1ff`, je einer auf sechs eingefrorene Stände), 12 Tabellen, R-Zeile;
  - erster Lauf alle 10 Fälle ok;
  - Tabellenbreite bei 390 px: 247 von 332 px, Zeilenköpfe `nowrap`.
- **Zwei Standorte** ok. Sichtprüfung: [Mobil hell](mobile.png), [Desktop dunkel](desktop.png).
- **Nicht ausgeführt:** Gesamtsuite, `verify_packet.py`, `tools/verify_*`; Sampler, Ziehungen, Trainings,
  Fossilrekonstruktionen.

## Grenzen

Siebte App-Anbindung des geprüften Nachbaus, **keine historische Fallreproduktion**.

- Alle elf Nachweisbündel bleiben offen; keine Variante gewählt.
- Paketanbieter sind nur im Zeitfenster gebunden.
- Baseline, CLI, Artenmodelle, Paperreferenz und Gates sind unverändert. Plots jung links, alt rechts.

Nächste Übergabe: [Claude-Auftrag nach dem Dirichlet-Quotienten](../CLAUDE_NEXT_STEP_AFTER_DIRICHLET_RATIO_2026_11_03.md).
