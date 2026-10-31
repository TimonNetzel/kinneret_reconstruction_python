# Tatsächlicher Prüfstand

31.10.2026. Neue strukturelle Tests: **29 bestanden**.
Erste Ausführung unter tools/taxon_synthetic/test_prototype.py, anschließend
unverändert fachlich nach tests/test_synthetic_inversion.py für normale
CI-Auffindbarkeit verschoben (Import per importlib angepasst).

Alle 108 synthetischen Auswertungen und der separate Direktprüfer bestanden;
38 Beobachtungsmatrizen rekonstruiert. Frische zweite Produktion: fünf Dateien
bytegleich. Numerische Protokolle: verification.json und repeat_verification.json.

## Unnötig breit gestarteter Standardtest – abgebrochen

`python -m pytest -q` wurde zusätzlich gestartet. Dies war zu breit: der
unveränderte Standardtest enthält unter anderem kurze Kinneret-Regressionsketten
mit bestehenden realen Testeingängen sowie umfangreiche Prior-only-Samplertests.
Ein Teil dieser technischen Tests lief, bevor die Prüfung per SIGINT gestoppt
wurde. Keine neue wissenschaftliche Kernkurve ausgewertet/gesichert und kein
Referenzergebnis ersetzt. Es ist deshalb unzutreffend, die gesamte Sitzung als
„ausschließlich synthetische Testeingänge gelesen“ zu bezeichnen.

Tatsächliche Abschlussmeldung vor dem Abbruch:

```text
FAILED tests/test_app.py::test_standort_waehler_im_browser
FAILED tests/test_app.py::test_im_browser
KeyboardInterrupt
2 failed, 56 passed, 2 deselected in 74.01s
```

Beide Browserfehler melden denselben fehlenden erwarteten Playwright-Pfad:
`/home/agent/.cache/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell`.
Playwright ist importierbar, das erwartete Browser-Binary fehlt.
Die beiden Fälle sind **fehlgeschlagen**, nicht übersprungen oder bestanden.
Die übrigen nach dem Interrupt ausstehenden Tests sind nicht ausgeführt;
zwei mit slow markierte Tests waren durch Repository-Standardoption abgewählt.
Keine Browserinstallation und keine Änderungen an CLI/App/alten Tests im Paket.
Der neue 29-Fälle-Test wird abschließend separat ausgeführt und ist nicht durch
diesen unvollständigen Repository-Lauf ersetzt.

Designabweichungen: keine fachlichen Parameter-/Kriterienänderungen.
Operative Abweichung: der unnötige Standardtest oben. Vorhandene Dateien werden
gegen den Basiscommit geprüft (preservation.json); der Prototyp selbst verwendet
ausschließlich synthetische Eingänge und importiert keine alten Modellpakete.
