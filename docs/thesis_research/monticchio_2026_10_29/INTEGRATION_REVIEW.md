# Unabhängige Prüfung von Claude-PR #19

Geprüfter Stand: `c8977a57fecabcef8f54abd7eafd6545f53487f9`,
[PR #19](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/19),
Basis `f9acc16c58a47a51b1063690a69a2db01381c76a` aus dem offenen PR #18.
Beide Heads vor Abschluss erneut kontrolliert, unverändert.

**Keine blockierenden Codebefunde im geprüften Umfang.** Die vollständige
Browserdarstellung wurde in dieser Umgebung nicht unabhängig ausgeführt.
Dies ist keine Freigabe einer fossilen Rekonstruktion und kein Merge.

## Prüfung

- Diff aller 14 geänderten Dateien gegen #18 gelesen. Evaluator, Auditor,
  Konfiguration, Richtlinie, Metriken, Referenzmanifest und Importer sind
  unverändert. CLI lädt denselben Evaluator per Dateipfad; App liest die
  gespeicherten Metriken. Keine doppelte Modellimplementierung gefunden.
- App-Daten, Berechnung der kompakten CLI-Werte, Modellrollen, Einheiten,
  Gewichtungsumschaltung, regionale Teilansicht und Exportpfade im Code geprüft.
  k=5 ist das Hauptmodell, k=1/15 bleiben Sensitivitäten. Die Hinweise zu
  modernen Testfehlern und nachgelagerter regionaler Diagnose sind vorhanden.
- Sechs Tests in `tests/test_thesis_baseline.py` bestanden, einschließlich
  Referenzvergleich, Quellenabweichung und Schutz bestehender Ausgaben.
- Zusammen mit `tests/test_app.py` bestanden zwölf Pythonprüfungen,
  darunter beide App-Bauformen und die mitgeführten Validierungsdaten.
  Zwei Browserprüfungen konnten Chromium nicht starten: zunächst fehlte das
  Standard-Temporärverzeichnis, nach Setzen eines beschreibbaren `TMPDIR`
  fehlte das Playwright-Chromium-Executable. Die beiden Fehler werden nicht
  als Produktfehler und auch nicht als bestandene Browsertests ausgewiesen.
- GitHub-Checks auf diesem SHA: Python 3.11 und 3.13, Push und PR, viermal
  erfolgreich. Die lokale C++-Erweiterung wurde mit GCC erstellt, da das
  voreingestellte `clang++` in dieser Umgebung fehlt. Keine Quelldatei angepasst.

## Entscheidender vollständiger CLI-Lauf

Die Shared37-Tabellen wurden aus frisch geladenen, SHA256-geprüften Quellen
mit dem vorhandenen Präparer reproduziert; dessen Manifest stimmt vollständig
mit der gespeicherten Referenz überein. Anschließend auf Claudes unverändertem
Branch:

```bash
python -m kinneret_recon.cli thesis-baseline \
  --input-dir /path/to/shared37 --output-dir /path/to/new_integration_run
```

Ergebnis: Exitcode 0, **`Referenz: uebereinstimmend`** mit der gespeicherten
`reference_manifest.json`. Alle Eingangs-, Code-, Konfigurations- und
Ausgabevergleiche des Wrappers bestanden. Manifest-SHA256:
`2c6e6474eefb9cb991cb43cc285a5d3831f86d9487677368c4027f776f6eb8de`.
Hauptwerte k=5: Tann 3,765/3,926 °C und Pann 365,2/373,1 mm/Jahr (5°/10°).
Umgebung Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0.

Der Lauf prüft den tatsächlichen Integrationspfad mit Originaldaten; es wurde
keine neue Baseline entwickelt. Claudes Integrationsdateien blieben unverändert.

## Anschluss

#19 kann nach Merge von #18 auf `main` umgestellt werden. Die neue
Monticchio-Diagnose ist ein getrennter Forschungs-PR auf derselben Basis.
Die dokumentierten Distanzflags und Altersgrenzen rechtfertigen weiterhin
keine fossile Klimakurve in der App.
