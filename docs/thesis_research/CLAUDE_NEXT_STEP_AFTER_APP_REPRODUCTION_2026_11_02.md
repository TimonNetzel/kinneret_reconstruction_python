# Claude-Übergabe: nach der App-Anbindung des Dissertationsnachbaus

Stand 02.11.2026. Gilt ebenso für ChatGPT/Codex. Vorrangiges Nutzerziel bleibt:
**die lokalen taxonbasierten Rekonstruktionen der Dissertation historisch
nachvollziehbar nachbauen und in der App umsetzen; Verbesserungen danach.**
Deutsch, kurze Fortschrittsmeldungen, keine Subagenten. Nur den vom Nutzer
benannten vorhandenen Checkout verwenden; keine Klone/Worktrees. Bestehender
Branch `dissertation-reconstruction`, [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
kein automatischer Merge. Zuerst Git-/PR-Stand und lokale Änderungen prüfen.
Ausgangspunkt dieses Pakets `26d65af`; den Sicherungscommit liefert
`git log -1 -- src/kinneret_recon/nachbaustand.py`.

## Was jetzt fertig ist

1. Lies `CLAUDE.md`, `AGENTS.md`, den neuesten [Projektstatus](PROJECT_STATUS.md),
   den [Arbeitsplan](LOCAL_DISS_RECONSTRUCTION_PLAN.md) und das
   [App-Nachbaupaket](app_reproduction_2026_11_02/README.md).
2. Die Ansicht **Forschungsstand** zeigt in beiden Offlineformaten einen eigenen
   Abschnitt **Dissertationsnachbau**: Status „Teilmodule geprüft; vollständiger
   Originalfall offen“, fünf geprüfte Teilmodule mit Belegen (251 Teil-/Beleg-
   tests), elf offene Originaleingaben direkt aus dem unveränderten Belegvertrag
   und ein eingeklapptes, deutlich gekennzeichnetes **synthetisches**
   Rechenbeispiel (TF vor/nach Normierung und Maske, Summen 1 und 2/3, PDF der
   Kunstkette, Abweichung 2,22 × 10⁻¹⁶).
3. Daten: neue datierte Datei `app_reproduction_2026_11_02/reproduction_state.json`,
   im Leser `src/kinneret_recon/nachbaustand.py` per SHA-256 fest gebunden; sie
   bindet ihrerseits Paket-READMEs, `spec.json` und die vier TF-Beispieldateien.
   Fehlende Belege → sichtbar „nicht verfügbar“; geänderte Bytes oder
   Freigabeflags → Build-Abbruch. Älterer Forschungs-Snapshot unverändert und
   getrennt (`daten/forschung.js` vs. `daten/nachbau.js`).
4. Tatsächlich geprüft: 34 App-/Daten-/Buildtests (2 Python-Browserwrapper
   übersprungen, Playwright nicht im Standard-Node-Pfad) und die explizite
   Browserregression mit gebündeltem Playwright + Edge offline in beiden
   Formaten, Handy/PC, Hell/Dunkel, DE/EN, Gesten, zwei Standorte; 251
   historische Tests; beide Phase-1-Prüfer Exit 2. Einzelheiten und die
   Gesamtsuite: [validation.json](app_reproduction_2026_11_02/validation.json).

**Weiter offen:** kein vollständiger Originalfall. Alle elf Nachweisbündel
bleiben offen: finale Basis/Listenchronologie, numerische historische TFs,
Isotopen-Workspaces, globale CRU/kgc-Eingaben, Referenz-/Ankervariante,
Originalposterioren/Ziehungen, finale Nachbearbeitung, Laufzeit/RNG und
numerische Originalausgaben. Keine Variante gewählt.

**Bekannte Prüfgrenze:** `literal_tf_preparation_2026_11_02/verify_packet.py`
ist paketgebunden und bricht seit diesem Paket an „`src` seit `956951b`
unverändert“ ab; vorher schreibt er `preserved_evidence.json` bytegleich neu.
Nicht ändern und nicht als Regressionsprüfer verwenden; danach `git status`
kontrollieren, falls er doch läuft.
Die Gesamtsuite `pytest tests` hat vier vorbestehende/fremde Fehlschläge
(`test_formel`, `test_thesis_import` am Ausgangsstand bestätigt; Legacy-/
Samplerfälle). Sie enthält Sampler-Unittests; im Rahmen „keine Sampler“ künftig
gezielte Testdateien statt der Gesamtsuite ausführen.

## Nächster fachlicher Meilenstein (Phase 1)

Ein vollständiger lokaler Originalfall braucht weiterhin belegte Originaleingaben.
Die bekannte Speicherortfrage nicht wiederholen; keine pauschale Inventur.
Bis neue Originale auftauchen, den nächsten **begrenzten deterministischen**
Rechenschritt aus den bereits gebundenen Quellen isolieren:

- In `src/mcmc_prepare.R` / `src/mcmc_functions_v1.6.R` (über die privaten
  Pfade aus `.local/my_reconst_paths.json`, nur lesen) die im TF-Paket
  ausgeklammerten umgebenden Schritte prüfen: `weightedMedian`, Taxon-Pruning,
  Splines, `sample`-/Quantilschritte. Zuerst belegen, welcher davon ohne
  Zufallszustand deterministisch ist. Nur einen solchen Schritt als eigenes
  Modul mit unabhängiger Bruch-/Handreferenz und Gegenbeispielen nachbilden.
- Zufallsabhängige Schritte (`sample`, RNG) nur dokumentieren; ohne belegten
  Seed/RNG-Zustand keine Ersatzwahl. Keine `inside.range`-, Median- oder
  Variantenentscheidung ohne Beleg.
- Bisherige Fixtures, Module, Tests und Pakete bytegleich lassen; neue Kunst-
  verbindung nur als zusätzliche Probe.

## Nächster App-Schritt

Erst nach einem neuen geprüften Teilmodul: einen **neuen** datierten
Nachbaustand (neuer Ordner `app_reproduction_<Datum>/`, neuer Pin) ergänzen,
statt `app_reproduction_2026_11_02/reproduction_state.json` zu ändern. Leser so
erweitern, dass der jeweils neueste gebundene Stand angezeigt wird und ältere
Stände eingefroren bleiben; Modul-/Testzahlen aus dem neuen Paket übernehmen.
Weiter keine Erfolgsanzeige aus Testzahlen; Kunstbeispiele nie in Laufkatalog
oder Rekonstruktionen einspeisen. Beide Offlineformate, DE/EN, Hell/Dunkel,
Handy/PC, Tabellenüberlauf und zwei Standorte erneut prüfen:

```bash
PYTHONPATH=src python -m pytest tests/test_nachbaustand.py tests/test_forschungsstand.py tests/test_app.py tests/test_artenvergleich.py -q
NODE_PATH=<Codex-Laufzeit>/dependencies/node/node_modules node tools/app_pruefen.js results/app/index.html results/kinneret-app.html --kanal msedge
```

Keine weiteren modernen Modelle, Komfortfunktionen oder Phase-2-Verbesserungen
vorziehen.

## Arbeitsgrenzen

Originale nur lesen; `diss` und private Quellpfade nie ändern. Keine stillen
Ersatzdaten, Quellenkorrekturen oder Variantenentscheidungen. Keine Trainings,
Sampler, Altersmodelle, R-Läufe oder Fossilrekonstruktionen. Baseline, CLI,
sieben experimentelle Artenmodelle, Paperreferenz und fallbezogene
Monticchio-/Chronologie-Gates erhalten. Plots jung links, alt rechts.
Diese und frühere Übergaben eingefroren lassen. Am Ende neues Belegpaket,
aktualisierter Projektstatus/Arbeitsplan und **eine neue konkrete Claude-
Übergabe mit Link im Abschluss**; geprüftes Paket auf dem bestehenden Branch
committen/pushen, PR #77 fortschreiben, nicht mergen.
