@AGENTS.md

# Quellmaterial der Dissertation

- Die ungeordneten Originaldateien der Dissertation liegen lokal in einem
  zusätzlich eingebundenen Arbeitsverzeichnis `diss` (außerhalb des Repos).
- `diss` nur lesen: dort nichts ändern, verschieben oder löschen.
- Übernommene Inhalte im Repo ablegen und die Herkunft (Datei in `diss`)
  angeben; keine stillen Quellenkorrekturen.
- Ist `diss` in einer Sitzung nicht eingebunden (z. B. in der Cloud), das
  kurz melden statt Inhalte zu raten.

> **Aktuell: gewichteter Median nachgebildet, App-Stand 8, 08.11.2026:**
> [Modul und Belege](docs/thesis_research/literal_weighted_median_2026_11_08/README.md): `mcmc_prepare.R:1628–1664`;
> `weightedMedian` wörtlich nach dem C-Kern von matrixStats (normalisiert in allen 13 Fassungen des Fensters ein Text).
> NA, sobald irgendein Gewicht NA/NaN oder keines positiv ist; bei `+Inf` zählt die Reihenfolge. R 3.5.0 + matrixStats
> 0.56.0: 439 von 439 Werten bitgleich, `which.tf.missing` gleich. `tann <- FALSE` (Bedingung 1248, Zuweisung 1249) wird
> nur in `mcmc_functions_v1.6.R:1939` gelesen. 182 neue Tests, 934 grün.
> [App-Stand 8](docs/thesis_research/app_reproduction_2026_11_08_weighted_median/README.md) mit eigenem Pin; Stände 1–7
> eingefroren. Kein Originalfall; elf Bündel offen.
> [Neue konkrete Claude-Übergabe](docs/thesis_research/CLAUDE_NEXT_STEP_AFTER_WEIGHTED_MEDIAN_2026_11_08.md).
> Der folgende „Aktuell“-Hinweis ist Verlauf.

> **Aktuell: Dirichlet-Quotient nachgebildet, App-Stand 7, 03.11.2026:**
> [Modul und Belege](docs/thesis_research/literal_dirichlet_ratio_2026_11_03/README.md): `mcmc_functions_v1.6.R:1269–1274`
> mit **expliziten** Gewichten; `ddirichlet` wörtlich nach LaplacesDemon (im CRAN-Fenster gebunden): keine Normierung
> (Zeile ohne Zuweisung), Vorschlag ohne `dirichlet_spread`, NaN → 0, `Inf` bleibt; `lgamma`/`log`/`exp` korrekt
> gerundet. Bitgleich mit unabhängiger Spouge-Referenz; R 3.5.0 + LaplacesDemon 16.1.6: 59 von 62 Werten bitgleich,
> höchstens 4 ulp wegen R-`lgamma(0.4)`. 50 neue Tests, 752 grün.
> [App-Stand 7](docs/thesis_research/app_reproduction_2026_11_03_dirichlet_ratio/README.md) mit eigenem Pin, zeigt auch
> die R-Gegenprobe; Stände 1–6 eingefroren. Kein Originalfall; elf Bündel offen.
> [Neue konkrete Claude-Übergabe](docs/thesis_research/CLAUDE_NEXT_STEP_AFTER_DIRICHLET_RATIO_2026_11_03.md).
> Der folgende „Aktuell“-Hinweis ist Verlauf.

> **Aktuell: Laufzeitbelege und R-3.5.0-Gegenprobe, 03.11.2026:**
> [Belegpaket](docs/thesis_research/r_runtime_crosscheck_2026_11_03/README.md). Auf Nutzerauftrag:
> Kinneret ist **Slot 3** des Zusatzorte-Zweigs (`add.sites <- T`); `mcmc_prepare.R:1248` setzt für Kinneret
> `tann <- FALSE`. Alle Laufeingaben stammen aus **R 3.5.0**; die Laufzeitfassung bleibt offen.
> Im Fenster 2018-04-23…2023-07-07 liefert **LaplacesDemon** `rdirichlet`/`ddirichlet` (in allen vier Fassungen
> textgleich) und **matrixStats** `weightedMedian`. R-Läufe für Gegenproben sind jetzt erlaubt: R 3.5.0 lokal
> (`.local/R`); 389 Prüfungen der wörtlichen Originalzeilen auf Kunstfixtures sind **bitgleich** mit vier Modulen.
> 702 Tests grün. Kein Originalfall; elf Bündel offen.
> [Neue konkrete Claude-Übergabe](docs/thesis_research/CLAUDE_NEXT_STEP_AFTER_R_CROSSCHECK_2026_11_03.md).
> Der folgende „Aktuell“-Hinweis ist Verlauf.

> **Aktuell: Kern der Gewichtsstufe nachgebildet, App-Stand 6, 03.11.2026:**
> [Determinismusprüfung und Modul](docs/thesis_research/literal_weights_stage_2026_11_03/README.md):
> `mcmc_functions_v1.6.R:1259/1276–1280` mit **expliziten** Gewichten, ohne Zufall/Paketsemantik, kein
> Samplerlauf; Gewichte über die zeilenweisen Werte recycelt, `byrow = T`, `rowSums(na.rm = T)` (NA/NaN
> übersprungen, Akkumulator markiert), Division je Zeile (Summe 0 → NaN/±Inf, `-Inf` → NaN/`-0`), `z = 1:ncol`.
> `rdirichlet`/`ddirichlet` sind in keiner Quelle definiert; Anbieter ungebunden, nur dokumentiert.
> 134 neue Tests, 689 grün. [App-Stand 6](docs/thesis_research/app_reproduction_2026_11_03_weights_stage/README.md)
> mit eigenem Pin; Stände 1–5 eingefroren. Kein Originalfall; elf Bündel offen.
> [Neue konkrete Claude-Übergabe](docs/thesis_research/CLAUDE_NEXT_STEP_AFTER_WEIGHTS_STAGE_2026_11_03.md).
> Der folgende „Aktuell“-Hinweis ist Verlauf.

> **Aktuell: Sampler-Vorbereitung nachgebildet, App-Stand 5, 02.11.2026:**
> [Determinismusprüfung und Modul](docs/thesis_research/literal_sampler_preparation_2026_11_02/README.md):
> `mcmc_conduction.R:53–79` ohne Zufall/Paketsemantik, kein Samplerlauf; `gibbs <- F`
> (`main_v1.2.R:624`, einzige Zuweisung) → `how_many_taxa = num_taxa`; `c(t(.))` zeilenweise,
> Zeile 75 überschreibt 64, Spaltenmaximum mit NA/NaN/`-Inf`, Ganz-NA-Spalte abgelehnt,
> `colMeans` ohne Leser (Akkumulator markiert). 118 neue Tests, 555 grün.
> [App-Stand 5](docs/thesis_research/app_reproduction_2026_11_02_sampler_prep/README.md)
> mit eigenem Pin; Stände 1–4 eingefroren. Kein Originalfall; elf Bündel offen.
> [Neue konkrete Claude-Übergabe](docs/thesis_research/CLAUDE_NEXT_STEP_AFTER_SAMPLER_PREP_2026_11_02.md).
> Der folgende „Aktuell“-Hinweis ist Verlauf.

> **Aktuell: fehlende TF und `num_taxa`-Priors nachgebildet, App-Stand 4, 02.11.2026:**
> [Determinismusprüfung und Modul](docs/thesis_research/literal_tf_missing_priors_2026_11_02/README.md):
> `which.tf.missing` + Priors `mcmc_prepare.R:1664–1685` ohne Zufall/Paketsemantik, im
> gebundenen Lauf aktiv; NA-Maske explizit (weightedMedian offen), leere Maske → `num_taxa`
> aus 1435, `round(n/4)` zur geraden Ziffer, `phi1`/`phi2` veraltet. 85 neue Tests, 437 grün.
> [App-Stand 4](docs/thesis_research/app_reproduction_2026_11_02_tf_missing/README.md)
> mit eigenem Pin; Stände 1–3 eingefroren. Kein Originalfall; elf Bündel offen.
> [Neue konkrete Claude-Übergabe](docs/thesis_research/CLAUDE_NEXT_STEP_AFTER_TF_MISSING_2026_11_02.md).
> Der folgende „Aktuell“-Hinweis ist Verlauf.

> **Aktuell: Gamma-Prior-Raster nachgebildet, App-Stand 3, 02.11.2026:**
> [Determinismusprüfung und Modul](docs/thesis_research/literal_gamma_prior_grid_2026_11_02/README.md):
> Raster `mcmc_prepare.R:114–121/1270–1281` ohne Zufall und Paketsemantik, im
> gebundenen Hauptlauf unbedingt aktiv; Obergrenze 50000/5000 explizit, Rezentzweig
> 1733 ersetzt `rate`. 55 neue Tests, 352 grün.
> [App-Stand 3](docs/thesis_research/app_reproduction_2026_11_02_gamma_grid/README.md)
> mit eigenem Pin; Stände 1 und 2 eingefroren. Kein Originalfall; elf Bündel offen.
> [Neue konkrete Claude-Übergabe](docs/thesis_research/CLAUDE_NEXT_STEP_AFTER_GAMMA_GRID_2026_11_02.md).
> Der folgende „Aktuell“-Hinweis ist Verlauf.

> **Aktuell: Taxon-Pruning nachgebildet, App-Stand 2, 02.11.2026:**
> [Determinismusprüfung und Modul](docs/thesis_research/literal_taxon_pruning_2026_11_02/README.md):
> nur das Pruning ist ohne Zufall und ohne ungebundene Paketsemantik;
> im gebundenen Hauptlauf abgeschaltet (bedingter Schritt). 46 neue Tests,
> 297 grün. [App-Stand 2](docs/thesis_research/app_reproduction_2026_11_02_pruning/README.md)
> mit eigenem Pin; Stand 1 eingefroren. Kein Originalfall; elf Bündel offen.
> [Neue konkrete Claude-Übergabe](docs/thesis_research/CLAUDE_NEXT_STEP_AFTER_TAXON_PRUNING_2026_11_02.md).
> Der folgende „Aktuell“-Hinweis ist Verlauf.

> **Aktuell: Dissertationsnachbau in der App sichtbar, 02.11.2026:**
> [Ergebnis und Prüfung](docs/thesis_research/app_reproduction_2026_11_02/README.md):
> Forschungsansicht zeigt fünf geprüfte Teilmodule, elf offene Originaleingaben
> und ein ausdrücklich synthetisches Rechenbeispiel, beide Offlineformate geprüft.
> Kein Originalfall reproduziert; elf Nachweisbündel offen.
> [Neue konkrete Claude-Übergabe](docs/thesis_research/CLAUDE_NEXT_STEP_AFTER_APP_REPRODUCTION_2026_11_02.md).
> Der folgende „Aktuell“-Hinweis ist Verlauf.

> **Aktuell: TF-Vorbereitung und fünfteilige Kunstkette geprüft, 02.11.2026:**
> [Ergebnis und Quellen](docs/thesis_research/literal_tf_preparation_2026_11_02/README.md):
> 51 neue Tests, mit bisherigen Teilen 251 grün. TF-Zeilennormierung vor
> expliziter Zellmaske; keine anschließende Renormierung. Noch kein Originalfall.
> [Neue konkrete Claude-Übergabe: App-Anbindung](docs/thesis_research/CLAUDE_NEXT_STEP_APP_REPRODUCTION_2026_11_02.md).
> Vorrangiges Ziel bleibt der Dissertationsnachbau mit Umsetzung in der App;
> Verbesserungen danach. Jedes geprüfte Paket erhält eine eigene Claude-
> Übergabedatei und deren Link im Abschluss. Alte Startaufträge bleiben erhalten.
> Ältere nächste Schritte unten sind Verlauf. Elf Nachweisbündel bleiben offen.

> **Aktueller Stand nach climate_modelling-Abgleich, 02.11.2026:**
> [Neues Belegpaket](docs/thesis_research/climate_modelling_audit_2026_11_02/README.md):
> 1.828 Dateien, 161 R-Quellen, ein ZIP tatsächlich zusätzlich geprüft.
> main.R-Verbreitungsbasis bytegleich zur Kinneret-Kopie; F14/F15 sind
> CSV-Kandidaten ohne gesicherte SISAL-V2-Version. v4.0, finale Basis und
> numerische historische TFs weiter ungebunden; elf Nachweisbündel offen.
> 23 Parser-/Lesertests bestanden. Keine neuen Modell-/Rekonstruktionsläufe.
> [Aktueller nächster Auftrag für ChatGPT und Claude](docs/thesis_research/NEXT_STEP_AFTER_CLIMATE_AUDIT_2026_11_02.md):
> begrenzte deterministische TF-Vorbereitung gemäß eingefrorenem Auftrag.
> Frühere Hinweise „climate_modelling ungeprüft“ unten sind Verlauf.

# Aktueller Projekteinstieg für Claude

> **Aktueller Einstieg für den Wechsel zu Claude, 02.11.2026:**
> **[Hier beginnen: vollständige Übergabe und climate_modelling](docs/thesis_research/CLAUDE_RESUME_CLIMATE_MODELLING_2026_11_02.md).**
> Der Nutzer stellt in der nächsten Sitzung einen neuen Ordner
> `climate_modelling` bereit. Noch nicht geprüft, genauer Pfad noch nicht gebunden.
> Zuerst gegen `main.R`, `main_v1.2.R` und die
> [priorisierte Fehlenden-/Kopienliste](docs/thesis_research/MISSING_PROJECT_FILES_2026_11_02.md)
> abgleichen. Dann erst verbleibende Lücken/weiteren technischen Auftrag festlegen.
> `/automount/` lag auf gelöschter Festplatte; lokale Kopien können trotzdem
> existieren. Erst Kopien prüfen, nur verbleibende nötige Dateien neu erstellen.
> [Quellenregel](docs/thesis_research/SOURCE_RECOVERY_PLAN.md). Keine neuen Läufe.
> Bisher 200 historische Teiltests gesichert; acht neue Parserprüfungen bestanden.
> Elf Nachweisbündel offen, keine finale Variante. Frühere Einstiege sind Verlauf.

> **Neuester Teilfortschritt: Altersraster/Zeilenauswahl, 02.11.2026:**
> [Belege und synthetische Prüfung](docs/thesis_research/literal_age_selection_2026_11_02/README.md).
> 39 neue Tests, mit Vorpaketen 124 grün. Exakte Rastermitgliedschaft ist kein
> Intervalltest; vorläufige Holozänmaske wird im erhaltenen Kontrollfluss auf
> alle Zeilen zurückgesetzt. Taxonwechselzweig bleibt durch `& FALSE` aus.
> `age`/`site.age` und freie Maskenvariablen nicht still synchronisieren.
> Keine finale Laufbindung; elf Nachweisbündel bleiben offen.
> Nächster unabhängiger Einstieg: kumulativer Dichte-/EMD-Ausschnitt in
> `src/mcmc_functions_v1.6.R:366–370,485–490` gegen skalare Referenzargumente
> der Fallstudien prüfen. Keine neue Referenz/Kalibrierung oder Vollrunner.
> Ältere Einstiege unten sind Verlauf. Keine neuen Läufe oder Speicherortfrage.

> **Neuester Teilfortschritt, 02.11.2026:**
> [Historische Alters-/Tiefenübertragung](docs/thesis_research/literal_age_depth_2026_11_02/README.md)
> aus `check_ages.R` und `mcmc_prepare.R` begrenzt nachgebildet: 43 neue
> synthetische Tests, mit Vorpaketen 85 grün. Listen-/Matrixform, rho(D|A),
> Normierungszeitpunkt und einmalige Altersverschiebung bleiben getrennt.
> Keine finale Variante oder R-Laufzeitgleichheit; elf Nachweisbündel offen.
> Nächster unabhängiger Einstieg: `age`/`site.age`/`reconst_ages` im gebundenen
> `src/mcmc_functions_v1.6.R` auf Zeilen-/Zeitfensterauswahl prüfen, bevor Module
> verbunden werden. Keine neuen Läufe, Ersatzdaten oder Speicherortfrage.
> Die älteren nächsten Schritte unten sind Verlauf. Phase 1 bleibt vorrangig.

Aktuelle Priorität auf Nutzerauftrag vom 02.11.2026:
**Zuerst die lokalen Rekonstruktionen der Dissertation nachbauen,
anschließend Verbesserungen.** Verbindlicher Ablauf:
`docs/thesis_research/LOCAL_DISS_RECONSTRUCTION_PLAN.md`.

Lies zuerst `AGENTS.md`, den neuesten Abschnitt von
`docs/thesis_research/PROJECT_STATUS.md`, den neuen Arbeitsplan und
`docs/thesis_research/CLAUDE_HANDOFF_2026_11_02.md`.
Die quellengebundene Fallmatrix ist abgeschlossen:
`docs/thesis_research/local_case_matrix_2026_11_02/README.md`.
Kinnerets älterer Proben-/Taxonstand ist jetzt teilweise exakt gebunden:
`docs/thesis_research/kinneret_binding_2026_11_02/README.md`.
Die finale v1.2-Basisdatei und numerischen TFs fehlen weiter; der Nutzer kennt
keinen weiteren Speicherort. Noch kein Fall vollständig reproduzierbar.
Kinnerets Referenz-/Ankerpfad ist dokumentiert, numerische Eingaben und
Laufbindung bleiben offen:
`docs/thesis_research/kinneret_references_2026_11_02/README.md`.
Die historischen Nachbearbeitungsfassungen sind getrennt gebunden:
`docs/thesis_research/kinneret_postprocessing_2026_11_02/README.md`.
Originalposterioren und finale Auswahl-/Kombinationsfassung fehlen weiter.
Der maschinenprüfbare Phase-1-Belegvertrag ist abgeschlossen:
`docs/thesis_research/kinneret_phase1_spec_2026_11_02/README.md`.
Lesender Prüfer: `python tools/check_kinneret_phase1_spec.py` (erwartet Exit 2).
26 Belegdateien, 19 Befunde und 50 Snapshotdateien intakt; elf Nachweisbündel offen.
Für den vollständigen Fall fehlen finale Originalbasis/Chronologieliste,
numerische TFs und historische Laufzuordnung. Keine Quelle automatisch auswählen;
keine Wiederholung der bereits beantworteten Speicherortfrage.
Unabhängiger technischer Teilnachbau der erhaltenen Taxongewichtung/PDF-Summe:
`docs/thesis_research/literal_final_pdf_2026_11_02/README.md`.
22 neue synthetische Tests bestanden; keine finale Laufbindung hergestellt.
Als nächsten begrenzten Teil historische Alters-/Tiefenübertragung aus dem
bereits gebundenen `src/check_ages.R` isolieren und synthetisch prüfen.
Keine neue Chronologie berechnen oder Matrix-/Listenformen gleichsetzen.
Keine erneute pauschale Suche oder neuen Läufe; ältere Eingaben nicht ersetzen.
Neue Kalibrierung, methodische Verbesserungen und zusätzliche
Reliabilitätsansicht gehören in Phase 2 nach dem gesicherten Nachbau.

Die Fortsetzung liegt auf `dissertation-reconstruction`; vor Beginn Git-/PR-
Stand prüfen. Nicht voraussetzen, dass `main` PR #77 schon enthält.
Deutsch und kurze Fortschrittsmeldungen; keine Subagenten. Nur den vom Nutzer
benannten vorhandenen Checkout verwenden, keine neuen Klone oder Worktrees.

Historische Rechenschritte und spätere Verbesserungen getrennt halten.
App/CLI, Analog-Baseline, Paperreferenz und sieben experimentelle Artenmodelle
bewahren. Die sieben modernen Artenmodelle ersetzen keine historischen TFs;
279 historische numerische Funktionen gelten weiterhin nicht als wiederhergestellt.
Offene Originaleingaben, Quellkonflikte und fachliche Gates sichtbar halten.
Monticchio-/Chronologie-Gates bleiben fallbezogen erhalten. Diese Planänderung
startet keine Trainings-/Samplerläufe oder neuen Fossilrekonstruktionen.
Plots immer jung links, alt rechts. Technische Sicherung ist keine Pilotfreigabe.

Forschungsstands-/Quellenansicht und Sieben-Arten-Vergleich einschließlich
Foldansicht sind umgesetzt und geprüft:
`docs/thesis_research/app_folds_2026_11_02/README.md`.
Abgeschlossene, geprüfte Pakete jeweils auf dem bestehenden Branch in PR #77
committen und pushen; keine automatischen Merges.
