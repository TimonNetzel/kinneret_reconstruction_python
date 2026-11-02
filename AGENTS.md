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

# Kommunikation und Fortsetzung des Netzel-Projekts

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

> **Aktuelle Reihenfolge auf Nutzerauftrag, 02.11.2026:** Zuerst die lokalen
> Rekonstruktionen der Dissertation nachbauen, anschließend Verbesserungen.
> Maßgeblich: [Arbeitsplan](docs/thesis_research/LOCAL_DISS_RECONSTRUCTION_PLAN.md).
> Die [Fallmatrix](docs/thesis_research/local_case_matrix_2026_11_02/README.md)
> ist abgeschlossen. [Kinnerets ältere Eingaben](docs/thesis_research/kinneret_binding_2026_11_02/README.md)
> sind teilweise gebunden; finale Basis/TFs fehlen, weiterer Speicherort unbekannt.
> [Referenz-/Ankervertrag](docs/thesis_research/kinneret_references_2026_11_02/README.md)
> ist dokumentiert; [Nachbearbeitungsfassungen](docs/thesis_research/kinneret_postprocessing_2026_11_02/README.md)
> sind getrennt gebunden. Der [Phase-1-Belegvertrag](docs/thesis_research/kinneret_phase1_spec_2026_11_02/README.md)
> ist maschinenprüfbar (26 Belege/19 Befunde/50 Snapshotdateien, elf offene
> Nachweisbündel). Finaler Eingabestand und Laufbindung fehlen weiterhin.
> Vollständiger Nachbau braucht neue Originalbasis-/Chronologie-/TF-Belege.
> Unabhängig davon ist ein [deterministischer Teilnachbau](docs/thesis_research/literal_final_pdf_2026_11_02/README.md)
> mit 22 neuen synthetischen Tests umgesetzt; keine finale Variante gewählt.
> Nächster begrenzter technischer Teil: Alters-/Tiefenübertragung aus dem
> bereits gebundenen `src/check_ages.R` synthetisch prüfen, keine neue Chronologie.
> Kein Fall vollständig reproduzierbar; keine Modellläufe, pauschalen Inventuren
> oder wiederholten Speicherortfragen.
> Historische Befunde und wissenschaftliche Gates bleiben erhalten; frühere
> Prioritäten unten sind Verlauf.


> Aktueller Einstieg 02.11.2026: Der Nutzer hat die GitHub-Sicherung und
> Fortsetzung mit Claude beauftragt. Zuerst
> [CLAUDE_HANDOFF_2026_11_02.md](docs/thesis_research/CLAUDE_HANDOFF_2026_11_02.md)
> lesen; Branch `dissertation-reconstruction` bis zu einem belegten Merge
> verwenden. Historische main-/Upload-Aussagen unten bleiben als Verlauf erhalten.

Vom Nutzer am 30.10.2026 gewünscht; aus PR #26/#27 übernommen.

- Vor fachlicher Weiterarbeit PR-/Merge-Stand und
  [PROJECT_STATUS.md](docs/thesis_research/PROJECT_STATUS.md) prüfen.
- Zu Beginn und nach Arbeitspaketen kurz auf Deutsch berichten:
  **fertig / aktuell / als Nächstes**, mit Bezug zur taxonbasierten Multiproxy-App.
- Statusdatei mit Ergebnis und PR-Verweis fortschreiben; begonnen, blockiert
  und abgeschlossen unterscheiden. Fertige Analog-Baseline nicht als fertige
  taxonbasierte Hauptmethode oder Fossilrekonstruktion bezeichnen.
- Claude betreut CLI/App; Codex Kalibrierung, Quellen und unabhängige Prüfung.
  Bestehende Baseline und Claudes Integrationsdateien unverändert lassen.
- Keine stillen Quellenkorrekturen, automatischen Pilotfreigaben,
  Autorennachrichten. Merges nur mit ausdrücklichem Nutzerauftrag; die
  Zusammenführung von #18–#29 ist am 30.10.2026 ausdrücklich beauftragt.
  Diese technische Zusammenführung ist keine fachliche Pilotfreigabe.

Der Nutzer hat nach #27 die eigenständige Erstellung neuer Taxon-
Transferfunktionen parallel zur offenen Chronologieprüfung beauftragt.
Der Nutzer hat anschließend den Vergleich SMOTE + einfaches Netz nach seiner
Dissertation beauftragt. Aktueller Einstieg:
[CHAT_RESTART_SMOTE.md](docs/thesis_research/CHAT_RESTART_SMOTE.md).
Nach der beauftragten Zusammenführung `main` verwenden. #28 bleibt am
gesperrten Commit die wissenschaftliche Referenz für Daten und bisherige Modelle.
Aktuelle gemeinsame Übergabe: [HANDOFF.md](docs/thesis_research/HANDOFF.md).
Frühere getrennte Forschungsstände:
[PR #27](https://github.com/TimonNetzel/kinneret_reconstruction_python/blob/5fd3bad46d5108e8986c054171fba66a05c26589/docs/thesis_research/CHAT_RESTART_VARIABLES.md).
