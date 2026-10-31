# Umsetzung vor erstem Ergebnislauf

Autorisiert im bestehenden Chat am 31.10.2026; Basis PR #40 bei `19a7542`.
Der unveränderte [Plan](../counts_observation_design_2026_10_31/SYNTHETIC_PLAN.md)
legt alle wissenschaftlichen Parameter/Kriterien und 52 gültige Fälle fest.
Neue Pfade tools/taxon_counts, tests/test_synthetic_counts.py und dieses
Ergebnispaket; Status/Übergabe als einzige bestehende Dokumente änderbar.

Operative Präzisierungen vor Ausführung, keine Planänderung:

- shared-Generator nutzt `rng.choice(shifts, p=weights)` und anschließend
  einen multinomialen Ziehschritt; derselbe Datensatz für alle Inferenzvarianten.
- 25 Beobachtungsdatensätze (18 base, je shared/merged sowie fünf feste Fälle),
  52 Posterioren, daneben zwei explizite nicht schätzbare Negativfälle.
- Null steht in observations.json für fehlend; conditional_subset hat den
  ausgewählten Nenner 5 und ignoriert den ursprünglichen vollen Nenner 8
  entsprechend dem ausdrücklich künstlichen Auswahlprozess.
- PMFs inklusive Koeffizienten mit lgamma und Logarithmen. Separate Nachrechnung
  nutzt faktorielle Direktprodukte ohne Produzentenimport. Kein Import alter
  Modelle oder Ergebnisdateien. Prior-/HPD-Definition wie gesichert.
- Ausführbarer Code, Tests und Prüfer werden vor Tests/erstem Ergebnislauf
  committed und gepusht. Keine planmäßige wissenschaftliche Nachjustierung.
  Notwendige Implementierungsreparaturen würden hier dokumentiert.

Keine echten Modelle, Fossilcounts, Baseline oder CLI/App bearbeiten.
Keine Standardtests, realen Fits/Kernläufe, Unteragenten, Merges oder Kontakte.
W96 und reale T0/T1/T3/T4/Monticchio-P1-Grenzen bleiben offen.
