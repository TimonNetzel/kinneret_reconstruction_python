# Projektstand

Stand: 26.10.2026. Dieses Dokument fasst Ziel, Plan, Ergebnisse, Entscheidungen und offene Punkte
zusammen, damit die Arbeit jederzeit wieder aufgenommen werden kann.

## Ziel

Den Code zur Publikation Netzel et al. (2025), *Clim. Past* 21, 357–380
(https://doi.org/10.5194/cp-21-357-2025) prüfen und verbessern. Daraus eine App bauen, mit der sich
Rekonstruktionen für weitere Arbeiten und Publikationen automatisiert rechnen lassen.

Rahmen (vom Autor festgelegt):

- Nutzung nur lokal durch den Autor, Code in einem privaten GitHub-Repo.
- Vorerst nur Kinneret. Die Transferfunktion (neuronales Netz) gilt als gegeben, ihr Training kommt
  in einem späteren Schritt.
- Korrekturen direkt einbauen, das Original bleibt erhalten.
- Dokumentation auf Deutsch.
- Entwicklung im Branch `claude/netzel-python-package-analysis-adcv39`.

## Plan und Stand

| Schritt | Inhalt | Stand |
|---|---|---|
| 1 | Original lauffähig machen, Referenzlauf sichern | ✅ `legacy/`, `reference/v1.0-paper/` |
| 2 | Paper Gleichung für Gleichung gegen den Code prüfen | ⏳ **offen**: Volltext fehlt (siehe unten) |
| 3 | Umbau zum Paket ohne Verhaltensänderung | ✅ Legacy-Modus bitidentisch (Tests) |
| 4 | Korrekturen einzeln, mit Vorher/Nachher-Vergleich | ✅ `docs/pruefbericht.md`, `reference/laeufe/` |
| 5 | Performance | ✅ Auswertung 576 s → 10 s; MCMC 60 s → 37 s (Legacy) |
| 6 | Vollständiger Lauf, Bericht | ✅ `reference/v2-standard/`, `reference/laeufe/g_dir_hierarchisch/` |
| – | Flexibler Prior der Taxa-Gewichte | ✅ Dir(α) fest oder hierarchisch |

## Wichtigste Ergebnisse (Details: `docs/pruefbericht.md`)

1. **Fehler in der Hastings-Korrektur der Taxa-Gewichte:** κ fehlt. Belegt mit einem Prior-only-Test.
2. **Klimarekonstruktion robust:** Unter allen Korrekturen und Priors ändern sich die Mediane um
   ≤ 0,17 °C und ≤ 12 mm. Die Unsicherheit von P war um etwa 10–15 % unterschätzt.
3. **CI-Verhältnis der Transferfunktionen:** Für das IT- und das SA-Biom ist die Verengung ein
   Trunkierungseffekt. Echte Information aus den Daten gibt es nur für das mediterrane Biom (~20 %).
4. **Taxa-Gewichte:** Die Daten bestimmen sie nur schwach. Die Rangfolge ist ab α ≥ 1 bzw. im
   hierarchischen Modell stabil (*Amaranthaceae*, *Q. calliprinos* ↑; *Olea* ↓), die Beträge hängen
   vom Prior ab.
5. **Neuer Sampler (log-Gamma-Random-Walk):** gleiche Zielverteilung, etwa 50-mal effizienter als
   der Dirichlet-Random-Walk.

## Entscheidungen

| Datum | Entscheidung |
|---|---|
| 26.10.2026 | Originalcode bleibt erhalten (`legacy/`, Commit `782c6a0`), Korrekturen kommen ins neue Paket |
| 26.10.2026 | C++ bleibt für den Sampler; die Auswertung läuft in NumPy und C++ (Dichteschätzung) |
| 26.10.2026 | Der Prior der Taxa-Gewichte muss nicht starr sein. Der Autor bevorzugt den **hierarchischen Prior** (sauberer) |

## Offene Punkte (nächste Sitzung)

1. **Paper hochladen:** im Branch `claude/netzel-python-package-analysis-adcv39` unter
   `docs/paper/cp-21-357-2025.pdf`, das Supplement in denselben Ordner. Danach Schritt 2
   (Abgleich Paper ↔ Code) und den Prüfbericht ergänzen. Zu klären:
   - Steht das fehlende κ so im Paper?
   - Wie ist die 5-%/95-%-Trunkierung dort begründet?
2. **Standard auf den hierarchischen Prior umstellen:** In `SamplerConfig` steht noch
   `taxa_prior = "fixed"`. Umzustellen sind `taxa_prior = "hierarchical"` und die
   Standardlänge (8 Mio. Iterationen, Burn-in 2 Mio., Ausdünnung 600), danach Tests und README.
   Bis dahin den hierarchischen Prior über `configs/g_dir_hierarchisch.toml` verwenden.
3. **Tag `v1.0-paper` auf GitHub anlegen:** Er zeigt auf Commit `782c6a0`. Der Push des Tags war
   aus der Arbeitsumgebung nicht möglich. Es gibt außerdem einen Tag `v0.1.0` auf `1c7f9b2`, drei
   Commits früher. Zu klären ist, ob das die eigentliche Paper-Version ist.
4. **Laufzeit des hierarchischen Laufs:** etwa 13 min. Der Prior-Referenzlauf nimmt die Hälfte davon
   ein und ließe sich kürzen.
5. **Später:** Training der Transferfunktion, weitere Standorte, Mergen des Branches in `main`.

## Reproduktion

```bash
pip install -e ".[test]"
pytest                                    # schnelle Tests (~45 s)
pytest -m slow                            # lange Tests (Bitgleichheit mit dem Original, hierarchischer Prior)
kinneret-recon run --legacy               # Original, ~1 min
kinneret-recon run -c configs/standard.toml
kinneret-recon run -c configs/g_dir_hierarchisch.toml
```

Alle Läufe der Vergleichsreihe: `configs/*.toml`. Gesicherte Ergebnisse (Tabellen, Diagnostik,
Einstellungen, Metadaten): `reference/laeufe/<lauf>/`, Vergleiche in `reference/vergleiche/`.
Die großen Dateien (`posterior.npz`, `results.npz`, Abbildungen) sind nicht im Repo. Jeder Lauf ist
mit seiner `config.toml` (inkl. Seed) bitgenau wiederholbar.

## Arbeitsumgebung

- Das Netzwerk der Cloud-Umgebung erlaubt keinen Zugriff auf `cp.copernicus.org`. Deshalb muss das
  Paper ins Repo hochgeladen werden.
- Gemessen auf 4 CPU-Kernen mit Python 3.13, NumPy 2.5, SciPy 1.18, Matplotlib 3.11 und g++ 13.
