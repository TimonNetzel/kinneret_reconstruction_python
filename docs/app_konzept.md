# App-Konzept: Rekonstruktion und Sensitivitätsanalyse im Browser

Stand: 28.10.2026. Grundlage für die Entwicklung der App in einer neuen Sitzung. Die Antworten des Autors
vom 28.10.2026 sind eingearbeitet (Abschnitt 1). Vor dem Start außerdem `docs/projektstand.md` lesen.

## 1. Rahmen (vom Autor festgelegt)

| Frage | Festlegung |
|---|---|
| Nutzer | nur der Autor, lokal |
| Rechnen | vorerst nur **fertige Läufe anzeigen**; Läufe werden per Kommandozeile gerechnet |
| Plattform | Browser, optimiert für **PC und Handy** |
| Sprache | Oberfläche **Deutsch und Englisch** (umschaltbar) |
| Inhalt | **alle bisherigen Läufe** und künftige; die App muss in weiteren Sitzungen **leicht erweiterbar** sein (Ansichten, Läufe, Faktoren) |
| Standorte | **leicht für weitere Standorte erweiterbar** |
| Vergleichsdaten (Overlay) | noch keine Daten; nur eine Schnittstelle vorsehen, nicht bauen |
| Abbildungsexport | im Format der Zeitschrift *Climate of the Past* (Copernicus) |
| Kulturbäume | „AP ohne Kulturbäume“ als **Variante** anbieten; Standard bleibt v3 (Begründung: Prüfbericht 3.11, die Einordnung von *Olea* im Mittelholozän ist fachlich offen) |

## 2. Architektur

Rechnen und Anzeigen sind getrennt. Ein Lauf dauert 4–11 min, eine Ansicht muss sofort reagieren.

```
configs/*.toml ──kinneret-recon run──▶ results/<lauf>/ (Ketten, ~60 MB)
                                            │
                          kinneret-recon app-daten  (neu)
                                            ▼
                     reference/laeufe/<lauf>/app.json (klein, im Repo)
                                            │
                          kinneret-recon app  (neu)
                                            ▼
                         App im Browser (PC und Handy)
```

1. **Lauf-Katalog.** Jeder Lauf ist ein Ordner mit `config.toml`, `meta.json` und Ergebnissen. Neu
   kommt eine **Beschreibung je Lauf** hinzu (z. B. `lauf.toml`): Titel und Kurzbeschreibung (de/en),
   Standort, Rolle (Referenz, Sensitivität, Validierung, Experiment), Basislauf, Faktor und Stufe (z. B.
   Faktor „CRU-Radius“, Stufe „25 km“). So kann die App Läufe gruppieren und Sensitivitäten automatisch
   zusammenstellen.
2. **Kompakte Zusammenfassung je Lauf** (`app.json` oder `app.npz`, Ziel < 1 MB, im Repo). Enthält alles,
   was die Ansichten brauchen: Quantile von T und P je Alter und Tiefe (Mittelwert und Mischverteilung),
   Dichten der Mischverteilung auf einem groben Raster, Biomanteile, AP-Kurve, Quantile der
   Taxa-Gewichte, R², T0/P0, CI-Verhältnisse, Diagnostik. Die großen Ketten bleiben außerhalb des Repos.
3. **Faktor-Register im Code.** Jeder Sensitivitätsfaktor ist ein Eintrag (Schlüssel, Bezeichnung de/en,
   Konfigurationsfeld, Stufen, Gruppe). Ein neuer Faktor ist ein neuer Eintrag plus Läufe, keine neue
   Ansicht.
4. **Standort-Register.** Ein Standort ist ein Ordner mit Kerndaten, Altersmodell, Transferfunktion,
   Koordinaten, Seehöhe und den Einstellungen für den CRU-Prior (heute fest: See Genezareth in
   `cru.py` und `data/in/`). Die App zeigt einen Standort-Wähler, sobald es mehr als einen gibt.
5. **Overlay-Schnittstelle** (nur vorsehen): CSV mit Alter, Wert, Unsicherheit, Einheit, Quelle; wird
   auf der Zeitachse über die Rekonstruktion gelegt.

**Technik (Vorschlag, in der neuen Sitzung entscheiden):** Da vorerst nur angezeigt wird, genügt eine
**statische Web-App**: `kinneret-recon app` erzeugt aus dem Katalog einen Ordner mit HTML, JS und den
Zusammenfassungen und startet einen kleinen lokalen Server. Das Handy greift über das heimische WLAN
darauf zu (oder der Ordner wird aufs Handy kopiert). Diagramme mit Plotly.js oder ECharts (interaktiv,
responsiv). Vorteile: volle Kontrolle über das responsive Layout, keine Serverlogik, später leicht um
ein Backend (Läufe starten) erweiterbar. Alternative: Python-Frameworks wie Panel, Dash oder Streamlit
(schneller gebaut, auf dem Handy aber weniger gut gestaltbar).

## 3. Ansichten

1. **Rekonstruktion:** T und P über Alter oder Tiefe, Median mit 50- und 95-%-Bändern; Mischverteilung
   als Dichte statt Quantilen (sonst irreführend, Prüfbericht 3.10). Darunter AP-Kurve und Biomanteile.
   Verknüpft: Antippen eines Alters zeigt Biomanteile, Spektrum und Verteilung von T und P dort.
2. **Sensitivität (Kern der App):** Basislauf wählen, Varianten überlagern; Differenz zur Basis über
   das Alter; Übersicht je Faktor mit Niveau-Verschiebung, Änderung des IQR und Korrelation der
   Anomalien (wie die Tabellen in Prüfbericht 3.8–3.11). Faktoren gruppiert, z. B.:
   - Sampler-Korrekturen (Original, A–C, Standard)
   - Prior der Taxa-Gewichte (Dir(0,5), Dir(1), Dir(2), hierarchisch)
   - Prior des rezenten Klimas (Paper, Gitterpunkt, enge Sd, CRU 25/50/100 km)
   - R²-Prior (0,25 / 0,5 / 0,75; Beta(2,62; 2,62))
   - Baumpollenkurve (mit/ohne Kulturbäume, Kalibrierung 4–9 ka)
   - Modellform (Mittelwert/Mischverteilung, N_t als Prior, Trunkierung 5/95 % gegen 1/99 %)
3. **Taxa-Gewichte:** je Lauf und im Vergleich, mit Prior-Median.
4. **Transferfunktionen:** Klimaräume der Biome, CI-Verhältnis Posterior/Prior.
5. **Diagnostik** (eingeklappt): R̂, ESS, Akzeptanz, Spuren.
6. **Lauf-Steckbrief:** Einstellungen, Methodentext (aus `report.py`), Code-Stand, Laufzeit.
7. **Export:** Abbildungen als PDF/PNG im Copernicus-Format (Spaltenbreiten und Schriftgrößen in der
   neuen Sitzung anhand der aktuellen Autorenrichtlinien prüfen), CSV der angezeigten Daten.

**Gestaltung:** eine Farbe je Lauf über alle Ansichten, farbsehschwächen-tauglich; Hell- und
Dunkelmodus; auf dem Handy eine Spalte mit Reitern, auf dem PC Seitenleiste plus Diagramme;
Erklärungen zu Optionen als aufklappbare Hinweise (de/en).

## 4. Läufe

**Bestand** (24 Konfigurationen in `configs/`, Ergebnisse in `reference/laeufe/`): Original, A–C, E,
F1/F2, G, H25/H75, I25/I75, K, L, M (3), O (3), P (2), Standard v2, Standard v3, Mischverteilung v2 und
v3.

**Wichtig:** Im Repo liegen je Lauf nur Quantil-Tabellen und Diagnostik. Für Gewichte,
Mischverteilungsdichten und Transferfunktionen braucht die App die vollständigen Ergebnisse. Die neue
Sitzung muss die Läufe daher neu rechnen und daraus die kompakten Zusammenfassungen erzeugen
(Batch-Befehl vorsehen). Geschätzte Rechenzeit auf 4 Kernen: etwa 2 h im Hintergrund (v2-basierte Läufe
je ~4 min, v3-basierte je ~11 min, Original-Läufe je ~1 min). Mit gleichem Code-Stand sind die Läufe
bitgleich zu den bisherigen.

**Erweiterungen für das Anbieten von Läufen:**

- Voreinstellungen für einen Klick (Original, v2, v3, v3 ohne Kulturbäume, CRU-Radien, R²-Prior).
- Sensitivitäts-Entwurf: Faktoren und Stufen wählen, Konfigurationen erzeugen („ein Faktor nach dem
  anderen um den Standard“ als Vorgabe, volles Raster optional). Vorerst als Kommandozeile, später in
  der App.
- Fingerabdruck je Konfiguration, damit gleiche Läufe nicht doppelt gerechnet werden.
- Auswertungsvarianten ohne neues Sampling (Mittelwert/Mischverteilung, Quantil-Methode).
- Prior-Referenzlauf kürzen (halbiert die Laufzeit der v3-Läufe; offener Punkt im Projektstand).

## 5. Vorgeschlagene Reihenfolge

1. Lauf-Beschreibungen (`lauf.toml`), Faktor-Register, kompakte Zusammenfassung, Batch-Befehl; alle
   Läufe neu rechnen und Zusammenfassungen ins Repo.
2. App-Grundgerüst (statisch, responsiv, de/en) mit Rekonstruktion und Sensitivität.
3. Taxa-Gewichte, Transferfunktionen, Diagnostik, Steckbrief.
4. Abbildungsexport im Copernicus-Format.
5. Standort-Register (Kinneret als erster Standort).
6. Später: Läufe aus der App starten, Overlays.

## 6. Offene Entscheidungen für die neue Sitzung

- Technik: statische Web-App (Vorschlag) oder Python-Framework.
- Format der Zusammenfassung (JSON oder npz) und Rasterauflösung der Dichten.
- Zugriff vom Handy: über das heimische WLAN oder als kopierter Ordner.
