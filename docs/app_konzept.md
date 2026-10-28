# App-Konzept: Rekonstruktion und Sensitivitätsanalyse im Browser

Stand: 28.10.2026. Grundlage für die Entwicklung der App. Die Antworten des Autors vom 28.10.2026 sind
eingearbeitet (Abschnitt 1). Schritte 1 bis 3 sind umgesetzt (Abschnitte 7, 9 und 10). Vor dem Weiterarbeiten außerdem
`docs/projektstand.md` lesen.

## 1. Rahmen (vom Autor festgelegt)

| Frage | Festlegung |
|---|---|
| Nutzer | nur der Autor, lokal |
| Rechnen | vorerst nur **fertige Läufe anzeigen**; Läufe werden per Kommandozeile gerechnet |
| Plattform | Browser, optimiert für **PC und Handy** (Handy: **Android**) |
| Offline | Die App läuft **ohne Netz** (28.10.2026): auf dem PC als Ordner per Doppelklick auf `index.html` (kein Server nötig), auf dem Handy als **eine einzige HTML-Datei** zum Kopieren. Keine Abhängigkeiten aus dem Netz (Bibliotheken im Repo). Dazu Manifest und Service Worker, damit sie über `localhost` oder HTTPS auch als installierbare Web-App (PWA) offline läuft |
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

**Technik (entschieden 28.10.2026, Abschnitt 8):** statische, offline-fähige Web-App. Ursprünglicher
Vorschlag: Da vorerst nur angezeigt wird, genügt eine **statische Web-App**: `kinneret-recon app` erzeugt aus dem Katalog einen Ordner mit HTML, JS und den
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

## 6. Offene Entscheidungen

- ~~Technik~~: statische, offline-fähige Web-App mit Plotly.js (Abschnitt 8).
- ~~Format der Zusammenfassung und Rasterauflösung~~: entschieden in Schritt 1 (Abschnitt 7): JSON, Raster
  T −10…30 °C in 0,5 °C und P 0…1500 mm in 20 mm.
- ~~Zugriff vom Handy~~: Android, eine kopierte HTML-Datei (offline); über das WLAN optional
  (`kinneret-recon app --serve`).

## 7. Umsetzung Schritt 1 (28.10.2026)

**Lauf-Katalog.** Jeder Lauf ist ein Ordner `reference/laeufe/<id>/` mit `lauf.toml`; die `id` ist der
Ordnername. Felder:

| Feld | Bedeutung |
|---|---|
| `config` | Konfiguration, z. B. `"configs/o_cru_r25.toml"` |
| `standort` | Schlüssel im Standort-Register (`"kinneret"`) |
| `rolle` | `referenz`, `variante`, `sensitivitaet`, `validierung` oder `experiment` |
| `basis` | Basislauf (optional); die Konfiguration darf von ihm nur in den Feldern des Faktors und in Laufzeit-Feldern (Länge, Burn-in, Ausdünnung, Seed) abweichen |
| `faktor`, `stufe` | Schlüssel im Faktor-Register; die Konfiguration muss die Werte der Stufe haben |
| `quelle` | Abschnitt im Prüfbericht |
| `[titel]`, `[beschreibung]` | je `de` und `en`; der Titel dient als Legende |
| `[hinweis]` | optional, je `de` und `en` (z. B. „Kette hängt“) |

`kinneret-recon katalog` prüft alle Regeln. Der Katalog umfasst 27 Läufe: die 24 bisherigen Konfigurationen,
das Original (`configs/original.toml`, gleichwertig zu `--legacy`) und die beiden Mischverteilungen
(`configs/n_mixture_v2.toml`, `configs/v3_mixture.toml`). Standard v3 hat den neuen Ordner
`reference/laeufe/standard_v3/`. Nicht im Katalog ist `reference/laeufe/standard`: derselbe Lauf wie
`standard_aktuell` (Standard v2) mit einem älteren Code-Stand.

**Faktor-Register** (`FAKTOREN` in `src/kinneret_recon/katalog.py`): Schlüssel, Gruppe, Bezeichnung und
Beschreibung (de/en), erlaubte Konfigurationsfelder und Stufen mit ihren Werten. Faktoren: `standardversion`,
`korrekturen`, `gewichte_prior`, `klima_prior`, `r2_prior`, `ap_taxa`, `kalibrierung_alter`,
`rekonstruktion`, `tf_prior`, `tf_grenzen` (Gruppen wie in Abschnitt 3). Die Stufe jedes Laufs für jeden
Faktor wird aus seiner Konfiguration bestimmt; so steht z. B. Standard v2 automatisch als Stufe „Dir(0,5)“
in der Reihe der Gewichte-Priors.

**Fingerabdruck.** Prüfsumme der wirksamen Einstellungen (Prior des rezenten Klimas aus den Eingangsdaten
aufgelöst) und der Eingangsdaten. Ein zweiter Fingerabdruck nur über die Sampler-Einstellungen erkennt Läufe
mit gleichen Ketten (Mittelwert/Mischverteilung, `c_hastings_gamma`/`standard_aktuell`); `batch` sampelt sie
nur einmal.

**Befehle:** `kinneret-recon katalog`, `kinneret-recon batch` (fehlende oder veraltete Läufe rechnen, dann
App-Daten), `kinneret-recon app-daten` (nur Zusammenfassungen schreiben).

**Ausgaben:**

- `reference/laeufe/<id>/app.json` je Lauf (Modul `app_daten.py`): Beschreibung, wirksame Einstellungen,
  Fingerabdrücke, Metadaten, Methodentext (de); Rekonstruktion je Alter und Tiefe als Mittelwert *und*
  Mischverteilung (Quantile 2,5/25/50/75/97,5 % und Mittel; Mischverteilung zusätzlich als Dichte auf dem
  groben Raster), die Quartile, wie der Lauf sie berichtet; Biomanteile (Quantile); Baumpollenkurve, wie sie
  in R² eingeht; R², T0, P0 mit Quantilen, Prior-Parametern und Kurven (Posterior/Prior); Taxa-Gewichte
  (Quantile, Prior-Quantile, α); Transferfunktionen (Quantile Posterior und Prior-Referenz, Grenzen,
  CI-Verhältnis, 2D-Dichte der Posterior); Diagnostik (Tabelle, Akzeptanz, Spuren mit 300 Punkten je Kette);
  Kennzahlen. Dichten sind je Zeile auf 100 normiert und ganzzahlig.
- `reference/app/standort_kinneret.json`: Taxa mit Biom, Gehölz- und Kulturbaum-Kennung, Spektren je Alter und
  Tiefe, Baumpollenkurve mit und ohne Kulturbäume, Prior der Transferfunktion als 2D-Dichte.
- `reference/app/katalog.json`: Register (Gruppen, Rollen, Faktoren mit Stufen, Standorte) und alle Läufe
  mit Beschreibung, Stufen je Faktor, Kennzahlen und Vergleich mit dem Basislauf (Niveau, IQR, Korrelation
  der Anomalien wie in Prüfbericht 3.8–3.11).

Fehlende Tabellen und Metadaten eines Laufs (bisher nur bei `standard_v3`) kopiert `app-daten` in den
Katalog-Ordner. Die gesicherten Tabellen der übrigen Läufe bleiben, wie sie im Prüfbericht ausgewertet
wurden (`--ueberschreiben` ersetzt sie); `app.json` beschreibt den neu gerechneten Lauf mit seinem
Code-Stand (`meta.git`).

**Ergebnis (28.10.2026):** Alle 27 Läufe neu gerechnet (~2 h auf 4 Kernen), `app.json` je 284–381 KB,
`katalog.json` 53 KB, `standort_kinneret.json` 115 KB. 21 Läufe sind bitgleich mit den gesicherten Tabellen,
darunter Original, Standard v2 und v3. Fünf Läufe aus älteren Code-Ständen (`b_hastings`, `c_hastings_gamma`,
`e_tf_grenzen_1_99`, `f_dir_alpha_1`, `f_dir_alpha_2`) und die Mischverteilung v2 (Ketten des alten
Standardlaufs) weichen nur durch eine andere Zufallszahlenfolge ab: Mediane im Mittel ≤ 0,03 °C und 3 mm,
höchstens 0,06 °C und 3,1 mm (Mischverteilung 7,6 mm). Die Vergleiche mit dem Basislauf in `katalog.json`
reproduzieren die Tabellen in Prüfbericht 3.8–3.11.


## 8. Offline-Fähigkeit (Festlegung vom 28.10.2026)

Der Autor nutzt ein Android-Handy und möchte die App ohne Netz verwenden. Umsetzung:

- **Keine Netz-Abhängigkeiten:** Plotly.js (Cartesian-Paket, MIT-Lizenz) liegt im Paket
  (`src/kinneret_recon/app/vendor/`), Schriften vom System, keine externen Aufrufe.
- **PC:** `kinneret-recon app` baut einen Ordner (Standard `results/app/`). `index.html` per Doppelklick
  öffnen; die Daten liegen als JS-Dateien bei, weil Browser über `file://` kein JSON nachladen. Die Daten
  eines Laufs werden erst geladen, wenn er angezeigt wird.
- **Handy (Android):** `kinneret-recon app --eine-datei` schreibt eine einzige HTML-Datei mit allem Nötigen;
  die Daten der Läufe sind gzip-komprimiert eingebettet und werden im Browser entpackt
  (`DecompressionStream`). Datei aufs Handy kopieren und in Chrome oder Firefox öffnen.
- **Installierbar (PWA):** Manifest und Service Worker sind dabei. Sie greifen nur über `localhost` oder
  HTTPS (Vorgabe der Browser), nicht über `file://` oder einfaches `http://` im WLAN, und stören dort nicht.
- **WLAN (optional):** `kinneret-recon app --serve` startet einen kleinen lokalen Server; das Handy öffnet die
  angezeigte Adresse (dann online im heimischen Netz, ohne Offline-Speicher).

## 9. Umsetzung Schritt 2 (28.10.2026)

**Quellen** in `src/kinneret_recon/app/`: `index.html`, `style.css`, `i18n.js` (Texte de/en), die Logik
(seit Schritt 3 aufgeteilt, Abschnitt 10), `manifest.webmanifest`, `icon.svg`, `vendor/plotly-cartesian.min.js` (Plotly.js 2.35.2, MIT-Lizenz,
`vendor/LICENSE-plotly.txt`). Kein Build-Werkzeug nötig, klassische Skripte (ES-Module laden über `file://`
nicht). Gebaut wird mit `kinneret-recon app` (Modul `app_bauen.py`); das Ergebnis liegt in `results/` und
ist nicht im Repo.

**Ansichten:**

- *Rekonstruktion:* Lauf (gruppiert nach Rolle, ★ = Standard), Achse Alter/Tiefe, Mittelwert oder
  Mischverteilung (Voreinstellung: Standard des Laufs). Vier verknüpfte Diagramme: T und P (Median mit
  50-/95-%-Bändern bzw. Dichte der Mischverteilung), Biomanteile, Baumpollenkurve wie in R² (gestrichelt die
  volle Kurve, grau die Alter außerhalb der Kalibrierung). Antippen eines Alters: Verteilung von T und P
  (Dichte der Mischverteilung, darüber Median, 50 % und 95 % des Mittelwerts), Biomanteile, häufigste Taxa.
  Beschreibung, Faktor, Basislauf, Quelle und Kennzahlen aufklappbar, Hinweise (z. B. „Kette hängt“) immer
  sichtbar.
- *Sensitivität:* Faktor (gruppiert), Basislauf (Voreinstellung: Wurzel der Reihe mit den meisten
  Varianten), Läufe der Reihe zum An- und Abwählen, weitere beliebige Läufe hinzufügbar, 50-%-Bänder
  optional. Diagramme: Median von T und P, Differenz zum Basislauf; Tabelle mit Stufe, Niveau, ΔNiveau, IQR
  (mit relativer Änderung), *r* der Anomalien und min. ESS. Die Werte rechnet die App aus den Laufdaten, für
  jeden Basislauf; sie stimmen mit Prüfbericht 3.8–3.11 überein.

**Gestaltung:** eine Farbe je Lauf über alle Ansichten (Paletten nach Tol und Okabe-Ito), der Basislauf in
Schwarz bzw. Weiß; Hell/Dunkel/Automatisch; ab 1000 px Seitenleiste plus Diagramme, darunter eine Spalte;
Einstellungen bleiben im Browser gespeichert (`localStorage`, ohne Speicher Voreinstellungen).

**Geprüft** (`tools/app_pruefen.js`, Chromium ohne Netz, auch in `tests/test_app.py`): Ordner und eine Datei,
Handy (390 px) und PC (1366 px), hell und dunkel, ohne Fehler in der Konsole und ohne Netzanfragen. Über
`--serve` und `localhost` speichert der Service Worker alle Dateien; die App lädt danach auch offline neu.

## 10. Umsetzung Schritt 3 (28.10.2026)

**Aufbau der Logik** (erweiterbar): `kern.js` (Zustand, Laden der Daten, Farben, Plotly-Helfer, Texte,
Vergleichsauswahl) und je Ansicht eine Datei, die sich mit `KR.ansicht({key, init, zeige})` anmeldet:
`rekonstruktion.js`, `sensitivitaet.js`, `gewichte.js`, `transferfunktionen.js`, `lauf.js`; `start.js`
erzeugt daraus die Reiter. **Eine neue Ansicht** ist eine neue Datei plus ein `<section id="ansicht-<key>">` in
`index.html`, ein Eintrag in `SKRIPTE` (`app_bauen.py`) und die Texte in `i18n.js` (`ansicht_<key>` usw.).

**Neue Ansichten:**

- *Taxa-Gewichte:* Posterior der 33 Gewichte (Median, 50 %, 95 %) eines Laufs und von bis zu sechs
  Vergleichsläufen; Prior-Median je Lauf (gestrichelt) und 50-%-Bereich des Priors (grau), 1/33 gepunktet;
  Biom als Farbe am Namen; Sortierung nach Median, Biom oder Name; lineare oder logarithmische Achse; α des
  Priors (fest oder Posterior des hierarchischen α).
- *Transferfunktionen:* je Biom der Klimaraum: Prior der TF (Pr = 0,5, gestrichelt), Posterior der
  Klimawerte als Fläche, Vergleichsläufe als Linien, Median mit 95 %, Trunkierungsbereich; darunter das
  CI-Verhältnis Posterior/Prior für T und P und eine Tabelle; Erklärung der Prior-Referenz.
- *Lauf* (Steckbrief): Beschreibung, Rolle, Faktor und Stufe, Basislauf, Quelle, Konfiguration,
  Fingerabdrücke, Code-Stand, Umgebung, Ketten und Laufzeit; **Methodentext auf Deutsch und Englisch** mit
  Kopierknopf (für Manuskripte); Kalibrierung (R², T0, P0: Posterior gegen Prior); alle wirksamen
  Einstellungen, Abweichungen vom Basislauf markiert; eingeklappt die **Konvergenzdiagnostik** (Status,
  Akzeptanz, Spuren aller Ketten mit Ende des Burn-in, R̂ und ESS aller Größen, Tabelle).

Der Vergleich ist für Gewichte und Transferfunktionen gemeinsam, der gewählte Lauf für alle Ansichten außer
der Sensitivität. Der englische Methodentext entsteht in `report.methods_text(run, "en")`; der deutsche ist
unverändert.

**Offen für Schritt 4:** Export der Abbildungen im Copernicus-Format und CSV der angezeigten Daten (bisher nur
PNG über die Plotly-Leiste).

## 11. Formelschreibweise (28.10.2026, vom Autor gewünscht)

Formelzeichen stehen in allen Texten in **LaTeX-Schreibweise** zwischen `$...$`, z. B. `$T_\mathrm{DJF}$`,
`$P_\mathrm{ANN}$`, `$\Delta T_\mathrm{DJF}$`, `$T_0$`, `$R^2$`, `$\hat{R}$`, `$N_\mathrm{t}(\mu, \sigma)$`.
Das gilt für die Oberflächentexte (`i18n.js`), Faktor-Register, Lauf-Beschreibungen (`lauf.toml`), den
Methodentext und den HTML-Bericht.

- **Umsetzung:** `formel.py` (Python) und `formel()` in `kern.js` (App) setzen eine kleine LaTeX-Teilmenge um,
  nach HTML mit kursiven Variablen und echten Indizes (`<i>T</i><sub>DJF</sub>`), nach Unicode-Text
  (`T_DJF`, `T₀`, `R²`; für Auswahllisten und Tooltips, die kein HTML können) oder nach LaTeX (Text außerhalb
  der Formeln maskiert, z. B. `%` → `\%`). Plotly versteht dieselbe HTML-Auszeichnung; Diagramme und Text
  sehen daher gleich aus, ohne MathJax (die App bleibt klein und offline). Beide Umsetzungen werden
  gegeneinander getestet (`tests/test_formel.py`).
- **Methodentext:** „Kopieren“ legt ihn formatiert (HTML, z. B. für Word) und als Unicode-Text in die
  Zwischenablage, „LaTeX“ als LaTeX-Quelltext (für die Copernicus-Vorlage).
- **Diagnostik:** Parameternamen werden gesetzt angezeigt (`T0` → $T_0$, `T_Med` → $T_\mathrm{Med}$,
  `w[…]` → $w$[…]); in `diagnostik.csv` bleiben die Bezeichner.
- **Matplotlib:** Neue Abbildungen (Spuren, R̂/ESS, Vergleiche) verwenden `$T_\mathrm{DJF}$` usw. Die
  Abbildungen des Originals behalten `$\mathregular{T_{DJF}}$` (aufrecht, aber ebenfalls mit Index), damit sie
  bitidentisch zum Originalskript bleiben.
- **Nicht geändert:** Spaltennamen der Tabellen (`T_DJF_q50` …) und Bezeichner im Code.
