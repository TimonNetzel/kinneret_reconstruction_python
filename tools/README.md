# Hilfsskripte

Diese Skripte wurden bei der Prüfung und Abstimmung verwendet. Sie werden aus dem Hauptverzeichnis
aufgerufen, nachdem das Paket installiert ist (`pip install -e .`).

| Skript | Zweck | Aufruf |
|---|---|---|
| `analyse.py` | Kennzahlen mehrerer Läufe (Mediane, IQR, R², Gewichte, CI-Verhältnisse, ESS, R̂) | `python tools/analyse.py legacy standard g_dir_hierarchisch` (Namen der Ordner in `results/`) |
| `tune.py` | Sampler-Einstellungen vergleichen (Akzeptanz, min. ESS, max. R̂), nur Sampling | `python tools/tune.py '{"weights_proposal": "log_gamma"}' '{"dirichlet_spread_factor": 4}'` |
| `prior_only.py` | Prior-only-Validierung: Dirichlet-Vorschlag mit und ohne korrekte Hastings-Korrektur | `python tools/prior_only.py 400000` |
| `legacy_post.py` | Originalskript ohne erneutes MCMC (nur Auswertung und Abbildungen) | `cd legacy && python ../tools/legacy_post.py` |
| `runtimer.py` | Laufzeit und Spitzenspeicher eines Befehls messen | `python tools/runtimer.py python3 main.py` |
| `anomalien.py` | Niveau, IQR und Korrelation der Anomalien mehrerer Läufe (Tabellen in Prüfbericht 3.8 und 3.9); liest `results/` oder `reference/laeufe/` | `python tools/anomalien.py standard o_cru_r25 o_cru_r50 o_cru_r100` |
| `karte_cru.py` | Karte des CRU-Ausschnitts mit den Einzugsgebieten | `python tools/karte_cru.py` |
