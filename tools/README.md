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
