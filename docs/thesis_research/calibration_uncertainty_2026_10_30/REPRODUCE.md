# Reproduktion der bedingten Block-Sensitivität

Frischer Checkout dieses Branches; #31 ist als gesperrte Basis enthalten.
Keine temporären Dateien aus einem früheren Chat erforderlich. Keine neuen
Netzfits und kein neues SMOTE; nur die Kalibrierungsstufe wird blockweise variiert.
Vorabdesign unverändert veröffentlicht in `5a4a96f1d7c01b64a230fb8ad6a7fbaf84857697`.

```bash
python -m pip install -r docs/thesis_research/smote_nnet_2026_10_30/requirements-research.txt
python tools/taxon_smote/prepare.py \
  --sources docs/thesis_research/smote_nnet_2026_10_30/sources.json \
  --audit-sources docs/thesis_research/smote_nnet_2026_10_30/audit_sources.json \
  --previous work/pr28 --maps work/maps --climate work/wc2.1_10m_bio.zip
```

Bestehende Quellen werden nur bei exakten Hashes akzeptiert, nicht überschrieben.
Meusel/BfN-Attribution/Lizenz im gesperrten #28-Quellenregister; WorldClim-
Roharchive werden nicht neu verteilt. Die Forschungsabhängigkeiten ändern die
App nicht. Python- und Bibliotheksversionen stehen im Laufmanifest.

## Vollständige Läufe

Neue Ausgabeverzeichnisse verwenden. Sechs unabhängige Mitgliedsaufträge je
Ensemble in Prozessen, je ein BLAS-Thread. Seeds und Ausgabereihenfolge hängen
nicht von der Prozessreihenfolge ab. Jeder Auftrag bearbeitet seine 200 Replikate
in fester Reihenfolge. Dateien erst nach Programmende lesen; `.part`-Schreiber
werden vor Abschluss geschlossen und umbenannt, keine laufenden Checkpoints.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/taxon_uncertainty/run.py \
  --previous work/pr28 --maps work/maps --climate work/wc2.1_10m_bio.zip \
  --config docs/thesis_research/calibration_uncertainty_2026_10_30/config.json \
  --output work/uncertainty_run1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/taxon_uncertainty/run.py \
  --previous work/pr28 --maps work/maps --climate work/wc2.1_10m_bio.zip \
  --config docs/thesis_research/calibration_uncertainty_2026_10_30/config.json \
  --output work/uncertainty_run2
python tools/taxon_smote/repeat.py \
  --first work/uncertainty_run1 --second work/uncertainty_run2 \
  --report work/uncertainty_repeat.json
```

`parameters.csv.gz`: sämtliche Mitglied×Replikat-Parameter, Klassenanzahlen,
Zielfunktionen, KKT-Reste und Status. `block_draws.csv.gz`: alle nichtnulligen
Blockmultiplizitäten (fehlende Blöcke haben Multiplizität null).
`intervals.csv.gz`: äußere und Vollgebiets-Zellprognosen samt Support,
Verfügbarkeitsstatus und gegebenenfalls drei Quantilen. Große Zell-/Ziehungstabellen
werden anhand ihrer Manifest-Hashes reproduziert. Deterministisches Gzip ohne
Zeit-/Dateinamenstempel; keine plattformübergreifende Bitgleichheit zugesichert.

## Separater Audit

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/taxon_uncertainty/verify.py \
  --previous work/pr28 --maps work/maps --climate work/wc2.1_10m_bio.zip \
  --config docs/thesis_research/calibration_uncertainty_2026_10_30/config.json \
  --output work/uncertainty_run1 --report work/uncertainty_verification.json
python -m unittest discover -s tests -p test_calibration_uncertainty.py -v
```

Der Prüfer importiert weder neuen Produzenten noch Bootstrap-Fitter. Er verwendet
hashgesperrte alte Auditwerkzeuge, Vektorlabels, direkte TIFF-Indizes,
Haversine-BallTree, separat tabulierte Blockzählungen, tanh-Netzrechnung,
eigene KKT-Ziele und sortierte interpolierte Quantile. Gemeinsame Decoder und
Numerik sind offengelegt, keine unabhängige Bestätigung der Kartenwahrheit.

## Vorwärtsanwendung

Nur die unveränderten #31-Modelle am gesperrten Commit verwenden; ihr Hash ist
im neuen `config.json` hinterlegt. Das volle Parameterregister bewahrt auch
Replikate ohne schätzbare Kalibrierung. Der Vorwärtsevaluator erzwingt das
vorab festgelegte Mindestgate, statt fehlende Bänder durch Nullbreite zu ersetzen.

```python
import json
import sys
sys.path.insert(0, 'tools/taxon_uncertainty')
from bootstrap import predict_interval

models = json.load(open('docs/thesis_research/calibration_2026_10_30/models.json'))
draws = json.load(open('docs/thesis_research/calibration_uncertainty_2026_10_30/full_parameters.json'))
members = models['betula_pendula']['5']['original']
record = draws['betula_pendula']['5']['original']
point, quantiles, support = predict_interval(members, record, [[8., 800.]])
```

`quantiles`: punktweise 0,025 / 0,5 / 0,975, **bedingt auf feste Netze und
schätzbare Replikate**. Keine nominal validierte Deckung und kein fossiles
Fehlerband. Extrapolation außerhalb irgendeiner Mitgliedshülle wird standardmäßig
verweigert. Keine Pollentyp-Zuordnung, CLI/App-Integration oder Pilotfreigabe.

## Technischer Prüfhinweis

Ein früher Lauf hatte unvollständige Gzip-Protokolle und wird nicht als
erfolgreicher Reproduktionsnachweis verwendet. Der unabhängige Audit fand
zusätzlich eine konservative Fehlklassifikation eines exakten gespeicherten
Klimahülleneckpunkts durch Qhull/Delaunay. Der neue Evaluator schließt solche
exakt gleichen Eckpunkte definitionsgemäß ein; kein größerer Extrapolationspuffer,
keine geänderten Hüllen, Gewichte oder Kalibrierung. Regressionstest vorhanden.
[execution_notes.json](./execution_notes.json) dokumentiert beide Befunde.
Die maßgeblichen Endläufe werden mit korrigierter technischer Randfallbehandlung
und unverändertem Vorabdesign frisch gerechnet.
