# Reproduktion ohne alte temporäre Dateien

Arbeitsverzeichnis ist ein Checkout dieses PR-Branches. Keine lokalen Dateien
aus einem früheren Chat voraussetzen. Der offene #18 ist nur die gemeinsame
Branchbasis; #28 wird ausdrücklich über seinen gesperrten Commit bezogen.
Für diese moderne Kalibrierung wird kein ScienceDirect-ZIP gebraucht.

## Umgebung und öffentliche Eingaben

Die genaue Vergleichsumgebung ist Python 3.12.14 mit den Versionen aus
[requirements-research.txt](./requirements-research.txt). Diese optionalen
Forschungsabhängigkeiten ändern die App nicht. In einer eigenen Umgebung:

```bash
python -m pip install -r docs/thesis_research/smote_nnet_2026_10_30/requirements-research.txt
python tools/taxon_smote/prepare.py \
  --sources docs/thesis_research/smote_nnet_2026_10_30/sources.json \
  --audit-sources docs/thesis_research/smote_nnet_2026_10_30/audit_sources.json \
  --previous work/pr28 --maps work/maps \
  --climate work/wc2.1_10m_bio.zip
```

Das Vorbereitungsskript lädt ausschließlich die angegebenen öffentlichen
Dateien mit gesperrten URLs, Bytezahlen und SHA256. Vorhandene abweichende
Dateien werden nicht überschrieben. `--verify-only` prüft vorhandene Dateien
ohne Downloads. Es führt die geladenen Programme nicht aus. Der anschließende
Vergleich importiert die ausdrücklich gesperrten #28-Leser nach Hashprüfung;
der unabhängige Prüfer verwendet nur dessen gesperrten separaten Vektorleser.

Quellenrechte bleiben wie in #28: Meusel/BfN-Karten CC-BY 4.0 mit voller
Urheberzuordnung im gesperrten alten `sources.json`; WorldClim akademisch/nicht
kommerziell, Rohdaten nicht weitergegeben. Alle Änderungen oder abweichenden
Hashes erfordern eine dokumentierte neue Quellenentscheidung.

## Vollständige Läufe und unabhängige Prüfung

Jeder Lauf braucht ein **noch nicht existierendes** Ausgabeverzeichnis.
Ein Thread vermeidet eine zusätzliche BLAS-Parallelisierungsvariable.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/taxon_smote/compare.py \
  --previous work/pr28 --maps work/maps \
  --climate work/wc2.1_10m_bio.zip \
  --config docs/thesis_research/smote_nnet_2026_10_30/config.json \
  --sources docs/thesis_research/smote_nnet_2026_10_30/sources.json \
  --output work/run1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/taxon_smote/verify.py \
  --previous work/pr28 --maps work/maps \
  --climate work/wc2.1_10m_bio.zip \
  --config docs/thesis_research/smote_nnet_2026_10_30/config.json \
  --sources docs/thesis_research/smote_nnet_2026_10_30/sources.json \
  --audit-sources docs/thesis_research/smote_nnet_2026_10_30/audit_sources.json \
  --output work/run1 --report work/verification.json
```

Den Vergleich unverändert mit `--output work/run2` wiederholen. Die Dateimengen
und alle SHA256 aus `manifest.json` vergleichen, einschließlich großer
Vorhersage- und SMOTE-Elternprotokolle. Gzip wird ohne Zeitstempel oder
Dateinamen geschrieben. Die dokumentierte Bytegleichheit gilt für die genaue
Umgebung; andere BLAS-, Optimierer- oder Bibliotheksversionen sind nicht als
bitgleich zugesichert. Auch bei Abweichungen dürfen Parameter nicht still
nachjustiert werden.

```bash
python -m unittest discover -s tests -p test_smote_network.py -v
python tools/taxon_smote/repeat.py \
  --first work/run1 --second work/run2 --report work/repeat.json
python tools/taxon_smote/plot.py \
  --summary work/run1/metrics_summary.csv \
  --output work/comparison.png
```

## Gespeicherte Funktionen verwenden

`models.json` enthält pro Art die beiden vollständigen Fits einschließlich
Normierung, aller Gewichte und ursprünglicher Klimahülle. `fit_models.json`
enthält außerdem sämtliche räumlichen Fold-Fits und je drei Optimierungsstarts.
Parameterreihenfolge: Input→Hidden-Matrix (2 × H, zeilenweise), H Hidden-Biases,
H Hidden→Output-Gewichte, Output-Bias. Beide Aktivierungen sind sigmoid.
Der separat importierbare Vorwärtsevaluator benötigt nur NumPy/SciPy:

```python
import json
import sys
sys.path.insert(0, "tools/taxon_smote")
from network import predict

models = json.load(open("docs/thesis_research/smote_nnet_2026_10_30/models.json"))
m = models["fagus_sylvatica"]["nnet_smote"]
q, inside = predict(m, [[10., 1000.]])  # TANN °C, PANN mm/Jahr
p_diagnostic, _ = predict(m, [[10., 1000.]], prior_adjusted=True)
```

Außerhalb der ursprünglichen Trainings-Klimahülle verweigert `predict` die
Ausgabe standardmäßig. `reject_extrapolation=False` wurde ausschließlich zur
vollständigen, ungefilterten äußeren Testauswertung verwendet und kennzeichnet
den Support weiterhin. Die Prioranpassung ist eine Trainingsdiagnose und keine
garantierte Kalibrierung. Keine dieser Ausgaben ist eine freigegebene
Fossil-Transferfunktion, Klimarekonstruktion oder automatische Pollentyp-Zuordnung.

Die unabhängige Prüfung importiert weder `compare.py` noch `network.py`.
Gemeinsam verwendete GIS-Decoder, numerische Bibliotheken und der alte
unabhängige Vektorleser sind im Prüfbericht genannt. Der Audit bestätigt die
Nachvollziehbarkeit innerhalb des angegebenen Kartierungsmodells, nicht die
ökologische Richtigkeit jeder Kartengrenze oder fossile Gültigkeit.
