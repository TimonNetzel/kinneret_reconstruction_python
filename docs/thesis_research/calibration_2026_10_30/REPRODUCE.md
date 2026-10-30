# Reproduktion der verschachtelten Kalibrierung

Ein frischer Checkout dieses Branches genügt; keine Dateien aus alten Chats.
Vorabdesign und unveränderte Parameter: [DESIGN.md](./DESIGN.md),
[config.json](./config.json), veröffentlicht vor dem ersten Fit in
`afa0c8a8845a5f40daee4289cc429fdf4a6ecf6a`.
Basis ist main nach #30, nicht ein alter getrennter Forschungsbranch.

## Umgebung und gesperrte Eingänge

Python 3.12, genaue Bibliotheksversionen aus dem unveränderten #29-Register.
Die Quellenprüfung sperrt #28 am Originalcommit, Karten und Klima an SHA256;
abweichende vorhandene Dateien werden nicht überschrieben. Kein ScienceDirect-
ZIP nötig. Meusel/BfN-Zuordnung und CC-BY-4.0-Attribution stehen im gesperrten
alten sources.json; WorldClim-Roharchive werden nicht neu verteilt.

```bash
python -m pip install -r docs/thesis_research/smote_nnet_2026_10_30/requirements-research.txt
python tools/taxon_smote/prepare.py \
  --sources docs/thesis_research/smote_nnet_2026_10_30/sources.json \
  --audit-sources docs/thesis_research/smote_nnet_2026_10_30/audit_sources.json \
  --previous work/pr28 --maps work/maps --climate work/wc2.1_10m_bio.zip
```

## Zwei vollständige Läufe

Jeder Lauf braucht ein neues Ausgabeverzeichnis. Sechs unabhängige Netzfits
werden in getrennten Prozessen gerechnet und **in fester Fallreihenfolge**
gespeichert. Jeder Fit verwendet einen BLAS-Thread; die Prozessreihenfolge
bestimmt weder Seeds noch Startauswahl. Zwei frühe serielle Teilläufe wurden
allein zur Beschleunigung der Ausführung abgebrochen, ohne Parameteränderung.
67 gemeinsame Parameterdatensätze wurden zwischen serieller und paralleler
Ausführung zusätzlich byteinhaltlich gleich bestätigt. Die maßgeblichen
Reproduktionsnachweise beziehen sich auf die zwei vollständigen Endläufe.
Ein weiterer vollständiger Fitlauf wurde wegen einer abgeschnittenen Gzip-
Elternspur vom Audit verworfen: neun andere Ausgaben waren bereits bytegleich,
aber die Spur war unvollständig. Die Ursache ist nicht belegt. Deshalb wurde
ein zusätzlicher kompletter Lauf unverändert gerechnet; keine Modell-, Score-
oder Designanpassung zur Herstellung besserer Ergebnisse. Im Ersatzlauf
blieb die Checkpointdatei bei 503 Einträgen stehen, während das vollständige
Vollgebietsmodell desselben Laufs und sein Manifest bereits alle 504 enthielten.
Der letzte Eintrag wurde ausschließlich aus dieser gleichlaufenden Modellkopie
wiederhergestellt; die Kandidatendatei entsprach bereits **vor dem Schreiben**
dem vorhandenen Manifest-Hash. Keine Gewichte neu berechnet oder angepasst.
[checkpoint_recovery.json](./checkpoint_recovery.json) dokumentiert die Wiederherstellung.
Danach sind alle elf Enddateien der zwei vollständigen Trainingsausführungen
bytegleich ([repeat.json](./repeat.json)); dies ist keine Behauptung zweier
völlig störungsfreier I/O-Läufe.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/taxon_calibration/run.py \
  --previous work/pr28 --maps work/maps --climate work/wc2.1_10m_bio.zip \
  --config docs/thesis_research/calibration_2026_10_30/config.json \
  --output work/calibration_run1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/taxon_calibration/run.py \
  --previous work/pr28 --maps work/maps --climate work/wc2.1_10m_bio.zip \
  --config docs/thesis_research/calibration_2026_10_30/config.json \
  --output work/calibration_run2
python tools/taxon_smote/repeat.py \
  --first work/calibration_run1 --second work/calibration_run2 \
  --report work/calibration_repeat.json
```

`manifest.json` sperrt sämtliche Ausgabebytes einschließlich großer
Vorhersagetabelle und vollständiger SMOTE-Elternspur. Gzip ohne Zeitstempel
oder Dateinamen. Bitgleichheit gilt für die aufgezeichnete Umgebung, nicht
pauschal über andere Optimierer-/BLAS-/Bibliotheksversionen.

## Separater Audit und Tests

Der Prüfer importiert weder den neuen Produzenten noch den neuen Kalibrator.
Er benutzt den gesperrten unabhängigen #28-Vektorleser und unveränderte,
zusätzlich hashgesperrte #29-Auditarithmetik. KKT-Prüfung des konvexen
Kalibrierungsziels bestätigt die Lösung einschließlich der a=0-Randfälle.
Gemeinsame GIS-/Numerikbibliotheken sind ausdrücklich keine unabhängige
Bestätigung der botanischen Kartenwahrheit. Alle Originaleltern, Zellrollen,
Starts, Kalibratoren, äußeren Prognosen, Scores und Deltas werden geprüft.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/taxon_calibration/verify.py \
  --previous work/pr28 --maps work/maps --climate work/wc2.1_10m_bio.zip \
  --config docs/thesis_research/calibration_2026_10_30/config.json \
  --output work/calibration_run1 --report work/calibration_verification.json
python -m unittest discover -s tests -p test_spatial_calibration.py -v
```

Der gewöhnliche Repositorytest benötigt zusätzlich den kompilierten C++-Kern:
`pip install -e ".[test]"`, dann `pytest -q`. Die neue Forschung selbst benötigt
den C++-Kern nicht. Eine abweichende lokale Build-/Browserumgebung ist separat
von wissenschaftlichen Prüfungen zu dokumentieren.

## Gespeicherte Funktionen

`models.json` enthält für jede Art, beide Blockgrößen und beide Netzvarianten
je drei vollständige Netz-/Kalibrator-Paare. `fit_models.json` enthält zusätzlich
alle äußeren Modelle und Optimierungsstarts. Kein Refit unter fremdem Kalibrator.

```python
import json
import sys
sys.path.insert(0, 'tools/taxon_calibration')
from model import predict

models = json.load(open('docs/thesis_research/calibration_2026_10_30/models.json'))
members = models['fagus_sylvatica']['5']['smote']
p, support_count = predict(members, [[10., 1000.]])  # TANN °C, PANN mm/Jahr
raw, _ = predict(members, [[10., 1000.]], calibrated=False)
```

Standardmäßig wird bei Extrapolation aus irgendeiner Mitgliedshülle abgebrochen.
Nur die vollständige äußere Bewertung verwendet `reject_extrapolation=False`
und behält alle Testzellen samt `support_members` (0–3). Das ist keine fossile
Transferfreigabe, Pollentyp-Zuordnung oder Unsicherheitsschätzung. Bereits
untersuchte Gebiete bleiben als solche benannt; kein vollständig ungesehener Test.
