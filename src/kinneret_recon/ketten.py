"""Ablage der Ketten ausserhalb des Repos (Daten-Repo ``kinneret_reconstruction_laeufe``).

Die Ketten eines Laufs (``posterior.npz``, ``prior.npz``) liegen unter ihrem Fingerabdruck der
Sampler-Einstellungen und Daten: ``<ablage>/ketten/<fingerabdruck_ketten>/``. Laeufe mit gleichen Ketten
(z. B. Mittelwert und Mischverteilung) teilen sich einen Ordner. ``index.json`` ordnet die Laeufe ihren Ketten zu.
``kinneret-recon batch --ketten <ablage>`` wertet vorhandene Ketten neu aus, statt neu zu sampeln.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path

from .katalog import Lauf, Pruefung

DATEIEN = ("posterior.npz", "prior.npz")
README = """# Ketten der Kinneret-Rekonstruktion

Ketten (`posterior.npz`, `prior.npz`) der Läufe aus
[kinneret_reconstruction_python](https://github.com/TimonNetzel/kinneret_reconstruction_python), abgelegt nach dem
Fingerabdruck ihrer Sampler-Einstellungen und Eingangsdaten: `ketten/<fingerabdruck>/`. `index.json` ordnet die
Läufe des Katalogs ihren Ketten zu; `ketten/<fingerabdruck>/ketten.json` beschreibt Einstellungen und Code-Stand.

Verwendung (im Hauptverzeichnis von kinneret_reconstruction_python):

```bash
git clone <dieses Repo> ../kinneret_reconstruction_laeufe
kinneret-recon batch --ketten ../kinneret_reconstruction_laeufe      # Ketten neu auswerten statt sampeln (~15 min)
kinneret-recon ketten sichern ../kinneret_reconstruction_laeufe      # neue Ketten aus results/ hinzufügen
```
"""


def kettenordner(ablage: Path, fingerabdruck: str) -> Path:
    return Path(ablage) / "ketten" / fingerabdruck


def vorhanden(ablage: Path | None, fingerabdruck: str) -> Path | None:
    """Ordner mit den Ketten zu diesem Fingerabdruck, falls in der Ablage vorhanden."""
    if ablage is None:
        return None
    d = kettenordner(ablage, fingerabdruck)
    return d if (d / "posterior.npz").exists() else None


def sichern(laeufe: dict[str, Lauf], pruefungen: dict[str, Pruefung], results: Path, ablage: Path,
            log=print) -> list[str]:
    """Kopiert die Ketten der Laeufe (aus results/<lauf>/) in die Ablage; vorhandene Ketten bleiben unveraendert.
    Liefert die Fingerabdruecke der neu abgelegten Ketten."""
    ablage = Path(ablage)
    (ablage / "ketten").mkdir(parents=True, exist_ok=True)
    if not (ablage / "README.md").exists():
        (ablage / "README.md").write_text(README, encoding="utf-8")
    index_pfad = ablage / "index.json"
    index = json.loads(index_pfad.read_text(encoding="utf-8")) if index_pfad.exists() else {"laeufe": {}}
    neu = []
    for lid, lauf in laeufe.items():
        p = pruefungen[lid]
        fp = p.fingerabdruck_ketten
        quelle = Path(results) / lid
        ziel = kettenordner(ablage, fp)
        if not (ziel / "posterior.npz").exists():
            if not (quelle / "posterior.npz").exists():
                log(f"{lid}: keine Ketten in {quelle}")
                continue
            ziel.mkdir(parents=True, exist_ok=True)
            for f in DATEIEN:
                if (quelle / f).exists():
                    shutil.copy2(quelle / f, ziel / f)
            meta = json.loads((quelle / "meta.json").read_text(encoding="utf-8")) if (quelle / "meta.json").exists() else {}
            beschreibung = {
                "fingerabdruck_ketten": fp,
                "einstellungen": {k: v for k, v in p.einstellungen.items() if k.startswith("sampler.")},
                "git": meta.get("git"), "version": meta.get("version"), "python": meta.get("python"),
                "numpy": meta.get("numpy"), "timings_s": meta.get("timings_s"),
                "abgelegt": dt.date.today().isoformat(), "aus_lauf": lid,
            }
            (ziel / "ketten.json").write_text(json.dumps(beschreibung, indent=2, ensure_ascii=False), encoding="utf-8")
            neu.append(fp)
            log(f"{lid}: Ketten abgelegt unter ketten/{fp}/")
        k = json.loads((ziel / "ketten.json").read_text(encoding="utf-8")) if (ziel / "ketten.json").exists() else {}
        laeufe_dort = sorted(set(k.get("laeufe", [])) | {lid})
        if k and k.get("laeufe") != laeufe_dort:
            k["laeufe"] = laeufe_dort
            (ziel / "ketten.json").write_text(json.dumps(k, indent=2, ensure_ascii=False), encoding="utf-8")
        index["laeufe"][lid] = {"fingerabdruck_ketten": fp, "fingerabdruck": p.fingerabdruck, "config": lauf.config}
    index["laeufe"] = dict(sorted(index["laeufe"].items()))
    index_pfad.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return neu
