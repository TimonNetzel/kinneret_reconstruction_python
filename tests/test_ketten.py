"""Ketten sichern (Daten-Repo) und mit batch --ketten neu auswerten statt zu sampeln."""

import json
import shutil
from pathlib import Path

from kinneret_recon.cli import main

ROOT = Path(__file__).resolve().parents[1]
KURZ = "[sampler]\nsample_length = 6000\nburn_in = 2000\nthin = 20\nn_chains = 2\n"


def _katalog(root: Path) -> None:
    (root / "configs").mkdir()
    (root / "configs/basis.toml").write_text(KURZ)
    (root / "configs/mix.toml").write_text(KURZ + '\n[postprocess]\nreconstruction = "mixture"\n')
    for lid, extra in (("basis", ""), ("mix", 'basis = "basis"\nfaktor = "rekonstruktion"\nstufe = "mixture"\n')):
        d = root / "reference/laeufe" / lid
        d.mkdir(parents=True)
        (d / "lauf.toml").write_text(f'config = "configs/{lid}.toml"\n{extra}\n[titel]\nde = "x"\nen = "x"\n\n'
                                     '[beschreibung]\nde = "x"\nen = "x"\n')
    (root / "data").symlink_to(ROOT / "data")


def test_sichern_und_neu_auswerten(tmp_path, monkeypatch, capsys):
    root = tmp_path / "repo"
    root.mkdir()
    _katalog(root)
    monkeypatch.chdir(root)
    assert main(["batch"]) == 0
    app = {i: json.loads((root / f"reference/laeufe/{i}/app.json").read_text()) for i in ("basis", "mix")}
    ablage = tmp_path / "laeufe"
    assert main(["ketten", "sichern", str(ablage)]) == 0
    index = json.loads((ablage / "index.json").read_text())
    fp = index["laeufe"]["basis"]["fingerabdruck_ketten"]
    assert index["laeufe"]["mix"]["fingerabdruck_ketten"] == fp  # gleiche Ketten, ein Ordner
    assert sorted(p.name for p in (ablage / "ketten").iterdir()) == [fp]
    assert json.loads((ablage / f"ketten/{fp}/ketten.json").read_text())["laeufe"] == ["basis", "mix"]
    assert (ablage / f"ketten/{fp}/prior.npz").exists() and (ablage / "README.md").exists()
    # zweimal sichern aendert nichts
    assert main(["ketten", "sichern", str(ablage)]) == 0
    assert "0 Kettensaetze neu" in capsys.readouterr().out

    shutil.rmtree(root / "results")
    assert main(["batch", "-n", "--ketten", str(ablage)]) == 0
    out = capsys.readouterr().out
    assert "rechnen" not in out.replace("zu rechnen", "") and "neu auswerten" in out
    assert main(["batch", "--ketten", str(ablage)]) == 0
    for i in ("basis", "mix"):
        neu = json.loads((root / f"reference/laeufe/{i}/app.json").read_text())
        for k in ("rekonstruktion", "biome", "gewichte", "transferfunktionen", "kalibrierung", "fingerabdruck"):
            assert neu[k] == app[i][k], (i, k)
        assert neu["meta"]["timings_s"]["sampling"] == 0.0
