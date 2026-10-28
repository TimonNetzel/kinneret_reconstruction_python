"""App-Build: Ordner und eine Datei, offline (keine Netz-Adressen), eingebettete Daten vollstaendig."""

import base64
import gzip
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from kinneret_recon.app_bauen import STATISCH, baue_eine_datei, baue_ordner

ROOT = Path(__file__).resolve().parents[1]


def _js_daten(path: Path) -> tuple[str, dict]:
    m = re.fullmatch(r'\(window\.KR_DATEN=window\.KR_DATEN\|\|\{\}\)\[("[^"]+")\]=(.*);\n', path.read_text(encoding="utf-8"), re.S)
    assert m, path
    return json.loads(m.group(1)), json.loads(m.group(2))


@pytest.fixture(scope="module")
def ordner(tmp_path_factory):
    out = tmp_path_factory.mktemp("app") / "app"
    baue_ordner(out, ROOT)
    return out


@pytest.fixture(scope="module")
def eine_datei(tmp_path_factory):
    return baue_eine_datei(tmp_path_factory.mktemp("app1") / "kinneret-app.html", ROOT)


def test_ordner_vollstaendig(ordner):
    for f in STATISCH + ("sw.js", "daten/katalog.js", "daten/standort_kinneret.js"):
        assert (ordner / f).exists(), f
    name, katalog = _js_daten(ordner / "daten/katalog.js")
    assert name == "katalog" and katalog["standard_lauf"] == "standard_v3"
    for l in katalog["laeufe"]:
        name, d = _js_daten(ordner / f"daten/laeufe/{l['id']}.js")
        assert name == f"lauf/{l['id']}" and d["id"] == l["id"]
    sw = (ordner / "sw.js").read_text()
    assert "daten/laeufe/standard_v3.js" in sw and "vendor/plotly-cartesian.min.js" in sw


def test_keine_netzadressen(ordner, eine_datei):
    """Offline: HTML, Stile und Skripte der App laden nichts aus dem Netz."""
    for f in STATISCH:
        if f.startswith("vendor/"):
            continue
        text = (ordner / f).read_text(encoding="utf-8").replace('xmlns="http://www.w3.org/2000/svg"', "")
        assert "http://" not in text and "https://" not in text, f
    html = eine_datei.read_text(encoding="utf-8")
    assert not re.search(r"""<(script|link)[^>]+(src|href)=["']https?://""", html)
    assert '<script src="' not in html and 'rel="stylesheet"' not in html


def test_eine_datei_enthaelt_alle_laeufe(eine_datei):
    html = eine_datei.read_text(encoding="utf-8")
    katalog = json.loads((ROOT / "reference/app/katalog.json").read_text(encoding="utf-8"))
    ids = [l["id"] for l in katalog["laeufe"] if l.get("app_json")]
    blobs = dict(re.findall(r'<script type="application/gzip-base64" id="kr-lauf-([^"]+)">([^<]+)</script>', html))
    assert sorted(blobs) == sorted(ids)
    d = json.loads(gzip.decompress(base64.b64decode(blobs["standard_v3"])))
    assert d == json.loads((ROOT / "reference/laeufe/standard_v3/app.json").read_text(encoding="utf-8"))
    assert eine_datei.stat().st_size < 15e6


@pytest.mark.skipif(shutil.which("node") is None, reason="node fehlt")
def test_im_browser(ordner, eine_datei):
    """Smoketest in Chromium (tools/app_pruefen.js), falls Playwright installiert ist."""
    if subprocess.run(["node", "-e", "require('playwright')"], capture_output=True).returncode:
        pytest.skip("Playwright fehlt")
    r = subprocess.run(["node", str(ROOT / "tools/app_pruefen.js"), str(ordner / "index.html"), str(eine_datei)],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout + r.stderr
