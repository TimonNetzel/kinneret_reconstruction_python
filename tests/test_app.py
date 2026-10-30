"""App-Build: Ordner und eine Datei, offline (keine Netz-Adressen), eingebettete Daten vollstaendig."""

import base64
import gzip
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from kinneret_recon.app_bauen import STATISCH, _js, baue_eine_datei, baue_ordner

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


def _zweiter_standort(ordner: Path, ziel: Path) -> Path:
    """Kopie der gebauten App mit einem zweiten Standort "zweit" (ein Lauf, Daten wie Kinneret)."""
    shutil.copytree(ordner, ziel)
    _, k = _js_daten(ziel / "daten/katalog.js")
    _, st = _js_daten(ziel / "daten/standort_kinneret.js")
    _, d = _js_daten(ziel / "daten/laeufe/standard_v3.js")
    name = {"de": "Zweiter Standort", "en": "Second site"}
    l = {**next(x for x in k["laeufe"] if x["id"] == "standard_v3"), "id": "zweit_lauf", "standort": "zweit",
         "rolle": "referenz", "basis": None, "faktor": None, "stufe": None}
    k["laeufe"].append(l)
    k["standorte"].append({"key": "zweit", "name": name, "daten": "standort_zweit.json", "standard_lauf": "zweit_lauf"})
    d.update(id="zweit_lauf", lauf={**d["lauf"], "id": "zweit_lauf", "standort": "zweit"})
    for name_js, schluessel, inhalt in (("katalog.js", "katalog", k), ("standort_zweit.js", "standort/zweit",
                                        {**st, "key": "zweit", "name": name}),
                                        ("laeufe/zweit_lauf.js", "lauf/zweit_lauf", d)):
        (ziel / "daten" / name_js).write_text(_js(schluessel, inhalt), encoding="utf-8")
    return ziel / "index.html"


def test_validierung_in_ordner_und_einer_datei(ordner, eine_datei):
    """Moderne Validierung: gespeicherte Baseline-Metriken als eigene Daten, im Ordner und in der einen Datei."""
    from kinneret_recon.baseline import app_daten

    name, v = _js_daten(ordner / "daten/validierung.js")
    assert name == "validierung" and v == json.loads(json.dumps(app_daten(ROOT)))
    assert "daten/validierung.js" in (ordner / "sw.js").read_text()
    assert '(window.KR_DATEN=window.KR_DATEN||{})["validierung"]=' in eine_datei.read_text(encoding="utf-8")


def test_katalog_standorte():
    k = json.loads((ROOT / "reference/app/katalog.json").read_text(encoding="utf-8"))
    assert [(s["key"], s["standard_lauf"]) for s in k["standorte"]] == [("kinneret", "standard_v3")]
    assert {l["standort"] for l in k["laeufe"]} == {"kinneret"}


def _node_mit_playwright():
    if shutil.which("node") is None:
        pytest.skip("node fehlt")
    if subprocess.run(["node", "-e", "require('playwright')"], capture_output=True).returncode:
        pytest.skip("Playwright fehlt")


def test_standort_waehler_im_browser(ordner, tmp_path):
    """Ab zwei Standorten erscheint der Waehler; Lauflisten zeigen nur Laeufe des gewaehlten Standorts."""
    _node_mit_playwright()
    index = _zweiter_standort(ordner, tmp_path / "app2")
    r = subprocess.run(["node", str(ROOT / "tools/app_pruefen.js"), "--zwei-standorte", str(index)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node fehlt")
def test_im_browser(ordner, eine_datei):
    """Smoketest in Chromium (tools/app_pruefen.js), falls Playwright installiert ist."""
    if subprocess.run(["node", "-e", "require('playwright')"], capture_output=True).returncode:
        pytest.skip("Playwright fehlt")
    r = subprocess.run(["node", str(ROOT / "tools/app_pruefen.js"), str(ordner / "index.html"), str(eine_datei)],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout + r.stderr


def test_export_eingebunden(ordner, eine_datei):
    """Export (export.js, Dialog, Druckbereich) im Ordner und in der einen Datei, vor start.js geladen."""
    html = (ordner / "index.html").read_text(encoding="utf-8")
    assert html.index('src="export.js"') < html.index('src="start.js"')
    eine = eine_datei.read_text(encoding="utf-8")
    for teil in ('<dialog id="export"', 'id="druck"', 'id="druck-seite"', "KR.oeffneExport = oeffne"):
        assert teil in html or teil in (ordner / "export.js").read_text(encoding="utf-8"), teil
        assert teil in eine, teil
