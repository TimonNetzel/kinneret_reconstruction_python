"""Baut die App (statisch, offline-faehig) aus den App-Daten des Katalogs (docs/app_konzept.md, Abschnitt 8).

- Ordner (Standard ``results/app/``): ``index.html`` per Doppelklick oeffnen. Die Daten liegen als JS-Dateien bei
  (``daten/``), weil Browser ueber ``file://`` kein JSON nachladen; die Daten eines Laufs werden erst geladen,
  wenn er angezeigt wird.
- Eine Datei (``--eine-datei``): alles eingebettet, die Laufdaten gzip-komprimiert (Entpacken im Browser mit
  ``DecompressionStream``). Fuer das Handy.
- Manifest und Service Worker fuer den Offline-Betrieb ueber ``localhost`` oder HTTPS (PWA).
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import http.server
import json
import shutil
import socket
from functools import partial
from pathlib import Path

from .app_daten import APP_DIR
from .katalog import KATALOG_DIR

QUELLE = Path(__file__).parent / "app"
# Skripte in der Reihenfolge von index.html; die Reihenfolge der Ansichten ist die der Reiter
SKRIPTE = ("vendor/plotly-cartesian.min.js", "i18n.js", "kern.js", "rekonstruktion.js", "sensitivitaet.js",
           "gewichte.js", "transferfunktionen.js", "lauf.js", "export.js", "start.js")
STATISCH = ("index.html", "style.css", "manifest.webmanifest", "icon.svg", "vendor/LICENSE-plotly.txt", *SKRIPTE)


def _js(name: str, d) -> str:
    """Daten als JS-Anweisung (laedt auch ueber file://)."""
    return f"(window.KR_DATEN=window.KR_DATEN||{{}})[{json.dumps(name)}]=" + json.dumps(d, ensure_ascii=False,
                                                                                         separators=(",", ":")) + ";\n"


def lade_daten(root: Path) -> tuple[dict, dict[str, dict], dict[str, dict]]:
    """Katalog-Index, Standortdaten und App-Daten aller Laeufe mit app.json."""
    katalog = json.loads((root / APP_DIR / "katalog.json").read_text(encoding="utf-8"))
    standorte = {s["key"]: json.loads((root / APP_DIR / s["daten"]).read_text(encoding="utf-8"))
                 for s in katalog["standorte"]}
    laeufe = {}
    for l in katalog["laeufe"]:
        if l.get("app_json"):
            laeufe[l["id"]] = json.loads((root / l["app_json"]).read_text(encoding="utf-8"))
    if not laeufe:
        raise SystemExit(f"keine App-Daten in {root / KATALOG_DIR} (kinneret-recon batch oder app-daten)")
    return katalog, standorte, laeufe


def _service_worker(dateien: list[str], version: str) -> str:
    return f"""// Service Worker: haelt die App offline verfuegbar (nur ueber localhost oder HTTPS aktiv)
const CACHE = "kinneret-app-{version}";
const DATEIEN = {json.dumps(dateien)};
self.addEventListener("install", e => e.waitUntil(caches.open(CACHE).then(c => c.addAll(DATEIEN))));
self.addEventListener("activate", e => e.waitUntil(caches.keys().then(ks =>
  Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k))))));
self.addEventListener("fetch", e => e.respondWith(caches.match(e.request).then(r => r || fetch(e.request))));
"""


def baue_ordner(out: Path, root: Path | None = None) -> Path:
    root = Path(root or Path.cwd())
    katalog, standorte, laeufe = lade_daten(root)
    if out.exists():
        shutil.rmtree(out)
    (out / "daten/laeufe").mkdir(parents=True)
    (out / "vendor").mkdir()
    for f in STATISCH:
        shutil.copy2(QUELLE / f, out / f)
    (out / "daten/katalog.js").write_text(_js("katalog", katalog), encoding="utf-8")
    for key, d in standorte.items():
        (out / f"daten/standort_{key}.js").write_text(_js(f"standort/{key}", d), encoding="utf-8")
    for lid, d in laeufe.items():
        (out / f"daten/laeufe/{lid}.js").write_text(_js(f"lauf/{lid}", d), encoding="utf-8")
    dateien = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    h = hashlib.sha256()
    for f in dateien:
        h.update((out / f).read_bytes())
    (out / "sw.js").write_text(_service_worker(["./", *dateien], h.hexdigest()[:12]), encoding="utf-8")
    return out / "index.html"


def baue_eine_datei(path: Path, root: Path | None = None) -> Path:
    """Eine einzige HTML-Datei: Skripte, Stile und Daten eingebettet (Laufdaten gzip + base64)."""
    root = Path(root or Path.cwd())
    katalog, standorte, laeufe = lade_daten(root)
    html = (QUELLE / "index.html").read_text(encoding="utf-8")
    css = (QUELLE / "style.css").read_text(encoding="utf-8")
    teile = {
        '<link rel="stylesheet" href="style.css">': f"<style>\n{css}</style>",
        '<link rel="manifest" href="manifest.webmanifest">': "",
        '<link rel="icon" href="icon.svg" type="image/svg+xml">':
            '<link rel="icon" href="data:image/svg+xml;base64,' +
            base64.b64encode((QUELLE / "icon.svg").read_bytes()).decode() + '">',
    }
    for alt, neu in teile.items():
        if alt not in html:
            raise RuntimeError(f"index.html: {alt!r} fehlt")
        html = html.replace(alt, neu)
    skripte = []
    for src in SKRIPTE:
        code = (QUELLE / src).read_text(encoding="utf-8").replace("</script", "<\\/script")
        skripte.append((f'<script src="{src}"></script>', f"<script>\n{code}\n</script>"))
    daten = _js("katalog", katalog) + "".join(_js(f"standort/{k}", d) for k, d in standorte.items())
    daten = daten.replace("</", "<\\/")
    eingebettet = "".join(
        f'<script type="application/gzip-base64" id="kr-lauf-{lid}">' +
        base64.b64encode(gzip.compress(json.dumps(d, ensure_ascii=False, separators=(",", ":")).encode(), 9,
                                       mtime=0)).decode() + "</script>\n"
        for lid, d in laeufe.items())
    daten_tag = '<script src="daten/katalog.js"></script>'
    for alt, neu in skripte:
        if alt not in html:
            raise RuntimeError(f"index.html: {alt!r} fehlt")
        html = html.replace(alt, neu)
    if daten_tag not in html:
        raise RuntimeError("index.html: Einbindung der Daten fehlt")
    html = html.replace(daten_tag, f"{eingebettet}<script>\n{daten}</script>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def lan_adressen() -> list[str]:
    out = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))  # keine Verbindung noetig, nur Wahl der Schnittstelle
            out.append(s.getsockname()[0])
    except OSError:
        pass
    return out


def serve(ordner: Path, port: int = 8000, host: str = "0.0.0.0") -> None:
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(ordner))
    with http.server.ThreadingHTTPServer((host, port), handler) as httpd:
        print(f"App auf http://localhost:{port}/" +
              "".join(f" und http://{a}:{port}/ (WLAN)" for a in lan_adressen()) + "; beenden mit Strg+C")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
