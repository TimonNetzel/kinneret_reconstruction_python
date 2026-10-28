"""Formeln in Texten (LaTeX-Schreibweise): Python (formel.py) und App (kern.js) setzen gleich um."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from kinneret_recon.formel import zu_html, zu_latex, zu_text

ROOT = Path(__file__).resolve().parents[1]
FAELLE = [
    r"$T_\mathrm{DJF}$ [°C]",
    r"$\Delta P_\mathrm{ANN}$ [mm] und $T_0$",
    r"Prior $N_\mathrm{t}(\mu(\psi), \sigma(\psi))$ auf 95 % & mehr",
    r"Split-$\hat{R}$, $R^2$, $y_k \sim \mathrm{Gamma}(0{,}5)$",
    r"$\log\alpha \sim N(\log 1, 1^2)$; Dir($\alpha, \ldots, \alpha$), $\kappa = 4K^2$",
    "ohne Formel <b> & 8 000",
]


def test_beispiele():
    assert zu_html(FAELLE[0]) == "<i>T</i><sub>DJF</sub> [°C]"
    assert zu_text(FAELLE[0]) == "T_DJF [°C]"
    assert zu_latex(FAELLE[0]) == FAELLE[0]
    assert zu_html(r"$y_k \sim \mathrm{Gamma}(0{,}5)$") == "<i>y</i><sub><i>k</i></sub> ∼ Gamma(0,5)"
    assert zu_text(r"$R^2$, $T_0$, $\hat{R}$") == "R², T₀, R̂"
    assert zu_html(r"$\hat{R}$") == "<i>R̂</i>"
    assert zu_html(r"$\log\alpha \sim N(\log 1, 1^2)$") == "log <i>α</i> ∼ <i>N</i>(log 1, 1<sup>2</sup>)"
    assert zu_latex("95 % von 8 000 & a_b") == r"95 \% von 8\,000 \& a\_b"
    assert zu_html("a < b & c") == "a &lt; b &amp; c"


@pytest.mark.skipif(shutil.which("node") is None, reason="node fehlt")
def test_js_gleich_python():
    src = (ROOT / "src/kinneret_recon/app/kern.js").read_text(encoding="utf-8")
    esc0 = src.index("  function esc(s) {")
    esc = src[esc0:src.index("\n  }\n", esc0) + 4]
    teil = src[src.index("  // Formeln in Texten"):src.index("  const F = ")]
    prog = ("const fs = require('fs'); const formel = new Function(" + json.dumps(esc + teil + "; return formel;") +
            ")(); const f = JSON.parse(fs.readFileSync(0, 'utf8'));"
            "console.log(JSON.stringify(f.map(([s, m]) => formel(s, m))));")
    faelle = [[s, m] for s in FAELLE for m in ("html", "text", "latex")]
    r = subprocess.run(["node", "-e", prog], input=json.dumps(faelle), capture_output=True, text=True, check=True)
    erwartet = [{"html": zu_html, "text": zu_text, "latex": zu_latex}[m](s) for s, m in faelle]
    assert json.loads(r.stdout) == erwartet
