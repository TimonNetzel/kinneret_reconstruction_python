"""Formeln in Texten: LaTeX-Schreibweise zwischen $...$ (z. B. ``$T_\\mathrm{DJF}$``), umgesetzt nach HTML
(kursive Variablen, echte Indizes; auch fuer Plotly), nach LaTeX (Text ausserhalb der Formeln maskiert) oder nach
reinem Text (Unicode, so weit moeglich).

Unterstuetzt wird eine kleine Teilmenge: Buchstaben (kursiv), Ziffern und Zeichen, ``_`` und ``^`` mit einem
Zeichen oder ``{...}``, ``\\mathrm{...}``, ``\\hat{...}``, griechische Buchstaben, ``\\log``, ``\\sim``, ``\\ldots``
und einige weitere Symbole. Leerzeichen in Formeln werden wie in LaTeX ignoriert; um Relationen und nach Kommas
wird Abstand gesetzt. Dieselbe Umsetzung steht in der App in ``app/kern.js`` (``formel``).
"""

from __future__ import annotations

import html
import re

GRIECH = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "kappa": "κ", "mu": "μ", "sigma": "σ", "psi": "ψ",
          "chi": "χ", "Sigma": "Σ", "Delta": "Δ", "Gamma": "Γ"}
SYMBOL = {"ldots": "…", "dots": "…", "cdot": "·", "times": "×", "pm": "±", ",": "\u2009", ";": " ", "%": "%"}
RELATION = {"sim": "∼", "le": "≤", "ge": "≥", "to": "→", "approx": "≈"}
FUNKTION = {"log", "exp", "ln"}
MIT_ARGUMENT = {"mathrm", "text", "hat"}
TIEF = dict(zip("0123456789+-()aehijklmnoprstuvx", "₀₁₂₃₄₅₆₇₈₉₊₋₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ"))
HOCH = dict(zip("0123456789+-()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁽⁾ⁿ"))
_BEFEHL = re.compile(r"\\([A-Za-z]+|.)")


def _mathe(s: str, modus: str) -> str:
    i, out = 0, []
    esc = html.escape if modus == "html" else (lambda x: x)

    def gruppe() -> str:
        nonlocal i
        if s[i] == "{":
            tiefe, j = 1, i + 1
            i += 1
            while i < len(s) and tiefe:
                tiefe += {"{": 1, "}": -1}.get(s[i], 0)
                i += 1
            return s[j:i - 1]
        if s[i] == "\\":
            m = _BEFEHL.match(s, i)
            i = m.end()
            if m.group(1) in MIT_ARGUMENT and i < len(s):  # z. B. T_\mathrm{DJF}
                return m.group(0) + "{" + gruppe() + "}"
            return m.group(0)
        i += 1
        return s[i - 1]

    while i < len(s):
        c = s[i]
        if c == "\\":
            m = _BEFEHL.match(s, i)
            i, n = m.end(), m.group(1)
            if n in ("mathrm", "text"):
                out.append(esc(gruppe()))
            elif n == "hat":
                inner = _mathe(gruppe(), modus)
                out.append(inner[:-4] + "̂</i>" if inner.endswith("</i>") else inner + "̂")
            elif n in GRIECH:
                g = GRIECH[n]
                out.append(f"<i>{g}</i>" if modus == "html" and n[0].islower() else g)
            elif n in FUNKTION:
                out.append(n + " ")
            elif n in RELATION:
                out.append(f" {RELATION[n]} ")
            elif n in SYMBOL:
                out.append(SYMBOL[n])
            else:
                out.append(esc(n))
        elif c in "_^":
            i += 1
            inner = _mathe(gruppe(), modus)
            if modus == "html":
                tag = "sub" if c == "_" else "sup"
                out.append(f"<{tag}>{inner}</{tag}>")
            else:
                tab = TIEF if c == "_" else HOCH
                out.append("".join(tab[ch] for ch in inner) if inner and all(ch in tab for ch in inner) else c + inner)
        elif c.isascii() and c.isalpha():
            j = i
            while j < len(s) and s[j].isascii() and s[j].isalpha():
                j += 1
            out.append(f"<i>{s[i:j]}</i>" if modus == "html" else s[i:j])
            i = j
        elif s.startswith("{,}", i):  # Dezimalkomma wie in LaTeX: 0{,}5
            out.append(",")
            i += 3
        elif c in "{} ":  # Leerzeichen setzt LaTeX in Formeln selbst
            i += 1
        elif c in "=<>":
            out.append(f" {esc(c)} ")
            i += 1
        elif c == ",":
            out.append(", ")
            i += 1
        else:
            out.append(esc(c))
            i += 1
    return "".join(out).strip(" ")


def _teile(s: str):
    """(ist_formel, Inhalt) fuer die Abschnitte zwischen $...$."""
    for k, teil in enumerate(s.split("$")):
        if teil:
            yield k % 2 == 1, teil


def zu_html(s: str) -> str:
    return "".join(_mathe(t, "html") if f else html.escape(t) for f, t in _teile(s))


def zu_text(s: str) -> str:
    return "".join(_mathe(t, "text") if f else t for f, t in _teile(s))


_LATEX_TEXT = {"\\": r"\textbackslash{}", "%": r"\%", "&": r"\&", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
               "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", " ": r"\,"}


def zu_latex(s: str) -> str:
    return "".join(f"${t}$" if f else "".join(_LATEX_TEXT.get(ch, ch) for ch in t) for f, t in _teile(s))
