"""Baumpollenkurve (AP) mit und ohne Kulturbaeume ueber das Alter.

Aufruf: python tools/ap_kurven.py [docs/abbildungen/ap_kulturbaeume.png]
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kinneret_recon.config import SamplerConfig
from kinneret_recon.data import CULTIVATED_TAXA, load_inputs
from kinneret_recon.model import build_setup

out = sys.argv[1] if len(sys.argv) > 1 else "docs/abbildungen/ap_kulturbaeume.png"
inp = load_inputs("data/in")
ap = build_setup(inp, SamplerConfig()).ap_age
ap_ohne = build_setup(inp, SamplerConfig(ap_exclude_taxa=CULTIVATED_TAXA)).ap_age
fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
ax.fill_between(inp.age, ap_ohne, ap, color="#b5d3a7", label="Kulturbäume (Olea, Vitis, Juglans)")
ax.plot(inp.age, ap, color="black", lw=2, label="AP (wie im Modell)")
ax.plot(inp.age, ap_ohne, color="#2f6f3e", lw=2, label="AP ohne Kulturbäume")
ax.axvspan(4000, 9000, color="0.92", zorder=0, label="Kalibrierung Test 2 (4–9 ka)")
ax.set_xlabel("Alter [cal a BP]")
ax.set_ylabel("Baumpollen [%]")
ax.set_xlim(inp.age.min(), inp.age.max())
ax.set_ylim(0, None)
ax.legend(fontsize=8, loc="upper right")
fig.savefig(out, dpi=130)
print(out)
