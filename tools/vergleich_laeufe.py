"""Kennzahlen mehrerer Laeufe relativ zu einem Basislauf (erster Name).

Aufruf: python tools/vergleich_laeufe.py standard k_tf_normal_prior l_beta_262 ...
"""
import csv
import sys

import numpy as np

from kinneret_recon.postprocess import Results
from kinneret_recon.sampler import Posterior

names = list(np.load("data/in/core_data.npz")["taxa_names"])
ix = [names.index(n) for n in ("Olea europaea", "Quercus calliprinos type", "Amaranthaceae")]
runs = sys.argv[1:]
base = Results.load(f"results/{runs[0]}/results.npz")
iqr = lambda q: (q[2] - q[0]).mean()
print(f"{'Lauf':24s} {'T':>5s} {'dTmax':>6s} {'IQR_T':>6s} {'P':>5s} {'dPmax':>6s} {'dPmit':>6s} {'IQR_P':>6s} "
      f"{'R2':>5s} {'T0':>5s} {'P0':>5s} {'CI_T Med/IT/SA':>16s} {'CI_P Med/IT/SA':>16s} {'w Olea/Qcal/Amar':>18s} "
      f"{'minESS':>7s} {'maxRh':>6s}")
for r in runs:
    x = Results.load(f"results/{r}/results.npz")
    p = Posterior.load(f"results/{r}/posterior.npz")
    d = list(csv.DictReader(open(f"results/{r}/diagnostik.csv")))
    ess = np.nanmin([float(z["ess_bulk"]) for z in d])
    rh = np.nanmax([float(z["rhat"]) for z in d])
    dt = x.temp_age_q[1] - base.temp_age_q[1]
    dp = x.pann_age_q[1] - base.pann_age_q[1]
    ct = "/".join(f"{v:.2f}" for v in x.post_temp_ci_size / x.prior_temp_ci_size)
    cp = "/".join(f"{v:.2f}" for v in x.post_pann_ci_size / x.prior_pann_ci_size)
    w = "/".join(f"{v:.3f}" for v in np.median(x.post_weights[:, ix], axis=0))
    print(f"{r:24s} {x.temp_age_q[1].mean():5.2f} {np.abs(dt).max():6.2f} {iqr(x.temp_age_q):6.2f} "
          f"{x.pann_age_q[1].mean():5.0f} {np.abs(dp).max():6.0f} {dp.mean():+6.1f} {iqr(x.pann_age_q):6.0f} "
          f"{np.median(p.pooled('expl_variance')):5.2f} {np.median(p.pooled('recent_temp')):5.1f} "
          f"{np.median(p.pooled('recent_pann')):5.0f} {ct:>16s} {cp:>16s} {w:>18s} {ess:7.0f} {rh:6.3f}")
