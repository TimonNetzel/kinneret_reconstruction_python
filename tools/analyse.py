# Kennzahlen der Vergleichsreihe fuer den Pruefbericht
import csv, json, sys
import numpy as np
from kinneret_recon.postprocess import Results
from kinneret_recon.sampler import Posterior
from kinneret_recon.data import load_inputs
names = load_inputs("data/in").taxa_names
runs = sys.argv[1:]
R = {r: Results.load(f"results/{r}/results.npz") for r in runs}
P = {r: Posterior.load(f"results/{r}/posterior.npz") for r in runs}
base = runs[0]
def iqr(q): return (q[2]-q[0]).mean()
print(f"{'Lauf':24s} {'T med':>7s} {'dT max':>7s} {'T IQR':>6s} {'P med':>7s} {'dP max':>7s} {'P IQR':>6s} {'R2':>5s} {'sd(w)':>6s} {'acc':>6s} {'minESS':>7s} {'maxRh':>6s}")
for r in runs:
    x = R[r]; b = R[base]
    d = list(csv.DictReader(open(f"results/{r}/diagnostik.csv")))
    ess = min(float(z["ess_bulk"]) for z in d); rh = max(float(z["rhat"]) for z in d)
    print(f"{r:24s} {x.temp_age_q[1].mean():7.2f} {np.abs(x.temp_age_q[1]-b.temp_age_q[1]).max():7.3f} {iqr(x.temp_age_q):6.2f} "
          f"{x.pann_age_q[1].mean():7.1f} {np.abs(x.pann_age_q[1]-b.pann_age_q[1]).max():7.1f} {iqr(x.pann_age_q):6.1f} "
          f"{P[r].pooled('expl_variance').mean():5.2f} {x.post_weights.std():6.4f} {P[r].acceptance_rate.mean():6.3f} {ess:7.0f} {rh:6.3f}")
print("\nTF-Mediane (T Med/IT/SA [°C], P Med/IT/SA [mm]) und CI-Verhaeltnis post/prior")
for r in runs:
    x = R[r]
    print(f"{r:24s} T {np.median(x.tf_post_temp,0).round(1)} P {np.median(x.tf_post_pann,0).round(0)} "
          f"CI_T {(x.post_temp_ci_size/x.prior_temp_ci_size).round(2)} CI_P {(x.post_pann_ci_size/x.prior_pann_ci_size).round(2)}")
print("\nTaxa-Gewichte (Median): groesste 6 je Lauf")
for r in runs:
    m = np.median(R[r].post_weights, 0); o = np.argsort(-m)[:6]
    print(f"{r:24s} " + ", ".join(f"{names[k]} {m[k]:.3f}" for k in o))
print("\nMittel der Gewichte: Olea, Quercus calliprinos, Amaranthaceae")
ix = [list(names).index(n) for n in ("Olea europaea", "Quercus calliprinos type", "Amaranthaceae")]
for r in runs:
    w = R[r].post_weights
    print(f"{r:24s} median {np.median(w[:, ix],0).round(4)}  P(w<0.001) {(w[:, ix] < 1e-3).mean(0).round(3)}")
