# Prior-only-Validierung: nur Dir(0.5)-Prior auf w und Gleichverteilung von c in den Grenzen.
import sys, time, numpy as np
from scipy import stats
from kinneret_recon.config import SamplerConfig
from kinneret_recon.data import load_inputs
from kinneret_recon.model import build_setup
from kinneret_recon import sampler
n = int(sys.argv[1]) if len(sys.argv) > 1 else 400_000
inp = load_inputs("data/in")
for fix in (True, False):
    cfg = SamplerConfig(sample_length=n, burn_in=n//4, thin=20, n_chains=4, dirichlet_hastings_fix=fix,
                        use_calibration=False, use_tf_likelihood=False)
    setup = build_setup(inp, cfg)
    t = time.time(); post = sampler.run(setup, cfg); dt = time.time()-t
    w = post.pooled("taxa_weights"); c = post.pooled("tf_sample")
    K = w.shape[1]
    # Sollwerte Dir(0.5,...,0.5): Randverteilung Beta(0.5, 0.5*(K-1))
    a, b = 0.5, 0.5*(K-1)
    ref = stats.beta(a, b)
    print(f"--- hastings_fix={fix}  ({dt:.1f}s, Akzeptanz {post.acceptance_rate.round(3)}, min log w {np.min(post.meta['min_log_weight']):.1f})")
    print(f"  w: Mittel {w.mean():.4f} (soll {ref.mean():.4f}), Sd {w.std():.4f} (soll {ref.std():.4f}), "
          f"Median {np.median(w):.5f} (soll {ref.median():.5f}), P(w<0.001) {np.mean(w<1e-3):.3f} (soll {ref.cdf(1e-3):.3f})")
    u = (c - setup.tf_lower) / (setup.tf_upper - setup.tf_lower)
    print(f"  c (skaliert auf [0,1]): Mittel {u.mean(0).round(3)} (soll 0.5), Sd {u.std(0).round(3)} (soll {np.sqrt(1/12):.3f})")
