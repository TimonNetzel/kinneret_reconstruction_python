# Abstimmung des Samplers: Akzeptanz, min. ESS und max. R-hat je Einstellung (nur Sampling)
import sys, time, json, numpy as np
from kinneret_recon.config import SamplerConfig
from kinneret_recon.data import load_inputs
from kinneret_recon.model import build_setup
from kinneret_recon import sampler, diagnostics
inp = load_inputs("data/in")
K = inp.num_taxa
for spec in sys.argv[1:]:
    kw = json.loads(spec)
    cfg = SamplerConfig(n_chains=4, **kw)
    setup = build_setup(inp, cfg)
    t = time.time(); post = sampler.run(setup, cfg); dt = time.time() - t
    s = diagnostics.summarize(post, list(inp.taxa_names), ["Med", "IT", "SA"])
    ess = np.array([r["ess_bulk"] for r in s]); rh = np.array([r["rhat"] for r in s])
    w = post.pooled("taxa_weights")
    print(f"{spec:70s} {dt:5.1f}s acc {post.acceptance_rate.mean():.3f} minESS {ess.min():6.0f} medESS {np.median(ess):6.0f} "
          f"maxRhat {rh.max():.3f} sd(w) {w.std():.4f}", flush=True)
