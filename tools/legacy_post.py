# Fuehrt main.py des Originals ohne erneutes MCMC-Sampling aus und misst die Laufzeit.
import time
t0 = time.time()
src = open("main.py").read().replace("mcmc_sampling = True", "mcmc_sampling = False")
exec(src)
print(f"postprocessing+plots: {time.time()-t0:.1f}s")
