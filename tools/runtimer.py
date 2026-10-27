import resource, subprocess, sys, time
t = time.time()
r = subprocess.run(sys.argv[1:])
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
print(f"exit={r.returncode} wall={time.time()-t:.1f}s user={ru.ru_utime:.1f}s maxrss={ru.ru_maxrss/1024:.0f}MB", file=sys.stderr)
