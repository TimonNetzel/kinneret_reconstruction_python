import difflib, io, re, sys, tarfile
from pathlib import Path
cache = Path(sys.argv[1]) / 'tarballs'
pairs = [('0.54.0', '0.55.0'), ('0.57.0', '0.58.0'), ('0.58.0', '0.59.0'), ('0.60.0', '0.60.1'), ('0.63.0', '1.0.0')]


def files(v):
    out = {}
    with tarfile.open(cache / f'matrixStats_{v}.tar.gz', 'r:gz') as tar:
        for m in tar.getmembers():
            if m.isfile() and re.match(r'matrixStats/(R/weightedMedian\.R|src/weightedMedian[^/]*\.(c|h))$', m.name):
                out[m.name.split('/', 1)[1]] = tar.extractfile(m).read().decode('utf-8', 'replace').splitlines()
    return out


for a, b in pairs:
    fa, fb = files(a), files(b)
    print(f'===== {a} -> {b}')
    for name in sorted(set(fa) | set(fb)):
        d = list(difflib.unified_diff(fa.get(name, []), fb.get(name, []), f'{a}/{name}', f'{b}/{name}', n=1, lineterm=''))
        if d:
            print('\n'.join(d[:80]))
