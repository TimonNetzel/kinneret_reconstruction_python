"""Extract rdirichlet and ddirichlet of LaplacesDemon from every CRAN version in the window and compare them.

Tarballs are cached and read in memory only; the function text is copied from `name <- function` to the line that
closes its outermost brace. Writes dirichlet_versions.json with the text of each distinct version (GPL source, short).
Usage: compare_dirichlet.py <cache folder> <cran_exports.json>
"""
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent
PKG, NAMES = 'LaplacesDemon', ('rdirichlet', 'ddirichlet')


def fetch(version, cache, current):
    path = cache / 'tarballs' / f'{PKG}_{version}.tar.gz'
    if not path.exists():
        urls = [f'https://cran.r-project.org/src/contrib/Archive/{PKG}/{PKG}_{version}.tar.gz']
        if version == current:
            urls.insert(0, f'https://cran.r-project.org/src/contrib/{PKG}_{version}.tar.gz')
        for url in urls * 6:  # CRAN throttles bursts
            done = subprocess.run(['curl', '-s', '-f', '-L', '--retry', '5', '--retry-delay', '3', url],
                                  capture_output=True)
            if done.returncode == 0 and done.stdout[:2] == b'\x1f\x8b':
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(done.stdout)
                break
            time.sleep(5)
        else:
            raise RuntimeError(f'no tarball for {version}')
    return path.read_bytes()


def functions(data):
    found = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
        for m in tar.getmembers():
            if not (m.isfile() and re.match(PKG + r'/R/[^/]+\.[Rr]$', m.name)):
                continue
            lines = tar.extractfile(m).read().decode('utf-8', 'replace').splitlines()
            for i, line in enumerate(lines):
                for name in NAMES:
                    if re.match(r'\s*' + name + r'\s*<-\s*function', line):
                        depth, j, seen = 0, i, False
                        while j < len(lines):
                            depth += lines[j].count('{') - lines[j].count('}')
                            seen = seen or '{' in lines[j]
                            if seen and depth == 0:
                                break
                            j += 1
                        found[name] = {'file': m.name.split('/', 1)[1], 'text': '\n'.join(lines[i:j + 1])}
    return found


def main():
    cache, exports = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
    versions = exports['packages'][PKG]['window_versions']
    current = max(versions, key=lambda v: v['published'])['version']
    rows, texts = [], {}
    for v in versions:
        data = fetch(v['version'], cache, current)
        f = functions(data)
        row = {'version': v['version'], 'published': v['published'], 'tarball_sha256': hashlib.sha256(data).hexdigest()}
        for name in NAMES:
            key = hashlib.sha256(f[name]['text'].encode()).hexdigest()[:12]
            texts.setdefault(name, {})[key] = f[name]
            row[name] = key
        rows.append(row)
    result = {'kind': 'LaplacesDemon_dirichlet_source_comparison_not_runtime_binding', 'window': exports['window'],
              'versions': rows, 'distinct': {n: len(t) for n, t in texts.items()}, 'texts': texts}
    (OUT / 'dirichlet_versions.json').write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(rows, result['distinct'])
    for name, t in texts.items():
        for key, f in t.items():
            print('-----', name, key, f['file'])
            print(f['text'])


if __name__ == '__main__':
    main()
