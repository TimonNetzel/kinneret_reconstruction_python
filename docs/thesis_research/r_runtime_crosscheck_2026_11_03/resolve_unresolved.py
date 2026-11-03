"""Resolve the packages that provider_analysis.json cannot settle from NAMESPACE alone.

For every package in front of a provider on the search path that uses exportPattern() or has window versions missing
in the GitHub mirror, the CRAN tarball of each such version is downloaded (cached, never extracted or executed) and
read in memory: NAMESPACE exports and top-level definitions `name <- function` / `name = function` in R/*.R.
Adds the result to provider_analysis.json under "tarball_resolution".
Usage: resolve_unresolved.py <cache folder> <cran_exports.json>
"""
import hashlib
import io
import json
import re
import sys
import tarfile
import time
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent
NAMES = ('rdirichlet', 'ddirichlet', 'weightedMedian')


def fetch(pkg, version, cache, current):
    path = cache / 'tarballs' / f'{pkg}_{version}.tar.gz'
    if not path.exists():
        urls = [f'https://cran.r-project.org/src/contrib/Archive/{pkg}/{pkg}_{version}.tar.gz']
        if version == current:
            urls.insert(0, f'https://cran.r-project.org/src/contrib/{pkg}_{version}.tar.gz')
        for url in urls * 6:  # CRAN throttles bursts: retry each address with growing pauses
            done = subprocess.run(['curl', '-s', '-f', '-L', '--retry', '5', '--retry-delay', '3', url],
                                  capture_output=True)
            if done.returncode == 0 and done.stdout[:2] == b'\x1f\x8b':
                data = done.stdout
                break
            time.sleep(5 + 10 * urls.index(url))
        else:
            raise RuntimeError(f'no tarball for {pkg} {version}')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return path.read_bytes()


def inspect(data, pkg):
    defined, ns = set(), ''
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = member.name
            if name == f'{pkg}/NAMESPACE':
                ns = tar.extractfile(member).read().decode('utf-8', 'replace')
            elif re.match(re.escape(pkg) + r'/R/[^/]+\.[Rr]$', name):
                text = tar.extractfile(member).read().decode('utf-8', 'replace')
                for n in NAMES:
                    if re.search(r'^[ \t]*[`"\']?' + n + r'[`"\']?\s*(<-|=)\s*function', text, re.M):
                        defined.add(n)
    code = '\n'.join(line.split('#', 1)[0] for line in ns.splitlines())
    explicit = set()
    for m in re.finditer(r'\bexport\s*\((.*?)\)', code, re.S):
        explicit.update(x.strip().strip('"\'`') for x in m.group(1).split(','))
    patterns = re.findall(r'exportPattern\s*\(\s*["\']([^"\']+)["\']', code)
    exported = {n: n in explicit or (n in defined and any(re.search(p.replace('[[:alpha:]]', '[A-Za-z]'), n)
                                                         for p in patterns)) for n in NAMES}
    return {'defined': sorted(defined), 'export_pattern': patterns, **exported}


def main():
    cache, exports = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
    analysis = json.loads((OUT / 'provider_analysis.json').read_text(encoding='utf-8'))
    path = analysis['search_path_front_to_back']
    todo = {}
    for name, provider in (('rdirichlet', 'LaplacesDemon'), ('ddirichlet', 'LaplacesDemon'),
                           ('weightedMedian', 'matrixStats')):
        if analysis['providers'][name]['first_on_search_path'] != provider:
            raise ValueError(f'unexpected first provider of {name}')
        for pkg in path[:path.index(provider)]:
            if pkg in analysis['export_pattern_packages_not_resolved_by_namespace'] or \
                    pkg in analysis['versions_missing_in_mirror']:
                todo.setdefault(pkg, set()).add(name)
    resolution = {}
    for pkg in sorted(todo):
        versions = exports['packages'][pkg]['window_versions']
        current = max(versions, key=lambda v: v['published'])['version']
        rows = []
        for v in versions:
            data = fetch(pkg, v['version'], cache, current)
            rows.append({'version': v['version'], 'sha256': hashlib.sha256(data).hexdigest(), **inspect(data, pkg)})
        resolution[pkg] = {'checked_for': sorted(todo[pkg]), 'versions': rows,
                           'exports_any': {n: any(r[n] for r in rows) for n in NAMES}}
        print(pkg, {n: any(r[n] for r in rows) for n in NAMES}, flush=True)
    blocking = {n: [p for p, r in resolution.items() if n in r['checked_for'] and r['exports_any'][n]] for n in NAMES}
    analysis['tarball_resolution'] = {'packages': resolution, 'blocking_providers_in_front': blocking}
    analysis['providers_bound_within_window'] = {
        n: (analysis['providers'][n]['first_on_search_path'] if not blocking[n] else None) for n in NAMES}
    (OUT / 'provider_analysis.json').write_text(json.dumps(analysis, indent=1) + '\n', encoding='utf-8', newline='\n')
    print('blocking:', blocking)


if __name__ == '__main__':
    main()
