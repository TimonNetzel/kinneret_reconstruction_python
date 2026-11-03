"""Which attached packages (and their Depends) export rdirichlet / ddirichlet in versions available 2018-04-23..2023-07-07?

Version dates from the CRAN archive and contrib listings (authoritative); NAMESPACE and DESCRIPTION per version from the
read-only GitHub mirror github.com/cran/<pkg> (one tag per CRAN version). Downloads are cached as plain text under the
cache folder given on the command line; nothing is executed. Writes one JSON with every observation.
"""
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CACHE = Path(sys.argv[1])
OUT = Path(sys.argv[2])
START, END = '2018-04-23', '2023-07-07'
ATTACHED = ['SBMSplitMerge', 'doParallel', 'foreach', 'doSNOW', 'ncdf4', 'rbacon', 'Hmisc', 'gdata', 'clam', 'fields',
            'maps', 'smotefamily', 'verification', 'NeuralNetTools', 'RColorBrewer', 'mvtnorm', 'gtools', 'abind', 'kgc',
            'emdist', 'signal', 'broman', 'MatrixCorrelation', 'matrixStats', 'fitdistrplus', 'Rcpp', 'RcppArmadillo',
            'RcppZiggurat', 'extraDistr', 'lsa', 'NbClust', 'Rfast', 'PTAk', 'magic', 'LaplacesDemon', 'wrswoR',
            'stringr', 'readxl', 'pracma', 'EnvStats', 'rPref', 'spatstat.utils', 'WaveletComp', 'biwavelet', 'caret',
            'gbm', 'nnet', 'earth', 'mda', 'kernlab', 'RRF', 'MASS', 'npreg', 'truncnorm', 'colorRamps']
BASE = {'R', 'base', 'stats', 'graphics', 'grDevices', 'utils', 'datasets', 'methods', 'grid', 'parallel', 'splines',
        'stats4', 'tools', 'tcltk', 'compiler'}
MONTHS = {m: i for i, m in enumerate(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov',
                                      'Dec'], 1)}


def get(url, name):
    path = CACHE / name
    if path.exists():
        return path.read_text(encoding='utf-8', errors='replace')
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                text = r.read().decode('utf-8', errors='replace')
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                text = '__404__'
                break
            time.sleep(2 + attempt * 3)
        except Exception:
            time.sleep(2 + attempt * 3)
    else:
        raise RuntimeError(f'failed {url}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return text


def listing_dates(html, pkg):
    """version -> ISO date from an Apache index (both 'YYYY-MM-DD' and 'DD-Mon-YYYY' styles)."""
    out = {}
    for m in re.finditer(r'(?<=[">])' + re.escape(pkg) + r'_([0-9][0-9.\-]*)\.tar\.gz</a>(?:</td><td[^>]*>)?\s*(\S+)', html):
        v, d = m.group(1), m.group(2)
        if re.match(r'\d{4}-\d{2}-\d{2}', d):
            out[v] = d[:10]
        elif re.match(r'\d{2}-[A-Z][a-z]{2}-\d{4}', d):
            day, mon, year = d.split('-')
            out[v] = f'{year}-{MONTHS[mon]:02d}-{day}'
    return out


def versions(pkg, contrib):
    dates = listing_dates(get(f'https://cran.r-project.org/src/contrib/Archive/{pkg}/', f'archive/{pkg}.html'), pkg)
    dates.update(listing_dates(contrib, pkg))
    ordered = sorted(dates.items(), key=lambda kv: kv[1])
    window = []
    for i, (v, d) in enumerate(ordered):
        nxt = ordered[i + 1][1] if i + 1 < len(ordered) else '9999-12-31'
        if d <= END and nxt > START:
            window.append({'version': v, 'published': d})
    return window, len(ordered)


def deps(desc):
    m = re.search(r'^Depends:(.*?)(?=^\S)', desc + '\nX:', re.S | re.M)
    if not m:
        return []
    return [re.split(r'[\s(]', p.strip())[0] for p in m.group(1).replace('\n', ' ').split(',') if p.strip()]


def exports(ns):
    code = '\n'.join(line.split('#', 1)[0] for line in ns.splitlines())
    explicit = set()
    for m in re.finditer(r'\bexport\s*\((.*?)\)', code, re.S):
        explicit.update(x.strip().strip('"\'`') for x in m.group(1).split(','))
    patterns = re.findall(r'exportPattern\s*\(\s*["\']([^"\']+)["\']', code)
    return explicit, patterns


def main():
    contrib = get('https://cran.r-project.org/src/contrib/', 'contrib.html')
    todo, seen, result = list(ATTACHED), set(), {}
    while todo:
        pkg = todo.pop(0)
        if pkg in seen or pkg in BASE:
            continue
        seen.add(pkg)
        window, total = versions(pkg, contrib)
        rows = []
        for item in window:
            v = item['version']
            ns = get(f'https://raw.githubusercontent.com/cran/{pkg}/{v}/NAMESPACE', f'mirror/{pkg}/{v}/NAMESPACE')
            desc = get(f'https://raw.githubusercontent.com/cran/{pkg}/{v}/DESCRIPTION', f'mirror/{pkg}/{v}/DESCRIPTION')
            if ns == '__404__':
                rows.append({**item, 'mirror': False})
                continue
            explicit, patterns = exports(ns)
            d = deps(desc)
            rows.append({**item, 'mirror': True, 'depends': d, 'export_pattern': patterns,
                         'exports_rdirichlet': 'rdirichlet' in explicit, 'exports_ddirichlet': 'ddirichlet' in explicit})
            for dep in d:
                if dep not in seen and dep not in BASE:
                    todo.append(dep)
        result[pkg] = {'attached_directly': pkg in ATTACHED, 'cran_versions_total': total, 'window_versions': rows}
        print(pkg, len(rows), flush=True)
    OUT.write_text(json.dumps({'window': [START, END], 'packages': result}, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
