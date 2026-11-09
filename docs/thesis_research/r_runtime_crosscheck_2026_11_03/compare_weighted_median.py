"""Compare matrixStats::weightedMedian across all versions available 2018-04-23..2023-07-07.

Reads R/weightedMedian.R and src/weightedMedian*.c / .h of each window version from the CRAN tarball (cached, read in
memory, never built or executed) and groups versions by identical source text. The call in mcmc_prepare.R
1631/1651 is weightedMedian(x, w) with all defaults, so the R signature and the C kernel decide the result.
Writes weighted_median_versions.json. Usage: compare_weighted_median.py <cache folder> <cran_exports.json>
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


def fetch(version, cache, current):
    path = cache / 'tarballs' / f'matrixStats_{version}.tar.gz'
    if not path.exists():
        urls = [f'https://cran.r-project.org/src/contrib/Archive/matrixStats/matrixStats_{version}.tar.gz']
        if version == current:
            urls.insert(0, f'https://cran.r-project.org/src/contrib/matrixStats_{version}.tar.gz')
        for url in urls * 6:  # CRAN throttles bursts: retry each address with growing pauses
            done = subprocess.run(['curl', '-s', '-f', '-L', '--retry', '5', '--retry-delay', '3', url],
                                  capture_output=True)
            if done.returncode == 0 and done.stdout[:2] == b'\x1f\x8b':
                data = done.stdout
                break
            time.sleep(5 + 10 * urls.index(url))
        else:
            raise RuntimeError(f'no tarball for {version}')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return path.read_bytes()


def main():
    cache, exports = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
    versions = exports['packages']['matrixStats']['window_versions']
    current = max(versions, key=lambda v: v['published'])['version']
    rows, groups = [], {}
    for v in versions:
        data = fetch(v['version'], cache, current)
        parts = {}
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
            for m in tar.getmembers():
                if m.isfile() and re.match(r'matrixStats/(R/weightedMedian\.R|src/weightedMedian[^/]*\.(c|h))$', m.name):
                    parts[m.name.split('/', 1)[1]] = tar.extractfile(m).read().decode('utf-8', 'replace')
        r_text = parts.get('R/weightedMedian.R', '')
        signature = re.search(r'weightedMedian\s*<-\s*function\s*\((.*?)\)\s*\{', r_text, re.S)
        key = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()
        groups.setdefault(key, []).append(v['version'])
        rows.append({'version': v['version'], 'published': v['published'],
                     'tarball_sha256': hashlib.sha256(data).hexdigest(), 'files': sorted(parts),
                     'signature': ' '.join(signature.group(1).split()) if signature else None,
                     'source_group': key[:12]})
    result = {'kind': 'matrixStats_weightedMedian_source_comparison_not_runtime_binding',
              'window': exports['window'], 'call_in_source': 'weightedMedian(x1range.bck, tf.x1.current[j,]) and '
              'weightedMedian(x2range.bck, tf.x2.current[j,]) (mcmc_prepare.R 1631/1651), all defaults',
              'versions': rows, 'distinct_sources': len(groups),
              'groups': {k[:12]: v for k, v in groups.items()}}
    (OUT / 'weighted_median_versions.json').write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8',
                                                       newline='\n')
    print(len(rows), 'versions,', len(groups), 'distinct sources')
    for r in rows:
        print(r['version'], r['published'], r['source_group'], r['signature'])


if __name__ == '__main__':
    main()
