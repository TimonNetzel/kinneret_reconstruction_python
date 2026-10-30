"""Confirm byte identity of two whole comparisons, including all large traces."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def compare(first, second, report):
    names = {p.name for p in first.iterdir()}
    if names != {p.name for p in second.iterdir()}:
        raise ValueError('Output file sets differ')
    files = []
    for name in sorted(names):
        a, b = first/name, second/name
        if not a.is_file() or not b.is_file() or a.stat().st_size != b.stat().st_size or digest(a) != digest(b):
            raise ValueError('Outputs differ: '+name)
        files.append({'file': name, 'bytes': a.stat().st_size, 'sha256': digest(a)})
    for folder in (first, second):
        manifest = json.loads((folder/'manifest.json').read_text())
        if names != set(manifest['outputs_sha256']) | {'manifest.json'}:
            raise ValueError('Manifest does not describe complete output set')
        for filename, expected in manifest['outputs_sha256'].items():
            if digest(folder/filename) != expected:
                raise ValueError('Manifest differs: '+filename)
    report.write_text(json.dumps({'result': 'all output bytes identical', 'complete_runs': 2,
        'files_checked': files, 'code_sha256': digest(Path(__file__)),
        'scope': 'Recorded Python/library environment; no cross-platform bitwise guarantee'}, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('first', 'second', 'report'):
        p.add_argument('--'+key, required=True, type=Path)
    a = p.parse_args()
    compare(a.first, a.second, a.report)
