"""Materialize only the declared, immutable public inputs; verify before use.

No external code is executed here. Existing mismatched files are never replaced.
The separate comparison/audit later import their explicitly pinned PR28 readers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def materialize(record, target, verify_only):
    def valid(path):
        return path.stat().st_size == record['bytes'] and digest(path) == record['sha256']
    if target.exists():
        if not valid(target):
            raise ValueError('Existing input differs; not overwritten: '+str(target))
    elif verify_only:
        raise FileNotFoundError(target)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name+'.part')
        # Exclusive creation prevents replacing unrelated partial work.
        created = False
        try:
            with partial.open('xb') as dest:
                created = True
                with urllib.request.urlopen(record['url'], timeout=60) as response:
                    while chunk := response.read(1024*1024):
                        dest.write(chunk)
            if not valid(partial):
                raise ValueError('Downloaded input hash/size differs: '+str(target))
            partial.rename(target)
        except Exception:
            # Only remove the file this invocation created.
            if created and partial.exists():
                partial.unlink()
            raise
    print('verified', target, flush=True)


def prepare(sources_path, audit_sources_path, previous, maps, climate, verify_only):
    sources = json.loads(sources_path.read_text())
    audit_sources = json.loads(audit_sources_path.read_text())
    for record in sources['previous_files']+[audit_sources['previous_raw_checker']]:
        if Path(record['file']).name != record['file']:
            raise ValueError('Input filename must be a basename')
        materialize(record, previous/record['file'], verify_only)
    old = json.loads((previous/'sources.json').read_text())
    for record in old['map_files']:
        path = Path(record['path'])
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe map path')
        materialize(record, maps/path, verify_only)
    materialize(old['climate_file'], climate, verify_only)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('sources', 'audit_sources', 'previous', 'maps', 'climate'):
        parser.add_argument('--'+key.replace('_', '-'), required=True, type=Path)
    parser.add_argument('--verify-only', action='store_true')
    a = parser.parse_args()
    prepare(a.sources, a.audit_sources, a.previous, a.maps, a.climate, a.verify_only)
