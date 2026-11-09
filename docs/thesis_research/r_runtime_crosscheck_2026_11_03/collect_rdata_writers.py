"""Read only the serialization header of every .rdata/.RData/.rds below a root; print writer R versions.

Reads at most a few hundred decompressed bytes per file; nothing is written into the root.
"""
import bz2
import collections
import gzip
import json
import lzma
import os
import struct
import sys
from datetime import datetime, timezone

root, out = sys.argv[1], sys.argv[2]


def header(path):
    with open(path, 'rb') as f:
        magic = f.read(6)
    opener = {b'\x1f\x8b': gzip.open, b'BZ': bz2.open, b'\xfd7': lzma.open}.get(magic[:2], open)
    with opener(path, 'rb') as f:
        data = f.read(64)
    if data[:5] in (b'RDX2\n', b'RDX3\n', b'RDA2\n', b'RDA3\n'):
        data = data[5:]
    if data[:2] != b'X\n':
        return None
    version, writer, reader = struct.unpack('>iii', data[2:14])
    fmt = lambda v: f'{v >> 16}.{(v >> 8) & 255}.{v & 255}'
    return {'serialization': version, 'writer_R': fmt(writer)}


rows, counts = [], collections.Counter()
for dirpath, _, files in os.walk(root):
    for name in files:
        if not name.lower().endswith(('.rdata', '.rda', '.rds')):
            continue
        path = os.path.join(dirpath, name)
        try:
            h = header(path)
        except Exception as error:  # damaged or unreadable container
            h = {'error': type(error).__name__}
        mtime = datetime.fromtimestamp(os.path.getmtime(path), timezone.utc).date().isoformat()
        rows.append({'path': os.path.relpath(path, root).replace('\\', '/'), 'mtime': mtime, **(h or {'error': 'no X header'})})
        counts[(h or {}).get('writer_R', 'unknown')] += 1
json.dump({'root_label': 'diss', 'files': len(rows), 'writer_counts': dict(counts), 'rows': rows},
          open(out, 'w', encoding='utf-8'), indent=1)
print(len(rows), dict(counts))
