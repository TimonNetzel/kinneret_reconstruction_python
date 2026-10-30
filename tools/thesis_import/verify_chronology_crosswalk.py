#!/usr/bin/env python3
"""Independent Fraction-based check of every pilot correspondence candidate."""
import argparse
from collections import Counter
import csv
from decimal import Decimal, ROUND_HALF_EVEN
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path


def verify(raw, run):
    def pangaea(name):
        lines = (raw/name).read_text().splitlines()
        start = lines.index('*/')+1
        return list(csv.DictReader(lines[start:], delimiter='\t'))
    def output(name):
        with (run/name).open(newline='') as stream:
            return list(csv.DictReader(stream))
    manifest = json.loads((run/'manifest.json').read_text())
    for name, lock in manifest['outputs'].items():
        data = (run/name).read_bytes()
        assert len(data) == lock['byte_count'] and hashlib.sha256(data).hexdigest() == lock['sha256']
    for name, lock in manifest['sources'].items():
        assert hashlib.sha256((raw/name).read_bytes()).hexdigest() == lock
    original = list(csv.reader((raw/'age.txt').read_text().splitlines()[3:], delimiter='\t'))
    original = [r for r in original if r and r[0].strip()]
    old = Counter((Decimal(r[0]).quantize(Decimal('.001'), rounding=ROUND_HALF_EVEN), Decimal(r[1])) for r in original)
    new = Counter((Decimal(r['Depth sed [m]']), Decimal(r['Age model [ka]'])) for r in pangaea('monticchio_707050.tab'))
    assert old == new
    summary = pangaea('pollen_707075.tab')
    source = pangaea('monticchio_739452.tab')
    # Deliberately independent mapping and exact rational arithmetic; no production imports.
    names = {'Pin [#]':'Pinus [%]', 'Bet [#]':'Betula [%]', 'Que [#]':'Quercus [%]',
             'Aln [#]':'Alnus [%]', 'Abi [#]':'Abies [%]', 'Fag [#]':'Fagus [%]',
             'Art [#]':'Artemisia [%]', 'Cheae [#]':'Chenopod [%]'}
    matches = 0
    for r in output('sample_correspondence.csv'):
        s = source[int(r['source_row'])-1]
        candidates = []
        for i, p in enumerate(summary, 1):
            if all(abs(100*F(s[k])/F(p['Pollen tot [#]'])-F(p[v])) <= F(1, 200) for k, v in names.items()):
                candidates.append(i)
        assert len(candidates) == int(r['candidate_count'])
        if len(candidates) == 1:
            assert int(r['summary_source_row']) == candidates[0]
            assert F(r['original_summary_age_bp']) == 1000*F(summary[candidates[0]-1]['Age [ka BP]'])
            matches += 1
        else:
            assert r['status'] == 'unresolved' and r['summary_source_row'] == ''
    assert matches == 66
    print('Verified: source/output hashes, complete age multiset, all 69 candidate searches with Fraction arithmetic.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-dir', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    args = p.parse_args()
    verify(args.raw_dir, args.run_dir)
