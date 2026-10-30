#!/usr/bin/env python3
"""Independent standard-library audit of the Monticchio diagnostic outputs."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def audit(directory):
    def rows(name):
        with (directory/name).open(encoding='utf-8', newline='') as stream:
            return list(csv.DictReader(stream))
    summary = json.loads((directory/'summary.json').read_text())
    manifest = json.loads((directory/'manifest.json').read_text())
    for name, expected in manifest['outputs'].items():
        data = (directory/name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected['sha256'], name
        assert len(data) == expected['byte_count'], name
    ref = rows('modern_distance_reference.csv')
    support = rows('fossil_support.csv')
    analogue = rows('fossil_analogues.csv')
    ages = rows('chronology_samples.csv')
    assert len(ref) == len({r['sample_id'] for r in ref}) == 7078
    assert len(ages) == len({r['sample_id'] for r in ages}) == 69
    assert len(support) == len({(r['sample_id'], r['scenario']) for r in support}) == 138
    assert len(analogue) == len({(r['sample_id'], r['scenario'], r['rank']) for r in analogue}) == 690
    neighbours = defaultdict(list)
    for row in analogue:
        neighbours[row['sample_id'], row['scenario']].append(row)
        assert row['modern_sample_id'] not in ('EPDcoretop_E932', 'EPDcoretop_E453')
        if row['scenario'] == 'buffer100km':
            assert float(row['geographic_distance_km']) > 100
    for scenario in ('other_coordinate', 'buffer100km'):
        selected = [r for r in support if r['scenario'] == scenario]
        assert len(selected) == 69
        for metric in ('d1', 'd5'):
            values = [float(r[scenario+'_'+metric]) for r in ref]
            # Inclusive order statistics, independent of numpy.quantile.
            quantiles = statistics.quantiles(values, n=100, method='inclusive')
            for name, cutoff in [('p95', quantiles[94]), ('p99', quantiles[98])]:
                assert math.isclose(cutoff, summary['reference_quantiles'][scenario][metric][name], rel_tol=1e-14)
                key = metric+'_above_'+name
                assert sum(int(r[key]) for r in selected) == summary['scenarios'][scenario]['flags'][key]
                for row in selected:
                    assert int(row[key]) == int(float(row[metric]) > cutoff)
        for row in selected:
            nn = sorted(neighbours[row['sample_id'], scenario], key=lambda r: int(r['rank']))
            assert [int(r['rank']) for r in nn] == [1, 2, 3, 4, 5]
            dist = [float(r['hellinger_distance']) for r in nn]
            assert dist == sorted(dist) and all(0 <= d <= 1 for d in dist)
            assert float(row['d1']) == dist[0] and float(row['d5']) == dist[-1]
            assert int(row['k5_coordinate_groups']) == len({r['coordinate_group'] for r in nn})
    widths = [float(r['age_max_bp'])-float(r['age_min_bp']) for r in ages]
    for row, width in zip(ages, widths):
        assert float(row['age_min_bp']) <= float(row['age_bp']) <= float(row['age_max_bp'])
        assert math.isclose(width, float(row['interval_width_years']), abs_tol=1e-10)
    assert statistics.median(widths) == summary['chronology']['interval_width_years']['median']
    assert dict(Counter(r['uncertainty_class'] for r in ages)) == summary['chronology']['uncertainty_classes']
    assert sum(float(r['previous_interval_overlap_years'] or 0) > 0 for r in ages) == 46
    print('Verified: all output hashes; inclusive quantiles and all flags; 138 top-5 tables; 69 age intervals.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    audit(parser.parse_args().run_dir)
