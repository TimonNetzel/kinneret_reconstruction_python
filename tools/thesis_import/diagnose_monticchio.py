#!/usr/bin/env python3
"""Read-only Shared37 analogue-support and source-chronology diagnostics.

No climate predictions and no import of the baseline evaluator. See DESIGN.md.
"""
import argparse
from collections import Counter
import csv
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.spatial.distance import cdist

from import_data import read_pangaea

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/thesis_research'
CONTROL_SHA = 'c96971bd5a9ff106663f8160f48e18cea34c5257258084ce3421190c5dc2bb05'
EXTRA_SOURCES = {
    'monticchio_740512.tab': 'b578952bddb100e4b4527c1fcbc89f5f97d3deb6c62ab3e3120c1be49c4a232a',
    'monticchio_741058.tab': 'f5bdb0dd008c9e714b87c60238b1cae452414567abd091a2c3616370125d2635',
    'monticchio_707050.tab': '80cde942aea6d0bc0c550cbd4cc4d20ee3a84366a9711d980dcb46275fb176f2',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, records):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def stats(values):
    return {k: float(v) for k, v in zip(('min', 'median', 'p95', 'p99', 'max'),
            np.quantile(values, [0, .5, .95, .99, 1], method='linear'))}


def geo_km(query, reference):
    """Independent haversine calculation, coordinates in degrees, lat/lon."""
    a, b = np.deg2rad(query), np.deg2rad(reference)
    dlat = a[:, None, 0]-b[None, :, 0]
    dlon = a[:, None, 1]-b[None, :, 1]
    h = np.sin(dlat/2)**2 + np.cos(a[:, None, 0])*np.cos(b[None, :, 0])*np.sin(dlon/2)**2
    return 2*6371.0088*np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def features(rows, names):
    counts = np.array([[float(r[n]) for n in names] for r in rows])
    require(np.isfinite(counts).all() and (counts >= 0).all() and (counts == np.floor(counts)).all(), 'Invalid counts')
    total = counts.sum(axis=1)
    require(np.all(total > 0), 'Empty subcomposition')
    require(np.allclose(total, [float(r['shared_count_sum']) for r in rows], rtol=0, atol=1e-8), 'Wrong sum')
    without = total-counts[:, names.index('pinus')]
    require(np.allclose(without, [float(r['shared_count_without_pinus']) for r in rows], rtol=0, atol=1e-8), 'Wrong no-Pinus sum')
    require(all(int(r['count_threshold_pass']) == int(v >= 100) for r, v in zip(rows, without)), 'Wrong threshold')
    return np.sqrt(counts/total[:, None])


def direct_distances(query, references):
    # Pure Python subtraction/fsum, independent of scipy.cdist and numpy reduction.
    return [math.sqrt(math.fsum((a-b)**2 for a, b in zip(query, ref))/2) for ref in references]


def chronology(raw, fossil, policy):
    def table(filename):
        return read_pangaea(raw/filename)[2]
    cut = [r for r in table('epd_ages_cut_804596.tab') if r['Event'] == 'LGM90D' and r['ID'] == '932']
    all_ages = [r for r in table('epd_ages_804595.tab') if r['Event'] == 'LGM90D' and r['ID'] == '932']
    entities = [r for r in table('epd_entity_804591.tab') if r['Event'] == 'LGM90D']
    require(len(entities) == 1, 'Ambiguous entity')
    entity = entities[0]
    require(next(v for k, v in entity.items() if k.startswith('ID (E#')) == '932', 'Wrong entity')
    require(next(v for k, v in entity.items() if k.startswith('Depth top')) == '0.00', 'Changed cut top')
    require(next(v for k, v in entity.items() if k.startswith('Depth bot')) == '12.000', 'Changed cut bottom')
    require(next(v for k, v in entity.items() if k.startswith('Age model type')) == '1', 'Changed model type')
    require(len(cut) == len(fossil) == 69 and len(all_ages) == 245, 'Changed source cohort')
    all_by_depth = {r['Depth sed [m]']: r for r in all_ages}
    require(all(all_by_depth[r['Depth sed [m]']] == r for r in cut), 'Cut/All mismatch')
    require(all(0 <= Decimal(r['Depth sed [m]']) <= 12 for r in cut), 'Cut outside interval')
    require(sum(0 <= Decimal(r['Depth sed [m]']) <= 12 for r in all_ages) == 69, 'Additional in-cut All samples')
    pollen = table('monticchio_739452.tab')
    used = set()
    pairs = []
    for row in cut:
        depth = Decimal(row['Depth sed [m]'])
        matches = [i for i, r in enumerate(pollen) if abs(Decimal(r['Depth sed [m]'])-depth) <= Decimal('.005')]
        require(len(matches) == 1 and matches[0] not in used, 'Nonunique independent depth join')
        i = matches[0]
        used.add(i)
        r = pollen[i]
        sample = next(f for f in fossil if f['sample_id'] == f'LGM90D:{i+1:06d}')
        for target, source in [('age_bp', 'Age dated [ka BP]'), ('age_min_bp', 'Age min [ka]'), ('age_max_bp', 'Age max [ka]')]:
            require(Decimal(sample[target]) == 1000*Decimal(row[source]), 'Fossil age differs from source')
        require(Decimal(sample['depth_m']) == Decimal(r['Depth sed [m]']), 'Fossil depth differs')
        require(Decimal(sample['chronology_depth_m']) == depth, 'Chronology depth differs')
        require(sample['chronology_uncertainty_class'] == row['Uncertainty class'], 'Wrong class')
        offset = Decimal(r['Depth sed [m]'])-depth
        require(Decimal(sample['chronology_depth_offset_m']) == offset, 'Wrong offset')
        require(sample['chronology_join_kind'] == ('exact' if offset == 0 else 'rounded_within_5mm'), 'Wrong join flag')
        for f in policy['features']:
            require(Decimal(sample[f['feature_id']]) == sum(Decimal(r[c]) for c in f['fossil_columns']), 'Independent fossil aggregation differs')
        pairs.append(sample)
    pairs.sort(key=lambda r: Decimal(r['chronology_depth_m']))
    depth = np.array([float(r['chronology_depth_m']) for r in pairs])
    age = np.array([float(r['age_bp']) for r in pairs])
    lo = np.array([float(r['age_min_bp']) for r in pairs])
    hi = np.array([float(r['age_max_bp']) for r in pairs])
    require((np.diff(depth) > 0).all() and (np.diff(age) > 0).all(), 'Nonmonotone central chronology')
    require(((lo <= age) & (age <= hi)).all(), 'Invalid age bounds')
    controls = [r for r in table('epd_control_804606.tab') if r['Event'] == 'LGM90D' and r['ID'] == '932']
    anl = [r for r in controls if r['Code'] == 'ANL']
    cdepth = np.array([float(r['Depth sed [m]']) for r in controls])
    cage = np.array([float(r['Age dated [ka BP]'])*1000 for r in controls])
    require((np.diff(cdepth) >= 0).all() and (np.diff(cage) >= 0).all(), 'Reversed control table')
    repeated_depths = {d: [r for r in controls if r['Depth sed [m]'] == d]
                       for d, n in Counter(r['Depth sed [m]'] for r in controls).items() if n > 1}
    rounding = []
    for i, r in enumerate(pairs):
        pdepth = float(r['depth_m'])
        # Local finite slope of the published central Cut chronology, not a new age model.
        left = max(0, i-1); right = min(len(pairs)-1, i+1)
        slope = (age[right]-age[left])/(depth[right]-depth[left])
        rounding.append(slope*float(r['chronology_depth_offset_m']))
    rows = [dict(sample_id=r['sample_id'], depth_m=r['depth_m'], chronology_depth_m=r['chronology_depth_m'],
                 depth_offset_m=r['chronology_depth_offset_m'], age_bp=age[i], age_min_bp=lo[i], age_max_bp=hi[i],
                 interval_width_years=hi[i]-lo[i], uncertainty_class=r['chronology_uncertainty_class'],
                 previous_central_gap_years=age[i]-age[i-1] if i else '',
                 previous_interval_overlap_years=max(0, min(hi[i], hi[i-1])-max(lo[i], lo[i-1])) if i else '',
                 offset_times_local_slope_years=rounding[i],
                 nearest_control_age_gap_years=float(np.min(np.abs(cage-age[i])))) for i, r in enumerate(pairs)]
    interval_controls = [r for r in controls if 0 <= float(r['Depth sed [m]']) <= 12]
    excluded = [r for r in all_ages if r['Depth sed [m]'] not in {x['Depth sed [m]'] for x in cut}]
    return rows, dict(source_samples=len(pollen), cut_samples=len(cut), excluded_samples=len(excluded),
        central_age_bp=[float(age.min()), float(age.max())], age_bounds_envelope_bp=[float(lo.min()), float(hi.max())],
        depth_interval_m=[0, 12], age_model_type='1=linear',
        uncertainty_classes=dict(Counter(r['chronology_uncertainty_class'] for r in pairs)),
        interval_width_years=stats(hi-lo), central_gap_years=stats(np.diff(age)),
        adjacent_interval_overlaps=int(sum(r['previous_interval_overlap_years'] > 0 for r in rows[1:])),
        rounded_depth_matches=int(sum(Decimal(r['chronology_depth_offset_m']) != 0 for r in pairs)),
        maximum_abs_offset_times_local_slope_years=float(max(abs(x) for x in rounding)),
        all_excluded_central_age_bp=stats([1000*float(r['Age dated [ka BP]']) for r in excluded]),
        controls=len(controls), control_codes=dict(Counter(r['Code'] for r in controls)),
        controls_in_cut_depth_range=len(interval_controls), control_codes_in_cut=dict(Counter(r['Code'] for r in interval_controls)),
        anl_errors_match_rounded_one_percent_with_one_year_floor=all(
            abs(float(r['Age e [±]'])-max(.001, .01*float(r['Age dated [ka BP]']))) <= .00005001 for r in anl),
        all_control_calendar_ages_present=all(r['Age dated [ka BP]'] != '' for r in controls),
        control_radiocarbon_age_fields_nonempty=sum(r['Age dated [ka]'] != '' for r in controls),
        repeated_control_depths=len(repeated_depths),
        repeated_depths_with_different_ages=sum(len({r['Age dated [ka BP]'] for r in rr}) > 1 for rr in repeated_depths.values()),
        exact_duplicate_control_rows=len(controls)-len({tuple(r.values()) for r in controls}),
        control_age_plateaus=int(sum(np.diff(cage) == 0)),
        joint_ensemble_available=False, retained_all_rows_equal_cut=True, independent_fossil_aggregations_checked=69*37), controls


def run(shared, raw, output):
    require(not output.exists(), 'Output exists')
    policy_path = DOC/'modern2025_2026_10_29/shared_features.json'
    policy = json.loads(policy_path.read_text())
    names = [f['feature_id'] for f in policy['features']]
    manifest = json.loads((shared/'manifest.json').read_text())
    reference = json.loads((DOC/'modern2025_2026_10_29/reference_manifest.json').read_text())
    require(manifest == reference, 'Shared37 manifest differs from locked reference')
    require(sha(policy_path) == manifest['feature_policy_sha256'], 'Changed policy')
    for name in ('modern_shared_counts.csv', 'fossil_shared_counts.csv'):
        require(sha(shared/name) == manifest['outputs'][name]['sha256'], 'Changed shared table')
    catalog = json.loads((DOC/'source_audit_2026_10_29/sources.json').read_text())
    source_locks = {}
    for s in catalog['files']:
        if s['source_id'] in ('monticchio_739452', 'epd_entity_804591', 'epd_ages_cut_804596', 'epd_ages_804595'):
            require(sha(raw/s['file']) == s['sha256'] and (raw/s['file']).stat().st_size == s['byte_count'], 'Changed raw source')
            source_locks[s['file']] = s['sha256']
    require(sha(raw/'epd_control_804606.tab') == CONTROL_SHA, 'Changed Control_P source')
    source_locks['epd_control_804606.tab'] = CONTROL_SHA
    for name, expected in EXTRA_SOURCES.items():
        require(sha(raw/name) == expected, 'Changed extra chronology source')
        source_locks[name] = expected
    modern_all = read_csv(shared/'modern_shared_counts.csv')
    features(modern_all, names)
    modern = sorted([r for r in modern_all if r['count_threshold_pass'] == '1'], key=lambda r: r['sample_id'])
    fossil = sorted(read_csv(shared/'fossil_shared_counts.csv'), key=lambda r: r['sample_id'])
    require(len(modern) == 7078 and len(fossil) == 69 and len(names) == 37, 'Cohort differs')
    x, z = features(modern, names), features(fossil, names)
    coordinates = np.array([[float(r['latitude']), float(r['longitude'])] for r in modern])
    coordinates[:, 1] = (coordinates[:, 1]+180) % 360-180
    groups = np.array([f'{a:.6f}|{b:.6f}' for a, b in coordinates])
    fossil_coordinates = np.tile([40.94444, 15.6], (len(fossil), 1))
    modern_reference = []
    modern_spotchecks = 0
    for start in range(0, len(x), 128):
        distance = cdist(x[start:start+128], x)/math.sqrt(2)
        geo = geo_km(coordinates[start:start+128], coordinates)
        for j, i in enumerate(range(start, min(start+128, len(x)))):
            record = dict(sample_id=modern[i]['sample_id'])
            direct = None
            if i in (0, len(x)//2, len(x)-1):
                direct = np.array(direct_distances(x[i].tolist(), x.tolist()))
                require(np.allclose(distance[j], direct, atol=1e-14, rtol=1e-13), 'Modern direct distances differ')
                modern_spotchecks += 1
            for scenario, mask in [('other_coordinate', groups == groups[i]), ('buffer100km', geo[j] <= 100)]:
                values = distance[j].copy(); values[mask] = np.inf
                order = np.argsort(values, kind='stable')[:5]
                require(np.isfinite(values[order]).all(), 'Not enough analogues')
                record[scenario+'_d1'] = float(values[order[0]])
                record[scenario+'_d5'] = float(values[order[-1]])
                if direct is not None:
                    direct_masked = direct.copy(); direct_masked[mask] = np.inf
                    require(np.array_equal(order, np.argsort(direct_masked, kind='stable')[:5]), 'Modern direct rank differs')
            modern_reference.append(record)
    thresholds = {scenario: {metric: stats([r[scenario+'_'+metric] for r in modern_reference]) for metric in ('d1', 'd5')}
                  for scenario in ('other_coordinate', 'buffer100km')}
    fd = cdist(z, x)/math.sqrt(2)
    fg = geo_km(fossil_coordinates, coordinates)
    summaries, analogues = [], []
    direct_rows = x.tolist()
    for i, sample in enumerate(fossil):
        direct = np.array(direct_distances(z[i].tolist(), direct_rows))
        require(np.allclose(fd[i], direct, atol=1e-14, rtol=1e-13), 'Fossil direct distance differs')
        for scenario, mask in [('other_coordinate', groups == '40.944440|15.600000'), ('buffer100km', fg[i] <= 100)]:
            values = fd[i].copy(); values[mask] = np.inf
            order = np.argsort(values, kind='stable')[:5]
            direct_masked = direct.copy(); direct_masked[mask] = np.inf
            require(np.array_equal(order, np.argsort(direct_masked, kind='stable')[:5]), 'Fossil direct rank differs')
            distances = values[order]
            record = dict(sample_id=sample['sample_id'], scenario=scenario, age_bp=float(sample['age_bp']),
                          d1=float(distances[0]), d5=float(distances[-1]),
                          k5_coordinate_groups=len(set(groups[order])), eligible_reference_samples=int((~mask).sum()))
            for metric in ('d1', 'd5'):
                for quantile in ('p95', 'p99'):
                    record[metric+'_above_'+quantile] = int(record[metric] > thresholds[scenario][metric][quantile])
            summaries.append(record)
            for rank, j in enumerate(order, 1):
                analogues.append(dict(sample_id=sample['sample_id'], scenario=scenario, rank=rank,
                    modern_sample_id=modern[j]['sample_id'], hellinger_distance=float(values[j]),
                    latitude=float(coordinates[j, 0]), longitude=float(coordinates[j, 1]),
                    coordinate_group=groups[j], geographic_distance_km=float(fg[i, j]),
                    sample_context=modern[j]['sample_context']))
    chron_rows, chron_summary, controls = chronology(raw, fossil, policy)
    allen = read_pangaea(raw/'monticchio_707050.tab')[2]
    radiocarbon = read_pangaea(raw/'monticchio_740512.tab')[2]
    tephra = read_pangaea(raw/'monticchio_741058.tab')[2]
    allen_pairs = {(Decimal(r['Depth sed [m]']), Decimal(r['Age model [ka]'])) for r in allen}
    control_pairs = {(Decimal(r['Depth sed [m]']), Decimal(r['Age dated [ka BP]'])) for r in controls if r['Code'] == 'ANL'}
    extra_summary = dict(
        allen1999_rows=len(allen), allen1999_unique_pairs=len(allen_pairs),
        allen1999_pairs_exactly_shared_with_epd_controls=len(allen_pairs & control_pairs),
        allen1999_pairs_within_cut_depth_range=sum(0 <= d <= 12 for d, age in allen_pairs),
        radiocarbon_dates=len(radiocarbon), radiocarbon_dates_inside_cut=sum(0 <= float(r['Depth sed [m]']) <= 12 for r in radiocarbon),
        radiocarbon_dates_depth_by_pollen_correlation=sum('pollen correlation' in r['Comment'] for r in radiocarbon),
        tephra_layers=len(tephra), tephra_layers_inside_cut=sum(0 <= float(r['Depth sed [m]']) <= 12 for r in tephra),
        source_14c_values_recalibrated=False, allen_composite_depths_adopted=False)
    output.mkdir(parents=True)
    write_csv(output/'fossil_support.csv', summaries)
    write_csv(output/'fossil_analogues.csv', analogues)
    write_csv(output/'modern_distance_reference.csv', modern_reference)
    write_csv(output/'chronology_samples.csv', chron_rows)
    write_csv(output/'chronology_controls.csv', controls)
    summary = dict(schema_version=1, modern_samples=len(modern), fossil_samples=len(fossil), features=names,
        modern_coordinate_groups=len(set(groups)), distance='Hellinger [0,1]',
        fossil_location_source='EPD Entity 932; 40.94444 N, 15.60000 E; altitude unused',
        reference_quantiles=thresholds, scenarios={}, chronology=chron_summary, extra_chronology_sources=extra_summary,
        modern_same_coordinates_excluded=[{k: r[k] for k in ('sample_id', 'source_site_code', 'sample_context')}
            for r, g in zip(modern, groups) if g == '40.944440|15.600000'],
        verification=dict(all_fossil_distance_cells_direct_checked=int(fd.size), all_fossil_top5_ranks_direct_checked=138,
                          modern_direct_spotchecks=modern_spotchecks, scipy_evaluator_imported=False),
        climate_predictions=False, distance_flags_are_validated_no_analogue_cutoffs=False)
    for scenario in thresholds:
        selected = [r for r in summaries if r['scenario'] == scenario]
        summary['scenarios'][scenario] = dict(d1=stats([r['d1'] for r in selected]), d5=stats([r['d5'] for r in selected]),
            flags={key: sum(r[key] for r in selected) for key in ('d1_above_p95', 'd1_above_p99', 'd5_above_p95', 'd5_above_p99')},
            k5_coordinate_group_counts=dict(Counter(r['k5_coordinate_groups'] for r in selected)),
            fossil_eligible_reference_samples=selected[0]['eligible_reference_samples'],
            top5_geographic_distance_km=stats([r['geographic_distance_km'] for r in analogues if r['scenario'] == scenario]))
    write_json(output/'summary.json', summary)
    write_json(output/'manifest.json', dict(script_sha256=sha(Path(__file__)), parser_sha256=sha(ROOT/'tools/thesis_import/import_data.py'),
        shared_manifest_sha256=sha(shared/'manifest.json'), source_sha256=source_locks,
        policy_sha256=sha(policy_path), python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        outputs={p.name: dict(sha256=sha(p), byte_count=p.stat().st_size) for p in sorted(output.iterdir())},
        source_licenses=dict(modern='CC-BY-4.0', fossil_counts='CC-BY-NC-3.0', chronology='CC-BY-3.0')))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--raw-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    run(a.input_dir, a.raw_dir, a.output_dir)
