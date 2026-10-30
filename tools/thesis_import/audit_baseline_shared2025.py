#!/usr/bin/env python3
"""Independent CSV audit: haversine buffers, fold roles, metrics and analogue checks.

Does not import the evaluator or use its distance/tree implementation.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(input_dir, run_dir, policy_path):
    manifest = json.loads((run_dir/'manifest.json').read_text())
    config = manifest['config']
    for name, expected in manifest['outputs'].items():
        path = run_dir/name
        assert path.stat().st_size == expected['byte_count'] and sha(path) == expected['sha256'], name
    assert sha(policy_path) == manifest['feature_policy_sha256']
    source_path = input_dir/'modern_shared_counts.csv'
    assert sha(source_path) == manifest['source_counts_sha256']
    selected = sorted((r for r in rows(source_path) if r['count_threshold_pass'] == '1'), key=lambda r: r['sample_id'])
    source = {r['sample_id']: r for r in selected}
    names = [f['feature_id'] for f in json.loads(policy_path.read_text())['features']]
    assignment = defaultdict(dict)
    for record in rows(run_dir/'fold_assignments.csv'):
        size = record['block_degrees']; key = record['sample_id']
        assert key not in assignment[size]
        assignment[size][key] = record
    roles = defaultdict(dict)
    for record in rows(run_dir/'training_roles.csv'):
        key = (record['block_degrees'], record['heldout_fold'])
        assert record['sample_id'] not in roles[key]
        roles[key][record['sample_id']] = record['role']
    predictions = defaultdict(dict)
    for record in rows(run_dir/'oof_predictions.csv'):
        key = (record['block_degrees'], record['model'])
        assert record['sample_id'] not in predictions[key]
        predictions[key][record['sample_id']] = record
    report = {'audit_implementation': 'Independent NumPy haversine / direct squared-distance / CSV arithmetic',
              'source_samples': len(source), 'folds_checked': 0, 'analogue_prediction_spotchecks': 0,
              'minimum_train_test_km': None, 'coordinate_group_split': False,
              'all_output_hashes_verified': True, 'all_prediction_metrics_verified': True}
    region = {k for k, r in source.items() if 25 <= float(r['latitude']) <= 75 and -15 <= float(r['longitude']) <= 45}
    report['posthoc_geographic_diagnostic'] = dict(
        note='Descriptive subset of the same OOF predictions, not an independent validation or a new fitted model; defined after the primary results.',
        latitude_bounds=[25, 75], longitude_bounds=[-15, 45], samples=len(region), metrics={})
    metrics = json.loads((run_dir/'metrics.json').read_text())
    for size, assigned in assignment.items():
        assert set(assigned) == set(source)
        for field in ['source_coordinate_group', 'canonical_coordinate_group', 'spatial_block']:
            seen = defaultdict(set)
            for record in assigned.values(): seen[record[field]].add(record['fold'])
            assert all(len(folds) == 1 for folds in seen.values())
        for fold in range(config['folds']):
            role = roles[(size, str(fold))]
            assert set(role) == set(source)
            tests = sorted(k for k in source if assigned[k]['fold'] == str(fold))
            train = sorted(k for k in source if role[k] == 'train')
            candidates = sorted(set(source)-set(tests))
            assert {k for k in role if role[k] == 'test'} == set(tests)
            test_lat = np.deg2rad([float(source[k]['latitude']) for k in tests])
            test_lon = np.deg2rad([float(source[k]['longitude']) for k in tests])
            closest_train = np.full(len(tests), np.inf)
            for start in range(0, len(candidates), 128):
                subset = candidates[start:start+128]
                lat = np.deg2rad([float(source[k]['latitude']) for k in subset])[:, None]
                lon = np.deg2rad([float(source[k]['longitude']) for k in subset])[:, None]
                a = np.sin((lat-test_lat)/2)**2+np.cos(lat)*np.cos(test_lat)*np.sin((lon-test_lon)/2)**2
                distance = 2*config['earth_radius_km']*np.arcsin(np.sqrt(np.clip(a, 0, 1)))
                minimum = distance.min(axis=1)
                mask = np.asarray([role[k] == 'train' for k in subset])
                for j, key in enumerate(subset):
                    expected = 'train' if minimum[j] > config['buffer_km'] else 'buffer_excluded'
                    assert role[key] == expected, (size, fold, key, minimum[j])
                if mask.any(): closest_train = np.minimum(closest_train, distance[mask].min(axis=0))
            assert closest_train.min() > config['buffer_km']-1e-7
            smallest = float(closest_train.min())
            report['minimum_train_test_km'] = min(report['minimum_train_test_km'] or smallest, smallest)
            training_targets = np.asarray([[float(source[k][t]) for t in config['targets']] for k in train])
            average = training_targets.mean(axis=0)
            for j, key in enumerate(tests):
                predicted = predictions[(size, 'training_mean')][key]
                assert predicted['fold'] == str(fold)
                np.testing.assert_allclose([float(predicted['predicted_tann_c']), float(predicted['predicted_pann_mm_per_year'])], average, atol=1e-9, rtol=0)
                assert abs(float(predicted['nearest_training_km'])-closest_train[j]) < 1e-6
            train_counts = np.asarray([[float(source[k][n]) for n in names] for k in train])
            train_root = np.sqrt(train_counts/train_counts.sum(axis=1)[:, None])
            for position in sorted(set([0, len(tests)//2, len(tests)-1])):
                key = tests[position]
                count = np.asarray([float(source[key][n]) for n in names])
                query = np.sqrt(count/count.sum())
                squared = np.sum((train_root-query)**2, axis=1)
                order = np.argsort(squared, kind='stable')
                for k in [config['primary_neighbors']]+config['sensitivity_neighbors']:
                    expected = training_targets[order[:k]].mean(axis=0)
                    predicted = predictions[(size, f'analog_k{k}')][key]
                    np.testing.assert_allclose([float(predicted['predicted_tann_c']), float(predicted['predicted_pann_mm_per_year'])], expected, atol=1e-9, rtol=0)
                    assert abs(float(predicted['kth_analog_distance'])-np.sqrt(squared[order[k-1]]/2)) < 1e-12
                    report['analogue_prediction_spotchecks'] += 1
            report['folds_checked'] += 1
        keys = sorted(source)
        groups = [assigned[k]['canonical_coordinate_group'] for k in keys]
        group_counts = Counter(groups)
        region_mask = np.asarray([k in region for k in keys])
        region_groups = [g for g, include in zip(groups, region_mask) if include]
        region_counts = Counter(region_groups)
        regional_models = {}
        for model, target_metrics in metrics['scenarios'][size]['metrics'].items():
            regional_models[model] = {}
            model_rows = predictions[(size, model)]
            assert set(model_rows) == set(source)
            for j, target in enumerate(config['targets']):
                observed_field = 'observed_tann_c' if j == 0 else 'observed_pann_mm_per_year'
                predicted_field = 'predicted_tann_c' if j == 0 else 'predicted_pann_mm_per_year'
                observed = np.asarray([float(source[k][target]) for k in keys])
                np.testing.assert_allclose([float(model_rows[k][observed_field]) for k in keys], observed, atol=0, rtol=0)
                predicted = np.asarray([float(model_rows[k][predicted_field]) for k in keys])
                reference = np.asarray([float(predictions[(size, 'training_mean')][k][predicted_field]) for k in keys])
                error = predicted-observed
                for weighting, weights in [('sample', np.ones(len(keys))),
                    ('equal_coordinate_group', np.asarray([1/group_counts[g] for g in groups]))]:
                    w = weights/weights.sum()
                    mse = np.dot(w, error**2)
                    expected = dict(rmse=np.sqrt(mse), mae=np.dot(w, np.abs(error)), bias=np.dot(w, error),
                                    mse_skill=1-mse/np.dot(w, (reference-observed)**2))
                    for name, value in expected.items():
                        assert abs(value-target_metrics[target][weighting][name]) < 1e-8
                regional_models[model][target] = {}
                for weighting, weights in [('sample', np.ones(len(region_groups))),
                    ('equal_coordinate_group', np.asarray([1/region_counts[g] for g in region_groups]))]:
                    w = weights/weights.sum(); e = error[region_mask]
                    mse = np.dot(w, e**2)
                    regional_models[model][target][weighting] = dict(rmse=float(np.sqrt(mse)),
                        mae=float(np.dot(w, np.abs(e))), bias=float(np.dot(w, e)),
                        mse_skill=float(1-mse/np.dot(w, (reference[region_mask]-observed[region_mask])**2)))
        report['posthoc_geographic_diagnostic']['metrics'][size] = regional_models
    report['prediction_rows_checked'] = sum(len(r) for r in predictions.values())
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input_dir, args.run_dir, args.policy)
    args.report.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
