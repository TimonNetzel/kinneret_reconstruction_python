#!/usr/bin/env python3
"""Reproducible modern-only spatial evaluation of the separate Shared37 variant."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import platform
import tempfile

import numpy as np
import scipy
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def canonical_coordinates(latitude, longitude):
    lat = np.round(np.asarray(latitude, dtype=float), 6)
    lon = np.round(np.asarray(longitude, dtype=float), 6)
    require(np.isfinite(lat).all() and np.isfinite(lon).all(), 'Nonfinite coordinates')
    require(((lat >= -90) & (lat <= 90)).all() and ((lon >= -180) & (lon <= 180)).all(), 'Invalid coordinates')
    return lat, (lon+180) % 360-180


def sphere_xyz(latitude, longitude):
    lat, lon = np.deg2rad(latitude), np.deg2rad(longitude)
    return np.column_stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)))


def nearest_km(query_xyz, reference_xyz, radius):
    chord = cKDTree(reference_xyz).query(query_xyz, k=1, workers=1)[0]
    return 2*radius*np.arcsin(np.clip(chord/2, 0, 1))


def spatial_folds(latitude, longitude, degrees, folds, seed):
    lat, lon = canonical_coordinates(latitude, longitude)
    require(degrees > 0 and 180 % degrees == 0 and 360 % degrees == 0, 'Unsupported block size')
    row = np.minimum(np.floor((lat+90)/degrees).astype(int), int(180/degrees)-1)
    col = np.floor((lon+180)/degrees).astype(int)
    blocks = np.asarray([f'{r}:{c}' for r, c in zip(row, col)])
    unique = np.asarray(sorted(set(blocks)))
    require(len(unique) >= folds >= 2, 'Not enough spatial blocks')
    order = np.random.Generator(np.random.PCG64(seed)).permutation(len(unique))
    assignment = {unique[index]: position % folds for position, index in enumerate(order)}
    return blocks, np.asarray([assignment[b] for b in blocks])


def hellinger_features(counts):
    counts = np.asarray(counts, dtype=float)
    require(counts.ndim == 2 and np.isfinite(counts).all() and (counts >= 0).all(), 'Invalid pollen counts')
    total = counts.sum(axis=1)
    require((total > 0).all(), 'Empty pollen subcomposition')
    return np.sqrt(counts/total[:, None])


def analog_neighbors(train_x, test_x, k):
    """Training input must already be sorted by sample ID for deterministic ties."""
    require(0 < k <= len(train_x), 'Insufficient training analogues')
    indices = np.empty((len(test_x), k), dtype=int)
    distances = np.empty((len(test_x), k))
    for start in range(0, len(test_x), 128):
        d = cdist(test_x[start:start+128], train_x, metric='sqeuclidean')
        order = np.argsort(d, axis=1, kind='stable')[:, :k]
        indices[start:start+len(d)] = order
        distances[start:start+len(d)] = np.sqrt(np.take_along_axis(d, order, axis=1)/2)
    return indices, distances


def metric_values(observed, predicted, reference, weights):
    error = predicted-observed
    mse = np.average(error**2, weights=weights)
    reference_mse = np.average((reference-observed)**2, weights=weights)
    return dict(rmse=float(np.sqrt(mse)), mae=float(np.average(np.abs(error), weights=weights)),
                bias=float(np.average(error, weights=weights)),
                mse_skill=float(1-mse/reference_mse) if reference_mse > 0 else None)


def evaluate_metrics(observed, predicted, reference, groups, targets):
    counts = Counter(groups)
    weightings = {'sample': np.ones(len(groups)),
                  'equal_coordinate_group': np.asarray([1/counts[g] for g in groups])}
    return {target: {name: metric_values(observed[:, j], predicted[:, j], reference[:, j], weights)
                     for name, weights in weightings.items()} for j, target in enumerate(targets)}


def load_inputs(input_dir, policy_path, config):
    manifest_path = input_dir/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    require(digest(policy_path) == manifest['feature_policy_sha256'], 'Feature policy differs from prepared data')
    path = input_dir/'modern_shared_counts.csv'
    expected = manifest['outputs'][path.name]
    require(path.stat().st_size == expected['byte_count'] and digest(path) == expected['sha256'], 'Modern counts checksum mismatch')
    policy = json.loads(policy_path.read_text())
    require(policy['variant'] == config['variant'], 'Variant mismatch')
    names = [f['feature_id'] for f in policy['features']]
    require(len(names) == len(set(names)) == config['expected_features'], 'Feature contract changed')
    require(config['block_origin_lat_lon'] == [-90, -180], 'Unsupported grid origin')
    require(config['targets'] == ['tann_c', 'pann_mm_per_year'], 'Unsupported targets or target order')
    require(config['hyperparameter_tuning'] is False and config['fossil_predictions'] is False, 'Unsupported experiment scope')
    with path.open(encoding='utf-8', newline='') as stream:
        source = list(csv.DictReader(stream))
    require(len(source) == config['expected_input_rows'], 'Input cohort changed')
    require(len({r['sample_id'] for r in source}) == len(source), 'Duplicate sample ID')
    threshold = float(policy['minimum_shared_count_without_pinus'])
    for record in source:
        values = np.asarray([float(record[n]) for n in names])
        require(np.isfinite(values).all() and (values >= 0).all() and (values == np.floor(values)).all(), 'Missing, negative or non-integer counts')
        total = values.sum()
        require(total > 0 and abs(total-float(record['shared_count_sum'])) < 1e-8, 'Subcomposition sum mismatch')
        without = total-values[names.index('pinus')]
        require(abs(without-float(record['shared_count_without_pinus'])) < 1e-8, 'No-Pinus sum mismatch')
        require(int(record['count_threshold_pass']) == int(without >= threshold), 'Threshold flag mismatch')
    records = sorted((r for r in source if r['count_threshold_pass'] == '1'), key=lambda r: r['sample_id'])
    require(len(records) == config['expected_selected_rows'], 'Selected cohort changed')
    counts = np.asarray([[float(r[n]) for n in names] for r in records])
    targets = np.asarray([[float(r[n]) for n in config['targets']] for r in records])
    require(np.isfinite(targets).all() and (targets[:, 1] >= 0).all(), 'Invalid target climate')
    lat = np.asarray([float(r['latitude']) for r in records])
    lon = np.asarray([float(r['longitude']) for r in records])
    canonical_lat, canonical_lon = canonical_coordinates(lat, lon)
    groups = np.asarray([f'{a:.6f}|{b:.6f}' for a, b in zip(canonical_lat, canonical_lon)])
    # The original rounded source groups must never split either.
    source_groups = np.asarray([r['coordinate_group_1e6deg'] for r in records])
    return records, names, counts, targets, lat, lon, groups, source_groups, manifest_path, path


def evaluate(input_dir, policy_path, config_path, output_dir):
    require(not output_dir.exists(), 'Output directory already exists')
    config = json.loads(config_path.read_text())
    records, names, counts, y, lat, lon, groups, source_groups, input_manifest, input_path = load_inputs(input_dir, policy_path, config)
    x = hellinger_features(counts)
    xyz = sphere_xyz(lat, lon)
    neighbors = sorted(set([config['primary_neighbors']]+config['sensitivity_neighbors']))
    require(all(isinstance(k, int) and k > 0 for k in neighbors), 'Invalid neighbor counts')
    models = ['training_mean']+[f'analog_k{k}' for k in neighbors]
    summaries = {}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent) as temporary:
        stage = Path(temporary)/'run'
        stage.mkdir()
        with (stage/'oof_predictions.csv').open('w', encoding='utf-8', newline='') as predictions, \
             (stage/'fold_assignments.csv').open('w', encoding='utf-8', newline='') as assignments, \
             (stage/'training_roles.csv').open('w', encoding='utf-8', newline='') as roles:
            pred_fields = ['block_degrees', 'fold', 'sample_id', 'coordinate_group', 'model',
                           'observed_tann_c', 'predicted_tann_c', 'observed_pann_mm_per_year', 'predicted_pann_mm_per_year',
                           'nearest_training_km', 'nearest_analog_distance', 'kth_analog_distance']
            pw = csv.DictWriter(predictions, fieldnames=pred_fields, lineterminator='\n'); pw.writeheader()
            aw = csv.writer(assignments, lineterminator='\n'); aw.writerow(['block_degrees', 'sample_id', 'source_coordinate_group', 'canonical_coordinate_group', 'spatial_block', 'fold'])
            rw = csv.writer(roles, lineterminator='\n'); rw.writerow(['block_degrees', 'heldout_fold', 'sample_id', 'role'])
            for degrees in config['block_degrees']:
                blocks, folds = spatial_folds(lat, lon, degrees, config['folds'], config['seed'])
                for grouping in (groups, source_groups):
                    for group in set(grouping):
                        require(len(set(folds[grouping == group])) == 1, 'Coordinate group split across folds')
                for i, record in enumerate(records):
                    aw.writerow([degrees, record['sample_id'], source_groups[i], groups[i], blocks[i], folds[i]])
                result = {name: np.full_like(y, np.nan) for name in models}
                fold_reports = []
                for fold in range(config['folds']):
                    test = np.flatnonzero(folds == fold)
                    candidates = np.flatnonzero(folds != fold)
                    distance_to_test = nearest_km(xyz[candidates], xyz[test], config['earth_radius_km'])
                    train = candidates[distance_to_test > config['buffer_km']]
                    buffer = candidates[distance_to_test <= config['buffer_km']]
                    require(len(train) >= max(neighbors) and len(test) > 0, 'Fold has insufficient data')
                    require(not set(groups[train]) & set(groups[test]), 'Coordinate leakage')
                    require(not set(blocks[train]) & set(blocks[test]), 'Spatial block leakage')
                    nearest_train = nearest_km(xyz[test], xyz[train], config['earth_radius_km'])
                    require(nearest_train.min() > config['buffer_km']-1e-7, 'Spatial buffer violated')
                    indexes, distances = analog_neighbors(x[train], x[test], max(neighbors))
                    mean = np.mean(y[train], axis=0)
                    result['training_mean'][test] = mean
                    for k in neighbors:
                        result[f'analog_k{k}'][test] = y[train[indexes[:, :k]]].mean(axis=1)
                    for role, subset in [('test', test), ('train', train), ('buffer_excluded', buffer)]:
                        for i in subset: rw.writerow([degrees, fold, records[i]['sample_id'], role])
                    reference = result['training_mean'][test]
                    fold_metrics = {name: evaluate_metrics(y[test], result[name][test], reference, groups[test], config['targets']) for name in models}
                    fold_reports.append(dict(fold=fold, test_samples=len(test), train_candidates=len(candidates),
                        train_samples=len(train), buffer_excluded=len(buffer),
                        test_coordinate_groups=len(set(groups[test])), test_blocks=len(set(blocks[test])),
                        minimum_train_test_km=float(nearest_train.min()), metrics=fold_metrics))
                    for j, i in enumerate(test):
                        for name in models:
                            k = int(name.removeprefix('analog_k')) if name != 'training_mean' else None
                            pw.writerow(dict(block_degrees=degrees, fold=fold, sample_id=records[i]['sample_id'],
                                coordinate_group=groups[i], model=name, observed_tann_c=y[i, 0], predicted_tann_c=result[name][i, 0],
                                observed_pann_mm_per_year=y[i, 1], predicted_pann_mm_per_year=result[name][i, 1],
                                nearest_training_km=nearest_train[j],
                                nearest_analog_distance=distances[j, 0] if k else '',
                                kth_analog_distance=distances[j, k-1] if k else ''))
                    print(f'{degrees} degrees fold {fold}: {len(train)} train, {len(test)} test, {len(buffer)} buffered', flush=True)
                require(all(np.isfinite(v).all() for v in result.values()), 'Missing out-of-fold prediction')
                summaries[str(degrees)] = dict(blocks=len(set(blocks)), folds=fold_reports,
                    metrics={name: evaluate_metrics(y, result[name], result['training_mean'], groups, config['targets']) for name in models})
        summary = dict(experiment=config['experiment'], variant=config['variant'], input_samples=config['expected_input_rows'],
            selected_samples=len(records), threshold_excluded=config['expected_input_rows']-len(records),
            features=names, coordinate_groups=len(set(groups)), source_coordinate_groups=len(set(source_groups)),
            primary_model=f"analog_k{config['primary_neighbors']}", no_hyperparameter_selection=True,
            fossil_predictions=False, target_units=dict(zip(config['targets'], config['target_units'])), scenarios=summaries)
        write_json(stage/'metrics.json', summary)
        provenance = dict(config=config, config_sha256=digest(config_path), feature_policy_sha256=digest(policy_path),
            source_counts_sha256=digest(input_path), input_manifest_sha256=digest(input_manifest),
            evaluation_script_sha256=digest(Path(__file__)), python=platform.python_version(),
            numpy=np.__version__, scipy=scipy.__version__,
            outputs={p.name: dict(byte_count=p.stat().st_size, sha256=digest(p)) for p in sorted(stage.iterdir())},
            note='Modern holdout errors only; no palaeoclimate reconstruction, confidence interval, or model winner selection.')
        write_json(stage/'manifest.json', provenance)
        stage.rename(output_dir)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    docs = Path(__file__).resolve().parents[2]/'docs/thesis_research'
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--policy', type=Path, default=docs/'modern2025_2026_10_29/shared_features.json')
    parser.add_argument('--config', type=Path, default=docs/'baseline2025_2026_10_29/baseline_config.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.input_dir, args.policy, args.config, args.output_dir)
    print(json.dumps({k: result[k] for k in ('experiment', 'selected_samples', 'coordinate_groups', 'primary_model')}))


if __name__ == '__main__':
    main()
