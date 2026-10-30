"""Train-only SMOTE/network comparison on the exact, locked PR28 source/split recipe."""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy.spatial import ConvexHull

from network import derived_seed, fit_network, make_smote, normalize_fit, normalize, predict, require


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_out(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def load_previous(folder, sources):
    for record in sources['previous_files']:
        require(Path(record['file']).name == record['file'], 'Unsafe input filename')
        path = folder/record['file']
        require(path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], 'PR28 input changed: '+record['file'])
    def module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    pm = module('locked_pr28_model', folder/'model.py')
    existing = sys.modules.get('model')
    sys.modules['model'] = pm
    try:
        pc = module('locked_pr28_calibrate', folder/'calibrate.py')
    finally:
        if existing is None:
            del sys.modules['model']
        else:
            sys.modules['model'] = existing
    return pm, pc, json.loads((folder/'config.json').read_text()), json.loads((folder/'sources.json').read_text())


def scores(y, probability, base, weights):
    positive, negative = y == 1, y == 0
    guessed = probability >= .5
    sensitivity = float(np.mean(guessed[positive])) if positive.any() else None
    specificity = float(np.mean(~guessed[negative])) if negative.any() else None
    ws = float(np.average(guessed[positive], weights=weights[positive])) if positive.any() else None
    wn = float(np.average(~guessed[negative], weights=weights[negative])) if negative.any() else None
    normalized = weights/weights.sum()
    brier = float(normalized @ ((probability-y)**2))
    brier_base = float(normalized @ ((base-y)**2))
    safe = np.clip(probability, 1e-12, 1-1e-12)
    return {'cells': len(y), 'positive_cells': int(positive.sum()), 'sensitivity': sensitivity, 'specificity': specificity,
            'balanced_accuracy': (sensitivity+specificity)/2 if sensitivity is not None and specificity is not None else None,
            'area_weighted_balanced_accuracy': (ws+wn)/2 if ws is not None and wn is not None else None,
            'brier': brier, 'comparator_brier': brier_base, 'brier_skill': 1-brier/brier_base if brier_base else None,
            'log_loss': float(-normalized @ (y*np.log(safe)+(1-y)*np.log1p(-safe))),
            'mean_prediction': float(normalized @ probability), 'observed_prevalence': float(normalized @ y)}


def run(previous, maps, climate, config_path, sources_path, output):
    require(not output.exists(), 'Use a new output directory')
    config, sources = json.loads(config_path.read_text()), json.loads(sources_path.read_text())
    pm, pc, old_config, old_sources = load_previous(previous, sources)
    pc.source_gate(maps, climate, old_sources)
    xx, yy, raw_climate, land, transform = pc.climate_grid(climate, old_config, old_sources)
    baseline_folds = {(m['taxon'], m['block_degrees'], m['fold']): m for m in json.loads((previous/'fold_models.json').read_text())}
    old_scores = {(r['taxon'], int(r['block_degrees']), r['scope']): r for r in pc.table(previous/'scores.csv')} if hasattr(pc, 'table') else None
    if old_scores is None:
        with (previous/'scores.csv').open() as handle:
            old_scores = {(r['taxon'], int(r['block_degrees']), r['scope']): r for r in csv.DictReader(handle)}
    output.mkdir(parents=True)
    fitted, final, all_scores, roles, inventory, reliability = [], {}, [], [], [], []
    method_ids = config['comparison_models']
    prediction_fields = ['taxon', 'block_degrees', 'cell_index', 'fold', 'label', 'comparator', 'in_training_climate_hull']+method_ids
    synthetic_total = 0
    with (output/'predictions.csv').open('w', newline='') as pred_handle, (output/'smote_trace.csv.gz').open('wb') as trace_raw:
        pred_writer = csv.DictWriter(pred_handle, fieldnames=prediction_fields, lineterminator='\n')
        pred_writer.writeheader()
        with gzip.GzipFile(filename='', mode='wb', fileobj=trace_raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding='utf-8', newline='') as trace_handle:
                trace_writer = csv.writer(trace_handle, lineterminator='\n')
                trace_writer.writerow(['case', 'base_cell', 'other_cell', 'fraction'])
                for taxon in old_config['taxa']:
                    labels, use, evidence = pc.range_labels(maps/f"Data_in/Areale/geojson/taxon_id_{taxon['taxon_id']}.geojson", taxon, old_config, xx.shape, transform)
                    selected = np.flatnonzero(land & use)
                    x, y = raw_climate[selected], labels[selected]
                    lon, lat = xx.ravel()[selected], yy.ravel()[selected]
                    area = np.cos(np.deg2rad(lat))
                    inventory.append({'taxon': taxon['id'], 'cells': len(y), 'positive_cells': int(y.sum()), 'masked_source_evidence': json.dumps(evidence, sort_keys=True)})

                    def train_case(case, train):
                        nonlocal synthetic_total
                        original_x, original_y, w = x[train], y[train], area[train]
                        require(min(np.sum(original_y == 0), np.sum(original_y == 1)) >= old_config['model']['minimum_training_cells_per_class'], 'Original training class gate')
                        norm = normalize_fit(original_x, w, config['normalization']['cdf_clip'])
                        z = normalize(original_x, norm)
                        prior = float(np.average(original_y, weights=w))
                        models = {}
                        seed = derived_seed(config['randomness']['master_seed'], case, 'smote')
                        sx, sy, sw, trace = make_smote(z, original_y, w, config['smote']['k_neighbors'], seed)
                        cells = selected[train]
                        for base, other, fraction in zip(cells[trace['base']], cells[trace['other']], trace['fraction']):
                            trace_writer.writerow([case, int(base), int(other), float(fraction)])
                        synthetic_total += trace['synthetic_count']
                        for name, nx, ny, nw in [('nnet_original', z, original_y, w), ('nnet_smote', sx, sy, sw)]:
                            model = fit_network(nx, ny, nw, config['network'], config['randomness']['master_seed'], case)
                            model.update(case_id=case, method=name, taxon=taxon, normalization=norm,
                                original_training_cells=len(original_y), original_training_positive_cells=int(original_y.sum()),
                                original_area_weighted_prior=prior, climate_hull=original_x[ConvexHull(original_x).vertices].tolist(),
                                synthetic_cells=trace['synthetic_count'] if name == 'nnet_smote' else 0,
                                smote_seed=seed if name == 'nnet_smote' else None,
                                smote_minority_label=trace['label'] if name == 'nnet_smote' else None,
                                variant=config['variant'], target='native_mapped_range_under_declared_sampling',
                                original_training_prior_preserved_in_output=name == 'nnet_original',
                                prior_adjustment_is_only_diagnostic=True)
                            models[name] = model
                            fitted.append(model)
                            json_out(output/'fit_models.json', fitted)
                            print('fit', case, name, 'synthetic', model['synthetic_cells'], 'stationary', sum(s['stationary'] for s in model['starts']), flush=True)
                        return models, trace

                    for block in old_config['validation']['block_sizes_degrees']:
                        collected = {name: np.empty(len(y)) for name in method_ids}
                        base_all = np.empty(len(y))
                        support_all = np.empty(len(y), bool)
                        for fold, train, test, evidence_role in pc.splits(lon, lat, block, old_config['validation']):
                            case = f"{taxon['id']}:b{block}:f{fold}"
                            current, trace = train_case(case, train)
                            prior = float(np.average(y[train], weights=area[train]))
                            base = np.full(test.sum(), prior)
                            baseline, inside = pm.probabilities(baseline_folds[taxon['id'], block, fold], x[test], reject_extrapolation=False)
                            estimates = {'pr28_quadratic': baseline}
                            estimates['nnet_original'], nn_inside = predict(current['nnet_original'], x[test], reject_extrapolation=False)
                            require(np.array_equal(inside, nn_inside), 'Original training climate hull changed')
                            estimates['nnet_smote'], _ = predict(current['nnet_smote'], x[test], reject_extrapolation=False)
                            estimates['nnet_smote_prior_adjusted'], _ = predict(current['nnet_smote'], x[test], prior_adjusted=True, reject_extrapolation=False)
                            base_all[test], support_all[test] = base, inside
                            for name, values in estimates.items():
                                collected[name][test] = values
                                result = {'taxon': taxon['id'], 'block_degrees': block, 'scope': str(fold), 'method': name, **scores(y[test], values, base, area[test])}
                                all_scores.append(result)
                                if name == 'pr28_quadratic':
                                    require(abs(result['brier']-float(old_scores[taxon['id'], block, str(fold)]['brier'])) <= 1e-12, 'PR28 comparator drift')
                            roles.append({'case': case, 'taxon': taxon['id'], 'block_degrees': block, 'fold': fold,
                                **evidence_role, 'synthetic_training_cells': trace['synthetic_count'], 'original_training_prior': prior})
                            for k, cell in enumerate(selected[test]):
                                values = [taxon['id'], block, int(cell), fold, int(y[test][k]), prior, bool(inside[k])]
                                pred_writer.writerow(dict(zip(prediction_fields, values+[float(estimates[name][k]) for name in method_ids])))
                        for name, values in collected.items():
                            result = {'taxon': taxon['id'], 'block_degrees': block, 'scope': 'all', 'method': name, **scores(y, values, base_all, area)}
                            all_scores.append(result)
                            if name == 'pr28_quadratic':
                                require(abs(result['brier']-float(old_scores[taxon['id'], block, 'all']['brier'])) <= 1e-12, 'PR28 aggregate drift')
                            bins = np.minimum((values*10).astype(int), 9)
                            for j in range(10):
                                choose = bins == j
                                reliability.append({'taxon': taxon['id'], 'block_degrees': block, 'method': name, 'bin': j,
                                    'cells': int(choose.sum()), 'area_weight': float(area[choose].sum()/area.sum()),
                                    'mean_prediction': float(np.average(values[choose], weights=area[choose])) if choose.any() else None,
                                    'observed_fraction': float(np.average(y[choose], weights=area[choose])) if choose.any() else None})
                    current, _ = train_case(taxon['id']+':full', np.ones(len(y), bool))
                    final[taxon['id']] = current
                    json_out(output/'models.json', final)
    pc.write_csv(output/'scores.csv', all_scores)
    pc.write_csv(output/'split_inventory.csv', roles)
    pc.write_csv(output/'sample_inventory.csv', inventory)
    pc.write_csv(output/'reliability.csv', reliability)
    pc.write_csv(output/'metrics_summary.csv', [r for r in all_scores if r['scope'] == 'all'])
    json_out(output/'manifest.json', {'variant': config['variant'], 'config_sha256': sha(config_path), 'sources_sha256': sha(sources_path),
        'previous_commit': sources['previous_commit'], 'synthetic_rows': synthetic_total,
        'code_sha256': {p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name('network.py')]},
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
        'outputs_sha256': {p.name: sha(p) for p in sorted(output.iterdir())}, 'fossil_application_approved': False, 'p1_released': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('previous', 'maps', 'climate', 'config', 'sources', 'output'):
        parser.add_argument('--'+key, required=True, type=Path)
    a = parser.parse_args()
    run(a.previous, a.maps, a.climate, a.config, a.sources, a.output)
