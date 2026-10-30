"""Nested spatial member calibration on locked Meusel7 inputs, no outer tuning."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import io
import json
import platform
import multiprocessing
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.spatial import ConvexHull

from model import fit_calibrator, logits, predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'taxon_smote'))
from compare import load_previous, sha, json_out, scores
from network import derived_seed, fit_network, make_smote, normalize, normalize_fit, require


def fit_member(args, metadata, cal_x, cal_y, cal_w, settings):
    m = fit_network(*args)
    m.update(metadata)
    m['calibrator'] = fit_calibrator(logits(m, cal_x), cal_y, cal_w, settings)
    return m


def run(previous, maps, climate, config_path, output):
    require(not output.exists(), 'Use a new output directory')
    config = json.loads(config_path.read_text())
    olddir = config_path.parent.parent/'smote_nnet_2026_10_30'
    for name, digest in config['locked_inputs_sha256'].items():
        require(sha(olddir/name) == digest, 'Locked SMOTE input drift: '+name)
    recipe = config['network_recipe']
    require(recipe == json.loads((olddir/'config.json').read_text()), 'Network recipe drift')
    sources = json.loads((olddir/'sources.json').read_text())
    _, pc, old, source28 = load_previous(previous, sources)
    pc.source_gate(maps, climate, source28)
    xx, yy, all_x, land, transform = pc.climate_grid(climate, old, source28)
    output.mkdir(parents=True)
    fits, final, score_rows, reliability, roles, inventory = [], {}, [], [], [], []
    synthetic_total = 0
    fields = ['taxon', 'block_degrees', 'cell_index', 'fold', 'label', 'comparator', 'support_members']+config['methods']
    with ProcessPoolExecutor(max_workers=6, mp_context=multiprocessing.get_context('spawn')) as workers, \
            (output/'predictions.csv').open('w', newline='') as ph, (output/'smote_trace.csv.gz').open('wb') as th:
        writer = csv.DictWriter(ph, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        with gzip.GzipFile(filename='', mode='wb', fileobj=th, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding='utf-8', newline='') as text:
                tracewriter = csv.writer(text, lineterminator='\n')
                tracewriter.writerow(['case', 'base_cell', 'other_cell', 'fraction'])
                for taxon in old['taxa']:
                    labels, use, evidence = pc.range_labels(maps/f"Data_in/Areale/geojson/taxon_id_{taxon['taxon_id']}.geojson", taxon, old, xx.shape, transform)
                    cells = np.flatnonzero(land & use)
                    x, y = all_x[cells], labels[cells]
                    lon, lat = xx.ravel()[cells], yy.ravel()[cells]
                    w = np.cos(np.deg2rad(lat))
                    inventory.append({'taxon': taxon['id'], 'cells': len(y), 'positive_cells': int(y.sum()),
                                      'evidence': json.dumps(evidence, sort_keys=True)})
                    final[taxon['id']] = {}

                    def train_ensemble(block, outer, outer_train):
                        nonlocal synthetic_total
                        pool = np.flatnonzero(outer_train)
                        settings = {**old['validation'], 'folds': config['inner_folds'], 'buffer_km': config['inner_buffer_km']}
                        members = {'original': [], 'smote': []}
                        pending = []
                        for inner, train_local, cal_local, info in pc.splits(lon[pool], lat[pool], block, settings):
                            train, cal = pool[train_local], pool[cal_local]
                            case = f"{taxon['id']}:b{block}:o{outer}:i{inner}"
                            require(np.bincount(y[train], minlength=2).min() >= config['minimum_training_class'], 'Inner original training class gate: '+case)
                            norm = normalize_fit(x[train], w[train], recipe['normalization']['cdf_clip'])
                            z = normalize(x[train], norm)
                            seed = derived_seed(recipe['randomness']['master_seed'], case, 'smote')
                            sx, sy, sw, trace = make_smote(z, y[train], w[train], recipe['smote']['k_neighbors'], seed)
                            for base, other, fraction in zip(cells[train][trace['base']], cells[train][trace['other']], trace['fraction']):
                                tracewriter.writerow([case, int(base), int(other), float(fraction)])
                            synthetic_total += trace['synthetic_count']
                            role = {'case': case, 'taxon': taxon['id'], 'block_degrees': block, 'outer': str(outer), 'inner': inner,
                                    'outer_train_cells': len(pool), 'calibration_cells': len(cal), 'calibration_positive': int(y[cal].sum()),
                                    'synthetic_cells': trace['synthetic_count'], **info}
                            roles.append(role)
                            for method, nx, ny, nw in [('original', z, y[train], w[train]), ('smote', sx, sy, sw)]:
                                metadata = dict(case_id=case, method=method, taxon=taxon['id'], block_degrees=block, outer=str(outer), inner=inner,
                                         normalization=norm, climate_hull=x[train][ConvexHull(x[train]).vertices].tolist(),
                                         original_training_cells=len(train), original_training_positive=int(y[train].sum()),
                                         original_area_weighted_prior=float(np.average(y[train], weights=w[train])),
                                         calibration_cells=len(cal), synthetic_cells=trace['synthetic_count'] if method == 'smote' else 0,
                                         smote_seed=seed if method == 'smote' else None)
                                args = (nx, ny, nw, recipe['network'], recipe['randomness']['master_seed'], case)
                                pending.append((method, workers.submit(fit_member, args, metadata, x[cal], y[cal], w[cal], config['calibration'])))
                        # Consume in declared case/method order, regardless of completion order.
                        for method, future in pending:
                            m = future.result()
                            members[method].append(m)
                            fits.append(m)
                            json_out(output/'fit_models.json', fits)
                            print('fit', m['case_id'], method, m['calibrator']['status'], flush=True)
                        return members

                    for block in old['validation']['block_sizes_degrees']:
                        collected = {method: np.empty(len(y)) for method in config['methods']}
                        bases = np.empty(len(y))
                        supports = np.empty(len(y), int)
                        for outer, train, test, info in pc.splits(lon, lat, block, old['validation']):
                            members = train_ensemble(block, outer, train)
                            prior = float(np.average(y[train], weights=w[train]))
                            bases[test] = prior
                            for method in config['methods']:
                                family, kind = method.split('_')
                                p, support = predict(members[family], x[test], calibrated=kind == 'calibrated', reject_extrapolation=False)
                                collected[method][test] = p
                                supports[test] = support
                                score_rows.append({'taxon': taxon['id'], 'block_degrees': block, 'scope': str(outer), 'method': method,
                                                   **scores(y[test], p, np.full(test.sum(), prior), w[test]),
                                                   'cells_outside_any_member_hull': int((support < 3).sum())})
                            for k in np.flatnonzero(test):
                                writer.writerow(dict(zip(fields, [taxon['id'], block, int(cells[k]), outer, int(y[k]), prior, int(supports[k])]+
                                                         [float(collected[m][k]) for m in config['methods']])))
                        for method, p in collected.items():
                            score_rows.append({'taxon': taxon['id'], 'block_degrees': block, 'scope': 'all', 'method': method,
                                               **scores(y, p, bases, w), 'cells_outside_any_member_hull': int((supports < 3).sum())})
                            bins = np.minimum((p*10).astype(int), 9)
                            for j in range(10):
                                mask = bins == j
                                reliability.append({'taxon': taxon['id'], 'block_degrees': block, 'method': method, 'bin': j,
                                                    'cells': int(mask.sum()), 'area_weight': float(w[mask].sum()/w.sum()),
                                                    'mean_prediction': float(np.average(p[mask], weights=w[mask])) if mask.any() else None,
                                                    'observed_fraction': float(np.average(y[mask], weights=w[mask])) if mask.any() else None})
                        final[taxon['id']][str(block)] = train_ensemble(block, 'full', np.ones(len(y), bool))
                        json_out(output/'models.json', final)
    for name, rows in [('scores.csv', score_rows), ('metrics_summary.csv', [r for r in score_rows if r['scope'] == 'all']),
                       ('reliability.csv', reliability), ('split_inventory.csv', roles), ('sample_inventory.csv', inventory)]:
        pc.write_csv(output/name, rows)
    paired = []
    for taxon in old['taxa']:
        for block in old['validation']['block_sizes_degrees']:
            for scope in [str(j) for j in range(5)]+['all']:
                subset = {r['method']: r for r in score_rows if r['taxon'] == taxon['id'] and r['block_degrees'] == block and r['scope'] == scope}
                for family in ['original', 'smote']:
                    raw, cal = subset[family+'_raw'], subset[family+'_calibrated']
                    paired.append({'taxon': taxon['id'], 'block_degrees': block, 'scope': scope, 'family': family,
                                   **{key+'_delta_cal_minus_raw': cal[key]-raw[key] if cal[key] is not None and raw[key] is not None else None
                                      for key in ['brier', 'log_loss', 'balanced_accuracy']}})
    pc.write_csv(output/'paired_deltas.csv', paired)
    code = [Path(__file__), Path(__file__).with_name('model.py'), Path(__file__).parents[1]/'taxon_smote/network.py',
            Path(__file__).parents[1]/'taxon_smote/compare.py']
    json_out(output/'manifest.json', {'variant': config['variant'], 'config_sha256': sha(config_path),
             'design_sha256': sha(config_path.with_name('DESIGN.md')), 'design_commit': 'afa0c8a8845a5f40daee4289cc429fdf4a6ecf6a',
             'locked_inputs_sha256': config['locked_inputs_sha256'], 'synthetic_rows': synthetic_total, 'network_fits': len(fits),
             'code_sha256': {str(p.resolve().relative_to(Path(__file__).resolve().parents[2])): sha(p) for p in code},
             'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
             'outputs_sha256': {p.name: sha(p) for p in sorted(output.iterdir())},
             'regions_previously_examined': True, 'outer_test_tuning': False, 'fossil_application_approved': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['previous', 'maps', 'climate', 'config', 'output']:
        parser.add_argument('--'+key, required=True, type=Path)
    a = parser.parse_args()
    run(a.previous, a.maps, a.climate, a.config, a.output)
