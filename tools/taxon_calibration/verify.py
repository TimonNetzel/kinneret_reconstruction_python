"""Separate nested-calibration audit. No new producer/model imports.

Reuses hash-locked independent PR28 vector reader and PR29 audit arithmetic;
shared numerical/GIS decoders are explicitly recorded, not called independent.
"""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import zipfile

import numpy as np
import rasterio
from scipy.special import digamma
from scipy.spatial import ConvexHull
from sklearn.neighbors import BallTree


AUDIT29_SHA = '30473367db13694862d1d857a2a8edbcca386d78a79c307486f9982b7fdea160'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def table(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def split(lo, la, block, count, buffer, radius):
    assigned = (np.floor((la+90)/block).astype(int)+2*np.floor((lo+180)/block).astype(int)) % count
    coords = np.deg2rad(np.stack([la, lo], axis=1))
    for fold in range(count):
        held = assigned == fold
        assert held.any()
        distance = BallTree(coords[held], metric='haversine').query(coords, k=1)[0][:, 0]*radius
        train = ~held & (distance > buffer)
        assert train.any()
        yield fold, train, held, {'train_cells': int(train.sum()), 'test_cells': int(held.sum()),
              'buffered_cells': int((~train & ~held).sum()), 'minimum_train_test_km': float(distance[train].min())}


def verify(previous, maps, climate, config_path, output, report):
    config = json.loads(config_path.read_text())
    recipe = config['network_recipe']
    root = Path(__file__).resolve().parents[2]
    olddir = config_path.parent.parent/'smote_nnet_2026_10_30'
    for file, digest in config['locked_inputs_sha256'].items():
        assert sha(olddir/file) == digest
    audit29 = root/'tools/taxon_smote/verify.py'
    assert sha(audit29) == AUDIT29_SHA
    a29 = load_module('locked_pr29_audit_arithmetic', audit29)
    close, check = a29.close, a29.check_record
    sources = json.loads((olddir/'sources.json').read_text())
    audit_sources = json.loads((olddir/'audit_sources.json').read_text())
    for r in sources['previous_files']+[audit_sources['previous_raw_checker']]:
        assert Path(r['file']).name == r['file']
        assert (previous/r['file']).stat().st_size == r['bytes'] and sha(previous/r['file']) == r['sha256']
    raw = load_module('locked_pr28_independent_vector_reader', previous/'verify.py')
    old = json.loads((previous/'config.json').read_text())
    source28 = json.loads((previous/'sources.json').read_text())
    for r in source28['map_files']:
        assert sha(maps/r['path']) == r['sha256']
    assert sha(climate) == source28['climate_file']['sha256']
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['config_sha256'] == sha(config_path)
    assert manifest['design_sha256'] == sha(config_path.with_name('DESIGN.md'))
    for file, digest in manifest['outputs_sha256'].items():
        assert Path(file).name == file and sha(output/file) == digest
    for file, digest in manifest['code_sha256'].items():
        assert sha(root/file) == digest
    d, step = old['geographic_domain'], old['grid_degrees']
    xx, yy = np.meshgrid(np.arange(d['west']+step/2, d['east'], step), np.arange(d['north']-step/2, d['south'], -step))
    lon, lat = xx.ravel(), yy.ravel()
    arrays = []
    with zipfile.ZipFile(climate) as archive:
        for file in ['wc2.1_10m_bio_1.tif', 'wc2.1_10m_bio_12.tif']:
            blob = archive.read(file)
            assert hashlib.sha256(blob).hexdigest() == source28['worldclim_members'][file]['sha256']
            with rasterio.io.MemoryFile(blob) as memory:
                with memory.open() as ds:
                    v = ds.read(1)[np.floor((90-lat)*6).astype(int), np.floor((lon+180)*6).astype(int)].astype(float)
                    v[v == ds.nodata] = np.nan
                    arrays.append(v)
    all_x = np.stack(arrays, axis=1)
    land = np.isfinite(all_x).all(axis=1)
    fits_list = json.loads((output/'fit_models.json').read_text())
    fits = {(m['case_id'], m['method']): m for m in fits_list}
    final = json.loads((output/'models.json').read_text())
    roles_list = table(output/'split_inventory.csv')
    roles = {r['case']: r for r in roles_list}
    inventory = {r['taxon']: r for r in table(output/'sample_inventory.csv')}
    score_list = table(output/'scores.csv')
    scores = {(r['taxon'], int(r['block_degrees']), r['scope'], r['method']): r for r in score_list}
    rel_list = table(output/'reliability.csv')
    rel = {(r['taxon'], int(r['block_degrees']), r['method'], int(r['bin'])): r for r in rel_list}
    assert len(fits) == len(fits_list) == 504
    assert len(roles) == len(roles_list) == 252
    assert len(scores) == len(score_list) == 336 and len(rel) == len(rel_list) == 560
    assert table(output/'metrics_summary.csv') == [r for r in score_list if r['scope'] == 'all']
    counters = dict(label_cells=0, prediction_rows=0, synthetic_rows=0, network_starts=0,
                    calibrators_fitted=0, calibrators_identity=0)
    min_outer, min_inner, max_cal_grad = float('inf'), float('inf'), 0.
    visited = set()
    with gzip.open(output/'smote_trace.csv.gz', 'rt') as trace_f, (output/'predictions.csv').open() as pred_f:
        traces = iter(itertools.groupby(csv.DictReader(trace_f), key=lambda r: r['case']))
        predictions = iter(itertools.groupby(csv.DictReader(pred_f), key=lambda r: (r['taxon'], int(r['block_degrees']))))
        for taxon in old['taxa']:
            name = taxon['id']
            labels, use = raw.geometry_labels(maps/f"Data_in/Areale/geojson/taxon_id_{taxon['taxon_id']}.geojson", taxon, old, lon, lat, xx.shape)
            selected = np.flatnonzero(land & use)
            x, y, lo, la = all_x[selected], labels[selected], lon[selected], lat[selected]
            w = np.cos(np.deg2rad(la))
            assert int(inventory[name]['cells']) == len(y) and int(inventory[name]['positive_cells']) == y.sum()
            counters['label_cells'] += len(y)

            def audit_ensemble(block, outer, outer_train, evaluation):
                nonlocal min_inner, max_cal_grad
                pool = np.flatnonzero(outer_train)
                values = {m: [] for m in config['methods']}
                support = np.zeros(len(evaluation), int)
                members = {'original': [], 'smote': []}
                for inner, train_local, cal_local, info in split(lo[pool], la[pool], block, 3, 100, old['validation']['sphere_radius_km']):
                    train, cal = pool[train_local], pool[cal_local]
                    assert not np.intersect1d(train, cal).size
                    case = f'{name}:b{block}:o{outer}:i{inner}'
                    visited.add(case)
                    a, b, weights, ids = x[train], y[train], w[train], selected[train]
                    counts = np.bincount(b, minlength=2)
                    assert counts.min() >= config['minimum_training_class']
                    role = roles[case]
                    check(role, {**info, 'outer_train_cells': len(pool), 'calibration_cells': len(cal), 'calibration_positive': y[cal].sum()})
                    min_inner = min(min_inner, info['minimum_train_test_km'])
                    original, smote = fits[case, 'original'], fits[case, 'smote']
                    norm = original['normalization']
                    assert norm == smote['normalization'] and norm['fit_on_original_training_only']
                    close(norm['temperature_mean'], np.average(a[:, 0], weights=weights))
                    close(norm['temperature_sd'], np.sqrt(np.average((a[:, 0]-norm['temperature_mean'])**2, weights=weights)))
                    positive = a[:, 1] > 0
                    assert norm['positive_precipitation_training_cells'] == positive.sum()
                    v = np.sqrt(a[positive, 1])
                    mean = np.average(v, weights=weights[positive])
                    close(norm['gamma_rate'], norm['gamma_shape']/mean)
                    close(np.log(norm['gamma_shape'])-digamma(norm['gamma_shape']), np.log(mean)-np.average(np.log(v), weights=weights[positive]))
                    assert norm['cdf_clip'] == recipe['normalization']['cdf_clip']
                    z = a29.transform(a, norm)
                    trace_case, trace_rows = next(traces)
                    assert trace_case == case
                    trace_rows = list(trace_rows)
                    minority = np.flatnonzero(b == np.argmin(counts))
                    n = int(abs(counts[0]-counts[1]))
                    assert len(trace_rows) == n == int(role['synthetic_cells'])
                    lookup = {int(cell): i for i, cell in enumerate(ids)}
                    parent = np.array([lookup[int(r['base_cell'])] for r in trace_rows])
                    other = np.array([lookup[int(r['other_cell'])] for r in trace_rows])
                    fraction = np.array([float(r['fraction']) for r in trace_rows])
                    assert (parent != other).all() and (b[parent] == np.argmin(counts)).all() and (b[other] == b[parent]).all()
                    assert ((fraction >= 0) & (fraction < 1)).all()
                    seed = a29.seed(recipe['randomness']['master_seed'], case, 'smote')
                    rng = np.random.Generator(np.random.PCG64(seed))
                    np.testing.assert_array_equal(parent, minority[rng.integers(len(minority), size=n)])
                    rng.integers(recipe['smote']['k_neighbors'], size=n)
                    np.testing.assert_array_equal(fraction, rng.random(n))
                    kth = BallTree(z[minority], metric='euclidean').query(z[parent], k=recipe['smote']['k_neighbors']+1)[0][:, -1]
                    assert (np.linalg.norm(z[parent]-z[other], axis=1) <= kth+1e-10).all()
                    sx = np.r_[z, z[parent]+fraction[:, None]*(z[other]-z[parent])]
                    sy, sw = np.r_[b, b[parent]], np.r_[weights, weights[parent]+fraction*(weights[other]-weights[parent])]
                    counters['synthetic_rows'] += n
                    hull = ConvexHull(a)
                    support += (evaluation @ hull.equations[:, :-1].T+hull.equations[:, -1] <= 1e-8).all(axis=1)
                    for method, m, nx, ny, nw in [('original', original, z, b, weights), ('smote', smote, sx, sy, sw)]:
                        assert m['case_id'] == case and m['taxon'] == name and m['method'] == method
                        assert m['block_degrees'] == block and m['outer'] == str(outer) and m['inner'] == inner
                        assert m['original_training_cells'] == len(b) and m['original_training_positive'] == b.sum()
                        assert m['calibration_cells'] == len(cal) and m['resampled_case_count'] == len(ny)
                        assert m['synthetic_cells'] == (n if method == 'smote' else 0)
                        assert m['smote_seed'] == (seed if method == 'smote' else None)
                        close(m['original_area_weighted_prior'], np.average(b, weights=weights))
                        close(m['resampled_case_weighted_prior'], np.average(ny, weights=nw))
                        assert set(map(tuple, m['climate_hull'])) == set(map(tuple, a[hull.vertices]))
                        assert m['hidden_units'] == recipe['network']['hidden_units'] and m['weight_decay'] == recipe['network']['weight_decay']
                        assert len(m['starts']) == recipe['network']['initializations']
                        for j, start in enumerate(m['starts']):
                            assert start['seed'] == a29.seed(recipe['randomness']['master_seed'], case, f'initialization:{j}')
                            loss, grad = a29.calculation(np.array(start['weights']), nx, ny, nw, m['weight_decay'])
                            close(start['objective'], loss, atol=1e-8)
                            close(start['gradient_per_case'], grad, atol=1e-12)
                            assert start['stationary'] == (grad <= recipe['network']['accepted_gradient_per_effective_case'])
                            counters['network_starts'] += 1
                        valid = [j for j, s in enumerate(m['starts']) if s['stationary']]
                        chosen = min(valid, key=lambda j: (m['starts'][j]['objective'], j))
                        assert m['selected_start'] == chosen and m['weights'] == m['starts'][chosen]['weights']
                        eta, _ = a29.calculation(np.array(m['weights']), a29.transform(x[cal], norm))
                        c = m['calibrator']
                        cc = np.bincount(y[cal], minlength=2)
                        assert c['class_counts'] == cc.tolist()
                        if cc.min() < config['calibration']['minimum_class']:
                            assert c['status'] == 'insufficient_calibration_class' and c['slope'] == 1 and c['intercept'] == 0
                            assert c['objective'] is None and c['projected_gradient'] is None
                            counters['calibrators_identity'] += 1
                        else:
                            assert c['status'] == 'fitted' and c['slope'] >= 0
                            aa, bb = c['slope'], c['intercept']
                            zz = aa*eta+bb
                            ww = w[cal]/w[cal].sum()
                            residual = ww*(a29.sigmoid(zz)-y[cal])
                            ridge = config['calibration']['ridge_slope']
                            gradient = np.array([(residual*eta).sum()+ridge*aa, residual.sum()])
                            if aa == 0 and gradient[0] > 0:
                                gradient[0] = 0
                            pg = abs(gradient).max()
                            # Convex NLL plus positive slope penalty: KKT certifies global optimum.
                            assert pg <= config['calibration']['projected_gradient_tolerance']
                            close(c['projected_gradient'], pg, atol=1e-12)
                            objective = np.sum(ww*(np.maximum(zz, 0)+np.log1p(np.exp(-abs(zz)))-y[cal]*zz))+ridge*aa**2/2
                            close(c['objective'], objective)
                            max_cal_grad = max(max_cal_grad, float(pg))
                            counters['calibrators_fitted'] += 1
                        eta, p = a29.calculation(np.array(m['weights']), a29.transform(evaluation, norm))
                        values[method+'_raw'].append(p)
                        values[method+'_calibrated'].append(a29.sigmoid(c['slope']*eta+c['intercept']))
                        members[method].append(m)
                return {k: np.mean(v, axis=0) for k, v in values.items()}, support, members

            for block in old['validation']['block_sizes_degrees']:
                key, rows = next(predictions)
                assert key == (name, block)
                rows = list(rows)
                by_id = {int(r['cell_index']): r for r in rows}
                assert len(rows) == len(by_id) == len(y) and set(by_id) == set(selected)
                ordered = [by_id[int(i)] for i in selected]
                np.testing.assert_array_equal([int(r['label']) for r in ordered], y)
                probs = {m: np.array([float(r[m]) for r in ordered]) for m in config['methods']}
                base = np.array([float(r['comparator']) for r in ordered])
                saved_support = np.array([int(r['support_members']) for r in ordered])
                for outer, train, test, info in split(lo, la, block, 5, 100, old['validation']['sphere_radius_km']):
                    assert all(int(r['fold']) == outer for r in np.array(ordered, object)[test])
                    min_outer = min(min_outer, info['minimum_train_test_km'])
                    expected, support, _ = audit_ensemble(block, outer, train, x[test])
                    np.testing.assert_array_equal(saved_support[test], support)
                    close(base[test], np.average(y[train], weights=w[train]))
                    for method, p in expected.items():
                        close(probs[method][test], p, atol=2e-11)
                        check(scores[name, block, str(outer), method], {**a29.metric(y[test], probs[method][test], base[test], w[test]),
                              'cells_outside_any_member_hull': (support < 3).sum()})
                for method, p in probs.items():
                    check(scores[name, block, 'all', method], {**a29.metric(y, p, base, w),
                          'cells_outside_any_member_hull': (saved_support < 3).sum()})
                    bins = np.minimum((p*10).astype(int), 9)
                    for j in range(10):
                        choose = bins == j
                        check(rel[name, block, method, j], {'cells': choose.sum(), 'area_weight': w[choose].sum()/w.sum(),
                              'mean_prediction': np.average(p[choose], weights=w[choose]) if choose.any() else None,
                              'observed_fraction': np.average(y[choose], weights=w[choose]) if choose.any() else None})
                counters['prediction_rows'] += len(y)
                _, _, members = audit_ensemble(block, 'full', np.ones(len(y), bool), x[:1])
                assert final[name][str(block)] == members
            print('verified', name, len(y), flush=True)
        assert next(traces, None) is None and next(predictions, None) is None
    deltas = table(output/'paired_deltas.csv')
    assert len(deltas) == 168
    for r in deltas:
        raw_score = scores[r['taxon'], int(r['block_degrees']), r['scope'], r['family']+'_raw']
        cal_score = scores[r['taxon'], int(r['block_degrees']), r['scope'], r['family']+'_calibrated']
        for metric in ['brier', 'log_loss', 'balanced_accuracy']:
            key = metric+'_delta_cal_minus_raw'
            if cal_score[metric] == '' or raw_score[metric] == '':
                assert r[key] == ''
            else:
                close(float(r[key]), float(cal_score[metric])-float(raw_score[metric]))
    assert counters['synthetic_rows'] == manifest['synthetic_rows']
    assert counters['network_starts'] == 1512 and len(visited) == 252
    result = {'result': 'passed', **counters, 'network_fits': len(fits), 'individual_outer_probabilities': 4*counters['prediction_rows'],
              'minimum_outer_haversine_km': min_outer, 'minimum_inner_haversine_km': min_inner,
              'maximum_calibration_projected_gradient': max_cal_grad, 'score_rows': 336, 'reliability_rows': 560, 'paired_deltas': 168,
              'producer_or_new_calibrator_imported': False, 'audit_code_sha256': sha(Path(__file__)),
              'old_audit_arithmetic_sha256': AUDIT29_SHA, 'old_raw_checker_sha256': sha(previous/'verify.py'),
              'shared_dependencies': ['hash-locked PR28 independent vector reader', 'hash-locked PR29 audit arithmetic',
                                      'NumPy/SciPy', 'rasterio/GDAL', 'shapely/GEOS', 'scikit-learn BallTree'],
              'independent_paths': ['vector predicates rather than producer rasterization', 'direct TIFF indices',
                                   'Haversine/Euclidean BallTree rather than cKDTree', 'scipy.stats CDF/PPF rather than special functions',
                                   'tanh forward/backpropagation', 'convex calibration objective/KKT', 'all scores/deltas/reliability'],
              'code_source_output_hashes_checked': True, 'regions_previously_examined': True,
              'scientific_map_truth_established': False, 'fossil_application_approved': False, 'p1_released': False}
    report.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['previous', 'maps', 'climate', 'config', 'output', 'report']:
        parser.add_argument('--'+key, required=True, type=Path)
    a = parser.parse_args()
    verify(a.previous, a.maps, a.climate, a.config, a.output, a.report)
