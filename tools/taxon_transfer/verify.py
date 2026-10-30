"""Separate source, Haversine, gradient and score audit. No producer import.

Shared decoders: rasterio/GDAL and shapely/GEOS. Raster membership is checked
with vector predicates instead of rasterize; BallTree instead of cKDTree.
"""
import argparse
import collections
import csv
import hashlib
import itertools
import json
from pathlib import Path
import zipfile

import numpy as np
import rasterio
from scipy.spatial import ConvexHull
from shapely.geometry import shape
import shapely
from sklearn.neighbors import BallTree


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def geometry_labels(path, taxon, config, lon, lat, grid_shape):
    doc = json.loads(path.read_text())
    y, point_cells = np.zeros(len(lon), bool), np.zeros(len(lon), bool)
    border, vicinity, unknown = np.zeros(len(lon), bool), np.zeros(len(lon), bool), np.zeros(len(lon), bool)
    d, step = config['geographic_domain'], config['grid_degrees']
    for f in doc['features']:
        p, g = f['properties'], shape(f['geometry'])
        assert p['taxon_id'] == taxon['taxon_id'] and p['name_use_id_orig'] == taxon['name_use_id_orig']
        assert p['name_choro_orig'] == taxon['source_name'] and g.is_valid and not g.is_empty
        native = p['occurrence_status'] == 'Vorkommen' and p['occurrence_origin'] == 'nativ'
        if not native:
            unknown |= shapely.intersects_xy(g.buffer(config['labels']['other_status_or_origin_exclusion_degrees']), lon, lat)
        elif g.geom_type == 'Point':
            col = int(np.floor((g.x-d['west'])/step))
            row = int(np.floor((d['north']-g.y)/step))
            if 0 <= row < grid_shape[0] and 0 <= col < grid_shape[1]:
                point_cells[row*grid_shape[1]+col] = True
            vicinity |= shapely.intersects_xy(g.buffer(config['labels']['point_neighbourhood_exclusion_degrees']), lon, lat)
        else:
            y |= shapely.intersects_xy(g, lon, lat)
            border |= shapely.intersects_xy(g.boundary.buffer(config['labels']['polygon_boundary_exclusion_degrees']), lon, lat)
    y |= point_cells
    use = ~border & ~unknown & (~vicinity | point_cells)
    return y.astype(int), use


def checked_fit(model, x, y, w, config):
    mu = np.average(x, axis=0, weights=w)
    sd = np.sqrt(np.average((x-mu)**2, axis=0, weights=w))
    np.testing.assert_allclose(model['mean'], mu, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(model['scale'], sd, rtol=1e-12, atol=1e-12)
    assert model['training_cells'] == len(y) and model['training_positive_cells'] == int(y.sum())
    assert min(y.sum(), len(y)-y.sum()) >= config['model']['minimum_training_cells_per_class']
    np.testing.assert_allclose(model['training_prevalence'], np.average(y, weights=w), atol=1e-14)
    a, b = ((x-mu)/sd).T
    matrix = np.stack([np.ones(len(x)), a, b, a*b, a*a, b*b], axis=1)
    beta = np.array(model['coefficients'])
    eta = matrix @ beta
    probability = .5 + .5*np.tanh(eta/2)
    gradient = np.sum(matrix * ((probability-y)*w/w.sum())[:, None], axis=0)
    gradient[1:] += config['model']['ridge_lambda']*beta[1:]
    maximum = float(np.max(np.abs(gradient)))
    assert maximum <= config['model']['accepted_gradient_max']
    np.testing.assert_allclose(model['gradient_max'], maximum, atol=1e-13)
    loss = np.average(np.maximum(eta, 0)+np.log1p(np.exp(-np.abs(eta)))-y*eta, weights=w)
    loss += config['model']['ridge_lambda']/2 * np.sum(beta[1:]**2)
    np.testing.assert_allclose(model['objective'], loss, atol=1e-12)
    hessian = matrix.T @ (matrix * (w/w.sum()*probability*(1-probability))[:, None])
    hessian[1:, 1:] += config['model']['ridge_lambda']*np.eye(5)
    assert np.linalg.eigvalsh(hessian).min() > 0
    hull = ConvexHull(x)
    assert set(map(tuple, x[hull.vertices])) == set(map(tuple, model['climate_hull']))
    return maximum


def forward(model, x):
    t, p = ((x-np.array(model['mean']))/np.array(model['scale'])).T
    b = model['coefficients']
    eta = b[0]+b[1]*t+b[2]*p+b[3]*t*p+b[4]*t*t+b[5]*p*p
    predicted = .5+.5*np.tanh(eta/2)
    hull = ConvexHull(np.array(model['climate_hull']))
    inside = np.max(x @ hull.equations[:, :2].T+hull.equations[:, 2], axis=1) <= 1e-9
    return predicted, inside


def verify(maps, climate, config_path, sources_path, output, report):
    config, sources = json.loads(config_path.read_text()), json.loads(sources_path.read_text())
    for r in sources['map_files']:
        assert sha(maps/r['path']) == r['sha256']
    assert sha(climate) == sources['climate_file']['sha256']
    manifest = json.loads((output/'manifest.json').read_text())
    assert sha(config_path) == manifest['config_sha256'] and sha(sources_path) == manifest['sources_sha256']
    assert all(sha(output/name) == value for name, value in manifest['output_sha256'].items())
    for name, value in manifest['code_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == value
    d, step = config['geographic_domain'], config['grid_degrees']
    xs = d['west']+(np.arange(int((d['east']-d['west'])/step))+.5)*step
    ys = d['north']-(np.arange(int((d['north']-d['south'])/step))+.5)*step
    xx, yy = np.meshgrid(xs, ys)
    lon, lat = xx.ravel(), yy.ravel()
    climate_arrays = []
    with zipfile.ZipFile(climate) as archive:
        for name in ('wc2.1_10m_bio_1.tif', 'wc2.1_10m_bio_12.tif'):
            blob = archive.read(name)
            assert hashlib.sha256(blob).hexdigest() == sources['worldclim_members'][name]['sha256']
            with rasterio.io.MemoryFile(blob) as memory:
                with memory.open() as ds:
                    image = ds.read(1)
                    rr, cc = np.floor((90-lat)*6).astype(int), np.floor((lon+180)*6).astype(int)
                    values = image[rr, cc].astype(float)
                    values[values == ds.nodata] = np.nan
                    climate_arrays.append(values)
    climate_values = np.stack(climate_arrays, axis=1)
    land = np.isfinite(climate_values).all(axis=1)
    models = json.loads((output/'models.json').read_text())
    folds = {(r['taxon'], r['block_degrees'], r['fold']): r for r in json.loads((output/'fold_models.json').read_text())}
    assert len(models) == 7 and len(folds) == 70
    roles = {(r['taxon'], int(r['block_degrees']), int(r['fold'])): r for r in table(output/'split_inventory.csv')}
    scores = {(r['taxon'], int(r['block_degrees']), r['scope']): r for r in table(output/'scores.csv')}
    reliability = {(r['taxon'], int(r['block_degrees']), int(r['bin'])): r for r in table(output/'reliability.csv')}
    inventory = {r['taxon']: r for r in table(output/'sample_inventory.csv')}
    assert len(roles) == 70 and len(scores) == 84 and len(reliability) == 140 and len(inventory) == 7
    total_predictions, label_cells, max_gradient, min_distance = 0, 0, 0., float('inf')

    def audit_scores(key, y, p, b, w, inside):
        record = scores[key]
        normalized = w/w.sum()
        bs = np.sum(normalized*(p-y)**2)
        base_bs = np.sum(normalized*(b-y)**2)
        def logloss(prob):
            safe = np.maximum(1e-12, np.minimum(1-1e-12, prob))
            return -np.sum(normalized*(y*np.log(safe)+(1-y)*np.log(1-safe)))
        expected = {'brier': bs, 'comparator_brier': base_bs, 'brier_skill': 1-bs/base_bs,
                    'log_loss': logloss(p), 'comparator_log_loss': logloss(b),
                    'observed_prevalence': np.sum(normalized*y), 'mean_prediction': np.sum(normalized*p)}
        for col, value in expected.items():
            np.testing.assert_allclose(float(record[col]), value, atol=1e-12)
        assert int(record['test_cells']) == len(y) and int(record['outside_hull_cells']) == int((~inside).sum())

    with (output/'predictions.csv').open() as handle:
        groups = iter(itertools.groupby(csv.DictReader(handle), key=lambda r: (r['taxon'], int(r['block_degrees']))))
        for taxon in config['taxa']:
            name = taxon['id']
            y_grid, use = geometry_labels(maps/f"Data_in/Areale/geojson/taxon_id_{taxon['taxon_id']}.geojson", taxon, config, lon, lat, xx.shape)
            selected = np.flatnonzero(land & use)
            x, y, lo, la = climate_values[selected], y_grid[selected], lon[selected], lat[selected]
            w = np.cos(np.deg2rad(la))
            inv = inventory[name]
            assert int(inv['retained_cells']) == len(y) and int(inv['positive_cells']) == y.sum()
            assert int(inv['mapped_outside_cells']) == (y == 0).sum() and int(inv['land_cells_before_map_mask']) == land.sum()
            label_cells += len(y)
            max_gradient = max(max_gradient, checked_fit(models[name], x, y, w, config))
            for block in config['validation']['block_sizes_degrees']:
                group_key, rows = next(groups)
                assert group_key == (name, block)
                rows = list(rows)
                by_id = {int(r['cell_index']): r for r in rows}
                assert len(rows) == len(by_id) == len(y) and set(by_id) == set(selected)
                ordered = [by_id[i] for i in selected]
                saved_p = np.array([float(r['prediction']) for r in ordered])
                saved_b = np.array([float(r['comparator']) for r in ordered])
                saved_inside = np.array([r['in_training_climate_hull'] == 'True' for r in ordered])
                assert np.array_equal([int(r['label']) for r in ordered], y)
                assigned = ((np.floor((la+90)/block).astype(int)+2*np.floor((lo+180)/block).astype(int)) % 5)
                assert np.array_equal([int(r['fold']) for r in ordered], assigned)
                for fold in range(5):
                    test = assigned == fold
                    # Haversine BallTree is independent of producer chord/cKDTree.
                    distances = BallTree(np.deg2rad(np.column_stack([la[test], lo[test]])), metric='haversine').query(np.deg2rad(np.column_stack([la, lo])), k=1)[0][:, 0]*config['validation']['sphere_radius_km']
                    train = ~test & (distances > config['validation']['buffer_km'])
                    role = roles[name, block, fold]
                    assert int(role['train_cells']) == train.sum() and int(role['test_cells']) == test.sum()
                    assert int(role['buffered_cells']) == (~train & ~test).sum()
                    nearest = distances[train].min()
                    np.testing.assert_allclose(float(role['minimum_train_test_km']), nearest, atol=1e-8)
                    assert nearest > 100
                    min_distance = min(min_distance, nearest)
                    model = folds[name, block, fold]
                    max_gradient = max(max_gradient, checked_fit(model, x[train], y[train], w[train], config))
                    pp, support = forward(model, x[test])
                    np.testing.assert_allclose(saved_p[test], pp, atol=2e-12, rtol=1e-10)
                    assert np.array_equal(saved_inside[test], support)
                    np.testing.assert_allclose(saved_b[test], np.average(y[train], weights=w[train]), atol=1e-14)
                    audit_scores((name, block, str(fold)), y[test], saved_p[test], saved_b[test], w[test], support)
                audit_scores((name, block, 'all'), y, saved_p, saved_b, w, saved_inside)
                bins = np.minimum((saved_p*10).astype(int), 9)
                for j in range(10):
                    r, mask = reliability[name, block, j], bins == j
                    assert int(r['cells']) == mask.sum()
                    np.testing.assert_allclose(float(r['area_weight']), w[mask].sum()/w.sum(), atol=1e-14)
                    if mask.any():
                        np.testing.assert_allclose(float(r['mean_prediction']), np.average(saved_p[mask], weights=w[mask]), atol=1e-14)
                        np.testing.assert_allclose(float(r['observed_fraction']), np.average(y[mask], weights=w[mask]), atol=1e-14)
                    else:
                        assert r['mean_prediction'] == r['observed_fraction'] == ''
                total_predictions += len(y)
            print('verified', name, len(y), flush=True)
        assert next(groups, None) is None
    result = {'result': 'passed', 'species_models': 7, 'fold_models': 70,
              'source_label_cells_checked': label_cells, 'prediction_rows_checked': total_predictions,
              'score_rows_checked': 84, 'reliability_rows_checked': 140,
              'maximum_recomputed_gradient': max_gradient, 'minimum_haversine_train_test_km': min_distance,
              'producer_imported': False, 'shared_dependencies': ['rasterio/GDAL TIFF decoding', 'shapely/GEOS geometry', 'NumPy/SciPy numerics'],
              'independent_paths': ['vector predicates instead of rasterize', 'direct TIFF row/column indexing', 'Haversine BallTree instead of chord cKDTree', 'recomputed logistic probabilities, gradients, Hessians and scores'],
              'source_hashes_checked': True, 'output_hashes_checked': True, 'fossil_mapping_approved': False,
              'p1_released': False}
    report.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('maps', 'climate', 'config', 'sources', 'output', 'report'):
        parser.add_argument('--'+key, required=True, type=Path)
    a = parser.parse_args()
    verify(a.maps, a.climate, a.config, a.sources, a.output, a.report)
