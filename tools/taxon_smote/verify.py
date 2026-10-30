"""Independent raw-label, split, SMOTE-parent, network and score audit.

Never imports the new producer or network module. A hash-locked, independent
PR28 vector-label reader is reused; shared GDAL/GEOS/numerical libraries remain.
This checks derivations and numerical consistency, not source-map truth.
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
from scipy.stats import gamma, norm
from sklearn.neighbors import BallTree


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def close(a, b, atol=1e-11):
    np.testing.assert_allclose(a, b, atol=atol, rtol=1e-10)


def seed(master, case, component):
    return int.from_bytes(hashlib.sha256(f'{master}|{case}|{component}'.encode()).digest()[:4], 'big')


def sigmoid(value):
    # Independent forward path; tanh is stable for both tails.
    return (1+np.tanh(value/2))/2


def transform(x, p):
    cdf = gamma.cdf(np.sqrt(x[:, 1]), a=p['gamma_shape'], scale=1/p['gamma_rate'])
    return np.stack([(x[:, 0]-p['temperature_mean'])/p['temperature_sd'],
                     norm.ppf(np.clip(cdf, p['cdf_clip'], 1-p['cdf_clip']))], axis=1)


def calculation(theta, x, y=None, weights=None, decay=.1):
    h = (len(theta)-1)//4
    ih, hb, ho, ob = theta[:2*h].reshape(2, h), theta[2*h:3*h], theta[3*h:4*h], theta[-1]
    activations = sigmoid(np.einsum('ni,ij->nj', x, ih)+hb)
    logits = np.sum(activations*ho, axis=1)+ob
    prob = sigmoid(logits)
    if y is None:
        return logits, prob
    w = weights/weights.mean()
    residual = (prob-y)*w
    hidden_residual = residual[:, None]*ho*activations*(1-activations)
    grad = np.r_[np.einsum('ni,nj->ij', x, hidden_residual).ravel(),
                 hidden_residual.sum(axis=0), (activations*residual[:, None]).sum(axis=0), residual.sum()]
    grad += 2*decay*theta
    loss = np.sum(w*(np.maximum(logits, 0)+np.log1p(np.exp(-np.abs(logits)))-y*logits))+decay*np.square(theta).sum()
    return loss, np.max(np.abs(grad))/w.sum()


def metric(y, p, base, w):
    guessed, pos, neg = p >= .5, y == 1, y == 0
    sen = guessed[pos].mean() if pos.any() else None
    spe = (~guessed[neg]).mean() if neg.any() else None
    ba = (sen+spe)/2 if pos.any() and neg.any() else None
    wba = (np.average(guessed[pos], weights=w[pos])+np.average(~guessed[neg], weights=w[neg]))/2 if ba is not None else None
    bs, bb = np.average((p-y)**2, weights=w), np.average((base-y)**2, weights=w)
    safe = np.clip(p, 1e-12, 1-1e-12)
    return dict(cells=len(y), positive_cells=int(pos.sum()), sensitivity=sen, specificity=spe,
                balanced_accuracy=ba, area_weighted_balanced_accuracy=wba, brier=bs,
                comparator_brier=bb, brier_skill=1-bs/bb if bb else None,
                log_loss=-np.average(y*np.log(safe)+(1-y)*np.log1p(-safe), weights=w),
                mean_prediction=np.average(p, weights=w), observed_prevalence=np.average(y, weights=w))


def check_record(record, expected):
    for key, value in expected.items():
        if value is None:
            assert record[key] == '', (key, record[key])
        else:
            close(float(record[key]), value)


def verify(previous, maps, climate, config_path, sources_path, audit_sources_path, output, report):
    config = json.loads(config_path.read_text())
    sources = json.loads(sources_path.read_text())
    audit_sources = json.loads(audit_sources_path.read_text())
    for r in sources['previous_files']+[audit_sources['previous_raw_checker']]:
        assert Path(r['file']).name == r['file']
        assert (previous/r['file']).stat().st_size == r['bytes'] and sha(previous/r['file']) == r['sha256']
    spec = importlib.util.spec_from_file_location('locked_independent_raw_reader', previous/'verify.py')
    raw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(raw)
    old = json.loads((previous/'config.json').read_text())
    old_sources = json.loads((previous/'sources.json').read_text())
    for r in old_sources['map_files']:
        assert sha(maps/r['path']) == r['sha256']
    assert sha(climate) == old_sources['climate_file']['sha256']
    manifest = json.loads((output/'manifest.json').read_text())
    assert sha(config_path) == manifest['config_sha256'] and sha(sources_path) == manifest['sources_sha256']
    for filename, checksum in manifest['outputs_sha256'].items():
        assert sha(output/filename) == checksum
    for filename, checksum in manifest['code_sha256'].items():
        assert sha(Path(__file__).with_name(filename)) == checksum

    d, step = old['geographic_domain'], old['grid_degrees']
    xx, yy = np.meshgrid(np.arange(d['west']+step/2, d['east'], step),
                         np.arange(d['north']-step/2, d['south'], -step))
    lon, lat = xx.ravel(), yy.ravel()
    arrays = []
    with zipfile.ZipFile(climate) as archive:
        for name in ('wc2.1_10m_bio_1.tif', 'wc2.1_10m_bio_12.tif'):
            blob = archive.read(name)
            assert hashlib.sha256(blob).hexdigest() == old_sources['worldclim_members'][name]['sha256']
            with rasterio.io.MemoryFile(blob) as memory:
                with memory.open() as ds:
                    values = ds.read(1)[np.floor((90-lat)*6).astype(int), np.floor((lon+180)*6).astype(int)].astype(float)
                    values[values == ds.nodata] = np.nan
                    arrays.append(values)
    climate_values = np.stack(arrays, axis=1)
    land = np.isfinite(climate_values).all(axis=1)
    fitted_list = json.loads((output/'fit_models.json').read_text())
    fitted = {(m['case_id'], m['method']): m for m in fitted_list}
    final = json.loads((output/'models.json').read_text())
    baseline = {(m['taxon'], m['block_degrees'], m['fold']): m for m in json.loads((previous/'fold_models.json').read_text())}
    roles = {r['case']: r for r in table(output/'split_inventory.csv')}
    scores = {(r['taxon'], int(r['block_degrees']), r['scope'], r['method']): r for r in table(output/'scores.csv')}
    reliability = {(r['taxon'], int(r['block_degrees']), r['method'], int(r['bin'])): r for r in table(output/'reliability.csv')}
    inventory = {r['taxon']: r for r in table(output/'sample_inventory.csv')}
    assert len(fitted) == len(fitted_list) == 154 and len(final) == len(inventory) == 7
    assert len(roles) == 70 and len(scores) == 336 and len(reliability) == 560
    assert table(output/'metrics_summary.csv') == [r for r in table(output/'scores.csv') if r['scope'] == 'all']
    total_trace, starts, predictions, cells_checked, max_gradient = 0, 0, 0, 0, 0.
    minimum_distance = float('inf')

    with gzip.open(output/'smote_trace.csv.gz', 'rt') as trace_handle, (output/'predictions.csv').open() as prediction_handle:
        traces = iter(itertools.groupby(csv.DictReader(trace_handle), key=lambda r: r['case']))
        prediction_groups = iter(itertools.groupby(csv.DictReader(prediction_handle), key=lambda r: (r['taxon'], int(r['block_degrees']))))
        for taxon in old['taxa']:
            name = taxon['id']
            labels, use = raw.geometry_labels(maps/f"Data_in/Areale/geojson/taxon_id_{taxon['taxon_id']}.geojson", taxon, old, lon, lat, xx.shape)
            selected = np.flatnonzero(land & use)
            x, y, lo, la = climate_values[selected], labels[selected], lon[selected], lat[selected]
            area = np.cos(np.deg2rad(la))
            assert int(inventory[name]['cells']) == len(y) and int(inventory[name]['positive_cells']) == y.sum()
            cells_checked += len(y)

            def audit_case(case, train):
                nonlocal total_trace, starts, max_gradient
                a, b, w, ids = x[train], y[train], area[train], selected[train]
                original, smote = fitted[case, 'nnet_original'], fitted[case, 'nnet_smote']
                p = original['normalization']
                assert p == smote['normalization'] and p['fit_on_original_training_only']
                close(p['temperature_mean'], np.average(a[:, 0], weights=w))
                close(p['temperature_sd'], np.sqrt(np.average((a[:, 0]-p['temperature_mean'])**2, weights=w)))
                positive = a[:, 1] > 0
                assert p['positive_precipitation_training_cells'] == positive.sum()
                v = np.sqrt(a[positive, 1])
                mean = np.average(v, weights=w[positive])
                close(p['gamma_rate'], p['gamma_shape']/mean)
                close(np.log(p['gamma_shape'])-digamma(p['gamma_shape']), np.log(mean)-np.average(np.log(v), weights=w[positive]))
                assert p['cdf_clip'] == config['normalization']['cdf_clip']
                z = transform(a, p)
                trace_case, trace_rows = next(traces)
                assert trace_case == case
                trace_rows = list(trace_rows)
                counts = np.bincount(b, minlength=2)
                minority = np.flatnonzero(b == np.argmin(counts))
                n = int(abs(counts[0]-counts[1]))
                assert len(trace_rows) == n
                lookup = {int(cell): i for i, cell in enumerate(ids)}
                # Lookup fails if either parent belongs to test/buffer or another taxon.
                parent = np.array([lookup[int(r['base_cell'])] for r in trace_rows])
                other = np.array([lookup[int(r['other_cell'])] for r in trace_rows])
                fraction = np.array([float(r['fraction']) for r in trace_rows])
                assert (parent != other).all() and (b[parent] == np.argmin(counts)).all() and (b[other] == b[parent]).all()
                assert ((fraction >= 0) & (fraction < 1)).all()
                rng = np.random.Generator(np.random.PCG64(seed(config['randomness']['master_seed'], case, 'smote')))
                np.testing.assert_array_equal(parent, minority[rng.integers(len(minority), size=n)])
                rng.integers(config['smote']['k_neighbors'], size=n)  # consume documented neighbor-slot draw
                np.testing.assert_array_equal(fraction, rng.random(n))
                # Independent Euclidean BallTree: chosen other must be among five
                # closest non-self minority rows; all equal-distance ties are valid.
                kth = BallTree(z[minority], metric='euclidean').query(z[parent], k=config['smote']['k_neighbors']+1)[0][:, -1]
                assert (np.linalg.norm(z[parent]-z[other], axis=1) <= kth+1e-10).all()
                sx = np.concatenate([z, z[parent]+fraction[:, None]*(z[other]-z[parent])])
                sy = np.r_[b, b[parent]]
                sw = np.r_[w, w[parent]+fraction*(w[other]-w[parent])]
                total_trace += n
                for model, nx, ny, nw in ((original, z, b, w), (smote, sx, sy, sw)):
                    assert model['taxon'] == taxon and model['original_training_cells'] == len(b)
                    assert model['original_training_positive_cells'] == b.sum()
                    assert model['synthetic_cells'] == (n if model['method'] == 'nnet_smote' else 0)
                    assert model['resampled_case_count'] == len(ny)
                    close(model['original_area_weighted_prior'], np.average(b, weights=w))
                    close(model['resampled_case_weighted_prior'], np.average(ny, weights=nw))
                    assert set(map(tuple, model['climate_hull'])) == set(map(tuple, a[ConvexHull(a).vertices]))
                    assert model['hidden_units'] == config['network']['hidden_units'] and model['weight_decay'] == config['network']['weight_decay']
                    assert len(model['starts']) == config['network']['initializations']
                    for j, start in enumerate(model['starts']):
                        assert start['seed'] == seed(config['randomness']['master_seed'], case, f'initialization:{j}')
                        loss, gradient = calculation(np.array(start['weights']), nx, ny, nw, model['weight_decay'])
                        close(start['objective'], loss, atol=1e-8)
                        close(start['gradient_per_case'], gradient, atol=1e-12)
                        assert start['stationary'] == (gradient <= config['network']['accepted_gradient_per_effective_case'])
                        if start['stationary']:
                            max_gradient = max(max_gradient, gradient)
                        starts += 1
                    stationary = [j for j, r in enumerate(model['starts']) if r['stationary']]
                    chosen = min(stationary, key=lambda j: (model['starts'][j]['objective'], j))
                    assert model['selected_start'] == chosen and model['weights'] == model['starts'][chosen]['weights']
                assert smote['smote_seed'] == seed(config['randomness']['master_seed'], case, 'smote')
                assert smote['smote_minority_label'] == np.argmin(counts)
                return original, smote

            for block in old['validation']['block_sizes_degrees']:
                key, rows = next(prediction_groups)
                assert key == (name, block)
                rows = list(rows)
                by_id = {int(r['cell_index']): r for r in rows}
                assert len(rows) == len(by_id) == len(y) and set(by_id) == set(selected)
                ordered = [by_id[i] for i in selected]
                assigned = (np.floor((la+90)/block).astype(int)+2*np.floor((lo+180)/block).astype(int)) % 5
                np.testing.assert_array_equal([int(r['label']) for r in ordered], y)
                np.testing.assert_array_equal([int(r['fold']) for r in ordered], assigned)
                probabilities = {method: np.array([float(r[method]) for r in ordered]) for method in config['comparison_models']}
                saved_base = np.array([float(r['comparator']) for r in ordered])
                saved_support = np.array([r['in_training_climate_hull'] == 'True' for r in ordered])
                for fold in range(5):
                    test = assigned == fold
                    distances = BallTree(np.deg2rad(np.column_stack([la[test], lo[test]])), metric='haversine').query(np.deg2rad(np.column_stack([la, lo])), k=1)[0][:, 0]*old['validation']['sphere_radius_km']
                    train = ~test & (distances > old['validation']['buffer_km'])
                    case = f'{name}:b{block}:f{fold}'
                    role = roles[case]
                    check_record(role, dict(train_cells=train.sum(), test_cells=test.sum(), buffered_cells=(~train & ~test).sum(), minimum_train_test_km=distances[train].min(), original_training_prior=np.average(y[train], weights=area[train])))
                    minimum_distance = min(minimum_distance, distances[train].min())
                    original, smote = audit_case(case, train)
                    assert int(role['synthetic_training_cells']) == smote['synthetic_cells']
                    pp, inside = raw.forward(baseline[name, block, fold], x[test])
                    expected = {'pr28_quadratic': pp}
                    expected['nnet_original'] = calculation(np.array(original['weights']), transform(x[test], original['normalization']))[1]
                    logits, expected['nnet_smote'] = calculation(np.array(smote['weights']), transform(x[test], smote['normalization']))
                    a, b = smote['original_area_weighted_prior'], smote['resampled_case_weighted_prior']
                    expected['nnet_smote_prior_adjusted'] = sigmoid(logits+np.log(a/(1-a))-np.log(b/(1-b)))
                    np.testing.assert_array_equal(saved_support[test], inside)
                    close(saved_base[test], np.average(y[train], weights=area[train]))
                    for method, expected_p in expected.items():
                        close(probabilities[method][test], expected_p, atol=2e-11)
                        # Audit the recorded scores on their actual recorded inputs.
                        # Independent tanh forward values are checked above; their
                        # sub-ULP tail differences can amplify inside log-loss.
                        check_record(scores[name, block, str(fold), method], metric(y[test], probabilities[method][test], saved_base[test], area[test]))
                for method, pp in probabilities.items():
                    check_record(scores[name, block, 'all', method], metric(y, pp, saved_base, area))
                    bins = np.minimum((pp*10).astype(int), 9)
                    for j in range(10):
                        mask = bins == j
                        check_record(reliability[name, block, method, j], dict(cells=mask.sum(), area_weight=area[mask].sum()/area.sum(),
                            mean_prediction=np.average(pp[mask], weights=area[mask]) if mask.any() else None,
                            observed_fraction=np.average(y[mask], weights=area[mask]) if mask.any() else None))
                predictions += len(y)
            original, smote = audit_case(name+':full', np.ones(len(y), bool))
            assert final[name] == {'nnet_original': original, 'nnet_smote': smote}
            print('verified', name, len(y), flush=True)
        assert next(traces, None) is None and next(prediction_groups, None) is None
    assert total_trace == manifest['synthetic_rows'] and starts == 462
    result = dict(result='passed', source_label_cells_checked=cells_checked, prediction_rows_checked=predictions,
                  individual_probabilities_checked=4*predictions, network_models_checked=154, optimizer_starts_checked=starts,
                  synthetic_rows_with_both_original_training_parents_checked=total_trace, score_rows_checked=336,
                  reliability_rows_checked=560, maximum_accepted_gradient_per_case=max_gradient,
                  minimum_haversine_train_test_km=minimum_distance, producer_imported=False,
                  independent_paths=['locked vector predicates instead of producer rasterization', 'direct TIFF indexing',
                    'Haversine and Euclidean BallTree instead of producer cKDTree', 'scipy.stats Gamma-CDF/normal-PPF',
                    'tanh network forward/backpropagation; every start objective and selection', 'classification/probability scores'],
                  shared_dependencies=['PR28 independent vector-label reader (hash locked)', 'rasterio/GDAL', 'shapely/GEOS', 'NumPy/SciPy'],
                  source_hashes_checked=True, code_and_output_hashes_checked=True,
                  audit_code_sha256=sha(Path(__file__)), audit_sources_sha256=sha(audit_sources_path),
                  scientific_map_truth_established=False, fossil_application_approved=False, p1_released=False)
    report.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('previous', 'maps', 'climate', 'config', 'sources', 'audit_sources', 'output', 'report'):
        parser.add_argument('--'+key.replace('_', '-'), required=True, type=Path)
    a = parser.parse_args()
    verify(a.previous, a.maps, a.climate, a.config, a.sources, a.audit_sources, a.output, a.report)
