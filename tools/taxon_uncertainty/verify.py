"""Independent conditional-bootstrap audit: no new producer/bootstrap imports."""
import argparse
from collections import Counter
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
from scipy.spatial import ConvexHull


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def quantiles(values, probabilities):
    ordered = np.sort(values, axis=0)
    result = []
    for probability in probabilities:
        position = probability*(len(ordered)-1)
        lo, hi = int(np.floor(position)), int(np.ceil(position))
        result.append(ordered[lo]+(position-lo)*(ordered[hi]-ordered[lo]))
    return np.array(result)


def verify(previous, maps, climate, config_path, output, report):
    root = Path(__file__).resolve().parents[2]
    config = json.loads(config_path.read_text())
    for file, digest in config['locked_sha256'].items():
        assert sha(root/file) == digest, file
    audit31 = module('locked_spatial_audit_31', root/'tools/taxon_calibration/verify.py')
    a29 = module('locked_arithmetic_audit_29', root/'tools/taxon_smote/verify.py')
    close, check = a29.close, a29.check_record
    d31 = config_path.parent.parent/'calibration_2026_10_30'
    c31 = json.loads((d31/'config.json').read_text())
    d29 = d31.parent/'smote_nnet_2026_10_30'
    for file, digest in c31['locked_inputs_sha256'].items():
        assert sha(d29/file) == digest
    sources = json.loads((d29/'sources.json').read_text())
    audit_sources = json.loads((d29/'audit_sources.json').read_text())
    for r in sources['previous_files']+[audit_sources['previous_raw_checker']]:
        assert Path(r['file']).name == r['file']
        assert sha(previous/r['file']) == r['sha256'] and (previous/r['file']).stat().st_size == r['bytes']
    raw = module('locked_vector_audit_28', previous/'verify.py')
    old = json.loads((previous/'config.json').read_text())
    src = json.loads((previous/'sources.json').read_text())
    for r in src['map_files']:
        assert sha(maps/r['path']) == r['sha256']
    assert sha(climate) == src['climate_file']['sha256']
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['config_sha256'] == sha(config_path) and manifest['design_sha256'] == sha(config_path.with_name('DESIGN.md'))
    assert manifest['source_model_sha256'] == config['locked_sha256']
    for file, digest in manifest['outputs_sha256'].items():
        assert Path(file).name == file and sha(output/file) == digest
    for file, digest in manifest['code_sha256'].items():
        assert sha(root/file) == digest
    dd, step = old['geographic_domain'], old['grid_degrees']
    xx, yy = np.meshgrid(np.arange(dd['west']+step/2, dd['east'], step), np.arange(dd['north']-step/2, dd['south'], -step))
    lo_all, la_all = xx.ravel(), yy.ravel()
    arrays = []
    with zipfile.ZipFile(climate) as archive:
        for name in ['wc2.1_10m_bio_1.tif', 'wc2.1_10m_bio_12.tif']:
            blob = archive.read(name)
            assert hashlib.sha256(blob).hexdigest() == src['worldclim_members'][name]['sha256']
            with rasterio.io.MemoryFile(blob) as memory:
                with memory.open() as ds:
                    v = ds.read(1)[np.floor((90-la_all)*6).astype(int), np.floor((lo_all+180)*6).astype(int)].astype(float)
                    v[v == ds.nodata] = np.nan
                    arrays.append(v)
    all_x = np.stack(arrays, axis=1)
    land = np.isfinite(all_x).all(axis=1)
    fits = {(m['case_id'], m['method']): m for m in json.loads((d31/'fit_models.json').read_text())}
    inventory_list = json.loads((output/'inventory.json').read_text())
    inventory = {r['case']: r for r in inventory_list}
    with (output/'summary.csv').open() as f:
        summary_list = list(csv.DictReader(f))
    summary = {(r['taxon'], int(r['block_degrees']), r['outer'], r['family']): r for r in summary_list}
    full = json.loads((output/'full_parameters.json').read_text())
    assert len(inventory) == len(inventory_list) == 84 and len(summary) == len(summary_list) == 168
    B, minimum = config['replicates'], config['minimum_valid_replicates']
    totals = Counter()
    minimum_outer, minimum_inner, max_gradient = float('inf'), float('inf'), 0.
    with gzip.open(output/'block_draws.csv.gz', 'rt') as df, gzip.open(output/'parameters.csv.gz', 'rt') as pf, gzip.open(output/'intervals.csv.gz', 'rt') as vf:
        draw_groups = iter(itertools.groupby(csv.DictReader(df), key=lambda r: r['case']))
        parameter_groups = iter(itertools.groupby(csv.DictReader(pf), key=lambda r: (r['case'], r['family'], int(r['inner']))))
        prediction_groups = iter(itertools.groupby(csv.DictReader(vf), key=lambda r: (r['taxon'], int(r['block_degrees']), r['outer'], r['family'])))
        for taxon in old['taxa']:
            labels, use = raw.geometry_labels(maps/f"Data_in/Areale/geojson/taxon_id_{taxon['taxon_id']}.geojson", taxon, old, lo_all, la_all, xx.shape)
            selected = np.flatnonzero(land & use)
            x, y, lon, lat = all_x[selected], labels[selected], lo_all[selected], la_all[selected]
            area = np.cos(np.deg2rad(lat))
            totals['source_label_cells'] += len(y)
            for block in old['validation']['block_sizes_degrees']:
                cases = []
                for outer, train, test, info in audit31.split(lon, lat, block, 5, 100, old['validation']['sphere_radius_km']):
                    cases.append((str(outer), train, test))
                    minimum_outer = min(minimum_outer, info['minimum_train_test_km'])
                cases.append(('full', np.ones(len(y), bool), np.ones(len(y), bool)))
                for outer, train, evaluation in cases:
                    case = f"{taxon['id']}:b{block}:o{outer}"
                    pool = np.flatnonzero(train)
                    block_pairs = [(int(np.floor((lat[k]+90)/block)), int(np.floor((lon[k]+180)/block))) for k in pool]
                    ids = sorted(set(block_pairs))
                    lookup = {pair: j for j, pair in enumerate(ids)}
                    inverse = np.array([lookup[pair] for pair in block_pairs])
                    G = len(ids)
                    trace_case, trace_rows = next(draw_groups)
                    assert trace_case == case
                    counts = np.zeros((B, G), int)
                    seen = set()
                    for r in trace_rows:
                        rep = int(r['replicate'])
                        pair = int(r['latitude_block']), int(r['longitude_block'])
                        j = lookup[pair]
                        assert 0 <= rep < B and (rep, j) not in seen and int(r['multiplicity']) > 0
                        seen.add((rep, j))
                        counts[rep, j] = int(r['multiplicity'])
                        totals['nonzero_block_multiplicities'] += 1
                    seed = a29.seed(config['master_seed'], case, 'block_bootstrap')
                    rng = np.random.Generator(np.random.PCG64(seed))
                    for rep in range(B):
                        counter = Counter(int(v) for v in rng.integers(G, size=G))
                        np.testing.assert_array_equal(counts[rep], [counter[j] for j in range(G)])
                    assert (counts.sum(axis=1) == G).all()
                    record = inventory[case]
                    assert record['blocks'] == G and record['pool_cells'] == len(pool) and record['seed'] == seed
                    inner_cases = []
                    metadata = []
                    for inner, tr, ca, info in audit31.split(lon[pool], lat[pool], block, 3, 100, old['validation']['sphere_radius_km']):
                        cal = pool[ca]
                        inner_cases.append((tr, cal, inverse[ca]))
                        minimum_inner = min(minimum_inner, info['minimum_train_test_km'])
                        metadata.append({'calibration_cells': len(cal), 'positive': int(y[cal].sum()),
                                         'positive_blocks': len(set(inverse[ca][y[cal] == 1])), 'calibration_blocks': len(set(inverse[ca]))})
                    assert record['members'] == metadata
                    for family in ['original', 'smote']:
                        members = [fits[case+f':i{j}', family] for j in range(3)]
                        available = all(m['calibrator']['status'] == 'fitted' for m in members)
                        saved, valid_member, coeff, failure = [], [], [], Counter()
                        for inner, (tr, cal, cb) in enumerate(inner_cases):
                            key, rows = next(parameter_groups)
                            assert key == (case, family, inner)
                            rows = list(rows)
                            assert len(rows) == B
                            m = members[inner]
                            assert m['original_training_cells'] == tr.sum()
                            eta = a29.calculation(np.array(m['weights']), a29.transform(x[cal], m['normalization']))[0]
                            valid = np.zeros(B, bool)
                            co = []
                            for rep, r in enumerate(rows):
                                assert int(r['replicate']) == rep
                                active = counts[rep, cb] > 0
                                unique = np.bincount(y[cal][active], minlength=2)
                                assert int(r['negative']) == unique[0] and int(r['positive']) == unique[1]
                                expected = 'base_member_uncalibrated' if not available else ('insufficient_unique_class' if unique.min() < config['minimum_unique_active_class'] else None)
                                if expected is not None:
                                    assert r['status'] == expected and all(r[k] == '' for k in ['slope', 'intercept', 'objective', 'projected_gradient'])
                                    co.append([None, None])
                                else:
                                    aa, bb = float(r['slope']), float(r['intercept'])
                                    assert np.isfinite([aa, bb]).all() and aa >= 0
                                    ee, yy = eta[active], y[cal][active]
                                    w = area[cal][active]*counts[rep, cb][active]
                                    w /= w.sum()
                                    z = aa*ee+bb
                                    residual = w*(a29.sigmoid(z)-yy)
                                    ridge = c31['calibration']['ridge_slope']
                                    gradient = np.array([np.sum(residual*ee)+ridge*aa, residual.sum()])
                                    if aa == 0 and gradient[0] > 0:
                                        gradient[0] = 0
                                    pg = float(abs(gradient).max())
                                    loss = float(np.sum(w*(np.maximum(z, 0)+np.log1p(np.exp(-abs(z)))-yy*z))+ridge*aa**2/2)
                                    close(float(r['objective']), loss)
                                    close(float(r['projected_gradient']), pg, atol=1e-12)
                                    fitted = pg <= c31['calibration']['projected_gradient_tolerance']
                                    assert r['status'] == ('fitted' if fitted else 'numerical_failure')
                                    valid[rep] = fitted
                                    if fitted:
                                        max_gradient = max(max_gradient, pg)
                                    co.append([aa, bb])
                                failure[r['status']] += 1
                                totals['member_replicates_'+r['status']] += 1
                            saved.append(rows)
                            valid_member.append(valid)
                            coeff.append(co)
                        valid = np.all(valid_member, axis=0)
                        nvalid = int(valid.sum())
                        coefficients = [[coeff[j][r] for j in range(3)] for r in range(B)]
                        etas = [a29.calculation(np.array(m['weights']), a29.transform(x[evaluation], m['normalization']))[0] for m in members]
                        point = sum(a29.sigmoid(m['calibrator']['slope']*eta+m['calibrator']['intercept']) for m, eta in zip(members, etas))/3
                        support = sum((x[evaluation] @ ConvexHull(np.asarray(m['climate_hull'])).equations[:, :-1].T+
                                       ConvexHull(np.asarray(m['climate_hull'])).equations[:, -1] <= 1e-8).all(axis=1) for m in members)
                        band = None
                        if nvalid >= minimum:
                            pp = np.zeros((nvalid, evaluation.sum()))
                            for index, rep in enumerate(np.flatnonzero(valid)):
                                pp[index] = sum(a29.sigmoid(coeff[j][rep][0]*etas[j]+coeff[j][rep][1]) for j in range(3))/3
                            band = quantiles(pp, config['quantiles'])
                        status = 'available_conditional_band' if band is not None else ('base_member_uncalibrated' if not available else 'insufficient_estimable_replicates')
                        key, rows = next(prediction_groups)
                        assert key == (taxon['id'], block, outer, family)
                        rows = list(rows)
                        np.testing.assert_array_equal([int(r['cell_index']) for r in rows], selected[evaluation])
                        np.testing.assert_array_equal([int(r['support_members']) for r in rows], support)
                        assert all(int(r['valid_replicates']) == nvalid and r['status'] == status for r in rows)
                        close([float(r['point']) for r in rows], point, atol=2e-11)
                        if band is None:
                            assert all(all(r[k] == '' for k in ['lower', 'median', 'upper']) for r in rows)
                        else:
                            close(np.array([[float(r[k]) for r in rows] for k in ['lower', 'median', 'upper']]), band, atol=2e-11)
                        totals['interval_rows'] += len(rows)
                        totals['ensemble_cases'] += 1
                        totals['ensemble_cases_'+status] += 1
                        rr = summary[key]
                        assert rr['status'] == status
                        width = band[2]-band[0] if band is not None else None
                        check(rr, {'valid_replicates': nvalid, 'total_replicates': B, 'evaluation_cells': evaluation.sum(),
                            'cells_outside_any_hull': (support < 3).sum(),
                            'mean_width_area_weighted': np.average(width, weights=area[evaluation]) if width is not None else None,
                            **{f'width_unweighted_q{q}': quantiles(width[:, None], [q/100])[0, 0] if width is not None else None for q in [5, 50, 95]},
                            **{'member_replicates_'+s: failure[s] for s in ['fitted', 'insufficient_unique_class', 'numerical_failure', 'base_member_uncalibrated']}})
                        if outer == 'full':
                            record = full[taxon['id']][str(block)][family]
                            assert record['coefficients'] == coefficients and record['valid'] == valid.tolist()
                            assert record['minimum_valid'] == minimum and record['status'] == status
                            assert record['conditional_fixed_networks'] and not record['nominal_coverage_validated']
            print('verified bootstrap', taxon['id'], flush=True)
        assert next(draw_groups, None) is None and next(parameter_groups, None) is None and next(prediction_groups, None) is None
    assert totals['ensemble_cases'] == 168 and sum(totals['member_replicates_'+s] for s in
        ['fitted', 'insufficient_unique_class', 'numerical_failure', 'base_member_uncalibrated']) == 100800
    result = {'result': 'passed', **totals, 'minimum_outer_haversine_km': minimum_outer,
              'minimum_inner_haversine_km': minimum_inner, 'maximum_fitted_projected_gradient': max_gradient,
              'producer_or_bootstrap_fitter_imported': False, 'audit_code_sha256': sha(Path(__file__)),
              'independent_paths': ['vector labels and direct TIFF indexing', 'Haversine BallTree splits',
                'Counter block tabulation from PCG64 draws', 'tanh neural forward', 'convex KKT and objective',
                'sorted linear order-statistic quantiles', 'all cell bands, missingness and summaries'],
              'shared_dependencies': ['hash-locked PR28/29/31 audit helpers', 'NumPy/SciPy', 'rasterio/GDAL', 'shapely/GEOS', 'scikit-learn BallTree'],
              'outer_labels_used_for_bands': False, 'nominal_coverage_validated': False, 'fossil_application_approved': False}
    report.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['previous', 'maps', 'climate', 'config', 'output', 'report']:
        parser.add_argument('--'+key, required=True, type=Path)
    a = parser.parse_args()
    verify(a.previous, a.maps, a.climate, a.config, a.output, a.report)
