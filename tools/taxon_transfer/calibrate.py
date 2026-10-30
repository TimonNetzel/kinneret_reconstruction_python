"""Locked seven-species calibration, independently auditable modern-only outputs."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import platform
import zipfile

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import scipy
from scipy.spatial import cKDTree
import shapely
from shapely.geometry import shape
from shapely.ops import unary_union

from model import fit, probabilities, require


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_csv(path, records):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)


def source_gate(maps, climate, sources):
    for source in sources['map_files']:
        p = maps / source['path']
        require(p.stat().st_size == source['bytes'] and sha(p) == source['sha256'], 'Map source changed: '+source['path'])
    record = sources['climate_file']
    require(climate.stat().st_size == record['bytes'] and sha(climate) == record['sha256'], 'Climate archive changed')


def climate_grid(path, config, sources):
    d = config['geographic_domain']
    grid = config['grid_degrees']
    lon = np.arange(d['west']+grid/2, d['east'], grid)
    lat = np.arange(d['north']-grid/2, d['south'], -grid)
    xx, yy = np.meshgrid(lon, lat)
    arrays = []
    with zipfile.ZipFile(path) as archive:
        require(archive.testzip() is None, 'Climate ZIP CRC failure')
        for name in ('wc2.1_10m_bio_1.tif', 'wc2.1_10m_bio_12.tif'):
            b = archive.read(name)
            lock = sources['worldclim_members'][name]
            require(len(b) == lock['bytes'] and hashlib.sha256(b).hexdigest() == lock['sha256'], 'TIFF hash differs')
            with rasterio.io.MemoryFile(b) as memory:
                with memory.open() as ds:
                    require(ds.crs.to_epsg() == 4326 and ds.scales == (1.0,) and ds.offsets == (0.0,), 'Climate CRS/scaling differs')
                    require(ds.width == 2160 and ds.height == 1080, 'Climate layout differs')
                    rr, cc = rasterio.transform.rowcol(ds.transform, xx.ravel(), yy.ravel())
                    values = ds.read(1, masked=True)[np.asarray(rr), np.asarray(cc)]
                    arrays.append(values.filled(np.nan).astype(float))
    x = np.column_stack(arrays)
    valid = np.isfinite(x).all(axis=1)
    require(((x[valid, 0] >= -100) & (x[valid, 0] <= 80) & (x[valid, 1] >= 0)).all(), 'Implausible climate units')
    return xx, yy, x, valid, from_origin(d['west'], d['north'], grid, grid)


def range_labels(path, taxon, config, grid_shape, transform):
    raw = json.loads(path.read_text())
    require(raw['type'] == 'FeatureCollection', 'Not a GeoJSON FeatureCollection')
    if 'crs' in raw:
        require(raw['crs']['properties']['name'] == 'urn:ogc:def:crs:OGC:1.3:CRS84', 'Unexpected map CRS')
    native, points, boundaries, unknown = [], [], [], []
    statuses = collections.Counter()
    for feature in raw['features']:
        p = feature['properties']
        require(p['taxon_id'] == taxon['taxon_id'] and p['name_use_id_orig'] == taxon['name_use_id_orig']
                and p['name_choro_orig'] == taxon['source_name'], 'Original taxon concept differs')
        geom = shape(feature['geometry'])
        require(not geom.is_empty and geom.is_valid and geom.geom_type in ('Point', 'Polygon', 'MultiPolygon'), 'Invalid or unsupported original geometry; no repair')
        require(-180 <= geom.bounds[0] <= geom.bounds[2] <= 180 and -90 <= geom.bounds[1] <= geom.bounds[3] <= 90, 'Map coordinates differ')
        require(p['occurrence_status'] in ('Vorkommen', 'fragliches Vorkommen', 'erloschenes Vorkommen')
                and p['occurrence_origin'] in ('nativ', 'synanthrop'), 'Unknown source status')
        statuses[p['occurrence_status']+'|'+p['occurrence_origin']+'|'+geom.geom_type] += 1
        if p['occurrence_status'] == 'Vorkommen' and p['occurrence_origin'] == 'nativ':
            native.append(geom)
            if geom.geom_type == 'Point':
                points.append(geom)
            else:
                boundaries.append(geom.boundary)
        else:
            unknown.append(geom)
    require(native and boundaries, 'No usable native polygon')
    def burn(geometries):
        if not geometries:
            return np.zeros(grid_shape, bool)
        return rasterize([(g, 1) for g in geometries], out_shape=grid_shape, transform=transform,
                         fill=0, all_touched=False, dtype='uint8').astype(bool)
    rules = config['labels']
    y = burn(native)
    point_cells = burn(points)
    boundary_mask = burn([g.buffer(rules['polygon_boundary_exclusion_degrees']) for g in boundaries])
    point_mask = burn([g.buffer(rules['point_neighbourhood_exclusion_degrees']) for g in points])
    unknown_mask = burn([g.buffer(rules['other_status_or_origin_exclusion_degrees']) for g in unknown])
    use = ~boundary_mask & ~unknown_mask & (~point_mask | point_cells)
    return y.ravel().astype(int), use.ravel(), {'source_feature_status_counts': dict(statuses),
        'source_features': len(raw['features']), 'confirmed_native_point_cells_in_domain': int(point_cells.sum()),
        'boundary_mask_cells': int(boundary_mask.sum()), 'point_neighbourhood_mask_cells': int((point_mask & ~point_cells).sum()),
        'uncertain_origin_status_mask_cells': int(unknown_mask.sum())}


def xyz(lon, lat):
    lo, la = np.deg2rad(lon), np.deg2rad(lat)
    return np.column_stack([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)])


def splits(lon, lat, block, settings):
    bx, by = np.floor((lon+180)/block).astype(int), np.floor((lat+90)/block).astype(int)
    folds = (by+2*bx) % settings['folds']
    unit = xyz(lon, lat)
    threshold = 2*np.sin(settings['buffer_km']/(2*settings['sphere_radius_km']))
    for fold in range(settings['folds']):
        test = folds == fold
        require(test.any(), 'Empty spatial fold')
        distance = cKDTree(unit[test]).query(unit, k=1)[0]
        train = ~test & (distance > threshold)
        require(train.any(), 'Empty buffered training set')
        closest = cKDTree(unit[train]).query(unit[test], k=1)[0].min()
        km = 2*settings['sphere_radius_km']*np.arcsin(min(1, closest/2))
        yield fold, train, test, {'train_cells': int(train.sum()), 'test_cells': int(test.sum()),
            'buffered_cells': int((~test & ~train).sum()), 'minimum_train_test_km': float(km)}


def score(y, p, base, w):
    w = w/w.sum()
    brier = float(w @ (y-p)**2)
    comparator = float(w @ (y-base)**2)
    safe = np.clip(p, 1e-12, 1-1e-12)
    b = np.clip(base, 1e-12, 1-1e-12)
    return {'brier': brier, 'comparator_brier': comparator,
            'brier_skill': 1-brier/comparator if comparator else None,
            'log_loss': float(-w @ (y*np.log(safe)+(1-y)*np.log1p(-safe))),
            'comparator_log_loss': float(-w @ (y*np.log(b)+(1-y)*np.log1p(-b))),
            'observed_prevalence': float(w @ y), 'mean_prediction': float(w @ p)}


def run(maps, climate, config_path, sources_path, output):
    require(not output.exists(), 'Use a new result directory')
    config, sources = json.loads(config_path.read_text()), json.loads(sources_path.read_text())
    source_gate(maps, climate, sources)
    xx, yy, climate_values, land, transform = climate_grid(climate, config, sources)
    mapping = list(csv.DictReader((maps/'Data_in/Areale/Mapping_areale.csv').open()))
    output.mkdir(parents=True)
    final_models, fold_models, scores, roles, inventory, reliability = {}, [], [], [], [], []
    pred_fields = ['taxon', 'block_degrees', 'cell_index', 'fold', 'label', 'prediction', 'comparator', 'in_training_climate_hull']
    with (output/'predictions.csv').open('w', newline='') as handle:
        predictions = csv.DictWriter(handle, fieldnames=pred_fields, lineterminator='\n')
        predictions.writeheader()
        for taxon in config['taxa']:
            match = [r for r in mapping if r['name_choro_orig'] == taxon['source_name'] and int(r['taxon_id']) == taxon['taxon_id'] and int(r['name_use_id_orig']) == taxon['name_use_id_orig']]
            require(len(match) == 1, 'Ambiguous taxon mapping')
            labels, mask, evidence = range_labels(maps/f"Data_in/Areale/geojson/taxon_id_{taxon['taxon_id']}.geojson", taxon, config, xx.shape, transform)
            chosen = np.flatnonzero(land & mask)
            x, y = climate_values[chosen], labels[chosen]
            lon, lat = xx.ravel()[chosen], yy.ravel()[chosen]
            weights = np.cos(np.deg2rad(lat))
            inventory.append({'taxon': taxon['id'], 'name': taxon['name'], 'source_name': taxon['source_name'],
                'bfn_taxon_id': taxon['taxon_id'], 'bfn_name_use_id_orig': taxon['name_use_id_orig'],
                'land_cells_before_map_mask': int(land.sum()), 'retained_cells': len(y),
                'positive_cells': int(y.sum()), 'mapped_outside_cells': int((y == 0).sum()),
                'source_evidence': json.dumps(evidence, sort_keys=True)})
            print(taxon['id'], 'cells', len(y), 'positive', int(y.sum()), flush=True)
            for block in config['validation']['block_sizes_degrees']:
                all_p, all_base, all_inside = np.empty(len(y)), np.empty(len(y)), np.zeros(len(y), bool)
                for fold, train, test, role in splits(lon, lat, block, config['validation']):
                    model = fit(x[train], y[train], weights[train], config['model'])
                    model.update(taxon=taxon['id'], block_degrees=block, fold=fold)
                    fold_models.append(model)
                    p, inside = probabilities(model, x[test], reject_extrapolation=False)
                    base = np.full(test.sum(), model['training_prevalence'])
                    all_p[test], all_base[test], all_inside[test] = p, base, inside
                    scores.append({'taxon': taxon['id'], 'block_degrees': block, 'scope': str(fold),
                        'test_cells': int(test.sum()), 'outside_hull_cells': int((~inside).sum()),
                        **score(y[test], p, base, weights[test])})
                    roles.append({'taxon': taxon['id'], 'block_degrees': block, 'fold': fold, **role})
                    for j, pp, bb, inn in zip(np.flatnonzero(test), p, base, inside):
                        predictions.writerow(dict(zip(pred_fields, [taxon['id'], block, int(chosen[j]), fold,
                            int(y[j]), float(pp), float(bb), bool(inn)])))
                scores.append({'taxon': taxon['id'], 'block_degrees': block, 'scope': 'all',
                    'test_cells': len(y), 'outside_hull_cells': int((~all_inside).sum()),
                    **score(y, all_p, all_base, weights)})
                bins = np.minimum((all_p*10).astype(int), 9)
                for bin_index in range(10):
                    sel = bins == bin_index
                    reliability.append({'taxon': taxon['id'], 'block_degrees': block, 'bin': bin_index,
                        'cells': int(sel.sum()), 'area_weight': float(weights[sel].sum()/weights.sum()),
                        'mean_prediction': float(np.average(all_p[sel], weights=weights[sel])) if sel.any() else None,
                        'observed_fraction': float(np.average(y[sel], weights=weights[sel])) if sel.any() else None})
            model = fit(x, y, weights, config['model'])
            model.update(taxon=taxon, variant=config['variant'], climate=config['climate'],
                         inference=config['inference'])
            final_models[taxon['id']] = model
    write_json(output/'models.json', final_models)
    write_json(output/'fold_models.json', fold_models)
    write_csv(output/'scores.csv', scores)
    write_csv(output/'split_inventory.csv', roles)
    write_csv(output/'sample_inventory.csv', inventory)
    write_csv(output/'reliability.csv', reliability)
    manifest = {'variant': config['variant'], 'config_sha256': sha(config_path),
        'sources_sha256': sha(sources_path), 'code_sha256': {p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name('model.py')]},
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                        'shapely': shapely.__version__, 'rasterio': rasterio.__version__},
        'output_sha256': {p.name: sha(p) for p in sorted(output.iterdir())},
        'fossil_predictions_created': False, 'p1_released': False}
    write_json(output/'manifest.json', manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('maps', 'climate', 'config', 'sources', 'output'):
        parser.add_argument('--'+key, required=True, type=Path)
    a = parser.parse_args()
    run(a.maps, a.climate, a.config, a.sources, a.output)
