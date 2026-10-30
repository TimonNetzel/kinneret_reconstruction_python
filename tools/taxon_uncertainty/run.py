"""Fixed-network block-bootstrap calibration sensitivity on all PR31 spatial cases."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import csv
import gzip
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import platform
import sys

import numpy as np
import scipy

from bootstrap import block_counts, fit_replicates, propagate, calibration

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'taxon_smote'))
from compare import load_previous


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_out(path, value):
    temp = path.with_name(path.name+'.part')
    with temp.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    temp.rename(path)


def gz_writer(stack, path, fields):
    raw = stack.enter_context(path.with_name(path.name+'.part').open('xb'))
    gz = stack.enter_context(gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0))
    text = stack.enter_context(io.TextIOWrapper(gz, encoding='utf-8', newline=''))
    writer = csv.DictWriter(text, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    return writer


def run(previous, maps, climate, config_path, output):
    if output.exists():
        raise ValueError('Use a new output directory')
    config = json.loads(config_path.read_text())
    root = Path(__file__).resolve().parents[2]
    for file, digest in config['locked_sha256'].items():
        if sha(root/file) != digest:
            raise ValueError('Locked PR31 input changed: '+file)
    d31 = config_path.parent.parent/'calibration_2026_10_30'
    old31 = json.loads((d31/'config.json').read_text())
    if config['quantiles'] != [.025, .5, .975] or config['quantile_method'] != 'linear' or config['minimum_unique_active_class'] != old31['calibration']['minimum_class']:
        raise ValueError('Unsupported deviation from predeclared quantile/class recipe')
    d29 = d31.parent/'smote_nnet_2026_10_30'
    for file, digest in old31['locked_inputs_sha256'].items():
        if sha(d29/file) != digest:
            raise ValueError('Locked PR29 source chain changed: '+file)
    fits = {(m['case_id'], m['method']): m for m in json.loads((d31/'fit_models.json').read_text())}
    source = json.loads((d29/'sources.json').read_text())
    _, pc, old, raw_source = load_previous(previous, source)
    pc.source_gate(maps, climate, raw_source)
    xx, yy, all_x, land, transform = pc.climate_grid(climate, old, raw_source)
    output.mkdir(parents=True)
    summaries, inventory, full = [], [], {}
    B = config['replicates']
    param_fields = ['case', 'family', 'inner', 'replicate', 'status', 'negative', 'positive', 'slope', 'intercept', 'objective', 'projected_gradient']
    draw_fields = ['case', 'replicate', 'latitude_block', 'longitude_block', 'multiplicity']
    pred_fields = ['taxon', 'block_degrees', 'outer', 'family', 'cell_index', 'support_members', 'valid_replicates', 'status', 'point', 'lower', 'median', 'upper']
    with ExitStack() as stack:
        workers = stack.enter_context(ProcessPoolExecutor(max_workers=6, mp_context=multiprocessing.get_context('spawn')))
        parameters = gz_writer(stack, output/'parameters.csv.gz', param_fields)
        draws = gz_writer(stack, output/'block_draws.csv.gz', draw_fields)
        predictions = gz_writer(stack, output/'intervals.csv.gz', pred_fields)
        for taxon in old['taxa']:
            labels, use, _ = pc.range_labels(maps/f"Data_in/Areale/geojson/taxon_id_{taxon['taxon_id']}.geojson", taxon, old, xx.shape, transform)
            selected = np.flatnonzero(land & use)
            x, y = all_x[selected], labels[selected]
            lon, lat = xx.ravel()[selected], yy.ravel()[selected]
            area = np.cos(np.deg2rad(lat))
            full[taxon['id']] = {}
            for block in old['validation']['block_sizes_degrees']:
                full[taxon['id']][str(block)] = {}
                cases = [(str(f), tr, te) for f, tr, te, _ in pc.splits(lon, lat, block, old['validation'])]
                cases.append(('full', np.ones(len(y), bool), np.ones(len(y), bool)))
                for outer, train_mask, eval_mask in cases:
                    case = f"{taxon['id']}:b{block}:o{outer}"
                    pool = np.flatnonzero(train_mask)
                    ids, inverse, counts, seed = block_counts(lon[pool], lat[pool], block, B, config['master_seed'], case)
                    for rep in range(B):
                        for j in np.flatnonzero(counts[rep]):
                            draws.writerow(dict(zip(draw_fields, [case, rep, int(ids[j, 0]), int(ids[j, 1]), int(counts[rep, j])])))
                    members = {f: [fits[case+f':i{j}', f] for j in range(3)] for f in ['original', 'smote']}
                    available = {f: all(m['calibrator']['status'] == 'fitted' for m in members[f]) for f in members}
                    pending, metadata = {}, []
                    for inner, _, cal, info in pc.splits(lon[pool], lat[pool], block, {**old['validation'], 'folds': 3}):
                        cal_ids = pool[cal]
                        cal_blocks = inverse[cal]
                        positive_blocks = len(np.unique(cal_blocks[y[cal_ids] == 1]))
                        metadata.append({'calibration_cells': len(cal_ids), 'positive': int(y[cal_ids].sum()),
                                         'positive_blocks': positive_blocks, 'calibration_blocks': len(np.unique(cal_blocks))})
                        for family in members:
                            eta = calibration.logits(members[family][inner], x[cal_ids])
                            pending[family, inner] = workers.submit(fit_replicates, eta, y[cal_ids], area[cal_ids], counts[:, cal_blocks],
                                                                   available[family], old31['calibration'])
                    inventory.append({'case': case, 'blocks': len(ids), 'pool_cells': len(pool), 'seed': seed, 'members': metadata})
                    for family in members:
                        results = [pending[family, j].result() for j in range(3)]
                        valid = np.array([all(results[j][r]['status'] == 'fitted' for j in range(3)) for r in range(B)])
                        coefficients = [[[results[j][r]['slope'], results[j][r]['intercept']] for j in range(3)] for r in range(B)]
                        for j in range(3):
                            for rep, record in enumerate(results[j]):
                                parameters.writerow({'case': case, 'family': family, 'inner': j, 'replicate': rep, **record})
                        point, band, support = propagate(members[family], coefficients, valid, x[eval_mask], config['minimum_valid_replicates'], False)
                        status = 'available_conditional_band' if band is not None else ('base_member_uncalibrated' if not available[family] else 'insufficient_estimable_replicates')
                        for k, cell in enumerate(selected[eval_mask]):
                            predictions.writerow(dict(zip(pred_fields, [taxon['id'], block, outer, family, int(cell), int(support[k]), int(valid.sum()), status,
                                                                       float(point[k])]+([float(v[k]) for v in band] if band is not None else [None]*3))))
                        width = band[2]-band[0] if band is not None else None
                        failures = {s: sum(rec['status'] == s for group in results for rec in group) for s in
                                    ['fitted', 'insufficient_unique_class', 'numerical_failure', 'base_member_uncalibrated']}
                        summaries.append({'taxon': taxon['id'], 'block_degrees': block, 'outer': outer, 'family': family,
                                          'status': status, 'valid_replicates': int(valid.sum()), 'total_replicates': B, 'evaluation_cells': int(eval_mask.sum()),
                                          'cells_outside_any_hull': int((support < 3).sum()), 'mean_width_area_weighted': float(np.average(width, weights=area[eval_mask])) if band is not None else None,
                                          **{f'width_unweighted_q{q}': float(np.quantile(width, q/100)) if band is not None else None for q in [5, 50, 95]},
                                          **{'member_replicates_'+k: v for k, v in failures.items()}})
                        if outer == 'full':
                            full[taxon['id']][str(block)][family] = {'minimum_valid': config['minimum_valid_replicates'], 'valid': valid.tolist(),
                                'status': status, 'coefficients': coefficients, 'conditional_fixed_networks': True, 'nominal_coverage_validated': False}
                    print('bootstrap', case, {f: sum(all(pending[f, j].result()[r]['status'] == 'fitted' for j in range(3)) for r in range(B)) for f in members}, flush=True)
    # Closed streams first; never publish running/checkpoint files as complete outputs.
    for name in ['parameters.csv.gz', 'block_draws.csv.gz', 'intervals.csv.gz']:
        (output/(name+'.part')).rename(output/name)
    json_out(output/'full_parameters.json', full)
    json_out(output/'inventory.json', inventory)
    with (output/'summary.csv.part').open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(summaries)
    (output/'summary.csv.part').rename(output/'summary.csv')
    json_out(output/'manifest.json', {'variant': config['variant'], 'base_commit': config['base_commit'], 'replicates': B,
        'design_commit': '5a4a96f1d7c01b64a230fb8ad6a7fbaf84857697', 'config_sha256': sha(config_path), 'design_sha256': sha(config_path.with_name('DESIGN.md')),
        'source_model_sha256': config['locked_sha256'], 'code_sha256': {str(p.relative_to(root)): sha(p) for p in [Path(__file__).resolve(), Path(__file__).with_name('bootstrap.py').resolve()]},
        'outputs_sha256': {p.name: sha(p) for p in sorted(output.iterdir())},
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
        'outer_labels_used_for_bands': False, 'nominal_coverage_validated': False, 'fossil_application_approved': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['previous', 'maps', 'climate', 'config', 'output']:
        parser.add_argument('--'+key, required=True, type=Path)
    a = parser.parse_args()
    run(a.previous, a.maps, a.climate, a.config, a.output)
