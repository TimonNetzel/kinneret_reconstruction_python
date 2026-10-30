"""Conditional spatial block sensitivity; fixed networks, no nominal coverage claim."""
import hashlib
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.spatial import Delaunay

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'taxon_calibration'))
import model as calibration


def block_counts(lon, lat, block, replicates, master, case):
    ids, inverse = np.unique(np.stack([np.floor((lat+90)/block).astype(int),
                                      np.floor((lon+180)/block).astype(int)], axis=1), axis=0, return_inverse=True)
    seed = int(hashlib.sha256(f'{master}|{case}|block_bootstrap'.encode()).hexdigest()[:8], 16)
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = rng.integers(len(ids), size=(replicates, len(ids)))
    counts = np.stack([np.bincount(row, minlength=len(ids)) for row in draws])
    return ids, inverse, counts, seed


def fit_replicates(eta, y, area, multiplicities, base_available, settings):
    records = []
    for counts in multiplicities:
        active = counts > 0
        unique = np.bincount(y[active], minlength=2)
        record = dict(negative=int(unique[0]), positive=int(unique[1]), slope=None,
                      intercept=None, objective=None, projected_gradient=None)
        if not base_available:
            record['status'] = 'base_member_uncalibrated'
        elif unique.min() < settings['minimum_class']:
            record['status'] = 'insufficient_unique_class'
        else:
            e, yy, w = eta[active], y[active], area[active]*counts[active]
            result = minimize(calibration.objective, settings['initial'],
                              args=(e, yy, w, settings['ridge_slope']), method='L-BFGS-B', jac=True,
                              bounds=[(0, None), (None, None)],
                              options={'maxiter': settings['maxiter'], 'gtol': 1e-10, 'ftol': 1e-15, 'maxls': 50})
            loss, gradient = calibration.objective(result.x, e, yy, w, settings['ridge_slope'])
            pg = calibration.projected_gradient(result.x, gradient)
            finite = bool(np.isfinite(result.x).all() and np.isfinite(loss) and np.isfinite(pg))
            record.update(status='fitted' if finite and pg <= settings['projected_gradient_tolerance'] else 'numerical_failure',
                          slope=float(result.x[0]) if np.isfinite(result.x[0]) else None,
                          intercept=float(result.x[1]) if np.isfinite(result.x[1]) else None,
                          objective=loss if np.isfinite(loss) else None,
                          projected_gradient=pg if np.isfinite(pg) else None)
        records.append(record)
    return records


def propagate(members, coefficients, valid, x, minimum, reject_extrapolation=True):
    support = sum(Delaunay(np.asarray(m['climate_hull'])).find_simplex(x) >= 0 for m in members)
    if reject_extrapolation and (support < 3).any():
        raise ValueError('Climate outside a member training hull')
    point = np.mean([expit(m['calibrator']['slope']*calibration.logits(m, x)+m['calibrator']['intercept']) for m in members], axis=0)
    if np.sum(valid) < minimum:
        return point, None, support
    c = np.asarray(coefficients, float)[valid]
    values = np.zeros((len(c), len(x)))
    for j, member in enumerate(members):
        eta = calibration.logits(member, x)
        values += expit(c[:, j, 0, None]*eta+c[:, j, 1, None])/3
    return point, np.quantile(values, [.025, .5, .975], axis=0, method='linear'), support


def predict_interval(members, record, climate, reject_extrapolation=True):
    """Point and conditional 2.5/50/97.5% sensitivity quantiles; never a fossil CI."""
    x = np.asarray(climate, float)
    point, interval, support = propagate(members, record['coefficients'], np.array(record['valid'], bool),
                                         x, record['minimum_valid'], reject_extrapolation)
    if interval is None:
        raise ValueError('Conditional calibration interval not estimable under fixed gate')
    return point, interval, support
