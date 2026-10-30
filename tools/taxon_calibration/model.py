"""Fixed monotone logistic recalibration; paired spatial member ensembles."""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.spatial import Delaunay

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'taxon_smote'))
from network import normalize, require, unpack


def logits(model, climate):
    z = normalize(climate, model['normalization'])
    ih, hb, ho, ob = unpack(np.asarray(model['weights']), model['hidden_units'])
    return expit(z @ ih+hb) @ ho+ob


def objective(theta, eta, y, weights, ridge):
    a, b = theta
    z = a*eta+b
    w = weights/weights.sum()
    residual = w*(expit(z)-y)
    return (float(w @ (np.logaddexp(0, z)-y*z)+ridge*a*a/2),
            np.array([residual @ eta+ridge*a, residual.sum()]))


def projected_gradient(theta, gradient):
    g = np.array(gradient, copy=True)
    if theta[0] == 0 and g[0] > 0:
        g[0] = 0
    return float(np.max(np.abs(g)))


def fit_calibrator(eta, y, weights, settings):
    eta, y, weights = np.asarray(eta), np.asarray(y), np.asarray(weights)
    require(eta.shape == y.shape == weights.shape and len(y) > 0, 'Calibration shape')
    require(np.isfinite(eta).all() and np.isfinite(weights).all() and (weights > 0).all()
            and np.isin(y, [0, 1]).all(), 'Invalid calibration data')
    counts = np.bincount(y.astype(int), minlength=2)
    if counts.min() < settings['minimum_class']:
        return {'slope': 1., 'intercept': 0., 'status': 'insufficient_calibration_class',
                'class_counts': counts.tolist(), 'objective': None, 'projected_gradient': None}
    result = minimize(objective, settings['initial'], args=(eta, y, weights, settings['ridge_slope']),
                      method='L-BFGS-B', jac=True, bounds=[(0, None), (None, None)],
                      options={'maxiter': settings['maxiter'], 'gtol': 1e-10, 'ftol': 1e-15, 'maxls': 50})
    loss, grad = objective(result.x, eta, y, weights, settings['ridge_slope'])
    pg = projected_gradient(result.x, grad)
    require(np.isfinite(result.x).all() and np.isfinite(loss) and
            pg <= settings['projected_gradient_tolerance'], 'Calibration stationarity failed')
    return {'slope': float(result.x[0]), 'intercept': float(result.x[1]), 'status': 'fitted',
            'class_counts': counts.tolist(), 'objective': loss, 'projected_gradient': pg,
            'optimizer_status': int(result.status), 'optimizer_iterations': int(result.nit)}


def predict(members, climate, calibrated=True, reject_extrapolation=True):
    require(len(members) == 3, 'Expected all three spatial members')
    x = np.asarray(climate, float)
    values, support = [], np.zeros(len(x), int)
    for member in members:
        eta = logits(member, x)
        support += Delaunay(np.asarray(member['climate_hull'])).find_simplex(x) >= 0
        if calibrated:
            c = member['calibrator']
            eta = c['slope']*eta+c['intercept']
        values.append(expit(eta))
    require(not reject_extrapolation or (support == len(members)).all(), 'Climate outside a member training hull')
    return np.mean(values, axis=0), support
