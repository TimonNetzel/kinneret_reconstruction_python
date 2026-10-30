"""Forward botanical range response; no fossil reconstruction or taxon aggregation."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, Delaunay
from scipy.special import expit


def require(ok, message):
    if not ok:
        raise ValueError(message)


def design(climate, mean, scale):
    x = np.asarray(climate, dtype=float)
    require(x.ndim == 2 and x.shape[1] == 2 and np.isfinite(x).all(), 'Expected finite N x 2 TANN/PANN')
    require((x[:, 1] >= 0).all(), 'Negative annual precipitation')
    scale = np.asarray(scale, dtype=float)
    require(scale.shape == (2,) and (scale > 0).all(), 'Degenerate climate scale')
    t, p = ((x - mean) / scale).T
    return np.column_stack([np.ones(len(x)), t, p, t*p, t*t, p*p])


def objective(beta, matrix, labels, weights, ridge):
    w = weights / weights.sum()
    eta = matrix @ beta
    penalty = np.array(beta, copy=True)
    penalty[0] = 0
    loss = np.sum(w * (np.logaddexp(0, eta) - labels * eta)) + ridge/2 * (penalty @ penalty)
    gradient = matrix.T @ (w * (expit(eta) - labels)) + ridge * penalty
    return float(loss), gradient


def fit(climate, labels, weights, options):
    x, y, w = np.asarray(climate, float), np.asarray(labels, float), np.asarray(weights, float)
    require(y.shape == w.shape == (len(x),), 'Fit dimensions differ')
    require(np.isfinite(x).all() and np.isfinite(w).all() and (w > 0).all(), 'Invalid training values')
    require(np.isin(y, [0, 1]).all(), 'Only explicit binary map labels allowed')
    require(min(np.sum(y == 0), np.sum(y == 1)) >= options['minimum_training_cells_per_class'], 'Insufficient training class')
    mean = np.average(x, axis=0, weights=w)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=w))
    matrix = design(x, mean, scale)
    prevalence = float(np.average(y, weights=w))
    initial = np.zeros(6)
    initial[0] = np.log(prevalence/(1-prevalence))
    result = minimize(objective, initial, args=(matrix, y, w, options['ridge_lambda']),
                      jac=True, method='L-BFGS-B', options={'maxiter': options['maxiter'],
                      'gtol': options['gradient_tolerance'], 'ftol': 1e-13, 'maxls': 50})
    loss, grad = objective(result.x, matrix, y, w, options['ridge_lambda'])
    require(result.success and np.max(np.abs(grad)) <= options['accepted_gradient_max'], 'Optimizer failed convergence gate')
    hull = x[ConvexHull(x).vertices]
    return {'mean': mean.tolist(), 'scale': scale.tolist(), 'coefficients': result.x.tolist(),
            'climate_hull': hull.tolist(), 'training_prevalence': prevalence,
            'training_cells': len(x), 'training_positive_cells': int(y.sum()),
            'ridge_lambda': options['ridge_lambda'], 'objective': loss,
            'gradient_max': float(np.max(np.abs(grad))), 'optimizer_iterations': int(result.nit),
            'target': 'confirmed_native_mapped_range_indicator',
            'probability_is_pollen_detection': False, 'fossil_mapping_approved': False}


def probabilities(model, climate, reject_extrapolation=True):
    matrix = design(climate, model['mean'], model['scale'])
    inside = Delaunay(np.asarray(model['climate_hull'])).find_simplex(np.asarray(climate)) >= 0
    require(not reject_extrapolation or inside.all(), 'Climate outside training convex hull')
    return expit(matrix @ np.asarray(model['coefficients'])), inside
