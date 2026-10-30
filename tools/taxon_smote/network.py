"""Independent Python implementation of a binary, single-hidden-layer nnet class.

Sources describe the architecture and objective; no R source code is copied.
SMOTE samples are training-only derivations, not new geographic observations.
"""
import hashlib
import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import digamma, expit, gammainc, ndtri
from scipy.spatial import cKDTree, Delaunay


def require(ok, message):
    if not ok:
        raise ValueError(message)


def derived_seed(master, case, component):
    return int(hashlib.sha256(f'{master}|{case}|{component}'.encode()).hexdigest()[:8], 16)


def normalize_fit(climate, weights, clip):
    x, w = np.asarray(climate, float), np.asarray(weights, float)
    require(x.ndim == 2 and x.shape[1] == 2 and w.shape == (len(x),), 'Climate dimensions differ')
    require(np.isfinite(x).all() and np.isfinite(w).all() and (w > 0).all() and (x[:, 1] >= 0).all(), 'Invalid original training values')
    mean = np.average(x[:, 0], weights=w)
    sd = np.sqrt(np.average((x[:, 0]-mean)**2, weights=w))
    positive = x[:, 1] > 0
    require(sd > 0 and positive.sum() >= 2, 'Degenerate original climate')
    v = np.sqrt(x[positive, 1])
    mw = np.average(v, weights=w[positive])
    s = np.log(mw)-np.average(np.log(v), weights=w[positive])
    require(s > 0, 'Degenerate gamma fit')
    shape = brentq(lambda a: np.log(a)-digamma(a)-s, 1e-8, 1e8, xtol=1e-12)
    return {'temperature_mean': float(mean), 'temperature_sd': float(sd),
            'gamma_shape': float(shape), 'gamma_rate': float(shape/mw), 'cdf_clip': clip,
            'positive_precipitation_training_cells': int(positive.sum()),
            'fit_on_original_training_only': True}


def normalize(climate, params):
    x = np.asarray(climate, float)
    require(x.ndim == 2 and x.shape[1] == 2 and np.isfinite(x).all() and (x[:, 1] >= 0).all(), 'Expected finite N x 2 TANN/PANN')
    cdf = gammainc(params['gamma_shape'], np.sqrt(x[:, 1])*params['gamma_rate'])
    return np.column_stack([(x[:, 0]-params['temperature_mean'])/params['temperature_sd'],
                            ndtri(np.clip(cdf, params['cdf_clip'], 1-params['cdf_clip']))])


def make_smote(x, y, weights, neighbors, seed):
    x, y, weights = np.asarray(x, float), np.asarray(y, int), np.asarray(weights, float)
    require(x.shape == (len(y), 2) and weights.shape == y.shape and np.isfinite(x).all(), 'SMOTE shape differs')
    require(np.isin(y, [0, 1]).all() and (weights > 0).all(), 'Invalid original SMOTE values')
    counts = np.bincount(y, minlength=2)
    label = int(np.argmin(counts))
    minority = np.flatnonzero(y == label)
    require(len(minority) > neighbors, 'Insufficient minority neighbors')
    count = int(counts.max()-counts.min())
    distances, nearest = cKDTree(x[minority]).query(x[minority], k=neighbors+1)
    # Exclude self by row ID, including when several rows have identical climate.
    lookup = np.array([row[row != i][:neighbors] for i, row in enumerate(nearest)], dtype=int)
    require(lookup.shape == (len(minority), neighbors), 'Invalid neighbor table')
    rng = np.random.Generator(np.random.PCG64(seed))
    base_local = rng.integers(len(minority), size=count)
    slot = rng.integers(neighbors, size=count)
    base = minority[base_local]
    other = minority[lookup[base_local, slot]]
    fraction = rng.random(count)
    synth = x[base]+fraction[:, None]*(x[other]-x[base])
    sw = weights[base]+fraction*(weights[other]-weights[base])
    return (np.concatenate([x, synth]), np.concatenate([y, np.full(count, label)]),
            np.concatenate([weights, sw]), {'base': base, 'other': other, 'fraction': fraction,
            'label': label, 'synthetic_count': count})


def unpack(theta, hidden):
    a = hidden*2
    return theta[:a].reshape(2, hidden), theta[a:a+hidden], theta[a+hidden:a+2*hidden], theta[-1]


def objective(theta, x, y, weights, hidden, decay):
    ih, hb, ho, ob = unpack(theta, hidden)
    activation = expit(x @ ih+hb)
    logits = activation @ ho+ob
    loss = np.sum(weights*(np.logaddexp(0, logits)-y*logits))+decay*(theta @ theta)
    delta = weights*(expit(logits)-y)
    dh = delta[:, None]*ho*activation*(1-activation)
    gradient = np.concatenate([(x.T @ dh).ravel(), dh.sum(axis=0), activation.T @ delta,
                               np.array([delta.sum()])])+2*decay*theta
    return float(loss), gradient


def fit_network(x, y, weights, settings, master, case):
    require(np.isfinite(x).all() and np.isfinite(weights).all() and (weights > 0).all(), 'Nonfinite fit')
    require(np.isin(y, [0, 1]).all() and len(np.unique(y)) == 2, 'Binary fit requires both classes')
    w = weights/np.mean(weights)
    hidden, decay = settings['hidden_units'], settings['weight_decay']
    starts = []
    for j in range(settings['initializations']):
        seed = derived_seed(master, case, f'initialization:{j}')
        initial = np.random.Generator(np.random.PCG64(seed)).uniform(-settings['initial_range'], settings['initial_range'], 4*hidden+1)
        result = minimize(objective, initial, args=(x, y, w, hidden, decay), method='BFGS', jac=True,
                          options={'maxiter': settings['maxiter'], 'gtol': settings['raw_gradient_tolerance']})
        loss, gradient = objective(result.x, x, y, w, hidden, decay)
        criterion = float(np.max(np.abs(gradient))/w.sum())
        stationary = bool(np.isfinite(result.x).all() and np.isfinite(loss) and criterion <= settings['accepted_gradient_per_effective_case'])
        starts.append({'seed': seed, 'weights': result.x.tolist(), 'objective': loss,
                       'gradient_per_case': criterion, 'stationary': stationary,
                       'optimizer_success': bool(result.success), 'optimizer_status': int(result.status),
                       'optimizer_iterations': int(result.nit)})
    accepted = [j for j, value in enumerate(starts) if value['stationary']]
    require(accepted, 'All predetermined network starts failed stationarity gate')
    chosen = min(accepted, key=lambda j: (starts[j]['objective'], j))
    return {'hidden_units': hidden, 'weights': starts[chosen]['weights'], 'selected_start': chosen,
            'starts': starts, 'resampled_case_count': len(y), 'resampled_case_weighted_prior': float(np.average(y, weights=weights)),
            'weight_decay': decay, 'loss_weight_normalization': 'mean case weight equals 1',
            'network_architecture': '2 inputs; one sigmoid hidden layer; one sigmoid output; no skip',
            'r_training_bitwise_reproduction': False, 'fossil_mapping_approved': False}


def predict(model, climate, prior_adjusted=False, reject_extrapolation=True):
    x = np.asarray(climate, float)
    z = normalize(x, model['normalization'])
    ih, hb, ho, ob = unpack(np.asarray(model['weights']), model['hidden_units'])
    eta = expit(z @ ih+hb) @ ho+ob
    if prior_adjusted:
        original, trained = model['original_area_weighted_prior'], model['resampled_case_weighted_prior']
        require(0 < original < 1 and 0 < trained < 1, 'Invalid training priors')
        eta = eta+np.log(original/(1-original))-np.log(trained/(1-trained))
    inside = Delaunay(np.asarray(model['climate_hull'])).find_simplex(x) >= 0
    require(not reject_extrapolation or inside.all(), 'Climate outside original training hull')
    return expit(eta), inside
