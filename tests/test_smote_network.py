"""Core network/SMOTE invariants; no GIS data or R installation needed."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy.special import expit
from scipy.stats import gamma, norm

path = Path(__file__).resolve().parents[1]/'tools/taxon_smote/network.py'
spec = importlib.util.spec_from_file_location('taxon_smote_network', path)
network = importlib.util.module_from_spec(spec)
spec.loader.exec_module(network)


class NetworkTests(unittest.TestCase):
    def test_all_network_derivatives_with_decay_and_case_weights(self):
        rng = np.random.default_rng(13)
        x, y = rng.normal(size=(19, 2)), rng.integers(0, 2, 19)
        w = rng.uniform(.2, 1, 19)
        theta = rng.uniform(-.7, .7, 25)
        loss, gradient = network.objective(theta, x, y, w, 6, .1)
        numerical = []
        for j in range(len(theta)):
            d = np.eye(len(theta))[j]*1e-6
            numerical.append((network.objective(theta+d, x, y, w, 6, .1)[0]-network.objective(theta-d, x, y, w, 6, .1)[0])/2e-6)
        self.assertTrue(np.isfinite(loss))
        np.testing.assert_allclose(gradient, numerical, atol=3e-8, rtol=1e-6)

    def test_training_gamma_transform_zero_and_negative_precipitation(self):
        x = np.array([[0, 0], [5, 100], [10, 200], [15, 1000], [20, 2000]], float)
        w = np.array([.2, .5, .7, .9, 1.])
        p = network.normalize_fit(x, w, 1e-7)
        z = network.normalize(x, p)
        np.testing.assert_allclose(p['temperature_mean'], np.average(x[:, 0], weights=w))
        np.testing.assert_allclose(z[:, 1], norm.ppf(np.clip(gamma.cdf(np.sqrt(x[:, 1]), a=p['gamma_shape'], scale=1/p['gamma_rate']), 1e-7, 1-1e-7)))
        self.assertTrue(np.isfinite(z).all())
        self.assertEqual(p['positive_precipitation_training_cells'], 4)
        with self.assertRaises(ValueError):
            network.normalize([[10, -1]], p)
        with self.assertRaises(ValueError):
            network.normalize_fit(np.full((5, 2), 1.), np.ones(5), 1e-7)

    def test_smote_preserves_originals_and_interpolates_only_minority_parents(self):
        rng = np.random.default_rng(31)
        x = rng.normal(size=(28, 2))
        y, w = np.r_[np.zeros(20, int), np.ones(8, int)], rng.uniform(.2, 1, 28)
        sx, sy, sw, trace = network.make_smote(x, y, w, 5, 5)
        np.testing.assert_array_equal(sx[:28], x)
        np.testing.assert_array_equal(sy[:28], y)
        np.testing.assert_array_equal(sw[:28], w)
        self.assertEqual(tuple(np.bincount(sy)), (20, 20))
        self.assertTrue(np.all(y[trace['base']] == 1))
        self.assertTrue(np.all(y[trace['other']] == 1))
        self.assertTrue(np.all(trace['base'] != trace['other']))
        u = trace['fraction']
        np.testing.assert_allclose(sx[28:], x[trace['base']]+u[:, None]*(x[trace['other']]-x[trace['base']]))
        np.testing.assert_allclose(sw[28:], w[trace['base']]+u*(w[trace['other']]-w[trace['base']]))
        again = network.make_smote(x, y, w, 5, 5)
        np.testing.assert_array_equal(again[0], sx)

    def test_duplicate_climates_do_not_allow_same_row_as_both_parents(self):
        x = np.zeros((22, 2))
        y = np.r_[np.zeros(14, int), np.ones(8, int)]
        _, _, _, trace = network.make_smote(x, y, np.ones(22), 5, 19)
        self.assertTrue(np.all(trace['base'] != trace['other']))
        with self.assertRaises(ValueError):
            network.make_smote(x[:16], y[:16], np.ones(16), 5, 19)

    def test_forward_formula_prior_adjustment_and_extrapolation(self):
        m = {'normalization': {'temperature_mean': 10., 'temperature_sd': 5., 'gamma_shape': 2., 'gamma_rate': .1, 'cdf_clip': 1e-7},
             'hidden_units': 1, 'weights': [.2, -.4, .1, .7, -.2],
             'climate_hull': [[0, 0], [20, 0], [20, 2000], [0, 2000]],
             'original_area_weighted_prior': .1, 'resampled_case_weighted_prior': .5}
        x = np.array([[8., 500.], [12., 1000.]])
        z = network.normalize(x, m['normalization'])
        eta = .7*expit(z[:, 0]*.2-z[:, 1]*.4+.1)-.2
        p, inside = network.predict(m, x)
        np.testing.assert_allclose(p, expit(eta))
        adjusted, _ = network.predict(m, x, prior_adjusted=True)
        np.testing.assert_allclose(adjusted, expit(eta+np.log(.1/.9)))
        self.assertTrue(np.all(adjusted < p) and inside.all())
        with self.assertRaisesRegex(ValueError, 'hull'):
            network.predict(m, [[50., 500.]])

    def test_training_start_selection_uses_objective_and_fixed_seeds(self):
        rng = np.random.default_rng(92)
        x = rng.normal(size=(120, 2))
        y = (x[:, 0] > 0).astype(int)
        settings = {'hidden_units': 2, 'weight_decay': .1, 'initial_range': .7, 'initializations': 3,
                    'maxiter': 2000, 'raw_gradient_tolerance': 1e-6, 'accepted_gradient_per_effective_case': 1e-6}
        result = network.fit_network(x, y, np.ones(120), settings, 42, 'toy')
        allowed = [i for i, r in enumerate(result['starts']) if r['stationary']]
        expected = min(allowed, key=lambda i: (result['starts'][i]['objective'], i))
        self.assertEqual(result['selected_start'], expected)
        self.assertEqual(result['weights'], result['starts'][expected]['weights'])
        self.assertEqual(result['starts'][0]['seed'], network.derived_seed(42, 'toy', 'initialization:0'))


if __name__ == '__main__':
    unittest.main()
