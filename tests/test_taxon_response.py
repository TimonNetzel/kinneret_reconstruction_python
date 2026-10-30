"""Core response invariants run in ordinary CI without optional GIS packages."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy.special import expit

path = Path(__file__).resolve().parents[1]/'tools/taxon_transfer/model.py'
spec = importlib.util.spec_from_file_location('taxon_response_model', path)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
OPTIONS = {'ridge_lambda': .001, 'minimum_training_cells_per_class': 30,
           'maxiter': 2000, 'gradient_tolerance': 1e-7, 'accepted_gradient_max': 1e-6}


class ResponseTests(unittest.TestCase):
    def test_gradient_against_finite_differences(self):
        rng = np.random.default_rng(27)
        x = np.column_stack([rng.normal(size=80), rng.uniform(0, 2000, 80)])
        z = model.design(x, np.array([0, 1000]), np.array([1, 600]))
        y, w, beta = rng.integers(0, 2, 80), rng.uniform(.2, 1, 80), rng.normal(size=6)
        _, gradient = model.objective(beta, z, y, w, .1)
        numerical = []
        for j in range(6):
            step = np.eye(6)[j]*1e-6
            numerical.append((model.objective(beta+step, z, y, w, .1)[0]-model.objective(beta-step, z, y, w, .1)[0])/2e-6)
        np.testing.assert_allclose(gradient, numerical, atol=1e-8)

    def test_training_scaler_and_extrapolation_gate(self):
        rng = np.random.default_rng(28)
        x = np.column_stack([rng.uniform(-10, 30, 1600), rng.uniform(0, 2500, 1600)])
        y = rng.binomial(1, expit((x[:, 0]-10)/8-(x[:, 1]-1200)**2/1e6))
        weights = rng.uniform(.2, 1, len(x))
        fitted = model.fit(x, y, weights, OPTIONS)
        np.testing.assert_allclose(fitted['mean'], np.average(x, axis=0, weights=weights))
        p, inside = model.probabilities(fitted, x)
        self.assertTrue(inside.all())
        self.assertTrue(np.all((p >= 0) & (p <= 1)))
        with self.assertRaisesRegex(ValueError, 'convex hull'):
            model.probabilities(fitted, [[80, 10000]])
        _, inside = model.probabilities(fitted, [[80, 10000]], reject_extrapolation=False)
        self.assertFalse(inside[0])

    def test_invalid_values_and_single_class_fail(self):
        x = np.column_stack([np.arange(100), np.arange(100)])
        for y in (np.zeros(100), np.full(100, np.nan), np.full(100, .5)):
            with self.assertRaises(ValueError):
                model.fit(x, y, np.ones(100), OPTIONS)
        with self.assertRaises(ValueError):
            model.design([[10, -1]], [0, 0], [1, 1])
        with self.assertRaises(ValueError):
            model.design([[np.nan, 10]], [0, 0], [1, 1])


if __name__ == '__main__':
    unittest.main()
