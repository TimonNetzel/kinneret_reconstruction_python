"""Probability calibration invariants; no external data downloads."""
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('spatial_calibration_model', ROOT/'tools/taxon_calibration/model.py')
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
SETTINGS = json.loads((ROOT/'docs/thesis_research/calibration_2026_10_30/config.json').read_text())['calibration']


class CalibrationTests(unittest.TestCase):
    def test_weighted_calibration_gradient_extreme_logits(self):
        eta = np.array([-1000., -5., -.1, 0., 2., 1000.])
        y, w = np.array([0, 1, 0, 1, 0, 1]), np.arange(1., 7.)
        theta = np.array([.3, -.7])
        loss, grad = model.objective(theta, eta, y, w, .0001)
        finite = []
        for j in range(2):
            delta = np.eye(2)[j]*1e-5
            finite.append((model.objective(theta+delta, eta, y, w, .0001)[0]-
                           model.objective(theta-delta, eta, y, w, .0001)[0])/2e-5)
        self.assertTrue(np.isfinite(loss))
        np.testing.assert_allclose(grad, finite, atol=1e-9)

    def test_calibration_recovers_shift_and_preserves_order(self):
        eta = np.linspace(-4, 4, 101)
        # Two real class rows at each logit, weighted to exactly known frequencies.
        q = expit(1.4*eta-.9)
        m = model.fit_calibrator(np.repeat(eta, 2), np.tile([0, 1], len(eta)),
                                 np.column_stack([1-q, q]).ravel(), SETTINGS)
        self.assertEqual(m['status'], 'fitted')
        self.assertLess(abs(m['slope']-1.4), .01)
        self.assertLess(abs(m['intercept']+.9), .01)
        self.assertLessEqual(m['projected_gradient'], 1e-7)
        self.assertTrue((np.diff(expit(m['slope']*eta+m['intercept'])) >= 0).all())

    def test_insufficient_class_identity_and_boundary_solution(self):
        c = model.fit_calibrator(np.arange(20.), np.r_[np.zeros(16), np.ones(4)], np.ones(20), SETTINGS)
        self.assertEqual(c['status'], 'insufficient_calibration_class')
        self.assertEqual((c['slope'], c['intercept']), (1., 0.))
        eta, y = np.arange(40.), np.r_[np.ones(20), np.zeros(20)]
        c = model.fit_calibrator(eta, y, np.ones(40), SETTINGS)
        self.assertEqual(c['slope'], 0.)
        self.assertLessEqual(c['projected_gradient'], 1e-7)

    def test_ensemble_means_probabilities_and_requires_all_hulls(self):
        base = {'normalization': {'temperature_mean': 10., 'temperature_sd': 5., 'gamma_shape': 2., 'gamma_rate': .1, 'cdf_clip': 1e-7},
                'hidden_units': 1, 'climate_hull': [[0, 0], [20, 0], [20, 2000], [0, 2000]]}
        members = [{**base, 'weights': [0, 0, 0, 0, t], 'calibrator': {'slope': 2., 'intercept': -.5}} for t in [-2., 0., 3.]]
        p, support = model.predict(members, [[10., 500.]])
        np.testing.assert_allclose(p, np.mean(expit(2*np.array([-2., 0., 3.])-.5)))
        self.assertEqual(support[0], 3)
        members[0] = {**members[0], 'climate_hull': [[0, 0], [5, 0], [5, 2000], [0, 2000]]}
        with self.assertRaisesRegex(ValueError, 'hull'):
            model.predict(members, [[10., 500.]])
        _, support = model.predict(members, [[10., 500.]], reject_extrapolation=False)
        self.assertEqual(support[0], 2)

    @unittest.skipUnless(importlib.util.find_spec('rasterio') and importlib.util.find_spec('shapely'),
                         'Optional research GIS dependencies absent')
    def test_nested_spatial_roles_exclude_outer_test_and_both_buffers(self):
        spec = importlib.util.spec_from_file_location('old_transfer_calibrate', ROOT/'tools/taxon_transfer/calibrate.py')
        import sys
        old_model = sys.modules.get('model')
        sm = importlib.util.spec_from_file_location('model', ROOT/'tools/taxon_transfer/model.py')
        module = importlib.util.module_from_spec(sm)
        sm.loader.exec_module(module)
        sys.modules['model'] = module
        try:
            pc = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(pc)
        finally:
            if old_model is None:
                del sys.modules['model']
            else:
                sys.modules['model'] = old_model
        lon, lat = np.meshgrid(np.arange(-10., 40., 1.), np.arange(30., 65., 1.))
        lon, lat = lon.ravel(), lat.ravel()
        settings = {'folds': 5, 'buffer_km': 100., 'sphere_radius_km': 6371.0088}
        for _, outer_train, outer_test, _ in pc.splits(lon, lat, 5, settings):
            pool = np.flatnonzero(outer_train)
            for _, train, cal, evidence in pc.splits(lon[pool], lat[pool], 5, {**settings, 'folds': 3}):
                self.assertFalse(outer_test[pool[train]].any())
                self.assertFalse(outer_test[pool[cal]].any())
                self.assertFalse((train & cal).any())
                self.assertGreater(evidence['minimum_train_test_km'], 100.)


if __name__ == '__main__':
    unittest.main()
