"""Conditional block resampling semantics, missingness and prediction gates."""
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('calibration_block_bootstrap', ROOT/'tools/taxon_uncertainty/bootstrap.py')
boot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boot)
SETTINGS = json.loads((ROOT/'docs/thesis_research/calibration_2026_10_30/config.json').read_text())['calibration']


class UncertaintyTests(unittest.TestCase):
    def test_whole_block_draws_preserve_shared_multiplicity_and_seed(self):
        lo, la = np.array([1., 2., 7., 8., 11.]), np.array([31., 32., 31., 32., 31.])
        ids, inverse, counts, seed = boot.block_counts(lo, la, 5, 20, 42, 'case')
        self.assertEqual(counts.shape, (20, 3))
        np.testing.assert_array_equal(counts.sum(axis=1), 3)
        np.testing.assert_array_equal(counts[:, inverse[0]], counts[:, inverse[1]])
        np.testing.assert_array_equal(counts[:, inverse[2]], counts[:, inverse[3]])
        again = boot.block_counts(lo, la, 5, 20, 42, 'case')
        np.testing.assert_array_equal(again[2], counts)
        self.assertEqual(again[3], seed)

    def test_repeated_positive_rows_do_not_manufacture_unique_class_support(self):
        eta, y = np.arange(10.), np.r_[np.zeros(6, int), np.ones(4, int)]
        r = boot.fit_replicates(eta, y, np.ones(10), np.full((1, 10), 20), True, SETTINGS)[0]
        self.assertEqual(r['status'], 'insufficient_unique_class')
        self.assertEqual(r['positive'], 4)
        self.assertIsNone(r['slope'])

    def test_count_weights_match_literal_block_row_duplication(self):
        eta = np.linspace(-3, 3, 40)
        y = np.tile([0, 1], 20)
        weights = np.linspace(.2, 1., 40)
        counts = np.tile([1, 3, 0, 2], 10)
        result = boot.fit_replicates(eta, y, weights, counts[None, :], True, SETTINGS)[0]
        index = np.repeat(np.arange(40), counts)
        repeated = boot.calibration.fit_calibrator(eta[index], y[index], weights[index], SETTINGS)
        self.assertEqual(result['status'], 'fitted')
        np.testing.assert_allclose([result['slope'], result['intercept']], [repeated['slope'], repeated['intercept']], atol=1e-7)

    def test_missing_base_calibration_is_unavailable_not_zero_uncertainty(self):
        eta, y = np.arange(20.), np.tile([0, 1], 10)
        result = boot.fit_replicates(eta, y, np.ones(20), np.ones((2, 20), int), False, SETTINGS)
        self.assertTrue(all(r['status'] == 'base_member_uncalibrated' and r['slope'] is None for r in result))

    def test_member_ensemble_quantiles_gate_and_extrapolation(self):
        base = {'normalization': {'temperature_mean': 10., 'temperature_sd': 5., 'gamma_shape': 2., 'gamma_rate': .1, 'cdf_clip': 1e-7},
                'hidden_units': 1, 'climate_hull': [[0, 0], [20, 0], [20, 2000], [0, 2000]],
                'calibrator': {'slope': 1., 'intercept': 0.}}
        members = [{**base, 'weights': [0, 0, 0, 0, t]} for t in [-2., 0., 3.]]
        c = [[[1., b]]*3 for b in [-.4, 0., .5, 1.]]
        record = {'coefficients': c, 'valid': [True]*4, 'minimum_valid': 4}
        point, band, support = boot.predict_interval(members, record, [[10., 500.]])
        pp = np.array([np.mean(expit(np.array([-2., 0., 3.])+b)) for b in [-.4, 0., .5, 1.]])
        np.testing.assert_allclose(band[:, 0], np.quantile(pp, [.025, .5, .975]))
        np.testing.assert_allclose(point[0], np.mean(expit([-2., 0., 3.])))
        self.assertEqual(support[0], 3)
        with self.assertRaisesRegex(ValueError, 'not estimable'):
            boot.predict_interval(members, {**record, 'valid': [True, True, True, False]}, [[10., 500.]])
        with self.assertRaisesRegex(ValueError, 'hull'):
            boot.predict_interval(members, record, [[50., 500.]])

    def test_exact_stored_hull_vertices_are_supported_in_anisotropic_real_hull(self):
        models = json.loads((ROOT/'docs/thesis_research/calibration_2026_10_30/models.json').read_text())
        member = models['abies_alba']['10']['original'][0]
        record = {'coefficients': [[[1., 0.]]*3]*4, 'valid': [True]*4, 'minimum_valid': 4}
        for vertex in member['climate_hull']:
            _, _, support = boot.predict_interval([member]*3, record, [vertex])
            self.assertEqual(support[0], 3)


if __name__ == '__main__':
    unittest.main()
