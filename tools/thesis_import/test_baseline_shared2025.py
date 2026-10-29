"""Scientific guardrails for the modern-only spatial baseline."""
import unittest

import numpy as np

from evaluate_shared2025 import (analog_neighbors, canonical_coordinates,
    evaluate_metrics, hellinger_features, nearest_km, spatial_folds, sphere_xyz)


class BaselineTests(unittest.TestCase):
    def test_composition_scaling_and_known_hellinger_distance(self):
        counts = np.array([[4., 0.], [1., 3.], [0., 8.]])
        transformed = hellinger_features(counts)
        np.testing.assert_allclose(transformed, [[1, 0], [.5, np.sqrt(.75)], [0, 1]])
        np.testing.assert_allclose(transformed, hellinger_features(counts*7))
        indices, distances = analog_neighbors(transformed[[0]], transformed[[2]], 1)
        self.assertAlmostEqual(distances[0, 0], 1)

    def test_missing_negative_and_empty_counts_fail(self):
        for counts in ([[0., 0.]], [[-1., 2.]], [[np.nan, 2.]]):
            with self.assertRaises(ValueError):
                hellinger_features(counts)

    def test_exact_pollen_ties_follow_sorted_training_order(self):
        train = np.array([[1., 0.], [0., 1.], [1., 0.]])
        indices, distances = analog_neighbors(train, train[[0]], 2)
        np.testing.assert_array_equal(indices, [[0, 2]])
        np.testing.assert_allclose(distances, 0)
        # Only these training labels enter the analogue mean.
        climate = np.array([10., 999., 30.])
        self.assertEqual(climate[indices].mean(), 20)

    def test_date_line_distance_cannot_escape_buffer(self):
        query = sphere_xyz(np.array([0., 0.]), np.array([179.9, -170.]))
        reference = sphere_xyz(np.array([0.]), np.array([-179.9]))
        km = nearest_km(query, reference, 6371.0088)
        self.assertAlmostEqual(km[0], 22.239016, places=5)
        self.assertLess(km[0], 100)
        self.assertGreater(km[1], 100)
        lat, lon = canonical_coordinates([0, 0], [180, -180])
        np.testing.assert_array_equal(lon, [-180, -180])

    def test_rounded_groups_and_blocks_remain_together(self):
        lat = [49.9999999, 50, 40, 20, 0, -40]
        lon = [20, 20, 0, 90, -100, -180]
        block, fold = spatial_folds(lat, lon, 5, 3, 42)
        self.assertEqual(block[0], block[1])
        self.assertEqual(fold[0], fold[1])
        np.testing.assert_array_equal(fold, spatial_folds(lat, lon, 5, 3, 42)[1])
        for group in set(block):
            self.assertEqual(len(set(fold[block == group])), 1)

    def test_equal_group_metrics_resist_duplicate_sampling(self):
        a = evaluate_metrics(np.zeros((2, 1)), np.array([[1.], [3.]]),
                             np.full((2, 1), 4.), ['a', 'b'], ['t'])
        b = evaluate_metrics(np.zeros((4, 1)), np.array([[1.], [1.], [1.], [3.]]),
                             np.full((4, 1), 4.), ['a', 'a', 'a', 'b'], ['t'])
        self.assertAlmostEqual(a['t']['equal_coordinate_group']['rmse'], np.sqrt(5))
        self.assertEqual(a['t']['equal_coordinate_group'], b['t']['equal_coordinate_group'])
        self.assertNotEqual(a['t']['sample']['rmse'], b['t']['sample']['rmse'])


if __name__ == '__main__':
    unittest.main()
