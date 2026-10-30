import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from rasterio.transform import from_origin
from calibrate import range_labels, splits


class TransferTests(unittest.TestCase):
    def test_geodesic_buffer_across_dateline(self):
        settings = {'folds': 5, 'buffer_km': 100, 'sphere_radius_km': 6371.0088}
        lon, lat = np.array([179.9, -179.9, 0, 15, 30, 45, 60]), np.array([10.25]*7)
        observed = False
        for _, train, test, evidence in splits(lon, lat, 5, settings):
            if test[0] and not test[1]:
                self.assertFalse(train[1])
                observed = True
            self.assertGreater(evidence['minimum_train_test_km'], 100)
        self.assertTrue(observed)

    def test_source_status_point_cells_and_invalid_geometries(self):
        properties = dict(taxon_id=1, name_use_id_orig=1, name_choro_orig='Demo', occurrence_status='Vorkommen', occurrence_origin='nativ')
        polygon = {'type': 'Polygon', 'coordinates': [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]]}
        features = [{'type': 'Feature', 'geometry': polygon, 'properties': properties}]
        features.append({'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [3.25, 3.25]}, 'properties': properties})
        features.append({'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [.75, .75]}, 'properties': properties | {'occurrence_status': 'fragliches Vorkommen'}})
        config = {'labels': {'polygon_boundary_exclusion_degrees': .1, 'point_neighbourhood_exclusion_degrees': .5, 'other_status_or_origin_exclusion_degrees': .5}}
        taxon = {'taxon_id': 1, 'name_use_id_orig': 1, 'source_name': 'Demo'}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'map.json'
            path.write_text(json.dumps({'type': 'FeatureCollection', 'features': features}))
            y, use, _ = range_labels(path, taxon, config, (8, 8), from_origin(0, 4, .5, .5))
            self.assertEqual(y[1*8+6], 1)  # isolated native point's containing cell
            self.assertTrue(use[1*8+6])
            self.assertFalse(use[6*8+1])  # uncertain occurrence within native polygon
            features[0]['geometry'] = {'type': 'Polygon', 'coordinates': [[[0, 0], [2, 2], [0, 2], [2, 0], [0, 0]]]}
            path.write_text(json.dumps({'type': 'FeatureCollection', 'features': features}))
            with self.assertRaisesRegex(ValueError, 'no repair'):
                range_labels(path, taxon, config, (8, 8), from_origin(0, 4, .5, .5))


if __name__ == '__main__':
    unittest.main()
