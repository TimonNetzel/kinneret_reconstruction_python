"""Scientific boundary cases: depth coincidence is not age identity."""
from decimal import Decimal as D
import unittest
from audit_marker_crosswalk import position


class MarkerPositionTests(unittest.TestCase):
    def setUp(self):
        self.markers = [('L4', D('6.62')), ('L5', D('7.87')), ('L6', D('8.62'))]

    def test_exact_marker_is_not_interpolated(self):
        self.assertEqual(position(D('8.62'), self.markers), ('coincident_listed_depth', 'L6', 'L6'))

    def test_native_bracket_preserves_order(self):
        self.assertEqual(position(D('7'), self.markers), ('between_listed_marker_depths', 'L4', 'L5'))

    def test_outside_span_is_not_extrapolated(self):
        self.assertEqual(position(D('0'), self.markers), ('shallower_than_marker_span', '', 'L4'))
        self.assertEqual(position(D('12'), self.markers), ('deeper_than_marker_span', 'L6', ''))

    def test_ambiguous_duplicate_depth_is_rejected(self):
        with self.assertRaises(ValueError):
            position(D('7'), [('L4', D('6.62')), ('different_layer', D('6.62'))])


if __name__ == '__main__':
    unittest.main()
