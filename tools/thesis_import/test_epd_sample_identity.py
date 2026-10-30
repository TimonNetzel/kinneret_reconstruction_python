"""Archive NULL/escape handling and depth precision failure cases."""
from decimal import Decimal as D
import unittest
from audit_epd_sample_identity import copy_field, rounded_metres


class ArchiveParsingTests(unittest.TestCase):
    def test_null_is_distinct_from_literal_null_marker(self):
        self.assertIsNone(copy_field(r'\N'))
        self.assertEqual(copy_field(r'\\N'), r'\N')

    def test_copy_escaped_tab_is_part_of_field(self):
        self.assertEqual(copy_field(r'core\tnote\nnext'), 'core\tnote\nnext')

    def test_observed_binary_format_differs_from_decimal_half_even(self):
        self.assertEqual(rounded_metres('513.50'), D('5.13'))
        self.assertEqual(rounded_metres('724.50'), D('7.25'))

    def test_export_cannot_recover_unrounded_depth(self):
        self.assertEqual(rounded_metres('513.49'), rounded_metres('513.00'))
        self.assertNotEqual(D('513.49'), D('513.00'))


if __name__ == '__main__':
    unittest.main()
