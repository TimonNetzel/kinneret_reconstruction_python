import unittest
from decimal import Decimal
from audit_monticchio_variables import measurement


class SourceValues(unittest.TestCase):
    def row(self, value, **change):
        return dict(taxonid=1, units='NISP', element='pollen', elementtype='pollen',
                    context=None, symmetry=None, value=value) | change

    def test_fractional_counts_are_preserved(self):
        self.assertEqual(measurement([self.row(1.5)], 1), Decimal('1.5'))

    def test_null_is_rejected_but_sparse_absence_is_zero(self):
        with self.assertRaises(ValueError):
            measurement([self.row(None)], 1)
        self.assertEqual(measurement([], 1), Decimal(0))

    def test_conflicting_duplicates_are_not_summed(self):
        with self.assertRaises(ValueError):
            measurement([self.row(1), self.row(2)], 1)
        self.assertEqual(measurement([self.row(1), self.row(1)], 1), Decimal(1))

    def test_different_contexts_and_units_remain_separate(self):
        rows = [self.row(2), self.row(90, context='anachronic'), self.row(80, units='percent')]
        self.assertEqual(measurement(rows, 1), Decimal(2))


if __name__ == '__main__':
    unittest.main()
