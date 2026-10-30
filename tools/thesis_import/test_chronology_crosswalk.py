"""Meaningful boundary checks for source matching and archive multiplicities."""
from decimal import Decimal as D
import unittest

import audit_chronology_crosswalk as a


class CrosswalkTests(unittest.TestCase):
    def test_rounding_ties_and_reordered_archive(self):
        original = [['5.0525', '6.655'], ['5.0535', '6.656']]
        published = [{'Depth sed [m]':'5.054', 'Age model [ka]':'6.656'},
                     {'Depth sed [m]':'5.052', 'Age model [ka]':'6.655'}]
        rows = a.age_archive_alignment(original, published)
        self.assertEqual([r['pangaea_source_row'] for r in rows], [2, 1])

    def test_duplicate_pairs_cannot_reuse_a_source_row(self):
        source = [['1.0000', '2.0'], ['1.0000', '2.0']]
        incompatible = [{'Depth sed [m]':'1.000', 'Age model [ka]':'2.0'},
                        {'Depth sed [m]':'1.001', 'Age model [ka]':'2.0'}]
        with self.assertRaisesRegex(ValueError, 'multiplicity'):
            a.age_archive_alignment(source, incompatible)

    def test_ambiguous_fingerprints_remain_multiple(self):
        raw = {col:'1' for col, _, _ in a.TAXA}
        row = {'Pollen tot [#]':'100', **{col:'1.00' for _, col, _ in a.TAXA}}
        self.assertEqual(a.candidate_rows(raw, [dict(row), dict(row)], D('.005'), 'pangaea'), [0, 1])

    def test_missing_or_fractional_raw_counts_are_not_zero(self):
        for value in ('', '.5', '-1'):
            with self.assertRaises((ValueError, ArithmeticError)):
                a.percent_error(value, '100', '0.0')

    def test_half_grain_denominator_is_preserved(self):
        self.assertEqual(a.percent_error('1', '417.5', '0'), D(100)/D('417.5'))
        self.assertNotEqual(a.percent_error('1', '417.5', '0'), a.percent_error('1', '418', '0'))


if __name__ == '__main__':
    unittest.main()
