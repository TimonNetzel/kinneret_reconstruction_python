"""Negative cases for the new numerical provenance comparison, without RDS."""
import unittest
from decimal import Decimal

from audit_neotoma_workshop import comparable_keys, fingerprint, measurement_map


def record(value, taxon=312):
    return dict(taxonid=taxon, units="NISP", element="pollen/spore",
                context=None, symmetry=None, elementtype="pollen/spore", value=value)


class ProvenanceComparisonTest(unittest.TestCase):
    def test_collapsed_variables_with_different_values_are_not_added(self):
        old = measurement_map([record(1), record(2)])
        new = measurement_map([record(3)])
        keys, ambiguous, _, _ = comparable_keys([old], [new])
        self.assertEqual(keys, [])
        self.assertEqual(len(ambiguous), 1)
        with self.assertRaises(ValueError):
            fingerprint(old, list(old))

    def test_repeated_identical_value_is_not_counted_twice(self):
        m = measurement_map([record(3), record(3)])
        self.assertEqual(fingerprint(m, list(m)), (Decimal(3),))

    def test_null_measurement_cannot_become_zero(self):
        with self.assertRaises(ValueError):
            measurement_map([record(None)])
        zero = measurement_map([record(0)])
        self.assertEqual(fingerprint({}, list(zero)), fingerprint(zero, list(zero)))

    def test_distinct_measurement_contexts_are_not_merged(self):
        a, b = record(5), record(7)
        b["units"] = "percent"
        m = measurement_map([a, b])
        self.assertEqual(len(m), 2)
        self.assertEqual(set(fingerprint(m, list(m))), {Decimal(5), Decimal(7)})


if __name__ == "__main__":
    unittest.main()
