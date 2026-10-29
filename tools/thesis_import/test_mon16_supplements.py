"""Tests for source integrity and scientific failure cases; no source downloads."""
from decimal import Decimal as D
import io
import unittest
import zipfile

import audit_mon16_supplements as audit


class SupplementSafetyTests(unittest.TestCase):
    def test_interval_violation_is_preserved_not_clipped(self):
        source = [(137, [D('135.5'), D('1437.6'), D('1435.2'), D('1437.2')])]
        bad, _, _ = audit.check_series(source)
        self.assertEqual(bad[0]['median_minus_maximum'], D('.4'))
        self.assertEqual(source[0][1][1], D('1437.6'))

    def test_depth_gap_is_not_resampled(self):
        rows = [(1, [D('1'), D('100'), D('80'), D('120')]),
                (2, [D('3.8'), D('101'), D('81'), D('121')])]
        _, gaps, _ = audit.check_series(rows)
        self.assertEqual(gaps[0]['step_mm'], D('2.8'))
        self.assertEqual(len(rows), 2)

    def test_reversed_age_series_is_flagged(self):
        rows = [(1, list(map(D, ['1', '100', '80', '120']))),
                (2, list(map(D, ['2', '99', '81', '121'])))]
        self.assertEqual(audit.check_series(rows)[2]['median'], [2])

    def test_nonfinite_is_rejected(self):
        with self.assertRaises(ValueError):
            audit.check_series([(5, [D('1'), D('NaN'), D('0'), D('2')])])

    def test_changed_source_bytes_are_rejected(self):
        content = b'original'
        with self.assertRaises(ValueError):
            audit.locked(b'changed!', len(content), audit.digest(content))

    def test_uncached_formula_does_not_become_zero(self):
        buf = io.BytesIO()
        ns = audit.NS['s']
        with zipfile.ZipFile(buf, 'w') as z:
            z.writestr('xl/workbook.xml', f'<workbook xmlns="{ns}" xmlns:r="{audit.REL}"><sheets><sheet name="Test" r:id="id9"/></sheets></workbook>')
            z.writestr('xl/_rels/workbook.xml.rels', '<Relationships><Relationship Id="id9" Target="worksheets/different-name.xml"/></Relationships>')
            z.writestr('xl/worksheets/different-name.xml', f'<worksheet xmlns="{ns}"><sheetData><row r="1"><c r="A1"><f>1+1</f></c></row></sheetData></worksheet>')
        with self.assertRaisesRegex(ValueError, 'without cached'):
            audit.read_workbook(buf.getvalue())


if __name__ == '__main__':
    unittest.main()
