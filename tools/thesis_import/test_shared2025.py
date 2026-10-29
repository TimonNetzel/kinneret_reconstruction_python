"""Small parser regressions; the full source run checks the scientific tables."""
from pathlib import Path
import tempfile
import unittest
import zipfile

from prepare_shared2025 import XlsxReader, s1_records, unaccented


class ParserTests(unittest.TestCase):
    def test_explicit_zero_and_missing_cell_are_distinct(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'fixture.xlsx'
            with zipfile.ZipFile(path,'w') as archive:
                archive.writestr('xl/sharedStrings.xml','<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"/>')
                archive.writestr('xl/worksheets/sheet2.xml','''<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="3"><c r="D3"><v>0</v></c><c r="E3"/><c r="F3"><f>1+1</f><v>2</v></c></row></sheetData></worksheet>''')
            reader=XlsxReader(path)
            try: rows=list(reader.rows(2))
            finally: reader.close()
            self.assertEqual(rows[0][1]['D'],('0','n',False))
            self.assertNotIn('E',rows[0][1])
            self.assertEqual(rows[0][1]['F'],('2','n',True))

    def test_only_explicit_word_vertical_merge_inherits_group(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'fixture.docx'
            xml='''<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:tbl>
            <w:tr><w:tc><w:p><w:r><w:t>A</w:t></w:r></w:p></w:tc><w:tc/><w:tc><w:tcPr><w:vMerge w:val="restart"/></w:tcPr><w:p><w:r><w:t>Group</w:t></w:r></w:p></w:tc></w:tr>
            <w:tr><w:tc><w:p><w:r><w:t>B</w:t></w:r></w:p></w:tc><w:tc/><w:tc><w:tcPr><w:vMerge/></w:tcPr></w:tc></w:tr>
            <w:tr><w:tc><w:p><w:r><w:t>C</w:t></w:r></w:p></w:tc><w:tc/><w:tc/></w:tr>
            </w:tbl></w:body></w:document>'''
            with zipfile.ZipFile(path,'w') as archive: archive.writestr('word/document.xml',xml)
            self.assertEqual([r['group'] for r in s1_records(path)],['Group','Group',''])

    def test_name_equivalence_does_not_erase_nonaccent_differences(self):
        self.assertEqual(unaccented('Jiménez_a1'),unaccented('Jimenez_a1'))
        self.assertNotEqual(unaccented('Jimenez_a1'),unaccented('Jimenez_a10'))


if __name__=='__main__':unittest.main()
