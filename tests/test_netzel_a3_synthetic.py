"""Only the authorized finite A3 examination. Standard library, no project model."""
import ast
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import unittest
import sys

# pytest's console entry point need not put the repository root on sys.path.
# Add only this checkout to locate the new isolated scripts, without installing
# or importing the reconstruction package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from scripts.netzel_a3_synthetic import audit, kernel, reference
from scripts.netzel_a3_synthetic.constants import DESIGN_BLOB


class FiniteA3Acceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = audit.examine()

    def test_all_locked_witnesses_pass(self):
        failures = [r for r in self.tables['witnesses'] if r['status'] != 'bestanden']
        self.assertEqual(failures, [])

    def test_all_subcases_present(self):
        # Independent coverage list, not derived from the implementation's cases.
        required = {'S00','S01','S02-scaled','S02-witness','S03','S04','S05',
                    'S06a','S06b','S06c','S06d','S07','S08a','S08b',
                    'S08c-a','S08c-b','S08d','S09-base','S09-opposite',
                    'S09-noY','S10','S11-equal','S11-unequal','S12'}
        self.assertEqual({r['case'] for r in self.tables['cases']},required)
        self.assertTrue(all(r['status']=='bestanden' for r in self.tables['cases']))
        errors = self.tables['invalid']
        self.assertEqual(len(errors),33)
        self.assertEqual({r['case'].split('-')[0] for r in errors},
                         {'I01','I02','I03','I04','I05','I06','I07','I08','I09'})
        self.assertTrue(all(r['actual']==r['expected'] and not r['Q_returned'] for r in errors))

    def test_each_defective_method_really_fails_its_witness(self):
        rows = self.tables['countermethods']
        self.assertEqual({r['method'] for r in rows},
            {'global_normalization','independent_psi','independent_categories',
             'independent_theta','independent_C_marginals','D_only_chronology'})
        self.assertTrue(all(r['difference']!=0 for r in rows))

    def test_saved_tables_are_complete_current_bytes(self):
        directory = ROOT/'docs/thesis_research/A3_SYNTHETIC_AUDIT'
        payload = audit.artifacts(self.tables)
        self.assertEqual({p.name for p in directory.iterdir()},set(payload))
        for name,data in payload.items():
            with self.subTest(file=name):
                self.assertEqual((directory/name).read_bytes(),data)
        manifest = json.loads(payload['manifest.json'])
        for name,entry in manifest['files'].items():
            self.assertEqual(hashlib.sha256(payload[name]).hexdigest(),entry['sha256'])

    def test_independent_code_paths_and_locked_design(self):
        for module in (kernel,reference):
            tree = ast.parse(Path(module.__file__).read_text())
            relative = [n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom) and n.level]
            self.assertEqual(relative,['constants'])
        data = (ROOT/'docs/thesis_research/A3_SYNTHETIC_DESIGN.md').read_bytes()
        self.assertEqual(hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest(),DESIGN_BLOB)

    def test_no_silent_extension_after_input_validation(self):
        candidate = kernel.base_metadata()
        candidate['n'] = (1,1,0)  # valid counts, but outside the fixed S01 input
        with self.assertRaisesRegex(kernel.Rejected,'OUT_OF_DESIGN'):
            kernel.calculate('S01',metadata=candidate)
        with self.assertRaisesRegex(kernel.Rejected,'OUT_OF_DESIGN'):
            kernel.calculate('S01',q={-1:Fraction(1,2),1:Fraction(1,2)})
        # A repeated parent is permitted; it is not duplicated measurement data.
        kernel.validate(kernel.base_metadata(),{-1:Fraction(1,3),1:Fraction(2,3)})


if __name__ == '__main__':
    unittest.main()
