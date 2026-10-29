"""Contract and failure-mode tests; synthetic fixtures, no network needed."""
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import import_data


def pangaea(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("/* DATA DESCRIPTION:\nSynthetic test fixture, not research data.\n*/\n")
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


class ImportContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.catalog = self.root / "sources.json"
        with sqlite3.connect(self.raw / "empd2.sqlite3") as con:
            con.executescript("""
                CREATE TABLE metadata (
                    samplename TEXT PRIMARY KEY, originalsamplename TEXT, sitename TEXT,
                    country TEXT, latitude NUMERIC, longitude NUMERIC, elevation NUMERIC,
                    locationreliability TEXT, sampletype TEXT, samplecontext TEXT,
                    agebp NUMERIC, ageuncertainty TEXT, ispercent INTEGER,
                    isdigitised INTEGER, empd_version TEXT);
                CREATE TABLE climate (samplename TEXT PRIMARY KEY, t_ann NUMERIC, p_ann NUMERIC);
                CREATE TABLE p_counts (samplename TEXT, var_ INTEGER, count NUMERIC,
                                       PRIMARY KEY(samplename,var_));
                CREATE TABLE p_vars (var_ INTEGER PRIMARY KEY, acc_var_ INTEGER,
                    original_varname TEXT, acc_varname TEXT, groupid TEXT);
                CREATE TABLE groupid (groupid TEXT PRIMARY KEY, groupname TEXT,
                    higher_groupid TEXT, used_in_sum INTEGER, percent_values INTEGER);
                INSERT INTO p_vars VALUES (1,10,'Original A','Accepted A','AP'),
                                          (2,20,'Original B','Accepted B','AP');
                INSERT INTO groupid VALUES ('AP','Arboreal pollen','P',1,1);
                INSERT INTO p_counts VALUES ('A',1,1.5),('A',2,0),('B',2,3),('C',1,4);
                INSERT INTO climate VALUES ('A',12,800),('B',NULL,700),('C',10,600);
            """)
            for sample, reliability in (("A", "A"), ("B", "NULL"), ("C", "X")):
                con.execute("INSERT INTO metadata VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (sample, sample, 'Synthetic site', 'Test', 40, 15, 100,
                             reliability, 'moss', 'land', -10, 'A', 0, 0, 'fixture'))
        self.pollen = self.raw / "monticchio_739452.tab"
        self.entity = self.raw / "epd_entity_804591.tab"
        self.ages = self.raw / "epd_ages_cut_804596.tab"
        pangaea(self.pollen, ["Depth sed [m]", "Abi [#]", "vol.sp.a [cm**3]"],
                 [["0.05", "5", "0.5"], ["0.62", "7", "1"], ["13.00", "9", "1"]])
        pangaea(self.entity, ["ID (E#, fixture)", "Event", "Depth top [m]", "Depth bot [m]"],
                 [["932", "LGM90D", "0", "12"]])
        self.age_header = ["Event", "ID", "Depth sed [m]", "Age dated [ka BP]",
                           "Age min [ka]", "Age max [ka]", "Uncertainty class"]
        self.age_rows = [["LGM90D", "932", "0.050", "0.1", "0", "0.2", "3"],
                         ["LGM90D", "932", "0.625", "1.1", "1", "1.2", "4"]]
        pangaea(self.ages, self.age_header, self.age_rows)
        self.lock()

    def lock(self):
        names = ["empd2.sqlite3", "monticchio_739452.tab", "epd_entity_804591.tab", "epd_ages_cut_804596.tab"]
        files = [dict(source_id=sid, file=name, byte_count=(self.raw/name).stat().st_size,
                      sha256=hashlib.sha256((self.raw/name).read_bytes()).hexdigest(),
                      download_url="https://example.invalid/fixture", doi=None, license="synthetic fixture")
                 for sid, name in zip(import_data.REQUIRED, names)]
        self.catalog.write_text(json.dumps(dict(files=files)))

    def run_import(self, name="output"):
        return import_data.run(self.raw, self.root/name, self.catalog)

    def csv(self, name, folder="output"):
        with (self.root/folder/name).open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_preserves_fractional_counts_sparse_rows_and_quality_codes(self):
        report = self.run_import()
        values = self.csv("empd_measurements.csv")
        self.assertEqual(len(values), 4)  # no implicit zero-filled Cartesian product
        self.assertEqual(values[0]["count"], "1.5")
        self.assertEqual(values[0]["is_fractional"], "1")
        self.assertEqual(values[1]["count"], "0")
        samples = {r["sample_id"]: r for r in self.csv("empd_samples.csv")}
        self.assertEqual(samples["A"]["age_bp"], "-10")
        self.assertIn("location_quality_unknown", samples["B"]["quality_flags"])
        self.assertEqual(samples["B"]["tann_c"], "")
        self.assertIn("location_X", samples["C"]["quality_flags"])
        self.assertEqual(report["empd"]["samples_with_fractional_counts"], 1)

    def test_decimal_join_units_and_exclusions(self):
        report = self.run_import()
        samples = self.csv("monticchio_samples.csv")
        self.assertEqual(samples[1]["depth_offset_m"], "-0.005")
        self.assertEqual(samples[1]["age_bp"], "1100.0")
        self.assertEqual(samples[1]["age_min_bp"], "1000")
        self.assertEqual(len(self.csv("monticchio_values.csv")), 4)
        self.assertEqual(self.csv("monticchio_excluded.csv")[0]["pollen_depth_m"], "13.00")
        self.assertFalse(report["monticchio"]["age_bounds_are_joint_ensemble"])

    def test_literal_null_is_missing_and_raw_token_is_preserved(self):
        with sqlite3.connect(self.raw/"empd2.sqlite3") as con:
            con.execute("UPDATE climate SET t_ann='NULL',p_ann='NULL' WHERE samplename='B'")
            con.execute("UPDATE metadata SET latitude='NULL',longitude='NULL' WHERE samplename='B'")
        self.lock()
        self.run_import()
        sample = next(r for r in self.csv("empd_samples.csv") if r["sample_id"] == "B")
        self.assertEqual(sample["tann_c"], "")
        self.assertEqual(sample["tann_raw"], "NULL")
        self.assertEqual(sample["latitude_raw"], "NULL")
        self.assertIn("missing_climate", sample["quality_flags"])
        self.assertIn("invalid_latitude", sample["quality_flags"])

    def test_unrecognised_numeric_text_is_not_silently_coerced(self):
        with sqlite3.connect(self.raw/"empd2.sqlite3") as con:
            con.execute("UPDATE climate SET t_ann='bad_value' WHERE samplename='B'")
        self.lock()
        with self.assertRaisesRegex(import_data.ImportFailure, "Invalid numeric"):
            self.run_import()
        self.assertFalse((self.root/"output").exists())

    def test_rejects_changed_hash_before_writing_output(self):
        with self.pollen.open("a") as stream:
            stream.write("modified")
        with self.assertRaises(import_data.ImportFailure):
            self.run_import()
        self.assertFalse((self.root/"output").exists())

    def test_rejects_ambiguous_depth_match(self):
        pangaea(self.pollen, ["Depth sed [m]", "Abi [#]"], [["0.05",5],["0.620",7],["0.630",8]])
        self.lock()
        with self.assertRaisesRegex(import_data.ImportFailure, "Ambiguous"):
            self.run_import()
        self.assertFalse((self.root/"output").exists())

    def test_rejects_reuse_of_one_sample(self):
        self.age_rows.append(["LGM90D", "932", "0.620", "1.1", "1", "1.2", "4"])
        pangaea(self.ages, self.age_header, self.age_rows)
        self.lock()
        with self.assertRaisesRegex(import_data.ImportFailure, "reused"):
            self.run_import()

    def test_rejects_cut_age_outside_entity_limits(self):
        self.age_rows.append(["LGM90D", "932", "13", "21", "20", "22", "3"])
        pangaea(self.ages, self.age_header, self.age_rows)
        self.lock()
        with self.assertRaisesRegex(import_data.ImportFailure, "outside"):
            self.run_import()

    def test_orphan_measurement_does_not_disappear(self):
        with sqlite3.connect(self.raw/"empd2.sqlite3") as con:
            con.execute("INSERT INTO p_counts VALUES ('missing_sample',1,2)")
        self.lock()
        with self.assertRaisesRegex(import_data.ImportFailure, "Orphan"):
            self.run_import()
        self.assertFalse((self.root/"output").exists())

    def test_output_is_deterministic_and_never_overwritten(self):
        self.run_import("one")
        self.run_import("two")
        one = {p.name:p.read_bytes() for p in (self.root/"one").iterdir()}
        two = {p.name:p.read_bytes() for p in (self.root/"two").iterdir()}
        self.assertEqual(one, two)
        with self.assertRaisesRegex(import_data.ImportFailure, "already exists"):
            self.run_import("one")
        self.assertEqual(one, {p.name:p.read_bytes() for p in (self.root/"one").iterdir()})

    def test_reference_accepts_same_inputs_and_rejects_drift(self):
        self.run_import("baseline")
        reference = dict(report=json.loads((self.root/"baseline/report.json").read_text()),
                         manifest=json.loads((self.root/"baseline/manifest.json").read_text()))
        path = self.root/"reference.json"
        path.write_text(json.dumps(reference))
        import_data.run(self.raw, self.root/"matching", self.catalog, reference_path=path)
        reference["report"]["empd"]["samples"] += 1
        path.write_text(json.dumps(reference))
        with self.assertRaisesRegex(import_data.ImportFailure, "Reference mismatch"):
            import_data.run(self.raw, self.root/"mismatch", self.catalog, reference_path=path)
        self.assertFalse((self.root/"mismatch").exists())


if __name__ == "__main__":
    unittest.main()
