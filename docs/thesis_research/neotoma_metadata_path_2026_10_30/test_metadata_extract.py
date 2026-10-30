"""Synthetic relational checks only; no real Neotoma observations or server use."""
import pathlib
import sqlite3
import unittest


QUERY = pathlib.Path(__file__).with_name("metadata_extract.sql").read_text()


class MetadataExtractTest(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            ATTACH DATABASE ':memory:' AS ndb;
            CREATE TABLE ndb.datasets (
                datasetid INTEGER PRIMARY KEY, datasetname TEXT,
                notes TEXT, collectionunitid INTEGER);
            CREATE TABLE ndb.collectionunits (
                collectionunitid INTEGER PRIMARY KEY, handle TEXT,
                collunitname TEXT, location TEXT, notes TEXT, siteid INTEGER);
            CREATE TABLE ndb.sites (siteid INTEGER PRIMARY KEY, sitename TEXT);
            CREATE TABLE ndb.samples (
                sampleid INTEGER PRIMARY KEY, datasetid INTEGER, samplename TEXT,
                labnumber TEXT, notes TEXT, analysisunitid INTEGER);
            CREATE TABLE ndb.analysisunits (
                analysisunitid INTEGER PRIMARY KEY, collectionunitid INTEGER,
                analysisunitname TEXT, depth REAL, thickness REAL,
                igsn TEXT, notes TEXT);
            INSERT INTO ndb.datasets VALUES
                (47525, 'synthetic target', 'dataset note', 10),
                (999, 'synthetic unrelated', 'do not select', 10);
            INSERT INTO ndb.collectionunits VALUES
                (10, 'synthetic', 'fixture unit', 'fixture location', 'unit note', 20);
            INSERT INTO ndb.sites VALUES (20, 'synthetic site');
        """)

    def tearDown(self):
        self.db.close()

    def rows(self):
        return [dict(row) for row in self.db.execute(QUERY)]

    def test_dataset_filter_nulls_and_notes_are_preserved(self):
        self.db.executescript("""
            INSERT INTO ndb.analysisunits VALUES
                (30, 10, 'fixture section', 513.5, NULL, NULL, 'analysis note');
            INSERT INTO ndb.samples VALUES
                (1, 47525, NULL, 'fixture label', 'sample note', 30),
                (2, 999, 'exclude me', NULL, NULL, 30);
        """)
        rows = self.rows()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row['sampleid'], 1)
        self.assertIsNone(row['samplename'])
        self.assertIsNone(row['analysisunit_thickness_cm'])
        self.assertEqual(row['analysisunit_depth_cm'], 513.5)
        self.assertEqual([row[k] for k in ('dataset_notes', 'collectionunit_notes',
                                          'sample_notes', 'analysisunit_notes')],
                         ['dataset note', 'unit note', 'sample note', 'analysis note'])
        self.assertEqual(row['relation_status'], 'joined')

    def test_broken_and_mismatched_links_remain_visible(self):
        self.db.executescript("""
            INSERT INTO ndb.analysisunits VALUES (31, 99, NULL, NULL, NULL, NULL, NULL);
            INSERT INTO ndb.samples VALUES
                (3, 47525, 'mismatch', NULL, NULL, 31),
                (4, 47525, 'broken link', NULL, NULL, 99999);
        """)
        rows = self.rows()
        self.assertEqual([r['sampleid'] for r in rows], [3, 4])
        self.assertEqual([r['relation_status'] for r in rows],
                         ['collectionunit_mismatch', 'missing_analysisunit'])
        self.assertEqual(rows[0]['analysisunit_collectionunitid'], 99)
        self.assertIsNone(rows[1]['resolved_analysisunitid'])

    def test_dataset_without_samples_is_not_hidden(self):
        row, = self.rows()
        self.assertEqual(row['datasetid'], 47525)
        self.assertEqual(row['relation_status'], 'no_samples')
        self.assertIsNone(row['sampleid'])


if __name__ == '__main__':
    unittest.main()
