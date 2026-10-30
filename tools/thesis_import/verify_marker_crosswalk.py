#!/usr/bin/env python3
"""Independent PDF/cell and rational-arithmetic audit; optional fitz/openpyxl.

Does not import the producing auditor. ETH-12464's age was visually read from
the page image (the PDF text layer omits that column); see the evidence record.
"""
import argparse
import bisect
import csv
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import re

import fitz
import openpyxl


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(source, output, evidence_path):
    evidence = json.loads(evidence_path.read_text())
    summary = json.loads((output / 'summary.json').read_text())
    for name, expected in json.loads((output / 'reference_manifest.json').read_text()).items():
        assert sha(output / name) == expected, name
    for name, lock in evidence['file_locks'].items():
        assert (source / name).stat().st_size == lock['bytes'] and sha(source / name) == lock['sha256']
    assert sha(evidence_path) == summary['evidence_sha256']
    doc = fitz.open(source / 'wulf2001.pdf')
    # Independent extraction by physical column positions on the two inspected pages.
    for record in evidence['wulf_appendix_rows']:
        words = doc[record['pdf_page'] - 1].get_text('words')
        y = next(w[1] for w in words if w[0] < 65 and w[4] == str(record['row']))
        for left, right, field in [(70, 115, 'varve_age_bp'), (115, 165, 'thickness_mm'),
                                  (165, 222, 'composite_base_cm'), (280, 360, 'core_section'),
                                  (720, 842, 'aliases')]:
            actual = ' '.join(w[4] for w in words if left <= w[0] < right and abs(w[1] - y) < 2)
            assert actual == record[field], (record['row'], field, actual)
    links = read_csv(output / 'marker_links.csv')
    rows = {x['row']: x for x in evidence['wulf_appendix_rows']}
    book = openpyxl.load_workbook(source / '1-s2.0-S0277379126003252-mmc1.xlsx', read_only=True, data_only=True)
    a1 = book.worksheets[0]
    native = list(csv.DictReader((source / 'monticchio_741058.tab').read_text().split('*/', 1)[1].strip().splitlines(), delimiter='\t'))
    by_layer = {x['Name']: x for x in native}
    for x, link in zip(links, evidence['marker_identities']):
        r = link['a1_row']
        assert a1[f'A{r}'].value == x['tm_label'] == link['modern_label']
        assert x['legacy_layer'] == link['legacy_label']
        assert F(x['legacy_listed_depth_cm']) == F(by_layer[x['legacy_layer']]['Depth sed [m]']) * 100
        expected_base = F(rows[link['appendix_row']]['composite_base_cm'])
        assert F(x['historical_composite_base_cm']) == expected_base
        assert F(x['listed_to_historical_base_difference_cm']) == expected_base - F(x['legacy_listed_depth_cm'])
        for col, key in [('C', 'mon16_base_cm'), ('D', 'mon16_top_cm'), ('K', 'mon12_base_cm')]:
            assert abs(F(str(a1[f'{col}{r}'].value)) - F(x[key])) < F(1, 1000000)
    assert len(links) == 6
    original = read_csv(source / 'pr21_sample_correspondence.csv')
    contexts = read_csv(output / 'sample_marker_context.csv')
    depths = [F(x['legacy_listed_depth_cm']) / 100 for x in links]
    names = [x['legacy_layer'] for x in links]
    for x, c in zip(original, contexts):
        assert x['sample_id'] == c['sample_id'] and x['pollen_depth_m'] == c['pollen_depth_m']
        value = F(x['pollen_depth_m'])
        i = bisect.bisect_left(depths, value)
        if i < len(depths) and value == depths[i]:
            expected = 'coincident_listed_depth', names[i], names[i]
        elif i == 0:
            expected = 'shallower_than_marker_span', '', names[0]
        elif i == len(depths):
            expected = 'deeper_than_marker_span', names[-1], ''
        else:
            expected = 'between_listed_marker_depths', names[i - 1], names[i]
        assert tuple(c[k] for k in ['native_depth_position', 'shallower_marker', 'deeper_marker']) == expected
        assert c['mon16_depth_mm'] == c['mon16_age_cal_bp'] == '' and c['sample_transfer_allowed'] == 'false'
    assert len(contexts) == len(original) == 69
    # Verify the identifiable part of the ETH source text and the depth arithmetic.
    eth_pdf = fitz.open(source / 'hajdas1998.pdf')
    text = re.sub(r'\s+', ' ', eth_pdf[2].get_text())
    assert 'ETH-12464 LGM7.1 1382.5 360' in text
    eth = summary['eth12464']
    assert F(eth['epd_minus_source_depth_cm']) == F(eth['epd_listed_depth_cm']) - F(eth['source_composite_depth_cm']) == F(5, 2)
    assert summary['gate']['automatic_go_to_fossil_climate_run'] is False
    return {'passed': True, 'independent_method': 'Physical PDF column extraction, openpyxl cells, Fraction/bisect; no producer import',
            'pdf_appendix_rows_checked': len(rows), 'pdf_fields_checked': len(rows) * 5,
            'marker_chains_numerically_checked': 6, 'native_sample_contexts_checked': 69,
            'original_age_values_machine_extracted_from_hajdas_table': False,
            'hajdas_table_age_and_material_visually_checked': True,
            'source_hashes_checked': len(evidence['file_locks']), 'output_hashes_checked': 3}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--evidence', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(verify(a.source_dir, a.output_dir, a.evidence), indent=2))
