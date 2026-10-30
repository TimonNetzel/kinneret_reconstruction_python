#!/usr/bin/env python3
"""Independent verification from the full SQL COPY stream, never executing SQL.

The producer reads separate table dumps; this verifier reads dumpall SQL.gz.
No producer imports. Exact Fraction arithmetic for all data comparisons.
"""
import argparse
import csv
from fractions import Fraction as F
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile


def sha(b):
    return hashlib.sha256(b).hexdigest()


def csv_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def pangaea(path):
    return list(csv.DictReader(io.StringIO(path.read_text().split('*/', 1)[1].strip()), delimiter='\t'))


def full_dump(archive, wanted):
    rows, totals, fields = {}, {}, {}
    with zipfile.ZipFile(archive) as z:
        with gzip.GzipFile(fileobj=io.BytesIO(z.read('dumpall_epd_db.sql.gz'))) as f:
            active = None
            for raw in f:
                line = raw.decode('latin1').rstrip('\r\n')
                if active is not None:
                    if line == r'\.':
                        active = None
                        continue
                    if active in wanted:
                        totals[active] += 1
                        values = line.split('\t')
                        if active == 'p_vars' or values[0] == '932':
                            assert len(values) == len(fields[active])
                            rows[active].append(dict(zip(fields[active], values)))
                    continue
                m = re.fullmatch(r'COPY (\w+) \(([^)]+)\) FROM stdin;', line)
                if m:
                    active = m[1]
                    if active in wanted:
                        assert active not in rows
                        rows[active], totals[active] = [], 0
                        fields[active] = m[2].split(', ')
    assert set(rows) == wanted
    return rows, totals


def verify(source, output):
    summary = json.loads((output / 'summary.json').read_text())
    for name, lock in summary['source_locks'].items():
        p = source / name
        assert p.stat().st_size == lock['bytes'] and sha(p.read_bytes()) == lock['sha256']
    for name, expected in json.loads((output / 'reference_manifest.json').read_text()).items():
        assert sha((output / name).read_bytes()) == expected
    inventory = csv_rows(output / 'archive_table_status.csv')
    db, totals = full_dump(source / 'epd-postgres-distribution-20171031.zip', {x['table'] for x in inventory})
    for x in inventory:
        assert totals[x['table']] == int(x['archive_rows'])
        assert len(db[x['table']]) == int(x['selected_rows'])
    samples = {int(x['sample_']): x for x in db['p_sample']}
    variables = {int(x['var_']): x for x in db['p_vars']}
    values = {(int(x['sample_']), int(x['var_'])): F(x['count']) for x in db['p_counts']}
    assert len(values) == len(db['p_counts']) == 8333
    assert sum(x == 0 for x in values.values()) == 48
    mapping = csv_rows(output / 'variable_export_map.csv')
    assert len(mapping) == len({x['epd_variable_id'] for x in mapping}) == 135
    for m in mapping:
        v = variables[int(m['epd_variable_id'])]
        assert v['varcode'] == m['epd_variable_code'] and v['varname'] == m['epd_variable_name']
    pollen = pangaea(source / 'monticchio_739452.tab')
    fingerprints = json.loads((output / 'numeric_fingerprints.json').read_text())
    assert len(pollen) == len(samples) == len(fingerprints) == 245
    expected_rounding = []
    for s, row in enumerate(pollen, 1):
        nums = []
        for m in mapping:
            actual = F(row[m['export_column']])
            assert actual == values.get((s, int(m['epd_variable_id'])), F(0)), (s, m['export_column'])
            # All 135 fields in this source are integral, including lab quantities.
            assert actual.denominator == 1
            nums.append(str(actual.numerator))
        assert fingerprints[s - 1]['sample_number'] == s
        assert fingerprints[s - 1]['sha256_joined_values'] == sha((','.join(nums) + '\n').encode())
        exact_m = F(samples[s]['depthcm']) / 100
        exported = F(row['Depth sed [m]'])
        assert abs(exported - exact_m) <= F(1, 200)
        if exported != exact_m:
            expected_rounding.append((s, exact_m, exported, (exported - exact_m) * 1000))
    rounding = csv_rows(output / 'depth_rounding.csv')
    assert len(rounding) == len(expected_rounding) == 19
    for x, expected in zip(rounding, expected_rounding):
        assert (int(x['sample_number']), F(x['archive_midpoint_m']),
                F(x['pangaea_export_depth_m']), F(x['export_minus_archive_mm'])) == expected
    cut = {F(x['Depth sed [m]']): x for x in pangaea(source / 'epd_ages_cut_804596.tab') if x['ID'] == '932'}
    identities = csv_rows(output / 'sample_identity.csv')
    prior = csv_rows(source / 'pr21_sample_correspondence.csv')
    assert len(identities) == len(prior) == len(cut) == 69
    for x, old in zip(identities, prior):
        s = int(x['epd_sample_number'])
        assert s == int(old['source_row']) and x['pilot_sample_id'] == old['sample_id']
        assert F(x['archive_midpoint_m']) == F(samples[s]['depthcm']) / 100 == F(x['epd_cut_depth_m'])
        c = cut[F(x['archive_midpoint_m'])]
        assert F(x['epd_cut_age_bp']) == F(c['Age dated [ka BP]']) * 1000 == F(old['epd_cut_age_bp'])
        assert x['original_core_section'] == x['mon16_age_cal_bp'] == ''
        assert x['new_age_transfer_allowed'] == 'false'
    assert not db['section'] and not db['coredriv']
    assert all(x['notes'] == x['thickness'] == r'\N' for x in samples.values())
    assert db['entity'][0]['c14depthadj'] == r'\N'
    chron = {int(x['chron_']): x for x in db['chron']}
    assert chron[1]['defaultchron'] == 'N' and chron[2]['defaultchron'] == 'Y'
    for x in summary['legacy_chronologies']:
        assert x['source_notes_sha256'] == sha(chron[x['number']]['notes'].encode())
    ages = {(int(x['chron_']), int(x['sample_'])): x for x in db['p_agedpt']}
    differences = []
    for x in csv_rows(output / 'legacy_age_comparison.csv'):
        s = int(x['epd_sample_number'])
        delta = F(ages[1, s]['agebp']) - F(x['retained_epd_cut_age_bp'])
        assert delta == F(x['chron1_minus_retained_years'])
        assert F(ages[2, s]['agebp']) == F(x['archive_default_chron2_reverse_calibrated_14c_age_bp'])
        assert x['replacement_allowed'] == 'false'
        differences.append(abs(delta))
    assert sorted(differences)[34] == F(summary['chron1_minus_cut_abs_years']['median'])
    assert max(differences) == F(summary['chron1_minus_cut_abs_years']['max'])
    assert summary['gate']['automatic_go_to_fossil_climate_run'] is False
    return {'passed': True, 'independent_reader': 'Full SQL.gz COPY stream instead of per-table ZIP; Fraction arithmetic; no SQL execution or producer import',
            'source_files_checked': 5, 'archive_tables_independently_checked': len(inventory),
            'numeric_cells_checked': 33075, 'sample_fingerprints_checked': 245,
            'exact_pilot_depth_and_age_joins_checked': 69, 'legacy_age_comparisons_checked': 69,
            'depth_rounding_rows_checked': 19, 'explicit_zeros_preserved': 48,
            'output_hashes_checked': 7}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(verify(a.source_dir, a.output_dir), indent=2))
