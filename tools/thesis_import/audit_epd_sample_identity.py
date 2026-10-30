#!/usr/bin/env python3
"""Read-only EPD snapshot audit. SQL is parsed as text, never executed.

Confirms archival sample identifiers, complete numeric export identity and depth
precision; does not replace chronologies or infer missing core sections.
"""
import argparse
from collections import Counter
import csv
from decimal import Decimal as D, ROUND_HALF_EVEN
import hashlib
import io
import json
from pathlib import Path
import re
import statistics
import zipfile

from import_data import read_pangaea, write_csv

ARCHIVE = 'epd-postgres-distribution-20171031.zip'
LOCKS = {
    ARCHIVE: (29210271, '6ce720d51e0cf92f3f207ed4527d6cda00dd565cecbb8d3793112a723dbfc072'),
    'pollen-database-manual-20071011.pdf': (615953, '98ef7675888d46060e2be52c9a7fcc3ee9df6aade8474ab6fcafc5076bd1fb3f'),
    'monticchio_739452.tab': (86174, '46ab6cea3a636d1a25b8c15f4b9b79a94c8fd1bd41f2943ae5c8941f9050a4de'),
    'epd_ages_cut_804596.tab': (3613957, 'd332d5c7921b914e84da689c918607befdb162f5c68bbcce9cd3534d1f5e58e0'),
    'pr21_sample_correspondence.csv': (9369, 'b3b9247594539bd0afd858d92fde2172bc3ff8b199c63bd106b506f82a122f83'),
}
TABLES = ['entity', 'p_entity', 'p_sample', 'p_counts', 'p_vars', 'section',
          'coredriv', 'chron', 'p_agedpt', 'geochron', 'c14', 'p_anldpt', 'alsegs']
# Explicit archive/export column correspondences for this dataset only. These
# do not authorize changing the Shared37 taxonomy or merging other variables.
ALIASES = {'Can': 'Canae', 'Cicoid': 'Cm.Sf.C', 'Euc': 'Euc(ct)', 'Lil-T': 'Lilae-T',
           'Cen.cfc-T': 'Cen.cfcT', 'Cen.cfn-T': 'Cen.cfnT', 'Cen.cfs-T': 'Cen.cfsT',
           'trilete spores': 'Trl.s'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def copy_field(value):
    """Postgres COPY text NULL and escapes, preserving a literal escaped \\N."""
    if value == r'\N':
        return None
    escapes = {'b': '\b', 'f': '\f', 'n': '\n', 'r': '\r', 't': '\t', 'v': '\v', '\\': '\\'}
    return re.sub(r'\\([0-7]{1,3}|.)', lambda m: chr(int(m[1], 8))
                  if m[1][0] in '01234567' else escapes.get(m[1], m[1]), value)


def archive_tables(path):
    selected, inventory = {}, []
    with zipfile.ZipFile(path) as outer:
        require(outer.testzip() is None, 'Archive CRC failure')
        schema = outer.read('dump_epd_db_schema_only.sql').decode('latin1')
        require("SET client_encoding = 'LATIN1'" in schema, 'Unexpected archive encoding')
        definitions = {}
        for m in re.finditer(r'CREATE TABLE (\w+) \((.*?)\n\);', schema, re.S):
            definitions[m[1]] = [x.strip().split()[0].rstrip(',') for x in m[2].splitlines() if x.strip()]
        with zipfile.ZipFile(io.BytesIO(outer.read('dumps_epd_all_tables.zip'))) as inner:
            for table in TABLES:
                fields = definitions[table]
                total, rows = 0, []
                with inner.open(table + '.dump') as f:
                    for line in f:
                        total += 1
                        if table != 'p_vars' and not line.startswith(b'932\t'):
                            continue
                        values = line.decode('latin1').rstrip('\r\n').split('\t')
                        require(len(values) == len(fields), f'Field count: {table}')
                        rows.append(dict(zip(fields, map(copy_field, values))))
                selected[table] = rows
                inventory.append({'table': table, 'archive_rows': total,
                                  'selected_rows': len(rows),
                                  'selection': 'all_variable_definitions' if table == 'p_vars' else 'entity_932'})
    return selected, inventory


def rounded_metres(depth_cm):
    # A diagnostic consistent with all observed exports, not proof of the
    # historical exporter implementation. Preserve Decimal originals elsewhere.
    return D(format(float(depth_cm) / 100, '.2f'))


def json_value(v):
    if isinstance(v, D):
        return format(v.normalize(), 'f')
    raise TypeError(type(v).__name__)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_value) + '\n')


def save_csv(path, rows):
    write_csv(path, list(rows[0]), [{k: json_value(v) if isinstance(v, D) else v
                                 for k, v in row.items()} for row in rows])


def run(source, output):
    require(not output.exists(), 'Output must be new')
    for name, (size, expected) in LOCKS.items():
        p = source / name
        require(p.stat().st_size == size and sha(p.read_bytes()) == expected, f'Source mismatch: {name}')
    tables, inventory = archive_tables(source / ARCHIVE)
    samples = {int(x['sample_']): x for x in tables['p_sample']}
    require(len(samples) == 245 and set(samples) == set(range(1, 246)), 'Sample IDs changed')
    counts = {}
    for row in tables['p_counts']:
        key = int(row['sample_']), int(row['var_'])
        require(key not in counts, 'Duplicate count key')
        require(row['count'] is not None and D(row['count']).is_finite()
                and D(row['count']) >= 0, 'Invalid numeric value')
        counts[key] = D(row['count'])
    by_code = {}
    for row in tables['p_vars']:
        require(row['varcode'] not in by_code, 'Duplicate archive variable code')
        by_code[row['varcode']] = row
    _, header, pollen = read_pangaea(source / 'monticchio_739452.tab')
    require(len(pollen) == len(samples), 'Export sample count mismatch')
    mapping = []
    for column in header[1:]:
        code = column.split(' [')[0]
        archive_code = ALIASES.get(code, code)
        v = by_code[archive_code]
        mapping.append({'export_column': column, 'epd_variable_id': int(v['var_']),
            'epd_variable_code': archive_code, 'epd_variable_name': v['varname'],
            'mapping': 'explicit_export_archive_alias' if code in ALIASES else 'exact_variable_code'})
    require(len(mapping) == len({x['epd_variable_id'] for x in mapping}) == 135, 'Variable map not one-to-one')
    require({v for _, v in counts} == {x['epd_variable_id'] for x in mapping}, 'Unmapped target variables')
    depth_differences, fingerprints = [], []
    for sample_id, row in enumerate(pollen, 1):
        numeric = []
        for m in mapping:
            value = D(row[m['export_column']])
            require(value == counts.get((sample_id, m['epd_variable_id']), D(0)),
                    f'Value mismatch sample {sample_id}, {m["export_column"]}')
            numeric.append(json_value(value))
        original_m = D(samples[sample_id]['depthcm']) / 100
        exported_m = D(row['Depth sed [m]'])
        require(rounded_metres(samples[sample_id]['depthcm']) == exported_m, 'Depth rounding mismatch')
        if original_m != exported_m:
            depth_differences.append({'sample_number': sample_id, 'archive_midpoint_m': original_m,
                'pangaea_export_depth_m': exported_m, 'export_minus_archive_mm': (exported_m - original_m) * 1000})
        fingerprints.append({'sample_number': sample_id, 'numeric_columns': len(numeric),
                             'sha256_joined_values': sha((','.join(numeric) + '\n').encode())})
    with (source / 'pr21_sample_correspondence.csv').open() as f:
        prior = list(csv.DictReader(f))
    _, _, cut = read_pangaea(source / 'epd_ages_cut_804596.tab')
    cut = {D(x['Depth sed [m]']): x for x in cut if x['ID'] == '932'}
    require(len(cut) == len(prior) == 69, 'Expected 69 pilot samples')
    ages = {(int(x['chron_']), int(x['sample_'])): x for x in tables['p_agedpt']}
    chron = {int(x['chron_']): x for x in tables['chron']}
    require(set(chron) == {1, 2}, 'Legacy chronology IDs changed')
    default = [i for i, x in chron.items() if x['defaultchron'] == 'Y']
    require(default == [2], 'Legacy default flag changed')
    identities, comparisons = [], []
    for old in prior:
        s = int(old['source_row'])
        r = samples[s]
        depth_m = D(r['depthcm']) / 100
        c = cut[depth_m]
        require(depth_m == D(old['chronology_depth_m']), 'Pilot exact depth mismatch')
        require(D(c['Age dated [ka BP]']) * 1000 == D(old['epd_cut_age_bp']), 'Retained age changed')
        identities.append({'pilot_sample_id': old['sample_id'], 'epd_entity': 932, 'epd_sample_number': s,
            'archive_midpoint_cm': r['depthcm'], 'archive_midpoint_m': depth_m,
            'pangaea_export_depth_m': old['pollen_depth_m'],
            'export_minus_archive_mm': (D(old['pollen_depth_m']) - depth_m) * 1000,
            'epd_cut_depth_m': old['chronology_depth_m'], 'epd_cut_age_bp': old['epd_cut_age_bp'],
            'database_sample_identity': 'confirmed_all_135_numeric_values',
            'original_core_section': '', 'core_section_status': 'not_stored_for_entity_932',
            'mon16_age_cal_bp': '', 'new_age_transfer_allowed': 'false'})
        legacy = D(ages[1, s]['agebp'])
        comparisons.append({'pilot_sample_id': old['sample_id'], 'epd_sample_number': s,
            'retained_epd_cut_age_bp': old['epd_cut_age_bp'], 'archive_chron1_calendar_age_bp': legacy,
            'chron1_minus_retained_years': legacy - D(old['epd_cut_age_bp']),
            'archive_default_chron2_reverse_calibrated_14c_age_bp': ages[2, s]['agebp'],
            'replacement_allowed': 'false'})
    require(not tables['section'] and not tables['coredriv'], 'New section/drive evidence requires review')
    differences = [abs(x['chron1_minus_retained_years']) for x in comparisons]
    summary = {'schema_version': 1, 'date': '2026-10-30', 'entity': 932,
        'audit_code_sha256': sha(Path(__file__).read_bytes()),
        'source_locks': {n: {'bytes': s, 'sha256': h} for n, (s, h) in LOCKS.items()},
        'numeric_export_identity': {'samples': 245, 'variables': 135, 'cells': 33075,
            'archive_value_rows': len(counts), 'explicit_zero_rows': sum(v == 0 for v in counts.values()),
            'cell_differences': 0, 'source_variable_map': dict(Counter(x['mapping'] for x in mapping))},
        'depth_precision': {'binary64_2decimal_format_diagnostic_matches_all_245': True,
            'historical_export_algorithm_proven': False,
            'exact_decimal_half_even_mismatches': sum((D(samples[i]['depthcm']) / 100).quantize(
                D('.01'), rounding=ROUND_HALF_EVEN) != D(row['Depth sed [m]']) for i, row in enumerate(pollen, 1)),
            'full_export_depth_differences': len(depth_differences),
            'pilot_depth_differences': sum(D(x['export_minus_archive_mm']) != 0 for x in identities),
            'pilot_exact_depth_matches_cut': 69},
        'missing_metadata': {'section_rows': len(tables['section']), 'coredriv_rows': len(tables['coredriv']),
            'sample_notes_nonnull': sum(x['notes'] is not None for x in samples.values()),
            'sample_thickness_nonnull': sum(x['thickness'] is not None for x in samples.values()),
            'c14depthadj': tables['entity'][0]['c14depthadj']},
        'legacy_chronologies': [{'number': i, 'name': x['name'], 'default': x['defaultchron'],
            'prepared_date': x['dateprepared'], 'model': x['model'],
            'source_notes_sha256': sha(x['notes'].encode()),
            'declared_age_basis': 'lamination_calendar_years' if i == 1 else
                'reverse_calibrated_14c_with_documented_calendar_reversion'} for i, x in chron.items()],
        'chron1_minus_cut_abs_years': {'median': statistics.median(differences), 'max': max(differences)},
        'gate': {'database_sample_ids_confirmed': 69, 'original_core_sections_identified': 0,
            'new_mon16_ages': 0, 'automatic_go_to_fossil_climate_run': False,
            'status': 'sample_ids_and_midpoints_confirmed_physical_core_crosswalk_missing'}}
    output.mkdir(parents=True)
    for name, rows in [('variable_export_map.csv', mapping), ('sample_identity.csv', identities),
                       ('legacy_age_comparison.csv', comparisons), ('depth_rounding.csv', depth_differences),
                       ('archive_table_status.csv', inventory)]:
        save_csv(output / name, rows)
    save_json(output / 'numeric_fingerprints.json', fingerprints)
    save_json(output / 'summary.json', summary)
    save_json(output / 'reference_manifest.json', {p.name: sha(p.read_bytes()) for p in sorted(output.iterdir())})
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.source_dir, a.output_dir)['gate']))
