#!/usr/bin/env python3
"""Audit named tephra marker links, without interpolating sample depths or ages."""
import argparse
from collections import Counter
import csv
from decimal import Decimal as D
import hashlib
import json
from pathlib import Path

from import_data import read_pangaea, write_csv
from prepare_shared2025 import XlsxReader

EVIDENCE = Path(__file__).resolve().parents[2] / 'docs/thesis_research/marker_crosswalk_2026_10_30/evidence.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def position(depth, markers):
    """Order on the published native depth axis only; never a transfer function."""
    require(bool(markers), 'No marker depths')
    require(all(b[1] > a[1] for a, b in zip(markers, markers[1:])), 'Unordered/duplicate markers')
    for name, value in markers:
        if depth == value:
            return 'coincident_listed_depth', name, name
    if depth < markers[0][1]:
        return 'shallower_than_marker_span', '', markers[0][0]
    if depth > markers[-1][1]:
        return 'deeper_than_marker_span', markers[-1][0], ''
    for a, b in zip(markers, markers[1:]):
        if a[1] < depth < b[1]:
            return 'between_listed_marker_depths', a[0], b[0]
    raise ValueError('Unclassified native depth')


def json_value(v):
    if isinstance(v, D):
        return format(v.normalize(), 'f')
    raise TypeError(type(v).__name__)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_value) + '\n')


def run(source_dir, output, evidence_path=EVIDENCE):
    require(not output.exists(), 'Output must be new')
    evidence = json.loads(evidence_path.read_text())
    for name, lock in evidence['file_locks'].items():
        p = source_dir / name
        require(p.stat().st_size == lock['bytes'] and sha(p) == lock['sha256'], f'Source mismatch: {name}')
    _, th, tephras = read_pangaea(source_dir / 'monticchio_741058.tab')
    require(th[:3] == ['Depth sed [m]', 'Thick [mm]', 'Name'], 'Legacy depth/unit schema changed')
    by_name = {x['Name']: x for x in tephras}
    require(len(by_name) == len(tephras), 'Duplicate legacy layer names')
    reader = XlsxReader(source_dir / '1-s2.0-S0277379126003252-mmc1.xlsx')
    try:
        a1, a2 = dict(reader.rows(1)), dict(reader.rows(2))
    finally:
        reader.close()
    require(a1[6]['C'][0].endswith('(cm)'), 'MON16 composite unit changed')
    require(a1[6]['K'][0] == 'Depth in MON12 (cm)', 'MON12 depth schema changed')
    historical = {x['row']: x for x in evidence['wulf_appendix_rows']}
    links = []
    for link in evidence['marker_identities']:
        old, new = link['legacy_label'], link['modern_label']
        h = historical[link['appendix_row']]
        require(old in h['aliases'].split(' / '), f'No explicit original alias: {old}')
        src = by_name[old]
        r = link['a1_row']
        cells = a1[r]
        require(cells['A'][0].strip() == new, f'Changed A-1 label: {new}')
        legacy_cm, hist_cm, mon12_cm = D(src['Depth sed [m]']) * 100, D(h['composite_base_cm']), D(cells['K'][0])
        # 0.05 cm is a presentation-rounding comparison, never a depth uncertainty.
        require(abs(hist_cm - mon12_cm) <= D('.05'), f'Historical base mismatch: {old}')
        links.append({'legacy_layer': old, 'tm_label': new, 'legacy_listed_depth_cm': legacy_cm,
            'legacy_thickness_mm': src['Thick [mm]'], 'appendix_pdf_page': h['pdf_page'],
            'appendix_row': h['row'], 'appendix_aliases': h['aliases'], 'historical_core_section': h['core_section'],
            'historical_composite_base_cm': hist_cm, 'mon12_base_cm': mon12_cm,
            'listed_to_historical_base_difference_cm': hist_cm - legacy_cm,
            'mon16_top_cm': D(cells['D'][0]), 'mon16_base_cm': D(cells['C'][0]),
            'identity_status': 'published_marker_identity_chain',
            'sample_depth_transform_status': 'unresolved_depth_conventions_and_intermarker_mapping',
            'mon16_conflict': 'mercato_top_and_labelled_age_conflict' if old == 'L4' else ''})
    markers = [(x['legacy_layer'], x['legacy_listed_depth_cm'] / 100) for x in links]
    with (source_dir / 'pr21_sample_correspondence.csv').open() as f:
        samples = list(csv.DictReader(f))
    require(len(samples) == len({x['sample_id'] for x in samples}) == 69, 'Expected 69 unique samples')
    context = []
    for x in samples:
        category, younger, older = position(D(x['pollen_depth_m']), markers)
        context.append({'sample_id': x['sample_id'], 'pollen_depth_m': x['pollen_depth_m'],
            'legacy_correspondence_status': x['status'], 'native_depth_position': category,
            'shallower_marker': younger, 'deeper_marker': older,
            'mon16_depth_mm': '', 'mon16_age_cal_bp': '', 'sample_transfer_allowed': 'false'})
    _, _, ams = read_pangaea(source_dir / 'monticchio_740512.tab')
    eth = [x for x in ams if x['Lab no'] == 'ETH-12464']
    require(len(eth) == 1, 'Expected one ETH-12464')
    eth = eth[0]
    h = evidence['hajdas_ETH12464']
    require(D(eth['Age dated [ka BP]']) * 1000 == D(h['radiocarbon_age_bp']), 'ETH age mismatch')
    require(all(D(eth[k]) * 1000 == D(h['radiocarbon_error_years']) for k in ['Age e + [ka +]', 'Age e - [ka -]']), 'ETH error mismatch')
    eth_audit = {'lab_code': 'ETH-12464', 'sample_identified': h['sample'],
        'source_composite_depth_cm': h['composite_depth_cm'], 'epd_listed_depth_cm': D(eth['Depth sed [m]']) * 100,
        'epd_minus_source_depth_cm': D(eth['Depth sed [m]']) * 100 - D(h['composite_depth_cm']),
        'radiocarbon_age_bp': h['radiocarbon_age_bp'], 'error_years': h['radiocarbon_error_years'],
        'status': 'source_study_and_sample_identified_depth_transfer_unresolved',
        'single_original_core_identified': False, 'recalibrated': False}
    # Original historical layer at 878.47 cm corroborates A-1's rounded MON12
    # base, not the conflicting A-2 cell. This is evidence, not a correction.
    hist = D(historical[58]['composite_base_cm'])
    a1_base, a2_base = D(a1[49]['K'][0]), D(a2[16]['L'][0])
    source_comparison = {'tephra': 'TM-8-1b', 'historical_row': 58, 'historical_base_cm': hist,
        'a1_K49_cm': a1_base, 'a2_L16_cm': a2_base,
        'a1_minus_historical_cm': a1_base - hist, 'a2_minus_historical_cm': a2_base - hist,
        'interpretation': 'historical_numeric_support_for_A1_only_no_source_correction'}
    summary = {'schema_version': 1, 'date': '2026-10-30', 'audit_code_sha256': sha(Path(__file__)),
        'evidence_sha256': sha(evidence_path), 'source_locks': evidence['file_locks'],
        'marker_identity_chains': len(links), 'native_sample_context': dict(Counter(x['native_depth_position'] for x in context)),
        'eth12464': eth_audit, 'tm8_1b_historical_comparison': source_comparison,
        'gate': {'phase': 'P1', 'complete_sample_depth_crosswalk': False,
            'marker_identity_links_available': 6, 'samples_with_new_mon16_age': 0,
            'automatic_go_to_fossil_climate_run': False,
            'remaining': ['Original pollen core/section-to-composite mapping and depth conventions',
                          'Intermarker correlation uncertainty', 'MON16 Mercato source conflict and A-3 interval inconsistencies']}}
    output.mkdir(parents=True)
    for name, rows in [('marker_links.csv', links), ('sample_marker_context.csv', context)]:
        serial = [{k: json_value(v) if isinstance(v, D) else v for k, v in x.items()} for x in rows]
        write_csv(output / name, list(rows[0]), serial)
    save_json(output / 'summary.json', summary)
    save_json(output / 'reference_manifest.json', {p.name: sha(p) for p in sorted(output.iterdir())})
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--evidence', type=Path, default=EVIDENCE)
    a = p.parse_args()
    print(json.dumps(run(a.source_dir, a.output_dir, a.evidence)['gate']))
