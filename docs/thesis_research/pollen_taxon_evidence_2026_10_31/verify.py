"""Local provenance audit; no source downloads, model imports, fits or taxonomy approval."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify():
    registry = json.loads((HERE / 'sources.json').read_text())
    for entry in registry['inputs']:
        data = (ROOT / entry['path']).read_bytes()
        require(hashlib.sha256(data).hexdigest() == entry['sha256'], entry['path'])
        stored = subprocess.check_output(['git', 'show', registry['base_commit'] + ':' + entry['path']], cwd=ROOT)
        require(data == stored, 'Input differs from fixed base: ' + entry['path'])
    research = ROOT / 'docs/thesis_research'
    taxa = json.loads((research / 'taxon_transfer_2026_10_30/config.json').read_text())['taxa']
    fossil = {r['original_name']: r for r in read_csv(research / 'taxon_decisions_2026_10_29/fossil_decisions.csv')}
    features = {r['feature_id']: r for r in read_csv(research / 'modern2025_2026_10_29/feature_evidence.csv')}
    shared = {r['feature_id']: r for r in json.loads((research / 'modern2025_2026_10_29/shared_features.json').read_text())['features']}
    rows = read_csv(HERE / 'inventory.csv')
    require(len(rows) == len(taxa) == 7, 'Expected exactly seven targets')
    require(len({r['model_id'] for r in rows}) == 7, 'Duplicate model IDs')
    require({r['model_id'] for r in rows} == {t['id'] for t in taxa}, 'Target set differs')
    evidence = {e['evidence_id']: e for e in registry['evidence']}
    require(len(evidence) == len(registry['evidence']) == 21, 'Expected 21 unique locators')
    for t in taxa:
        row = next(r for r in rows if r['model_id'] == t['id'])
        genus = t['name'].split()[0]
        f, feature = fossil[genus], features[genus.lower()]
        expected = {'model_species': t['name'], 'botanical_source_name': t['source_name'],
                    'bfn_taxon_id': str(t['taxon_id']), 'bfn_name_use_id': str(t['name_use_id_orig']),
                    'empd2_candidate_accepted_id': f['candidate_modern_ids'],
                    'empd2_candidate_accepted_name': f['candidate_modern_names'],
                    'shared37_feature': feature['feature_id'], 'published2025_code': feature['modern_codes'],
                    'published2025_label': feature['published_labels'],
                    'published_member_count': feature['published_member_count'],
                    'published_member_names': feature['member_examples']}
        expected.update({'fossil_column': f['original_column'], 'fossil_original_name': f['original_name'],
                         'fossil_archive_comment': f['archive_comment'], 'catalogue_action': f['action'],
                         'catalogue_scope': f['scope'], 'catalogue_scientific_status': f['scientific_status']})
        for field, value in expected.items():
            require(row[field] == value, t['id'] + ':' + field)
        require(f['scope'] == 'fossil_catalogue_only' and f['scientific_status'] == 'not_training_approved', 'Catalogue gate')
        require(row['fossil_model_mapping_approved'] == 'false', 'Unexpected approval')
        require(row['morphological_species_assignment'] == 'not_established', 'Unsupported morphology claim')
        relation = 'species_name_in_archive_comment_only' if genus == 'Taxus' else 'fossil_genus_broader_than_model_species'
        require(row['label_relation'] == relation, 'Incorrect label relation')
        require(row['gap_id'] == 'G-' + t['id'], 'Gap reference')
        require(row['gap_id'] in (HERE / 'SOURCE_GAP.md').read_text(), 'Missing per-species gap')
        require(shared[genus.lower()]['fossil_columns'] == [f['original_column']], 'Shared37 fossil mismatch')
        require(shared[genus.lower()]['modern_codes'] == [feature['modern_codes']], 'Shared37 code mismatch')
        names = row['published_member_names'].split('|')
        require(len(names) == int(row['published_member_count']) <= 8, 'Truncated/duplicate name list')
        require(len(set(names)) == len(names), 'Duplicate published names')
        refs = row['evidence_ids'].split('|')
        require(len(refs) == 3 and set(refs) == {t['id'] + ':' + k for k in ['model', 'fossil', 'modern']}, 'Evidence coverage')
        for ref in refs:
            entry = evidence[ref]
            path = ROOT / entry['path']
            require(entry['primary_bytes_retrieved_now'] is False, 'Unexpected primary retrieval claim')
            if entry['locator'].startswith('line:'):
                index = int(entry['locator'].split(':')[1]) - 2
                source_row = read_csv(path)[index]
                require(source_row == (f if ref.endswith(':fossil') else feature), 'Locator points to wrong row')
            else:
                pointer = entry['locator'].split('/')
                require(json.loads(path.read_text())[pointer[1]][int(pointer[2])] == t, 'Model pointer')
    historical = json.loads((research / 'taxon_decisions_2026_10_29/provenance.json').read_text())
    actual = hashlib.sha256((research / 'taxon_decisions_2026_10_29/fossil_decisions.csv').read_bytes()).hexdigest()
    require(actual == historical['outputs']['fossil_decisions.csv'], 'Historical fossil output hash')
    abies = read_csv(research / 'variable_provenance_2026_10_30/original_variable_definitions.csv')[0]
    require(abies['epd_original_name'] == 'Abies' and abies['epd_variable_id'] == '1', 'Historical Abies EPD record')
    original_catalog = json.loads((research / 'source_audit_2026_10_29/sources.json').read_text())['files']
    modern_catalog = json.loads((research / 'modern2025_2026_10_29/sources_2025.json').read_text())['files']
    require(len(registry['registered_primary_sources']) == 4, 'Expected four original registrations')
    for source in registry['registered_primary_sources']:
        old = next(r for r in original_catalog + modern_catalog if r['source_id'] == source['source_id'])
        require(source['sha256'] == old['sha256'] and source['bytes'] == old['byte_count'], 'Original-source hash metadata')
        require(source['url'] == old['download_url'] and source['doi'] == old['doi'], 'Original-source locator')
        require(source['hash_status'] == 'historical_registry_only_not_rehashed_now', 'Hash claim too strong')
    return {'result': 'passed', 'scope': 'repository_evidence_inventory_only',
            'base_commit': registry['base_commit'], 'repository_inputs_hash_and_base_checked': len(registry['inputs']),
            'model_targets_checked': len(rows), 'evidence_locators_checked': len(evidence),
            'fossil_genus_labels': 6, 'fossil_species_name_comments': 1,
            'new_species_mappings_approved': 0, 'registered_primary_sources_metadata_checked': 4,
            'original_files_retrieved_or_rehashed': 0, 'models_fitted': 0,
            'primary_access_limit': 'CONNECT HTTP 403 for doi.pangaea.de and store.pangaea.de',
            'independent_palynological_review': False,
            'package_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(HERE.iterdir()) if p.is_file() and p.name != 'verification.json'}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = verify()
    text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.report:
        args.report.write_text(text)
    print(json.dumps({k: v for k, v in report.items() if k != 'package_sha256'}, ensure_ascii=False))
