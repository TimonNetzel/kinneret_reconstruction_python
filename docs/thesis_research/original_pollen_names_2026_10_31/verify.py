"""Verify saved name chains directly against locked originals; no producer import."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import subprocess

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def read(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def verify(raw_dir, output):
    manifest = json.loads((output / 'manifest.json').read_text())
    registered = json.loads((ROOT / 'docs/thesis_research/pollen_taxon_evidence_2026_10_31/sources.json').read_text())
    locks = {s['source_id']: s for s in registered['registered_primary_sources']}
    originals = {}
    for source in manifest['source_files']:
        path = raw_dir / source['file']
        lock = locks[source['source_id']]
        check(digest(path) == source['sha256'] == lock['sha256'], 'Original SHA256 mismatch')
        check(path.stat().st_size == source['bytes'] == lock['bytes'], 'Original byte size mismatch')
        check(source['url'] == lock['url'] and source['doi'] == lock['doi'], 'Original source metadata mismatch')
        originals[source['source_id']] = path
    check(set(originals) == {'empd2', 'monticchio_739452'}, 'Expected two originals')
    for name, sha in manifest['repository_inputs'].items():
        path = ROOT / name
        check(digest(path) == sha, 'Repository input SHA256 mismatch')
        check(subprocess.check_output(['git', 'show', manifest['base_commit'] + ':' + name], cwd=ROOT) == path.read_bytes(), 'Base input differs')
    for name, sha in manifest['outputs_sha256'].items():
        check(digest(output / name) == sha, 'Output SHA256 mismatch: ' + name)
    with sqlite3.connect(originals['empd2'].resolve().as_uri() + '?mode=ro', uri=True) as con:
        con.row_factory = sqlite3.Row
        sql_rows = {r['var_']: dict(r) for r in con.execute('SELECT * FROM p_vars')}
        schema = [dict(r) for r in con.execute('PRAGMA table_info(p_vars)')]
    check(len(sql_rows) == manifest['p_vars_total_rows'] == 5230, 'Dictionary size')
    check(schema == manifest['p_vars_schema'], 'Dictionary schema')
    check({r['name'] for r in schema} == {'var_', 'acc_var_', 'original_varname', 'acc_varname', 'groupid'}, 'Unexpected taxon-note fields')
    check(manifest['p_vars_taxon_notes_column_present'] is False, 'Unsupported note claim')
    old = {r['model_id']: r for r in read(ROOT / 'docs/thesis_research/pollen_taxon_evidence_2026_10_31/inventory.csv')}
    chain_rows = read(output / 'empd_name_chains.csv')
    definitions = read(output / 'monticchio_definitions.csv')
    links = read(output / 'species_name_links.csv')
    for table in [definitions, links]:
        check(len(table) == 7 and {r['model_id'] for r in table} == set(old), 'Missing/duplicate target')
    check(len({(r['model_id'], r['var_']) for r in chain_rows}) == len(chain_rows), 'Duplicate chain rows')
    # Different lookup path: tokenize every name, collect exact genus tokens, and
    # demand every reference target is present, with no extra unexplained row.
    query_counts = []
    for key, previous in old.items():
        genus = previous['model_species'].split()[0]
        name_ids = {code for code, record in sql_rows.items()
                    if genus.casefold() in set(re.findall(r'\w+', ((record['original_varname'] or '') + ' ' + (record['acc_varname'] or '')).casefold()))}
        selected = set(name_ids)
        while True:
            expanded = selected | {sql_rows[code]['acc_var_'] for code in selected if sql_rows[code]['acc_var_'] in sql_rows}
            if expanded == selected:
                break
            selected = expanded
        actual = [r for r in chain_rows if r['model_id'] == key]
        check({int(r['var_']) for r in actual} == selected, 'Incomplete lexical/reference coverage: ' + key)
        for row in actual:
            code = int(row['var_'])
            source = sql_rows[code]
            for field in ['var_', 'acc_var_', 'original_varname', 'acc_varname', 'groupid']:
                expected = '' if source[field] is None else str(source[field])
                check(row[field] == expected, 'Altered original SQL field: ' + key + ':' + field)
            check(row['query_genus'] == genus, 'Query genus')
            check(row['locator'] == 'p_vars WHERE var_=' + str(code), 'SQL locator')
            check(row['selection_role'] == ('lexical_name_hit' if code in name_ids else 'referenced_target_only'), 'Selection role')
            for field, flag in [('original_varname', 'original_name_hit'), ('acc_varname', 'accepted_name_hit')]:
                hit = genus.casefold() in re.findall(r'\w+', (source[field] or '').casefold())
                check(row[flag] == str(hit).lower(), 'Field-specific lexical-hit flag')
            ids = [int(n) for n in row['chain_ids'].split('|')]
            check(ids[0] == code and len(set(ids)) == len(ids), 'Chain start or repetition')
            check(all(sql_rows[left]['acc_var_'] == right for left, right in zip(ids, ids[1:])), 'Broken accepted-ID edge')
            check(sql_rows[ids[-1]]['acc_var_'] == ids[-1] and row['terminal_status'] == 'self_reference', 'Unclosed chain')
            check(row['approved_species_mapping'] == 'false', 'Unexpected species approval')
        query_counts.append(dict(model_id=key, lexical_name_hits=len(name_ids), rows_with_referenced_targets=len(selected)))
        link = next(r for r in links if r['model_id'] == key)
        exact = [r for r in sql_rows.values() if r['original_varname'] == previous['model_species']]
        check(len(exact) == 1, 'Exact species-name match cardinality')
        source = exact[0]
        check(link['model_species'] == previous['model_species'], 'Altered model name')
        for field in ['original_varname', 'acc_var_', 'acc_varname']:
            check(link[field] == str(source[field]), 'Species-name link mismatch')
        check(link['original_var_'] == str(source['var_']), 'Species-name original key')
        check(link['accepted_record_original_varname'] == sql_rows[source['acc_var_']]['original_varname'], 'Accepted record locator')
        check(link['catalogue_candidate_acc_var_'] == previous['empd2_candidate_accepted_id'], 'Changed catalogue candidate')
        check(link['fossil_column'] == previous['fossil_column'], 'Changed fossil column')
        check(link['same_id_as_catalogue_candidate'] == str(str(source['acc_var_']) == previous['empd2_candidate_accepted_id']).lower(), 'False ID equality')
        check(link['morphological_equivalence_established'] == link['approved_species_mapping'] == 'false', 'Unexpected morphology claim')
    lines = originals['monticchio_739452'].read_text(encoding='utf-8').splitlines()
    metadata_end = lines.index('*/')
    for row in definitions:
        previous = old[row['model_id']]
        line_index = int(row['metadata_line']) - 1
        check(line_index < metadata_end, 'Definition outside source metadata')
        source = lines[line_index]
        check(source.strip().startswith(row['original_parameter'] + ' [#] (' + row['abbreviation'] + ') * PI:'), 'Parameter definition locator')
        check(source.split(' * COMMENT: ', 1)[1] == row['archive_comment'] == previous['fossil_archive_comment'], 'Archive comment mismatch')
        check(row['original_parameter'] == previous['fossil_original_name'], 'Original parameter mismatch')
        check(row['fossil_column'] == previous['fossil_column'], 'Fossil column mismatch')
        check(row['approved_species_mapping'] == 'false', 'Unexpected mapping approval')
        check(row['archive_comment_role'] == 'accepted_EPD_name_not_proven_original_identification', 'Archive context claim')
        check(row['species_identification_protocol_in_parameter'] == 'not_stated', 'Unsupported identification protocol')
        check(not any('METHOD:' in part for part in source.split(' * ')), 'Identification-method claim requires review')
        headers = next(csv.reader([lines[int(row['data_header_line']) - 1]], delimiter='\t'))
        check(int(row['data_header_line']) > metadata_end + 1, 'Header locator outside data section')
        check(headers[int(row['data_column_1based']) - 1] == row['fossil_column'], 'Data column locator')
        context = lines[int(row['archive_context_line']) - 1]
        check('accepted species name is given in the parameter comment' in context, 'Unsubstantiated comment-role claim')
        check(manifest['monticchio_archive_context'] == dict(line=int(row['archive_context_line']), text=context.split('\t', 1)[1]), 'Context manifest differs')
    check(query_counts == manifest['query_counts'], 'Query-count report differs')
    check(len(chain_rows) == manifest['extracted_chain_rows'], 'Chain-row count differs')
    check(len({r['var_'] for r in chain_rows}) == manifest['unique_dictionary_rows'], 'Unique dictionary count differs')
    check(manifest['count_values_parsed_or_used'] is False, 'Count-use claim')
    return dict(result='passed', scope='metadata_and_database_name_links_only', source_hashes_checked=2,
                repository_input_hash_and_base_checks=3, dictionary_rows_scanned=5230,
                fossil_definitions_checked=len(definitions), dictionary_chain_rows_checked=len(chain_rows),
                unique_dictionary_rows_checked=len({r['var_'] for r in chain_rows}), species_name_links_checked=len(links),
                lexical_seed_rows_across_queries=sum(q['lexical_name_hits'] for q in query_counts),
                unclosed_selected_chains=0, new_species_mappings_approved=0, models_fitted=0,
                palynological_identification_review=False,
                outputs_sha256={p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != 'verification.json'},
                audit_code_sha256=digest(Path(__file__)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = verify(args.raw_dir, args.output_dir)
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if not k.endswith('sha256')}, ensure_ascii=False))
