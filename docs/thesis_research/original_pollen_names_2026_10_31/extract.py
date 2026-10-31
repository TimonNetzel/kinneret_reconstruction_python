"""Extract locked source metadata only; never alter counts, names or scientific mappings."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESEARCH = ROOT / 'docs/thesis_research'
BASE = '98ef1bf980aaa71903fa8fe40ef7797bd9d16b87'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, records):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)


def extract(raw_dir, output):
    if output.exists():
        raise ValueError('Use a new output directory')
    sources = json.loads((RESEARCH / 'pollen_taxon_evidence_2026_10_31/sources.json').read_text())
    registrations = {s['source_id']: s for s in sources['registered_primary_sources']}
    files = {'empd2': raw_dir / 'EMPD2.sqlite3', 'monticchio_739452': raw_dir / 'monticchio_739452.tab'}
    source_checks = []
    for key, path in files.items():
        reg = registrations[key]
        if sha(path) != reg['sha256'] or path.stat().st_size != reg['bytes']:
            raise ValueError('Locked source mismatch: ' + key)
        source_checks.append(dict(source_id=key, file=path.name, sha256=reg['sha256'], bytes=reg['bytes'],
                                  url=reg['url'], doi=reg['doi'], license=reg['license'],
                                  byte_verification='matched_historical_lock_2026-10-31'))
    with sqlite3.connect(files['empd2'].resolve().as_uri() + '?mode=ro', uri=True) as con:
        con.row_factory = sqlite3.Row
        schema = [dict(r) for r in con.execute('PRAGMA table_info(p_vars)')]
        records = {r['var_']: dict(r) for r in con.execute('SELECT * FROM p_vars ORDER BY var_')}
    lines = files['monticchio_739452'].read_text(encoding='utf-8').splitlines()
    metadata_end = lines.index('*/')
    header_line = next(i for i in range(metadata_end + 1, len(lines)) if lines[i].strip())
    header = next(csv.reader([lines[header_line]], delimiter='\t'))
    context = next((i + 1, line.split('\t', 1)[1]) for i, line in enumerate(lines[:metadata_end]) if line.startswith('Comment:\t'))
    taxa = json.loads((RESEARCH / 'taxon_transfer_2026_10_30/config.json').read_text())['taxa']
    with (RESEARCH / 'pollen_taxon_evidence_2026_10_31/inventory.csv').open(encoding='utf-8') as stream:
        inventory = {r['model_id']: r for r in csv.DictReader(stream)}
    definitions, chains, links, query_counts = [], [], [], []
    for taxon in taxa:
        key, species = taxon['id'], taxon['name']
        genus = species.split()[0]
        old = inventory[key]
        pattern = re.compile(r'\b' + re.escape(genus) + r'\b', re.I)
        seeds = {code for code, row in records.items()
                 if any(pattern.search(row[field] or '') for field in ('original_varname', 'acc_varname'))}
        selected = set(seeds)
        for start in sorted(seeds):
            seen = set()
            code = start
            while code in records and code not in seen:
                selected.add(code)
                seen.add(code)
                code = records[code]['acc_var_']
        for code in sorted(selected):
            row = records[code]
            visited, current = [], code
            while current in records and current not in visited:
                visited.append(current)
                following = records[current]['acc_var_']
                if following == current:
                    status = 'self_reference'
                    break
                current = following
            else:
                status = 'cycle' if current in visited else 'null_reference' if current is None else 'missing_reference'
            chains.append(dict(model_id=key, query_genus=genus, var_=row['var_'], acc_var_=row['acc_var_'],
                               original_varname=row['original_varname'], acc_varname=row['acc_varname'], groupid=row['groupid'],
                               selection_role='lexical_name_hit' if code in seeds else 'referenced_target_only',
                               original_name_hit=str(bool(pattern.search(row['original_varname'] or ''))).lower(),
                               accepted_name_hit=str(bool(pattern.search(row['acc_varname'] or ''))).lower(),
                               chain_ids='|'.join(map(str, visited)), terminal_status=status,
                               locator='p_vars WHERE var_=' + str(code), approved_species_mapping='false'))
        exact = [r for r in records.values() if r['original_varname'] == species]
        if len(exact) != 1:
            raise ValueError('Missing or ambiguous exact source spelling: ' + species)
        row = exact[0]
        target = records.get(row['acc_var_'])
        links.append(dict(model_id=key, model_species=species, original_var_=row['var_'],
                          original_varname=row['original_varname'], acc_var_=row['acc_var_'], acc_varname=row['acc_varname'],
                          accepted_record_original_varname=target['original_varname'] if target else None,
                          fossil_column=old['fossil_column'], catalogue_candidate_acc_var_=old['empd2_candidate_accepted_id'],
                          same_id_as_catalogue_candidate=str(str(row['acc_var_']) == old['empd2_candidate_accepted_id']).lower(),
                          morphological_equivalence_established='false', approved_species_mapping='false'))
        abbr = old['fossil_column'].split(' [')[0]
        matches = [(i + 1, line) for i, line in enumerate(lines[:metadata_end]) if re.match(r'^\t.* \[#\] \(' + re.escape(abbr) + r'\) \* PI:', line)]
        if len(matches) != 1:
            raise ValueError('Ambiguous fossil parameter: ' + abbr)
        number, definition = matches[0]
        original = definition.strip().split(' [#] (', 1)[0]
        comment = definition.split(' * COMMENT: ', 1)[1]
        column = abbr + ' [#]'
        if column not in header or original != old['fossil_original_name'] or comment != old['fossil_archive_comment']:
            raise ValueError('Fossil inventory differs: ' + key)
        definitions.append(dict(model_id=key, fossil_column=column, original_parameter=original, abbreviation=abbr,
                                archive_comment=comment, metadata_line=number, data_header_line=header_line + 1,
                                data_column_1based=header.index(column) + 1, archive_context_line=context[0],
                                archive_comment_role='accepted_EPD_name_not_proven_original_identification',
                                species_identification_protocol_in_parameter='not_stated', approved_species_mapping='false'))
        query_counts.append(dict(model_id=key, lexical_name_hits=len(seeds), rows_with_referenced_targets=len(selected)))
    output.mkdir(parents=True)
    for name, rows in [('monticchio_definitions.csv', definitions), ('empd_name_chains.csv', chains), ('species_name_links.csv', links)]:
        write_csv(output / name, rows)
    input_paths = ['docs/thesis_research/pollen_taxon_evidence_2026_10_31/sources.json',
                   'docs/thesis_research/pollen_taxon_evidence_2026_10_31/inventory.csv',
                   'docs/thesis_research/taxon_transfer_2026_10_30/config.json']
    manifest = dict(checked_on='2026-10-31', base_commit=BASE, scope='original_metadata_and_name_chains_not_taxonomic_approval',
                    source_files=source_checks, repository_inputs={p: sha(ROOT / p) for p in input_paths},
                    p_vars_schema=schema, p_vars_total_rows=len(records), p_vars_taxon_notes_column_present=False,
                    monticchio_archive_context=dict(line=context[0], text=context[1]),
                    selection='case-insensitive whole-word genus search in both name fields over all p_vars, then recursive acc_var_ targets; lexical hits are not genus membership or approved aggregation pools',
                    query_counts=query_counts, extracted_chain_rows=len(chains),
                    unique_dictionary_rows=len({r['var_'] for r in chains}), count_values_parsed_or_used=False,
                    outputs_sha256={p.name: sha(p) for p in sorted(output.iterdir())})
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return dict(definitions=len(definitions), chain_rows=len(chains), species_links=len(links),
                unique_dictionary_rows=manifest['unique_dictionary_rows'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(extract(args.raw_dir, args.output_dir)))
