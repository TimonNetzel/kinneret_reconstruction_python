"""Post-hoc variable diagnostics on FIXED PR26 pairs; never a matcher or correction.

Uses the hash-locked PR26 converted reader. No R, SQL or age model is executed.
All quantities remain source values; balances are arithmetic hypotheses only.
"""
import argparse
import collections
import csv
from decimal import Decimal
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import zipfile

LOCKS = {
    'itDownload.RDS': '03536b6f1d3974de77ff997f6d86674a78882b2ebe305bb4e135e38289a2a829',
    'epd-postgres-distribution-20171031.zip': '6ce720d51e0cf92f3f207ed4527d6cda00dd565cecbb8d3793112a723dbfc072',
    'monticchio_739452.tab': '46ab6cea3a636d1a25b8c15f4b9b79a94c8fd1bd41f2943ae5c8941f9050a4de',
    'pr26_correspondence.csv': '08b90b71e5aa22b4f923a7f82579a2d18e6cc9d9425f551e6065a0709b71c9bf',
    'pr26_diagnostics.csv': '3368c3296e4166199ce26a1c9d54ac6a38ed7976fa2f83bec0bd1c9bee266373',
    'pr26_differences.csv': '9161157b577aeaf79876d658ab9587910395b45c3475c3dcaf0cbdceff76c345',
}
READER_SHA = 'bce9dc26e452fbc269e9c0c739bbc4f50deca7a2c102f85c121be01d898d6c54'
# Direct archive/export identifiers, not a new taxonomic policy.
FIELDS = {1: ('Abi', 1), 480: ('Pla', 576), 699: ('Pla.l-T', 580),
          2905: ('Pla.m/m', 583), 3590: ('Pla.c-T', 1478),
          284: ('Scoae', 763), 275: ('Sam', 715)}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reader(path):
    require(digest(path) == READER_SHA, 'Wrong PR26 reader')
    spec = importlib.util.spec_from_file_location('locked_pr26_reader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measurement(rows, tid):
    selected = [r for r in rows if int(r['taxonid']) == tid
                and r['units'] == 'NISP' and r['element'] == 'pollen'
                and r['elementtype'] == 'pollen'
                and all(r[k] is None for k in ('context', 'symmetry'))]
    require(all(r['value'] is not None for r in selected), 'NULL is not zero')
    values = {Decimal(str(r['value'])) for r in selected}
    require(all(v.is_finite() and v >= 0 for v in values), 'Invalid source value')
    require(len(values) <= 1, 'Conflicting duplicate source values')
    return next(iter(values), Decimal(0))


def read_epd(path):
    # Parse COPY dumps as inert text. Only two tables needed here.
    with zipfile.ZipFile(path) as z:
        require(z.testzip() is None, 'ZIP CRC failure')
        schema = z.read('dump_epd_db_schema_only.sql').decode('latin1')
        fields = re.search(r'CREATE TABLE p_vars \((.*?)\n\);', schema, re.S)
        names = [line.strip().split()[0].rstrip(',') for line in fields[1].splitlines() if line.strip()]
        with zipfile.ZipFile(io.BytesIO(z.read('dumps_epd_all_tables.zip'))) as inner:
            variables = {}
            for line in inner.read('p_vars.dump').decode('latin1').splitlines():
                row = dict(zip(names, [None if s == r'\N' else s for s in line.split('\t')]))
                require(len(line.split('\t')) == len(names), 'Wrong p_vars fields')
                key = int(row['var_'])
                require(key not in variables, 'Duplicate variable ID')
                variables[key] = row
            counts = {}
            with inner.open('p_counts.dump') as handle:
                for line in handle:
                    if not line.startswith(b'932\t'):
                        continue
                    e, sample, var, value = line.decode('latin1').rstrip('\r\n').split('\t')
                    key = int(sample), int(var)
                    require(key not in counts and value != r'\N', 'Duplicate or NULL count')
                    counts[key] = Decimal(value)
    return variables, counts


def table(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def csv_out(path, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def json_out(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def run(inputs, reader_path, output):
    require(not output.exists(), 'Use a new output directory')
    for name, expected in LOCKS.items():
        require(digest(inputs / name) == expected, 'Wrong input: ' + name)
    reader = load_reader(reader_path)
    site, info = reader.read_site(inputs / 'itDownload.RDS')
    datasets = {}
    for cu in reader.collection_values(site.collunits.collunits):
        for ds in reader.collection_values(cu.datasets.datasets):
            did = int(reader.one(ds.datasetid))
            if did in (26608, 47525):
                datasets[did] = {int(reader.one(s.sampleid)): s
                                for s in reader.collection_values(ds.samples.samples)}
    require([len(datasets[k]) for k in (26608, 47525)] == [245, 924], 'Dataset inventory')
    rows = {did: {sid: [{str(k): reader.atom(v) for k, v in r.items()}
                       for r in s.datum.to_dict('records')]
                 for sid, s in samples.items()} for did, samples in datasets.items()}
    variables, counts = read_epd(inputs / 'epd-postgres-distribution-20171031.zip')
    pg = list(csv.DictReader(io.StringIO((inputs / 'monticchio_739452.tab').read_text().split('*/', 1)[1].strip()), delimiter='\t'))
    prior = table(inputs / 'pr26_correspondence.csv')
    diagnostics = {p['pilot_sample_id']: p for p in table(inputs / 'pr26_diagnostics.csv')}
    differences = table(inputs / 'pr26_differences.csv')
    require(len(prior) == 69 and len(diagnostics) == 41 and len(differences) == 47, 'PR26 inventory')
    pairs = []
    pair_values = {}
    checked = 0
    for p in prior:
        pilot = p['pilot_sample_id']
        oldid = int(p['legacy_sampleid'])
        if p['candidate_sampleid']:
            newid, role = int(p['candidate_sampleid']), 'pr26_candidate'
        else:
            d = diagnostics[pilot]
            ids = json.loads(d['nearest_sampleids'])
            require(d['nearest_count'] == '1' and len(ids) == 1, 'Ambiguous fixed diagnostic')
            newid, role = int(ids[0]), 'pr26_diagnostic_only'
        oldrows, newrows = rows[26608][oldid], rows[47525][newid]
        old = {t: measurement(oldrows, t) for t in FIELDS}
        new = {t: measurement(newrows, t) for t in (*FIELDS, 4114, 3430, 476, 902)}
        sample_number = int(p['epd_sample_number'])
        for tid, (code, varid) in FIELDS.items():
            require(variables[varid]['varcode'] == code, 'Original variable changed')
            require(old[tid] == counts.get((sample_number, varid), Decimal(0))
                    == Decimal(pg[sample_number - 1][code + ' [#]']), 'Three-source value disagreement')
            checked += 1
        plantago = sum(old[t] for t in (480, 699, 2905, 3590))
        record = {'pilot_sample_id': pilot, 'legacy_sampleid': oldid,
                  'fixed_composite_sampleid': newid, 'pair_role': role,
                  'archived_analysisunitname': reader.one(datasets[47525][newid].analysisunitname),
                  'old_Pla': old[480], 'old_Pla_l_T': old[699], 'old_Pla_m_m': old[2905],
                  'old_Pla_c_T': old[3590], 'old_four_Plantago_fields_sum': plantago,
                  'new_Plantago': new[480], 'Plantago_balance_equal': plantago == new[480],
                  'old_Scrophulariaceae': old[284], 'new_Scrophulariaceae': new[284],
                  'new_Digitalis_type': new[4114], 'new_Veronica_type': new[3430],
                  'new_Pedicularis': new[476], 'Scroph_Digitalis_value_equal': old[284] == new[4114],
                  'old_Sambucus': old[275], 'new_Sambucus': new[275], 'new_cf_Sambucus': new[902],
                  'old_Abies': old[1], 'new_Abies': new[1],
                  'pr26_status_retained': p['status'], 'source_recoding_proven': False,
                  'physical_identity_proven': False, 'mon16_depth_cm': '', 'mon16_age_cal_bp': ''}
        pairs.append(record)
        pair_values[pilot] = old, new, newid
    explanations = []
    for d in differences:
        old, new, newid = pair_values[d['pilot_sample_id']]
        tid = int(d['taxonid'])
        require(newid == int(d['diagnostic_sampleid']) and old[tid] == Decimal(d['legacy_value'])
                and new[tid] == Decimal(d['diagnostic_value']), 'PR26 difference changed')
        if tid == 480:
            label = 'four_old_Plantago_fields_balance_only'
            left, right = sum(old[t] for t in (480, 699, 2905, 3590)), new[480]
        elif tid == 284:
            label = 'Digitalis_type_value_equal_only'
            left, right = old[284], new[4114]
        else:
            label = 'unexplained_source_difference'
            left, right = old[tid], new[tid]
        explanations.append(d | {'diagnostic_evidence': label, 'diagnostic_left': left,
                                 'diagnostic_right': right, 'diagnostic_equal': left == right,
                                 'provenance_status': 'unresolved_no_submission_record',
                                 'pr26_decision_changed': False})
    definitions = []
    for tid, (code, varid) in FIELDS.items():
        v = variables[varid]
        accepted = variables[int(v['accvar_'])]
        definitions.append({'neotoma_taxonid': tid, 'epd_variable_id': varid,
                            'epd_code': code, 'epd_original_name': v['varname'],
                            'epd_accepted_variable_id': accepted['var_'],
                            'epd_accepted_name': accepted['varname'],
                            'epd_notes': v['notes'], 'source': 'p_vars.dump'})
    inventory = []
    for did in (26608, 47525):
        for tid in (*FIELDS, 4114, 3430, 476, 902):
            relevant = [r for rr in rows[did].values() for r in rr if int(r['taxonid']) == tid]
            inventory.append({'datasetid': did, 'taxonid': tid,
                              'stored_names': json.dumps(sorted({r['variablename'] for r in relevant})),
                              'samples_with_stored_rows': sum(any(int(r['taxonid']) == tid for r in rr) for rr in rows[did].values()),
                              'variableid_exported': False, 'synonymy_exported': False})
    output.mkdir(parents=True)
    csv_out(output / 'fixed_pair_diagnostics.csv', pairs)
    csv_out(output / 'difference_evidence.csv', explanations)
    csv_out(output / 'original_variable_definitions.csv', definitions)
    csv_out(output / 'affected_native_inventory.csv', inventory)
    json_out(output / 'summary.json', {
        'input_sha256': LOCKS, 'pr26_reader_sha256': READER_SHA,
        'rdata_version': info['rdata_version'], 'fixed_pairs': len(pairs),
        'new_pair_search_performed': False, 'three_source_original_cells_checked': checked,
        'plantago_balances_equal': sum(r['Plantago_balance_equal'] for r in pairs),
        'scroph_digitalis_values_equal': sum(r['Scroph_Digitalis_value_equal'] for r in pairs),
        'difference_evidence_counts': dict(collections.Counter(r['diagnostic_evidence'] for r in explanations)),
        'pr26_status_counts_retained': dict(collections.Counter(r['pr26_status_retained'] for r in pairs)),
        'all_47_provenance_cases_resolved': False, 'submission_records_acquired': False,
        'source_values_corrected': False, 'p1_status': 'open', 'mon16_mapping_proven': False})
    json_out(output / 'manifest.json', {p.name: digest(p) for p in sorted(output.iterdir())})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', required=True, type=Path)
    p.add_argument('--pr26-reader', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    run(a.inputs, a.pr26_reader, a.output)
