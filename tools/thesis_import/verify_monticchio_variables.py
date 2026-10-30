"""Independent Fraction/raw-slot check of fixed-pair diagnostics.

Imports only the hash-locked PR26 raw reader, not the new producer. Reads the
full SQL COPY stream rather than the producer's separate table dumps.
Shared dependency: rdata's binary serialization parser (not an independent decoder).
"""
import argparse
import collections
import csv
from fractions import Fraction as F
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import warnings
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def verify(inputs, helper, output, report):
    import rdata
    assert sha(helper) == '45994cddec4b1e70509d84a0b7b79dd3911f3722de2e6d4fde829336e0f021fe'
    module_spec = importlib.util.spec_from_file_location('locked_raw_reader', helper)
    raw = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(raw)
    locks = {
        'itDownload.RDS': '03536b6f1d3974de77ff997f6d86674a78882b2ebe305bb4e135e38289a2a829',
        'epd-postgres-distribution-20171031.zip': '6ce720d51e0cf92f3f207ed4527d6cda00dd565cecbb8d3793112a723dbfc072',
        'monticchio_739452.tab': '46ab6cea3a636d1a25b8c15f4b9b79a94c8fd1bd41f2943ae5c8941f9050a4de',
        'pr26_correspondence.csv': '08b90b71e5aa22b4f923a7f82579a2d18e6cc9d9425f551e6065a0709b71c9bf',
        'pr26_diagnostics.csv': '3368c3296e4166199ce26a1c9d54ac6a38ed7976fa2f83bec0bd1c9bee266373',
        'pr26_differences.csv': '9161157b577aeaf79876d658ab9587910395b45c3475c3dcaf0cbdceff76c345',
    }
    assert all(sha(inputs / name) == value for name, value in locks.items())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        parsed = rdata.parser.parse_file(inputs / 'itDownload.RDS')
    sites = raw.children(parsed.object, 'sites')
    assert len(sites) == 59
    selected = [s for s in sites if raw.scalar(raw.attributes(s)['siteid']) == 3295]
    assert len(selected) == 1
    ds = {}
    for cu in raw.children(raw.attributes(selected[0])['collunits'], 'collunits'):
        for dataset in raw.children(raw.attributes(cu)['datasets'], 'datasets'):
            attr = raw.attributes(dataset)
            did = raw.scalar(attr['datasetid'])
            if did in (26608, 47525):
                ds[did] = {raw.scalar(raw.attributes(s)['sampleid']): raw.attributes(s)
                           for s in raw.children(attr['samples'], 'samples')}
    assert len(ds[26608]) == 245 and len(ds[47525]) == 924
    rows = {did: {sid: raw.records(s['datum']) for sid, s in samples.items()}
            for did, samples in ds.items()}

    def value(rr, tid):
        numbers = set()
        for r in rr:
            if (r['taxonid'], r['units'], r['element'], r['context'], r['symmetry'], r['elementtype']) == (tid, 'NISP', 'pollen', None, None, 'pollen'):
                assert r['value'] is not None
                numbers.add(F(r['value']))
        assert len(numbers) <= 1 and all(n >= 0 for n in numbers)
        return next(iter(numbers), F(0))

    # The full database COPY stream is a different archive member/reading path.
    definitions, counts = {}, {}
    with zipfile.ZipFile(inputs / 'epd-postgres-distribution-20171031.zip') as archive:
        with gzip.GzipFile(fileobj=archive.open('dumpall_epd_db.sql.gz')) as handle:
            current, columns = None, []
            for line in io.TextIOWrapper(handle, encoding='latin1'):
                if line.startswith('COPY '):
                    match = re.match(r'COPY (?:public\.)?(\w+) \((.*?)\) FROM stdin;', line)
                    current = match[1] if match else None
                    columns = match[2].split(', ') if match else []
                elif line.rstrip() == r'\.':
                    current = None
                elif current == 'p_vars':
                    r = dict(zip(columns, [None if s == r'\N' else s for s in line.rstrip('\n').split('\t')]))
                    definitions[int(r['var_'])] = r
                elif current == 'p_counts' and line.startswith('932\t'):
                    r = dict(zip(columns, line.rstrip('\n').split('\t')))
                    key = int(r['sample_']), int(r['var_'])
                    assert key not in counts
                    counts[key] = F(r['count'])
    assert definitions and counts
    field_map = {1: ('Abi', 1), 480: ('Pla', 576), 699: ('Pla.l-T', 580),
                 2905: ('Pla.m/m', 583), 3590: ('Pla.c-T', 1478), 284: ('Scoae', 763), 275: ('Sam', 715)}
    pg = list(csv.DictReader(io.StringIO((inputs / 'monticchio_739452.tab').read_text().split('*/')[1].strip()), delimiter='\t'))
    prior = table(inputs / 'pr26_correspondence.csv')
    diag = {p['pilot_sample_id']: p for p in table(inputs / 'pr26_diagnostics.csv')}
    actual = table(output / 'fixed_pair_diagnostics.csv')
    assert len(actual) == len(prior) == 69
    checks, preserved = 0, collections.Counter()
    pair_lookup = {}
    for source, result in zip(prior, actual):
        pilot = source['pilot_sample_id']
        oldid = int(source['legacy_sampleid'])
        if source['candidate_sampleid']:
            newid, role = int(source['candidate_sampleid']), 'pr26_candidate'
        else:
            ids = json.loads(diag[pilot]['nearest_sampleids'])
            assert len(ids) == int(diag[pilot]['nearest_count']) == 1
            newid, role = ids[0], 'pr26_diagnostic_only'
        old, new = rows[26608][oldid], rows[47525][newid]
        assert result['pilot_sample_id'] == pilot
        assert int(result['legacy_sampleid']) == oldid and int(result['fixed_composite_sampleid']) == newid
        assert result['pair_role'] == role
        assert result['archived_analysisunitname'] == raw.scalar(ds[47525][newid]['analysisunitname'])
        assert result['pr26_status_retained'] == source['status']
        preserved[source['status']] += 1
        fields = {'old_Pla': (old, 480), 'old_Pla_l_T': (old, 699), 'old_Pla_m_m': (old, 2905),
                  'old_Pla_c_T': (old, 3590), 'new_Plantago': (new, 480),
                  'old_Scrophulariaceae': (old, 284), 'new_Scrophulariaceae': (new, 284),
                  'new_Digitalis_type': (new, 4114), 'new_Veronica_type': (new, 3430),
                  'new_Pedicularis': (new, 476), 'old_Sambucus': (old, 275),
                  'new_Sambucus': (new, 275), 'new_cf_Sambucus': (new, 902),
                  'old_Abies': (old, 1), 'new_Abies': (new, 1)}
        for col, (rr, tid) in fields.items():
            assert F(result[col]) == value(rr, tid)
        total = sum(value(old, tid) for tid in (480, 699, 2905, 3590))
        assert F(result['old_four_Plantago_fields_sum']) == total == value(new, 480)
        assert value(old, 284) == value(new, 4114)
        assert result['Plantago_balance_equal'] == result['Scroph_Digitalis_value_equal'] == 'True'
        assert result['source_recoding_proven'] == result['physical_identity_proven'] == 'False'
        assert result['mon16_depth_cm'] == result['mon16_age_cal_bp'] == ''
        for tid, (code, varid) in field_map.items():
            assert value(old, tid) == counts.get((int(source['epd_sample_number']), varid), F(0)) == F(pg[int(source['epd_sample_number'])-1][code+' [#]'])
            checks += 1
        pair_lookup[pilot] = old, new, newid
    definition_rows = table(output / 'original_variable_definitions.csv')
    assert len(definition_rows) == 7
    assert {int(r['neotoma_taxonid']) for r in definition_rows} == set(field_map)
    for r in definition_rows:
        v = definitions[int(r['epd_variable_id'])]
        assert r['epd_code'] == v['varcode'] and r['epd_original_name'] == v['varname']
        assert r['epd_accepted_variable_id'] == v['accvar_']
        assert r['epd_accepted_name'] == definitions[int(v['accvar_'])]['varname']
        assert r['epd_notes'] == (v['notes'] or '')
    inventory = table(output / 'affected_native_inventory.csv')
    assert len(inventory) == 22
    for r in inventory:
        did, tid = int(r['datasetid']), int(r['taxonid'])
        selected_rows = [x for rr in rows[did].values() for x in rr if x['taxonid'] == tid]
        assert json.loads(r['stored_names']) == sorted({x['variablename'] for x in selected_rows})
        assert int(r['samples_with_stored_rows']) == sum(any(x['taxonid'] == tid for x in rr) for rr in rows[did].values())
        assert r['variableid_exported'] == r['synonymy_exported'] == 'False'
        assert all('variableid' not in x for x in selected_rows)
    prior_diffs = table(inputs / 'pr26_differences.csv')
    evidence = table(output / 'difference_evidence.csv')
    assert len(prior_diffs) == len(evidence) == 47
    evidence_counts = collections.Counter()
    for before, after in zip(prior_diffs, evidence):
        assert all(after[k] == v for k, v in before.items())
        old, new, newid = pair_lookup[before['pilot_sample_id']]
        tid = int(before['taxonid'])
        assert int(before['diagnostic_sampleid']) == newid
        assert F(before['legacy_value']) == value(old, tid)
        assert F(before['diagnostic_value']) == value(new, tid)
        if tid == 480:
            left, right, label = sum(value(old, t) for t in (480, 699, 2905, 3590)), value(new, 480), 'four_old_Plantago_fields_balance_only'
        elif tid == 284:
            left, right, label = value(old, 284), value(new, 4114), 'Digitalis_type_value_equal_only'
        else:
            left, right, label = value(old, tid), value(new, tid), 'unexplained_source_difference'
        assert (F(after['diagnostic_left']), F(after['diagnostic_right'])) == (left, right)
        assert after['diagnostic_equal'] == str(left == right)
        assert after['diagnostic_evidence'] == label
        assert after['pr26_decision_changed'] == 'False'
        assert after['provenance_status'] == 'unresolved_no_submission_record'
        evidence_counts[label] += 1
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['input_sha256'] == locks
    assert summary['difference_evidence_counts'] == dict(evidence_counts)
    assert summary['pr26_status_counts_retained'] == dict(preserved)
    assert summary['three_source_original_cells_checked'] == checks == 483
    assert summary['plantago_balances_equal'] == summary['scroph_digitalis_values_equal'] == 69
    assert summary['fixed_pairs'] == 69 and summary['new_pair_search_performed'] is False
    assert summary['p1_status'] == 'open' and summary['source_values_corrected'] is False
    manifest = json.loads((output / 'manifest.json').read_text())
    # The published result directory may also contain its README and sources.
    assert set(manifest) == {'fixed_pair_diagnostics.csv', 'difference_evidence.csv',
                             'original_variable_definitions.csv', 'affected_native_inventory.csv',
                             'summary.json'}
    assert all(sha(output / name) == value for name, value in manifest.items())
    report.write_text(json.dumps({'result': 'passed', 'fixed_pairs_checked': 69,
        'original_cells_checked_in_three_sources': checks, 'difference_rows_preserved': 47,
        'evidence_counts': dict(evidence_counts), 'pr26_status_counts_retained': dict(preserved),
        'independent_of_producer': True, 'reader': 'raw S4 slots and Fraction; full SQL COPY stream',
        'shared_binary_parser': 'rdata ' + rdata.__version__,
        'raw_reader_sha256': sha(helper), 'output_sha256': manifest,
        'physical_identity_or_recoding_proven': False, 'p1_status': 'open'}, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('inputs', 'pr26-raw-reader', 'output', 'report'):
        p.add_argument('--' + key, required=True, type=Path)
    a = p.parse_args()
    verify(a.inputs, a.pr26_raw_reader, a.output, a.report)
