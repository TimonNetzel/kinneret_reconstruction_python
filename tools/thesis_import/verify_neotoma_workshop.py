"""Independent raw-slot reader and Fraction checker; no producer import.

Uses rdata's serialization parser, but not its S4/data-frame conversion.
Does not execute R code, workshop code, SQL, or age models.
"""
import argparse
import collections
import csv
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import warnings


def ref(x):
    while x is not None and x.info.type.name == 'REF':
        x = x.referenced_object
    return x


def attributes(x):
    attrs = {}
    pair = ref(ref(x).attributes)
    while pair is not None and pair.info.type.name == 'LIST':
        label = ref(ref(pair.tag).value).value.decode('utf-8')
        assert label not in attrs
        attrs[label] = pair.value[0]
        pair = ref(pair.value[1])
    return attrs


def values(x):
    import numpy as np
    x = ref(x)
    if x.info.type.name == 'STR':
        return [None if ref(v).value is None else ref(v).value.decode('utf-8') for v in x.value]
    assert x.info.type.name in ('INT', 'REAL', 'LGL')
    out = []
    for v in x.value:
        if np.ma.is_masked(v):
            out.append(None)
        else:
            v = v.item() if isinstance(v, np.generic) else v
            out.append(None if isinstance(v, float) and math.isnan(v) else v)
    return out


def scalar(x):
    v = values(x)
    assert len(v) == 1
    return v[0]


def children(x, slot):
    v = ref(attributes(x)[slot])
    assert v.info.type.name == 'VEC'
    return v.value


def records(x):
    x = ref(x)
    assert x.info.type.name == 'VEC'
    names = values(attributes(x)['names'])
    assert len(names) == len(set(names)) == len(x.value)
    columns = [values(c) for c in x.value]
    assert len(set(map(len, columns))) == 1
    return [dict(zip(names, row)) for row in zip(*columns)]


def number_for(rows, taxon):
    vals = {Fraction(r['value']) for r in rows if r['taxonid'] == taxon
            and r['units'] == 'NISP' and r['element'] == 'pollen'}
    assert len(vals) <= 1
    return next(iter(vals), Fraction(0))


def verify(rds, pangaea, prior, output, report):
    import rdata
    blob = rds.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == '03536b6f1d3974de77ff997f6d86674a78882b2ebe305bb4e135e38289a2a829'
    assert hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest() == 'eaa1e04645575ecd13ba5b2432e877ae57848db1'
    assert hashlib.sha256(pangaea.read_bytes()).hexdigest() == '46ab6cea3a636d1a25b8c15f4b9b79a94c8fd1bd41f2943ae5c8941f9050a4de'
    assert hashlib.sha256(prior.read_bytes()).hexdigest() == '4ab39ab51bb54b76257f2ffd0460044fe15d17e962dbb49fce94d38c2ba35f03'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data = rdata.parser.parse_file(rds)
    sites = ref(attributes(data.object)['sites'])
    assert len(sites.value) == 59
    assert values(attributes(sites)['names']) == ['site'] * 59
    selected = [s for s in sites.value if scalar(attributes(s)['siteid']) == 3295]
    assert len(selected) == 1
    collectionunits = children(attributes(selected[0])['collunits'], 'collunits')
    assert len(collectionunits) == 13
    datasets = {}
    inventory = []
    for cu in collectionunits:
        ca = attributes(cu)
        for ds in children(ca['datasets'], 'datasets'):
            da = attributes(ds)
            did = scalar(da['datasetid'])
            assert did not in datasets
            samples = [attributes(s) for s in children(da['samples'], 'samples')]
            datasets[did] = (ca, da, samples)
            inventory.append((did, scalar(ca['collectionunitid']), scalar(ca['handle']), len(samples),
                              sum(scalar(s['analysisunitname']) is not None for s in samples),
                              sum(scalar(s['samplename']) is not None for s in samples)))
    old, new = datasets[26608][2], datasets[47525][2]
    assert len(old) == 245 and len(new) == 924
    assert scalar(datasets[26608][0]['handle']) == 'LGM90D'
    assert scalar(datasets[47525][0]['handle']) == 'MONTICCHIO'
    old_records = [records(s['datum']) for s in old]
    new_records = [records(s['datum']) for s in new]
    pg = list(csv.DictReader(io.StringIO(pangaea.read_text().split('*/', 1)[1].strip()), delimiter='\t'))
    assert len(pg) == len(old)
    check_taxa = [('Abi [#]', 1), ('Aln [#]', 330), ('Art [#]', 25), ('Bet [#]', 29), ('Fag [#]', 107)]
    aligned = []
    for s, rows in zip(old, old_records):
        # Independent precision check: the published metre depths have two
        # decimals, so the stored cm depth must be within half a centimetre.
        hits = [j for j, p in enumerate(pg)
                if abs(Fraction(p['Depth sed [m]'])*100-Fraction(scalar(s['depth']))) <= Fraction(1,2)
                and all(number_for(rows,tid)==Fraction(p[col]) for col,tid in check_taxa)]
        assert len(hits)==1
        aligned.append(hits[0])
    assert len(set(aligned))==245
    alignment=list(csv.DictReader((output/'legacy_source_alignment.csv').open()))
    assert len(alignment)==245
    moved=0
    for i,(s,j,actual) in enumerate(zip(old,aligned,alignment)):
        assert int(actual['rds_list_position'])==i+1
        assert int(actual['epd_row_number'])==j+1
        assert int(actual['neotoma_sampleid'])==scalar(s['sampleid'])
        assert actual['position_changed']==str(i!=j)
        moved+=i!=j
    assert moved==4
    expected_quercus = []
    for s, rows, j in zip(old, old_records, aligned):
        source=pg[j]
        source_value, native = Fraction(source['Que [#]']), number_for(rows, 2943)
        if source_value != native:
            expected_quercus.append((j+1, scalar(s['sampleid']), source_value, native, number_for(rows, 4717)))
    expected_quercus.sort()
    qtable = list(csv.DictReader((output/'legacy_quercus_differences.csv').open()))
    assert len(qtable) == len(expected_quercus) == 16
    for actual, expected in zip(qtable, expected_quercus):
        assert int(actual['epd_row_number']) == expected[0]
        assert int(actual['neotoma_sampleid']) == expected[1]
        assert tuple(Fraction(actual[k]) for k in ('pangaea_Quercus','neotoma_Quercus_undiff','neotoma_Quercus_ilex')) == expected[2:]
    key_names = ('taxonid','units','element','context','symmetry','elementtype')
    def make(rows):
        result = collections.defaultdict(set)
        for r in rows:
            key = tuple(None if r[k] is None else str(r[k]) for k in key_names)
            assert r['value'] is not None
            result[key].add(Fraction(r['value']))
        return dict(result)
    om, nm = [make(r) for r in old_records], [make(r) for r in new_records]
    ok, nk = set().union(*om), set().union(*nm)
    bad = {key for m in om+nm for key, vals in m.items() if len(vals)>1}
    common = sorted((ok & nk)-bad, key=str)
    assert (len(ok), len(nk), len(bad), len(common)) == (131,182,1,76)
    policies = list(csv.DictReader((output/'native_variable_policy.csv').open()))
    actual_keys = {tuple(r[k] if r[k] else None for k in key_names) for r in policies if r['used_for_comparison']=='True'}
    assert actual_keys == set(common)
    def vector(m):
        return tuple(next(iter(m[k])) if k in m else Fraction(0) for k in common)
    new_vectors = [vector(m) for m in nm]
    by_depth = {Fraction(scalar(s['depth'])): i for i,s in enumerate(old)}
    assert len(by_depth) == 245
    previous = list(csv.DictReader(prior.open()))
    table = list(csv.DictReader((output/'sample_correspondence.csv').open()))
    assert len(table) == len(previous) == 69
    unique, hit_ids = 0, []
    expected_diagnostics=[]
    expected_differences=[]
    for actual, fixed in zip(table, previous):
        i = by_depth[Fraction(fixed['archive_midpoint_cm'])]
        source = old[i]
        assert actual['pilot_sample_id'] == fixed['pilot_sample_id']
        assert int(actual['legacy_sampleid']) == scalar(source['sampleid'])
        assert int(actual['legacy_analysisunitid']) == scalar(source['analysisunitid'])
        v = vector(om[i])
        hits = [j for j, nv in enumerate(new_vectors) if nv == v]
        assert int(actual['candidate_count']) == len(hits)
        assert json.loads(actual['candidate_sampleids']) == [scalar(new[j]['sampleid']) for j in hits]
        assert actual['mon16_depth_cm'] == actual['mon16_age_cal_bp'] == ''
        assert actual['new_age_transfer_allowed'] == 'false'
        if len(hits) == 1:
            unique += 1
            target = new[hits[0]]
            tid = scalar(target['sampleid']);hit_ids.append(tid)
            assert actual['status'] == 'unique_shared_variable_candidate'
            assert int(actual['candidate_sampleid']) == tid
            assert int(actual['candidate_analysisunitid']) == scalar(target['analysisunitid'])
            assert actual['candidate_analysisunitname'] == scalar(target['analysisunitname'])
            assert Fraction(actual['candidate_composite_depth_cm']) == Fraction(scalar(target['depth']))
            assert Fraction(actual['candidate_minus_legacy_cm']) == Fraction(scalar(target['depth'])) - Fraction(fixed['archive_midpoint_cm'])
        else:
            assert len(hits) == 0
            assert actual['status'] == 'no_shared_variable_match'
            assert actual['candidate_sampleid'] == actual['candidate_analysisunitname'] == ''
            distances=[sum(a!=b for a,b in zip(v,nv)) for nv in new_vectors]
            minimum=min(distances)
            nearest=[j for j,d in enumerate(distances) if d==minimum]
            expected_diagnostics.append((actual['pilot_sample_id'],scalar(source['sampleid']),minimum,
                                         [scalar(new[j]['sampleid']) for j in nearest]))
            for j in nearest:
                for key,a,b in zip(common,v,new_vectors[j]):
                    if a!=b:
                        expected_differences.append((actual['pilot_sample_id'],scalar(new[j]['sampleid']),key,a,b))
    assert unique == len(set(hit_ids)) == 28
    diagnostics=list(csv.DictReader((output/'unmatched_diagnostics.csv').open()))
    assert len(diagnostics)==len(expected_diagnostics)==41
    for r,(pid,sid,minimum,nearest) in zip(diagnostics,expected_diagnostics):
        assert r['pilot_sample_id']==pid and int(r['legacy_sampleid'])==sid
        assert int(r['minimum_differing_fields'])==minimum and int(r['nearest_count'])==len(nearest)
        assert json.loads(r['nearest_sampleids'])==nearest
        assert r['status']=='diagnostic_only_not_correspondence'
    differing=list(csv.DictReader((output/'unmatched_variable_differences.csv').open()))
    assert len(differing)==len(expected_differences)==47
    for r,(pid,sid,key,a,b) in zip(differing,expected_differences):
        assert r['pilot_sample_id']==pid and int(r['diagnostic_sampleid'])==sid
        assert tuple(r[k] if r[k] else None for k in key_names)==key
        assert Fraction(r['legacy_value'])==a and Fraction(r['diagnostic_value'])==b
        assert r['status']=='unresolved_source_difference'
    for r in csv.DictReader((output/'field_availability.csv').open()):
        samples = datasets[int(r['datasetid'])][2]
        field = r['field'];exists = sum(field in s for s in samples)
        nonmissing = sum(field in s and scalar(s[field]) is not None for s in samples)
        assert int(r['slot_present']) == exists and int(r['nonmissing']) == nonmissing
    actual_inventory = list(csv.DictReader((output/'dataset_inventory.csv').open()))
    expected_inventory = sorted(inventory)
    assert len(actual_inventory) == len(expected_inventory)
    for row, expected in zip(actual_inventory, expected_inventory):
        assert (int(row['datasetid']),int(row['collectionunitid']),row['handle'],int(row['samples']),
                int(row['analysisunit_names_present']),int(row['sample_names_present'])) == expected
    manifest = json.loads((output/'manifest.json').read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((output/name).read_bytes()).hexdigest() == digest
    result = {'raw_R_sites_checked':59, 'named_list_collision_avoided':True,
              'collectionunits_checked':13, 'source_cells_checked':1225,
              'source_Quercus_differences_preserved':16,
              'source_alignment_positions_changed':4,
              'quercus_difference_equals_native_ilex_rows':sum(a-b==c for _,_,a,b,c in expected_quercus),
              'native_variable_keys_checked':[131,182], 'ambiguous_native_keys_excluded':1,
              'shared_keys_checked':76, 'pilot_rows_checked':69, 'unique_candidates_checked':28,
              'unmatched_rows_preserved':41,'candidate_names_verified_against_raw_slots':True,
              'unmatched_diagnostic_cells_checked':47,
              'independent_number_system':'Fraction of stored numeric values; no tolerance',
              'shared_dependency':'rdata serialization parser only; no producer import or S4/dataframe converter',
              'R_or_workshop_code_executed':False,'mon16_mapping_proven':False,
              'p1_status':'open','output_hashes_checked':manifest}
    Path(report).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='output_hashes_checked'},indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rds',type=Path,required=True)
    p.add_argument('--pangaea',type=Path,required=True)
    p.add_argument('--prior',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();verify(a.rds,a.pangaea,a.prior,a.output,a.report)
