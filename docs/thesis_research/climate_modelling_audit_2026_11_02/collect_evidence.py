"""Supplement the frozen two-root audit with climate_modelling, read-only.

Only this explicit third source tree is walked. The old audit is imported for
static R parsing; its collect() is never invoked and its outputs stay frozen.
Large containers are not decoded by this metadata/filename collection step.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
OLD = ROOT / 'docs/thesis_research/required_files_2026_11_02'
spec = importlib.util.spec_from_file_location('frozen_required_files', OLD / 'collect_evidence.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def emit(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def table(name, rows, fields=None):
    with (OUT / name).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    cfg = json.loads((ROOT / '.local/my_reconst_paths.json').read_text(encoding='utf-8-sig'))
    source = Path(cfg['climate_modelling_source_root']).resolve(strict=True)
    roots = {'climate_modelling': source, 'original': Path(cfg['original_data_root']),
             'additional': Path(cfg['additional_source_root']), 'repository': ROOT}
    for root in [source, roots['original'], roots['additional']]:
        if root == OUT or root in OUT.parents or OUT in root.parents:
            raise ValueError('Evidence output must be outside all source trees')
    entries = []; errors = []
    def walk_error(error):
        errors.append({'operation': 'walk', 'error_type': type(error).__name__})
    for folder, dirs, names in os.walk(source, onerror=walk_error):
        dirs[:] = sorted(d for d in dirs if d not in legacy.EXCLUDED_DIRS)
        for name in sorted(names):
            p = Path(folder) / name
            if not p.is_file():
                errors.append({'operation':'not_regular_file','source':p.relative_to(source).as_posix()})
                continue
            s = p.stat()
            entries.append({'source': 'climate_modelling:' + p.relative_to(source).as_posix(),
                            'relative':p.relative_to(source).as_posix(), 'path':p, 'bytes':s.st_size,
                            'mtime_ns':s.st_mtime_ns, 'ext':p.suffix.lower()})
    by_name = defaultdict(list); archive_names = defaultdict(list); archives = []
    for e in entries: by_name[e['path'].name.lower()].append(e)
    for e in entries:
        if e['ext'] != '.zip': continue
        try:
            with zipfile.ZipFile(e['path']) as z:
                members = [{'name':i.filename,'bytes':i.file_size,'crc32':i.CRC} for i in z.infolist() if not i.is_dir()]
            for i in members:
                archive_names[Path(i['name']).name.lower()].append(e['source']+'!'+i['name'])
            archives.append({'source':e['source'],'members':len(members),'directory_sha256':
                             hashlib.sha256(json.dumps(members,sort_keys=True).encode()).hexdigest()})
        except (OSError,zipfile.BadZipFile) as error:
            errors.append({'operation':'zip_directory','source':e['source'],'error_type':type(error).__name__})
    requirements = list(csv.DictReader((OLD/'priority_requirements.csv').open(encoding='utf-8-sig')))
    queries = [{'id':r['id'],'priority':r['priority'],'consumer':r['source']+':'+r['line'],
                'query':r['filename_or_pattern']} for r in requirements]
    queries += [{'id':'ENTRY_MAIN_128','priority':'entrypoint','consumer':'original:main.R:128',
                 'query':'pollen/transfer_functions/input_data/data_out/taxa_distribution_ww.rdata'}]
    posterior = json.loads((OLD/'posterior_variants_recheck.json').read_text())
    queries += [{'id':'POSTERIOR_'+str(i+1),'priority':'historical_output','consumer':'prior_posterior_contract',
                 'query':r['relative_path']} for i,r in enumerate(posterior)]
    matches = []
    selected = {}
    for q in queries:
        name = Path(q['query']).name.lower()
        candidates = by_name.get(name,[]); zipped = archive_names.get(name,[])
        for e in candidates: selected[e['source']] = e
        matches.append({**q,'new_file_candidates': ';'.join(e['source'] for e in candidates),
                        'new_archive_candidates': ';'.join(zipped),
                        'status':'filename_candidate_not_historical_identity' if candidates or zipped else 'no_named_copy_in_new_scope'})
    table('requirement_crosswalk.csv',matches)
    table('file_metadata.csv',[{k:e[k] for k in ['source','bytes','mtime_ns','ext']} for e in entries])
    # Bind nearby versions without interpreting their numeric payloads.
    for e in entries:
        if e['path'].name.lower() in {'pollen_g_climate_all_giesecke.rdata','pollen_g_climate_all_giesecke_v1.4.rdata',
                                      'pmip3_21k_sftlf_v0.nc'}:
            selected[e['source']] = e
    hashes = [{**{k:e[k] for k in ['source','bytes','mtime_ns']},'sha256':sha(e['path'])} for e in selected.values()]
    emit('candidate_hashes.json',hashes)
    old_copies = json.loads((OLD/'preserved_copies.json').read_text())
    emit('byte_identity_comparisons.json',[{'new_source':h['source'],'prior_source':p['source'],
             'same_sha256':h['sha256']==p['sha256'],'historical_run_identity_claimed':False}
             for h in hashes if h['source'].endswith('/taxa_distribution_ww.rdata') for p in old_copies])
    sources = []; calls = []
    for e in entries:
        if e['ext'] != '.r': continue
        raw=e['path'].read_bytes(); s=e['path'].stat()
        text=raw.decode('utf-8-sig',errors='replace')
        recent=s.st_mtime>=legacy.CUTOFF
        in_src=any(part.lower()=='src' for part in Path(e['relative']).parts)
        sources.append({'source':e['source'],'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                        'modified_utc':datetime.fromtimestamp(s.st_mtime,timezone.utc).isoformat(),
                        'is_literal_src_directory':in_src,'modified_after_2021':recent})
        for c in legacy.calls(text):
            expr=legacy.path_expression(c)
            if not expr: continue
            rendered=legacy.render_path(expr,legacy.constants_before(c['code'],c['position']))
            if not any(word in rendered.lower() for word in ['giesecke','taxa_distribution','speleo_info','ages_info','pmip3_21k','mixed_partial','cru','isotope','d18_d13']): continue
            calls.append({'source':e['source'],'line':c['line'],'function':c['function'],
                          'path_or_pattern':legacy.public(rendered,roots),'modified_after_2021':recent,
                          'static_only_not_execution_proof':True})
    table('source_dates.csv',sources)
    table('targeted_source_calls.csv',calls,['source','line','function','path_or_pattern','modified_after_2021','static_only_not_execution_proof'])
    # Re-evaluate only the old reference names against the NEW root, not the old trees.
    old_reference_matches=[]
    for row in csv.DictReader((OLD/'dependency_calls.csv').open(encoding='utf-8-sig')):
        rendered=row['path_or_pattern']; name=Path(rendered).name.lower()
        if '{' in name: continue  # no guessed expansion of dynamic R expressions
        candidates=by_name.get(name,[]); zipped=archive_names.get(name,[])
        if candidates or zipped:
            old_reference_matches.append({'source':row['source'],'line':row['line'],'path_or_pattern':rendered,
                'previous_availability':row.get('availability',''),
                'new_file_candidates':';'.join(e['source'] for e in candidates),'new_archive_candidates':';'.join(zipped)})
    table('prior_reference_new_candidates.csv',old_reference_matches)
    changed=[]
    for e in entries:
        s=e['path'].stat()
        if (s.st_size,s.st_mtime_ns)!=(e['bytes'],e['mtime_ns']): changed.append(e['source'])
    if changed: errors.append({'operation':'source_changed_during_read','sources':changed})
    # Bind exact consumer lines from the two original entrypoints; no R execution.
    consumers=[]
    for file,lines in [('main.R',[128,129,130]),('main_v1.2.R',[391,392,858,860])]:
        p=roots['original']/file; raw=p.read_bytes(); text=raw.decode('utf-8-sig',errors='replace').splitlines()
        consumers.append({'source':'original:'+file,'sha256':hashlib.sha256(raw).hexdigest(),
            'lines':[{'line':n,'text':legacy.public(text[n-1],roots)} for n in lines]})
    emit('entrypoint_bindings.json',consumers)
    emit('archive_directories.json',archives)
    summary={'basis_commit':'705cf7b68b2bfdfc99919ad50e4091dd113d907e',
       'scope':'Only configured climate_modelling_source_root recursively; old inventories read, not rerun',
       'private_config_key':'climate_modelling_source_root','excluded_directories':sorted(legacy.EXCLUDED_DIRS),
       'files_indexed':len(entries),'extensions':dict(Counter(e['ext'] for e in entries)),
       'R_sources_read_and_hashed':len(sources),'R_sources_modified_after_2021':sum(x['modified_after_2021'] for x in sources),
       'literal_src_files_modified_after_2021':sum(x['modified_after_2021'] and x['is_literal_src_directory'] for x in sources),
       'targeted_static_calls':len(calls),'archives_read_directory_only':len(archives),
       'queries':len(queries),'priority_requirements':39,'posterior_variants':8,
       'priority_requirements_with_new_named_candidates':[x['id'] for x in matches if x['id'].startswith('F') and (x['new_file_candidates'] or x['new_archive_candidates'])],
       'prior_read_calls_with_new_static_name_candidates':len(old_reference_matches),
       'candidate_files_hashed':len(hashes),'source_metadata_unchanged':not changed,'errors':errors,
       'large_containers_decoded_by_collector':False,'R_executed':False,'models_or_samplers_started':False,
       'final_variant_selected':False,'limitations':['Names/paths do not prove historical identity or final execution.',
       'R parser is static, not an execution/reachability interpreter. Dynamic names are not guessed.',
       'mtime after 2021 is a priority signal, not historical creation/execution proof.',
       'Renamed equivalent objects are not excluded by filename absence; no exhaustive numeric search.']}
    emit('summary.json',summary)
    print(json.dumps(summary,ensure_ascii=True))
    if errors: raise SystemExit(1)


if __name__=='__main__': main()
