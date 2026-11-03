"""Verify the portable research snapshot without local source data or third-party packages."""
from pathlib import Path, PurePosixPath
import argparse,gzip,hashlib,json
import xml.etree.ElementTree as ET


def sha(blob):return hashlib.sha256(blob).hexdigest()


def verify(base,with_local=False):
    manifest=json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    if manifest['schema_version']!=1:raise ValueError('Unknown snapshot schema')
    seen=set();total=0
    for entry in manifest['entries']:
        rel=PurePosixPath(entry['published'])
        if rel.is_absolute() or '..' in rel.parts or str(rel) in seen:raise ValueError('Unsafe or duplicate snapshot path')
        seen.add(str(rel));path=base.joinpath(*rel.parts);stored=path.read_bytes()
        if sha(stored)!=entry['published_sha256'] or len(stored)!=entry['published_bytes']:raise ValueError(f'Published bytes changed: {rel}')
        content=gzip.decompress(stored) if path.suffix=='.gz' else stored
        if sha(content)!=entry['decoded_sha256']:raise ValueError(f'Decoded bytes changed: {rel}')
        if with_local:
            repo=base.parents[2];source=repo/entry['source']
            if sha(source.read_bytes())!=entry['local_original_sha256']:raise ValueError(f'Local source changed: {entry["source"]}')
        if '.json' in path.suffixes:
            text=content.decode('utf-8');json.loads(text)
            if any(fragment in text for fragment in ['C:\\\\Users\\\\tnetz','C:/Users/tnetz']):raise ValueError(f'Personal path left in export: {rel}')
        total+=len(stored)
    summary=json.loads((base/'summary.json').read_text())
    if summary['plots']['numeric_TF_arrays_recovered'] or summary['gates']['fossil_pilot_release']:raise ValueError('Unexpected scientific release claim')
    cases=ET.parse(base/'handoff_tests/pytest_v2.xml').findall('.//testcase')
    if len(cases)!=67 or any(c.find('failure') is not None or c.find('error') is not None or c.find('skipped') is not None for c in cases):raise ValueError('Targeted test record not passing')
    print(json.dumps({'status':'passed','files_checked':len(seen),'stored_bytes_checked':total,'targeted_test_cases':len(cases),'raw_inputs_required':False,'local_original_hashes_checked':with_local},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',type=Path,default=Path(__file__).resolve().parents[1]/'docs/thesis_research/research_snapshot_2026_11_02');parser.add_argument('--with-local',action='store_true');args=parser.parse_args();verify(args.base,args.with_local)