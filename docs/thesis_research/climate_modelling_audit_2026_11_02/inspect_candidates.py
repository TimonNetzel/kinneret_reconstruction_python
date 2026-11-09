"""Bounded structure checks for selected, explicitly relevant candidates only."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.io import netcdf_file
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tools'))
from local_rdata_stream import read_structure,children,attributes,small_value
OUT=Path(__file__).resolve().parent


def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    cfg=json.loads((ROOT/'.local/my_reconst_paths.json').read_text(encoding='utf-8-sig'))
    root=Path(cfg['climate_modelling_source_root']).resolve(strict=True)
    hashes=json.loads((OUT/'candidate_hashes.json').read_text())
    expected={x['source']:x['sha256'] for x in hashes}
    def verify(rel):
        p=root/rel; before=sha(p)
        if 'climate_modelling:'+rel in expected and before!=expected['climate_modelling:'+rel]:
            raise ValueError('Candidate changed since collection: '+rel)
        return p,before
    rel='pollen/transfer_functions/input_data/data_out/taxa_distribution_ww.rdata'
    p,h=verify(rel)
    data,parser=read_structure(p,max_vector_bytes=8192)
    top=children(data.object)
    fields=children(top['taxa.distribution.ww'])
    dims=list(map(int,attributes(fields['dist'])['dim']))
    axes={}
    for k in ['lon','lat']:
        a=np.asarray(small_value(fields[k],parser)).ravel()
        axes[k]={'length':len(a),'first':float(a[0]),'last':float(a[-1]),
                 'strictly_increasing':bool(np.all(np.diff(a)>0)),
                 'unique_steps':np.unique(np.diff(a)).tolist(),
                 'units':'not stored as units attribute' if 'units' not in attributes(fields[k]) else str(attributes(fields[k])['units'])}
    names=np.asarray(small_value(fields['names'],parser)).ravel().tolist()
    assert dims==[axes['lon']['length'],axes['lat']['length'],len(names)]
    dist={'source':'climate_modelling:'+rel,'sha256':h,'top_level_objects':list(top),
          'fields':list(fields),'dimensions_lon_lat_taxa':dims,'axes':axes,'taxa_count':len(names),
          'taxa_names_ordered_sha256':hashlib.sha256(json.dumps(names,ensure_ascii=False).encode()).hexdigest(),
          'numeric_payload_materialized':False,'skipped_numeric_blocks':len(parser.blocks),
          'skipped_numeric_bytes':sum(b['bytes'] for b in parser.blocks.values()),
          'small_numeric_bytes_retained':parser.retained_numeric_bytes,
          'consumer_fields_match_main_R_129_130':all(k in fields for k in ['dist','names']),
          'role':'geographic occurrence grid; not numerical historical climate transfer functions',
          'historical_run_identity_claimed':False,'hash_unchanged_after_read':sha(p)==h}
    csvs=[]
    for name in ['speleo_info.csv','ages_info.csv']:
        rel='speleothems/speleo_data/'+name;p,h=verify(rel)
        with p.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f); rows=list(reader); fields=reader.fieldnames
        required=['site_name','site_id','entity_id','latitude','longitude','elevation'] if name=='speleo_info.csv' else ['entity_id','depth_dating','corr_age','corr_age_uncert_pos']
        widths=CounterWidths(p)
        csvs.append({'source':'climate_modelling:'+rel,'sha256':h,'rows':len(rows),'columns':fields,
          'row_widths':widths,'required_metadata_fields_present':all(k in fields for k in required),
          'entity_id_unique_count':len({r['entity_id'] for r in rows}),
          'units_metadata':'not declared in CSV headers; no inferred age/depth units',
          'target_reference':'original:main_v1.2.R:391-392 points to SISAL_V2, not this source folder',
          'SISAL_V2_identity_proven':False,'historical_run_identity_claimed':False,
          'hash_unchanged_after_read':sha(p)==h})
    nc=[]
    for rel in ['verification/Data/my/pmip3_21k_sftlf_v0.nc',
                'pollen/transfer_functions/input_data/data_in/cru/pann_cru4.nc',
                'pollen/transfer_functions/input_data/data_in/cru/tann_cru4.nc']:
        p,h=verify(rel)
        with netcdf_file(p,mmap=True) as f:
            variables={name:{'dimensions':list(v.dimensions),'shape':list(v.shape),
                       'attributes':{a:(b.decode('utf-8',errors='replace') if isinstance(b,bytes) else str(b))
                       for a,b in v._attributes.items() if a in ['units','long_name','standard_name']}}
                       for name,v in f.variables.items()}
            axes={}
            for name in ['lon','lat','time']:
                if name not in f.variables: continue
                v=f.variables[name]
                if v.data.size>4096: continue
                a=np.array(v[:],copy=True).ravel()
                axes[name]={'length':len(a),'first':float(a[0]),'last':float(a[-1])}
            del v  # release the last mmap-backed variable before closing the file
            nc.append({'source':'climate_modelling:'+rel,'sha256':h,'dimensions':dict(f.dimensions),
                       'variables':variables,'small_axes':axes,'climate_or_mask_arrays_loaded':False,
                       'has_required_mask2_lon_lat':all(x in variables for x in ['mask2','lon','lat']),
                       'historical_version_identity_claimed':False})
        nc[-1]['hash_unchanged_after_read']=sha(p)==h
    report={'distribution':dist,'metadata_csv_candidates':csvs,'nearby_netCDF_variants':nc,
        'large_pollen_RData_variants':'Hashes/filenames only; no full object or numeric decoding; not accepted as v4.0',
        'R_execution':False,'models_or_samplers_started':False,'final_variant_selected':False}
    (OUT/'candidate_structures.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8', newline='\n')
    print(json.dumps(report,ensure_ascii=True))


def CounterWidths(p):
    from collections import Counter
    with p.open(encoding='utf-8-sig',newline='') as f:
        return dict(Counter(map(len,csv.reader(f))))


if __name__=='__main__': main()
