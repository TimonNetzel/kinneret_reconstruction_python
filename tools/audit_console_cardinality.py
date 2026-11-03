"""Compare a historical console trace to saved raster class cardinalities.

No R execution, model fitting, synthetic sample generation, or fossil inference.
Matching counts constrain a candidate path; they do not identify climate values,
random seeds, spatial membership, or serialized model weights.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import numpy as np
from audit_my_reconst_sources import sha
from local_rdata_metadata import read_metadata
from local_rdata_stream import children, attributes, read_planes, small_value

ANSI = re.compile(r'\x1b\[[0-9;]*m')


def parse_console(text):
    lines=ANSI.sub('', text).splitlines()
    starts=[(i,int(m.group(1))) for i,line in enumerate(lines)
            if (m:=re.fullmatch(r'\[1\] (\d+)',line))]
    if not starts or [p for _,p in starts]!=list(range(1,len(starts)+1)):
        raise ValueError('Console positions must be consecutive, ordered, and 1-based')
    blocks=[]
    for j,(start,position) in enumerate(starts):
        end=starts[j+1][0] if j+1<len(starts) else len(lines)
        chunk=lines[start:end]
        labels=[m.group(1) for line in chunk if (m:=re.search(r'model label\s*:\s*(\S+)',line))]
        shapes=[(int(m.group(1)),int(m.group(2))) for line in chunk
                if (m:=re.search(r'data\s*:\s*(\d+)\s+rows\s+(\d+)\s+cols',line))]
        if len(labels)!=len(shapes):
            raise ValueError('Model labels and data shapes are not paired')
        if shapes and len(set(shapes))!=1:
            raise ValueError('Models at one printed position have different data shapes')
        blocks.append({'printed_position':position,'first_line':start+1,'last_line':end,
                       'model_labels':labels,'logged_rows':shapes[0][0] if shapes else None,
                       'logged_columns':shapes[0][1] if shapes else None})
    return blocks


def class_cardinality(presence, absence, fraction=0.7):
    if presence<0 or absence<0 or not 0<fraction<=1:
        raise ValueError('Invalid class sizes or fraction')
    pos,neg=math.ceil(presence*fraction),math.ceil(absence*fraction)
    branch=2*pos<neg and pos>1
    # smotefamily 1.3.1 n_dup_max and SMOTE$syn_data: original minority
    # observations are absent from syn_data and are not re-added by this R branch.
    duplicates=(neg-pos)//pos if branch else None
    return {'presence':presence,'absence':absence,'train_presence':pos,'train_absence':neg,
            'source_skip':presence<=10,'source_smote_branch':branch,
            'duplications_per_minority':duplicates,
            'synthetic_only_rows':neg+pos*duplicates if branch else None,
            'with_original_minority_rows':neg+pos*(duplicates+1) if branch else None,
            'unbalanced_training_rows':pos+neg}


def run(config, output):
    repo=Path(__file__).resolve().parents[1]
    cfg=json.loads(config.read_text(encoding='utf-8-sig'))
    roots={'original':Path(cfg['original_data_root']).resolve(strict=True),
           'additional':Path(cfg['additional_source_root']).resolve(strict=True)}
    output=output.resolve()
    if repo not in output.parents or any(output==p or p in output.parents for p in roots.values()):
        raise ValueError('Output must remain in this checkout, outside both original trees')
    output.mkdir(parents=True,exist_ok=False)
    log_path=roots['original']/'logfile_trafo.txt'
    distribution=roots['additional']/'my_ohrid/data/in/climate_data/taxa_distribution.rdata'
    initial_hashes={'console':sha(log_path),'distribution':sha(distribution)}
    blocks=parse_console(log_path.read_text(encoding='utf-8'))
    data,parser=read_metadata(distribution)
    fields=children(children(data.object)['taxa.distribution'])
    obj=fields['distribution'];dimensions=tuple(int(x) for x in attributes(obj)['dim'])
    lat=np.asarray(small_value(fields['lat'],parser));lon=np.asarray(small_value(fields['lon'],parser))
    names=np.asarray(small_value(fields['names'],parser)).tolist()
    if dimensions!=(len(lon),len(lat),len(blocks)) or len(names)!=len(blocks):
        raise ValueError('Stored axes and complete console positions do not align')
    if not np.all(np.diff(lat)>0) or not np.all(np.diff(lon)>0) or not np.any(lat>=0):
        raise ValueError('Unexpected stored coordinate axes')
    if any(b['model_labels'] not in ([],['nnet','mda','qda','gbm']) for b in blocks):
        raise ValueError('Unexpected model-output sequence')
    rows=[];reference_mask=None
    for i,plane in read_planes(distribution,parser.blocks[id(obj.value)],dimensions):
        if not np.all(np.isnan(plane)|(plane==0)|(plane==1)):
            raise ValueError('Source is not a binary/missing distribution')
        mask=np.isnan(plane)
        if reference_mask is None:reference_mask=mask.copy()
        if not np.array_equal(mask,reference_mask):raise ValueError('Source planes have different masks')
        row={**blocks[i],'snapshot_name_not_historical_taxon_assignment':names[i]}
        for label,y in [('north',lat>=0),('world',np.ones(len(lat),dtype=bool))]:
            values=plane[:,y]
            pos=int(np.count_nonzero(values==1));neg=int(np.count_nonzero(values==0))
            row[label]=class_cardinality(pos,neg)
            row[label+'_all_data']=class_cardinality(pos,neg,1.0)
        rows.append(row)
    logged=[r for r in rows if r['logged_rows'] is not None]
    skipped=[r['printed_position'] for r in rows if r['logged_rows'] is None]
    controls={key:sum(r['logged_rows']==r[region][field] for r in logged)
              for key,region,field in [('north_synthetic_only','north','synthetic_only_rows'),
                                      ('world_synthetic_only','world','synthetic_only_rows'),
                                      ('north_original_minority_included','north','with_original_minority_rows'),
                                      ('north_no_SMOTE','north','unbalanced_training_rows'),
                                      ('north_no_holdout_synthetic_only','north_all_data','synthetic_only_rows')]}
    groups=defaultdict(list)
    for r in logged:groups[r['logged_rows']].append(r['printed_position'])
    result={'source_hashes':initial_hashes,'dimensions':list(dimensions),'coordinate_ranges':{'lon':[float(lon[0]),float(lon[-1])],'lat':[float(lat[0]),float(lat[-1])]},
            'console_positions':len(rows),'positions_with_model_output':len(logged),'all_model_shapes_count':sum(len(r['model_labels']) for r in rows),
            'skipped_console_positions':skipped,'predicted_low_presence_positions':[r['printed_position'] for r in rows if r['north']['source_skip']],
            'control_match_counts':controls,'distinct_logged_row_counts':len(groups),
            'row_count_collision_groups':[{'logged_rows':k,'positions':v} for k,v in groups.items() if len(v)>1],
            'all_logged_cases_enter_source_smote_branch':all(r['north']['source_smote_branch'] for r in logged),
            'all_logged_columns_are_five':all(r['logged_columns']==5 for r in logged),
            'northern_finite_cell_counts':sorted({r['north']['presence']+r['north']['absence'] for r in rows}),
            'names_are_snapshot_metadata_only':True,'no_historical_taxon_or_seed_or_climate_value_assignment':True,
            'R_prediction_validated':False,'no_training_or_sampler':True,'rows':rows,
            'source_hashes_unchanged':initial_hashes=={'console':sha(log_path),'distribution':sha(distribution)},
            'tool_sha256':sha(Path(__file__))}
    (output/'console_cardinality.json').write_text(json.dumps(result,ensure_ascii=True,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','row_count_collision_groups','source_hashes')},ensure_ascii=True),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args();run(args.config,args.output)


if __name__=='__main__':main()
