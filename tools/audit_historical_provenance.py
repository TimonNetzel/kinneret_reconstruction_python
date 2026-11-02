"""Read-only historical raster/model/climate provenance audit. Never executes R.

Writes summaries only to a fresh directory outside both source trees. Models
are inspected in bounded subprocesses; parse failures remain explicit failures.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import warnings
import numpy as np
import rdata
from scipy.io import netcdf_file
from audit_my_reconst_sources import metadata, sha, exact_indices
from local_rdata_stream import read_structure, read_planes, children, attributes, small_value, resolve
from local_rdata_metadata import read_metadata, MetadataSpecial
from rdata.parser._parser import RObject, RObjectType


def emit(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def leaf(obj, parser):
    obj = resolve(obj)
    if id(obj.value) in parser.blocks:
        return {'omitted_numeric_payload': parser.blocks[id(obj.value)]}
    if obj.info.type in (RObjectType.STR, RObjectType.CHAR, RObjectType.INT, RObjectType.REAL, RObjectType.LGL):
        v = small_value(obj, parser)
        if isinstance(v, bytes): return v.decode('utf-8', errors='replace')
        arr = np.asarray(v)
        if arr.size > 100: return {'length':int(arr.size), 'not_exported': True}
        return [x.item() if hasattr(x,'item') else x for x in arr.ravel()]
    return {'R_type':obj.info.type.name}


def model_attributes(obj):
    """Read only class/names strings; never convert formula environments."""
    obj = resolve(obj)
    node = resolve(obj.attributes)
    result, seen = {}, set()
    while node is not None and node.info.type == RObjectType.LIST:
        if id(node) in seen:
            raise ValueError('Cyclic model attribute pairlist')
        seen.add(id(node))
        symbol = resolve(node.tag)
        if symbol is None or symbol.info.type != RObjectType.SYM:
            raise ValueError('Model attribute tag is not a symbol')
        name = resolve(symbol.value)
        if name.info.type != RObjectType.CHAR or not isinstance(name.value, bytes):
            raise ValueError('Model attribute name is not a character string')
        key = name.value.decode('utf-8', errors='strict')
        value = resolve(node.value[0])
        if key in ('class', 'names'):
            if key in result:
                raise ValueError('Repeated model class/names attribute')
            if value.info.type != RObjectType.STR:
                raise ValueError('Model class/names attribute is not a string vector')
            chars = [resolve(v) for v in value.value]
            if any(v.info.type != RObjectType.CHAR or not isinstance(v.value, bytes) for v in chars):
                raise ValueError('Model class/names contains missing or non-character entries')
            result[key] = [v.value.decode('utf-8', errors='strict') for v in chars]
        node = resolve(node.value[1])
    if node is not None and node.info.type not in (RObjectType.NIL, RObjectType.NILVALUE):
        raise ValueError('Malformed model attribute pairlist terminator')
    return result


def model_children(obj):
    obj = resolve(obj)
    if obj.info.type != RObjectType.VEC:
        return children(obj)
    names = model_attributes(obj).get('names', [str(i+1) for i in range(len(obj.value))])
    if len(names) != len(obj.value) or len(set(names)) != len(names):
        raise ValueError('Repeated or mismatched model keys require explicit handling')
    return dict(zip(names, obj.value))


def model_worker(path):
    data, parser = read_metadata(path)
    fields = model_children(data.object)
    fits, namespaces, seen = [], set(), set()
    def walk(obj, trail, depth=0):
        if obj is None or not isinstance(obj, RObject): return
        obj = resolve(obj)
        if id(obj) in seen: return
        seen.add(id(obj))
        if depth > 300: raise RecursionError('Metadata graph depth exceeded')
        if obj.info.type == RObjectType.NAMESPACE:
            parts=tuple(v.value.decode('utf-8', errors='replace') for v in obj.value if isinstance(v.value,bytes))
            if parts: namespaces.add(parts)
        if obj.info.type == RObjectType.VEC:
            attrs = model_attributes(obj)
            classes = np.asarray(attrs.get('class',[])).tolist()
            labels = attrs.get('names', [str(i+1) for i in range(len(obj.value))])
            if 'train' in classes:
                f = model_children(obj)
                result = {'object_path':trail, 'class':classes}
                for key in ('method','levels','coefnames','metric','maximize','modelType'):
                    if key in f: result[key] = leaf(f[key],parser)
                if 'trainingData' in f:
                    td = model_children(f['trainingData'])
                    result['training_columns'] = list(td)
                    if td:
                        first=resolve(next(iter(td.values())))
                        result['training_rows'] = parser.blocks.get(id(first.value),{}).get('length',len(first.value))
                if 'bestTune' in f: result['bestTune']={k:leaf(v,parser) for k,v in model_children(f['bestTune']).items()}
                if 'finalModel' in f:
                    final = f['finalModel']; result['final_model_class']=np.asarray(model_attributes(final).get('class',[])).tolist()
                    try:
                        final_fields=model_children(final)
                        for key in ('n','decay','softmax','entropy','convergence','lev','xNames'):
                            if key in final_fields: result['final_'+key]=leaf(final_fields[key],parser)
                    except ValueError: pass
                fits.append(result)
            for key,value in zip(labels,obj.value): walk(value,trail+'/'+str(key),depth+1)
        elif isinstance(obj.value,(list,tuple)):
            for i,value in enumerate(obj.value): walk(value,trail+'/'+str(i+1),depth+1)
        elif isinstance(obj.value,RObject): walk(obj.value,trail+'/value',depth+1)
        walk(obj.attributes,trail+'/attributes',depth+1)
        walk(obj.tag,trail+'/tag',depth+1)
    with warnings.catch_warnings(record=True) as caught:
        for key,value in fields.items(): walk(value,key)
    binary=sum(len(f.get('levels',[]))==2 for f in fits)
    multi=sum(len(f.get('levels',[]))>2 for f in fits)
    return {'parse_status':'complete_structure_not_R_validation','workspace_objects':list(fields),
            'serialization_version':data.versions.format,'R_writer_integer':data.versions.serialized,
            'R_minimum_reader_integer':data.versions.minimum,'special_markers':dict(parser.special_counts),
            'numeric_blocks_omitted':len(parser.blocks),'visited_objects':len(seen),
            'parsed_objects':parser.object_count,'reused_character_objects':parser.reused_characters,
            'character_cache_entries':len(parser.character_cache),'retained_numeric_bytes':parser.retained_numeric_bytes,'namespace_references':[{'name':x[0],'serialized_version_fields':list(x[1:])} for x in sorted(namespaces)],
            'caret_fit_count':len(fits),'binary_caret_fit_count':binary,'multiclass_caret_fit_count':multi,
            'fits':fits,'warnings':sorted(set(str(w.message) for w in caught)),
            'interpretation':'caret_multiclass_models_not_279_binary_taxon_functions' if multi and not binary
                else 'requires_object_level_interpretation_no_filename_assignment',
            'R_prediction_validated':False}


def nc_array(path):
    with netcdf_file(path,mmap=False) as n:
        x=n.variables['longitude'][:].copy(); y=n.variables['latitude'][:].copy()
        v=n.variables['Taxa distributions']
        if v.dimensions != ('taxon','latitude','longitude'): raise ValueError('Unexpected declared dimension order')
        a=v[:].copy().astype(float).transpose(2,1,0)
        a[a==v._attributes['_FillValue']]=np.nan
    return x,y,a


def stats(a,b):
    both=np.isfinite(a)&np.isfinite(b)
    return {'finite_value_mismatches':int(np.count_nonzero(a[both]!=b[both])),
            'mask_mismatches':int(np.count_nonzero(np.isnan(a)!=np.isnan(b))),
            'a1_b0':int(np.count_nonzero((a==1)&(b==0))),
            'a0_b1':int(np.count_nonzero((a==0)&(b==1)))}


def raster(cfg,out):
    source=Path(cfg['additional_source_root']); original=Path(cfg['original_data_root'])
    files={'global':source/'my_kinneret/data/in/climate_data/taxa_distribution_ww.rdata',
           'eurasia_nc':source/'my_prespa/data/in/climate_data/taxa_old_new_mediterr_eurasia.nc',
           'eurasia_nc_ohrid':source/'my_ohrid/data/in/climate_data/taxa_old_new_mediterr_eurasia.nc',
           'eurasia_RData':source/'my_ohrid/data/in/climate_data/taxa_old_new_mediterr_eurasia.RData',
           'regional_nc':source/'my_ohrid/data/in/climate_data/taxa_old_new_mediterr.nc',
           'regional_RData':original/'in/taxa_old_new_mediterr.rdata','regional_climate':original/'in/tann_cru4.nc'}
    fingerprints={k:sha(p) for k,p in files.items()}
    regional=rdata.read_rda(files['regional_RData'])['taxa_old_new_mediterr']
    a=np.asarray(regional['taxadist']); names=regional['taxanames'].iloc[:,0].tolist()
    with netcdf_file(files['regional_climate'],mmap=False) as n: rx=n.variables['lon'][:].copy(); ry=n.variables['lat'][:].copy()
    ex,ey,e=nc_array(files['eurasia_nc']); nx,ny,n=nc_array(files['regional_nc'])
    d,p=read_structure(files['eurasia_RData']); ef=children(children(d.object)['taxa_old_new_mediterr'])
    enames=small_value(ef['taxanames'],p).iloc[:,0].tolist()
    eblock=p.blocks[id(ef['taxadist'].value)]; edims=tuple(map(int,attributes(ef['taxadist'])['dim']))
    e_rdata_equal=all(np.array_equal(v,e[:,:,i],equal_nan=True) for i,v in read_planes(files['eurasia_RData'],eblock,edims))
    d,p=read_structure(files['global']); gf=children(children(d.object)['taxa.distribution.ww'])
    gx=np.asarray(small_value(gf['lon'],p)); gy=np.asarray(small_value(gf['lat'],p)); gnames=small_value(gf['names'],p).tolist()
    if names!=enames or names!=gnames: raise ValueError('No exact ordered name mapping; refuse position guesses')
    gdims=tuple(map(int,attributes(gf['dist'])['dim'])); gb=p.blocks[id(gf['dist'].value)]
    gex=exact_indices(gx,ex); gey=exact_indices(gy,ey); grx=exact_indices(gx,rx); gry=exact_indices(gy,ry)
    rx_keep=np.flatnonzero(np.isin(rx,ex)); ry_keep=np.flatnonzero(np.isin(ry,ey))
    erx=exact_indices(ex,rx[rx_keep]); ery=exact_indices(ey,ry[ry_keep])
    nx_in=np.flatnonzero(np.isin(rx,nx)); ny_in=np.flatnonzero(np.isin(ry,ny))
    n_local=n[np.ix_(exact_indices(nx,rx[nx_in]),exact_indices(ny,ry[ny_in]),range(len(names)))]
    n_e=e[np.ix_(exact_indices(ex,nx),exact_indices(ey,ny),range(len(names)))]
    rows=[]; global_examples=[]; region_examples=[]; totals=Counter(); first_mask=None; common_masks=True
    for i,g in read_planes(files['global'],gb,gdims):
        if first_mask is None: first_mask=np.isnan(g)
        common_masks &= np.array_equal(np.isnan(g),first_mask)
        ge=g[np.ix_(gex,gey)]; raw=e[:,:,i]
        finite=np.isfinite(ge)&np.isfinite(raw); bad=np.argwhere(finite&(ge!=raw))
        for ix,iy in bad[:max(0,12-len(global_examples))]:
            global_examples.append(dict(position_R=i+1,name=names[i],lon=float(ex[ix]),lat=float(ey[iy]),global_value=float(ge[ix,iy]),eurasia_value=float(raw[ix,iy])))
        # Reproduce only the documented binary insertion, with the OBSERVED
        # mask. This is not recovery of the absent independent climate mask.
        predicted=np.where(np.isnan(g),np.nan,0.0)
        px,py=np.where(raw==1)
        predicted[gex[px],gey[py]]=1
        direct=stats(g,predicted)
        # R 1-based floor(k/nlon) indexing drops the final longitude column.
        predicted_edge=np.where(np.isnan(g),np.nan,0.0)
        keep=px!=len(ex)-1
        predicted_edge[gex[px[keep]],gey[py[keep]]]=1
        edge=stats(g,predicted_edge)
        crop=g[np.ix_(grx,gry)]; reference=a[:,:,i]
        local_diff=stats(crop,reference)
        for ix,iy in np.argwhere(np.isfinite(crop)&np.isfinite(reference)&(crop!=reference))[:max(0,8-len(region_examples))]:
            region_examples.append(dict(position_R=i+1,name=names[i],lon=float(rx[ix]),lat=float(ry[iy]),global_value=float(crop[ix,iy]),regional_value=float(reference[ix,iy])))
        ecrop=raw[np.ix_(erx,ery)]
        inside_g=crop[np.ix_(rx_keep,ry_keep)]; inside_r=reference[np.ix_(rx_keep,ry_keep)]
        ec_filled=np.where(np.isnan(inside_g),np.nan,np.where(ecrop==1,1.0,0.0))
        inside_compare=stats(inside_g,ec_filled)
        # cells outside source domain have the documented zero base layer
        outside=np.ones(reference.shape,dtype=bool); outside[np.ix_(rx_keep,ry_keep)]=False
        outside_diff=int(np.count_nonzero(outside&np.isfinite(crop)&np.isfinite(reference)&(crop!=reference)))
        row={'position_R':i+1,'name':names[i], 'global_vs_regional':local_diff,
             'global_vs_eurasia_on_finite_overlap':stats(ge,raw),
             'global_vs_direct_insertion_observed_mask':direct,
             'global_vs_drop_last_lon_observed_mask':edge,
             'regional_inside_vs_eurasia':stats(inside_r,ec_filled),
             'global_inside_vs_eurasia':inside_compare,'regional_differences_outside_eurasia':outside_diff}
        rows.append(row)
        for prefix,value in row.items():
            if isinstance(value,dict):
                for k,v in value.items(): totals[prefix+'.'+k]+=v
        totals['regional_differences_outside_eurasia']+=outside_diff
    summary={'fingerprints':fingerprints,'ordered_275_names_equal':names==enames==gnames,
             'eurasia_RData_and_NetCDF_arrays_equal':e_rdata_equal,
             'eurasia_nc_copies_byte_equal':fingerprints['eurasia_nc']==fingerprints['eurasia_nc_ohrid'],
             'regional_nc_vs_eurasia_on_shared_coordinates':stats(n,n_e),
             'regional_nc_vs_regional_RData_on_shared_coordinates':stats(n_local,a[np.ix_(nx_in,ny_in,range(len(names)))]),
             'regional_axes':{'lon':[float(rx[0]),float(rx[-1])],'lat':[float(ry[0]),float(ry[-1])]},
             'eurasia_axes':{'lon':[float(ex[0]),float(ex[-1])],'lat':[float(ey[0]),float(ey[-1])]},
             'regional_longitudes_outside_eurasia':[float(x) for x in rx if x not in ex],
             'global_mask_identical_all_planes':bool(common_masks),'totals':dict(totals),
             'global_vs_eurasia_examples':global_examples,'regional_difference_examples':region_examples,
             'limitations':['Insertion comparison uses observed global mask, not the absent climate mask.',
                 'No execution log identifies the exact historical generation run.',
                 'No raster corrections or substitutions; only comparisons of retained inputs.']}
    emit(out/'raster_planes.json',rows); emit(out/'raster.json',summary)
    print(json.dumps({'raster_totals':dict(totals)},ensure_ascii=False),flush=True)
    return summary


def models(cfg,out):
    source=Path(cfg['additional_source_root'])
    paths=sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower()=='.rdata'
                 and any(x in p.parts for x in ('transfer_function','ml_competition')))
    groups={}
    for path in paths: groups.setdefault(sha(path),[]).append(path)
    results=[]
    for i,(digest,group) in enumerate(groups.items()):
        path=group[0]
        try:
            proc=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',str(path)],
                                capture_output=True,text=True,encoding='utf-8',timeout=45)
            if proc.returncode!=0: result={'parse_status':'failed','error':proc.stderr[-1000:] or proc.stdout[-1000:]}
            else: result=json.loads(proc.stdout)
        except subprocess.TimeoutExpired: result={'parse_status':'timeout_45_seconds_not_classified'}
        result.update(sha256=digest,relative_paths=[p.relative_to(source).as_posix() for p in group])
        results.append(result)
        emit(out/f'model_{i+1:03d}.json',result)
        print(f'model {i+1}/{len(groups)}: {result["parse_status"]} fits={result.get("caret_fit_count","?")} {path.relative_to(source)}',flush=True)
    summary={'candidate_files':len(paths),'unique_byte_contents':len(groups),
             'parse_status_counts':dict(Counter(x['parse_status'] for x in results)),
             'complete_caret_fits':sum(x.get('caret_fit_count',0) for x in results),
             'complete_binary_caret_fits':sum(x.get('binary_caret_fit_count',0) for x in results),
             'complete_multiclass_caret_fits':sum(x.get('multiclass_caret_fit_count',0) for x in results),
             'results':results,'not_a_full_search_of_all_RData':True,
             'no_execution_no_training_no_prediction':True}
    emit(out/'models.json',summary)
    return summary


def climate(cfg,out):
    roots={'additional':Path(cfg['additional_source_root']), 'original':Path(cfg['original_data_root'])}
    results=[]
    def serial(v):
        if isinstance(v,bytes): return v.decode('utf-8',errors='replace')
        if isinstance(v,np.ndarray): return v.tolist()
        if isinstance(v,np.generic): return v.item()
        return v
    for label,root in roots.items():
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.suffix.lower()!='.nc': continue
            result={'source_tree':label,'relative_path':path.relative_to(root).as_posix(),'sha256':sha(path)}
            try:
                with netcdf_file(path,mmap=False) as n:
                    result.update(parse_status='NetCDF3_header_read',dimensions=n.dimensions,
                                  attributes={k:serial(v) for k,v in n._attributes.items()},variables={})
                    for key,v in n.variables.items():
                        info={'shape':list(v.shape),'dimensions':list(v.dimensions),
                              'attributes':{k:serial(x) for k,x in v._attributes.items()}}
                        if len(v.shape)==1 and v.shape[0]<=10000:
                            a=v[:].copy(); info['coordinate_values']=a.tolist()
                        result['variables'][key]=info
            except (TypeError,ValueError,OSError) as e: result.update(parse_status='not_decoded',error=str(e).replace(str(root),'<source_root>'))
            results.append(result)
    summary={'file_count':len(results),'parse_status_counts':dict(Counter(x['parse_status'] for x in results)),
             'no_temporal_period_inferred_from_filename_or_index':True,'results':results}
    emit(out/'climate_headers.json',summary)
    print(json.dumps({'climate_header_status':summary['parse_status_counts']}),flush=True)
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--worker',type=Path); ap.add_argument('--config',type=Path)
    ap.add_argument('--output',type=Path); ap.add_argument('--stage',choices=['raster','models','climate'])
    args=ap.parse_args()
    if args.worker:
        print(json.dumps(model_worker(args.worker),ensure_ascii=False,allow_nan=False)); return
    cfg=json.loads(args.config.read_text(encoding='utf-8-sig'))
    roots=[Path(cfg[k]).resolve(strict=True) for k in ('original_data_root','additional_source_root')]
    out=args.output.resolve()
    if any(out==root or root in out.parents or out in root.parents for root in roots): raise ValueError('Output overlaps original source tree')
    if out.exists(): raise FileExistsError('Refusing to overwrite audit output')
    before=[metadata(root) for root in roots]
    out.mkdir(parents=True)
    emit(out/'source_metadata_before.json',before)
    result=globals()[args.stage](cfg,out)
    checks={'original_metadata_unchanged':all(old==metadata(root) for old,root in zip(before,roots))}
    emit(out/'verification.json',{'stage':args.stage,'checks':checks,'all_checks_passed':all(checks.values()),
         'tool_hashes':{name:sha(Path(__file__).parent/name) for name in ('audit_historical_provenance.py','local_rdata_metadata.py','local_rdata_stream.py')},
         'parser_version':importlib.metadata.version('rdata')})
    if not all(checks.values()): raise SystemExit(1)


if __name__=='__main__': main()
