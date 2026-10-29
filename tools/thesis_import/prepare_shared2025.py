#!/usr/bin/env python3
"""Prepare a separate published-2025/common-taxon comparison dataset.

Read-only XLSX/DOCX extraction using Python's standard library. No workbook
authoring, source imputation, model fitting, or modification of the old import.
"""
import argparse
import collections
import csv
from decimal import Decimal
import json
from pathlib import Path
import re
import tempfile
import unicodedata
import zipfile
import xml.etree.ElementTree as ET

import import_data
from import_data import require, digest, number, fixed, read_pangaea, monticchio_exports

M='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
EPS=Decimal('0.000001')


def label(text):
    return re.sub(r'[^a-z0-9]','',text.casefold().replace('(continuation)',''))


def unaccented(text):
    return ''.join(c for c in unicodedata.normalize('NFD',text) if not unicodedata.combining(c))


def col_index(text):
    result=0
    for ch in text: result=result*26+ord(ch)-64
    return result


class XlsxReader:
    def __init__(self,path):
        self.archive=zipfile.ZipFile(path)
        root=ET.fromstring(self.archive.read('xl/sharedStrings.xml'))
        self.strings=[''.join(x.text or '' for x in cell.iter('{'+M+'}t')) for cell in root]

    def rows(self,sheet):
        with self.archive.open(f'xl/worksheets/sheet{sheet}.xml') as stream:
            for _,node in ET.iterparse(stream,events=('end',)):
                if node.tag=='{'+M+'}row':
                    row={}
                    for cell in node:
                        address=cell.get('r');value=cell.find('{'+M+'}v')
                        if value is None: continue
                        kind=cell.get('t','n')
                        text=self.strings[int(value.text)] if kind=='s' else value.text
                        row[re.sub(r'[0-9]','',address)]=(text,kind,cell.find('{'+M+'}f') is not None)
                    yield int(node.get('r')),row
                    node.clear()

    def close(self): self.archive.close()


def s1_records(path):
    with zipfile.ZipFile(path) as archive:
        root=ET.fromstring(archive.read('word/document.xml'))
    records=[]
    for table in root.findall('.//{'+W+'}tbl'):
        active={}
        for tr in table.findall('{'+W+'}tr'):
            cells=[];column=0
            for cell in tr.findall('{'+W+'}tc'):
                text=' '.join(x.text or '' for x in cell.iter('{'+W+'}t'))
                properties=cell.find('{'+W+'}tcPr')
                merge=properties.find('{'+W+'}vMerge') if properties is not None else None
                span=properties.find('{'+W+'}gridSpan') if properties is not None else None
                width=int(span.get('{'+W+'}val','1')) if span is not None else 1
                if merge is not None and merge.get('{'+W+'}val')!='restart': text=active.get(column,'')
                elif merge is not None: active[column]=text
                else: active.pop(column,None)
                cells.extend([text]+['']*(width-1));column+=width
            if len(cells)>=3: records.append(dict(original=cells[0],group=cells[2]))
    return records


def locked_file(source,directory):
    require(Path(source['file']).name==source['file'],'Invalid source filename')
    path=directory/source['file']
    require(path.stat().st_size==source['byte_count'] and digest(path)==source['sha256'],f"Source changed: {path.name}")
    return path


def prepare(old_raw,new_raw,old_catalog,new_catalog,features_path,output):
    require(not output.exists(),'Output directory already exists')
    old_sources={s['source_id']:s for s in json.loads(old_catalog.read_text())['files']}
    new_sources=json.loads(new_catalog.read_text())['files']
    old={key:locked_file(old_sources[key],old_raw) for key in
         ('monticchio_739452','epd_entity_804591','epd_ages_cut_804596')}
    new={s['source_id']:locked_file(s,new_raw) for s in new_sources}
    config=json.loads(features_path.read_text())
    features=config['features'];names=[x['feature_id'] for x in features]
    fossil_columns=[c for f in features for c in f['fossil_columns']]
    modern_codes=[c for f in features for c in f['modern_codes']]
    require(len(features)==37 and len(set(names))==37,'Feature contract changed')
    require(len(fossil_columns)==len(set(fossil_columns)) and len(modern_codes)==len(set(modern_codes)), 'Overlapping feature inputs')
    require(not set(fossil_columns)&set(config['exclusions']),'Excluded fossil field selected')
    threshold=number(config['minimum_shared_count_without_pinus'])
    numeric_cache={}
    def numeric(cell):
        text,kind,_=cell
        require(kind=='n','Missing or non-numeric observation; never replace with zero')
        if text not in numeric_cache: numeric_cache[text]=number(text)
        return numeric_cache[text]
    reader=XlsxReader(new['empd2_ready2025'])
    modern=[];full_values={};source_formula_cells=0;source_zero_cells=0;fractional_cells=0;name_variants=[]
    footer={};footer_labels={'Sum of pollen grains','Presence (# sites)','Absence (# sites)','Total (# sites)'}
    try:
        iterator=reader.rows(2)
        _,first=next(iterator);_,second=next(iterator)
        code_columns={v[0]:k for k,v in first.items() if col_index(k)>=4}
        require(len(code_columns)==90 and set(modern_codes)<=set(code_columns),'Published 90-code layout changed')
        codes=list(code_columns)
        headers={k:v[0] for k,v in second.items()}
        field=lambda name:next(k for k,v in headers.items() if v==name)
        total_col=field('TOTAL (# grains)');nopinus_col=field('Total without Pinus (# grains)')
        lon_col=field('Longitude');lat_col=field('Latitude');t_col=field('Tann');p_col=field('Pann')
        for _,row in iterator:
            if not row: continue
            if 'A' not in row and 'B' not in row and row.get('C',('',))[0] in footer_labels:
                footer[row['C'][0]]=row
                continue
            require('A' in row and 'B' in row and 'C' in row,'Incomplete sample key')
            sample=row['C'][0];values={code:numeric(row[code_columns[code]]) for code in codes}
            require(sample not in full_values,'Duplicate sample name')
            require(all(x>=0 for x in values.values()),'Negative pollen count')
            total=sum(values.values(),Decimal(0));without=total-values['PPINU']
            require(abs(total-numeric(row[total_col]))<=EPS,'Published total does not equal 90 counts')
            require(abs(without-numeric(row[nopinus_col]))<=EPS and without>=100,'Published no-Pinus total/filter failed')
            for column in ('CR','EG'):
                require(column in row and unaccented(row[column][0])==unaccented(sample),'Climate/ecology sample alignment mismatch')
                if row[column][0]!=sample:
                    name_variants.append(dict(sample_id=sample,column=column,source_label=row[column][0],rule='Unicode NFD accent-only equivalence'))
            lon,lat,tann,pann=(numeric(row[k]) for k in (lon_col,lat_col,t_col,p_col))
            require(-180<=lon<=180 and -90<=lat<=90 and pann>=0,'Invalid climate or coordinates')
            selected={f['feature_id']:sum((values[c] for c in f['modern_codes']),Decimal(0)) for f in features}
            selected_sum=sum(selected.values(),Decimal(0));selected_nopinus=selected_sum-selected['pinus']
            require(selected_sum>0,'Empty modern subcomposition')
            coordinate=f"{lat.quantize(Decimal('0.000001'))}|{lon.quantize(Decimal('0.000001'))}"
            modern.append(dict(sample_id=sample,source_site_id=row['A'][0],source_site_code=row['B'][0],
                longitude=fixed(lon),latitude=fixed(lat),tann_c=fixed(tann),pann_mm_per_year=fixed(pann),
                sample_context=row.get('EJ',('','s',False))[0],coordinate_group_1e6deg=coordinate,
                source90_count_sum=fixed(total),source90_count_without_pinus=fixed(without),
                shared_count_sum=fixed(selected_sum),shared_count_without_pinus=fixed(selected_nopinus),
                count_threshold_pass=int(selected_nopinus>=threshold),
                stage='comparison_data_not_model_validated',**{k:fixed(v) for k,v in selected.items()}))
            full_values[sample]=values
            fractional_cells+=sum(x!=x.to_integral_value() for x in values.values())
            source_zero_cells+=sum(x==0 for x in values.values())
            source_formula_cells+=sum(row[c][2] for c in code_columns.values())
        require(len(modern)==7634,'Published primary cohort size changed')
        require(len({r['source_site_code'] for r in modern})==7634,'Duplicate site code')
        require(set(footer)==footer_labels,'Published summary footer changed')
        for code,col in code_columns.items():
            observations=[v[code] for v in full_values.values()]
            presence=sum(v>0 for v in observations)
            for footer_label,expected in [('Sum of pollen grains',sum(observations)),('Presence (# sites)',presence),
                                          ('Absence (# sites)',7634-presence),('Total (# sites)',7634)]:
                require(abs(numeric(footer[footer_label][col])-expected)<=EPS,'Published summary footer mismatch')
        percentage_rows=0;max_percentage_error=Decimal(0);percentage_sites=set()
        by_sitecode={r['source_site_code']:r for r in modern}
        for index,row in reader.rows(3):
            if index<=2 or not row:continue
            if 'A' not in row and 'B' not in row and row.get('C',('',))[0] in {
                'MAX %','MIN % (>0)','MEAN % (>0)','MEDIAN %(>0)','SD % (>0)','Presence (# sites)'}:continue
            require('B' in row and row['B'][0] in by_sitecode,f'Unknown percentage site: {index} {row.get("C")}')
            record=by_sitecode[row['B'][0]];sample=record['sample_id']
            require(row['A'][0]==record['source_site_id'] and unaccented(row['C'][0])==unaccented(sample),'Percentage sample key mismatch')
            require(sample not in percentage_sites,'Duplicate percentage sample');percentage_sites.add(sample)
            if row['C'][0]!=sample:name_variants.append(dict(sample_id=sample,column='percentage:C',source_label=row['C'][0],rule='Unicode NFD accent-only equivalence'))
            vals=full_values[sample];den=sum(vals.values(),Decimal(0));pct_sum=Decimal(0)
            for code,col in code_columns.items():
                pct=numeric(row[col]);pct_sum+=pct
                max_percentage_error=max(max_percentage_error,abs(pct-100*vals[code]/den))
            require(abs(pct_sum-100)<=EPS,'Published percentages do not sum to 100')
            percentage_rows+=1
        require(percentage_rows==7634 and max_percentage_error<=EPS,'Published percentage contract failed')
        coordinate_rows=0;by_sitecode={r['source_site_code']:r for r in modern};bad_coordinate_headers=False
        for index,row in reader.rows(6):
            if index==1:
                bad_coordinate_headers=row['B'][0]=='Longitude (°N)' and row['C'][0]=='Latitude (°E)'
                continue
            if not row:continue
            record=by_sitecode[row['A'][0]]
            require(abs(number(record['longitude'])-numeric(row['B']))<=EPS and
                    abs(number(record['latitude'])-numeric(row['C']))<=EPS,'Separate coordinate table mismatch')
            coordinate_rows+=1
        require(coordinate_rows==7634,'Coordinate rows changed')
        climate_rows=0
        for index,row in reader.rows(7):
            if index==1:
                climate_columns={v[0]:k for k,v in row.items()}
                continue
            if not row:continue
            record=by_sitecode[row['A'][0]]
            require(abs(number(record['tann_c'])-numeric(row[climate_columns['Tann']]))<=EPS and
                    abs(number(record['pann_mm_per_year'])-numeric(row[climate_columns['Pann']]))<=EPS,'Separate climate table mismatch')
            climate_rows+=1
        require(climate_rows==7634,'Climate rows changed')
        intermediate_labels=[]
        for index,row in reader.rows(9):
            if index>=3 and 'A' in row:intermediate_labels.append(row['A'][0])
    finally: reader.close()
    s1=s1_records(new['mmc1'])
    labels_to_codes={label(headers[col]):code for code,col in code_columns.items()}
    labels_to_codes[label('Cannabaceae/Urticaceae')]='CANURT'  # explicit source spelling variant
    labels_to_codes[label('Apiaceae')]='APIACE'  # workbook additionally gives synonym Umbelliferae
    members=collections.defaultdict(set)
    for record in s1:
        code=labels_to_codes.get(label(record['group']))
        if code: members[code].add(record['original'].strip())
    for code in modern_codes:require(bool(members[code]),f'S1 group evidence not found: {code}')
    evidence=[dict(feature_id=f['feature_id'],fossil_columns='|'.join(f['fossil_columns']),modern_codes='|'.join(f['modern_codes']),
        published_labels='|'.join(headers[code_columns[c]] for c in f['modern_codes']),
        published_member_count=sum(len(members[c]) for c in f['modern_codes']),
        member_examples='|'.join(sorted({x for c in f['modern_codes'] for x in members[c]})[:8]),
        scope='explicit_shared_subcomposition_not_complete_spectrum') for f in features]
    _,_,pollen=read_pangaea(old['monticchio_739452'])
    require(all(c in pollen[0] for c in fossil_columns),'Fossil column missing')
    oak=[r for r in pollen if number(r['Que.r-T [#]'])+number(r['Que.i [#]'])>0]
    require(len(oak)==23 and all(number(r['Que [#]'])==number(r['Que.r-T [#]'])+number(r['Que.i [#]']) for r in oak),'Oak total/subdivision contract failed')
    fossil=[]
    with tempfile.TemporaryDirectory() as temp:
        work=Path(temp)
        monticchio_exports(old['monticchio_739452'],old['epd_entity_804591'],old['epd_ages_cut_804596'],work)
        with (work/'monticchio_samples.csv').open(encoding='utf-8',newline='') as stream:
            linked=list(csv.DictReader(stream))
        for sample in linked:
            raw=pollen[int(sample['source_row'])-1]
            vals={f['feature_id']:sum((number(raw[c]) for c in f['fossil_columns']),Decimal(0)) for f in features}
            den=sum(vals.values(),Decimal(0));without=den-vals['pinus']
            require(den>0,'Empty fossil subcomposition')
            fossil.append(dict(sample_id=sample['sample_id'],source_row=sample['source_row'],depth_m=sample['pollen_depth_m'],
                age_bp=sample['age_bp'],age_min_bp=sample['age_min_bp'],age_max_bp=sample['age_max_bp'],
                chronology_depth_m=sample['chronology_depth_m'],chronology_join_kind=sample['join_kind'],
                chronology_uncertainty_class=sample['uncertainty_class'],chronology_entity_id=sample['chronology_entity_id'],
                chronology_event=sample['chronology_event'],chronology_status=sample['scientific_status'],
                chronology_depth_offset_m=sample['depth_offset_m'],shared_count_sum=fixed(den),shared_count_without_pinus=fixed(without),
                count_threshold_pass=int(without>=threshold),stage='comparison_data_not_model_validated',**{k:fixed(v) for k,v in vals.items()}))
    require(len(fossil)==69,'Fossil cohort changed')
    coords=collections.Counter(r['coordinate_group_1e6deg'] for r in modern)
    summary=dict(schema_version=1,variant=config['variant'],modern_samples=len(modern),fossil_samples=len(fossil),features=len(features),
        modern_count_threshold_pass=sum(r['count_threshold_pass'] for r in modern),
        fossil_count_threshold_pass=sum(r['count_threshold_pass'] for r in fossil),
        minimum_shared_count_without_pinus=fixed(threshold),explicit_zero_cells_in_source90=source_zero_cells,
        fractional_cells_in_source90=fractional_cells,source_count_formula_cells=source_formula_cells,
        published_percentage_rows_checked=percentage_rows,max_published_percentage_error=fixed(max_percentage_error),
        separate_climate_rows_checked=climate_rows,
        published_percent_denominator='all_90_taxa_including_Pinus',
        output_percent_denominator=config['percentage_denominator'],
        coordinate_groups_1e6deg=len(coords),samples_in_repeated_coordinate_groups=sum(v for v in coords.values() if v>1),
        separate_coordinate_table_headers_swap_degree_directions=bad_coordinate_headers,
        combined_sheet_accent_only_name_variants=name_variants,
        intermediate_table_rows_with_original_labels=len(intermediate_labels),
        intermediate_tail_labels=intermediate_labels[-7:],intermediate_declared_taxa_in_publication=840,
        intermediate_table_used_to_rebuild_counts=False,old_sqlite_used_for_modern_values=False,
        model_fitted=False,scientific_status='comparison_data_not_model_validated')
    return modern,fossil,evidence,summary


def write_outputs(output,modern,fossil,evidence,summary,features_path,old_catalog,new_catalog):
    names=[x['feature_id'] for x in json.loads(features_path.read_text())['features']]
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        stage=Path(temp)/'result';stage.mkdir()
        for kind,rows in [('modern',modern),('fossil',fossil)]:
            for suffix,percent in [('counts',False),('percent',True)]:
                with (stage/f'{kind}_shared_{suffix}.csv').open('w',encoding='utf-8',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader()
                    for row in rows:
                        out=dict(row)
                        if percent:
                            den=number(row['shared_count_sum'])
                            for key in names:out[key]=fixed(100*number(row[key])/den)
                        writer.writerow(out)
        with (stage/'feature_evidence.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(evidence[0]),lineterminator='\n');writer.writeheader();writer.writerows(evidence)
        (stage/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        manifest=dict(schema_version=1,script_sha256=digest(Path(__file__)),helper_sha256=digest(Path(import_data.__file__)),
            feature_policy_sha256=digest(features_path),old_source_catalog_sha256=digest(old_catalog),new_source_catalog_sha256=digest(new_catalog),
            outputs={p.name:dict(sha256=digest(p),byte_count=p.stat().st_size) for p in sorted(stage.iterdir())},
            source_licenses={'modern':'CC-BY-4.0','fossil_counts':'CC-BY-NC-3.0','fossil_chronologies':'CC-BY-3.0'},
            note='Only the data inputs and the explicit shared subcomposition are prepared; no model or validation score.')
        (stage/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        stage.rename(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    base=Path(__file__).resolve().parents[2]/'docs/thesis_research'
    parser.add_argument('--raw-dir',type=Path,required=True)
    parser.add_argument('--published-dir',type=Path,required=True)
    parser.add_argument('--old-catalog',type=Path,default=base/'source_audit_2026_10_29/sources.json')
    parser.add_argument('--published-catalog',type=Path,default=base/'modern2025_2026_10_29/sources_2025.json')
    parser.add_argument('--features',type=Path,default=base/'modern2025_2026_10_29/shared_features.json')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    modern,fossil,evidence,summary=prepare(args.raw_dir,args.published_dir,args.old_catalog,args.published_catalog,args.features,args.output_dir)
    write_outputs(args.output_dir,modern,fossil,evidence,summary,args.features,args.old_catalog,args.published_catalog)
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
