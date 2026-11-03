"""Bounded read-only Kinneret input audit; no R, fits, sampling or traversal.

Run from this existing checkout. Private roots come from .local/my_reconst_paths.json.
Exports provenance and row/column crosswalks, never pollen counts or model arrays.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
import zipfile
import numpy as np
import rdata

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO/'tools'))
from local_rdata_metadata import read_metadata
from local_rdata_stream import children, small_value, resolve, attributes
cfg = json.loads((REPO/'.local/my_reconst_paths.json').read_text(encoding='utf-8'))
roots = {'original':Path(cfg['original_data_root']), 'additional':Path(cfg['additional_source_root']), 'repository':REPO}
sources = {}
def bind(scope, rel):
    key=scope+':'+rel
    if key not in sources:
        p=roots[scope]/rel
        sources[key]={'scope':scope,'relative_path':rel,'exists':p.is_file()}
        if p.is_file():sources[key].update(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    return key

def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.reader(f))
def write_csv(name, rows):
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
def array_hash(x):
    return hashlib.sha256(np.asarray(x,dtype='>f8').ravel(order='F').tobytes()).hexdigest()

old_rel='in/core_data_all_andrea_taxa_additional_sites.rdata'
raw_rel='my_kinneret/data/in/core/pollen_counts_epd_taxanomy.csv'
biom_rel='my_kinneret/data/in/core/core_data.rdata'
age_rel='my_kinneret/data/out/age_depth/rho_depth_given_age.rdata'
for scope,rel in [('original',old_rel),('additional',raw_rel),('additional',biom_rel),('additional',age_rel)]:bind(scope,rel)
data,reader=read_metadata(roots['original']/old_rel)
sites=children(children(data.object)['core.data'])
matching=[]
for index,obj in sites.items():
    f=children(obj)
    if 'sitename' in f and str(np.asarray(small_value(f['sitename'],reader)).ravel()[0])=='Kinneret':matching.append((index,f))
if len(matching)!=1:raise ValueError('Require one explicitly named Kinneret entry')
old_slot,fields=matching[0]
old=small_value(fields['all.available.taxa'],reader)
old_depth=np.asarray(small_value(fields['depth'],reader))
old_age=np.asarray(small_value(fields['age'],reader))
old_spectrum=np.asarray(old['taxa.spectrum'])
raw=read_csv(roots['additional']/raw_rel)
raw_depth=np.asarray([float(row[0]) for row in raw[1:]])
raw_values=np.asarray([[float(v) if v.strip() else np.nan for v in row[1:61]] for row in raw[1:]])
# A diagnostic identity only: missing raw cells count as zero in these row sums.
values_for_identity=np.nan_to_num(raw_values,nan=0.0)
row_sum=values_for_identity.sum(axis=1)
if not np.all(row_sum>0):raise ValueError('Invalid denominator')
biom=rdata.read_rda(roots['additional']/biom_rel)['core_data']
chrono=rdata.read_rda(roots['additional']/age_rel)['output']
for label,arr in [('old',old_depth),('biom',biom['depths']),('chronology',chrono['depth'])]:
    if not np.array_equal(raw_depth,arr):raise ValueError('Depth rows do not match exactly: '+label)
if len(np.unique(raw_depth))!=len(raw_depth):raise ValueError('Ambiguous duplicate sample depths')
if not np.array_equal(old_age,chrono['age']):raise ValueError('Old and chronology ages differ')
block=reader.blocks[id(resolve(fields['rho_depth_given_age']).value)]
chrono_equal=block['xdr_sha256']==array_hash(chrono['rho_depth_given_age'])
if not chrono_equal:raise ValueError('Chronology numeric payload differs')
sample_rows=[]
for i,depth in enumerate(raw_depth):
    sample_rows.append({'sample_row_R_1_based':i+1,'csv_depth_cm':float(depth),
      'older_collection_depth_cm':float(old_depth[i]),'biom_depth_cm':float(biom['depths'][i]),
      'chronology_depth_cm':float(chrono['depth'][i]),'older_age_cal_a_BP':float(old_age[i]),
      'chronology_age_cal_a_BP':float(chrono['age'][i]),
      'depths_exactly_equal':True,'ages_exactly_equal':True,
      'final_v1_2_sample_binding':'unresolved_missing_base_entry'})
write_csv('sample_crosswalk.csv',sample_rows)

column_rows=[]
normalized=values_for_identity/row_sum[:,None]
for j,name in enumerate(old['taxa.names']):
    matches=[k for k in range(60) if np.array_equal(old_spectrum[:,j],normalized[:,k])]
    if len(matches)!=1:raise ValueError('Require one exact 160-value column match: '+str(name))
    k=matches[0]
    column_rows.append({'older_column_R_1_based':j+1,'older_taxon_name':str(name),
      'raw_csv_column_R_1_based':k+2,'raw_pollen_type':raw[0][k+1],
      'identity':'raw count / row sum of 60 taxon columns; empty raw cells zero for this check',
      'compared_rows':len(raw_depth),'max_absolute_error':float(np.max(abs(old_spectrum[:,j]-normalized[:,k]))),
      'raw_column_empty_cells':int(np.isnan(raw_values[:,k]).sum()),
      'status':'exact_numeric_identity_of_older_candidate_not_final_v1_2'})
write_csv('older_taxon_crosswalk.csv',column_rows)

cross_rel='.local/my_reconst_source_audit_v3/global_taxon_crosswalk.csv'
bind('repository',cross_rel)
with (REPO/cross_rel).open(encoding='utf-8-sig',newline='') as f:cross=list(csv.DictReader(f))
requirements=[]
for source_pos,name,lowercase in [(5,'Olea Europaea','Olea europaea'),(6,'Phillyrea Latifolia','Phillyrea latifolia'),(8,'Pistacia Lentiscus','Pistacia lentiscus')]:
    def find(n):return [r for r in cross if r['source']=='my_ohrid' and r['name']==n and r['retained_after_documented_deletion']=='True']
    m=find(name);other=find(lowercase)
    if len(m)!=1 or len(other)!=1:raise ValueError('Ambiguous exact-case name binding')
    requirements.append({'base_column_before_selection_R_1_based':source_pos,'explicit_consumer_name':name,
      'original_source_line':{5:148,6:149,8:150}[source_pos],
      'older_name_at_same_position':str(old['taxa.names'][source_pos-1]),
      'older_raw_type_at_same_position':column_rows[source_pos-1]['raw_pollen_type'],
      'ohrid_global_position_before_documented_deletion':m[0]['global_position_R_1_based'],
      'consumer_pool_position_after_documented_deletion':m[0]['position_after_index_operation_R_1_based'],
      'separate_lowercase_name':lowercase,'separate_lowercase_consumer_pool_position':other[0]['position_after_index_operation_R_1_based'],
      'required_fields':'prod.x1;prod.x2;x1range.bck;x2range.bck',
      'numerical_tf_status':'missing','scope':'explicit current consumer assignments only; not complete final subset'})
write_csv('explicit_tf_requirements.csv',requirements)

for rel in ['main_v1.2.R','src/check_ages.R','src/mcmc_prepare.R','src/mcmc_case_studies.R']:
    bind('original',rel)
for rel in ['my_kinneret/main.R','my_kinneret/src/bacon_kinneret.R','my_kinneret/src/bbm_reconstruction.R']:
    bind('additional',rel)
for rel in ['in/core_data_additional_sites_v1.2.rdata','in/core_data_kinneret.rdata',
            'out/transfer_functions/mixed_partial_arts_emd.rdata','out/transfer_functions/mixed_partial_arts_emd_info.rdata']:
    bind('original',rel)

sys.path.insert(0,str(REPO/'.local/console_provenance_v1/pdf_runtime'))
import pymupdf
plot_relations=[]
plot_labels={}
archive_rel='out/plots/mcmc/v5.0/diss_final.zip'
bind('original',archive_rel)
with zipfile.ZipFile(roots['original']/archive_rel) as archive:
    for filename in ['taxa_weights.pdf','gibbs_taxa_weights.pdf','rw_diff_taxa_weights.pdf']:
        rel='out/plots/case_studies/kinneret/'+filename
        key=bind('original',rel)
        with pymupdf.open(roots['original']/rel) as pdf:
            text='\n'.join(p.get_text() for p in pdf)
        labels=[]
        for line in text.splitlines():
            try:float(line);break
            except ValueError:
                if line not in ['Difference of taxa weights', 'Taxa weights']:labels.append(line)
        plot_labels[filename]={'first_panel_labels_in_text_order':labels,
                               'first_panel_label_count':len(labels),
                               'scope':'display labels, not numerical TF keys or base-column indices'}
        member='figures/kinneret/'+filename
        if member in archive.namelist():
            plot_relations.append({'source':key,'member':member,
              'byte_equal':hashlib.sha256(archive.read(member)).hexdigest()==sources[key]['sha256']})

# A prior checksum alone cannot replace reading the current original sources.
unchanged=[]
for item in sources.values():
    p=roots[item['scope']]/item['relative_path']
    unchanged.append(p.is_file()==item['exists'] and (not item['exists'] or hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']))
if not all(unchanged):raise ValueError('Source changed while audit was reading')
summary={'schema_version':1,'date':'2026-11-02','base_commit':'756865bea123b6494fd3a0e7babfce968bfcb29e',
 'status':'partial_kinneret_binding_final_base_and_numeric_transfer_functions_missing',
 'older_collection':{'site_count':len(sites),'kinneret_slot_R_1_based':int(old_slot),
                    'sitename':'Kinneret','spectrum_shape':list(old_spectrum.shape),'taxon_names':list(map(str,old['taxa.names']))},
 'sample_binding':{'rows':len(raw_depth),'unique_depths':True,'all_four_depth_vectors_exactly_equal':True,
    'older_age_and_chronology_age_exactly_equal':True,'age_range_cal_a_BP':[float(old_age.min()),float(old_age.max())],
    'rho_matrix_shape':list(np.asarray(chrono['rho_depth_given_age']).shape),'rho_xdr_payload_sha256':block['xdr_sha256'],
    'rho_numeric_payload_exactly_equal':chrono_equal,'final_v1_2_binding':'unresolved'},
 'older_column_binding':{'exactly_bound_columns':len(column_rows),'compared_values':int(old_spectrum.size),
    'raw_taxon_columns':60,'raw_empty_cells':int(np.isnan(raw_values).sum()),
    'olea_max_error_if_AP_plus_NAP_denominator_used':float(np.max(abs(old_spectrum[:,0]-values_for_identity[:,0]/np.asarray([float(v[-3])+float(v[-2]) for v in raw[1:]]))))},
 'biom_candidate':{'spectrum_shape':list(np.asarray(biom['taxa.spectrum']).shape),'biome_shape':list(np.asarray(biom['biomes.appear']).shape),
    'replacement_for_final_dissertation_base':False},
 'reference_binding':{'older_qualitative_equals_csv_tree_percentage':bool(np.array_equal(small_value(fields['qualitativ_reconstruction'],reader),np.asarray([float(v[-1]) for v in raw[1:]]))),
    'current_consumer_replaces_reference_from_csv':True},
 'chronology_interface':{'preserved_candidate':'matrix plus flat age vector','bacon_source':'first saves list/age-grid variant, later overwrites same filename with mean-age matrix',
    'current_consumer':'age.reso=2 and [[age.reso]] expect a list; matrix candidate cannot be silently substituted'},
 'plot_labels':plot_labels,'plot_archive_comparisons':plot_relations,
 'explicit_consumer_tf_assignments':len(requirements),'complete_final_tf_subset_known':False,
 'all_explicit_sources_unchanged':True,'sources':sources,
 'limits':['A mathematical identity does not establish the historical producer execution',
           'No final v1.2 sample/taxon contract has been restored',
           'Display-name rewriting prevents inversion of the plot labels into a unique TF key list',
           'No numerical TF arrays, taxon draws, climate curves or new Bacon outputs generated']}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({k:summary[k] for k in ['status','sample_binding','older_column_binding','explicit_consumer_tf_assignments','complete_final_tf_subset_known','all_explicit_sources_unchanged']},ensure_ascii=False))
