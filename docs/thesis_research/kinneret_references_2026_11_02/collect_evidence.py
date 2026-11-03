"""Read-only, explicitly scoped Kinneret reference/anchor contract audit.

No R execution, calibration, random draws, wavelets, training or reconstruction.
Source snippets and metadata are exported; input arrays remain in original roots.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
import zipfile
import numpy as np
import rdata

REPO=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
cfg=json.loads((REPO/'.local/my_reconst_paths.json').read_text(encoding='utf-8'))
roots={'original':Path(cfg['original_data_root']),'additional':Path(cfg['additional_source_root']),'repository':REPO}
sources={}
def bind(scope,rel):
    key=scope+':'+rel
    if key not in sources:
        p=roots[scope]/rel
        item={'scope':scope,'relative_path':rel,'exists':p.is_file()}
        if p.is_file():item.update(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        sources[key]=item
    return key

def json_write(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')

ranges={
 'main_v1.2.R':[(143,154),(325,350),(388,401),(654,675)],
 'src/check_ages.R':[(219,231)],
 'src/prior_climate.R':[(6,28),(35,46),(68,130)],
 'src/mcmc_prepare.R':[(635,664),(710,730),(751,800),(820,835),(860,907),(1014,1046),(1138,1149),(1183,1204),(1247,1283),(1295,1304),(1729,1735),(1776,1818),(2054,2058)],
 'src/mcmc_prep_v1.2.R':[(1,30),(764,832),(980,990)],
 'src/mcmc_functions_v1.6.R':[(1843,1855),(1935,1942),(1964,1987),(1993,2027),(2317,2329)],
 'src/mcmc_conduction.R':[(153,164),(178,196)],
}
spans=[]
texts={}
for rel,rr in ranges.items():
    bind('original',rel)
    lines=(roots['original']/rel).read_text(encoding='utf-8-sig').splitlines()
    texts[rel]='\n'.join(lines)
    for first,last in rr:
        spans.append({'source':'original:'+rel,'first_line':first,'last_line':last,
                      'text':'\n'.join(lines[first-1:last])})
# Assertions are change guards for the inspected source, not an R interpreter.
needles={
 'main_v1.2.R':['$human_impact <- c(-60,6500)','age.reso <- 2','[,,50:115]'],
 'src/mcmc_prepare.R':['if(add.sites) hi_time <- FALSE','site.age <- seq(0,220500,by=500)',
    'site.age <- site.age[-c(6:29)]','iso.ref <- d18o.age$splined.series[[170]]',
    'if(sitename == "Lake Kinneret") which.site <- 105','if(any(sitename == c("Lake Kinneret", "")))',
    'iso.ref.time <- d18o.age$splined.series[[ (length(d18o.age$loc_nr)-3) + which.site ]]'],
 'src/prior_climate.R':['if(length(depths.timeslice.id[[1]]) > 0)',
    'if(prior.recent$sd.tmp < 2)  prior.recent$sd.tmp <- 2',
    'if(prior.recent$sd.pann < 150)  prior.recent$sd.pann <- 150'],
 'src/mcmc_functions_v1.6.R':['predictant_current[ages_pollen][hi_ages_time]',
    'posterior_temp$post_iso_hi <- 0.95','predictant_spec <- c(wt_power'],
}
for rel,items in needles.items():
    for needle in items:
        if needle not in texts[rel]:raise ValueError('Inspected code changed: '+rel+' '+needle)

archive_rel='out/plots/mcmc/v5.0/diss_final.zip'
bind('original',archive_rel)
archive_members=[]
with zipfile.ZipFile(roots['original']/archive_rel) as archive:
    for name,rr in [('thesis_kinneret.tex',[(28,43),(140,146)]),('thesis_dead_sea1.tex',[(150,160)])]:
        data=archive.read(name);lines=data.decode('utf-8-sig').splitlines()
        archive_members.append({'member':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        for first,last in rr:spans.append({'source':'archive:'+name,'first_line':first,'last_line':last,'text':'\n'.join(lines[first-1:last])})

raw_rel='my_kinneret/data/in/core/pollen_counts_epd_taxanomy.csv'
bind('additional',raw_rel)
with (roots['additional']/raw_rel).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.reader(f))
ap=np.asarray([float(r[-3]) for r in rows[1:]])
nap=np.asarray([float(r[-2]) for r in rows[1:]])
reference=np.asarray([float(r[-1]) for r in rows[1:]])
reference_error=float(np.max(abs(100*ap/(ap+nap)-reference)))
if reference_error>1e-12:raise ValueError('AP/NAP CSV percentage identity changed')

# Explicitly referenced saved candidate from a different, later Dead Sea branch.
speleo_rel='dead_sea_reconstruction_R/data/in/speleo_data.rdata'
bind('additional',speleo_rel)
v=rdata.read_rda(roots['additional']/speleo_rel)['speleo_data']
times=v['timeseries'];age=np.asarray(times['age'])
saved_speleo={'source':'additional:'+speleo_rel,'which_site':float(np.asarray(v['which_site']).ravel()[0]),
 'timeseries_shape':list(times.shape),'column_names':list(map(str,times.columns)),
 'age_range_as_stored':[float(age.min()),float(age.max())],
 'distinct_age_steps':list(map(float,np.unique(np.diff(age)))),
 'coi_shape':list(np.asarray(v['coi_array']).shape),'wb_shape':list(np.asarray(v['wb']).shape),
 'len_coi':float(np.asarray(v['len_coi']).ravel()[0]),
 'status':'saved later Dead Sea candidate; no proven identity with Kinneret inputs',
 'warning':'Numeric index 170 alone does not identify a Soreq publication or release'}

# Reuse existing negative searches and climate time bounds; no root traversal.
for rel in ['docs/thesis_research/research_snapshot_2026_11_02/historical_provenance/climate_time_bounds.json',
 '.local/console_provenance_v1/explicit_dependency_paths.json','docs/thesis_research/download_manifest.json',
 'docs/thesis_research/kinneret_binding_2026_11_02/summary.json']:
    bind('repository',rel)

loads=[
 ('main_v1.2.R',107,'in/core_data_additional_sites_v1.2.rdata','final Kinneret base','original_relative'),
 ('main_v1.2.R',325,'../../reconstructions/cru_nh/cru_pann_ww.nc','global mean PANN and stn mask','historical_relative'),
 ('main_v1.2.R',326,'../../reconstructions/cru_nh/cru_tann_ww.nc','global mean TANN','historical_relative'),
 ('main_v1.2.R',343,'../../reconstructions/cru_nh/cru4_pann_v1.2.nc','recent PANN series indices 50:115','historical_relative'),
 ('main_v1.2.R',344,'../../reconstructions/cru_nh/cru4_tann_v1.2.nc','recent TANN series indices 50:115','historical_relative'),
 ('main_v1.2.R',388,'in/reference_speleo_iso.rdata','loaded reference workspace','original_relative'),
 ('main_v1.2.R',391,'../../stalage_speleodatabase/sisal/SISAL_V2/speleo_info.csv','cave metadata','historical_relative'),
 ('main_v1.2.R',392,'../../stalage_speleodatabase/sisal/SISAL_V2/ages_info.csv','cave ages metadata','historical_relative'),
 ('main_v1.2.R',394,'../../stalage_speleodatabase/sisal/SISAL_V2/d18_d13.RData','loaded isotope workspace','historical_relative'),
 ('main_v1.2.R',397,'in/isotope_info.rdata','loaded isotope workspace','original_relative'),
 ('main_v1.2.R',401,'in/isotope_speleo_other.rdata','loaded isotope workspace','original_relative'),
 ('main_v1.2.R',569,'out/transfer_functions/mixed_partial_arts_emd.rdata','numeric TANN/PANN TF arrays','original_relative'),
]
deps=[]
for file,line,literal,role,scope in loads:
    path=(roots['original']/literal).resolve()
    # Probe only this literal consumer path, never search its parent.
    exists=path.is_file()
    deps.append({'source_file':file,'line':line,'literal_consumer_path':literal,'role':role,
                 'path_scope':scope,'exists_at_literal_consumer_path':exists,
                 'status':'present_not_automatically_validated' if exists else 'missing_at_explicit_path'})

# Control-flow illustrations use flags only. No isotope arrays, model values or R execute.
branches=[]
for name,core,speleo,marine,case in [('main_three_channels',1,1,1,False),('without_speleo',1,0,1,False),('case_study_switch',1,1,1,True)]:
    if case:speleo=marine=0
    idx=[i for i,val in enumerate([0,core,speleo,marine],1) if val]
    multiple=idx in [[1,4],[2,4],[1,3,4],[2,3,4]]
    branches.append({'scenario':name,'conditional_on_core_AP_source_available':True,
       'active_source_indices':idx,'multiple':multiple,'spectral':3 in idx,
       'pollen_and_WPS_climate_variable':'PANN','marine_climate_variable':'TANN',
       'hi_time':False,'AP_mask':'all ages' if case else '!inside.range(pollen.age,c(-60,6500))',
       'scope':'symbolic source-flag trace only; not a reconstructed historical run'})

for item in sources.values():
    p=roots[item['scope']]/item['relative_path']
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
        raise ValueError('Source changed while reading')

contract={
 'schema_version':1,'date':'2026-11-02','base_commit':'44092d3856d1ab5f686e0ad30c57b3caf66e0ebd',
 'status':'reference_anchor_contract_documented_numerical_inputs_and_run_binding_unresolved',
 'dissertation_contract':{'AP':'PANN reference only older than 6.5 cal ka BP',
  'Soreq':'WPS linked to PANN across Kinneret period; Bar-Matthews 2003 family in caption',
  'MEDSTACK':'TANN reference; Colleoni 2012 family in caption','anchor':'recent Csa climate class',
  'climate':'TANN degrees C; PANN mm/a; CRU4.01 1961-1990 as text reference'},
 'consumer_contract':{'AP_mask_literal':'!inside.range(pollen.age,c(-60,6500))',
  'AP_mask_not_disabled_by_hi_time_false':True,'AP_boundary_6500':'inside.range package implementation/version not restored; retain literal',
  'Soreq_forced_105_only_in_multi_candidate_branch':True,'Soreq_later_override':170,
  'site_age_override':{'start':0,'end':220500,'step':500,'deleted_R_indices':'6:29','expected_length':418},
  'marine_selection_if_Csa':'category 1, element length(d18o.age$loc_nr)-2; sign reversed',
  'marine_actual_object_identity':'unresolved without d18o.age metadata and load provenance',
  'anchor_if_nonpositive_age_present':'normal fit to local recent series indices 50:115',
  'anchor_else':'normal fit to class-matched regional global-mean cells',
  'anchor_sd_floors':{'TANN':2,'PANN':150},'anchor_density':{'TANN':'normal','PANN':'gamma'},
  'calendar_years_for_indices_50_115':'unverified without time axis; source comment says 1950-2015',
  'full_entry_execution':'not validated; preparation contains active stop at line 2057'},
 'source_variant_comparison':{'older_preparation':'src/mcmc_prep_v1.2.R',
  'older_Kinneret_forced_cave_index':37,'newer_initial_forced_index':105,'newer_later_override_index':170,
  'older_contains_220500_age_override': 'site.age <- seq(0,220500,by=500)' in texts['src/mcmc_prep_v1.2.R'],
  'run_binding':'No version selected as the final Kinneret run'},
 'AP_csv_check':{'rows':160,'identity':'tree_percentage = 100*AP/(AP+NAP)',
   'maximum_absolute_percentage_point_error':reference_error,
   'scope':'existing CSV only, no depth-age transformation performed'},
 'saved_speleo_candidate':saved_speleo,'control_flow_traces':branches,
 'sources':sources,'archive_members':archive_members,
 'checks':{'source_needles':sum(map(len,needles.values())),'all_sources_unchanged':True,
           'dependency_paths_probed':len(deps),'any_numerical_anchor_fitted':False},
 'limits':['Text-defined reconstruction and current source behavior remain separately identified',
  'No entry-point change, data repair, silent Soreq/MEDSTACK substitution or climate recalibration',
  'Final base, historical numeric TFs, original d18o.age workspaces and exact global CRU inputs still missing',
  'Prior reader/model audits and broad inventory were not rerun']}
json_write('contract.json',contract)
json_write('source_spans.json',spans)
with (OUT/'dependencies.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(deps[0]),lineterminator='\n');w.writeheader();w.writerows(deps)
print(json.dumps({'checks':contract['checks'],'AP_csv':contract['AP_csv_check'],'saved_candidate':saved_speleo},ensure_ascii=False))
