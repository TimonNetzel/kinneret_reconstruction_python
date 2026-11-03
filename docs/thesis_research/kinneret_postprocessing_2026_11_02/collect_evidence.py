"""Read-only audit of preserved Kinneret postprocessing sources.

Only explicit sources and exact posterior paths are inspected. Deterministic
fixtures contain synthetic integers/probabilities; no original draws, random
numbers, KDEs, IAT estimates, R execution or fossil reconstructions are produced.
"""
from pathlib import Path
from fractions import Fraction
import csv
import hashlib
import json
import re
import zipfile

REPO=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
cfg=json.loads((REPO/'.local/my_reconst_paths.json').read_text(encoding='utf-8'))
root=Path(cfg['original_data_root'])
sources={}
def bind(scope,rel):
    base=root if scope=='original' else REPO
    p=base/rel
    key=scope+':'+rel
    if key not in sources:
        sources[key]={'scope':scope,'relative_path':rel,'exists':p.is_file()}
        if p.is_file():sources[key].update(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    return key

def write_json(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
ranges={
 'main_v1.2.R':[(38,50),(74,79),(624,632),(654,683)],
 'src/mcmc_prepare.R':[(1295,1304),(1328,1364)],
 'src/mcmc_case_studies.R':[(8,43),(67,85),(95,142),(165,213),(232,246),(282,332)],
 'src/mcmc_functions_v1.6.R':[(366,388),(419,477),(539,611),(683,766),(843,851)],
 'src/mcmc_trimming.R':[(87,143),(145,187),(208,212)],
 'src/mcmc_algorithm_plots.R':[(1068,1123),(1141,1168)],
 'src/independent_samples.R':[(4,33)],
 'src/mcmc_trimming_posterior_trafo.R':[(94,113),(168,215),(235,261),(308,336)],
 'src/final_plotting.R':[(65,104)],
}
spans=[];texts={}
for rel,rr in ranges.items():
    bind('original',rel)
    lines=(root/rel).read_text(encoding='utf-8-sig').splitlines()
    texts[rel]='\n'.join(lines)
    for first,last in rr:spans.append({'source':'original:'+rel,'first_line':first,'last_line':last,'text':'\n'.join(lines[first-1:last])})
needles={
 'src/mcmc_case_studies.R':['averaging_pdfs <- T','averaging_data <- F',
  '#     independent_samples <- best_combs[independent_samples]',
  'if(length(energy_temp) >= 100 & FALSE)',
  'independent_samples <- dupl_id[independent_samples_temp]',
  'if(length(pareto_id) > 10) pareto_id <- pareto_id[1:10]'],
 'src/mcmc_trimming.R':['\nindependent_samples <- best_combs[independent_samples]',
  'while(iat_temp >= 1.05 & FALSE)','if(length(energy_temp) > 10 & FALSE)'],
 'src/mcmc_algorithm_plots.R':['averaging_data <- T','    independent_samples <- best_combs[independent_samples]'],
 'src/mcmc_functions_v1.6.R':['sample_size <- 25000','if(averaging_pdfs)',
  '}else if(averaging_data)','combs_unique <- which(!duplicated(energy))'],
 'src/mcmc_trimming_posterior_trafo.R':['which_samples <- seq(burn_in+1,sample_length, by=thin_size)',
  '# which_samples <- which_samples[which( climate_check & variance_check )]',
  'source("src/final_plotting.R")'],
 'src/final_plotting.R':['DescTools::Quantile','pann_depth_50[i] <- quantiles_tann_age[3]'],
}
for rel,nn in needles.items():
    for n in nn:
        if n not in texts[rel]:raise ValueError('Inspected source changed: '+rel+' '+n)

archive_rel='out/plots/mcmc/v5.0/diss_final.zip';bind('original',archive_rel)
with zipfile.ZipFile(root/archive_rel) as z:
    member='thesis_mcmc_results.tex';raw=z.read(member);lines=raw.decode('utf-8-sig').splitlines()
    member_meta={'member':member,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    for first,last in [(209,220),(274,293)]:spans.append({'source':'archive:'+member,'first_line':first,'last_line':last,'text':'\n'.join(lines[first-1:last])})
    rel='out/plots/case_studies/kinneret/reconstruction.pdf';key=bind('original',rel)
    archived_plot_equal=hashlib.sha256(z.read('figures/kinneret/reconstruction.pdf')).hexdigest()==sources[key]['sha256']
for rel in ['out/plots/case_studies/kinneret/mcmc_samples.pdf','out/plots/case_studies/kinneret/mcmc_samples.png']:
    bind('original',rel)

# Literal paths for the four declared source branches, no directory traversal.
paths=[]
for algo in ['rw','gibbs']:
    for counts in [True,False]:
        for case in [False,True]:
            suffix=('_case_studies' if case else '')+('_counts' if counts else '')
            rel=f'out/posterior/{algo}_posterior_add_3_1{suffix}.rdata'
            key=bind('original',rel)
            paths.append({'algorithm':algo,'percent':not counts,'case_studies':case,
             'station_id':3,'sim_nr':1,'relative_path':rel,'exists':sources[key]['exists'],
             'scope':'exact current source branch candidate, not a recovered historical run'})
with (OUT/'posterior_paths.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(paths[0]),lineterminator='\n');w.writeheader();w.writerows(paths)

# Synthetic relative-index counterexample. Fewer than eleven distinct scores:
# the preserved case-studies branch performs no IAT calculation in this fixture.
posterior_scores=[.10,.20,.30,.56,.10,.20,.56,.10,.57,.20,.30,.58]
best_combs=[4,7,9,12]
filtered=[round(posterior_scores[i-1],5) for i in best_combs]
seen=set();positions=[]
for i,value in enumerate(filtered,1):
    if value not in seen:positions.append(i);seen.add(value)
original_ids=[best_combs[i-1] for i in positions]
assert positions==[1,3,4] and original_ids==[4,9,12]
assert set(positions)-set(best_combs)=={1,3}
index_fixture={'kind':'synthetic_control_flow_counterexample_not_original_posterior',
 'posterior_scores':posterior_scores,'best_combs_original_draw_ids':best_combs,
 'scores_after_filter_and_rounding':filtered,'deduplicated_positions_within_filter':positions,
 'literal_case_studies_indices_before_Pareto':positions,
 'original_draw_ids_addressed_by_those_filtered_positions':original_ids,
 'indices_outside_filter_if_used_directly':[i for i in positions if i not in best_combs],
 'IAT_not_called_in_fixture':True,
 'limit':'No repair applied. The separate trimming/algorithm sources contain the explicit remapping.'}

# Exact arithmetic with normalized toy PMFs: mixture vs mean of independent RVs.
pmf={Fraction(0):Fraction(1,2),Fraction(2):Fraction(1,2)}
average={}
for x,px in pmf.items():
    for y,py in pmf.items():
        z=(x+y)/2;average[z]=average.get(z,Fraction(0))+px*py

def moments(d):
    m=sum(x*p for x,p in d.items())
    v=sum((x-m)**2*p for x,p in d.items())
    return {'mass':str(sum(d.values())),'mean':str(m),'variance':str(v)}
mixture=moments(pmf);mean_rv=moments(average)
assert mixture=={'mass':'1','mean':'1','variance':'1'}
assert mean_rv=={'mass':'1','mean':'1','variance':'1/2'}
density_fixture={'kind':'exact_synthetic_PMF_control_no_random_draws_or_climate_data',
 'input_PMF':[{'x':str(x),'probability':str(p)} for x,p in sorted(pmf.items())],
 'equal_mixture_of_two_identical_PMFs':mixture,
 'mean_of_two_independent_random_variables':{'PMF':[{'x':str(x),'probability':str(p)} for x,p in sorted(average.items())],**mean_rv},
 'scope':'Demonstrates non-equivalence of named source modes; no method selected or corrected'}
fixture={'index_mapping':index_fixture,'density_combination':density_fixture}
write_json('synthetic_checks.json',fixture)

# Count scalar energy use only within the inspected final_reconstruction body.
fun='\n'.join(texts['src/mcmc_functions_v1.6.R'].splitlines()[365:851])
energy_weight_occurrences=len(re.findall(r'\benergy_weight\b',fun))
if energy_weight_occurrences!=1:raise ValueError('Energy weighting use changed')
for item in sources.values():
    p=(root if item['scope']=='original' else REPO)/item['relative_path']
    if p.is_file()!=item['exists'] or (item['exists'] and hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']):raise ValueError('Source changed while reading')

contract={'schema_version':1,'date':'2026-11-02','base_commit':'edfd3f1be486b94915dad6bf618b0b82e2a57f19',
 'status':'postprocessing_variants_bound_final_run_and_original_draws_unresolved',
 'thesis_text':{'selection':['prior IQR','continue trimming until IAT below 1.3','Pareto proximity to prior means','ten individual reconstructions'],
                'combination':'convolution for mean of independent random variables, scaled by 1/N'},
 'case_studies_source':{'combination_flags':{'averaging_pdfs':True,'averaging_data':False},
   'deduplication':'post_pollen rounded to five decimals; fallback to post_iso under source condition',
   'index_mapping_back_to_best_combs':'commented out','IAT_refinement_loop':'inside >=100 & FALSE branch',
   'active_IAT_use':'one IAT estimate followed by stride ceiling(IAT) when more than ten unique scores',
   'final_selected_count':'at most ten, possibly fewer','explicit_burn_in_filter_in_selection_block':False,
   'post_prior_Pareto':'additional high(proxy scores) preference and cap',
   'stop_after_selection_and_reconstruction_line':332},
 'alternative_sources':{'mcmc_trimming.R':'explicit index remapping, entire outer IAT branch FALSE, PDF sum mode',
   'mcmc_algorithm_plots.R':'explicit index remapping, IAT refinement disabled in inspected block, random-variable-mean mode',
   'independent_samples.R':'separate IAT refinement to 1.05 or small-pool exit; no final Kinneret run binding'},
 'current_entry':{'active_postprocessor':'mcmc_trimming_posterior_trafo.R','case_studies_call':'commented',
   'selection':'fixed burn/thin, five-decimal score dedup; IQR filter commented',
   'output':'KDE of weighted taxon-climate point values over saved draws, not the selected TF-density combination',
   'rw_source_settings':{'sample_length':25000,'burn_in':5000,'thin_size':5,'pre_dedup_candidate_count':4000}},
 'final_reconstruction':{'selected_draw_dedup':'exact equality of supplied energy values','energy_weight_occurrences':energy_weight_occurrences,
   'mixture':'sum input density rows then normalize; no scalar energy weighting in sum',
   'random_variable_mean':'25000 random values per selected density/age or depth, sum/divide retained count then KDE; additional PANN remap',
   'mode_precedence':['averaging_pdfs','averaging_data','conflation'],'single_selected_row':'locally switches to PDF mode'},
 'display':{'quantiles':'spline to 10000 knots then DescTools::Quantile',
   'depth_PANN_median_literal':'pann_depth_50[i] <- quantiles_tann_age[3]',
   'status':'source-only display conflict; not repaired or asserted to affect archived age plot'},
 'sources':sources,'archive_member':member_meta,'archive_reconstruction_bytes_equal':archived_plot_equal,
 'checks':{'source_needles':sum(map(len,needles.values())),'all_explicit_sources_unchanged':True,
   'posterior_paths_probed':len(paths),'posterior_candidates_present':sum(x['exists'] for x in paths),
   'synthetic_index_counterexample_passed':True,'exact_PMF_control_passed':True},
 'limits':['No original posterior IDs, selected ten draws or final numeric densities restored',
  'No R/rPref/LaplacesDemon/DescTools behavior asserted without version binding',
  'Archived graphic bytes do not identify a unique postprocessing source version',
  'All historical source differences preserved; no source fixes or new reconstructions']}
write_json('contract.json',contract);write_json('source_spans.json',spans)
print(json.dumps(contract['checks'],ensure_ascii=False))
