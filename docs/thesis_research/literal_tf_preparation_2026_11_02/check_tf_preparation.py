"""Evidence-only preparation and bridge into four unchanged synthetic fragments."""
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tools'))
import historical_tf_preparation as preparation


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exact=load('tf_preparation_exact_reference',OUT/'rational_reference.py')
chain=load('tf_preparation_prior_chain',ROOT/'docs/thesis_research/literal_phase1_chain_2026_11_02/check_chain.py')
chain_exact=load('tf_preparation_prior_chain_exact',ROOT/'docs/thesis_research/literal_phase1_chain_2026_11_02/rational_reference.py')


def fixture():
    return json.loads((OUT/'synthetic_fixture.json').read_text())


def base_chain(f):
    return json.loads((ROOT/f['chain_bridge']['base_fixture']).read_text())


def numeric(value):
    return [numeric(v) for v in value] if isinstance(value,list) else float(Fraction(value))


def evaluate_case(case):
    norm=preparation.normalize_tf_rows(case['raw'])
    masked=preparation.mask_tf_cells(norm,case['keep'])
    return {'normalized':norm.tolist(),'masked':masked.tolist(),
            'row_masses_after_mask':masked.sum(axis=1).tolist()}


def evaluate_bridge(f):
    copied=deepcopy(base_chain(f))
    case=f['cases'][f['chain_bridge']['TF_case']]
    copied['transfer_functions_taxon_by_cell']=evaluate_case(case)['masked']
    return chain.evaluate_fixture(copied)


def exact_expectations(f):
    cases={name:exact.exact_preparation(case['raw'],case['keep']) for name,case in f['cases'].items()}
    copied=deepcopy(base_chain(f))
    copied['transfer_functions_taxon_by_cell']=cases[f['chain_bridge']['TF_case']]['masked']
    return {'cases':cases,'prepared_chain':chain_exact.exact_expectations(copied)}


def evaluate(f):
    return {'cases':{name:evaluate_case(case) for name,case in f['cases'].items()},
            'prepared_chain':evaluate_bridge(f)}


def max_error(actual,expected):
    errors=[]
    for name,result in expected['cases'].items():
        for stage,values in result.items():
            errors.append(float(np.max(np.abs(np.asarray(actual['cases'][name][stage])-numeric(values)))))
    for stage in ['site_age','age','depth_weights','pollen_at_ages','column_max_scaled','raw_sum','pdf']:
        errors.append(float(np.max(np.abs(np.asarray(actual['prepared_chain'][stage])-numeric(expected['prepared_chain'][stage])))))
    for draw,stages in expected['prepared_chain']['draws'].items():
        for stage,values in stages.items():
            errors.append(float(np.max(np.abs(np.asarray(actual['prepared_chain']['draws'][draw][stage])-numeric(values)))))
    return max(errors)


def main():
    f=fixture(); expected=exact_expectations(f); actual=evaluate(f)
    error=max_error(actual,expected)
    if error>f['tolerance']:raise AssertionError('Independent rational control failed')
    altered=deepcopy(base_chain(f))
    prepared=np.array(actual['cases'][f['chain_bridge']['TF_case']]['masked'])
    altered['transfer_functions_taxon_by_cell']=(prepared/prepared.sum(axis=1,keepdims=True)).tolist()
    wrong=chain.evaluate_fixture(altered)
    premature=preparation.normalize_tf_rows(preparation.mask_tf_cells(f['cases']['source_brief']['raw'],f['cases']['source_brief']['keep']))
    summary={'kind':'synthetic_fragment_check_not_historical_reproduction','maximum_absolute_rational_error':error,
        'tolerance':f['tolerance'],'post_mask_renormalization_TF_max_difference':float(np.max(np.abs(premature-prepared))),
        'post_mask_renormalization_chain_PDF_max_difference':float(np.max(np.abs(np.array(wrong['pdf'])-actual['prepared_chain']['pdf']))),
        'active_original_draw_ids':actual['prepared_chain']['draw_ids_R_1_based'],
        'R_execution':False,'historical_reproduction_ready':False,'final_variant_selected':False}
    for name,value in [('rational_expectations.json',expected),('synthetic_result.json',actual),('arithmetic_check.json',summary)]:
        (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
