"""Historical TF preparation: rational checks, source boundaries and composition."""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
PACKET=ROOT/'docs/thesis_research/literal_tf_preparation_2026_11_02'
sys.path.insert(0,str(PACKET))
from check_tf_preparation import preparation as source,fixture,numeric,evaluate,evaluate_case,exact_expectations,chain,base_chain


@pytest.mark.parametrize('case',['source_brief','rectangular','PANN_separate_width'])
@pytest.mark.parametrize('stage',['normalized','masked','row_masses_after_mask'])
def test_preparation_matches_independent_exact_rational_values(case,stage):
    f=fixture();expected=exact_expectations(f)
    np.testing.assert_allclose(evaluate_case(f['cases'][case])[stage],numeric(expected['cases'][case][stage]),rtol=0,atol=f['tolerance'])


def test_manual_source_brief_anchors_and_no_second_normalization():
    f=fixture();actual=evaluate_case(f['cases']['source_brief'])
    np.testing.assert_allclose(actual['normalized'],[[1,0,0,0],[0,0,1/3,2/3]],rtol=0,atol=1e-15)
    np.testing.assert_allclose(actual['masked'],[[1,0,0,0],[0,0,0,2/3]],rtol=0,atol=1e-15)
    assert actual['row_masses_after_mask']==[1,2/3]
    assert source.normalize_tf_rows(actual['masked'])[1,3]==1


def test_rectangular_cell_axis_not_taxon_axis():
    masked=evaluate_case(fixture()['cases']['rectangular'])['masked']
    np.testing.assert_allclose(masked,[[1/10,0,3/10,0,0],[0,0,0,0,1/3],[1/2,0,0,0,1/2]],rtol=0,atol=1e-15)


def test_all_true_mask_preserves_values_and_returns_independent_storage():
    values=source.normalize_tf_rows([[1,2,3],[3,2,1]])
    result=source.mask_tf_cells(values,[True,True,True])
    np.testing.assert_array_equal(result,values)
    result[0,0]=99
    assert values[0,0]==1/6


def test_fully_discarded_cells_remain_zero_and_are_not_repaired_downstream():
    zero=source.mask_tf_cells(source.normalize_tf_rows([[1,2],[2,1]]),[False,False])
    np.testing.assert_array_equal(zero,np.zeros((2,2)))
    with pytest.raises(source.UnsupportedSourceDomain):source.normalize_tf_rows(zero)
    from historical_final_pdf import sum_pdf_branch,UnsupportedSourceDomain
    with pytest.raises(UnsupportedSourceDomain):sum_pdf_branch(zero[None,:,:],[1])


def test_masking_is_separate_and_never_implicitly_normalizes_supplied_rows():
    np.testing.assert_array_equal(source.mask_tf_cells([[0,0],[1,4]],[False,True]),[[0,0],[0,4]])


def test_tiny_positive_rows_are_not_replaced_by_an_epsilon():
    np.testing.assert_array_equal(source.normalize_tf_rows([[5e-324,0],[0,5e-324]]),[[1,0],[0,1]])


@pytest.mark.parametrize('bad',[
 [],[1,2],[[1,2],[]],np.empty((2,0)),np.ones((1,1,1)),
 [[0,0],[1,2]],[[1,-1]],[[float('nan'),1]],[[float('inf'),1]],[[1e308,1e308]],
 [['1','2']],[[True,False]],[[1+2j,0]],
])
def test_unsupported_input_is_rejected_without_repairs(bad):
    with pytest.raises(source.UnsupportedSourceDomain):source.normalize_tf_rows(bad)


@pytest.mark.parametrize('bad',[True,[True],[True,False,True],[[True,False]],[1,0],['yes','no'],[True,None]])
def test_masks_require_explicit_booleans_on_exact_cell_axis(bad):
    with pytest.raises(source.UnsupportedSourceDomain):source.mask_tf_cells([[1,2],[3,4]],bad)


def test_all_false_mask_cannot_hide_nonfinite_source_values():
    with pytest.raises(source.UnsupportedSourceDomain):source.mask_tf_cells([[np.nan,1]],[False,False])


def test_inputs_and_separate_normalized_stage_are_unchanged():
    values=np.array([[2.,0,0,0],[0,0,1,2]])
    keep=np.array([True,True,False,True])
    before=values.copy();mask_before=keep.copy()
    normalized=source.normalize_tf_rows(values);norm_before=normalized.copy()
    masked=source.mask_tf_cells(normalized,keep)
    np.testing.assert_array_equal(values,before)
    np.testing.assert_array_equal(keep,mask_before)
    np.testing.assert_array_equal(normalized,norm_before)
    assert not np.shares_memory(masked,normalized)


def test_TANN_and_PANN_widths_and_masks_are_not_shared_or_synchronized():
    f=fixture();results=evaluate(f)['cases']
    assert np.shape(results['source_brief']['masked'])==(2,4)
    assert np.shape(results['PANN_separate_width']['masked'])==(2,3)
    np.testing.assert_array_equal(results['PANN_separate_width']['masked'],[[0,1/4,0],[0,2/5,3/5]])
    with pytest.raises(source.UnsupportedSourceDomain):source.mask_tf_cells(f['cases']['PANN_separate_width']['raw'],f['cases']['source_brief']['keep'])


@pytest.mark.parametrize('stage',['depth_weights','pollen_at_ages','column_max_scaled','raw_sum','pdf'])
def test_prepared_TFs_compose_with_unchanged_four_module_chain(stage):
    f=fixture();expected=exact_expectations(f);actual=evaluate(f)
    np.testing.assert_allclose(actual['prepared_chain'][stage],numeric(expected['prepared_chain'][stage]),rtol=0,atol=f['tolerance'])


@pytest.mark.parametrize('draw_id',['1','3'])
@pytest.mark.parametrize('stage',['raw_density','raw_cumulative','zero_reference_scores'])
def test_prepared_chain_active_draw_products_match_exact_oracle(draw_id,stage):
    f=fixture();expected=exact_expectations(f);actual=evaluate(f)
    np.testing.assert_allclose(actual['prepared_chain']['draws'][draw_id][stage],numeric(expected['prepared_chain']['draws'][draw_id][stage]),rtol=0,atol=f['tolerance'])


def test_reordering_preparation_changes_chain_and_does_not_overwrite_old_fixture():
    f=fixture();before=base_chain(f);actual=evaluate(f)['prepared_chain']
    changed=deepcopy(before);masked=np.array(evaluate_case(f['cases']['source_brief'])['masked'])
    changed['transfer_functions_taxon_by_cell']=(masked/masked.sum(axis=1,keepdims=True)).tolist()
    assert np.max(np.abs(np.array(chain.evaluate_fixture(changed)['pdf'])-actual['pdf']))>.01
    assert base_chain(f)==before
    assert actual['draw_ids_R_1_based']==[1,3]
    assert actual['historical_reproduction_ready'] is False


def test_frozen_expected_arithmetic_and_source_binding():
    f=fixture()
    assert exact_expectations(f)==json.loads((PACKET/'rational_expectations.json').read_text())
    binding=json.loads((PACKET/'source_binding.json').read_text())
    old=json.loads((ROOT/'docs/thesis_research/literal_phase1_chain_2026_11_02/next_step_source_excerpt.json').read_text())
    assert binding['source_sha256']==old['sha256']
    assert binding['run_binding'] is None
    assert binding['R_or_package_runtime_equivalence_claimed'] is False
    assert binding['actual_original_inputs_used_for_numerics'] is False
