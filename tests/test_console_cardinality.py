"""Cardinality checks never run SMOTE or train an estimator."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from audit_console_cardinality import class_cardinality,parse_console


def test_documented_counts_reproduce_first_console_case():
    x=class_cardinality(333,53692)
    assert (x['train_presence'],x['train_absence'])==(234,37585)
    assert x['duplications_per_minority']==159
    assert x['synthetic_only_rows']==74791
    assert x['with_original_minority_rows']==75025
    assert x['unbalanced_training_rows']==37819


def test_nondivisible_counts_use_classwise_ceiling_and_synthetic_only():
    x=class_cardinality(19,54006)
    assert (x['train_presence'],x['train_absence'])==(14,37805)
    assert x['synthetic_only_rows']==75591
    assert x['with_original_minority_rows']-x['synthetic_only_rows']==14


@pytest.mark.parametrize('presence,skip',[(0,True),(1,True),(10,True),(11,False)])
def test_source_presence_gate(presence,skip):
    assert class_cardinality(presence,1000)['source_skip'] is skip


def test_missing_else_branch_is_not_invented_for_balanced_classes():
    x=class_cardinality(100,100)
    assert x['source_smote_branch'] is False
    assert x['synthetic_only_rows'] is None


@pytest.mark.parametrize('p,n,f',[(-1,10,.7),(1,-1,.7),(1,10,0),(1,10,1.1)])
def test_invalid_counts_rejected(p,n,f):
    with pytest.raises(ValueError):class_cardinality(p,n,f)


def test_console_parser_preserves_skipped_positions_and_ansi_data():
    text='[1] 1\n  -> model label : nnet\n  -> data : 74791 rows 5 cols\n[1] 2\n[1] 3\n  -> model label : mda\n  -> data : \x1b[33m12\x1b[39m rows 5 cols\n'
    rows=parse_console(text)
    assert [r['printed_position'] for r in rows]==[1,2,3]
    assert [r['logged_rows'] for r in rows]==[74791,None,12]
    assert rows[1]['model_labels']==[]


@pytest.mark.parametrize('text',['','[1] 1\n[1] 1\n','[1] 1\n[1] 3\n'])
def test_unordered_or_missing_console_positions_fail(text):
    with pytest.raises(ValueError,match='consecutive'):parse_console(text)


def test_incomplete_console_model_record_is_not_filled_in():
    with pytest.raises(ValueError,match='not paired'):
        parse_console('[1] 1\n -> model label : nnet\n')


def test_disagreement_between_models_is_not_collapsed():
    with pytest.raises(ValueError,match='different data shapes'):
        parse_console('[1] 1\n -> model label : nnet\n -> data : 10 rows 5 cols\n -> model label : mda\n -> data : 11 rows 5 cols\n')
