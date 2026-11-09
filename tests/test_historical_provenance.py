"""Metadata classification must follow saved classes/levels, never filenames."""
import sys
from pathlib import Path
import numpy as np
import pytest
import rdata
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from audit_historical_provenance import model_worker, stats
from local_rdata_metadata import read_metadata
from local_rdata_stream import children, small_value


@pytest.mark.parametrize('version',[2,3])
@pytest.mark.parametrize('compression',[None,'gzip'])
def test_metadata_reader_matches_regular_reader_on_standard_values(tmp_path,version,compression):
    path=tmp_path/'plain.rdata'
    values={'names':np.array(['A','A','B']), 'values':np.array([1.,np.nan,3.])}
    rdata.write_rda(path,values,format_version=version,compression=compression)
    full=rdata.read_rda(path)
    data,parser=read_metadata(path)
    fields=children(data.object)
    for key in values:
        np.testing.assert_array_equal(small_value(fields[key],parser),full[key])
    assert parser.special_counts=={}
    assert data.versions.format==version


class SavedFit:
    def __init__(self, **fields): self.fields=fields


def fit_constructor(value,converter):
    obj=converter.convert_to_r_object(value.fields)
    obj.attributes=converter.convert_to_r_attributes({'names':np.array(list(value.fields)), 'class':np.array(['train','train.formula'])})
    obj.info.object=True
    return obj


def test_nested_binary_and_multiclass_fits_are_counted_without_merging(tmp_path):
    binary=SavedFit(method=np.array(['nnet']),levels=np.array(['absent','present']),
                    coefnames=np.array(['tann','pann']),trainingData={'.outcome':np.array([0,1,0]),'tann':np.ones(3)})
    multi=SavedFit(method=np.array(['qda']),levels=np.array(['taxa1','taxa2','unknown']))
    path=tmp_path/'all_279_transfer_functions.rdata'
    rdata.write_rda(path,{'models':{'first':binary,'second':multi,'third':binary}},constructor_dict={SavedFit:fit_constructor})
    result=model_worker(path)
    assert result['caret_fit_count']==3
    assert result['binary_caret_fit_count']==2
    assert result['multiclass_caret_fit_count']==1
    assert result['fits'][0]['training_rows']==3
    assert result['R_prediction_validated'] is False
    assert result['interpretation']=='requires_object_level_interpretation_no_filename_assignment'


def test_value_and_mask_mismatches_remain_separate():
    result=stats(np.array([0.,1.,np.nan,0.]),np.array([1.,0.,0.,np.nan]))
    assert result=={'finite_value_mismatches':2,'mask_mismatches':2,'a1_b0':1,'a0_b1':1}
