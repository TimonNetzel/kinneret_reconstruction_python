"""Independent wire fixtures follow R serialize.c ReadItem/ReadBC1 scopes."""
import io
import struct
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from local_rdata_metadata import MetadataXDR
from rdata.parser._parser import RObjectType


def wire(*values):
    return struct.pack('>' + 'i' * len(values), *values)


def test_ordinary_nested_bytecode_has_its_own_repetition_count():
    # Outer BC: 1 rep slot, code [12], one ordinary VEC constant.
    # Its nested BC has 53 slots, code [34], no constants; following int [99].
    raw = wire(21, 1, 13, 1, 12, 1, 19, 19, 1,
               21, 53, 13, 1, 34, 0, 13, 1, 99)
    parser = MetadataXDR(io.BytesIO(raw))
    obj = parser.parse_R_object()
    nested = obj.value[1][0].value[0]
    assert nested.info.type == RObjectType.BCODE
    assert nested.value[0].value.tolist() == [34]
    assert nested.value[1] == []
    assert parser.parse_R_object().value.tolist() == [99]
    parser.check_complete()


def test_compact_nested_bytecode_shares_repetition_table():
    # Compact BC constant has no new repetition-count word. It defines
    # a LANG in slot 0; the following outer constant references that slot.
    raw = wire(21, 1, 13, 1, 12, 2,
               21, 13, 1, 34, 1, 244, 0, 6, 254,
               0, 13, 1, 7, 0, 254,
               243, 0, 13, 1, 99)
    parser = MetadataXDR(io.BytesIO(raw))
    obj = parser.parse_R_object()
    nested, alias = obj.value[1]
    assert nested.value[0].value.tolist() == [34]
    assert nested.value[1][0] is alias
    assert alias.info.type == RObjectType.LANG
    assert alias.value[0].value.tolist() == [7]
    assert parser.parse_R_object().value.tolist() == [99]
    parser.check_complete()


def test_default_constant_discards_outer_repetition_scope():
    parser = MetadataXDR(io.BytesIO(wire(21, 53, 13, 1, 8, 0)))
    obj = parser.parse_R_object(bytecode_rep_list=[None], info_int=999)
    assert obj.value[0].value.tolist() == [8]
    parser.check_complete()


@pytest.mark.parametrize('raw', [b'', b'\x00', b'\x00\x00\x00'])
def test_scalar_fast_path_rejects_truncation(raw):
    with pytest.raises(EOFError, match='Truncated XDR integer'):
        MetadataXDR(io.BytesIO(raw)).parse_int()


def test_integer_tags_do_not_use_retained_vector_budget():
    parser = MetadataXDR(io.BytesIO(wire(-1, 0, 53)))
    assert [parser.parse_int() for _ in range(3)] == [-1, 0, 53]
    assert parser.retained_numeric_bytes == 0


def test_model_classification_does_not_convert_irrelevant_environments(monkeypatch):
    from rdata.parser._parser import RObject, RObjectInfo
    from audit_historical_provenance import model_attributes, model_children
    import rdata
    def obj(kind, value, *, attributes=None, tag=None):
        return RObject(info=RObjectInfo(type=kind, object=False, attributes=attributes is not None,
                                      tag=tag is not None, gp=0, reference=0),
                       value=value, attributes=attributes, tag=tag, referenced_object=None)
    nil=obj(RObjectType.NILVALUE, None)
    def attr(key, value, tail):
        return obj(RObjectType.LIST, (value, tail), tag=obj(RObjectType.SYM, obj(RObjectType.CHAR, key.encode())))
    strings=lambda *xs: obj(RObjectType.STR, [obj(RObjectType.CHAR, x.encode()) for x in xs])
    # Unconvertible closure and a skipped numeric payload in nonselected attrs.
    attrs=attr('terms', obj(RObjectType.CLO, (nil, nil)),
          attr('names', strings('x'), attr('class', strings('train'),
          attr('unrelated', obj(RObjectType.REAL, []), nil))))
    vector=obj(RObjectType.VEC, [nil], attributes=attrs)
    def forbid_conversion(*args, **kwargs):
        raise AssertionError('Must not use generic R conversion for attributes')
    # Pairlist tag decoding in children is still trivial and bounded; selected
    # values must never go through generic conversion or inspect terms.
    monkeypatch.setattr(rdata.conversion, 'convert', forbid_conversion)
    assert model_attributes(vector)=={'names':['x'], 'class':['train']}
    assert model_children(vector)=={'x':nil}
    # No attributes: no conversion at all, even with an opaque child.
    assert model_attributes(obj(RObjectType.VEC,[nil]))=={}


def test_repeated_characters_keep_values_with_a_bounded_cache():
    raw=wire(16,30)+b''.join(wire(9,4)+b'Fold' for _ in range(30))+wire(13,1,99)
    parser=MetadataXDR(io.BytesIO(raw), max_objects=8, max_parsed_objects=100)
    obj=parser.parse_R_object()
    assert [x.value for x in obj.value]==[b'Fold']*30
    assert len({id(x) for x in obj.value})==1
    assert parser.reused_characters==29
    assert len(parser.character_cache)==1
    assert parser.parse_R_object().value.tolist()==[99]
    parser.check_complete()


def test_character_cache_preserves_encoding_flags_and_is_bounded():
    raw=wire(16,4)+wire(9,1)+b'x'+wire(9|(64<<12),1)+b'x'+wire(9,1)+b'y'+wire(9,1)+b'y'
    parser=MetadataXDR(io.BytesIO(raw), max_cached_characters=2)
    obj=parser.parse_R_object()
    assert obj.value[0].info.gp != obj.value[1].info.gp
    assert obj.value[0] is not obj.value[1]
    assert obj.value[2] is obj.value[3]
    assert len(parser.character_cache)==2
    assert [x.value for x in obj.value]==[b'x',b'x',b'y',b'y']
    parser.check_complete()


def test_character_attributes_are_not_discarded_by_interning():
    raw=wire(16,2)+wire(9|512,1)+b'x'+wire(254)+wire(9|512,1)+b'x'+wire(254)
    parser=MetadataXDR(io.BytesIO(raw))
    obj=parser.parse_R_object()
    assert all(x.attributes.info.type==RObjectType.NILVALUE for x in obj.value)
    assert obj.value[0] is not obj.value[1]
    assert parser.character_cache=={}
    parser.check_complete()


def test_work_budget_remains_bounded_for_repeated_strings():
    raw=wire(16,30)+b''.join(wire(9,1)+b'x' for _ in range(30))
    with pytest.raises(MemoryError, match='work budget'):
        MetadataXDR(io.BytesIO(raw), max_parsed_objects=8).parse_R_object()


def test_unique_character_objects_still_obey_retained_budget():
    raw=wire(16,10)+b''.join(wire(9,1)+bytes([i+65]) for i in range(10))
    with pytest.raises(MemoryError, match='retained-object budget'):
        MetadataXDR(io.BytesIO(raw), max_objects=8).parse_R_object()


def test_character_cache_recovers_after_many_unique_row_names():
    values=[str(i).encode() for i in range(30)]+[b'Fold1']*30
    raw=wire(16,len(values))+b''.join(wire(9,len(v))+v for v in values)
    parser=MetadataXDR(io.BytesIO(raw), max_cached_characters=4, max_objects=40)
    obj=parser.parse_R_object()
    assert [x.value for x in obj.value]==values
    assert parser.reused_characters==29
    assert len(parser.character_cache)==4
    parser.check_complete()


def test_repeated_row_name_vectors_share_leaves_with_default_cache():
    values=[str(i).encode() for i in range(5000)]*2
    raw=wire(16,len(values))+b''.join(wire(9,len(v))+v for v in values)
    parser=MetadataXDR(io.BytesIO(raw), max_objects=6000)
    obj=parser.parse_R_object()
    assert [x.value for x in obj.value]==values
    assert parser.reused_characters==5000
    assert obj.value[0] is obj.value[5000]
    parser.check_complete()
