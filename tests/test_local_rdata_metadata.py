import io
import struct
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from local_rdata_metadata import MetadataSpecial, MetadataXDR
from rdata.parser._parser import RObjectType


def wire(*values):
    return io.BytesIO(struct.pack('>' + 'i' * len(values), *values))


@pytest.mark.parametrize('code,kind', [(252, MetadataSpecial.UNBOUNDVALUE), (250, MetadataSpecial.BASENAMESPACE)])
def test_special_markers_are_distinct_and_consume_no_payload(code, kind):
    stream = wire(code, 13, 1, 42)
    parser = MetadataXDR(stream)
    marker = parser.parse_R_object()
    assert marker.info.type == kind
    assert marker.value == kind.name
    assert stream.tell() == 4
    following = parser.parse_R_object()
    assert following.info.type == RObjectType.INT
    assert following.value.tolist() == [42]


def test_package_reference_preserves_table_and_next_object():
    # package: placeholder, one CHAR entry 'stats'; then REF to package, then int
    raw = struct.pack('>iiiii', 248, 0, 1, 9, 5) + b'stats' + struct.pack('>iiii', 511, 13, 1, 7)
    parser = MetadataXDR(io.BytesIO(raw))
    refs = []
    package = parser.parse_R_object(refs)
    assert package.info.type == MetadataSpecial.PACKAGE
    assert package.value[0].value == b'stats'
    assert refs == [package]
    assert parser.parse_R_object(refs).referenced_object is package
    assert parser.parse_R_object(refs).value.tolist() == [7]


def test_unknown_type_is_not_silently_substituted():
    with pytest.raises(ValueError):
        MetadataXDR(wire(247)).parse_R_object()


def test_truncated_package_is_rejected():
    with pytest.raises((EOFError, ValueError, struct.error)):
        MetadataXDR(wire(248, 0)).parse_R_object()


@pytest.mark.parametrize('outer_tag', [13, 252, 65279])
def test_default_bytecode_constant_tag_is_followed_by_read_item(outer_tag):
    # R serialize.c ReadBCConsts/ReadBCLang default branch reads a complete
    # object following the already-consumed outer tag. It is not a payload.
    stream=wire(13,1,9,13,1,8)
    parser=MetadataXDR(stream)
    assert parser.parse_R_object(info_int=outer_tag).value.tolist()==[9]
    assert parser.parse_R_object().value.tolist()==[8]
