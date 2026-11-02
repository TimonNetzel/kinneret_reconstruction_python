"""Verify bounded XDR reading against rdata's independent full reader."""
import gzip
import importlib.util
from pathlib import Path

import numpy as np
import pytest

rdata = pytest.importorskip("rdata")
tool = Path(__file__).resolve().parents[1] / "tools/local_rdata_stream.py"
spec = importlib.util.spec_from_file_location("local_rdata_stream", tool)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


@pytest.mark.parametrize("version", [2, 3])
@pytest.mark.parametrize("compression", [None, "gzip"])
def test_planes_and_names_match_full_reader(tmp_path, version, compression):
    array = np.arange(5 * 7 * 3, dtype=float).reshape(5, 7, 3, order="F")
    array[2, 4, 1] = np.nan
    path = tmp_path / "fixture.rdata"
    rdata.write_rda(path, {"pool": {"dist": array, "names": np.array(["A", "A", "B"])}},
                    format_version=version, compression=compression)
    full = rdata.read_rda(path)["pool"]
    data, parser = reader.read_structure(path, max_vector_bytes=128)
    fields = reader.children(reader.children(data.object)["pool"])
    block = parser.blocks[id(fields["dist"].value)]
    assert block["length"] == array.size
    assert fields["dist"].value.size == 0
    with pytest.raises(ValueError, match="deliberately absent"):
        reader.small_value(fields["dist"], parser)
    dims = tuple(reader.attributes(fields["dist"])["dim"])
    actual = np.stack([plane.copy() for _, plane in reader.read_planes(path, block, dims)], axis=2)
    np.testing.assert_array_equal(actual, full["dist"])
    np.testing.assert_array_equal(reader.small_value(fields["names"], parser), full["names"])


def test_truncated_payload_is_rejected(tmp_path):
    path = tmp_path / "fixture.rdata"
    rdata.write_rda(path, {"x": np.arange(100.0)}, compression=None, format_version=2)
    path.write_bytes(path.read_bytes()[:100])
    with pytest.raises(EOFError):
        reader.read_structure(path, max_vector_bytes=32)


def test_changed_payload_hash_is_rejected(tmp_path):
    path = tmp_path / "fixture.rdata"
    array = np.arange(24.0).reshape((2, 3, 4), order="F")
    rdata.write_rda(path, {"x": array}, compression=None, format_version=2)
    data, parser = reader.read_structure(path, max_vector_bytes=128)
    node = reader.children(data.object)["x"]
    block = parser.blocks[id(node.value)]
    raw = bytearray(path.read_bytes())
    raw[block["byte_offset"] + 7] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="hash changed"):
        list(reader.read_planes(path, block, (2, 3, 4)))


def test_dimensions_cannot_be_guessed(tmp_path):
    path = tmp_path / "fixture.rdata"
    rdata.write_rda(path, {"x": np.arange(24.0).reshape((2, 3, 4), order="F")},
                    compression="gzip", format_version=2)
    data, parser = reader.read_structure(path, max_vector_bytes=128)
    node = reader.children(data.object)["x"]
    block = parser.blocks[id(node.value)]
    with pytest.raises(ValueError, match="Dimensions"):
        list(reader.read_planes(path, block, (3, 3, 4)))
